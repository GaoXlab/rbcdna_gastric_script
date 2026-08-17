import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import tempfile
import shutil
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))

from configs.params import MODEL_PARAMS
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_sample_info, load_separate_cohorts
from hy.model import run_pipeline
from hy.Estimator import BedFeatureSelector
from hy.evaluate import calculate_thresholds, generate_report, save_report

def main():
    parser = argparse.ArgumentParser(description="Evaluate a full cross-validation over 5 manually created folds.")
    parser.add_argument('--basename', type=str, required=True, help='基线实验名前缀, 例如 gc')
    parser.add_argument('--rnd', action='store_true', help='Test randomized models against shuffled labels')
    parser.add_argument('--working_dir', type=str, default=".", help='工作目录')
    args = parser.parse_args()

    exp_basename = args.basename
    npcs = 50
    is_rnd = args.rnd

    human_model_dir = os.path.abspath(args.working_dir)
    model_data_dir = os.path.join(human_model_dir, "modelData")
    classification_dir = os.path.join(human_model_dir, "results", "4_Classification")
    os.makedirs(classification_dir, exist_ok=True)

    print(f"Working Directory: {human_model_dir}")
    print(f"Evaluating Full CV for basename '{exp_basename}', rnd={is_rnd}")

    sampleinfo_name = f"{exp_basename}_shuffled" if is_rnd else exp_basename
    sample_info = load_sample_info(model_data_dir, sampleinfo_name)

    trn_ids_path = os.path.join(model_data_dir, f"{exp_basename}.trn.ids.txt")
    print(f"Loading training IDs from {trn_ids_path}")
    with open(trn_ids_path, 'r') as f:
        trn_ids = [line.strip() for line in f if line.strip()]
    trn_info = sample_info[sample_info.index.isin(trn_ids)]
    seed = 1637
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_predictions = []

    start_time = time.time()

    for fold, (train_index, val_index) in enumerate(skf.split(trn_info, trn_info['stage']), 1):
        fold_base = f"{exp_basename}_trncv_{fold}"
        if is_rnd:
            fold_exp_name = f"{fold_base}_rnd"
        else:
            fold_exp_name = fold_base

        print(f"\n--- Processing Fold {fold}: {fold_exp_name} ---")

        if is_rnd:
            train_ids = load_separate_cohorts(model_data_dir, f"{fold_exp_name}", 'trn').index.tolist()
            val_ids = load_separate_cohorts(model_data_dir, f"{fold_exp_name}", 'test').index.tolist()
        else:
            train_ids = trn_info.index[train_index].tolist()
            val_ids = trn_info.index[val_index].tolist()

        exp_dir = os.path.join(human_model_dir, fold_exp_name)
        normalized_counts = load_normalized_data(exp_dir, fold_exp_name)

        bed_paths = [os.path.join(exp_dir, f"all.{fold_exp_name}.bed.out")]
        bed_file_path = next((p for p in bed_paths if os.path.exists(p)), None)
        if not bed_file_path:
            raise FileNotFoundError(f"Bed selection file not found for {fold_exp_name}")

        top_n_selector = BedFeatureSelector(bed_path=bed_file_path, top_n=1000)
        X_all_1000 = top_n_selector.fit_transform(normalized_counts)

        X_train = X_all_1000.loc[train_ids]
        y_train = sample_info.loc[train_ids]['target']
        X_test = X_all_1000.loc[val_ids]
        y_test = sample_info.loc[val_ids]['target']

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        model_params = MODEL_PARAMS.copy()
        model_params['pca_params'] = model_params['pca_params'].copy()
        model_params['pca_params']['n_pcas'] = npcs

        tmp_dir = tempfile.mkdtemp(prefix=f'eval_fullcv_fold{fold}_')
        model_params['catboost_params'] = model_params['catboost_params'].copy()
        model_params['catboost_params']['train_dir'] = tmp_dir

        builder = (PipelineBuilder(model_params)
                   .start_sub_pipeline('lr').add_pca_feature_combiner().add_lr_classifier().end_sub_pipeline()
                   .start_sub_pipeline('cb').add_pca_feature_combiner().add_catboost_classifier().end_sub_pipeline()
                   .add_voting_classifier())
        pipe = builder.build()

        try:
            pipe.fit(X_train, y_train)
            test_result = run_pipeline(pipe, X_test)

            separate_results = pipe.transform(X_test)
            test_result['lr'] = separate_results[:, 1]
            test_result['cb'] = separate_results[:, 3]

            test_result['fold'] = fold
            test_result['target'] = y_test

            try:
                f_auc_voting = roc_auc_score(test_result['target'], test_result[0])
                f_auc_lr = roc_auc_score(test_result['target'], test_result['lr'])
                f_auc_cb = roc_auc_score(test_result['target'], test_result['cb'])
                print(f"  --> Fold {fold} AUC: Voting={f_auc_voting:.5f}, LR={f_auc_lr:.5f}, CB={f_auc_cb:.5f}")
            except Exception as e:
                print(f"  --> Fold {fold} AUC Error: {e}")

            fold_predictions.append(test_result)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n==================================================")
    print("ALL FOLDS COMPLETED. CALCULATING GLOBAL RESULTS...")
    print("==================================================")

    combined_preds = pd.concat(fold_predictions)
    combined_preds.index.name = 'seqID'
    combined_preds.reset_index(inplace=True)
    combined_preds.rename(columns={0: '0'}, inplace=True)
    combined_preds['source_key'] = 'test'

    cols = ['seqID', '0', 'lr', 'cb', 'source_key', 'fold', 'target']
    combined_preds = combined_preds[cols]

    suffix = "_rnd" if is_rnd else ""

    pred_output_path = os.path.join(classification_dir, f"{exp_basename}_full_cv{suffix}_prediction.csv")
    combined_preds.to_csv(pred_output_path, index=False)
    print(f"1. Predictions saved to: {pred_output_path}")

    auc_voting = roc_auc_score(combined_preds['target'], combined_preds['0'])
    auc_lr = roc_auc_score(combined_preds['target'], combined_preds['lr'])
    auc_cb = roc_auc_score(combined_preds['target'], combined_preds['cb'])

    print("\n2. Global AUC Results:")
    print(f"   Voting AUC : {auc_voting:.6f}")
    print(f"   CatBoost AUC: {auc_cb:.6f}")
    print(f"   Logistic Reg: {auc_lr:.6f}")

    cutoffs = calculate_thresholds(combined_preds['target'].values, combined_preds[['0']].values.T)
    all_results = {
        'voting': combined_preds.set_index('seqID')['0'],
        'lr': combined_preds.set_index('seqID')['lr'],
        'cb': combined_preds.set_index('seqID')['cb']
    }
    report = generate_report(all_results, sample_info, cutoffs)
    for r in report:
        r['npcs'] = npcs
        r['p100_file'] = "FULL_CV"

    report_file = os.path.join(classification_dir, f"{exp_basename}_full_cv{suffix}_report.csv")
    if os.path.exists(report_file):
        os.remove(report_file)
    save_report(report, report_file)
    print(f"\n3. Comprehensive Standard Report saved to: {report_file}")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tempfile
import shutil
import time

from sklearn.model_selection import ParameterGrid, StratifiedKFold

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))

from configs.params import MODEL_PARAMS
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_sample_info, load_separate_cohorts
from hy.model import run_pipeline
from hy.Estimator import BedFeatureSelector
from hy.evaluate import calculate_thresholds, generate_report, save_report


DEFAULT_NPCS_CANDIDATES = (20, 30, 40, 50, 60)
DEFAULT_TOP_FEATURE_CANDIDATES = tuple(range(100, 1001, 100))


def combine_fold_predictions(fold_predictions):
    combined_preds = pd.concat(fold_predictions)
    combined_preds.index.name = 'seqID'
    combined_preds.reset_index(inplace=True)
    combined_preds.rename(columns={0: '0'}, inplace=True)
    combined_preds['source_key'] = 'test'

    cols = [
        'seqID', '0', 'lr', 'cb', 'source_key', 'fold',
        'top_feature', 'npcs', 'target',
    ]
    return combined_preds[cols]


def fit_predict_fold(fold_label, X_train, y_train, X_test, y_test, npcs):
    model_params = MODEL_PARAMS.copy()
    model_params['pca_params'] = model_params['pca_params'].copy()
    model_params['pca_params']['n_pcas'] = npcs

    tmp_dir = tempfile.mkdtemp(prefix=f'eval_fullcv_{fold_label}_npcs{npcs}_')
    model_params['catboost_params'] = model_params['catboost_params'].copy()
    model_params['catboost_params']['train_dir'] = tmp_dir

    builder = (PipelineBuilder(model_params)
               .start_sub_pipeline('lr').add_pca_feature_combiner().add_lr_classifier().end_sub_pipeline()
               .start_sub_pipeline('cb').add_pca_feature_combiner().add_catboost_classifier().end_sub_pipeline()
               .add_voting_classifier())
    pipe = builder.build()

    try:
        pipe.fit(X_train, y_train)
        predictions = run_pipeline(pipe, X_test)
        separate_results = pipe.transform(X_test)
        predictions['lr'] = separate_results[:, 1]
        predictions['cb'] = separate_results[:, 3]
        predictions['target'] = y_test
        return predictions
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def select_fold_params(
    fold,
    X_train,
    y_train,
    stage_train,
    top_feature_candidates,
    npcs_candidates,
    inner_splits,
    seed,
):
    """Select one (top_feature, npcs) pair using only outer-fold training samples."""
    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    split_indices = list(inner_cv.split(X_train, stage_train))

    score_rows = []
    for current_params in ParameterGrid({
        'top_feature': top_feature_candidates,
        'npcs': npcs_candidates,
    }):
        top_feature = current_params['top_feature']
        npcs = current_params['npcs']
        X_candidate = X_train.iloc[:, :top_feature]
        max_valid_npcs = min(
            min(len(inner_train_idx), X_candidate.shape[1])
            for inner_train_idx, _ in split_indices
        )
        if npcs > max_valid_npcs:
            continue

        inner_predictions = []
        inner_fold_aucs = []
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(split_indices, 1):
            X_inner_train = X_candidate.iloc[inner_train_idx]
            y_inner_train = y_train.iloc[inner_train_idx]
            X_inner_valid = X_candidate.iloc[inner_valid_idx]
            y_inner_valid = y_train.iloc[inner_valid_idx]

            predictions = fit_predict_fold(
                f'outer{fold}_inner{inner_fold}_top{top_feature}',
                X_inner_train,
                y_inner_train,
                X_inner_valid,
                y_inner_valid,
                npcs,
            )
            predictions['inner_fold'] = inner_fold
            inner_predictions.append(predictions)
            inner_fold_aucs.append(
                roc_auc_score(predictions['target'], predictions[0])
            )

        combined_inner = pd.concat(inner_predictions)
        score_rows.append({
            'fold': fold,
            'top_feature': top_feature,
            'npcs': npcs,
            'inner_splits': inner_splits,
            'mean_inner_auc': float(np.mean(inner_fold_aucs)),
            'pooled_inner_auc': roc_auc_score(
                combined_inner['target'],
                combined_inner[0],
            ),
            'pooled_inner_lr_auc': roc_auc_score(
                combined_inner['target'],
                combined_inner['lr'],
            ),
            'pooled_inner_cb_auc': roc_auc_score(
                combined_inner['target'],
                combined_inner['cb'],
            ),
        })

    if not score_rows:
        max_valid_npcs = min(
            min(len(inner_train_idx), X_train.shape[1])
            for inner_train_idx, _ in split_indices
        )
        raise ValueError(
            f"Fold {fold}: no (top_feature, npcs) candidate pair is valid; "
            f"maximum npcs allowed by the inner training matrices is {max_valid_npcs}"
        )

    best_row = sorted(
        score_rows,
        key=lambda row: (
            -row['pooled_inner_auc'],
            -row['mean_inner_auc'],
            row['npcs'],
            row['top_feature'],
        ),
    )[0]
    for row in score_rows:
        row['selected'] = int(
            row['top_feature'] == best_row['top_feature']
            and row['npcs'] == best_row['npcs']
        )
    return int(best_row['top_feature']), int(best_row['npcs']), score_rows


def evaluate_fold(data, top_feature_candidates, npcs_candidates, inner_splits, seed):
    fold = data['fold']
    print(f"\n--- Outer Fold {fold}: {data['fold_exp_name']} ---")
    print(f"  Outer train samples: {len(data['X_train'])}")
    print(f"  Outer valid samples: {len(data['X_test'])}")

    selected_top_feature, selected_npcs, score_rows = select_fold_params(
        fold,
        data['X_train'],
        data['y_train'],
        data['stage_train'],
        top_feature_candidates,
        npcs_candidates,
        inner_splits,
        seed,
    )
    score_text = ", ".join(
        f"top{row['top_feature']}*npcs{row['npcs']}:pooled={row['pooled_inner_auc']:.4f}"
        f"/mean={row['mean_inner_auc']:.4f}"
        for row in score_rows
    )
    print(
        f"  Inner-CV AUCs [{score_text}] -> selected "
        f"top_feature={selected_top_feature}, npcs={selected_npcs}"
    )

    predictions = fit_predict_fold(
        f'outer{fold}_final_top{selected_top_feature}',
        data['X_train'].iloc[:, :selected_top_feature],
        data['y_train'],
        data['X_test'].iloc[:, :selected_top_feature],
        data['y_test'],
        selected_npcs,
    )
    predictions['fold'] = fold
    predictions['top_feature'] = selected_top_feature
    predictions['npcs'] = selected_npcs

    print(
        f"  Outer-valid AUC: Voting={roc_auc_score(predictions['target'], predictions[0]):.5f}, "
        f"LR={roc_auc_score(predictions['target'], predictions['lr']):.5f}, "
        f"CB={roc_auc_score(predictions['target'], predictions['cb']):.5f}"
    )
    return predictions, score_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate a full cross-validation over 5 manually created folds.")
    parser.add_argument('--basename', type=str, required=True, help='基线实验名前缀，例如 gc')
    parser.add_argument('--rnd', action='store_true', help='Test randomized models against shuffled labels')
    parser.add_argument('--working_dir', type=str, default=".", help='工作目录')
    parser.add_argument(
        '--inner_splits',
        type=int,
        default=5,
        help='每个 outer fold 内搜索 npcs 时使用的 inner CV 折数 (默认: 5)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1637,
        help='outer CV 和 inner CV 使用的分层切分种子 (默认: 1637)',
    )
    parser.add_argument(
        '--npcs', '--npcas',
        dest='npcs_candidates',
        type=int,
        nargs='+',
        default=list(DEFAULT_NPCS_CANDIDATES),
        metavar='N',
        help=(
            'PCA component counts searched inside each outer training fold '
            '(default: 20 30 40 50 60).'
        ),
    )
    parser.add_argument(
        '--top_features',
        dest='top_feature_candidates',
        type=int,
        nargs='+',
        default=list(DEFAULT_TOP_FEATURE_CANDIDATES),
        metavar='N',
        help=(
            'Top ranked feature counts searched inside each outer training fold '
            '(default: 100 200 ... 1000).'
        ),
    )
    args = parser.parse_args()

    exp_basename = args.basename
    if args.inner_splits < 2:
        parser.error('--inner_splits must be at least 2')
    if any(npcs <= 0 for npcs in args.npcs_candidates):
        parser.error('--npcs/--npcas values must all be positive integers')
    if any(top_feature <= 0 for top_feature in args.top_feature_candidates):
        parser.error('--top_features values must all be positive integers')
    npcs_candidates = sorted(set(args.npcs_candidates))
    top_feature_candidates = sorted(set(args.top_feature_candidates))
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_data = []
    
    start_time = time.time()

    for fold, (train_index, val_index) in enumerate(skf.split(trn_info, trn_info['stage']), 1):
        fold_base = f"{exp_basename}_trncv_{fold}"
        if is_rnd:
            fold_exp_name = f"{fold_base}_rnd"
        else:
            fold_exp_name = fold_base

        print(f"\n--- Loading Fold {fold}: {fold_exp_name} ---")

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
            
        max_top_feature = max(top_feature_candidates)
        top_n_selector = BedFeatureSelector(
            bed_path=bed_file_path,
            top_n=max_top_feature,
        )
        X_all_ranked = top_n_selector.fit_transform(normalized_counts)
        
        X_train = X_all_ranked.loc[train_ids]
        y_train = sample_info.loc[train_ids]['target']
        X_test = X_all_ranked.loc[val_ids]
        y_test = sample_info.loc[val_ids]['target']
        
        fold_data.append({
            'fold': fold,
            'fold_exp_name': fold_exp_name,
            'X_train': X_train,
            'y_train': y_train,
            'stage_train': sample_info.loc[train_ids]['stage'],
            'X_test': X_test,
            'y_test': y_test,
        })

    print(
        f"\nNested top-feature/NPCS selection: {args.inner_splits}-fold inner CV, "
        f"top_feature_candidates={top_feature_candidates}, "
        f"npcs_candidates={npcs_candidates}, seed={args.seed}"
    )
    fold_predictions = []
    inner_score_rows = []
    for data in fold_data:
        predictions, score_rows = evaluate_fold(
            data,
            top_feature_candidates,
            npcs_candidates,
            args.inner_splits,
            args.seed,
        )
        fold_predictions.append(predictions)
        inner_score_rows.extend(score_rows)

    combined_preds = combine_fold_predictions(fold_predictions)
    inner_scores_df = pd.DataFrame(inner_score_rows).sort_values(
        ['fold', 'pooled_inner_auc', 'mean_inner_auc', 'npcs', 'top_feature'],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    selected_by_fold = inner_scores_df.loc[inner_scores_df['selected'].eq(1)].copy()
    suffix = "_rnd" if is_rnd else ""

    search_output_path = os.path.join(
        classification_dir,
        f"{exp_basename}_full_cv{suffix}_nested_inner_cv_scores.csv",
    )
    inner_scores_df.to_csv(search_output_path, index=False)
    print(f"\n1. Nested inner-CV top-feature/NPCS scores saved to: {search_output_path}")
    print(inner_scores_df.to_string(index=False))
    
    pred_output_path = os.path.join(
        classification_dir,
        f"{exp_basename}_full_cv{suffix}_nested_prediction.csv",
    )
    combined_preds.to_csv(pred_output_path, index=False)
    print(f"2. Nested outer-fold predictions saved to: {pred_output_path}")
    
    auc_voting = roc_auc_score(combined_preds['target'], combined_preds['0'])
    auc_lr = roc_auc_score(combined_preds['target'], combined_preds['lr'])
    auc_cb = roc_auc_score(combined_preds['target'], combined_preds['cb'])
    
    print("\n3. Pooled outer OOF AUC after inner-CV top-feature/NPCS selection:")
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
    npcs_by_fold = ';'.join(
        f"fold{int(row.fold)}={int(row.npcs)}"
        for row in selected_by_fold.sort_values('fold').itertuples(index=False)
    )
    top_features_by_fold = ';'.join(
        f"fold{int(row.fold)}={int(row.top_feature)}"
        for row in selected_by_fold.sort_values('fold').itertuples(index=False)
    )
    for r in report:
        r['top_feature'] = top_features_by_fold
        r['npcs'] = npcs_by_fold
        r['p100_file'] = "FULL_CV_NESTED_INNER_CV"
        
    report_file = os.path.join(
        classification_dir,
        f"{exp_basename}_full_cv{suffix}_nested_report.csv",
    )
    if os.path.exists(report_file):
        os.remove(report_file)
    save_report(report, report_file)
    print(f"\n4. Comprehensive Standard Report saved to: {report_file}")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

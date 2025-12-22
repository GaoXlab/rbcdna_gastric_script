import argparse
import os
from datetime import datetime

import pandas as pd
from numpy import arange
from sklearn.model_selection import ParameterGrid, train_test_split

from configs.params import MODEL_PARAMS
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_sample_info, load_separate_cohorts
from hy.evaluate import calculate_thresholds, generate_report, save_report, save_prediction
from hy.model import save_model, train_pipeline
from hy.Estimator import BedFeatureSelector
from hy.message import message_to_sns


def main(args):
    exp_name = args.exp_name
    discovery = load_separate_cohorts('modelData', exp_name, 'trn')
    sample_info = load_sample_info('modelData', 'gc')
    normalized_counts = load_normalized_data(args.working_dir + f"/{exp_name}", exp_name)
    model_params = MODEL_PARAMS.copy()
    model_params['bed_selector_file'] = args.working_dir + f"/{args.exp_name}/all.{args.exp_name}.bed.out"
    message_to_sns("Start step3 search.")
    normalized_data = {}
    for n in range(100, 1001, 100):
        top_n_selector = BedFeatureSelector(
            bed_path=model_params['bed_selector_file'],
            top_n=n,
        )
        normalized_data[n] = top_n_selector.fit_transform(normalized_counts)
    params_grid = {
        'top_feature': range(100, 1001, 100), # [1000], #
        'n_pcas': range(10, 70, 10), #[50], #
    }
    max_train_cv_auc = best_n_feature = 0
    best_pipe = best_cutoff = best_params = best_oof_results = None
    for current_params in ParameterGrid(params_grid):
        print(f"当前参数: {current_params}")
        model_params['pca_params'].update({
            'n_pcas': current_params['n_pcas'],
        })
        train_x = normalized_data[current_params['top_feature']].loc[discovery.index]
        train_y = sample_info.loc[discovery.index]['target']
        pipe = (PipelineBuilder(model_params)
                .start_sub_pipeline('lr').add_pca_feature_combiner().add_lr_classifier().end_sub_pipeline()
                .start_sub_pipeline('cb').add_pca_feature_combiner().add_catboost_classifier().end_sub_pipeline()
                .add_voting_classifier()
                )
        trained_pipeline, oof_results, fold_results = train_pipeline(pipe, train_x, train_y,cv_splits=5, model_params=model_params)
        all_result = {
            'TRAIN_CV': oof_results,
        }
        cutoff = calculate_thresholds(train_y, oof_results)
        report = generate_report(all_result, sample_info, cutoff)
        report[0]['top_feature'] = current_params['top_feature']
        report[0]['n_pcas'] = current_params['n_pcas']
        if report[0]['AUC'] > max_train_cv_auc:
            max_train_cv_auc = report[0]['AUC']
            best_pipe = trained_pipeline
            best_cutoff = cutoff
            best_params = model_params
            best_oof_results = oof_results
            best_fold_results = fold_results
            best_n_feature = current_params['top_feature']
            print(f"当前最佳AUC : {max_train_cv_auc} @ {current_params}")
        save_report(report, args.working_dir + f"/results/3_FeatureReduction/{exp_name}_trncv_detail.csv")
    transformer = PipelineBuilder(best_params).add_pca_feature_combiner().build()
    pca_x = transformer.fit_transform(normalized_data[best_n_feature].loc[discovery.index])
    pca_x.to_csv(args.working_dir + f"/results/3_FeatureReduction/{exp_name}_pca_train.csv")
    save_model(best_pipe, best_cutoff, best_n_feature, args.working_dir + f"/results/4_Classification/", exp_name)
    save_prediction({'TRAIN_CV': best_oof_results}, args.working_dir + f"/results/4_Classification/{exp_name}_prediction_trncv.csv")
    save_prediction(best_fold_results, args.working_dir + f"/results/4_Classification/{exp_name}_prediction_trncv_fold.csv")

    message_to_sns(f"Step3 finished.with best params: {best_params}, best AUC: {max_train_cv_auc}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")

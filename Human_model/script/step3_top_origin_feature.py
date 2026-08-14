import argparse
import os
from datetime import datetime

import pandas as pd
from numpy import arange
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score

from configs.params import MODEL_PARAMS
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_sample_info, load_separate_cohorts
from hy.evaluate import calculate_thresholds, generate_report, save_report, save_prediction
from hy.model import save_model, train_pipeline
from hy.Estimator import BedFeatureSelector
from hy.message import message_to_sns
from hy.model import run_pipeline


def main(args):
    exp_name = args.exp_name
    # top_feature = args.top_feature
    discovery = load_separate_cohorts('modelData', exp_name, 'trn')
    sample_info = load_sample_info('modelData', 'gc')
    normalized_counts = load_normalized_data(args.working_dir + f"/{exp_name}", exp_name)
    model_params = MODEL_PARAMS.copy()
    model_params['bed_selector_file'] = args.working_dir + f"/{args.exp_name}/all.{args.exp_name}.bed.out"
    message_to_sns("Start step3 search.")
    normalized_data = {}
    n_list = [10] + list(range(100, 1001, 100)) # [top_feature]
    for n in n_list:
        top_n_selector = BedFeatureSelector(
            bed_path=model_params['bed_selector_file'],
            top_n=n,
        )
        normalized_data[n] = top_n_selector.fit_transform(normalized_counts)
    params_grid = {
        'top_feature': n_list, # [1000], #
    }
    
    for current_params in ParameterGrid(params_grid):
        top_feature = current_params['top_feature']

        if top_feature == 10:
            npca_list = [0]
        else:
            npca_list = range(0, 70, 10)

        for npca in npca_list:
            print(f"当前参数: top_feature={top_feature}, PCA={npca}")
            train_x = normalized_data[top_feature].loc[discovery.index]
            train_y = sample_info.loc[discovery.index]['target']
            if npca == 0:
                pipe = (PipelineBuilder(model_params)
                        .start_sub_pipeline('lr').add_lr_classifier().end_sub_pipeline()
                        .start_sub_pipeline('cb').add_catboost_classifier().end_sub_pipeline()
                        .add_voting_classifier()
                        )
            else:
                model_params['pca_params'].update({
                    'n_pcas': npca,
                })
                pipe = (PipelineBuilder(model_params)
                        .start_sub_pipeline('lr').add_pca_feature_combiner().add_lr_classifier().end_sub_pipeline()
                        .start_sub_pipeline('cb').add_pca_feature_combiner().add_catboost_classifier().end_sub_pipeline()
                        .add_voting_classifier()
                        )

            trained_pipeline, oof_results, fold_results = train_pipeline(pipe,train_x,train_y,cv_splits=5,model_params=model_params)

            cb_auc = roc_auc_score(train_y, oof_results['cb'])
            lr_auc = roc_auc_score(train_y, oof_results['lr'])

            all_result = {'TRAIN_CV': oof_results,}

            cutoff = calculate_thresholds(train_y, oof_results)
            report = generate_report(all_result, sample_info, cutoff)

            report[0]['top_feature'] = current_params['top_feature']
            report[0]['n_pcas'] = npca
            report[0]['cb_AUC'] = cb_auc
            report[0]['lr_AUC'] = lr_auc

            save_report(report,args.working_dir + f"/results/3_FeatureReduction/{exp_name}_trncv_detail.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    # parser.add_argument('--top_feature', help='top_feature', type=int, default=10)

    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")

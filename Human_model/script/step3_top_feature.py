import argparse
import os
from datetime import datetime

import pandas as pd
from numpy import arange
from sklearn.model_selection import ParameterGrid, train_test_split

from configs.params import MODEL_PARAMS
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_cohorts, load_sample_info, load_separate_cohorts
from hy.evaluate import calculate_thresholds, generate_report, save_report, save_prediction
from hy.model import save_model, train_pipeline
from hy.Estimator import BedFeatureSelector
from hy.message import message_to_sns
from hy.model import run_pipeline


def main(args):
    exp_name = args.exp_name
    discovery = load_separate_cohorts('modelData', exp_name, 'trn')
    sample_info = load_sample_info('modelData', 'gc')
    normalized_counts = load_normalized_data(args.working_dir + f"/{exp_name}", exp_name)
    model_params = MODEL_PARAMS.copy()
    model_params['bed_selector_file'] = args.working_dir + f"/{args.exp_name}/all.{args.exp_name}.bed.out"
    message_to_sns("Start step3 search.")
    normalized_data = {}

    top_n_selector = BedFeatureSelector(
        bed_path=model_params['bed_selector_file'],
        top_n=10,
    )
    normalized_count = top_n_selector.fit_transform(normalized_counts)
    print(normalized_count.columns[1])
    for i in range(0, 6):
        train_x = normalized_count.loc[discovery.index][[normalized_count.columns[i]]]
        train_y = sample_info.loc[discovery.index]['target']

        pipe = (PipelineBuilder(model_params)
                .start_sub_pipeline('lr').add_lr_classifier().end_sub_pipeline()
                .start_sub_pipeline('cb').add_catboost_classifier().end_sub_pipeline()
                .add_voting_classifier()
                )
        trained_pipeline, oof_results, fold_results = train_pipeline(pipe, train_x, train_y,cv_splits=5, repeat=1, model_params=model_params)
        all_result = {
            'TRAIN_CV': oof_results,
        }
        cutoff = calculate_thresholds(train_y, oof_results)
        report = generate_report(all_result, sample_info, cutoff)
        report[0]['feature'] = normalized_count.columns[i]
        save_prediction(all_result, f"{exp_name}/top_n_feature_prediction_{i}.csv")

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

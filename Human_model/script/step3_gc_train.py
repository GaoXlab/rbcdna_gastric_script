import argparse
import os
from datetime import datetime

from hy.Estimator import ScoreOrderFilter
from hy.PipelineBuilder import PipelineBuilder
from sklearn.model_selection import ParameterGrid

from configs.params import MODEL_PARAMS
from hy.data_loader import load_normalized_data, load_sample_info
from hy.data_loader import load_separate_cohorts
from hy.evaluate import calculate_thresholds, generate_report, save_prediction, save_report
from hy.model import save_model, train_pipeline


def get_location(location):
    if location == "WORKING_DIR":
        return args.working_dir
    elif location == "MODEL_DATA":
        return os.path.join(args.working_dir, "modelData")
    elif location == "FR_LOCATION":
        return os.path.join(args.working_dir, "results/3_FeatureReduction")
    elif location == "RESULTS":
        return os.path.join(args.working_dir, "results/4_Classification")
    else:
        return os.path.join(args.working_dir, "results/")


def main(args):
    # 加载数据
    normalized_counts = load_normalized_data(get_location("WORKING_DIR"), args.exp_name)

    score_order_filter = ScoreOrderFilter(
        bed_path=get_location("WORKING_DIR") + "/all.gc.bed.out",
    )
    normalized_counts = score_order_filter.fit_transform(normalized_counts)

    discovery = load_separate_cohorts(get_location("MODEL_DATA"), args.exp_name, 'trn')
    sample_info = load_sample_info(get_location("MODEL_DATA"), args.exp_name)

    result_location = get_location("RESULTS")
    feature_selection_location = get_location("FR_LOCATION")
    X_train = normalized_counts.loc[discovery.index]
    y_train = sample_info.loc[discovery.index]['target']

    param_grid = {
        'n_pcas': range(10, 60, 10),
    }
    print(f"Searching for best params in {len(param_grid)} dim combinations")
    best_auc = 0
    best_param = best_cv_result = best_model = best_fold = None
    model_params = MODEL_PARAMS.copy()
    for current_param in ParameterGrid(param_grid):
        model_params.update(current_param)

        normal_pipeline = (
            PipelineBuilder(model_params)
            .start_sub_pipeline('rf')
                .add_pca_feature_combiner()
                .add_lasso_selector()
                .add_rf_classifier()
            .end_sub_pipeline()
            .start_sub_pipeline('lr')
                .add_pca_feature_combiner()
                .add_lasso_selector()
                .add_lr_classifier()
            .end_sub_pipeline()
            .add_voting_classifier()
        )

        # 训练并保存模型
        model, train_cv_oof_result, fold_pred = train_pipeline(normal_pipeline, X_train, y_train,
                                                               model_params=model_params)
        report = generate_report({'TRAIN_CV': train_cv_oof_result}, sample_info, [])
        print(f'{current_param},{report[0]["AUC"]}')
        if report[0]['AUC'] > best_auc:
            best_auc, best_model, best_param, best_cv_result, best_fold = report[0]['AUC'], model, current_param, train_cv_oof_result, fold_pred

    print(f"find best params {best_param}, AUC: {best_auc}")
    cutoffs = calculate_thresholds(y_train, best_cv_result)
    save_model(best_model, cutoffs, get_location("RESULTS"), args.exp_name)
    model_params.update(best_param)
    transform_pipeline = (
        PipelineBuilder(model_params)
        .add_pca_feature_combiner()
    )
    # save transform train X for future use
    X_train_transformed = transform_pipeline.build().fit_transform(X_train)
    X_train_transformed.to_csv(feature_selection_location + f"/{args.exp_name}.train.csv")
    report = generate_report({'train_cv': best_cv_result}, sample_info, cutoffs)
    save_report(report, result_location + f"/{args.exp_name}.train_cv.report.csv")
    save_prediction({'TRAIN_CV': best_cv_result}, result_location + f"/{args.exp_name}_prediction.csv")
    save_prediction(best_fold, result_location + f"/{args.exp_name}_fold_prediction.csv")


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

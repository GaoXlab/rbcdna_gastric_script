import argparse
import os
import shutil
import tempfile
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score

from configs.params import MODEL_PARAMS
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_sample_info, load_separate_cohorts
from hy.evaluate import calculate_thresholds, generate_report, save_report
from hy.model import train_pipeline
from hy.Estimator import BedFeatureSelector, PCABasedFeatureCombiner
from hy.message import message_to_sns


def main(args):
    exp_name = args.exp_name
    working_dir = args.working_dir
    npca_start = args.start
    npca_end = args.end
    # 1. 加载数据集与样本信息
    discovery = load_separate_cohorts('modelData', exp_name, 'trn')
    sample_info = load_sample_info('modelData', 'gc')
    normalized_counts = load_normalized_data(f"{working_dir}/{exp_name}", exp_name)
    model_params = MODEL_PARAMS.copy()

    model_params['bed_selector_file'] = f"{working_dir}/{exp_name}/all.{exp_name}.bed.out"

    tmp_dir = tempfile.mkdtemp(prefix='yaoxingyun_tmp')
    model_params['catboost_params'] = model_params['catboost_params'].copy()
    model_params['catboost_params']['train_dir'] = tmp_dir
    message_to_sns("Start step3_ori_topn_feature search.")

    # 2. 用 BedFeatureSelector 提取前 1000 个原始特征
    top_1000_selector = BedFeatureSelector(
        bed_path=model_params['bed_selector_file'],
        top_n=1000,
    )
    normalized_data_1000 = top_1000_selector.fit_transform(normalized_counts)

    train_x_all = normalized_data_1000.loc[discovery.index]
    train_y = sample_info.loc[discovery.index]['target']

    # 3. 按 PCABasedFeatureCombiner.py 第 61 行开始的注释计算 finalmodel 的 loadings 及 cumsum
    pca_params = model_params.get('pca_params', {})
    n_pcas = pca_params.get('n_pcas', 50)
    scaler_name = pca_params.get('scaler_name', 'StandardScaler')

    combiner = PCABasedFeatureCombiner(n_pcas=n_pcas, scaler_name=scaler_name)
    combiner.fit(train_x_all)

    r, c = combiner.pca_.components_.shape
    pc_columns = [f"PC{i + 1}" for i in range(r)]
    loadings_df = pd.DataFrame(
        combiner.loadings_.T,
        columns=pc_columns,
        index=train_x_all.columns,
    )

    weights = combiner.pca_.explained_variance_ratio_
    loadings_df['weighted_contribution'] = (loadings_df[pc_columns] ** 2).mul(weights, axis=1).sum(axis=1)
    loadings_df = loadings_df.sort_values(by='weighted_contribution', ascending=False)
    loadings_df['contribution_percent'] = (loadings_df['weighted_contribution'] / loadings_df['weighted_contribution'].sum()) * 100
    loadings_df['cumulative_contribution_percent'] = loadings_df['contribution_percent'].cumsum()

    # 保存 loadings 及累计贡献率表
    # os.makedirs(f"{working_dir}/results/3_FeatureReduction", exist_ok=True)
    # loadings_df.to_csv(f"{working_dir}/results/3_FeatureReduction/{exp_name}_loadings_cumsum.csv")

    # 获取按加权贡献度降序排列的原始特征列表
    sorted_features = loadings_df.index.tolist()

    # 4. 用 top10 到 top1000 的原始特征（10, 100, 200...，不用 PCA，只用 lr 和 cb 的均值）计算 AUC
    report_file = f"{working_dir}/results/3_FeatureReduction/{exp_name}_trncv_ori_topn_detail.csv"
    # if os.path.exists(report_file):
    #     os.remove(report_file)

    n_list = list(range(npca_start, npca_end + 1, 10))
    for n in n_list:
        print(f"Evaluating top {n} origin features...")
        top_n_features = sorted_features[:n]
        train_x_sub = train_x_all[top_n_features]

        # 不使用 PCA，只构建包含 lr 和 cb 的分类器组合 Pipeline (VotingClassifier 默认 soft voting 均值)
        pipe = (PipelineBuilder(model_params)
                .start_sub_pipeline('lr').add_lr_classifier().end_sub_pipeline()
                .start_sub_pipeline('cb').add_catboost_classifier().end_sub_pipeline()
                .add_voting_classifier()
                )

        trained_pipeline, oof_results, fold_results = train_pipeline(
            pipe, train_x_sub, train_y, cv_splits=5, model_params=model_params
        )

        cb_auc = roc_auc_score(train_y, oof_results['cb'])
        lr_auc = roc_auc_score(train_y, oof_results['lr'])

        all_result = {
            'TRAIN_CV': oof_results,
        }
        cutoff = calculate_thresholds(train_y, oof_results)
        report = generate_report(all_result, sample_info, cutoff)
        report[0]['top_feature'] = n
        report[0]['n_pcas'] = 0
        report[0]['cb_AUC'] = cb_auc
        report[0]['lr_AUC'] = lr_auc

        print(f"Top {n} origin features - AUC: {report[0].get('AUC')} - cb_AUC: {cb_auc:.4f} - lr_AUC: {lr_auc:.4f}")
        save_report(report, report_file)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    message_to_sns(f"Step3 origin topN feature search finished for {exp_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('--start', type=int, default=10)
    parser.add_argument('--end', type=int, default=1000)
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")



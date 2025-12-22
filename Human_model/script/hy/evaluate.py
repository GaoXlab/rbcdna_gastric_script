import math
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from pandas import json_normalize
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error,
    max_error, mean_absolute_percentage_error
)
def calculate_thresholds(y_true: np.ndarray, y_proba: np.ndarray, specificity_targets: tuple = (90, 95, 98)) -> dict:
    """根据交叉验证结果计算不同特异性要求的阈值

    Args:
        y_true: 真实标签 (ground truth)
        y_proba: 预测概率 (正类概率)
        specificity_targets: 目标特异性百分比列表 (0-100)

    Returns:
        dict: 键为特异性值，值为对应阈值及敏感性 {spec%: {'threshold': ..., 'sensitivity': ...}}
    """
    prob_df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_proba[0],
    }).sort_values(by='y_proba').reset_index(drop=True)
    prob_hd_df = prob_df[prob_df['y_true'] == 0].copy().reset_index()
    total_hd = len(prob_df[prob_df['y_true'] == 0])
    thresholds_dict = {}
    for target in specificity_targets:
        target_hd_index = math.floor(total_hd * (target / 100.0) + 1 + 1e-8)
        cutoff = prob_hd_df.loc[target_hd_index - 1, 'y_proba']
        min_cutoff = prob_df[prob_df['y_proba'] > cutoff]['y_proba'].min()
        thresholds_dict[f"{target}%"] = {
            'threshold': float(min_cutoff)
        }
    return thresholds_dict

def generate_report(t, info, cutoffs):
    results = []

    for t_type in t:
        pred = t[t_type]
        if isinstance(pred, pd.Series):
            pred = pred.to_frame(0)
        data = pred.join(info[['target', 'stage']], lsuffix='_left')
        # 如果有未知样本，默认是target0，
        data.fillna(0, inplace=True)
        all_data = data.copy()
        # Check if we have both positive and negative samples
        unique_targets = data['target'].unique()
        has_both_classes = len(unique_targets) > 1

        line = {'type': t_type}

        # Only calculate ROC and AUC if we have both classes
        if has_both_classes:
            fpr, tpr, _ = roc_curve(data['target'], data[0])
            try:
                roc_auc = round(roc_auc_score(data['target'], data[0]), 6)
            except ValueError:
                roc_auc = 0.0
            all_roc_auc = round(roc_auc_score(all_data['target'], all_data[0]), 6)
            line['AUC'] = roc_auc
            line['AUC-ALL'] = all_roc_auc

        else:
            line['AUC'] = 'N/A'
            line['AUC-ALL'] = 'N/A'

        if len(cutoffs) == 0:
            cutoffs = {'0.5': {'threshold': 0.5}}

        for spec in cutoffs:
            threshold = cutoffs[spec]['threshold']
            line[spec] = {}
            # Calculate stage-specific sensitivities
            for stage in ['I', 'II', 'III']:
                stage_data = data[data['stage'] == stage]
                stage_pos = stage_data[stage_data['target'] == 1]
                if len(stage_pos) > 0:
                    line[spec][f'sens-{stage}'] = round(stage_pos.apply(
                        lambda row: 1 if row[0] >= threshold else 0, axis=1).sum() / len(stage_pos), 3)
                else:
                    line[spec][f'sens-{stage}'] = 'N/A'

            # Calculate sensitivity only if we have positive samples
            if 1 in unique_targets:
                pos_samples = data[data['target'] == 1]
                line[spec]['sens'] = round(pos_samples.apply(
                    lambda row: 1 if row[0] >= threshold else 0, axis=1).sum() / len(pos_samples), 3)

            if 0 in unique_targets:
                neg_samples = data[data['target'] == 0]
                line[spec]['spec'] = round(neg_samples.apply(
                    lambda row: 1 if row[0] < threshold else 0, axis=1).sum() / len(neg_samples), 3)
            else:
                line[spec]['spec'] = 'N/A'

        results.append(line)
    return results

def save_report(metrics, filename):
    from filelock import FileLock  # 为了处理在HPC多节点多线程处理中的写文件问题，加上了hard file lock

    """保存指标到文本文件（使用filelock库）"""
    lock_file = filename + '.lock'

    with FileLock(lock_file):
        if not os.path.isfile(filename):
            json_normalize(metrics, max_level=3).to_csv(filename, index=False)
        else:
            json_normalize(metrics, max_level=3).to_csv(filename, mode='a', index=False, header=False)


def save_prediction(all_results, filename):
    # 为每个DataFrame添加对应的key列
    dfs = []
    for key, df in all_results.items():
        if isinstance(df, pd.Series):
            df = df.to_frame(0)
        df = df.copy()  # 避免修改原始DataFrame
        df['source_key'] = key  # 添加新列存储key
        dfs.append(df)

    # 合并所有DataFrame并保存，预测值保留8位小数
    pd.concat(dfs, axis=0).to_csv(filename, index=True, header=True, float_format="%.8f")
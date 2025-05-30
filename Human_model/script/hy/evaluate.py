import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def calculate_thresholds(y_true: np.ndarray,
                         y_proba: np.ndarray,
                         specificity_targets: tuple = (85, 90, 95)) -> dict:
    """根据交叉验证结果计算不同特异性要求的阈值
    
    Args:
        y_true: 真实标签 (ground truth)
        y_proba: 预测概率 (正类概率)
        specificity_targets: 目标特异性百分比列表 (0-100)
        
    Returns:
        dict: 键为特异性值，值为对应阈值及敏感性 {spec%: {'threshold': ..., 'sensitivity': ...}}
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    spec = 1 - fpr  # 特异性 = 1 - FPR

    thresholds_dict = {}
    for target in specificity_targets:
        target_spec = target / 100.0
        mask = spec >= target_spec

        if not np.any(mask):
            # 找不到满足条件的阈值，选择最接近的
            closest_idx = np.argmin(np.abs(spec - target_spec))
            closest_spec = round(spec[closest_idx], 3)
            if abs(closest_spec - target_spec) > 0.05:
                print(f"[Warning] 无法达到{target}%特异性，最相近为{closest_spec * 100:.1f}%")
            _threshold = thresholds[closest_idx]
            _sens = tpr[closest_idx]
        else:
            # 选择满足条件的最小阈值（最高灵敏度）   
            _threshold = thresholds[mask][-1]
            _sens = tpr[mask][-1]

        thresholds_dict[f"{target}%"] = {
            'threshold': float(_threshold),
            'sensitivity': float(_sens),
            'specificity': float(spec[mask][-1] if np.any(mask) else closest_spec)
        }

    return thresholds_dict


def save_report(metrics, filename):
    """保存指标到文本文件
    :param name:
    """
    # json_normalize(metrics).to_csv(filename, index=False)
    # return
    # 如果文件不存在，写入列名；否则追加数据
    if not os.path.isfile(filename):
        json_normalize(metrics).to_csv(filename, index=False)
    else:
        json_normalize(metrics).to_csv(filename, mode='a', index=False, header=False)

def generate_report(t, info, cutoffs, plot_auc=False, save_path=None):
    results = []

    # 如果要绘制AUC曲线，准备画布
    if plot_auc:
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.grid(True)

    for t_type in t:
        data = t[t_type].join(info[['target', 'stage']], lsuffix='_left')
        data = data[data['stage'] != '进展期腺瘤']

        #fill all nan target with 0
        data['target'] = data['target'].fillna(0)
        fpr, tpr, _ = roc_curve(data['target'], data[0])
        roc_auc = round(roc_auc_score(data['target'], data[0]), 4)

        line = {'type': t_type, 'AUC': roc_auc}

        # 绘制ROC曲线
        if plot_auc:
            plt.plot(fpr, tpr, label=f'{t_type} (AUC = {roc_auc})')

        for spec in cutoffs:
            threshold = cutoffs[spec]['threshold']
            line[spec] = {
                'sens': data.apply(lambda row: 1 if row['target'] == 1 and row[0] >= threshold else 0, axis=1).sum() /
                        data['target'].sum(),
                'sens-0I': data[data['stage'].isin(['I', 0])].apply(
                    lambda row: 1 if row['target'] == 1 and row[0] >= threshold else 0, axis=1).sum() /
                           data[data['stage'].isin(['I', 0])]['target'].sum(),
                'sens-II': data[data['stage'] == 'II'].apply(
                    lambda row: 1 if row['target'] == 1 and row[0] >= threshold else 0, axis=1).sum() /
                           data[data['stage'] == 'II']['target'].sum(),
                'sens-III': data[data['stage'] == 'III'].apply(
                    lambda row: 1 if row['target'] == 1 and row[0] >= threshold else 0, axis=1).sum() /
                            data[data['stage'] == 'III']['target'].sum(),
                'sens-III-IV': data[data['stage'].isin(['III', 'IV'])].apply(
                    lambda row: 1 if row['target'] == 1 and row[0] >= threshold else 0, axis=1).sum() /
                               data[data['stage'].isin(['III', 'IV'])]['target'].sum(),
                'spec': data.apply(lambda row: 1 if row['target'] == 0 and row[0] < threshold else 0,
                                   axis=1).sum() / sum(data['target'] == 0)
            }
            # round list to 3 decimal
            for k in line[spec]:
                line[spec][k] = round(line[spec][k], 3)
        results.append(line)

    if plot_auc:
        plt.legend(loc='lower right')
        if save_path:
            plt.savefig(save_path)  # 保存到文件
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()  # 尝试显示图形（可能有 GUI 或无效果）

    return results
def save_prediction(all_results, filename):
    # 为每个DataFrame添加对应的key列
    dfs = []
    for key, df in all_results.items():
        df = df.copy()  # 避免修改原始DataFrame
        df['source_key'] = key  # 添加新列存储key
        dfs.append(df)

    # 合并所有DataFrame并保存
    pd.concat(dfs, axis=0).to_csv(filename, index=True, header=True)
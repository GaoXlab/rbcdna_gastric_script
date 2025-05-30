import pandas as pd
from pathlib import Path


def load_normalized_data(file_path: str, model_type: str) -> pd.DataFrame:
    """加载预处理后的标准化数据"""
    return load_normalized_tab_file(Path(file_path) / f"all.{model_type}.tab")


def load_normalized_tab_file(file_path) -> pd.DataFrame:
    """加载预处理后的标准化数据"""
    cpm_file = file_path
    df = pd.read_csv(cpm_file, sep='\t', header=0, dtype={'#chr': str, 'start': int, 'end': int})
    df.columns = df.columns.str.replace(r"\.uniq\.nodup\.bam|'", "", regex=True)
    df.index = df.apply(lambda row: f"chr{str(int(row['#chr']))}:{str(int(row['start']))}-{str(int(row['end']))}",
                        axis=1).tolist()
    normalized_df = df.drop(columns=['#chr', 'start', 'end']).T
    return normalized_df


def load_sample_info(path: str, model_type: str) -> pd.DataFrame:
    """加载样本信息表"""
    sample_info = pd.read_csv(f'{path}/sampleinfo.{model_type}.txt', sep='\t', index_col=['seqID'])
    return sample_info


def load_separate_cohorts(path: str, model_type: str, cohort_type: str):
    return pd.read_csv(f"{path}/{model_type}.{cohort_type}.ids.txt", header=None, names=['seqID'], sep=',',
                       usecols=[0]).set_index('seqID')

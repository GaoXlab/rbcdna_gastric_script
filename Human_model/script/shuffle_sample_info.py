# 这个脚本用来生成
from hy.data_loader import load_sample_info, load_separate_cohorts

if __name__ == '__main__':
    sample_info = load_sample_info('modelData', 'gc')
    discovery = load_separate_cohorts('modelData', 'gc', 'trn')
    discovery_new = discovery.sample(frac=1, random_state=1637)
    rename_dict = dict(zip(discovery.index, discovery_new.index))
    sample_info.rename(index=rename_dict, inplace=True)
    sample_info.to_csv('modelData/sampleinfo.gc_shuffled.txt', sep='\t')
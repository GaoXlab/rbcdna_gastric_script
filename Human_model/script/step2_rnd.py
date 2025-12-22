import argparse
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from hy.data_loader import load_sample_info, load_separate_cohorts
from hy.message import message_to_sns


def get_location(location):
    if location == "WORKING_DIR":
        return args.working_dir
    elif location == "MODEL_DATA":
        return os.path.join(args.model_data)
    elif location == "SCRIPT":
        return os.path.join(args.script_dir)
    elif location == "REPORT":
        return os.path.join(args.working_dir, "results/3_FeatureReduction")
    else:
        return os.path.join(args.working_dir, "results")

def main(args):
    # p100 to p80 and p20
    sample_info = load_sample_info(get_location("MODEL_DATA"), 'gc_shuffled')
    seed = 1234
    # always generate p80_ids_path
    print(f"skip generating {args.exp_name}.ids.txt")
    p80 = load_separate_cohorts(get_location("MODEL_DATA"), args.exp_name, "trn")
    for i in range(1, 51):
        p64, _, _, _ = train_test_split(sample_info.loc[p80.index],
                                  sample_info.loc[p80.index]['target'],
                                  test_size=0.2,
                                  stratify=sample_info.loc[p80.index]['target'], #这个特意设置成了 target 维持总数
                                  random_state=i+seed)
        # 获取 p64 中 target=0 和 target=1 的数量
        count_0 = (p64['target'] == 0).sum()
        count_1 = (p64['target'] == 1).sum()

        # 取较少的一个类别数量
        min_count = min(count_0, count_1)

        # 对 target=0 和 target=1 分别采样 min_count 个样本
        balanced_p64 = pd.concat([
            p64[p64['target'] == 0].sample(n=min_count, random_state=i + seed),
            p64[p64['target'] == 1].sample(n=min_count, random_state=i + seed)
        ])

        with open(get_location("WORKING_DIR")+f"/all.{args.exp_name}.sample.info.{i}", 'w') as f:
            f.write(f"{len(balanced_p64)}\n")
            for index, row in balanced_p64.iterrows():
                f.write(f"{index} {row['target']} - -2 0 -1\n")

    # 生成 train.tab 文件
    script_dir = get_location("SCRIPT")
    model_data_dir = get_location("MODEL_DATA")
    # 这里是为了生成train.tab文件
    print(f"bash {script_dir}/make_tab.sh {model_data_dir}/{args.exp_name}.trn.ids.txt trim_q30_gcc_10k_cpm {args.working_dir}/train.tab")
    os.system(f"bash {script_dir}/make_tab.sh {model_data_dir}/{args.exp_name}.trn.ids.txt trim_q30_gcc_10k_cpm {args.working_dir}/train.tab")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('script_dir', help='工作目录')
    parser.add_argument('model_data', help='工作目录')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")

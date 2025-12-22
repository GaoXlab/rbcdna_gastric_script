import argparse
import os
from datetime import datetime

from hy.Enviroment import get_available_cpus
from joblib import Parallel, delayed

from hy.tab_files import aggregate_tab_file


def get_location(location):
    if location == "WORKING_DIR":
        return args.working_dir
    elif location == "MODEL_DATA":
        return args.model_data_dir
    elif location == "SCRIPT":
        return args.script_dir
    elif location == "REPORT":
        return os.path.join(args.working_dir, "results/3_FeatureReduction")
    else:
        return os.path.join(args.working_dir, "results")

def process_task(task_id, working_dir, script_dir):
    # 动态生成文件路径（例如 train_1.tab, train_2.tab, ...）
    tab_file = os.path.join(working_dir, f"train.tab")
    blacklist_file = os.path.join(script_dir, "blacklist.bed")
    genome_file = os.path.join(script_dir, "genome.txt")

    # 调用目标函数
    return aggregate_tab_file(tab_file, task_id, blacklist_file, genome_file)

def main(args):
    # 并行处理 1-100 的任务
    # message_to_feishu(f"生成train.tab文件完成，开始处理10k-1m任务")
    tab_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', args.tab_id))
    multi = args.multi
    print(f"生成train.tab文件完成，开始处理10k-1m任务, tab_id: {tab_id}, multi: {multi}")
    #process_task(tab_id, args.working_dir, args.script_dir)
    Parallel(n_jobs=get_available_cpus())(  # 使用所有可用CPU核心
        delayed(process_task)(i, args.working_dir, args.script_dir) for i in range((tab_id-1) * multi+1, tab_id * multi+1)  # 生成1到100的任务ID
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('script_dir', help='脚本目录')
    parser.add_argument('model_data_dir', help='模型数据目录')
    parser.add_argument("--tab_id", type=int, default=1, help="tab文件编号，默认1")
    parser.add_argument("--multi", type=int, default=1, help="每个job处理的个数")
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")

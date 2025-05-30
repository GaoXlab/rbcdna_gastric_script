import argparse
import os
from datetime import datetime

from hy.Estimator import ScoreOrderFilter
from hy.data_loader import load_normalized_data, load_separate_cohorts, \
    load_normalized_tab_file
from hy.data_loader import load_sample_info
from hy.evaluate import save_prediction, generate_report
from hy.evaluate import save_report
from hy.model import load_model, run_pipeline


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
    if args.tab_file_location:
        normalized_counts = load_normalized_tab_file(args.tab_file_location)
    else:
        normalized_counts = load_normalized_data(get_location("WORKING_DIR"), args.exp_name)

    score_order_filter = ScoreOrderFilter(
        bed_path=get_location("WORKING_DIR") + "/all.gc.bed.out",
    )
    normalized_counts = score_order_filter.fit_transform(normalized_counts)
    sample_info = load_sample_info(get_location("MODEL_DATA"), args.exp_name)
    test = load_separate_cohorts(get_location("MODEL_DATA"), args.exp_name, args.test_name)

    # 准备训练数据
    pipeline, cutoffs = load_model(get_location("RESULTS"), args.exp_name)
    # 测试集1评估
    if test.index.size > 0:
        print(f"Testing {args.test_name} cohort, total samples: {test.index.size}")
        test_result = run_pipeline(pipeline, normalized_counts.loc[test.index])
        report = generate_report({args.test_name : test_result}, sample_info, cutoffs)
        save_prediction({args.test_name: test_result}, get_location('RESULTS') + f"/{args.exp_name}_{args.test_name}_prediction.csv")
        save_report(report, get_location('RESULTS') + f"/{args.exp_name}_{args.test_name}_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('test_name', help='测试的ids名称，请保证modelData文件夹中有 exp_name.test_name.ids.txt文件')
    parser.add_argument('tab_file_location', help='默认是工作目录下all.exp_name.tab文件', default=None, nargs='?')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")
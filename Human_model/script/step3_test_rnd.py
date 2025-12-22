import argparse
from datetime import datetime

from configs.params import MODEL_PARAMS
from hy.Estimator import BedFeatureSelector
from hy.PipelineBuilder import PipelineBuilder
from hy.data_loader import load_normalized_data, load_sample_info, load_separate_cohorts
from hy.evaluate import generate_report, save_prediction, save_report
from hy.model import load_model, run_pipeline


def main(args):
    exp_name = args.exp_name
    pipeline, cutoffs, n_features = load_model(args.working_dir + f"/results/4_Classification/", args.exp_name)

    normalized_counts = load_normalized_data(args.working_dir + f"/{exp_name}", exp_name)
    top_n_selector = BedFeatureSelector(
            bed_path=args.working_dir + f"/{args.exp_name}/all.{args.exp_name}.bed.out",
            top_n=n_features,
        )
    normalized_counts = top_n_selector.fit_transform(normalized_counts)
    sample_info = load_sample_info('modelData', 'gc.test')
    test = load_separate_cohorts('modelData', exp_name, args.test_name)

    test_result = run_pipeline(pipeline, normalized_counts.loc[test.index])

    separate_results = pipeline.transform(normalized_counts.loc[test.index])
    test_result['lr'] = separate_results[:, 1]
    test_result['cb'] = separate_results[:, 3]
    report = generate_report({args.test_name: test_result}, sample_info, cutoffs)

    model_params = MODEL_PARAMS.copy()
    model_params['pca_params'].update({
        'n_pcas': 50,
    })

    save_prediction({args.test_name: test_result}, args.working_dir + f"/results/4_Classification/{exp_name}_prediction_{args.test_name}.csv")
    save_report(report, args.working_dir + f"/results/4_Classification/{exp_name}_report_{args.test_name}.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('test_name', help='测试的ids名称，请保证modelData文件夹中有 exp_name.test_name.ids.txt文件')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")
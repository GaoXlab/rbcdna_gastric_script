import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime


def load_ids(model_data_dir, exp_name, cohort):
    """Read one sample id per line from modelData/{exp_name}.{cohort}.ids.txt."""
    path = os.path.join(model_data_dir, f"{exp_name}.{cohort}.ids.txt")
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def write_ids(path, sample_ids):
    with open(path, 'w') as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, help='Experiment name, e.g., gc')
    parser.add_argument('--model_data_dir', type=str, default='modelData', help='Path to modelData directory')
    args = parser.parse_args()

    exp_name = args.exp_name
    model_data_dir = os.path.abspath(args.model_data_dir)

    sample_info_path = os.path.join(model_data_dir, f"sampleinfo.{exp_name}.txt")
    trn_ids_path = os.path.join(model_data_dir, f"{exp_name}.trn.ids.txt")

    print(f"Loading sample info from {sample_info_path}")
    sample_info = pd.read_csv(sample_info_path, sep='\t', index_col=['seqID'])

    seed = 1234
    if os.path.exists(trn_ids_path):
        print(f"{trn_ids_path} already exists, skip p100 split")
    else:
        # Split p100 into p80 (trn) and p20 (test), stratified by stage.
        # This logic was moved here from step2.py so that both the main
        # pipeline and the full-CV test can bootstrap their ids.
        print(f"generating {exp_name}.ids.txt from p100")
        p100_ids = load_ids(model_data_dir, exp_name, "p100")
        p100 = sample_info.loc[p100_ids]
        if p100.index.duplicated().any():
            print(f"{exp_name}以下样本ID重复，可能存在配置问题，请检查！\n{p100.index[p100.index.duplicated()]}")
            exit(1)

        p80, p20, y_p80, y_p20 = train_test_split(p100,
                                                  p100['target'],
                                                  test_size=0.2,
                                                  stratify=p100['stage'],
                                                  random_state=seed)
        write_ids(trn_ids_path, p80.index)
        write_ids(os.path.join(model_data_dir, f"{exp_name}.neg.ids.txt"),
                  p80[p80['target'] == 0].index)
        write_ids(os.path.join(model_data_dir, f"{exp_name}.test.ids.txt"),
                  p20.index)

    print(f"Loading training IDs from {trn_ids_path}")
    trn_ids = load_ids(model_data_dir, exp_name, "trn")

    trn_info = sample_info[sample_info.index.isin(trn_ids)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1637)
    print("\nSplitting and saving folds...")

    for fold_idx, (train_index, val_index) in enumerate(skf.split(trn_info, trn_info['stage']), 1):
        train_ids = trn_info.index[train_index].tolist()
        val_ids = trn_info.index[val_index].tolist()

        train_ids = [x for x in sample_info.index if x in train_ids]
        val_ids = [x for x in sample_info.index if x in val_ids]

        train_ids = sorted(train_ids)
        val_ids = sorted(val_ids)

        trn_output_path = os.path.join(
            model_data_dir, f"{exp_name}_trncv_{fold_idx}.trn.ids.txt")
        test_output_path = os.path.join(
            model_data_dir, f"{exp_name}_trncv_{fold_idx}.test.ids.txt")

        write_ids(trn_output_path, train_ids)
        write_ids(test_output_path, val_ids)

        print(f"Saved fold {fold_idx}:")
        print(f"  {os.path.basename(trn_output_path)} with {len(train_ids)} samples.")
        print(f"  {os.path.basename(test_output_path)} with {len(val_ids)} samples.")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    print(f"Done in {datetime.now() - start_time}")

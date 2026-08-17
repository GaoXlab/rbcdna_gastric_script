import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

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

    print(f"Loading training IDs from {trn_ids_path}")
    with open(trn_ids_path, 'r') as f:
        trn_ids = [line.strip() for line in f if line.strip()]

    trn_info = sample_info[sample_info.index.isin(trn_ids)]
    seed = 1637
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    print("\nSplitting and saving folds...")

    for fold_idx, (train_index, val_index) in enumerate(skf.split(trn_info, trn_info['stage']), 1):
        train_ids = trn_info.index[train_index].tolist()
        val_ids = trn_info.index[val_index].tolist()

        train_ids = [x for x in sample_info.index if x in train_ids]
        val_ids = [x for x in sample_info.index if x in val_ids]

        train_ids = sorted(train_ids)
        val_ids = sorted(val_ids)

        trn_output_filename = f"{exp_name}_trncv_{fold_idx}.trn.ids.txt"
        test_output_filename = f"{exp_name}_trncv_{fold_idx}.test.ids.txt"

        trn_output_path = os.path.join(model_data_dir, trn_output_filename)
        test_output_path = os.path.join(model_data_dir, test_output_filename)

        with open(trn_output_path, 'w') as f:
            for sample_id in train_ids:
                f.write(f"{sample_id}\n")

        with open(test_output_path, 'w') as f:
            for sample_id in val_ids:
                f.write(f"{sample_id}\n")

        print(f"Saved fold {fold_idx}:")
        print(f"  {trn_output_filename} with {len(train_ids)} samples.")
        print(f"  {test_output_filename} with {len(val_ids)} samples.")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    print(f"Done in {datetime.now() - start_time}")

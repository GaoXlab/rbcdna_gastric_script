import sys
import glob
import dask.dataframe as dd
import dask
import multiprocessing

def process_files(file_type, n_workers=16):
    # 获取所有匹配的文件
    files = glob.glob(f"all.{file_type}.tab.*[0-9]")
    df_list = [dd.read_csv(file, header=None, sep='\t', blocksize=None) for file in files]
    new_df = df_list[0][[0, 1, 2]].copy()
    selected_dfs = [df.iloc[:, 3:7] for df in df_list]  # 注意Python切片是左闭右开区间
    merged_df = dd.concat(selected_dfs, axis=1)
    new_df['mean_3'] = merged_df[3].mean(axis=1)
    new_df['mean_4'] = merged_df[4].mean(axis=1)
    new_df['mean_5'] = 0
    new_df.to_csv(
        f"all.{file_type}.bed",
        sep='\t',
        header=False,
        index=False,
        single_file=True,
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("必须提供类型参数")
        sys.exit(1)

    file_type = sys.argv[1]

    # 可选：从命令行参数读取进程数
    n_workers = 16  # 默认使用全部线程

    process_files(file_type, n_workers)
import csv
import os
from typing import List, Dict


def aggregate_tab_file(input_file: str, agg: int, blacklist_file: str) -> None:
    """
    聚合处理tab文件的主要函数

    参数:
        input_file: 输入文件名
        agg: 聚合窗口大小
        blacklist_file: 黑名单bed文件路径
    """
    output_filename = f"{input_file}.{agg}"

    # 第一次读取文件获取seq_id
    with open(input_file, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t', quotechar="'")
        head = next(reader)
        seq_id = [v.split('.')[0] for v in head[3:]]

    # 处理数据并写入输出文件
    with open(input_file, 'r') as fp, open(output_filename, 'w+', newline='') as fp_output:
        reader = csv.reader(fp, delimiter='\t')
        writer = csv.writer(fp_output, delimiter='\t')

        # 写入表头
        head = next(reader)
        writer.writerow(head)
        col = len(head)

        pos = {}
        sum_data = {0: [0] * col}  # 使用sum_data避免与内置函数sum冲突
        i = 1

        for line in reader:
            # 处理位置信息
            pos[i] = line[:3]
            pos[i][2] = str(int(pos[i][1]) + agg * 10000)

            # 计算累加值
            sum_data[i] = sum_data[i - 1].copy()
            for j in range(3, col):
                sum_data[i][j] = sum_data[i - 1][j] + float(line[j])

            # 当累计足够行数时输出
            if i >= agg:
                output_index = i - agg + 1
                output_data = []
                for j in range(3, col):
                    output_data.append(round(sum_data[i][j] - sum_data[output_index - 1][j], 6))

                del sum_data[output_index - 1]  # 删除不再需要的数据
                writer.writerow(pos[output_index] + output_data)  # 写入输出行

            i += 1
            if i % 50000 == 1:
                print(f"{output_filename} {i}")  # 进度打印

    # 使用bedtools清理数据
    os.system(
        f"head -n 1 {output_filename} > {output_filename}.clean && "
        f"bedtools subtract -a {output_filename} -b {blacklist_file} -A -g /mnt/sfs-data/tab_files/genome.txt -sorted >> "
        f"{output_filename}.clean && mv {output_filename}.clean {output_filename}"
    )
    print(f"{output_filename} 处理完成")

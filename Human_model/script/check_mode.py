import os
import sys
import glob
from typing import List
from hy.message import message_to_sns

def main():
    # 获取命令行参数
    if len(sys.argv) < 2:
        print("Usage: python script.py <mode> [close_notify]")
        sys.exit(1)

    mode = sys.argv[1]
    close_notify = len(sys.argv) > 2 and sys.argv[2]  # 简化处理，可根据需要调整

    dir_path = mode

    if not close_notify:
        message_to_sns(f"开始check {mode} raw文件的完整性")

    # 获取所有.raw文件
    files = glob.glob(f"{dir_path}/cleaned/*.raw")

    # 读取索引文件
    index_file = f"{dir_path}/sorted.tab.index"
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            index = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"索引文件不存在: {index_file}")
        return

    i = 0
    error_count = 0
    total_count = 0

    for file_path in files:
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]

            # 获取文件ID（去掉扩展名）
            file_id = os.path.splitext(os.path.basename(file_path))[0]

            # 获取数据行（排除第一行）
            line_data = lines[1:] if len(lines) > 1 else []
            if len(line_data) == 0:
                # 删除空文件
                print(f"Deleted {file_path}")
                os.remove(file_path)
                origin_file = f"{dir_path}/origin/{file_id}.raw"
                if os.path.exists(origin_file):
                    os.remove(origin_file)
                error_count += 1
                continue
            # 检查行数是否匹配
            if len(index) != len(lines):
                if len(index) == len(lines) + 1 and lines and lines[0].isdigit():
                    # 在第一行插入bam信息
                    new_first_line = f"'{file_id}.uniq.nodup.bam'"
                    new_content = [new_first_line] + lines
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_content))
                    print(f"Cleaned {file_path}")
                else:
                    # 行数错误，删除文件
                    print(f"Line error {file_path} {len(index)} {len(lines)}")
                    os.remove(file_path)
                    origin_file = f"{dir_path}/origin/{file_id}.raw"
                    if os.path.exists(origin_file):
                        os.remove(origin_file)

                error_count += 1
                continue

            # 检查数据行是否全为0
            if line_data:
                try:
                    data_sum = sum(float(x) for x in line_data if x.replace('.', '').replace('-', '').isdigit())
                    if data_sum == 0:
                        print(f"Empty {file_path}")
                        os.remove(file_path)
                        origin_file = f"{dir_path}/origin/{file_id}.raw"
                        if os.path.exists(origin_file):
                            os.remove(origin_file)
                        error_count += 1
                        continue
                except ValueError:
                    # 如果转换失败，跳过求和检查
                    pass

            # 检查第一行是否包含bam但没有单引号
            if lines and "bam" in lines[0] and "'" not in lines[0]:
                new_first_line = f"'{file_id}.uniq.nodup.bam'"
                new_content = [new_first_line] + lines[1:]
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_content))
                print(f"Add ' {file_path}")
                error_count += 1

            total_count += 1

            # 每处理1000个文件输出进度
            if i % 1000 == 1:
                print(f"Processed {i} files")
            i += 1

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            error_count += 1

    # 发送完成通知
    if not close_notify:
        message_to_sns(
            f"完成check {mode} raw文件的完整性, 清理了 {error_count} 个文件, {total_count} 个文件检查通过")


if __name__ == "__main__":
    main()

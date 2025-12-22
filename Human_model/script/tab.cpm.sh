#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

name=$1
id=$2
cd $BAM_FILE_DIR || exit 1
BAM_FILE="${id}.nodup.q30.bam"
TAB_PATH="$MODEL_DATA_DIR"
if [ ! -f $BAM_FILE ]; then
  echo "$BAM_FILE not exists"
  exit 1
fi
if [ ! -f ${TAB_PATH}/${name}/origin/"${id}".raw ]
then
  echo "Working ${id}"
  touch "$TAB_PATH/${name}/origin/${id}.raw"
  TMP_FILE="$TAB_PATH/${name}/origin/${id}.raw.tmp.$$"
  bedtools coverage -a "${TAB_PATH}/${name}/sorted.tab.index" -b "$BAM_FILE" -F 0.5 -counts -sorted -g "${SCRIPT_DIR}"/genome.txt | cut -f4 > $TMP_FILE
  echo "'${id}.uniq.nodup.bam'" | cat - $TMP_FILE > "$TAB_PATH/${name}/origin/${id}.raw"
  rm $TMP_FILE
else
  echo "Skip ${id}"
  exit 1
fi
# 使用samtools idxstats命令来获取所有染色体的reads数量
# 过滤掉XYM染色体，使用awk累加其它染色体的reads数
[ ! -f "$BAM_FILE".bai ] && echo "Build Index" && samtools index $BAM_FILE -@ 6
TOTAL_COUNT=$(samtools idxstats $BAM_FILE | grep -v '^[XYM]' | awk '{SUM+=$3} END {print SUM}')

# 打印错误信息如果未能计算总读取数
if [ -z "$TOTAL_COUNT" ]; then
    echo "Error: Unable to calculate total reads from BAM file."
    exit 1
fi

# 现在遍历输入文件的每一行（跳过第一行头部信息）
# 并使用awk对每个区间的reads进行CPM标准化
# 输出到指定的输出文件
echo $TOTAL_COUNT
cat "${TAB_PATH}/${name}"/origin/"${id}".raw | awk -v total=$TOTAL_COUNT '{if (NR == 1) {print $0} else{cpm = ($1 / total) * 2000000 ;printf "%.6f\n", cpm} }' > "${TAB_PATH}/${name}"/cleaned/"${id}".raw

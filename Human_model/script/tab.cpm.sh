name=$1
id=$2
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
BASE_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)

if [ ! -f "${id}.10m.nodup.q30.bam" ]; then
    echo " BAM file not found."
    exit 1
fi

BAM_FILE="${id}.10m.nodup.q30.bam"
if [ ! -f $BASE_DIR/${name}/origin/"${id}".raw ]
then
  echo "Working ${id}"
  TMP_FILE="$BASE_DIR/${name}/origin/${id}.raw.tmp.$$"
  bedtools coverage -a "$BASE_DIR/${name}/sorted.tab.index" -b "${id}.10m.nodup.q30.bam" -counts -sorted -g $SCRIPT_DIR/genome.txt | cut -f4 > $TMP_FILE
  echo "'${id}.uniq.nodup.bam'" | cat - $TMP_FILE > "$BASE_DIR/${name}/origin/${id}.raw"
  rm $TMP_FILE
else
  echo "Skip ${id}"
  exit 1
fi
# use samtools idxstats to get the total number of reads in the BAM file
# if the index file does not exist, build it first
[ ! -f "$BAM_FILE".bai ] && echo "Build Index" && samtools index $BAM_FILE -@ 6
# get total reads from BAM file excluding chrX, chrY, chrM
TOTAL_COUNT=$(samtools idxstats $BAM_FILE | grep -v '^[XYM]' | awk '{SUM+=$3} END {print SUM}')

# calculate CPM reads for each feature in the tab file, and write to a new file
echo $TOTAL_COUNT
cat $BASE_DIR/"${name}"/origin/"${id}".raw | awk -v total=$TOTAL_COUNT '{if (NR == 1) {print $0} else{cpm = ($1 / total) * 10000000 ;printf "%.1f\n", cpm} }' > $BASE_DIR/"${name}"/cleaned/"${id}".raw

name=$1
id=$2
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
BASE_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)
BAM_FILE="${id}.nodup.q30.bam"
if [ ! -f $BAM_FILE ]; then
  echo "$BAM_FILE not exists"
  exit 1
fi
if [ ! -f "${BASE_DIR}"/${name}/origin/"${id}".raw ]
then
  echo "Working ${id}"
  bedtools coverage -a "${BASE_DIR}/${name}/sorted.tab.index" -b "$BAM_FILE" -F 0.5 -counts -sorted -g "$BASE_DIR"/genome.txt | cut -f4 > "$BASE_DIR/${name}/origin/${id}.raw"
  sed -i "1i '${id}.uniq.nodup.bam'" "${BASE_DIR}/${name}/origin/${id}.raw"
else
  echo "Skip ${id}"
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
cat "${BASE_DIR}/${name}"/origin/"${id}".raw | awk -v total=$TOTAL_COUNT '{if (NR == 1) {print $0} else{cpm = ($1 / total) * 2000000 ;printf "%.6f\n", cpm} }' > "${BASE_DIR}/${name}"/cleaned/"${id}".raw

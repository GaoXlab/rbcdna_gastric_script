#!/bin/bash
GENOME_TYPE=$1  # e.g., GRCh38 or mm10
# Define URLs and output filenames based on genome type
case "$GENOME_TYPE" in
  "GRCh38")
    URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/"
    OUTPUT_FA="GRCh38.fa"
    ;;
  "mm10")
    URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes/"
    OUTPUT_FA="mm10.fa"
    ;;
  *)
    echo "Error: Unsupported genome type '$GENOME_TYPE'. Use 'GRCh38' or 'mm10'."
    exit 1
    ;;
esac

# Create a unique tmp directory in case overlap
TMP_DIR="tmp_$(date +%s)"
mkdir -p "$TMP_DIR" || { echo "Failed to create tmp dir"; exit 1; }

# Download files to tmp dir, excluding *_alt.fa.gz files
echo "Downloading $GENOME_TYPE chromosomes to $TMP_DIR..."
curl -s "$URL" |
  grep -oP 'href="\Kchr[^"]+\.fa\.gz(?=")' |  # 匹配所有chr开头的fa.gz文件
  grep -v '_alt\.fa\.gz' |                    # 排除包含_alt的文件
  awk -v url="$URL" '{print url $0}' |
  while read -r original_url; do
    chr_name=$(basename "$original_url" .fa.gz)
    new_name="$TMP_DIR/${chr_name}_online_version.fa.gz"
    wget -q -O "$new_name" "$original_url" || { echo "Download failed: $original_url"; exit 1; }
    echo "Downloaded: $chr_name"
  done

# Merge, remove 'chr' prefix, and clean up
echo "Building $OUTPUT_FA..."
zcat "$TMP_DIR"/*.fa.gz | sed 's/^>chr/>/' > "$OUTPUT_FA" || { echo "Failed to merge files"; exit 1; }

samtools faidx "$OUTPUT_FA"
echo "Generating ${GENOME_TYPE}.chrom.sizes..."
awk '{print $1 "\t" $2}' "${OUTPUT_FA}.fai" > "${GENOME_TYPE}.chrom.sizes"
echo "Generating ${GENOME_TYPE}.genome..."
cp "${GENOME_TYPE}.chrom.sizes" "${GENOME_TYPE}.genome"
bwa-mem2-2.2.1_x64-linux/bwa-mem2 index -p "$OUTPUT_FA" "$OUTPUT_FA"

# Delete tmp dir
rm -rf "$TMP_DIR"
echo "Done. Output: $OUTPUT_FA"

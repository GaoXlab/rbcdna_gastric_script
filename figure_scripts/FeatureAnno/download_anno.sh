#!/bin/bash

TARGET_DIR="$(pwd)/figure_scripts/FeatureAnno/refData"

mkdir -p "$TARGET_DIR" "$TARGET_DIR"/CellResearch

cd "$TARGET_DIR" || exit 1

# Download CR data
curl -L https://raw.githubusercontent.com/GaoXlab/MNDNA_scripts_forSun/main/Human/02.mndna_enriched_peak_calling/data/ref.RData > "$TARGET_DIR"/CellResearch/ref.RData
curl -L https://raw.githubusercontent.com/GaoXlab/MNDNA_scripts_forSun/main/Human/02.mndna_enriched_peak_calling/data/HD_deep_10sample.RData > "$TARGET_DIR"/CellResearch/HD_deep_10sample.RData
curl -L https://raw.githubusercontent.com/GaoXlab/MNDNA_scripts_forSun/main/Human/02.mndna_enriched_peak_calling/result/HD_deep_10sample_anno.RData > "$TARGET_DIR"/CellResearch/HD_deep_10sample_anno.RData
curl -L https://raw.githubusercontent.com/GaoXlab/MNDNA_scripts_forSun/main/Human/02.mndna_enriched_peak_calling/result/result_d_0_merge_t_rbcDNA_c_gDNA.10samples.broadPeak.bed > "$TARGET_DIR"/CellResearch/result_d_0_merge_t_rbcDNA_c_gDNA.10samples.broadPeak.bed

curl -L https://raw.githubusercontent.com/ernstlab/full_stack_ChromHMM_annotations/main/state_annotations_processed.csv > state_annotations_processed.csv
python3 -c "
import csv, sys
with open('state_annotations_processed.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 5:
            print(f'{row[0]}\t{row[1]}\t{row[4]}')
" > state_annotations_processed

wget "https://public.hoffman2.idre.ucla.edu/ernst/UUKP7/hg38lift_genome_100_segments.bed.gz"

gzip -d hg38lift_genome_100_segments.bed.gz
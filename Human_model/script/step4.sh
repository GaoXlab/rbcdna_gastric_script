#!/bin/bash

# You should run this script in the parent dir of 'Human_model'

mkdir Figures -p

WORKING_DIR="$(pwd)"

# Build sample_info.RData and prediction_final.Data
Rscript figure_scripts/Build_RData.R "$(pwd)"

# Build train_gam_100k.RData
cd Human_model || exit 1
script/build_feature_data.sh modelData/gc.trn.ids.txt trim_gcc_r100k_0start
# mkdir trim_gcc_r100k_0start -p

cd "$WORKING_DIR" || exit 1
# Build chromatin states annotation data
## Download anno data from official web site
figure_scripts/FeatureAnno/download_anno.sh "$(pwd)"
cp Human_model/results/2_FeatureSelection/all.gc.bed.out figure_scripts/FeatureAnno/gc.bed
python figure_scripts/FeatureAnno/chrom_full_stack_batch.py figure_scripts/FeatureAnno/refData figure_scripts/FeatureAnno figure_scripts/FeatureAnno/gc.bed

cut -f 1-3 Human_Model/gc/train.tab.* > figure_scripts/FeatureAnno/total_1000w.bed

# Build quality control data
cd Human_model ||  exit 1
script/qc_reads_check.sh "./bams/"
mv TotalSample_MT.noalt.log ../Figures/

cd "$WORKING_DIR" || exit 1
# Build r_g_10controls_smooth.RData and trn_nonAN_smooth_anno.RData
mkdir -p Figures/QDNA_bin_results
xargs -n 1 -I %1 -P 16 -a Human_model/modelData/rbcDNA_regions.ids.txt Rscript figure_scripts/build_QDNAseq_bin_count.R Human_model/bams/pipeline_trimmomatic/%1.nodup.q30.bam 100 hg38 Figures/QDNA_bin_results
xargs -n 1 -I %1 -P 16 -a Human_model/modelData/rbcDNA_regions.ids.txt Rscript figure_scripts/build_QDNAseq_bin_count.R Human_model/bams/pipeline_trimmomatic/%1.nodup.q30.bam 1000 hg38 Figures/QDNA_bin_results
xargs -n 1 -I %1 -P 16 -a Human_model/modelData/gc.neg.ids.txt Rscript figure_scripts/build_QDNAseq_bin_count.R Human_model/bams/pipeline_trimmomatic/%1.nodup.q30.bam 100 hg38 Figures/QDNA_bin_results
xargs -n 1 -I %1 -P 16 -a Human_model/modelData/gc.neg.ids.txt Rscript figure_scripts/build_QDNAseq_bin_count.R Human_model/bams/pipeline_trimmomatic/%1.nodup.q30.bam 1000 hg38 Figures/QDNA_bin_results
xargs -n 1 -I %1 -P 16 -a Human_model/modelData/sampleinfo.gc.storagetime.txt Rscript figure_scripts/build_QDNAseq_bin_count.R Human_model/bams/pipeline_trimmomatic/%1.nodup.q30.bam 100 hg38 Figures/QDNA_bin_results
xargs -n 1 -I %1 -P 16 -a Human_model/modelData/sampleinfo.gc.storagetime.txt Rscript figure_scripts/build_QDNAseq_bin_count.R Human_model/bams/pipeline_trimmomatic/%1.nodup.q30.bam 1000 hg38 Figures/QDNA_bin_results

Rscript figure_scripts/Build_Controls_Smooth_Data.R "$(pwd)"
Rscript figure_scripts/Build_Smooth_Anno_Data.R "$(pwd)"

# Start Run Fig*.R
for fig in 2 3 4 5 6 S2 S3 S4 S5 S6 S7 S8 S9 S11; do
  Rscript figure_scripts/Figure"$fig".R "$(pwd)"
done

Rscript figure_scripts/Table1.BasicCharacteristic.R "$(pwd)"
Rscript figure_scripts/Table2.Performance.R "$(pwd)"

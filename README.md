# rbcdna_gastric_script

This project provides a bioinformatics pipeline for rbcDNA whole-genome sequencing data analysis. The preprocessing pipeline and the subsequent modeling analysis for human samples are organized in the `Human_model` directory. The `Figure` directory contains the code used to generate all figures in the article.

## 1. Directory structure
```text
├── Figure
└── Human_model
    ├── fq
    ├── bams
    ├── modelData
    ├── results    # model outputs and evaluation results
    └── script     # analysis and modeling scripts
```

## 2. Preprocessing of the rbcDNA WGS data
The analysis pipeline for preprocessing the rbcDNA whole-genome sequencing (WGS) data is `01.pipeline_preprocess.sh`. (The human reference genomes (hg38) used is the "no alt" version, which excludes alternate contigs (_alt).)  
The pipeline reads the locations of the input FastQ files, the output directory, and the genome type from environment variables. Ensure that the specified input directory contains the paired-end files `sample_name_1.fq.gz` and `sample_name_2.fq.gz` before execution.  
To generate the primary reference genome file, run:
```bash
./00.build_primary_fa.sh GRCh38  # Generates GRCh38.fa
```
Then set the required environment variables and run the preprocessing pipeline:
```bash
export SAMPLE_NAME=$SAMPLE_NAME;export SOURCE=$SOURCE_DIR;export OUTPUT_DIR=$OUTPUT_DIR;export GENOME_TYPE=$GENOME_TYPE;./01.pipeline_preprocess.sh
```
For human rbcDNA WGS samples, an additional preprocessing step is required to generate GC-content calibration files using the following script:
```bash
# $TARGET_DIR should be set to ./bams/
./02.batch_build_gcc_files.sh $OUTPUT_DIR $TARGET_DIR
```
## 3. Subsequent modeling analysis for human samples  

- **./Human_model/**: Contains the scripts and processes for constructing classification models to detect gastric cancer (GC) patients based on tumor-associated rbcDNA regions. Detailed execution steps and analysis procedures are provided within the corresponding subdirectories.  

## 4. Comments for figure generation
Note: Run this script from the parent directory of the working directory. It will automatically generate all figures and table data used in the manuscript. This step requires a server with at least 48 CPU cores and 96 GB RAM. The file Supplementary_Tables.xlsx must be placed in the Figures directory.
```bash
./Human_Model/script/step4.sh `pwd`
```

## Software version and hardware requirements

- bwa-mem2@2.2.1
- samtools@1.10 (using htslib 1.10.2)
- bedtools@2.27.1
- deeptools@3.3.2
- Python@3.8
- R@4.2.2+



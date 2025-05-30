# rbcdna_gastric_script

This project provides a bioinformatics analysis pipeline for preprocessing and analyzing the rbcDNA whole-genome sequencing data. The preprocessing pipeline is implemented in shell scripts, and the subsequent analysis for human and mouse samples is organized in the `Human` and `Mouse` folders, respectively. The `Figure` folder contains the code for generating each figure in the article.

## 1. Directory structure
```text
├── Figure
└── Human_model
    ├── bam
    ├── modelData
    ├── results
    └── script
```

## 2. Preprocessing the rbcDNA WGS data
The analysis pipeline for preprocessing the rbcDNA whole-genome sequencing data is `01.pipeline_preprocess.sh`.  
The pipeline reads the location of the FastQ files and the output directory from environment variables, as well as the type of genome. When using the pipeline, ensure that the specified locations contain the `sample_name_1.fq.gz` and `sample_name_2.fq.gz` files.
```bash
export SAMPLE_NAME=$SAMPLE_NAME;export SOURCE=$SOURCE_DIR;export OUTPUT_DIR=$OUTPUT_DIR;export GENOME_TYPE=$GENOME_TYPE;./01.pipeline_preprocess.sh
```

For human whole-genome sequencing rbcDNA samples, you need to run an additional preprocessing step using the 02.batch_build_gcc_files.sh script to generate GC-content calibration files. 

```bash
# $TARGET_DIR should be set to Human_model/bam/
./02.batch_build_gcc_files.sh $OUTPUT_DIR $TARGET_DIR
```
## 3. Subsequent Analysis for Human Samples  

- **./Human_model/**: Contains the scripts and processes for building the model to detect GC patients using tumor-associated rbcDNA regions. The specific running process is described in the subfolders.  

## Software version and hardware requirements

- bwa-mem2@2.2.1
- samtools@1.10
- bedtools@2.27.1
- php@7.4
- deeptools@3.3.2
- Python@3.8
- R@4.2.2+


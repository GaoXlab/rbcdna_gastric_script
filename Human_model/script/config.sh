#!/bin/bash
# config.sh - 路径配置

export SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
export PROJECT_ROOT=$(cd $SCRIPT_DIR/..; pwd)

# 结果目录
export FEATURE_SELECTION_OUTPUT_DIR=$PROJECT_ROOT/results/2_FeatureSelection
export FEATURE_REDUCTION_OUTPUT_DIR=$PROJECT_ROOT/results/3_FeatureReduction
export FEATURE_CLASSIFICATION_DIR=$PROJECT_ROOT/results/4_Classification
export MODEL_DATA_DIR=$PROJECT_ROOT/modelData
export BAM_FILE_DIR=$PROJECT_ROOT/bams

mkdir -p $FEATURE_SELECTION_OUTPUT_DIR \
         $FEATURE_REDUCTION_OUTPUT_DIR \
         $FEATURE_CLASSIFICATION_DIR \
         $MODEL_DATA_DIR
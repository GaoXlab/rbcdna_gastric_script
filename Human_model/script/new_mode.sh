#!/bin/bash
WORKING_DIR=`pwd`

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
FEATURE_SELECTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/2_FeatureSelection; pwd)
FEATURE_REDUCTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/3_FeatureReduction; pwd)
FEATURE_CLASSIFICATION_DIR=$(cd $SCRIPT_DIR/../results/4_Classification; pwd)

MODEL_DATA_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)

NAME=$1
TAB_NAME=$2
if [ ! -n "$NAME" ]; then
    echo "usage new_mode.sh mode_name"
    exit
fi
if [ -d "$MODEL_DATA_DIR/$NAME" ]; then
    mkdir -p "$MODEL_DATA_DIR/to_be_deleted"
    # remove the old model data
    mv "$MODEL_DATA_DIR/$NAME" "$MODEL_DATA_DIR/to_be_deleted"
    rm -rf "$MODEL_DATA_DIR/to_be_deleted"
fi
cp -r "$MODEL_DATA_DIR/empty" "$MODEL_DATA_DIR/$NAME"
bedtools sort -g "$SCRIPT_DIR/genome.txt" -i $TAB_NAME | cut -f 1-3 >> "$MODEL_DATA_DIR/$NAME/sorted.tab.index"
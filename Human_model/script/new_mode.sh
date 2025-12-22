#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

NAME=$1
TAB_FILE=$2

if [ ! -n "$NAME" ]; then
    echo "usage new_mode.sh mode_name"
    exit
fi
if [ -d "$MODEL_DATA_DIR/$NAME" ]; then
    rm -rf "${MODEL_DATA_DIR:-not_real_dir}/${NAME:-not_real_name}"
fi

cp -r "$MODEL_DATA_DIR"/empty "$MODEL_DATA_DIR/$NAME"
cut -f 1-3 $TAB_FILE > $TAB_FILE.tmp
bedtools sort -g "$SCRIPT_DIR"/genome.txt -i $TAB_FILE.tmp >> $MODEL_DATA_DIR"/$NAME/sorted.tab.index"
rm $TAB_FILE.tmp
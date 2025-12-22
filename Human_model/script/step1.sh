#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

TYPE=$1

"$SCRIPT_DIR"/build_feature_data.sh "$MODEL_DATA_DIR"/"$TYPE".p100.ids.txt trim_q30_gcc_10k_cpm

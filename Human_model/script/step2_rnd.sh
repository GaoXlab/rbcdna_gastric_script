#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
FEATURE_SELECTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/2_FeatureSelection; pwd)
FEATURE_REDUCTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/3_FeatureReduction; pwd)
FEATURE_CLASSIFICATION_DIR=$(cd $SCRIPT_DIR/../results/4_Classification; pwd)
MODEL_DATA_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)

ORI_TYPE=$1
TYPE="$ORI_TYPE"_rnd

for ID_TYPE in trn "test"; do
  cp -v $MODEL_DATA_DIR/"$ORI_TYPE".$ID_TYPE.ids.txt $MODEL_DATA_DIR/"$TYPE".$ID_TYPE.ids.txt
done

message() {
    local message="$1"
    # 检查环境变量
    echo "[$TYPE] $message"
}

message "Making train.tab"

mkdir $TYPE -p
WORKING_ROOT=$(pwd)

cd $TYPE || exit 1

python $SCRIPT_DIR/step2_rnd.py $TYPE `pwd` "$SCRIPT_DIR" "$MODEL_DATA_DIR"

exit_code=$?
if [ ! $exit_code -eq 0 ]; then
    message "step2.py 执行失败，退出码：$exit_code"
    exit 1
fi
message "Build 10-1000k tab"
seq 1 4 | xargs -n 1 -P 2 -I %1 python $SCRIPT_DIR/step2_build_tab.py "$TYPE" "$WORKING_DIR" "$SCRIPT_DIR" "$MODEL_DATA_DIR" --tab_id %1 --multi 25
message "Start feature selection"
seq 1 50 | xargs -n 1 -I %1 -P 3 $SCRIPT_DIR/fs.sh $TYPE %1 16
message "Feature selection finished"
# merge all 50 top 1000 feature scores
message "Merge feature scores"
python $SCRIPT_DIR/merge_p80.py $TYPE
message "Merge feature scores finished"

$SCRIPT_DIR/bed_select all.$TYPE.bed all.$TYPE.bed.out 1000

rm train.tab*
rm all.*.tab.*[0-9]
cat all.$TYPE.bed.out | cut -f1-3 > $FEATURE_SELECTION_OUTPUT_DIR/all.$TYPE.bed.out
# mv all.$TYPE.bed $FEATURE_SELECTION_OUTPUT_DIR

message "Start building new mode"

"$SCRIPT_DIR"/new_mode.sh "$TYPE" all."$TYPE".bed.out
"$SCRIPT_DIR"/build_feature_data.sh "$MODEL_DATA_DIR"/gc.all.ids.txt "$TYPE"
python "$SCRIPT_DIR"/check_mode.py $TYPE

$SCRIPT_DIR/make_all_tab.sh "$TYPE" "all.$TYPE.tab"

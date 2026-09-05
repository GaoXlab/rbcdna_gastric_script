#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

TYPE=$1

message() {
    local message="$1"
    # 这里可以增加其他通知方式，样例中只是shuffle了一下
    echo "[$TYPE] $message"
}

message "Making train.tab"

mkdir $TYPE -p
WORKING_ROOT=$(pwd)

cd $TYPE || exit 1

python $SCRIPT_DIR/step2.py $TYPE `pwd` "$SCRIPT_DIR" "$MODEL_DATA_DIR"

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
nfiles=$(ls all.$TYPE.tab.*[0-9] 2>/dev/null | wc -l)
if [ "$nfiles" -ne 50 ]; then
   message "FS 分片数量异常：$nfiles（期望 50），终止 $TYPE"
   exit 1
fi
python $SCRIPT_DIR/merge_p80.py $TYPE
exit_code=$?
if [ ! $exit_code -eq 0 ]; then
   message "merge_p80.py 执行失败，退出码：$exit_code"
   exit 1
fi
message "Merge feature scores finished"

$SCRIPT_DIR/bed_select all.$TYPE.bed all.$TYPE.bed.out 1000
exit_code=$?
if [ ! $exit_code -eq 0 ]; then
   message "bed_select 执行失败，退出码：$exit_code"
   exit 1
fi

rm train.tab*
rm all.*.tab.*[0-9]
cat all.$TYPE.bed.out | cut -f1-3 > $FEATURE_SELECTION_OUTPUT_DIR/all.$TYPE.bed.out
# mv all.$TYPE.bed $FEATURE_SELECTION_OUTPUT_DIR

message "Start building new mode"

if [ -d "$MODEL_DATA_DIR/$TYPE/cleaned" ] && [ -n "$(ls -A "$MODEL_DATA_DIR/$TYPE/cleaned" 2>/dev/null | head -1)" ]; then
    message "modelData/$TYPE/cleaned already exists, reuse per-sample features and rebuild all.$TYPE.tab"
    python "$SCRIPT_DIR"/check_mode.py "$MODEL_DATA_DIR"/"$TYPE"
    "$SCRIPT_DIR"/make_all_tab.sh "$TYPE" "all.$TYPE.tab"
else
    "$SCRIPT_DIR"/new_mode.sh "$TYPE" all."$TYPE".bed.out
    "$SCRIPT_DIR"/build_feature_data.sh "$MODEL_DATA_DIR"/gc.p100.ids.txt "$TYPE"
    python "$SCRIPT_DIR"/check_mode.py "$MODEL_DATA_DIR"/"$TYPE"
    "$SCRIPT_DIR"/make_all_tab.sh "$TYPE" "all.$TYPE.tab"
fi

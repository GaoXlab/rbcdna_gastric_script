#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

EXP_NAME=$1

if [ -z "$EXP_NAME" ]; then
  echo "Usage: $0 <exp_name>"
  exit 1
fi

message() {
    local message="$1"
    echo "[$EXP_NAME] $message"
}

cd $PROJECT_ROOT || exit 1

message "Phase 1: Split 5-fold"
python $SCRIPT_DIR/split_5_fold_full_cv.py $EXP_NAME
exit_code=$?
if [ ! $exit_code -eq 0 ]; then
   message "split_5_fold_full_cv.py 执行失败，退出码：$exit_code"
   exit 1
fi

message "Phase 2: Processing folds"
for i in {1..5}; do
  message "Running step2.sh for fold $i..."
  $SCRIPT_DIR/step2.sh ${EXP_NAME}_trncv_$i
  exit_code=$?
  if [ ! $exit_code -eq 0 ]; then
     message "step2.sh (fold $i) 执行失败，退出码：$exit_code"
     exit 1
  fi
done

message "Phase 3: Processing random folds"
for i in {1..5}; do
  message "Running step2_rnd.sh for fold $i..."
  $SCRIPT_DIR/step2_rnd.sh ${EXP_NAME}_trncv_$i
  exit_code=$?
  if [ ! $exit_code -eq 0 ]; then
     message "step2_rnd.sh (fold $i) 执行失败，退出码：$exit_code"
     exit 1
  fi
done

message "Phase 4: Evaluating normal folds"
python $SCRIPT_DIR/evaluate_full_cv.py --basename $EXP_NAME --working_dir $PROJECT_ROOT
exit_code=$?
if [ ! $exit_code -eq 0 ]; then
   message "evaluate_full_cv.py (normal) 执行失败，退出码：$exit_code"
   exit 1
fi

message "Phase 5: Evaluating random folds"
python $SCRIPT_DIR/evaluate_full_cv.py --basename $EXP_NAME --rnd --working_dir $PROJECT_ROOT
exit_code=$?
if [ ! $exit_code -eq 0 ]; then
   message "evaluate_full_cv.py (random) 执行失败，退出码：$exit_code"
   exit 1
fi

message "All tasks finished successfully!"

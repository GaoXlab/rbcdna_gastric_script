#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
FEATURE_SELECTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/2_FeatureSelection; pwd)
FEATURE_REDUCTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/3_FeatureReduction; pwd)
FEATURE_CLASSIFICATION_DIR=$(cd $SCRIPT_DIR/../results/4_Classification; pwd)
MODEL_DATA_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)
FILE_LOCATION_BAMS=$(cd $SCRIPT_DIR/../../bam; pwd)
OUTPUT_PREFIX=$1
TYPE=$OUTPUT_PREFIX

message() {
  local message="$1"
  echo $message
}

message "Making train.tab"

python $SCRIPT_DIR/step2_gc.py $TYPE `pwd`

message "Start feature selection"
# calc 10m feature scores and select top 1000 for 50 random repeat
seq 1 50 | xargs -n 1 -I %1 -P 3 $SCRIPT_DIR/fs.sh $TYPE %1

message "Start merge p80 feature"
python $SCRIPT_DIR/merge_p80.py $TYPE

message "Start select top 1000 feature"
$SCRIPT_DIR/bed_select all.$TYPE.bed all.$TYPE.bed.out 1000

cat all.$TYPE.bed.out|cut -f1-3 > $FEATURE_SELECTION_OUTPUT_DIR/all.$TYPE.bed.out

mv all.$TYPE.bed $FEATURE_SELECTION_OUTPUT_DIR
# clean up workspace and backup all random ids
cp "all.${TYPE}.sample.info".* $FEATURE_SELECTION_OUTPUT_DIR
#rm train.tab.*

# build selected feature bed for feature selection

#$SCRIPT_DIR/new_mode.sh "manu_{$TYPE}_2025" $FEATURE_SELECTION_OUTPUT_DIR/all.${TYPE}.bed.out
#cd $FILE_LOCATION_BAMS
#ls *.nodup.q30.bam 2>/dev/null | cut -f 1 -d . | xargs -n 1 -P 8 -I %1 $SCRIPT_DIR/tab.cpm.sh "manu_${TYPE}_2025" %1

$SCRIPT_DIR/make_all_tab.sh manu_${TYPE}_2025 all.${TYPE}.tab
cp all.${TYPE}.tab $FEATURE_SELECTION_OUTPUT_DIR

cd $FEATURE_SELECTION_OUTPUT_DIR
tar zcf $OUTPUT_PREFIX.sample.info.tar.gz all.$TYPE.sample.info.*
rm $FEATURE_SELECTION_OUTPUT_DIR/all.$TYPE.sample.info.*
message "All $OUTPUT_PREFIX finished"
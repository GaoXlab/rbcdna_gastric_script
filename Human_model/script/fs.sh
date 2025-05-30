#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
TYPE=$1
ID=$2
DIR=`pwd`

export LC_ALL=C

TMP_DIR="$DIR/tmp/$$"
echo "Using $TMP_DIR for $ID"
mkdir $TMP_DIR -p

cd $TMP_DIR

cp -sn $DIR/train.tab.*[0-9] .
cp $DIR/all.$TYPE.sample.info.$ID ./all.sample.info

ls *.tab.* | xargs -n 1 -P 32 $SCRIPT_DIR/dim_reduction_single_step > /dev/null
cat *.bed > $DIR/all.$TYPE.tab.$ID

cd $DIR
$SCRIPT_DIR/bed_select all.$TYPE.tab.$ID all.$TYPE.tab.$ID.out 1000

rm -rf $TMP_DIR
echo "$ID finished"
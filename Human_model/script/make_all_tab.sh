NAME=$1
OUTPUT=$2

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
BASE_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)

if [ ! -n "$1" ] && [ ! -n "$2" ]; then
  echo "Parameter mode_name and output filename is required"
fi
DIR=`pwd`
OUTPUT_NAME=`realpath $OUTPUT`
cd $BASE_DIR/$NAME/cleaned || echo "cd $BASE_DIR/$NAME/cleaned failed"
paste ../sorted.tab.index  *.raw > $OUTPUT_NAME
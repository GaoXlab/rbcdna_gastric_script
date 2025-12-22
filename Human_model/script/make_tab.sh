INPUT=$1
NAME=$2
OUTPUT=$3

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
BASE_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)

INPUT_LOCATION=`realpath $INPUT`
OUTPUT_LOCATION=`realpath $OUTPUT`
if [ ! -n "$1" ] && [ ! -n "$2" ] && [ ! -n "$3" ]; then
  echo "Parameter input_file mode_name and output filename is required"
fi
DIR=`pwd`
cd $BASE_DIR/$NAME/cleaned;
sed 's/$/.raw/' $INPUT_LOCATION | xargs paste ../sorted.tab.index  > $OUTPUT_LOCATION
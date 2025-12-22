#!/bin/bash
# Usage: ./run_gcc_correction_merged.sh SOURCE_DIR TARGET_DIR [THREADS]

SOURCE_DIR=$1
TARGET_DIR=$2
THREADS=${3:-4}  # Default: 4 threads

# Check parameters
if [ -z "$SOURCE_DIR" ] || [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 SOURCE_DIR TARGET_DIR [THREADS]"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Main processing function
process_bam() {
    local filePath="$1"
    local core="$2"
    local output="$3"

    # Skip if output exists
    if [ -f "$output.bai" ]; then
        echo "$output exists!"
        return 0
    fi
    DATA_PATH=`pwd`
    echo "Using DATA_PATH $DATA_PATH"

    # Create temp directory
    tmp_dir="${DATA_PATH}/tmp_$$_$RANDOM"
    mkdir -p "$tmp_dir"
    # Copy reference genome
    cp ./GRCh38.2bit "$tmp_dir"
    cd "$tmp_dir" || exit 1
    export TMP_DIR="$tmp_dir"
    export TMPDIR="$tmp_dir"

    # cp $DATA_PATH/correctGCBias .
    # Index BAM if needed
    [ ! -e "$filePath.bai" ] && samtools index "$filePath" -@ "$core"

    # GC correction steps
    echo "Calculating gc.output for $(basename "$filePath")"
    computeGCBias -b "$filePath" --effectiveGenomeSize 2862010428 \
        --genome "$tmp_dir/GRCh38.2bit" -o "$tmp_dir/gc.output" \
        --numberOfProcessors "$core" 2>&1
    # the numberOfProcessors is set to 1 for reproducing the same result as the original script
    echo "Correcting GC bias for $(basename "$filePath")"
    correctGCBias -b "$filePath" --effectiveGenomeSize 2862010428 \
        --genome "$tmp_dir/GRCh38.2bit" --GCbiasFrequenciesFile "$tmp_dir/gc.output" \
        --numberOfProcessors 1 -o "$tmp_dir/gcc.bam" 2>&1

    # Move final files
    mv "$tmp_dir/gcc.bam" "$output"
    samtools index "$output" -@ "$core"

    # Cleanup
    rm -rf "$tmp_dir"
    echo "Finished processing $(basename "$filePath")"
}

# Export function for xargs
export -f process_bam

# Run parallel processing
find "$SOURCE_DIR" -maxdepth 1 -mindepth 1 -name "*.nodup.q30.bam" | \
    xargs -n 1 -P "$THREADS" -I {} bash -c \
    'process_bam "{}" '"$THREADS"' "'"$TARGET_DIR"'/$(basename "{}")"'

echo "All GC corrections completed!"

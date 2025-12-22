#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

NODE_CPU=${SLURM_CPUS_ON_NODE:-48}

xargs -n 1 -a "$1" -P "$NODE_CPU" -I %1 "$SCRIPT_DIR"/tab.cpm.sh "$2" %1
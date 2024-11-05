#!/bin/bash

cd /srv/essa-lab/flash3/jballoch6/code/sheeprl

source ~/.bashrc


OUTPUT_DIR="$1"
CONFIG_FILE="$2"  ## expected naming convention is task_noveltyarg_method, like walker_175_curious-replay
SEED=$3
EXTRA_ARGS=("${@:4}")

echo "OUTPUT DIR: $OUTPUT_DIR"
echo "CONFIG FILE: $CONFIG_FILE"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# conf_basename=$(basename "$CONFIG_FILE")

# # Check if the basename contains the word "slurm"
# if [[ "$basename" != *"slurm"* ]]; then
#     echo "\n WARNING: The config file does not contain 'slurm'. \n"
# fi

# Extract the file extension
file_extension="${CONFIG_FILE##*.}"

# Set TRUE for debugging
export PYTHONUNBUFFERED=FALSE

conda activate cbwmclone

# if [ -n "$4" ]; then
#     PRETRAINED_MODEL="$4"
#     cp -R "${PRETRAINED_MODEL}/run${SEED}" "${OUTPUT_DIR}/run$SEED"
# fi

# File of command line args
if [ "$file_extension" = "txt" ]; then
    # Read the string from the file
    my_string=$(<"$CONFIG_FILE")

    # Split the string into an array
    read -r -a CONFIG_LIST <<< "$my_string"

    #CONFIG_LIST=$(cat "$CONFIG_FILE")
    python sheeprl.py "seed=${SEED}" "root_dir=${OUTPUT_DIR}" "env=robosuite" "${CONFIG_LIST[@]}" "${EXTRA_ARGS[@]}"
else
    python sheeprl.py "seed=${SEED}" "root_dir=${OUTPUT_DIR}" "env=robosuite" "exp=${CONFIG_FILE}"
fi

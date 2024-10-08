#!/bin/bash


# -- helper
echo -e "## Running slurm experiments \n"

# Function to display the command help
show_help() {
    echo    "## Usage: $0 [options]"
    echo -e "          $0 <exp_output_dir> <parent_configs> \n"
    echo
    echo "Options:"
    echo "  --help          Show this help message and exit"
    echo "  --hard_configs  Use the configs hardcoded in this file"
    echo "  --configs       Use a list of configs after this flag"
    echo "  --config_dir    Use all configs in the specified directory"
}

# Check for the --help flag
#if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "--helper" ]]; then
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "--helper" ]]; then
    show_help
    exit 0
fi

# Define a date and time format
date_format="%Y-%m-%d_%H-%M-%S"

# Get the current date and time in the specified format
timestamp=$(date +"$date_format")

# -- set variables
ROOT="/srv/essa-lab/flash3/jballoch6"

PROJECT="sheeprl"

ENV="robosuite"

SCRIPT_DIR="$ROOT/code/${PROJECT}/slurm"

CONFIG_DIR="$ROOT/code/${PROJECT}/slurm/slurm_configs/${ENV}"

RUNNER_WRAPPER_SCRIPT="${SCRIPT_DIR}/template_wrapper_${PROJECT}_${ENV}.sh"

if [ $# -eq 2 ] && [ "$1" = "--hard_configs" ]; then
    # Use configs hardcoded in this file
    echo "Using hardcoded configs"
    CONFIGS=("train_bwm.txt" "train_cbwm.txt")
    #CONFIGS=("walker_175_dreamer.yml" "walker_175_cr.yml" "walker_175_does.yml" "walker_300_dreamer.yml" "walker_300_cr.yml" "walker_300_does.yml")
    #CONFIGS=("${hardcoded_files[@]/#/$directory/}")
# Check if specific files are provided as arguments
elif [ $# -gt 2 ] && [ "$1" = "--configs" ]; then
    # Shift to remove the directory argument
    echo "specified configs:"
    shift
    shift
    CONFIGS=("$@")
else
    # If no specific files are provided, use all of the non-".args" files in the directory
    echo "All configs in directory"
    # Loop through each file to filter out
    CONFIGS=()
    for file in "${CONFIG_DIR}"/*; do
        # Check if it is a file and does not end in ".args"
        echo "checking file ${file}?"
        if [ -f "$file" ] && [[ ! "$file" =~ \.args$ ]]; then
            bn=$(basename "$file")
            echo "basename: $bn"
            CONFIGS+=($(basename "$file"))
        fi
    done
fi

echo "Configs: ${CONFIGS[@]}"

#novelty = walker175
#method = curious-replay

EXPERIMENT_OUTPUT_DIR="${ROOT}/logs/${PROJECT}/${ENV}"
#PARENT_CONFIG="$2"

#UNIQUE_ID=$((1 + RANDOM % 100000))
UNIQUE_ID=$timestamp
CPT=0



# -- grab un-usable nodes
mapfile -t all_nodes < <( squeue -u abeedu3 -o "%N" )

all_nodes=("${all_nodes[@]:1}")

unique_nodes=($(printf '%s\n' "${all_nodes[@]}" | sort -u))

exclude_nodes=""
for x in "${unique_nodes[@]}"; do
    exclude_nodes+="$x,"
done

exclude_nodes="${exclude_nodes:0:-1}"
echo -e "## excluding these nodes: $exclude_nodes \n"




# -- launch SLURM scripts
for cfg in "${CONFIGS[@]}"; do

    config_file="$CONFIG_DIR/$cfg"
    basename="${cfg%.*}"
    echo "basename: ${basename}"
    echo "config path: $config_file"
    echo "output path: ${EXPERIMENT_OUTPUT_DIR}/${cfg%.*}"

    # Check for extra args in a ".args" file with the same name as the config
    args_file="$CONFIG_DIR/${basename}.args"
    EXTRA_ARGS=()
    if [ -f "$args_file" ]; then
        echo "extra slurm run args file present: $args_file"
        # Read the file line by line
        while IFS= read -r line; do
            # Add each line to the arguments array if not empty
            if [ -n "$line" ]; then
                # Trim whitespace
                trimmed_line=$(echo "$line" | awk '{$1=$1};1')
                if [ -n "$trimmed_line" ]; then
                    EXTRA_ARGS+=("$trimmed_line")
                fi
            fi
        done < "$args_file"
    fi

    # create copy of runner_wrapper.sh template
    tmp_wrapper_name="wrapper_${PROJECT}_${basename}_${UNIQUE_ID}_${CPT}.sh"
    dst_wrapper="$SCRIPT_DIR/tmp/$tmp_wrapper_name"

    cp "$RUNNER_WRAPPER_SCRIPT" "$dst_wrapper"

    chmod +x "$dst_wrapper"

    #sbatch --exclude="$exclude_nodes" autoslurm_script.sh "$dst_wrapper"     "$EXPERIMENT_OUTPUT_DIR" "$config_file" "$PARENT_CONFIG"
    #if [ -n "$PRETRAINED_MODEL" ]; then
    #    sbatch autoslurm_script.sh "$dst_wrapper"     "${EXPERIMENT_OUTPUT_DIR}/${cfg%.*}" "$config_file" "$PRETRAINED_MODEL"
    #else
    sbatch autoslurm_script.sh --job-name="${PROJECT}_${basename}_${UNIQUE_ID}_${CPT}" "JOB" "$dst_wrapper"  "${EXPERIMENT_OUTPUT_DIR}/${basename}_${UNIQUE_ID}_${CPT}" "${config_file}" "${EXTRA_ARGS[@]}"
    #fi
    ((CPT++))

done

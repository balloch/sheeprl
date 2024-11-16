#!/bin/bash

generate_arg_combinations() {
    local yaml_file="$1"
    declare -A args_dict
    local combinations=()
    local alone_keys=()

    # Read the combos file
    while IFS= read -r line; do
        # Parse the YAML file
        if [[ "$line" =~ ^([a-zA-Z0-9_.]+): ]]; then
            key="${BASH_REMATCH[1]}"
            read -r line
            if [[ "$line" =~ options:\ (.*) ]]; then
                options=(${BASH_REMATCH[1]//, / })
            fi
            read -r line
            if [[ "$line" =~ combo:\ (.*) ]]; then
                combo="${BASH_REMATCH[1]}"
                # Store the key if combo is "alone"
                if [[ "$combo" == "alone" ]]; then
                    alone_keys+=("$key")
                fi
            fi
            args_dict["$key"]="${options[@]}:$combo"
        fi
    done < "$yaml_file"

    # Generate combinations based on the parsed dictionary
    local group_combinations=()
    local alone_combinations=()

    for key in "${!args_dict[@]}"; do
        IFS=':' read -r options combo <<< "${args_dict[$key]}"
        IFS=' ' read -r -a options_array <<< "$options"

        if [[ "$combo" == "all" ]]; then
            # Generate all combinations for "all"
            local all_combinations=()
            local num_options=${#options_array[@]}
            for ((i=1; i < (1 << num_options); i++)); do
                local combination=""
                for ((j=0; j < num_options; j++)); do
                    if (( (i & (1 << j)) != 0 )); then
                        combination+="${options_array[j]},"
                    fi
                done
                combination="[${combination%,}]" # Remove trailing comma and add square brackets
                all_combinations+=("$key=$combination")
            done
            group_combinations+=("${all_combinations[@]}")
        elif [[ "$combo" == "alone" ]]; then
            # Generate alone combinations
            for option in "${options_array[@]}"; do
                alone_combinations+=("$key=$option")
            done
        fi
    done

    # Combine results
    # echo "group_combinations: ${group_combinations[@]}"
    # echo "Alone combinations: ${alone_combinations[@]}"
    combinations+=("${group_combinations[@]}")
    for alone in "${alone_combinations[@]}"; do
        combinations+=("$alone")
    done

    # Filter combinations to ensure no multiple arguments with alone options
    local final_combinations=()
    for combo in "${combinations[@]}"; do
        # Check if the combo contains any of the alone keys
        local keep_combo=true
        for alone_key in "${alone_keys[@]}"; do
            if [[ "$combo" == *"$alone_key="* ]]; then
                # If it contains an alone key, we need to check if it has other options
                if [[ $(echo "$combo" | tr -cd ' ' | wc -c) -ne 0 ]]; then
                    keep_combo=false
                    break
                fi
            fi
        done

        if $keep_combo; then
            final_combinations+=("$combo")
        fi
    done

    echo "${final_combinations[@]}"
}

###
### For testing
###

# ROOT="/srv/essa-lab/flash3/jballoch6"

# PROJECT="sheeprl"

# ENV="robosuite"

# SCRIPT_DIR="$ROOT/code/${PROJECT}/slurm"

# CONFIG_DIR="$ROOT/code/${PROJECT}/slurm/slurm_configs/${ENV}_pretrain"


# # Read permutations.yaml and generate argument combinations
# PERMUTATIONS_FILE="$CONFIG_DIR/permutations.combos"
# EXTRA_COMBINATIONS=()
# if [ -f "$PERMUTATIONS_FILE" ]; then
#     EXTRA_COMBINATIONS=($(generate_arg_combinations "$PERMUTATIONS_FILE"))
# fi

# echo "${EXTRA_COMBINATIONS[@]}"

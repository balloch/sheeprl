#!/bin/bash
#SBATCH --error=/srv/essa-lab/flash3/jballoch6/logs/slurm_logs/%x/sample-%j.err
#SBATCH --output=/srv/essa-lab/flash3/jballoch6/logs/slurm_logs/%x/sample-%j.out
#SBATCH --partition="ei-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="long"


# Initialize an array to store the arguments
ARGS_TO_PASS=()

# Flag to indicate whether we have encountered "JOB"
start_collecting=false

# Loop through all the arguments starting at $1
for arg in "$@"; do
    echo "arg: $arg"
    # Check if arguments are still SBATCH args by looking for the keyword "JOB"
    # Everything between "JOB" and "STOP" should be passed to srun
    if [ "$arg" = "JOB" ]; then
        echo "JOB found"
        start_collecting=true
        continue  # Skip adding "JOB" itself to the list
    fi

    # Check if the current argument is the stopping value
    if [ "$arg" = "STOP" ]; then
        break
    fi

    # Add the argument to the array
    if [ "$start_collecting" = true ]; then
        ARGS_TO_PASS+=("$arg")
    fi
done

echo "ARGS_TO_PASS: $ARGS_TO_PASS"

#srun $1 $2 $3 $4
srun ${ARGS_TO_PASS[@]}

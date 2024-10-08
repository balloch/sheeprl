#!/bin/bash
#SBATCH --job-name=cem_dreamer_scratch
#SBATCH --output=cem_dreamer_scratch.out
#SBATCH --error=cem_dreamer_scratch.err
#SBATCH --partition="ei-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="long"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate cbwmclone
cd code/sheeprl
srun -u python sheeprl.py exp=offline_dreamer_robosuite env.wrapper.bddl_file='scenes/LIBERO_OBJECT_SCENE_pick_up_the_butter_and_place_it_in_the_basket.bddl' algo.learning_starts=1000 logger@metric.logger=wandb

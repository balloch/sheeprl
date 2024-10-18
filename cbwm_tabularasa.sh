#!/bin/bash
#SBATCH --job-name=nocem_cbwm_tabularasa
#SBATCH --output=nocem_cbwm_tabularasa.out
#SBATCH --error=nocem_cbwm_tabularasa.err
#SBATCH --partition="ei-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="long"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate cbwmclone
cd ~/code/sheeprl
srun -u python sheeprl.py \
exp=offline_dreamer_robosuite \
env.wrapper.bddl_file='scenes/LIBERO_OBJECT_SCENE_pick_up_the_butter_and_place_it_in_the_basket.bddl' \
algo.learning_starts=10000 \
algo.world_model.cbm_model.use_cbm=False \
logger@metric.logger=wandb

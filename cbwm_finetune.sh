#!/bin/bash
#SBATCH --job-name=cbwm_finetune
#SBATCH --output=cbwm_finetune.out
#SBATCH --error=cbwm_finetune.err
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
exp=offline_dreamer_robosuite_finetune \
env.wrapper.bddl_file='scenes/LIBERO_OBJECT_SCENE_pick_up_the_butter_and_place_it_in_the_basket.bddl' \
algo.learning_starts=10000 \
algo.world_model.cbm_model.use_cbm=true \
checkpoint.pretrain_chkpt_path='/nethome/jballoch6/code/sheeprl/logs/runs/offline_dreamer/Panda_PickPlace/2024-09-30_00-05-38_offline_dreamer_Panda_PickPlace_5/checkpoint/ckpt_24500_0.ckpt'
logger@metric.logger=wandb

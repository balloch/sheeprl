#!/bin/bash
#SBATCH --job-name=nocem_cbwm_pretrain2
#SBATCH --output=nocem_cbwm_pretrain2.out
#SBATCH --error=nocem_cbwm_pretrain2.err
#SBATCH --partition="ei-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate cbwmclone
cd code/sheeprl

srun -u python sheeprl.py exp=offline_dreamer_robosuite_pretrain algo.offline=True checkpoint.every=500 env.num_envs=1 algo.num_epochs=25 algo.per_rank_batch_size=256 algo.offline_train_split=0.95 metric.log_every=10 algo.world_model.optimizer.lr=1e-2 algo.world_model.cbm_model.use_cbm=False logger@metric.logger=wandb

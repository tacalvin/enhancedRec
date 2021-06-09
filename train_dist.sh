#!/bin/bash -l
 
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 # Each task needs only one CPU
#SBATCH --time=2-00:01:00  # 1 day and 1 minute 
#SBATCH --mail-user=cta003@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="train_enhance dist"
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=./slurm_logs/output_%j-

source activate enhanceRec
hostname
date
w
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
srun python train_dist.py --model_config ./configs/model.yaml --experiment_config ./configs/experiment.yaml
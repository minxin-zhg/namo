#!/bin/bash
#SBATCH -A mth260006p
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
#SBATCH -t 24:00:00
#SBATCH -J wandb-h100
#SBATCH -o /ocean/projects/mth260006p/mzhang37/logs/wandb-h100-%j.out

set -eo pipefail

cd /ocean/projects/mth260006p/mzhang37/code/namo_private

source ~/.bashrc
conda activate namo

export OMP_NUM_THREADS=13
export WANDB_DIR=/ocean/projects/mth260006p/mzhang37/code/namo_private/wandb

wandb agent "$1"

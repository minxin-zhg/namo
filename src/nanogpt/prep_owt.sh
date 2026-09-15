#!/bin/bash
#SBATCH -J prep-owt
#SBATCH -A mth260006p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -n 5
#SBATCH -t 04:00:00
#SBATCH -o /ocean/projects/mth260006p/mzhang37/logs/prep-owt-%j.out

module load anaconda3
source /ocean/projects/mth260006p/mzhang37/envs/llm/bin/activate
export HF_HOME=/ocean/projects/mth260006p/mzhang37/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd /ocean/projects/mth260006p/mzhang37/code/namo_private/src/nanogpt
python data/openwebtext/prepare.py

#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-80:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=252G
#SBATCH --time=08:00:00
#SBATCH --job-name=wandb-sweep
#SBATCH --output=logs/%x-%A_%a.out

# Submit from the repository root; create logs/ before calling sbatch.
# Each array task runs one W&B trial. Use --array=1-N%1 for serial trials.
set -eo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[^/]+/[^/]+/[^/]+$ ]]; then
    echo "Usage: sbatch -A ACCOUNT [--array=1-N%1] scripts/run_wandb_bridges2.sbatch ENTITY/PROJECT/SWEEP_ID" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:?Submit this job from the namo repository root}"
source "$HOME/.bashrc"
conda activate "${NAMO_CONDA_ENV:-namo}"
set -u

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
export WANDB_DIR="$PWD/wandb"
mkdir -p "$WANDB_DIR"

# Fail before claiming a sweep trial if the local dataset is unavailable.
for split in train val; do
    if [[ ! -s "src/nanogpt/data/openwebtext/$split.bin" ]]; then
        echo "Missing dataset: src/nanogpt/data/openwebtext/$split.bin" >&2
        exit 1
    fi
done

python - <<'CHECK_GPUS'
import torch
count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"Expected four allocated GPUs, found {count}")
print(f"Using {count} GPUs: {torch.cuda.get_device_name(0)}", flush=True)
CHECK_GPUS

# Sweep names can overlap; keep checkpoints separate without changing W&B configs.
sweep_work_dir="$PWD/outputs/sweeps/${1//\//_}"
mkdir -p "$sweep_work_dir"
if [[ ! -e "$sweep_work_dir/src" ]]; then
    ln -s "$PWD/src" "$sweep_work_dir/src"
fi
cd "$sweep_work_dir"

# Bound each allocation to one ~7-hour trial, leaving an hour of margin.
exec wandb agent --count 1 "$1"

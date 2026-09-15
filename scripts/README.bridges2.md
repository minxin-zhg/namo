# W&B sweeps on Bridges-2

Submit from the repository root with the `namo` Conda environment installed
and OpenWebText prepared under `src/nanogpt/data/openwebtext/`.
Authenticate with W&B before submitting jobs. Set `NAMO_CONDA_ENV` to use
a different Conda environment.

```bash
mkdir -p logs
sbatch -A YOUR_ALLOCATION --job-name=wandb-SWEEP_ID \
  --array=1-20%1 scripts/run_wandb_bridges2.sbatch ENTITY/PROJECT/SWEEP_ID
```

Each array task requests four H100 80 GB GPUs in `GPU-shared`, 48 CPU cores,
252 GiB of host memory, and eight hours. It runs `wandb agent --count 1`,
which allows about an hour of margin for a seven-hour training run and
prevents the agent from starting another trial near the time limit.
The sweep command must launch four training processes.

Set the array size to the number of remaining trials. `%1` allows one trial
at a time in each sweep; three such arrays can use at most twelve GPUs in
total. W&B assigns the parameter combinations. Already claimed or finished
trials are not deliberately restarted, and agents exit when no trials remain.
Omit `--array` to submit only one trial.

Each sweep runs under `outputs/sweeps/ENTITY_PROJECT_SWEEP_ID/`, with a
symlink to the repository's `src/` directory. This keeps checkpoints separate
even when two sweeps use identical experiment names. Checkpoints are under
that directory's `outputs/EXPERIMENT/EXPERIMENT_NAME/` subdirectory.

Logs are written to `logs/JOB_NAME-ARRAY_JOB_ID_TASK_ID.out`.
Use `squeue -u "$USER"` to inspect the queue. Failed or timed-out trials need
to be reviewed before retrying; this launcher does not resume checkpoints.

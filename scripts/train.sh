# SWEEP_ID=$(wandb sweep -p muon -e schaefferlab1 scripts/sweeps/adago2.yaml 2>&1 | tee /dev/tty \
#            | grep -oP '(?<=wandb agent ).+')
# wandb agent $SWEEP_ID


# SWEEP_ID=$(wandb sweep -p muon -e schaefferlab1 scripts/sweeps/adamnormmuon.yaml 2>&1 | tee /dev/tty \
#            | grep -oP '(?<=wandb agent ).+')
# wandb agent $SWEEP_ID


## gpt2-small training
torchrun --standalone --nproc_per_node=4 src/train_gpt2.py optim=adamw wandb_log=0 exp_name=adam

torchrun --standalone --nproc_per_node=4 src/train_gpt2.py optim=muon wandb_log=0 exp_name=muon

torchrun --standalone --nproc_per_node=4 src/train_gpt2.py optim=namo wandb_log=0 exp_name=namo

torchrun --standalone --nproc_per_node=4 src/train_gpt2.py optim=namo-d wandb_log=0 exp_name=namo_d
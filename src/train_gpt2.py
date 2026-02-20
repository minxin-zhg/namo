"""
Modified based on nanogpt/train.py to support wandb/hydra/different optimizers.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanogpt.model import GPTConfig, GPT

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from hydra.utils import get_original_cwd
from muon import Muon
from namo import NAMO, NAMO_D

# system configs
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


def configure_optimizers(model, cfg, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # apply weight decay only to hidden matrix parameters
    decay_params, nodecay_params = [], []
    nodecay_keys = ["lm_head", "wte", "wpe"]

    for n, p in param_dict.items():
        if p.dim() < 2 or any([s in n for s in nodecay_keys]):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    if cfg.type == "adamw":
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), **extra_args)
        print(f"using fused AdamW: {use_fused}")

    else:
        # muon and variants
        adamw_params = nodecay_params
        muon_params = decay_params

        num_adamw_params = sum(p.numel() for p in adamw_params)
        num_muon_params = sum(p.numel() for p in muon_params)
        print(f"num muon parameter tensors: {len(muon_params)}, with {num_muon_params:,} parameters")
        print(f"num adamw parameter tensors: {len(adamw_params)}, with {num_adamw_params:,} parameters")

        # shared configs
        kwargs = dict(
            lr=cfg.learning_rate,
            wd=cfg.weight_decay,
            momentum=cfg.beta,
            adamw_betas=(cfg.beta1, cfg.beta2),
            adamw_wd=cfg.adamw_weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

        match cfg.type:
            case "muon":
                optimizer = Muon(**kwargs)
                print("using Muon optimizer")

            case "namo":
                kwargs["mu2"] = cfg.mu2
                optimizer = NAMO(**kwargs)
                print("using NAMO optimizer")

            case "namo_d":
                kwargs["mu2"] = cfg.mu2
                if "adamnorm_eps" in cfg:
                    kwargs["adamnorm_eps"] = cfg.adamnorm_eps
                if "scale_coeff" in cfg:
                    kwargs["scale_coeff"] = cfg.scale_coeff
                if "col_state_clamp_c" in cfg:
                    kwargs["col_state_clamp_c"] = cfg.col_state_clamp_c
                optimizer = NAMO_D(**kwargs)
                print("using NAMO-D optimizer")

            case _:
                raise ValueError(f"Unknown muon optimizer type: {cfg.type}")

    return optimizer


@hydra.main(version_base=None, config_path="hydra_configs", config_name="main")
def main(cfg: DictConfig):
    # various inits, derived attributes, I/O setup

    # Hydra may chdir into a run dir; keep paths rooted at the original cwd.
    orig_cwd = get_original_cwd()

    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(orig_cwd, p)

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    out_dir = _abs(os.path.join("outputs", cfg.exp, cfg.exp_name))
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(out_dir, "configs.yaml"))

    # logging
    if cfg.wandb_log and master_process:
        if not cfg.wandb.id:
            cfg.wandb.id = wandb.util.generate_id()
        wandb.init(
            **cfg.wandb,
            resume="allow",
            dir=out_dir,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
        wandb.define_metric("iter")
        wandb.define_metric("*", step_metric="iter")

    tokens_per_iter = cfg.gradient_accumulation_steps * ddp_world_size * cfg.batch_size * cfg.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    data_dir = os.path.join(_abs(cfg.data_dir), cfg.dataset)

    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + cfg.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + cfg.block_size]).astype(np.int64)) for i in ix])

        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    gd_norm = None

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Safety: GPT-2 pretrained checkpoints require GPT-2-tokenized datasets (~50k vocab).
    # Catch accidental char-level datasets early (e.g. shakespeare_char).
    if isinstance(cfg.init_from, str) and cfg.init_from.startswith("gpt2"):
        if meta_vocab_size is not None and meta_vocab_size < 1000:
            raise ValueError(
                f"init_from={cfg.init_from} expects GPT-2-tokenized data (vocab ~50k), "
                f"but meta.pkl reports vocab_size={meta_vocab_size}. "
                "Use dataset=shakespeare/openwebtext (BPE), not shakespeare_char."
            )
    # model init
    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        block_size=cfg.block_size,
        vocab_size=None,
    )  # start with model_args from command line

    match cfg.init_from:
        case "scratch":
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304

            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

        case "resume":
            print(f"Resuming training from {out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint["model_args"]

            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                model_args[k] = checkpoint_model_args[k]

            # create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint["model"]

            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]

        case _:
            if cfg.init_from.startswith("gpt2"):
                print(f"Initializing from OpenAI GPT-2 weights: {cfg.init_from}")
                # initialize from OpenAI GPT-2 weights
                override_args = dict(dropout=cfg.model.dropout)
                model = GPT.from_pretrained(cfg.init_from, override_args)

                # read off the created config params, so we can store them into checkpoint correctly
                for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
                    model_args[k] = getattr(model.config, k)

    # crop down the model block size if desired, using model surgery
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args["block_size"] = cfg.block_size  # so that the checkpoint will have the right value
    model.to(device)

    if master_process:
        print(model)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

    # optimizer
    # optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    optimizer = configure_optimizers(model, cfg.optim, device_type)
    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate schedule: warmup stable decay
    def get_lr_ratio(step: int):
        if step < cfg.optim.warmup:
            # linear warmup phase
            return 1.0 * step / cfg.optim.warmup
        else:
            x = step / cfg.max_iters  # progress in training
            assert 0 <= x <= 1
            if x <= 1 - cfg.optim.decay:
                # stable phase
                return 1.0
            else:
                match cfg.optim.get("decay_type", "linear"):
                    case "linear":
                        w = (1 - x) / cfg.optim.decay
                    case "cosine":
                        decay_ratio = (step - cfg.optim.warmup) / (cfg.max_iters - cfg.optim.warmup)
                        w = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                    case _:
                        assert False, "unknown lr decay type"

                return w * 1.0 + (1 - w) * cfg.optim.min_lr_ratio

    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr_ratio = get_lr_ratio(iter_num)
        lr = lr_ratio * cfg.optim.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if cfg.wandb_log:
                log = {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr/lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
                if gd_norm is not None:
                    log["train/grad_norm"] = gd_norm
                wandb.log(log, step=iter_num)

            if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        # "config": cfg,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

        if iter_num == 0 and cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(cfg.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == cfg.gradient_accumulation_steps - 1
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps  # scale the loss to account for gradient accumulation

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            gd_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            if cfg.wandb_log:
                log = {
                    "iter": iter_num,
                    "train/loss_iter": lossf,
                    "lr/lr": lr,
                    "mfu": running_mfu * 100,
                }
                wandb.log(log, step=iter_num)

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > cfg.max_iters:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        print(f"Max Memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MB")
        if torch.distributed.is_initialized():
            destroy_process_group()

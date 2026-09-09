"""Training orchestration for all six methods.

Reward (Eq. 7): r = A_after - A_before, the change in validation accuracy from
updating the selected arm(s) for one epoch. Bandit statistics are updated with
this reward (UCB Lines 14-15 / TS posterior update).
"""

import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .bandit import UCB, GaussianThompsonSampling, build_random_arms, union_arms
from .bandit.arms import Arm
from .models import (build_model, wrap_paca, wrap_lora, freeze_backbone,
                     get_head_parameters)
from .sensitivity import compute_sensitivity, build_gradient_chains
from .utils import AverageMeter, accuracy, set_seed, resolve_device, count_trainable


# --------------------------------------------------------------------------- #
# low-level train / eval
# --------------------------------------------------------------------------- #
def _build_optimizer(params, cfg):
    return torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                             weight_decay=cfg.weight_decay)


def train_one_epoch(model, loader, optimizer, device, criterion, scaler=None, amp=False):
    model.train()
    meter = AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        meter.update(accuracy(logits, targets), images.size(0))
    return meter.avg


# --------------------------------------------------------------------------- #
# helpers shared by the PaCA-family
# --------------------------------------------------------------------------- #
def _random_columns(layer_in_features: Dict[str, int], rank: int, rng) -> Arm:
    arm: Arm = {}
    for name, in_f in layer_in_features.items():
        r = min(rank, in_f)
        arm[name] = [int(c) for c in rng.permutation(in_f)[:r]]
    return arm


def _make_optimizer_for(manager, model, cfg):
    params = list(manager.trainable_parameters())
    if cfg.train_head:
        params += get_head_parameters(model)
    return _build_optimizer(params, cfg)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def run(cfg, loaders, num_classes, device, logger=print):
    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    target_subs = [s.strip() for s in cfg.target_modules.split(",") if s.strip()]

    set_seed(cfg.seed)
    model = build_model(cfg.model, num_classes, cfg.image_size).to(device)

    history: List[dict] = []
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_timer_start = time.perf_counter()

    if cfg.method == "lora":
        _dispatch_lora(cfg, model, target_subs, train_loader, val_loader,
                       device, criterion, scaler, history, logger)
        trainable = count_trainable(model)          # LoRA adapters + head (still active)
    else:
        model, manager = wrap_paca(model, target_subs)
        model.to(device)
        freeze_backbone(model, train_head=cfg.train_head)

        if cfg.method == "paca":
            _dispatch_paca_fixed(cfg, model, manager, train_loader, val_loader,
                                 device, criterion, scaler, history, logger)
        elif cfg.method == "r_paca":
            _dispatch_r_paca(cfg, model, manager, train_loader, val_loader,
                             device, criterion, scaler, history, logger)
        elif cfg.method in ("ucb_paca", "ts_paca"):
            _dispatch_random_bandit(cfg, model, manager, train_loader, val_loader,
                                    device, criterion, scaler, history, logger)
        elif cfg.method == "gradient_paca":
            _dispatch_gradient_paca(cfg, model, manager, train_loader, val_loader,
                                    device, criterion, scaler, history, logger)
        else:
            raise ValueError(cfg.method)
        # Count trainable params while the last arm's deltas are still active;
        # merge_all() below sets them to None, which would otherwise hide the
        # adapter budget and report only the head.
        trainable = count_trainable(model)
        manager.merge_all()  # persist learned columns before final test

    total_time = time.perf_counter() - total_timer_start

    total_params = (sum(p.numel() for p in model.parameters())
                    + sum(b.numel() for b in model.buffers()))
    trainable_pct = 100.0 * trainable / max(total_params, 1)

    test_acc = evaluate(model, test_loader, device)
    best_val = max((h["val_after"] for h in history), default=0.0)

    if device.type == "cuda":
        peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
    else:
        peak_mem_mb = None

    result = {
        "method": cfg.method,
        "dataset": cfg.dataset,
        "total_epochs": len(history),
        "total_training_time_sec": round(total_time, 2),
        "best_val_accuracy": round(best_val * 100, 4),
        "test_accuracy": round(test_acc * 100, 4),
        "trainable_params": trainable,
        "total_params": total_params,
        "trainable_params_pct": round(trainable_pct, 4),
        "peak_gpu_mem_mb": peak_mem_mb,
        "history": history,
    }
    logger(f"[params] trainable={trainable:,} "
           f"({trainable_pct:.3f}% of {total_params:,} total)")
    if peak_mem_mb is not None:
        logger(f"[gpu] peak memory allocated = {peak_mem_mb:.1f} MB")
    logger(f"[done] method={cfg.method} dataset={cfg.dataset} "
           f"epochs={len(history)} time={total_time:.1f}s test_acc={test_acc*100:.2f}%")
    return result


# --------------------------------------------------------------------------- #
# early-stopping wrapper
# --------------------------------------------------------------------------- #
class _EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = -1.0
        self.bad = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.patience <= 0:
            return False
        if value > self.best + 1e-6:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience


# --------------------------------------------------------------------------- #
# method dispatchers
# --------------------------------------------------------------------------- #
def _dispatch_lora(cfg, model, target_subs, train_loader, val_loader,
                   device, criterion, scaler, history, logger):
    model = wrap_lora(model, target_subs, cfg.lora_rank, cfg.lora_alpha)
    model.to(device)
    freeze_backbone(model, train_head=cfg.train_head)
    # unfreeze LoRA params
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(params, cfg)
    stopper = _EarlyStopper(cfg.patience)
    prev_val = evaluate(model, val_loader, device)  # baseline once; reuse thereafter
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                               scaler, cfg.amp)
        val_after = evaluate(model, val_loader, device)
        history.append(_record(epoch, None, loss, prev_val, val_after))
        logger(_fmt(epoch, "-", loss, prev_val, val_after))
        prev_val = val_after
        if stopper.step(val_after):
            logger(f"[early-stop] no val improvement for {cfg.patience} epochs.")
            break


def _dispatch_paca_fixed(cfg, model, manager, train_loader, val_loader,
                         device, criterion, scaler, history, logger):
    rng = np.random.default_rng(cfg.seed)
    arm = _random_columns(manager.layer_in_features(), cfg.rank, rng)
    manager.activate(arm)
    optimizer = _make_optimizer_for(manager, model, cfg)  # fixed selection -> one optimizer
    stopper = _EarlyStopper(cfg.patience)
    prev_val = evaluate(model, val_loader, device)  # baseline once; reuse thereafter
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                               scaler, cfg.amp)
        val_after = evaluate(model, val_loader, device)
        history.append(_record(epoch, None, loss, prev_val, val_after))
        logger(_fmt(epoch, "fixed", loss, prev_val, val_after))
        prev_val = val_after
        if stopper.step(val_after):
            logger(f"[early-stop] no val improvement for {cfg.patience} epochs.")
            break


def _dispatch_r_paca(cfg, model, manager, train_loader, val_loader,
                     device, criterion, scaler, history, logger):
    rng = np.random.default_rng(cfg.seed)
    stopper = _EarlyStopper(cfg.patience)
    prev_val = evaluate(model, val_loader, device)  # baseline once; reuse thereafter
    for epoch in range(cfg.epochs):
        arm = _random_columns(manager.layer_in_features(), cfg.rank, rng)
        manager.activate(arm)                      # new random subset each epoch
        optimizer = _make_optimizer_for(manager, model, cfg)
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                               scaler, cfg.amp)
        val_after = evaluate(model, val_loader, device)
        history.append(_record(epoch, None, loss, prev_val, val_after))
        logger(_fmt(epoch, "random", loss, prev_val, val_after))
        prev_val = val_after
        if stopper.step(val_after):
            logger(f"[early-stop] no val improvement for {cfg.patience} epochs.")
            break


def _dispatch_random_bandit(cfg, model, manager, train_loader, val_loader,
                            device, criterion, scaler, history, logger):
    rng = np.random.default_rng(cfg.seed)
    # Paper Sec 4.1: the random arms exhaustively and disjointly partition ALL
    # adapted columns. Each arm holds ~`rank` columns, so N is derived here
    # (ceil(in_features / rank)) rather than taken from --num-arms.
    arms = build_random_arms(manager.layer_in_features(), cfg.rank, rng)
    num_arms = len(arms)
    select_size = max(1, min(cfg.select_size, num_arms))

    if cfg.num_arms != num_arms:
        logger(f"[arms] --num-arms is not used for {cfg.method}: random arms form "
               f"an exhaustive partition, so N is set by --rank.")
    logger(f"[arms] {cfg.method}: N={num_arms} exhaustive disjoint arms "
           f"(~{cfg.rank} cols/arm/layer), K={select_size} selected/epoch; "
           f"union = all adapted columns.")

    if cfg.method == "ucb_paca":
        bandit = UCB(num_arms, alpha=cfg.ucb_alpha)
    else:  # ts_paca
        bandit = GaussianThompsonSampling(
            num_arms, mu0=cfg.ts_mu0, sigma0=cfg.ts_sigma0,
            sigma_obs=cfg.ts_sigma_obs, rng=np.random.default_rng(cfg.seed + 1))

    _bandit_loop(cfg, model, manager, arms, bandit, select_size,
                 train_loader, val_loader, device, criterion, scaler, history, logger)


def _dispatch_gradient_paca(cfg, model, manager, train_loader, val_loader,
                            device, criterion, scaler, history, logger):
    rng = np.random.default_rng(cfg.seed)

    # ---- warm start: train a broad random adapter set for a few epochs ------
    warm_arm = _random_columns(manager.layer_in_features(),
                               cfg.rank * cfg.num_arms, rng)
    manager.activate(warm_arm)
    optimizer = _make_optimizer_for(manager, model, cfg)
    prev_val = evaluate(model, val_loader, device)  # baseline once; reuse thereafter
    for epoch in range(cfg.warmup_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                               scaler, cfg.amp)
        val_after = evaluate(model, val_loader, device)
        history.append(_record(epoch, "warmup", loss, prev_val, val_after))
        logger(_fmt(epoch, "warmup", loss, prev_val, val_after))
        prev_val = val_after
    manager.merge_all()  # persist warm-start updates into the frozen weights

    # ---- sensitivity analysis + chain construction --------------------------
    sens = compute_sensitivity(model, manager, train_loader, device,
                               num_batches=cfg.sens_batches)
    layer_order = manager.layer_names  # depth order preserved during wrapping
    arms = build_gradient_chains(sens, layer_order, cfg.num_arms, cfg.chain_width)
    num_arms = len(arms)
    select_size = max(1, min(cfg.select_size, num_arms))
    logger(f"[gradient] built {num_arms} gradient-aligned chain arms "
           f"(N={num_arms}, K={select_size}).")

    # ---- Thompson Sampling over chain arms (paper reports TS for gradient) ---
    bandit = GaussianThompsonSampling(
        num_arms, mu0=cfg.ts_mu0, sigma0=cfg.ts_sigma0,
        sigma_obs=cfg.ts_sigma_obs, rng=np.random.default_rng(cfg.seed + 2))
    _bandit_loop(cfg, model, manager, arms, bandit, select_size,
                 train_loader, val_loader, device, criterion, scaler, history, logger,
                 start_epoch=cfg.warmup_epochs)


def _bandit_loop(cfg, model, manager, arms, bandit, select_size,
                 train_loader, val_loader, device, criterion, scaler, history, logger,
                 start_epoch=0):
    stopper = _EarlyStopper(cfg.patience)
    remaining = cfg.epochs - start_epoch
    # Re-selecting an arm never changes the model's output (delta is re-init'd so
    # the correction starts at zero), so the previous epoch's post-training
    # accuracy IS this epoch's A_before. Evaluate the baseline once, then reuse
    # each epoch's val_after as the next epoch's "before" -> one eval per epoch.
    prev_val = evaluate(model, val_loader, device)
    for step in range(remaining):
        epoch = start_epoch + step
        selected = bandit.select(select_size)          # arm indices this epoch
        cols = union_arms(arms, selected)              # union of their columns
        manager.activate(cols)
        optimizer = _make_optimizer_for(manager, model, cfg)

        loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                               scaler, cfg.amp)
        val_after = evaluate(model, val_loader, device)

        reward = val_after - prev_val                  # Eq. 7 (A_after - A_before)
        bandit.update(selected, reward)

        history.append(_record(epoch, selected, loss, prev_val, val_after, reward))
        logger(_fmt(epoch, selected, loss, prev_val, val_after, reward))
        prev_val = val_after                           # next epoch's A_before
        if stopper.step(val_after):
            logger(f"[early-stop] no val improvement for {cfg.patience} epochs.")
            break


# --------------------------------------------------------------------------- #
# logging records
# --------------------------------------------------------------------------- #
def _record(epoch, selected, loss, val_before, val_after, reward=None):
    rec = {
        "epoch": epoch,
        "arm": (selected if isinstance(selected, (list, type(None))) else str(selected)),
        "train_loss": round(float(loss), 4),
        "val_before": round(float(val_before), 4),
        "val_after": round(float(val_after), 4),
    }
    if reward is not None:
        rec["reward"] = round(float(reward), 4)
    return rec


def _fmt(epoch, arm, loss, vb, va, reward=None):
    base = (f"[epoch {epoch:>3}] arm={arm} loss={loss:.4f} "
            f"val {vb*100:.2f}->{va*100:.2f}%")
    if reward is not None:
        base += f" reward={reward*100:+.2f}"
    return base


# --------------------------------------------------------------------------- #
# throughput benchmark
# --------------------------------------------------------------------------- #
def benchmark_throughput(cfg, num_classes, device, logger=print):
    """Measure training throughput (images/sec) for several batch sizes."""
    criterion = nn.CrossEntropyLoss()
    target_subs = [s.strip() for s in cfg.target_modules.split(",") if s.strip()]
    batch_sizes = [int(b) for b in cfg.throughput_batch_sizes.split(",") if b.strip()]

    set_seed(cfg.seed)
    model = build_model(cfg.model, num_classes, cfg.image_size).to(device)

    if cfg.method == "lora":
        model = wrap_lora(model, target_subs, cfg.lora_rank, cfg.lora_alpha).to(device)
        freeze_backbone(model, train_head=cfg.train_head)
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        model, manager = wrap_paca(model, target_subs)
        model.to(device)
        freeze_backbone(model, train_head=cfg.train_head)
        rng = np.random.default_rng(cfg.seed)
        arm = _random_columns(manager.layer_in_features(), cfg.rank, rng)
        manager.activate(arm)
        params = list(manager.trainable_parameters())
        if cfg.train_head:
            params += get_head_parameters(model)

    optimizer = _build_optimizer(params, cfg)

    trainable = count_trainable(model)
    total_params = (sum(p.numel() for p in model.parameters())
                    + sum(b.numel() for b in model.buffers()))
    logger(f"[params] trainable={trainable:,} "
           f"({100.0 * trainable / max(total_params, 1):.3f}% of {total_params:,} total)")

    results = {}
    peak_mem = {}
    for bs in batch_sizes:
        images = torch.randn(bs, 3, cfg.image_size, cfg.image_size, device=device)
        targets = torch.randint(0, num_classes, (bs,), device=device)
        # warmup
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(cfg.throughput_iters):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ips = cfg.throughput_iters * bs / dt
        results[bs] = round(ips, 2)
        if device.type == "cuda":
            mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)
            peak_mem[bs] = mb
            logger(f"[throughput] method={cfg.method} bs={bs} -> {ips:.2f} images/sec "
                   f"(peak {mb:.1f} MB)")
        else:
            logger(f"[throughput] method={cfg.method} bs={bs} -> {ips:.2f} images/sec")
    return {
        "method": cfg.method,
        "throughput_images_per_sec": results,
        "peak_gpu_mem_mb": peak_mem or None,
        "trainable_params": trainable,
        "total_params": total_params,
        "trainable_params_pct": round(100.0 * trainable / max(total_params, 1), 4),
    }

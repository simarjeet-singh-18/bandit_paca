#!/usr/bin/env python
"""Entry point for bandit-based PEFT fine-tuning.

Examples
--------
# Gradient-PaCA on CIFAR-100
python main.py --method gradient_paca --dataset cifar100 --num-arms 3 --select-size 1

# TS-PaCA on Flowers-102
python main.py --method ts_paca --dataset flowers102 --num-arms 6 --select-size 2

# UCB-PaCA on Caltech-101
python main.py --method ucb_paca --dataset caltech101 --ucb-alpha 1.0

# LoRA baseline on SVHN
python main.py --method lora --dataset svhn --lora-rank 8 --lora-alpha 16

# Fast pipeline smoke test (no downloads, tiny fake data)
python main.py --method gradient_paca --fake-data --epochs 2 --warmup-epochs 1 \
    --batch-size 8 --num-workers 0

# Throughput benchmark
python main.py --method ts_paca --dataset cifar100 --measure-throughput
"""

import os

from src.config import parse_args
from src.utils import set_seed, resolve_device, save_json
from src.data import build_datasets, build_fake_datasets, make_loaders
from src.trainer import run, benchmark_throughput

NUM_CLASSES = {"cifar100": 100, "flowers102": 102, "caltech101": 101, "svhn": 10}


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[config] method={cfg.method} dataset={cfg.dataset} device={device}")

    # ---- throughput mode (no dataset download needed) ---------------------
    if cfg.measure_throughput:
        num_classes = cfg.fake_classes if cfg.fake_data else NUM_CLASSES[cfg.dataset]
        result = benchmark_throughput(cfg, num_classes, device)
        tag = f"_{cfg.tag}" if cfg.tag else ""
        out = os.path.join(cfg.output_dir,
                           f"throughput_{cfg.method}_{cfg.dataset}{tag}.json")
        save_json(result, out)
        print(f"[saved] {out}")
        return

    # ---- data -------------------------------------------------------------
    if cfg.fake_data:
        num_classes = cfg.fake_classes
        datasets = build_fake_datasets(num_classes, cfg.fake_size,
                                       cfg.image_size, cfg.val_frac)
    else:
        datasets = build_datasets(cfg.dataset, cfg.data_root, cfg.image_size,
                                  cfg.val_frac, cfg.seed)
    train_ds, val_ds, test_ds, num_classes = datasets

    # ---- training ---------------------------------------------------------
    loaders = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size,
                           cfg.eval_batch_size, cfg.num_workers)
    result = run(cfg, loaders, num_classes, device)
    result["config"] = vars(cfg)

    tag = f"_{cfg.tag}" if cfg.tag else ""
    out = os.path.join(cfg.output_dir, f"{cfg.method}_{cfg.dataset}{tag}.json")
    save_json(result, out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

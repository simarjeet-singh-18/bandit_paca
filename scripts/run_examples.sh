#!/usr/bin/env bash
# Example invocations for every method. Adjust flags as needed.
set -e

DATA_ROOT=${DATA_ROOT:-./data}
COMMON="--data-root ${DATA_ROOT} --batch-size 64 --epochs 30 --patience 5"

# ---- baselines ------------------------------------------------------------
python main.py --method lora  --dataset cifar100 --lora-rank 8 --lora-alpha 16 ${COMMON}
python main.py --method paca  --dataset cifar100 --rank 16 ${COMMON}
python main.py --method r_paca --dataset cifar100 --rank 16 ${COMMON}

# ---- bandit methods (random arms) -----------------------------------------
python main.py --method ucb_paca --dataset cifar100 --rank 16 --num-arms 6 --select-size 1 --ucb-alpha 1.0 ${COMMON}
python main.py --method ts_paca  --dataset cifar100 --rank 16 --num-arms 6 --select-size 2 ${COMMON}

# ---- gradient-aligned chains ----------------------------------------------
python main.py --method gradient_paca --dataset cifar100 --num-arms 3 --select-size 1 \
    --warmup-epochs 2 --sens-batches 8 --chain-width 1 ${COMMON}

# ---- throughput benchmark -------------------------------------------------
python main.py --method ts_paca --dataset cifar100 --measure-throughput \
    --throughput-batch-sizes 16,32,64

# ---- across all datasets (Gradient-PaCA) ----------------------------------
for DS in cifar100 flowers102 caltech101 svhn; do
    python main.py --method gradient_paca --dataset ${DS} --num-arms 3 --select-size 1 ${COMMON}
done

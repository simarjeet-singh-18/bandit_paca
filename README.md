# Bandit-based Connection Selection for Efficient Fine-Tuning

Reference implementation of **"Learning What to Tune: Bandit-based Connection
Selection for Efficient Fine-Tuning"** (Tripathi & Diddigi).

The paper reframes parameter-efficient fine-tuning (PEFT) — specifically
Partial Connection Adaptation (PaCA) — as a **multi-armed bandit (MAB)**
problem: each *arm* is a subset of pre-trained weights, and Upper Confidence
Bound (UCB) or Thompson Sampling (TS) decides *which connections to tune* at
each epoch. A gradient-aligned **chain** construction builds task-aware arms
from a sensitivity analysis.

This repository implements every method the paper compares:

| Method            | Arms                              | Selection            |
|-------------------|-----------------------------------|----------------------|
| `lora`            | — (low-rank adapters)             | —                    |
| `paca`            | one fixed random column subset    | fixed                |
| `r_paca`          | new random subset each epoch      | random / epoch       |
| `ucb_paca`        | random, disjoint arms             | UCB                  |
| `ts_paca`         | random, disjoint arms             | Thompson Sampling    |
| `gradient_paca`   | gradient-aligned chains           | Thompson Sampling    |

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch`/`torchvision` should match your CUDA setup — see
<https://pytorch.org/get-started/locally/>. `timm` provides the
ImageNet-pretrained ViT-Base/16 backbone.

## 2. Download the datasets

```bash
python download_data.py --data-root ./data
# or a subset:
python download_data.py --data-root ./data --datasets cifar100 svhn
# harder / less ImageNet-aligned tasks:
python download_data.py --data-root ./data --datasets dtd fgvc_aircraft eurosat
```

Available datasets: the four originals (`cifar100`, `flowers102`, `caltech101`,
`svhn`) plus four harder, less-saturated tasks — `dtd` (textures, 47 classes),
`fgvc_aircraft` (fine-grained, 100 variants), `eurosat` (satellite, 10 classes),
and `sun397` (scenes, 397 classes; **~38 GB download**, a lighter in-genre
alternative to Places365). The first four saturate quickly for an ImageNet-1K
ViT-B; the harder four, especially combined with `--max-train-samples`, leave
room for method differences to show.

Each dataset lands in `./data/<dataset>/`, exactly where training expects it.
Caltech-101 is served from Google Drive and is occasionally rate-limited — just
re-run the script if it fails.

## 3. Run

```bash
# Gradient-PaCA on CIFAR-100 (best config from the ablation: N=3, K=1)
python main.py --method gradient_paca --dataset cifar100 --num-arms 3 --select-size 1

# UCB-PaCA on Caltech-101
python main.py --method ucb_paca --dataset caltech101 --rank 16 --num-arms 6 --ucb-alpha 1.0

# TS-PaCA on Flowers-102 selecting K=2 arms per epoch
python main.py --method ts_paca --dataset flowers102 --num-arms 6 --select-size 2

# LoRA baseline on SVHN
python main.py --method lora --dataset svhn --lora-rank 8 --lora-alpha 16

# Low-data, harder task (de-saturated): TS-PaCA on FGVC-Aircraft, 1000 train imgs
python main.py --method ts_paca --dataset fgvc_aircraft --max-train-samples 1000 \
    --num-arms 6 --select-size 1
```

Results (total epochs, total training time, best val accuracy, test accuracy,
trainable-param count + percentage, peak GPU memory, and a per-epoch history
with arm selections + rewards)
are written as JSON to `./runs/`.

`scripts/run_examples.sh` runs a representative sweep.

### Smoke test (no downloads, no GPU required)

```bash
python main.py --method gradient_paca --fake-data --epochs 2 --warmup-epochs 1 \
    --batch-size 8 --eval-batch-size 8 --num-workers 0
```

`--fake-data` swaps in random tensors so the full pipeline (wrapping,
sensitivity, chain construction, bandit loop, rewards) can be exercised in
seconds.

### Throughput benchmark

```bash
python main.py --method ts_paca --dataset cifar100 --measure-throughput \
    --throughput-batch-sizes 16,32,64
```

## 4. Key CLI flags

| Flag | Meaning |
|------|---------|
| `--method` | one of `lora, paca, r_paca, ucb_paca, ts_paca, gradient_paca` |
| `--dataset` | `cifar100, flowers102, caltech101, svhn, dtd, fgvc_aircraft, eurosat, sun397` |
| `--max-train-samples` | cap training set via class-stratified subsample (0 = all); e.g. `1000` for a VTAB-1k-style low-data regime. Val/test are untouched |
| `--target-modules` | substrings of Linear layers to adapt (default `attn.qkv,attn.proj`) |
| `--rank` | trainable columns per adapted layer, per arm (PaCA budget `r`) |
| `--num-arms` | number of arms `N` |
| `--select-size` | arms selected per epoch `K` (UCB defaults to 1) |
| `--ucb-alpha` | UCB exploration coefficient `α` |
| `--ts-mu0 / --ts-sigma0 / --ts-sigma-obs` | Gaussian TS prior / noise |
| `--warmup-epochs` | warm start before sensitivity analysis (gradient method) |
| `--sens-batches` | batches used to estimate the sensitivity matrix |
| `--chain-width` | columns per layer per chain (`1` = paper's pure chain) |
| `--lora-rank / --lora-alpha` | LoRA hyper-parameters |
| `--epochs / --patience` | max epochs and early-stopping patience |
| `--lr / --weight-decay / --beta1 / --beta2` | AdamW settings (β₁=0.90 per paper) |

Run `python main.py --help` for the full list.

---

## 5. How the paper maps onto the code

**PaCA mechanism (`src/models/paca.py`).** For a frozen weight `W ∈ R^{out×in}`,
PaCA tunes a subset of *input columns* (each column = all outgoing connections
from one input neuron). `PaCALinear` keeps `W` as a frozen buffer and holds a
small trainable `delta` for only the active columns, using the identity

```
h = W x  −  W[:, active] x_active  +  delta x_active
```

so it is *exactly* equal to using `delta` for the active columns and `W`
elsewhere, while only `delta` (`out × r`) participates in autograd. When the
active set changes between epochs the trained columns are merged back into `W`
so learning persists.

**MAB formulation (Sec. 4.1).** An arm is `{layer_name: [column indices]}`.
Random arms (`src/bandit/arms.py`) partition each layer's columns into `N`
disjoint groups (empty pairwise intersection, as required). Selecting arms
activates the union of their columns.

**Sensitivity & chains (`src/sensitivity.py`).**
- Proxy loss (Eq. 4): `L_proxy = mean(logits)` — label-agnostic.
- Sensitivity (Eq. 5): `S^{(l)} = E_batch[ |∇_{W^{(l)}} L_proxy| ]`, computed
  exactly as `(dL/dy)^T x` from hooked forward inputs and backward grad-outputs.
- Chains (Eq. 6): from `K` distinct high-sensitivity start columns, greedily
  follow `i* = argmax_i S^{(l)}_{i,j}` layer by layer. Each chain owns one
  column per layer and becomes an arm.

**Bandits.**
- UCB (`src/bandit/ucb.py`, Algorithm 1): `U_i = μ_i + α·√(log t /(N_i+1))`,
  incremental mean update.
- Gaussian TS (`src/bandit/thompson.py`, Algorithm 2): sample per arm, select
  top-K, exact Bayesian posterior update with known observation noise.

**Reward (Eq. 7).** `r = A_after − A_before`, the validation-accuracy jump from
one epoch of tuning the selected arm(s). These non-stationary rewards feed the
bandit updates.

Because re-selecting an arm re-initialises `delta` to the current frozen values
(zero correction), the model's output is unchanged at activation, so
`A_before` for epoch *t* is exactly `A_after` from epoch *t−1*. We therefore
evaluate the validation set **once per epoch** and reuse the previous result as
the next epoch's baseline — one baseline evaluation at the start of a run, then
a single evaluation per epoch. Rewards are numerically identical to computing
`A_before` and `A_after` separately.

---

## 6. Design decisions & assumptions

The paper leaves a few implementation details unspecified; the choices below
are documented so behaviour is transparent and adjustable. None affect the
correctness of the core algorithms.

1. **Classification head** is always trained (a fresh `Linear` for the new label
   space); pass `--freeze-head` to disable. Adapter arms control the *backbone*
   column selection.
2. **Random arm width.** Each random arm owns exactly `--rank` columns per
   layer, and arms are disjoint per layer (`N·rank ≤ in_features`). This keeps
   each arm's per-epoch budget equal to PaCA's while satisfying the paper's
   non-overlap requirement.
3. **Gradient chains across heterogeneous layers.** Adapted layers need not have
   matching dimensions (e.g. `qkv` has `out=3·in`), so the propagated output
   index `i*` is mapped into the next layer's input range modulo its width.
   `--chain-width > 1` broadens each chain with the next most column-sensitive
   inputs (an optional extension; `1` reproduces the paper's pure chains).
4. **Gradient warm start** tunes a broad random adapter set (`rank·num_arms`
   columns/layer) for `--warmup-epochs`, after which those updates are merged
   and the sensitivity analysis is run.
5. **Optimizer** is AdamW (`β₁=0.90` per the paper). Because the active
   parameter set changes each epoch for the dynamic methods, the optimizer is
   rebuilt at each epoch boundary.
6. **Early stopping / total epochs.** The paper reports method-dependent epoch
   counts; we reproduce this via early stopping on validation accuracy
   (`--patience`) under an `--epochs` cap.
7. **Dataset splits.** CIFAR-100 / SVHN / DTD / FGVC-Aircraft use official
   splits (DTD uses partition 1, train+val pooled for training); Flowers-102,
   Caltech-101, EuroSAT and SUN397 (no official train/test split of the needed
   size) use deterministic stratified 80/20 splits. A `--val-frac` hold-out is
   carved from train for rewards and early stopping, and `--max-train-samples`
   optionally caps the *training* portion (class-stratified) while leaving the
   validation and test sets fixed, so runs at different budgets stay comparable.
8. **Reported metrics.** Every run logs and saves `trainable_params` (the active
   adapter budget plus the classification head, measured before the final merge
   so deltas are counted), its percentage of the full model, and — on CUDA —
   `peak_gpu_mem_mb` (peak allocated memory over the run). Throughput mode also
   reports per-batch-size peak memory.

---

## 7. Project layout

```
bandit-paca/
├── README.md
├── requirements.txt
├── download_data.py          # downloads all datasets to ./data/<name>
├── main.py                   # CLI entry point
├── scripts/
│   └── run_examples.sh
└── src/
    ├── config.py             # argparse / all CLI flags
    ├── data.py               # dataset builders + transforms + fake data
    ├── sensitivity.py        # sensitivity matrix + gradient chains (Eqs. 4-6)
    ├── trainer.py            # training loops for all six methods
    ├── utils.py              # seeding, metrics, timing, logging
    ├── models/
    │   ├── paca.py           # PaCALinear + PaCAManager
    │   ├── lora.py           # LoRALinear baseline
    │   └── vit.py            # backbone build + layer wrapping
    └── bandit/
        ├── arms.py           # random arm construction
        ├── ucb.py            # UCB (Algorithm 1)
        └── thompson.py       # Gaussian Thompson Sampling (Algorithm 2)
```

"""Command line configuration.

Every knob described in the paper is exposed as a CLI flag so that all six
methods and their ablations can be run from a single entry point.
"""

import argparse

METHODS = ["lora", "paca", "r_paca", "ucb_paca", "ts_paca", "gradient_paca"]
DATASETS = ["cifar100", "flowers102", "caltech101", "svhn",
            "dtd", "fgvc_aircraft", "eurosat", "sun397"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bandit-based Connection Selection for Efficient Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- method / data ----------------------------------------------------
    p.add_argument("--method", type=str, default="gradient_paca", choices=METHODS,
                   help="Fine-tuning method to run.")
    p.add_argument("--dataset", type=str, default="cifar100", choices=DATASETS,
                   help="Downstream image-classification dataset.")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Root folder where datasets are stored / downloaded.")
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="If > 0, cap the number of TRAINING samples via a class-"
                        "stratified subsample (proportional per class). The validation "
                        "hold-out and test set are unaffected, so runs stay comparable. "
                        "0 = use all training data. Use e.g. 1000 for a VTAB-1k-style "
                        "low-data regime that de-saturates strong backbones.")

    # ---- backbone ---------------------------------------------------------
    p.add_argument("--model", type=str, default="vit_base_patch16_224",
                   help="timm model name (ViT-Base/16 pre-trained on ImageNet-1K by default).")
    p.add_argument("--image-size", type=int, default=224, help="Input resolution.")
    p.add_argument("--target-modules", type=str, default="attn.qkv,attn.proj",
                   help="Comma separated substrings identifying Linear layers to adapt. "
                        "Examples: 'attn.qkv,attn.proj' or 'attn.qkv,attn.proj,mlp.fc1,mlp.fc2'.")
    p.add_argument("--train-head", action="store_true", default=True,
                   help="Train the (freshly initialised) classification head.")
    p.add_argument("--freeze-head", dest="train_head", action="store_false",
                   help="Freeze the classification head (not recommended).")

    # ---- PaCA / arms ------------------------------------------------------
    p.add_argument("--rank", type=int, default=16,
                   help="Number of trainable columns (input neurons) per adapted layer, "
                        "per arm. This is the PaCA budget r.")
    p.add_argument("--num-arms", type=int, default=6,
                   help="Number of arms N in the multi-armed-bandit formulation.")
    p.add_argument("--select-size", type=int, default=1,
                   help="Number of arms K selected per epoch. UCB defaults to 1.")

    # ---- LoRA -------------------------------------------------------------
    p.add_argument("--lora-rank", type=int, default=8, help="Rank r for LoRA.")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="Scaling alpha for LoRA.")

    # ---- UCB --------------------------------------------------------------
    p.add_argument("--ucb-alpha", type=float, default=1.0,
                   help="Exploration coefficient alpha in the UCB rule.")

    # ---- Thompson Sampling ------------------------------------------------
    p.add_argument("--ts-mu0", type=float, default=0.0, help="Prior mean mu0.")
    p.add_argument("--ts-sigma0", type=float, default=1.0, help="Prior std sigma0.")
    p.add_argument("--ts-sigma-obs", type=float, default=1.0,
                   help="Observation noise std sigma_obs for the Bayesian update.")

    # ---- Gradient chains --------------------------------------------------
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="Warm-start epochs before sensitivity analysis / bandit selection.")
    p.add_argument("--sens-batches", type=int, default=8,
                   help="Number of batches used to estimate the sensitivity matrix.")
    p.add_argument("--chain-width", type=int, default=1,
                   help="Columns owned per layer by each gradient chain (1 = paper's pure chain).")

    # ---- optimisation -----------------------------------------------------
    p.add_argument("--epochs", type=int, default=30, help="Maximum number of epochs.")
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience on validation accuracy (0 disables).")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    p.add_argument("--eval-batch-size", type=int, default=128, help="Evaluation batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    p.add_argument("--beta1", type=float, default=0.90,
                   help="AdamW beta1 (paper uses momentum 0.90).")
    p.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2.")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of the training set held out for validation / rewards.")
    p.add_argument("--amp", action="store_true", help="Enable mixed-precision training (CUDA).")

    # ---- infra ------------------------------------------------------------
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'cpu' or 'cuda:0'.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--output-dir", type=str, default="./runs",
                   help="Where JSON logs / results are written.")
    p.add_argument("--tag", type=str, default="", help="Optional run tag appended to the output name.")

    # ---- smoke test / throughput -----------------------------------------
    p.add_argument("--fake-data", action="store_true",
                   help="Use random tensors instead of real datasets (fast pipeline smoke test).")
    p.add_argument("--fake-classes", type=int, default=10, help="Classes for --fake-data.")
    p.add_argument("--fake-size", type=int, default=256, help="Samples for --fake-data.")
    p.add_argument("--measure-throughput", action="store_true",
                   help="Benchmark training throughput (images/sec) instead of full training.")
    p.add_argument("--throughput-batch-sizes", type=str, default="16,32,64",
                   help="Comma separated batch sizes for throughput benchmarking.")
    p.add_argument("--throughput-iters", type=int, default=20,
                   help="Timed iterations per batch size when benchmarking throughput.")

    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)

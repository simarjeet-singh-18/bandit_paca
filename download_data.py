#!/usr/bin/env python
"""Download all four benchmark datasets into the correct paths.

Usage
-----
    python download_data.py --data-root ./data
    python download_data.py --data-root ./data --datasets cifar100 svhn

Each dataset is fetched with torchvision's ``download=True`` into
``<data-root>/<dataset>``. Caltech-101 downloads from Google Drive and may be
rate-limited; re-run if it fails.
"""

import argparse
import os

DATASETS = ["cifar100", "flowers102", "caltech101", "svhn"]


def download_cifar100(root):
    from torchvision.datasets import CIFAR100
    CIFAR100(root=root, train=True, download=True)
    CIFAR100(root=root, train=False, download=True)


def download_svhn(root):
    from torchvision.datasets import SVHN
    SVHN(root=root, split="train", download=True)
    SVHN(root=root, split="test", download=True)


def download_flowers102(root):
    from torchvision.datasets import Flowers102
    for split in ("train", "val", "test"):
        Flowers102(root=root, split=split, download=True)


def download_caltech101(root):
    from torchvision.datasets import Caltech101
    Caltech101(root=root, target_type="category", download=True)


DOWNLOADERS = {
    "cifar100": download_cifar100,
    "svhn": download_svhn,
    "flowers102": download_flowers102,
    "caltech101": download_caltech101,
}


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root folder for all datasets.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                        help="Subset of datasets to download.")
    args = parser.parse_args()

    for name in args.datasets:
        root = os.path.join(args.data_root, name)
        os.makedirs(root, exist_ok=True)
        print(f"==> Downloading {name} into {root} ...")
        try:
            DOWNLOADERS[name](root)
            print(f"    {name}: done.")
        except Exception as exc:  # noqa: BLE001
            print(f"    {name}: FAILED ({exc}).")
            if name == "caltech101":
                print("    Caltech-101 uses Google Drive and is often rate-limited; "
                      "please re-run this script to retry.")

    print("All requested downloads processed.")


if __name__ == "__main__":
    main()

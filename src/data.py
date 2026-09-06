"""Dataset construction for the four image-classification benchmarks.

All images are resized to ``image_size`` and normalised with ImageNet
statistics (the ViT backbone's pre-training statistics). Grayscale images
(e.g. some Caltech-101 samples) are converted to RGB.

Returned splits:
    train_ds  -- training data minus a validation hold-out
    val_ds    -- validation hold-out (used for rewards + early stopping)
    test_ds   -- official / held-out test set
    num_classes

Notes on splits (documented so the pipeline is transparent):
    * CIFAR-100 / SVHN use their official train/test splits.
    * Flowers-102 merges the tiny official train+val+test into one pool and
      re-splits 80/20 (the paper reports a custom split of all 8189 images).
    * Caltech-101 has no official split; we make a deterministic stratified
      80/20 split.
    * The validation hold-out is a fraction ``val_frac`` carved from *train*.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _transforms(image_size: int, train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _RGB(Dataset):
    """Wrap a dataset so every image is RGB before its transform runs."""

    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(label)


def _stratified_split(labels, frac: float, seed: int):
    """Deterministic stratified split returning (idx_a, idx_b)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx_a, idx_b = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1.0 - frac)))
        idx_a.extend(idx[:cut].tolist())
        idx_b.extend(idx[cut:].tolist())
    return idx_a, idx_b


# --------------------------------------------------------------------------- #
# per-dataset raw builders (return PIL-image datasets, no tensor transform yet)
# --------------------------------------------------------------------------- #
def _build_cifar100(root):
    from torchvision.datasets import CIFAR100
    train = CIFAR100(root=root, train=True, download=False)
    test = CIFAR100(root=root, train=False, download=False)
    return train, test, 100


def _build_svhn(root):
    from torchvision.datasets import SVHN
    train = SVHN(root=root, split="train", download=False)
    test = SVHN(root=root, split="test", download=False)
    return train, test, 10


def _build_flowers102(root, seed):
    from torchvision.datasets import Flowers102
    parts = [Flowers102(root=root, split=s, download=False) for s in ("train", "val", "test")]
    pool = ConcatDataset(parts)
    labels = []
    for part in parts:
        labels.extend(list(part._labels))  # torchvision Flowers102 stores integer labels here
    labels = np.asarray(labels)
    train_idx, test_idx = _stratified_split(labels, frac=0.2, seed=seed)
    train = Subset(pool, train_idx)
    test = Subset(pool, test_idx)
    return train, test, 102


def _build_caltech101(root, seed):
    from torchvision.datasets import Caltech101
    full = Caltech101(root=root, target_type="category", download=False)
    labels = np.asarray(full.y)
    train_idx, test_idx = _stratified_split(labels, frac=0.2, seed=seed)
    train = Subset(full, train_idx)
    test = Subset(full, test_idx)
    return train, test, 101


class FakeImageDataset(Dataset):
    """Random tensors that look like normalised images -- pipeline smoke test."""

    def __init__(self, n, num_classes, image_size, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, image_size, image_size, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], int(self.y[idx])


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_datasets(name: str, root: str, image_size: int, val_frac: float, seed: int):
    """Return (train_ds, val_ds, test_ds, num_classes) ready for DataLoaders."""
    ds_root = os.path.join(root, name)  # matches download_data.py layout
    if name == "cifar100":
        raw_train, raw_test, num_classes = _build_cifar100(ds_root)
    elif name == "svhn":
        raw_train, raw_test, num_classes = _build_svhn(ds_root)
    elif name == "flowers102":
        raw_train, raw_test, num_classes = _build_flowers102(ds_root, seed)
    elif name == "caltech101":
        raw_train, raw_test, num_classes = _build_caltech101(ds_root, seed)
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

    train_tf = _transforms(image_size, train=True)
    eval_tf = _transforms(image_size, train=False)

    full_train = _RGB(raw_train, train_tf)
    test_ds = _RGB(raw_test, eval_tf)

    # validation hold-out carved from train
    n = len(full_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    # validation should use eval transforms -> build a separate RGB view
    val_view = _RGB(raw_train, eval_tf)
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(val_view, val_idx)

    return train_ds, val_ds, test_ds, num_classes


def build_fake_datasets(num_classes: int, size: int, image_size: int, val_frac: float):
    n_test = max(64, size // 4)
    train = FakeImageDataset(size, num_classes, image_size, seed=1)
    test = FakeImageDataset(n_test, num_classes, image_size, seed=2)
    n_val = max(16, int(size * val_frac))
    val = FakeImageDataset(n_val, num_classes, image_size, seed=3)
    return train, val, test, num_classes


def make_loaders(train_ds, val_ds, test_ds, batch_size, eval_batch_size,
                 num_workers) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

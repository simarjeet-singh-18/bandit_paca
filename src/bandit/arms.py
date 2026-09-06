"""Arm construction utilities.

An *arm* is a mapping ``{layer_name: [column indices]}`` describing which input
columns of which adapted layers it owns. Two constructors are provided:

* :func:`build_random_arms` -- the simple, scalable baseline used by UCB-PaCA
  and TS-PaCA. For every adapted layer we sample ``num_arms * rank`` distinct
  columns and partition them into ``num_arms`` disjoint groups of size ``rank``.
  Arms are therefore non-overlapping (empty pairwise intersection), matching
  the paper's requirement, with each arm holding exactly ``rank`` columns per
  layer so its per-epoch budget matches PaCA.

Gradient-aligned chain arms are built in :mod:`src.sensitivity`.
"""

from typing import Dict, List, Iterable

import numpy as np


Arm = Dict[str, List[int]]


def build_random_arms(layer_in_features: Dict[str, int], num_arms: int, rank: int,
                      rng: np.random.Generator) -> List[Arm]:
    """Partition columns of every layer into ``num_arms`` disjoint arms."""
    arms: List[Arm] = [dict() for _ in range(num_arms)]
    for name, in_f in layer_in_features.items():
        n_take = min(num_arms * rank, in_f)
        perm = rng.permutation(in_f)[:n_take]
        # split into num_arms (near-)equal disjoint chunks
        chunks = np.array_split(perm, num_arms)
        for i, chunk in enumerate(chunks):
            arms[i][name] = [int(c) for c in chunk]
    return arms


def union_arms(arms: List[Arm], selected: Iterable[int]) -> Arm:
    """Union of the columns owned by the ``selected`` arm indices, per layer."""
    merged: Dict[str, set] = {}
    for idx in selected:
        for name, cols in arms[idx].items():
            merged.setdefault(name, set()).update(cols)
    return {name: sorted(cols) for name, cols in merged.items()}


def arms_to_layer_dims(arms: List[Arm]) -> Dict[str, int]:
    """Number of distinct columns each layer contributes across all arms."""
    dims: Dict[str, set] = {}
    for arm in arms:
        for name, cols in arm.items():
            dims.setdefault(name, set()).update(cols)
    return {name: len(cols) for name, cols in dims.items()}

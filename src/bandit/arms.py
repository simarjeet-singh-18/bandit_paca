"""Arm construction utilities.

An *arm* is a mapping ``{layer_name: [column indices]}`` describing which input
columns of which adapted layers it owns. Two constructors are provided:

* :func:`build_random_arms` -- the random-arm baseline used by UCB-PaCA and
  TS-PaCA. Following Sec. 4.1 of the paper, the adapted weights are partitioned
  into *non-overlapping and exhaustive* groups: every column of every adapted
  layer belongs to exactly one arm, and the union of all arms is the full set of
  adapted weights. Each arm owns ``rank`` columns per layer, so the number of
  arms is derived as ``ceil(in_features / rank)`` (it is NOT a free parameter for
  these methods). Selecting a single arm therefore updates ``rank`` columns per
  layer, matching PaCA's budget r; the paper's rank ablation (Table 6) is
  reproduced by sweeping ``rank`` (which sweeps N with it).

Gradient-aligned chain arms are built in :mod:`src.sensitivity`; there the number
of arms (chains) IS a free knob, per the paper's N/K ablation (Table 7).
"""

from typing import Dict, List, Iterable

import numpy as np


Arm = Dict[str, List[int]]


def build_random_arms(layer_in_features: Dict[str, int], rank: int,
                      rng: np.random.Generator) -> List[Arm]:
    """Exhaustive, disjoint partition of every adapted layer's columns into arms.

    Each layer's ``in_features`` columns are randomly shuffled and split into
    ``ceil(in_features / rank)`` near-equal blocks of ~``rank`` columns; block
    ``i`` of every layer forms arm ``i``. The union of all arms recovers the full
    column set and any two arms are disjoint (paper Sec. 4.1). The number of arms
    equals the largest per-layer block count, so if adapted layers differ in width
    the widest layer sets N and narrower layers simply contribute to the first
    arms; exhaustiveness and disjointness still hold.
    """
    per_layer_blocks: Dict[str, List[List[int]]] = {}
    max_blocks = 0
    for name, in_f in layer_in_features.items():
        perm = rng.permutation(in_f)
        n_blocks = max(1, int(np.ceil(in_f / rank)))
        blocks = np.array_split(perm, n_blocks)      # near-equal, size ~rank
        per_layer_blocks[name] = [[int(c) for c in b] for b in blocks]
        max_blocks = max(max_blocks, n_blocks)

    arms: List[Arm] = [dict() for _ in range(max_blocks)]
    for name, blocks in per_layer_blocks.items():
        for i, block in enumerate(blocks):
            arms[i][name] = block
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

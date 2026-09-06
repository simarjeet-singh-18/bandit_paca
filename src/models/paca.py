"""Partial Connection Adaptation (PaCA) building blocks.

Paper mechanism (Sec. 3.3 / Fig. 1):
    - The pre-trained weight matrix W in R^{out x in} is kept frozen.
    - PaCA chooses a subset I of *input neurons* (columns of W). For every
      j in I the whole j-th column of W is updated during back-prop; every
      other column stays frozen.

To support the bandit methods (R-PaCA / UCB / TS / Gradient) the set of
trainable columns must be able to *change between epochs*. We therefore store
the full weight as a (frozen) buffer and materialise a small trainable
``delta`` parameter that holds *only the active columns*:

        h = W_frozen x  -  W_frozen[:, active] x_active  +  delta x_active

which is *exactly* equal to using ``delta`` for the active columns and the
frozen weight everywhere else. Only ``delta`` (out x r) participates in
autograd, giving the memory profile PaCA is designed for.

When the active set changes we ``merge()`` the trained columns back into the
frozen buffer (so the learned values persist) and re-extract the new columns.
"""

from typing import Iterable, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaCALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with column-wise partial tuning."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Full pre-trained weight, frozen. Kept as a buffer so the optimizer
        # never sees it and it is not counted as a trainable parameter.
        self.register_buffer("weight", linear.weight.detach().clone())

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

        # active_indices: LongTensor of currently trainable columns (or None)
        # delta: nn.Parameter [out, len(active_indices)] (or None)
        self.active_indices: Optional[torch.Tensor] = None
        self.delta: Optional[nn.Parameter] = None

    # ------------------------------------------------------------------ #
    # selection management
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def merge(self) -> None:
        """Write the currently-trained columns back into the frozen weight."""
        if self.delta is not None and self.active_indices is not None:
            self.weight[:, self.active_indices] = self.delta.detach().to(self.weight.dtype)
        self.delta = None
        self.active_indices = None

    def set_active(self, indices: Optional[Iterable[int]]) -> None:
        """Make ``indices`` (input columns) the trainable set for this layer.

        Any previously trained columns are merged back first so their learned
        values are preserved across selections.
        """
        self.merge()
        if indices is None:
            return
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=self.weight.device)
        if idx.numel() == 0:
            return
        idx = torch.unique(idx)  # guard against accidental duplicates
        self.active_indices = idx
        # Initialise delta with the current (possibly already-trained) values so
        # that the correction term starts at exactly zero.
        self.delta = nn.Parameter(self.weight.index_select(1, idx).clone())

    def is_active(self) -> bool:
        return self.delta is not None

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)  # frozen contribution (no grad)
        if self.delta is not None:
            idx = self.active_indices
            x_sel = x.index_select(-1, idx)                       # [..., r]
            frozen_sel = self.weight.index_select(1, idx)         # [out, r] (no grad)
            # delta - frozen_sel is zero at init and carries grad through delta.
            out = out + F.linear(x_sel, self.delta - frozen_sel)
        return out

    def extra_repr(self) -> str:
        r = 0 if self.active_indices is None else int(self.active_indices.numel())
        return f"in_features={self.in_features}, out_features={self.out_features}, active={r}"


class PaCAManager:
    """Coordinates a collection of :class:`PaCALinear` layers.

    An *arm* is represented as a mapping ``{layer_name: [column indices]}``.
    Selecting a set of arms simply activates the union of their columns.
    """

    def __init__(self, layers: Dict[str, PaCALinear]):
        # preserve insertion (=depth) order
        self.layers: Dict[str, PaCALinear] = dict(layers)

    # -- introspection ------------------------------------------------------
    @property
    def layer_names(self) -> List[str]:
        return list(self.layers.keys())

    def layer_in_features(self) -> Dict[str, int]:
        return {name: layer.in_features for name, layer in self.layers.items()}

    # -- activation ---------------------------------------------------------
    def activate(self, arm_columns: Dict[str, Iterable[int]]) -> None:
        """Activate exactly the columns in ``arm_columns`` (per layer).

        Layers absent from the mapping are deactivated (fully frozen).
        """
        for name, layer in self.layers.items():
            cols = arm_columns.get(name, None)
            layer.set_active(cols)

    def deactivate_all(self) -> None:
        for layer in self.layers.values():
            layer.set_active(None)

    def merge_all(self) -> None:
        for layer in self.layers.values():
            layer.merge()

    # -- parameters ---------------------------------------------------------
    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for layer in self.layers.values():
            if layer.delta is not None:
                params.append(layer.delta)
        return params

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

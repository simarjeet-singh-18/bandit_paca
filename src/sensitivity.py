"""Sensitivity analysis and gradient-aligned chain construction (Sec. 4.1).

Proxy objective (Eq. 4) -- label-agnostic mean of the output activations:
        L_proxy = mean_{b,d} y_{b,d}

Sensitivity matrix (Eq. 5) -- batch-averaged absolute gradient of L_proxy
w.r.t. each adapted layer's weight:
        S^{(l)} = E_batch [ | grad_{W^{(l)}} L_proxy | ]

For a linear layer y = x W^T the exact gradient of a scalar loss w.r.t. W is
(dL/dy)^T @ x, which we assemble from forward inputs and backward grad-outputs
captured via hooks -- no temporary parameters required.

Chain construction (Eq. 6) -- greedily follow the most influential outgoing
connection layer by layer:
        i* = argmax_i S^{(l)}_{i, j}
propagating i* forward as the input index of the next adapted layer. Each chain
(one column per layer) is an arm.
"""

from typing import Dict, List

import numpy as np
import torch

from .bandit.arms import Arm
from .models.paca import PaCAManager


@torch.no_grad()
def _accumulate(store, name, tensor):
    store[name] = tensor


def compute_sensitivity(model, manager: PaCAManager, loader, device,
                        num_batches: int = 8) -> Dict[str, np.ndarray]:
    """Estimate S^{(l)} for every adapted layer as a numpy array [out, in]."""
    was_training = model.training
    model.eval()

    captured_in: Dict[str, torch.Tensor] = {}
    captured_gout: Dict[str, torch.Tensor] = {}
    handles = []

    def make_fwd(name):
        def hook(module, inp, out):
            captured_in[name] = inp[0].detach()
        return hook

    def make_bwd(name):
        def hook(module, grad_in, grad_out):
            captured_gout[name] = grad_out[0].detach()
        return hook

    for name, layer in manager.layers.items():
        handles.append(layer.register_forward_hook(make_fwd(name)))
        handles.append(layer.register_full_backward_hook(make_bwd(name)))

    sens: Dict[str, np.ndarray] = {
        name: np.zeros((layer.out_features, layer.in_features), dtype=np.float64)
        for name, layer in manager.layers.items()
    }
    counts = 0

    it = iter(loader)
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        images = batch[0].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(images)                       # [B, D]
        loss = logits.mean()                         # Eq. 4 (label-agnostic)
        loss.backward()

        for name in manager.layers:
            if name not in captured_in or name not in captured_gout:
                continue
            x = captured_in[name]
            g = captured_gout[name]
            x = x.reshape(-1, x.shape[-1])            # [*, in]
            g = g.reshape(-1, g.shape[-1])            # [*, out]
            grad_w = g.transpose(0, 1) @ x           # [out, in] = (dL/dy)^T x
            sens[name] += grad_w.abs().double().cpu().numpy()
        counts += 1

    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    if counts == 0:
        raise RuntimeError("Sensitivity estimation saw no batches.")
    for name in sens:
        sens[name] /= counts                         # batch-average (Eq. 5)
    return sens


def build_gradient_chains(sens: Dict[str, np.ndarray], layer_order: List[str],
                          num_chains: int, chain_width: int = 1) -> List[Arm]:
    """Construct ``num_chains`` gradient-aligned chains, each an arm.

    Starting indices are the ``num_chains`` most column-sensitive input neurons
    of the first adapted layer (distinct), giving diverse, task-relevant chains.
    Because adapted layers need not have matching dimensions, the propagated
    output index i* is mapped into the next layer's input range modulo its width.
    """
    if not layer_order:
        return []
    first = layer_order[0]
    first_in = sens[first].shape[1]

    # column sensitivity of the first layer = sum over outputs
    col_score = sens[first].sum(axis=0)
    order = np.argsort(-col_score, kind="stable")
    starts = [int(order[c % len(order)]) for c in range(num_chains)]
    # ensure distinct starts where possible
    seen, uniq_starts = set(), []
    for s in starts:
        if s not in seen:
            uniq_starts.append(s); seen.add(s)
    for cand in order:
        if len(uniq_starts) >= num_chains:
            break
        if int(cand) not in seen:
            uniq_starts.append(int(cand)); seen.add(int(cand))
    starts = uniq_starts[:num_chains]

    arms: List[Arm] = []
    for start in starts:
        arm: Arm = {}
        j = start
        for li, name in enumerate(layer_order):
            S = sens[name]
            out_dim, in_dim = S.shape
            j = j % in_dim
            cols = [j]
            if chain_width > 1:
                # broaden the arm with the next most column-sensitive inputs
                cscore = S.sum(axis=0)
                cand = np.argsort(-cscore, kind="stable")
                for c in cand:
                    c = int(c)
                    if c not in cols:
                        cols.append(c)
                    if len(cols) >= chain_width:
                        break
            arm[name] = sorted(set(cols))
            # propagate strongest outgoing connection forward (Eq. 6)
            i_star = int(np.argmax(S[:, j]))
            j = i_star
        arms.append(arm)
    return arms

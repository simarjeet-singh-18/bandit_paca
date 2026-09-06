"""Backbone construction and layer wrapping.

Builds an ImageNet-pretrained ViT (via ``timm``) with a fresh classification
head for the downstream task, and replaces the target ``nn.Linear`` layers
with either :class:`PaCALinear` or :class:`LoRALinear`.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .paca import PaCALinear, PaCAManager
from .lora import LoRALinear


def build_model(model_name: str, num_classes: int, image_size: int = 224) -> nn.Module:
    """Create a pretrained backbone with a fresh head of ``num_classes``."""
    import timm

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        img_size=image_size,
    )
    return model


def _get_module(root: nn.Module, name: str) -> nn.Module:
    mod = root
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new)


def _matching_linears(model: nn.Module, target_substrings: List[str]) -> List[str]:
    """Return names of ``nn.Linear`` modules whose name contains any substring.

    The classification head (``head`` / ``fc``) is explicitly excluded so it is
    never turned into an adapter (it is trained directly instead).
    """
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in ("head", "fc") or name.endswith(".head") or name.endswith(".fc"):
            continue
        if any(sub in name for sub in target_substrings):
            names.append(name)
    return names


def wrap_paca(model: nn.Module, target_substrings: List[str]) -> Tuple[nn.Module, PaCAManager]:
    """Replace target Linear layers with :class:`PaCALinear` and return a manager."""
    names = _matching_linears(model, target_substrings)
    if not names:
        raise ValueError(
            f"No Linear layers matched target substrings {target_substrings}. "
            "Inspect the model with --measure-throughput or print(model)."
        )
    wrapped: Dict[str, PaCALinear] = {}
    for name in names:
        linear = _get_module(model, name)
        paca = PaCALinear(linear)
        _set_module(model, name, paca)
        wrapped[name] = paca
    return model, PaCAManager(wrapped)


def wrap_lora(model: nn.Module, target_substrings: List[str], r: int, alpha: float) -> nn.Module:
    """Replace target Linear layers with :class:`LoRALinear`."""
    names = _matching_linears(model, target_substrings)
    if not names:
        raise ValueError(f"No Linear layers matched target substrings {target_substrings}.")
    for name in names:
        linear = _get_module(model, name)
        _set_module(model, name, LoRALinear(linear, r=r, alpha=alpha))
    return model


def get_head_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return the classification-head parameters."""
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module) and any(True for _ in head.parameters()):
            return list(head.parameters())
    # fallbacks
    for attr in ("head", "fc", "classifier"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Module):
                return list(head.parameters())
    raise ValueError("Could not locate a classification head on the model.")


def freeze_backbone(model: nn.Module, train_head: bool = True) -> None:
    """Freeze every base parameter; optionally leave the head trainable.

    Adapter parameters (LoRA A/B, PaCA deltas) are created *after* this call
    or are managed dynamically, so they are unaffected.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    if train_head:
        for p in get_head_parameters(model):
            p.requires_grad_(True)

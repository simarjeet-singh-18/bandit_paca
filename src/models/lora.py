"""Low-Rank Adaptation (LoRA) baseline (Hu et al., 2022).

    h = W x + (alpha / r) * (B A) x,   A in R^{r x in}, B in R^{out x r}

The base weight is frozen; A is Kaiming-initialised and B is zero-initialised
so the adapter contributes nothing at the start of training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.register_buffer("weight", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + self.scaling * lora

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, scaling={self.scaling:.3f}")

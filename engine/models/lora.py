"""Low-rank adapters (LoRA) [Hu et al., ref 9].

The base weight W0 stays frozen and each targeted linear layer learns a small
update BA, so the layer computes W0 x + (alpha / r) B A x. Only A and B are
trained and saved, which keeps one writer's adapter to a few megabytes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("ih", "hh", "window", "head")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero: starts as the base model
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, targets=DEFAULT_TARGETS) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return model


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def save_adapter(path, model: nn.Module, r: int, alpha: float, meta: dict | None = None):
    torch.save({"r": r, "alpha": alpha, "weights": lora_state_dict(model), **(meta or {})}, path)


def load_adapter(model: nn.Module, path, device="cpu") -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
    apply_lora(model, ck["r"], ck["alpha"])
    missing = model.load_state_dict(ck["weights"], strict=False)
    if missing.unexpected_keys:
        raise ValueError(f"adapter does not match model: {missing.unexpected_keys[:3]}")
    return model.to(device).eval()

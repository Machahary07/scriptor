"""Engine wrapper for the LSTM-MDN baseline. Torch is imported only when used."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import GRAVES_WEIGHTS
from .base import HandwritingModel, ModelUnavailable, Style


@lru_cache(maxsize=8)
def _load(weights: str, adapter: str | None):
    from .graves_net import load_checkpoint
    from .lora import load_adapter

    net, norm = load_checkpoint(weights)
    if adapter:
        load_adapter(net, adapter)
    return net, norm


class GravesModel(HandwritingModel):
    name = "graves"
    label = "LSTM-MDN baseline"
    description = "Graves-style recurrent model with attention window, style encoder and optional LoRA adapter."

    def __init__(self, weights: Path = GRAVES_WEIGHTS):
        self.weights = Path(weights)

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "PyTorch is not installed"
        if not self.weights.exists():
            return False, f"no trained weights at {self.weights} (run training/pretrain.py)"
        return True, "ready"

    def write_line(self, text: str, style: Style, bias: float, rng: np.random.Generator) -> np.ndarray:
        ok, why = self.available()
        if not ok:
            raise ModelUnavailable(why)
        import torch

        adapter = style.adapter_path if style.adapter_path and Path(style.adapter_path).exists() else None
        net, norm = _load(str(self.weights), adapter)
        mean, std = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
        ref = _reference(style, mean, std)
        g = torch.Generator().manual_seed(int(rng.integers(2**31)))
        with torch.no_grad():
            r = torch.from_numpy(ref).unsqueeze(0)
            s = net.style_enc(r, torch.ones(r.shape[:2]))
            out = net.sample(text, s, bias=bias, generator=g)
        out[:, :2] = out[:, :2] * std + mean
        return out


def _reference(style: Style, mean, std, max_points: int = 1500) -> np.ndarray:
    """Concatenate the writer's normalised samples as input to the style encoder."""
    if style.samples:
        ref = np.concatenate(style.samples, axis=0)[:max_points].copy()
    else:
        ref = np.zeros((1, 3), np.float32)
    ref[:, :2] = (ref[:, :2] - mean) / std
    return ref.astype(np.float32)

"""Model registry. Every generator is reached through get_model(name)."""

from __future__ import annotations

from .base import HandwritingModel, ModelUnavailable, Style
from .external import DiffInkModel, SDTModel
from .font import FontModel
from .graves import GravesModel

_MODELS: dict[str, HandwritingModel] = {m.name: m for m in (FontModel(), GravesModel(), SDTModel(), DiffInkModel())}


def get_model(name: str) -> HandwritingModel:
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'")
    return _MODELS[name]


def list_models() -> list[dict]:
    out = []
    for m in _MODELS.values():
        ok, why = m.available()
        out.append({"name": m.name, "label": m.label, "description": m.description, "available": ok, "status": why})
    return out


__all__ = ["HandwritingModel", "ModelUnavailable", "Style", "get_model", "list_models"]

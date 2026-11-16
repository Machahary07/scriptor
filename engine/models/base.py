"""Common interface for every handwriting generator.

A model takes one line of text and a writer style and returns the line as a
stroke-3 array in x-height units: the baseline is y=0, y grows downwards, the
line starts at x=0 and the height of a lower-case 'x' is about 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class ModelUnavailable(RuntimeError):
    """Raised when a model cannot run here (missing weights, library or GPU)."""


@dataclass
class Style:
    writer_id: str | None = None
    # Measured from the writer's samples; used by the font generator.
    slant: float = 0.12  # radians, positive leans right
    width: float = 1.0  # horizontal stretch of letters
    spacing: float = 1.0  # letter spacing factor
    word_gap: float = 1.0  # word spacing factor
    messiness: float = 1.0  # amount of natural variation
    # Normalised stroke-3 samples, for models that read style from examples.
    samples: list[np.ndarray] = field(default_factory=list)
    adapter_path: str | None = None  # per-writer LoRA weights


class HandwritingModel(ABC):
    name: str = ""
    label: str = ""
    description: str = ""

    def available(self) -> tuple[bool, str]:
        return True, "ready"

    @abstractmethod
    def write_line(self, text: str, style: Style, bias: float, rng: np.random.Generator) -> np.ndarray:
        """Return the line as stroke-3 in x-height units."""

    def estimate_width(self, text: str, style: Style) -> float:
        """Approximate width of the written text in x-height units, for line wrapping."""
        return 0.62 * len(text) * style.width * style.spacing

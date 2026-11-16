"""Adapters for the published research models SDT [ref 4] and DiffInk [ref 5].

Both are separate research code bases with their own environments, so Scriptor
runs them as an external command rather than importing them. The command named
in the environment variable receives a JSON request on stdin:

    {"text": "...", "bias": 1.0, "seed": 7, "samples": [[[dx, dy, p], ...], ...]}

and must print a JSON stroke-3 list [[dx, dy, p], ...] in x-height units on
stdout. Until such a command is configured the model reports itself unavailable
and the rest of Scriptor keeps working with the other models.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess

import numpy as np

from .base import HandwritingModel, ModelUnavailable, Style


class ExternalModel(HandwritingModel):
    env_var = ""

    def available(self) -> tuple[bool, str]:
        if not os.environ.get(self.env_var):
            return False, f"set {self.env_var} to a command that runs the model"
        return True, "ready"

    def write_line(self, text: str, style: Style, bias: float, rng: np.random.Generator) -> np.ndarray:
        cmd = os.environ.get(self.env_var)
        if not cmd:
            raise ModelUnavailable(self.available()[1])
        req = {"text": text, "bias": bias, "seed": int(rng.integers(2**31)),
               "samples": [s.tolist() for s in style.samples]}
        res = subprocess.run(shlex.split(cmd), input=json.dumps(req), capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise ModelUnavailable(f"{self.name} failed: {res.stderr.strip()[-300:]}")
        return np.asarray(json.loads(res.stdout), dtype=np.float32).reshape(-1, 3)


class SDTModel(ExternalModel):
    name = "sdt"
    label = "SDT (few-shot style)"
    description = "Style-Disentangled Transformer; copies a writer from reference samples without training."
    env_var = "SCRIPTOR_SDT_CMD"


class DiffInkModel(ExternalModel):
    name = "diffink"
    label = "DiffInk (latent diffusion)"
    description = "Latent diffusion transformer for final, highest-quality pages. Needs a GPU."
    env_var = "SCRIPTOR_DIFFINK_CMD"

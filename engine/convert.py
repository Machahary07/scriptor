"""Photo-to-strokes conversion for writers without a tablet (InkSight, ref 6).

InkSight runs in its own TensorFlow environment, so it is called as an external
command named by SCRIPTOR_INKSIGHT_CMD. The command gets the image path as its
last argument and prints JSON: [{"text": "...", "strokes": [[[x, y, t], ...], ...]}, ...].
Converted lines are then stored like tablet samples. An OCR check [ref 20] can be
plugged in through SCRIPTOR_OCR_CMD (image path in, recognised text out) to drop
lines whose recognised text does not match.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path


class ConversionUnavailable(RuntimeError):
    pass


def inksight_available() -> bool:
    return bool(os.environ.get("SCRIPTOR_INKSIGHT_CMD"))


def photo_to_samples(image: Path) -> list[dict]:
    cmd = os.environ.get("SCRIPTOR_INKSIGHT_CMD")
    if not cmd:
        raise ConversionUnavailable("set SCRIPTOR_INKSIGHT_CMD to convert photos into strokes")
    res = subprocess.run([*shlex.split(cmd), str(image)], capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise ConversionUnavailable(f"InkSight failed: {res.stderr.strip()[-300:]}")
    return json.loads(res.stdout)

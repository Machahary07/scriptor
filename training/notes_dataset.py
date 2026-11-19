"""Dataset from photographed notes: writer_samples.json -> stroke-3 lines.

Uses the same preprocessing as IAM-OnDB and tablet samples (training/data.py:
normalise slant and x-height, resample, RDP, stroke-3). Lines are split 90/10
before augmentation, so jittered copies of a validation line never reach
training. Augmentation is a light random affine map per copy: scale, slant
(shear) and rotation.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.strokes import polylines_to_stroke3, stroke3_to_polylines  # noqa: E402
from training.data import Line, preprocess  # noqa: E402

WRITER = "me"


def load_samples(path: Path, writer: str = WRITER, min_points: int = 10) -> list[Line]:
    out = []
    for item in json.loads(Path(path).read_text()):
        if not item.get("text", "").strip():
            continue
        s3 = preprocess([np.asarray(s, dtype=np.float64)[:, :2] for s in item["strokes"] if len(s)])
        if s3 is not None and len(s3) >= min_points:
            out.append(Line(item["text"], writer, s3))
    return out


def split_lines(lines: list[Line], val_frac: float = 0.1, seed: int = 0) -> tuple[list[Line], list[Line]]:
    idx = list(range(len(lines)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(val_frac * len(lines)))
    val = set(idx[:n_val])
    return [l for i, l in enumerate(lines) if i not in val], [l for i, l in enumerate(lines) if i in val]


def jitter(s3: np.ndarray, rng: np.random.Generator, scale: float = 0.08, slant: float = 0.12,
           rotate_deg: float = 2.0) -> np.ndarray:
    s = rng.uniform(1 - scale, 1 + scale)
    S = np.diag([s * rng.uniform(0.95, 1.05), s])
    K = np.array([[1.0, rng.uniform(-slant, slant)], [0.0, 1.0]])
    a = np.radians(rng.uniform(-rotate_deg, rotate_deg))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = R @ K @ S
    return polylines_to_stroke3([p @ M.T for p in stroke3_to_polylines(s3)])


def augment(lines: list[Line], copies: int = 3, seed: int = 0) -> list[Line]:
    """The original lines plus `copies` jittered versions of each."""
    rng = np.random.default_rng(seed)
    return lines + [Line(l.text, l.writer, jitter(l.s3, rng)) for _ in range(copies) for l in lines]


def save_lines(path: Path, lines: list[Line]) -> None:
    """All lines in one .npz: concatenated points, line offsets and texts (no pickling)."""
    offsets = np.cumsum([0] + [len(l.s3) for l in lines])
    np.savez_compressed(path, points=np.concatenate([l.s3 for l in lines]).astype(np.float32), offsets=offsets,
                        texts=np.array([l.text for l in lines]), writers=np.array([l.writer for l in lines]))


def load_lines(path: Path) -> list[Line]:
    d = np.load(path)
    o = d["offsets"]
    return [Line(str(t), str(w), d["points"][o[i] : o[i + 1]]) for i, (t, w) in enumerate(zip(d["texts"], d["writers"]))]


def build(samples_json: Path, out_dir: Path, max_points: int = 1000, copies: int = 3, seed: int = 0) -> dict:
    """writer_samples.json -> train.npz (augmented), train_orig.npz, val.npz. Returns counts."""
    lines = load_samples(samples_json)
    n_loaded = len(lines)
    too_long = [l for l in lines if len(l.s3) > max_points]
    lines = [l for l in lines if len(l.s3) <= max_points]
    if len(lines) < 2:
        raise ValueError(f"only {len(lines)} usable lines in {samples_json}; need at least 2")
    train, val = split_lines(lines, seed=seed)
    train_aug = augment(train, copies, seed)
    train_aug = [l for l in train_aug if len(l.s3) <= max_points]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_lines(out_dir / "train.npz", train_aug)
    save_lines(out_dir / "train_orig.npz", train)
    save_lines(out_dir / "val.npz", val)
    return {"lines_in_json": len(json.loads(Path(samples_json).read_text())), "lines_loaded": n_loaded,
            "lines_too_long": len(too_long), "max_points": max_points, "train_lines": len(train),
            "val_lines": len(val), "train_lines_augmented": len(train_aug), "aug_copies": copies,
            "points_train_orig": int(sum(len(l.s3) for l in train)),
            "median_points_per_line": int(np.median([len(l.s3) for l in lines]))}

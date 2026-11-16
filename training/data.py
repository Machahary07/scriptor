"""Training data: IAM-OnDB loader, writer samples, preprocessing and batching.

Preprocessing follows the report: normalise slant and x-height, resample at even
spacing, simplify with RDP, encode as stroke-3, standardise offsets, and split
by writer so test writers are never seen in training.
"""

from __future__ import annotations

import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.paths import rdp  # noqa: E402
from engine.strokes import normalise_line, polylines_to_stroke3, resample  # noqa: E402

RESAMPLE = 0.12  # x-height units between points
RDP_EPS = 0.02


@dataclass
class Line:
    text: str
    writer: str
    s3: np.ndarray  # normalised stroke-3, not yet standardised


def preprocess(polys: list[np.ndarray]) -> np.ndarray | None:
    polys = [p for p in polys if len(p) > 0]
    if not polys:
        return None
    npolys, _ = normalise_line(polys)
    npolys = [rdp(resample(p, RESAMPLE), RDP_EPS) for p in npolys]
    return polylines_to_stroke3(npolys)


# ---- IAM-OnDB ----------------------------------------------------------------

def _transcriptions(ascii_dir: Path) -> dict[str, str]:
    """Map line id (e.g. a01-000u-01) to its text from the ascii/*.txt files."""
    out = {}
    for f in ascii_dir.rglob("*.txt"):
        lines = f.read_text(encoding="latin-1").splitlines()
        if "CSR:" not in "\n".join(lines):
            continue
        body = lines[[i for i, l in enumerate(lines) if l.strip() == "CSR:"][0] + 1 :]
        body = [l.strip() for l in body if l.strip()]
        for i, text in enumerate(body, 1):
            out[f"{f.stem}-{i:02d}"] = text
    return out


def _writers(original_dir: Path) -> dict[str, str]:
    """Map form id (e.g. a01-000u) to writer id, read from original/**/strokes*.xml."""
    out = {}
    if not original_dir.exists():
        return out
    for f in original_dir.rglob("strokes*.xml"):
        m = re.search(r'writerID="(\d+)"', f.read_text(encoding="latin-1", errors="replace")[:4000])
        if m:
            out[f.parent.name + f.stem.replace("strokes", "")] = m.group(1)
    return out


def load_iamondb(root: Path, limit: int | None = None) -> list[Line]:
    """Read IAM-OnDB [ref 2]. Expects root/lineStrokes, root/ascii and (optionally) root/original."""
    texts = _transcriptions(root / "ascii")
    writers = _writers(root / "original")
    lines = []
    for f in sorted((root / "lineStrokes").rglob("*.xml")):
        lid = f.stem
        if lid not in texts:
            continue
        form = lid.rsplit("-", 1)[0]
        tree = ET.parse(f)
        polys = [np.array([[float(p.get("x")), float(p.get("y"))] for p in s.iter("Point")]) for s in tree.iter("Stroke")]
        s3 = preprocess(polys)
        if s3 is None or len(s3) < 10:
            continue
        lines.append(Line(texts[lid], writers.get(form, form[:7]), s3))
        if limit and len(lines) >= limit:
            break
    return lines


def load_writer_samples(sample_dir: Path, writer: str) -> list[Line]:
    import json

    out = []
    for f in sorted(sample_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("text"):
            continue
        s3 = preprocess([np.asarray(s, dtype=np.float64)[:, :2] for s in d["strokes"] if len(s)])
        if s3 is not None:
            out.append(Line(d["text"], writer, s3))
    return out


def synthetic_lines(n: int, writers: int = 20, seed: int = 0) -> list[Line]:
    """Lines written by the font generator with random styles; for testing the training loop."""
    from engine.models import Style
    from engine.models.font import FontModel
    from engine.strokes import stroke3_to_polylines

    words = ("the quick brown fox jumps over a lazy dog while seven bright students write neat notes "
             "about circuits graphs and simple machines in their lab records").split()
    rng = np.random.default_rng(seed)
    model = FontModel()
    styles = [Style(slant=rng.uniform(-0.1, 0.4), width=rng.uniform(0.85, 1.2)) for _ in range(writers)]
    out = []
    for i in range(n):
        w = i % writers
        text = " ".join(rng.choice(words, size=rng.integers(2, 5)))
        s3 = model.write_line(text, styles[w], 0.5, rng)
        s3 = preprocess(stroke3_to_polylines(s3))
        out.append(Line(text, f"syn{w}", s3))
    return out


# ---- splitting and batching ----------------------------------------------------

def split_by_writer(lines: list[Line], seed: int = 0, frac=(0.8, 0.1, 0.1)):
    ws = sorted({l.writer for l in lines})
    random.Random(seed).shuffle(ws)
    n = len(ws)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    sets = set(ws[:a]), set(ws[a:b]), set(ws[b:])
    return tuple([l for l in lines if l.writer in s] for s in sets)


def norm_stats(lines: list[Line]) -> dict:
    allp = np.concatenate([l.s3[:, :2] for l in lines], axis=0)
    return {"mean": allp.mean(0).tolist(), "std": (allp.std(0) + 1e-6).tolist()}


def standardise(s3: np.ndarray, norm: dict) -> np.ndarray:
    out = s3.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] - np.asarray(norm["mean"])) / np.asarray(norm["std"])
    return out


def batches(lines: list[Line], norm: dict, batch_size: int, max_points: int, ref_points: int = 600,
            shuffle: bool = True, seed: int = 0):
    """Yield padded tensors. Each line's style reference is another line by the same writer."""
    import torch
    import torch.nn.functional as F

    from engine.models.graves_net import VOCAB_SIZE, encode_text

    lines = [l for l in lines if len(l.s3) <= max_points]
    by_writer: dict[str, list[Line]] = {}
    for l in lines:
        by_writer.setdefault(l.writer, []).append(l)
    rng = random.Random(seed)
    order = sorted(range(len(lines)), key=lambda i: len(lines[i].s3))  # bucket by length
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    for chunk in chunks:
        items = [lines[i] for i in chunk]
        refs = []
        for l in items:
            pool = [r for r in by_writer[l.writer] if r is not l] or [l]
            refs.append(standardise(rng.choice(pool).s3[:ref_points], norm))
        tgt = [standardise(l.s3, norm) for l in items]
        T, R, U = max(map(len, tgt)), max(map(len, refs)), max(len(l.text) for l in items)
        B = len(items)
        target = torch.zeros(B, T, 3)
        mask = torch.zeros(B, T)
        ref = torch.zeros(B, R, 3)
        ref_mask = torch.zeros(B, R)
        ids = torch.zeros(B, U, dtype=torch.long)
        char_mask = torch.zeros(B, U)
        for j, (t, r, l) in enumerate(zip(tgt, refs, items)):
            target[j, : len(t)] = torch.from_numpy(t)
            mask[j, : len(t)] = 1
            ref[j, : len(r)] = torch.from_numpy(r)
            ref_mask[j, : len(r)] = 1
            e = encode_text(l.text)
            ids[j, : len(e)] = torch.tensor(e)
            char_mask[j, : len(e)] = 1
        x = torch.cat([torch.zeros(B, 1, 3), target[:, :-1]], 1)
        yield dict(x=x, target=target, mask=mask, chars=F.one_hot(ids, VOCAB_SIZE).float(),
                   char_mask=char_mask, ref=ref, ref_mask=ref_mask)

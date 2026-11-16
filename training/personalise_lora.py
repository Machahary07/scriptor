"""Personalise the base model to one writer with a LoRA adapter.

    python training/personalise_lora.py --writer <id>

Uses the writer's tablet samples that have a transcription (10 to 20 lines).
Saves data/writers/<id>/adapter.pt; the engine loads it automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import GRAVES_WEIGHTS, WRITERS_DIR  # noqa: E402
from engine.models.graves_net import load_checkpoint  # noqa: E402
from engine.models.lora import apply_lora, save_adapter  # noqa: E402
from training.data import batches, load_writer_samples  # noqa: E402
from training.pretrain import run_epoch  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", required=True)
    ap.add_argument("--base", type=Path, default=GRAVES_WEIGHTS)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    wdir = WRITERS_DIR / a.writer
    lines = load_writer_samples(wdir / "samples", a.writer)
    if len(lines) < 3:
        sys.exit(f"need at least 3 transcribed samples, found {len(lines)}")
    val = lines[-max(1, len(lines) // 8):]
    train = lines[: len(lines) - len(val)]
    print(f"writer {a.writer}: {len(train)} train / {len(val)} validation lines")

    net, norm = load_checkpoint(a.base, a.device)
    apply_lora(net, a.rank, a.alpha)
    net.to(a.device)
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"trainable LoRA parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.Adam(params, lr=a.lr)
    best, bad, out = float("inf"), 0, wdir / "adapter.pt"
    for ep in range(1, a.epochs + 1):
        net.train()
        tr = run_epoch(net, batches(train, norm, a.batch, 10_000, seed=ep), opt, a.device, a.clip)
        net.eval()
        with torch.no_grad():
            va = run_epoch(net, batches(val, norm, a.batch, 10_000, shuffle=False), None, a.device, a.clip)
        print(f"epoch {ep}: train {tr:.4f}  val {va:.4f}", flush=True)
        if va < best:
            best, bad = va, 0
            save_adapter(out, net, a.rank, a.alpha, {"writer": a.writer, "val_loss": va})
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break
    print(f"saved {out}")


if __name__ == "__main__":
    main()

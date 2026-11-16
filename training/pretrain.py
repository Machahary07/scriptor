"""Pre-train the LSTM-MDN baseline on IAM-OnDB.

    python training/pretrain.py --data /path/to/IAM-OnDB              # full run (GPU / Colab)
    python training/pretrain.py --synthetic 400 --epochs 3 --hidden 64  # quick check of the loop on CPU

Saves data/models/graves_base.pt with the weights and the offset normalisation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import GRAVES_WEIGHTS  # noqa: E402
from engine.models.graves_net import GravesNet, mdn_loss, save_checkpoint  # noqa: E402
from training.data import batches, load_iamondb, norm_stats, split_by_writer, synthetic_lines  # noqa: E402


def run_epoch(net, data, opt, device, clip):
    total, n = 0.0, 0
    for b in data:
        b = {k: v.to(device) for k, v in b.items()}
        style = net.style_enc(b["ref"], b["ref_mask"])
        y = net(b["x"], b["chars"], b["char_mask"], style)
        loss = mdn_loss(net, y, b["target"], b["mask"])
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], clip)
            opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="IAM-OnDB root folder")
    ap.add_argument("--synthetic", type=int, default=0, help="train on N font-generated lines instead")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--max-points", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=400)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=GRAVES_WEIGHTS)
    a = ap.parse_args(argv)

    if a.synthetic:
        lines = synthetic_lines(a.synthetic)
    elif a.data:
        lines = load_iamondb(a.data, a.limit)
    else:
        ap.error("give --data or --synthetic")
    train, val, test = split_by_writer(lines)
    print(f"{len(lines)} lines: {len(train)} train / {len(val)} val / {len(test)} test (split by writer)")
    norm = norm_stats(train)
    net = GravesNet(hidden=a.hidden).to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best, bad = float("inf"), 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        net.train()
        tr = run_epoch(net, batches(train, norm, a.batch, a.max_points, seed=ep), opt, a.device, a.clip)
        net.eval()
        with torch.no_grad():
            va = run_epoch(net, batches(val or train, norm, a.batch, a.max_points, shuffle=False), None, a.device, a.clip)
        print(f"epoch {ep}: train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if va < best:
            best, bad = va, 0
            a.out.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(a.out, net, norm, {"val_loss": va, "epoch": ep})
            print(f"  saved {a.out}")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break


if __name__ == "__main__":
    main()

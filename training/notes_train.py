"""Time-boxed training for the notes -> model notebook.

Stage A trains the LSTM-MDN (graves_net.GravesNet) from scratch; stage B
freezes it and trains LoRA adapters (engine/models/lora.py). Both use the same
loop: one optimiser step per batch until the time budget runs out, with the
validation NLL measured at fixed intervals. Every number logged is measured.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data import batches  # noqa: E402
from training.pretrain import run_epoch  # noqa: E402


def count_params(net: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total)."""
    return (sum(p.numel() for p in net.parameters() if p.requires_grad), sum(p.numel() for p in net.parameters()))


def train_timeboxed(net, train, val, norm, *, minutes: float, lr: float, stage: str, on_best=None,
                    batch: int = 32, max_points: int = 1000, clip: float = 10.0, eval_every_s: float = 60.0,
                    device: str = "cuda", t_offset: float = 0.0, log=print) -> dict:
    """Train until `minutes` have passed. on_best(val_nll, step) is called on each new best.

    Returns steps, wall time and a history of {stage, step, t_min, train, val}
    where train is the mean batch NLL since the previous evaluation.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    def evaluate() -> float:
        net.eval()
        with torch.no_grad():
            v = run_epoch(net, batches(val, norm, batch, max_points, shuffle=False), None, device, clip)
        net.train()
        return v

    t0 = time.time()
    budget = minutes * 60
    history, best, step, epoch = [], float("inf"), 0, 0
    v = evaluate()
    history.append({"stage": stage, "step": 0, "t_min": round(t_offset, 3), "train": None, "val": v})
    log(f"[{stage}] step 0  val {v:.4f}")
    run_loss, run_n, last_eval = 0.0, 0, time.time()
    net.train()
    if not any(len(l.s3) <= max_points for l in train):
        raise ValueError(f"no training line has <= {max_points} points")
    while time.time() - t0 < budget:
        epoch += 1
        for b in batches(train, norm, batch, max_points, seed=epoch):
            run_loss += run_epoch(net, [b], opt, device, clip)
            run_n += 1
            step += 1
            out_of_time = time.time() - t0 >= budget
            if time.time() - last_eval >= eval_every_s or out_of_time:
                v = evaluate()
                tr = run_loss / run_n
                t_min = t_offset + (time.time() - t0) / 60
                history.append({"stage": stage, "step": step, "t_min": round(t_min, 3), "train": tr, "val": v})
                log(f"[{stage}] step {step}  epoch {epoch}  {t_min:5.1f} min  train {tr:.4f}  val {v:.4f}")
                if v < best:
                    best = v
                    if on_best:
                        on_best(v, step)
                run_loss, run_n, last_eval = 0.0, 0, time.time()
            if out_of_time:
                break
    trained = [h for h in history if h["train"] is not None]
    return {"stage": stage, "steps": step, "epochs": epoch, "wall_s": round(time.time() - t0, 1),
            "best_val": best, "final_train": trained[-1]["train"] if trained else None,
            "final_val": history[-1]["val"], "initial_val": history[0]["val"], "history": history}

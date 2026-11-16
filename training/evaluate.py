"""Evaluate generated handwriting and toolpaths.

    python training/evaluate.py --model font --pages 50            # toolpath metrics
    python training/evaluate.py --model graves --writer <id> --cer  # + legibility (needs an OCR)

Metrics (report, section 13):
  * toolpath validity: every point inside the sheet, every path non-empty;
  * efficiency: pen-up travel before and after ordering, plot time per page;
  * legibility (--cer): character error rate of an OCR reading of each rendered line.
    The OCR is TrOCR [ref 20] through the `transformers` package if installed, or any
    command given in SCRIPTOR_OCR_CMD (image path in, text out).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import store  # noqa: E402
from engine.models import get_model  # noqa: E402
from engine.pipeline import generate_strokes, make_gcode  # noqa: E402
from engine.planner import PageSpec  # noqa: E402
from engine.strokes import stroke3_to_polylines  # noqa: E402

SENTENCES = [
    "The experiment was repeated three times and the mean value was taken.",
    "A graph of current against voltage gives a straight line through the origin.",
    "Students should record every reading neatly in the laboratory notebook.",
    "The resistance of the wire increases slightly as its temperature rises.",
    "Water boils at a lower temperature at high altitudes because the pressure is lower.",
    "Every observation was checked twice before the table was completed.",
]


def cer(ref: str, hyp: str) -> float:
    """Levenshtein distance divided by reference length."""
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[-1] / max(len(ref), 1)


def render_line_png(polys, path: Path, px_per_unit: int = 40):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allp = np.concatenate(polys)
    w, h = np.ptp(allp[:, 0]) + 1, np.ptp(allp[:, 1]) + 1
    fig = plt.figure(figsize=(w * px_per_unit / 100, h * px_per_unit / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    for p in polys:
        ax.plot(p[:, 0], p[:, 1], color="black", lw=1.6)
    ax.set_xlim(allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5)
    ax.set_ylim(allp[:, 1].max() + 0.5, allp[:, 1].min() - 0.5)
    ax.axis("off")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def make_ocr():
    cmd = os.environ.get("SCRIPTOR_OCR_CMD")
    if cmd:
        return lambda img: subprocess.run([*shlex.split(cmd), str(img)], capture_output=True, text=True).stdout.strip()
    try:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        return None
    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    def ocr(img):
        px = proc(images=Image.open(img).convert("RGB"), return_tensors="pt").pixel_values
        return proc.batch_decode(model.generate(px, max_new_tokens=96), skip_special_tokens=True)[0]

    return ocr


def toolpath_metrics(model: str, style, pages: int, bias: float) -> dict:
    spec = PageSpec()
    rows, invalid = [], 0
    text = " ".join(SENTENCES * 3)
    for seed in range(pages):
        page = generate_strokes(text, style, model, bias, seed, spec)[0]
        res = make_gcode(page.paths, spec.height)
        pts = np.concatenate(res.paths)
        if (pts < 0).any() or (pts[:, 0] > spec.width).any() or (pts[:, 1] > spec.height).any():
            invalid += 1
        rows.append(res.stats)
    agg = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    return {
        "pages": pages,
        "valid_pages": pages - invalid,
        "mean_paths": agg("paths"),
        "mean_travel_written_mm": agg("travel_mm_written_order"),
        "mean_travel_ordered_mm": agg("travel_mm"),
        "travel_reduction": 1 - agg("travel_mm") / agg("travel_mm_written_order"),
        "mean_plot_s": agg("total_s"),
        "max_plot_s": float(max(r["total_s"] for r in rows)),
        "mean_pen_share": agg("pen_share"),
    }


def cer_metric(model_name: str, style, bias: float, ocr) -> dict:
    model = get_model(model_name)
    rng = np.random.default_rng(0)
    scores = []
    with tempfile.TemporaryDirectory() as d:
        for i, s in enumerate(SENTENCES):
            polys = stroke3_to_polylines(model.write_line(s, style, bias, rng))
            img = Path(d) / f"{i}.png"
            render_line_png(polys, img)
            hyp = ocr(img)
            scores.append(cer(s, hyp))
            print(f"  CER {scores[-1]:.3f}  | {hyp}")
    return {"cer_mean": float(np.mean(scores)), "lines": len(scores)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="font")
    ap.add_argument("--writer", default=None)
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--pages", type=int, default=50)
    ap.add_argument("--cer", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    style = store.load_style(a.writer)
    res = {"model": a.model, "writer": a.writer, "toolpath": toolpath_metrics(a.model, style, a.pages, a.bias)}
    if a.cer:
        ocr = make_ocr()
        if ocr is None:
            print("no OCR available: install transformers + pillow, or set SCRIPTOR_OCR_CMD")
        else:
            res["legibility"] = cer_metric(a.model, style, a.bias, ocr)
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

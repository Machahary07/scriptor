"""Regenerate the report's prototype-page numbers and Figs. 8 and 9.

    python docs/make_figures.py

Uses examples/lab-record.txt (the Studio's sample page) with the font model,
seed 7, bias 1 and the default page settings, and writes:
  docs/figures/fig08-toolpath.png   toolpath preview, drawn paths and pen-up travel
  docs/figures/fig09-ordering.png   pen-up travel and plot time for three toolpath variants
  docs/figures/prototype-page.json  every number quoted in the report
  docs/figures/prototype-page.gcode the final G-code
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.models import Style  # noqa: E402
from engine.pipeline import generate_strokes, make_gcode  # noqa: E402
from engine.planner import PageSpec  # noqa: E402

OUT = ROOT / "docs" / "figures"
SEED = 7

BLUE, RED, ORANGE, GREY = "#1d3fa8", "#d33b3b", "#f08a4b", "#9aa3b2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "examples" / "lab-record.txt").read_text(encoding="utf-8")
    spec = PageSpec()
    page = generate_strokes(text, Style(), "font", 1.0, SEED, spec)[0]
    page.paths = [p.round(3) for p in page.paths]  # as /api/generate returns them

    variants = {
        "written": make_gcode(page.paths, spec.height, order=False, join_mm=0),
        "ordered": make_gcode(page.paths, spec.height, order=True, join_mm=0),
        "joined": make_gcode(page.paths, spec.height, "job=report page=1/1", order=True, join_mm=0.3),
    }
    final = variants["joined"]
    (OUT / "prototype-page.gcode").write_text(final.gcode)
    numbers = {k: v.stats for k, v in variants.items()}
    numbers["gcode_head"] = final.gcode.splitlines()[:14]
    (OUT / "prototype-page.json").write_text(json.dumps(numbers, indent=2))

    # Fig. 8: toolpath preview of the full page.
    fig, ax = plt.subplots(figsize=(6.2, 8.6), dpi=200)
    ax.add_patch(plt.Rectangle((0, 0), spec.width, spec.height, fill=False, ec=GREY, lw=0.6))
    cur = (0.0, spec.height)
    for p in final.paths:
        ax.plot([cur[0], p[0][0]], [cur[1], p[0][1]], color=RED, lw=0.35, ls=(0, (3, 2)), alpha=0.8)
        ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=0.55, solid_capstyle="round")
        cur = p[-1]
    ax.plot([cur[0], 0], [cur[1], spec.height], color=RED, lw=0.35, ls=(0, (3, 2)), alpha=0.8)
    ax.set_xlim(-3, spec.width + 3)
    ax.set_ylim(spec.height + 3, -3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "fig08-toolpath.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Fig. 9: pen-up travel and plot time for the three variants.
    labels = ["Written order", "Nearest-neighbour\nordering", "Ordering +\nstroke joining"]
    st = [numbers[k] for k in ("written", "ordered", "joined")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200)
    bars = a1.bar(labels, [s["travel_mm"] / 1000 for s in st], color=[GREY, ORANGE, BLUE], width=0.6)
    a1.bar_label(bars, [f"{s['travel_mm'] / 1000:.2f} m" for s in st], fontsize=9, padding=2)
    a1.set_title("Pen-up travel", fontsize=11)
    a1.set_ylabel("metres")
    bottoms = [0.0] * 3
    for key, name, color in (("draw_s", "Drawing", BLUE), ("travel_s", "Travel", ORANGE), ("pen_s", "Pen lift and lower", "#f5b84a")):
        vals = [s[key] for s in st]
        a2.bar(labels, vals, bottom=bottoms, color=color, width=0.6, label=name)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, s in enumerate(st):
        a2.text(i, s["total_s"] + 3, f"{s['total_s']:.0f} s", ha="center", fontsize=9)
    a2.set_title("Simulated plot time", fontsize=11)
    a2.set_ylabel("seconds")
    a2.legend(fontsize=8, frameon=False, loc="upper right")
    a2.set_ylim(0, max(bottoms) * 1.18)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig09-ordering.png", facecolor="white")
    plt.close(fig)

    for k, s in numbers.items():
        if k != "gcode_head":
            print(f"{k:8s} paths {s['paths']:4d}  ink {s['draw_mm']:7.0f} mm  travel {s['travel_mm']:6.0f} mm  "
                  f"time {s['total_s']:5.1f} s (draw {s['draw_s']:.0f}, travel {s['travel_s']:.0f}, pen {s['pen_s']:.0f}, "
                  f"{100 * s['pen_share']:.0f}%)  lines {s['gcode_lines']}")


if __name__ == "__main__":
    main()

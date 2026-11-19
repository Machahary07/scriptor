"""Write one text in several handwriting styles, one G-code file each.

    python examples/handwriting_styles.py examples/linear-search.txt -o out/styles

Each style is a Hershey single-stroke font plus the font model's style settings
(slant, width, spacing, messiness) and bias. Writes <style>.gcode, <style>.svg,
styles.png (all pages side by side) and stats.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine.cli import to_svg  # noqa: E402
from engine.models import Style  # noqa: E402
from engine.models import font as font_model  # noqa: E402
from engine.pipeline import generate_strokes, make_gcode  # noqa: E402
from engine.planner import PageSpec  # noqa: E402

STYLES = {
    "neat_cursive": ("scripts", dict(slant=0.15, width=1.0, spacing=1.0, messiness=0.4), 2.0),
    "flowing_cursive": ("cursive", dict(slant=0.25, width=1.05, spacing=0.95, messiness=1.0), 1.0),
    "fast_messy": ("scripts", dict(slant=0.35, width=1.15, spacing=1.1, word_gap=1.2, messiness=1.5), 0.3),
    "casual_print": ("futural", dict(slant=0.08, width=0.95, spacing=1.05, messiness=1.2), 1.0),
    "upright_print": ("rowmans", dict(slant=0.0, width=0.9, spacing=1.0, messiness=0.6), 2.0),
    "bold_script": ("scriptc", dict(slant=0.18, width=1.0, spacing=1.0, messiness=0.7), 1.5),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="out/styles")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    text = Path(a.input).read_text(encoding="utf-8")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = PageSpec()
    fig, axes = plt.subplots(1, len(STYLES), figsize=(4 * len(STYLES), 6))
    stats = {}
    for ax, (name, (font, params, bias)) in zip(axes, STYLES.items()):
        font_model.FONT_NAME = font
        font_model._font.cache_clear()
        page = generate_strokes(text, Style(**params), "font", bias, a.seed, spec)[0]
        res = make_gcode(page.paths, spec.height, f"style={name} font={font}")
        (out / f"{name}.gcode").write_text(res.gcode)
        (out / f"{name}.svg").write_text(to_svg(res.paths, spec, travel=False))
        stats[name] = {"font": font, "bias": bias, **params, **res.stats}
        for p in page.paths:
            ax.plot(p[:, 0], p[:, 1], "-", color="#1d3fa8", lw=0.4)
        ax.set_xlim(0, spec.width)
        ax.set_ylim(spec.height * 0.72, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name}\n{res.stats['total_s'] / 60:.1f} min plot", fontsize=10)
        print(f"{name:16s} font={font:8s} {res.stats['paths']:4d} paths  {res.stats['total_s'] / 60:4.1f} min")
    fig.tight_layout()
    fig.savefig(out / "styles.png", dpi=130)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

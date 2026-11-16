"""Command line: python -m engine.cli input.txt -o out/ [--writer ID] [--model font] [--bias 1] [--seed 0]

Writes page<N>.gcode, page<N>.svg (toolpath preview) and stats.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import store
from .pipeline import check_request, generate_strokes, make_gcode
from .planner import PageSpec


def to_svg(paths, spec: PageSpec, travel: bool = True) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {spec.width} {spec.height}" '
           f'width="{spec.width}mm" height="{spec.height}mm"><rect width="100%" height="100%" fill="white"/>']
    cur = (0.0, 0.0)
    for p in paths:
        if travel:
            out.append(f'<line x1="{cur[0]:.2f}" y1="{cur[1]:.2f}" x2="{p[0][0]:.2f}" y2="{p[0][1]:.2f}" '
                       'stroke="#d33" stroke-width="0.15" stroke-dasharray="0.8 0.6"/>')
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in p)
        out.append(f'<polyline points="{pts}" fill="none" stroke="#1d3fa8" stroke-width="0.35" '
                   'stroke-linecap="round" stroke-linejoin="round"/>')
        cur = (p[-1][0], p[-1][1])
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write text as handwriting G-code.")
    ap.add_argument("input", help="text file, or '-' to read the --text argument")
    ap.add_argument("--text", default="")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--writer", default=None)
    ap.add_argument("--model", default="font")
    ap.add_argument("--bias", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--join", type=float, default=0.3, help="join strokes closer than this (mm); 0 disables")
    ap.add_argument("--no-order", action="store_true", help="keep written order (for comparison)")
    a = ap.parse_args(argv)

    text = a.text if a.input == "-" else Path(a.input).read_text(encoding="utf-8")
    writer = store.get_writer(a.writer) if a.writer else None
    check_request(text, writer["name"] if writer else None)
    style = store.load_style(a.writer)
    spec = PageSpec()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    pages = generate_strokes(text, style, a.model, a.bias, a.seed, spec)
    for i, page in enumerate(pages, 1):
        res = make_gcode(page.paths, spec.height, f"cli page={i}/{len(pages)}", order=not a.no_order, join_mm=a.join)
        (out / f"page{i}.gcode").write_text(res.gcode)
        (out / f"page{i}.svg").write_text(to_svg(res.paths, spec))
        stats.append(res.stats)
        s = res.stats
        print(f"page {i}: {s['paths']} paths, ink {s['draw_mm']:.0f} mm, travel {s['travel_mm']:.0f} mm "
              f"(written order {s['travel_mm_written_order']:.0f} mm), time {s['total_s']:.0f} s, "
              f"pen lifts {100 * s['pen_share']:.0f}% ({s['paths_before_join']} strokes joined into {s['paths']})")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

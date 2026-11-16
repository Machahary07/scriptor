"""Page planning and rendering.

plan_pages() wraps text into lines and places them on A4 pages in millimetres.
A line of the form ``[box 40 Circuit diagram]`` reserves a 40 mm high box for a
diagram; the outline is drawn and the caption is written under it.
render_page() asks the handwriting model for each line and returns the page as
polylines in page millimetres (origin top-left, y down).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from .models import HandwritingModel, Style
from .strokes import Polyline, stroke3_to_polylines

BOX_RE = re.compile(r"^\[box\s+(\d+(?:\.\d+)?)\s*(.*)\]$", re.IGNORECASE)


@dataclass
class PageSpec:
    width: float = 210.0
    height: float = 297.0
    margin_left: float = 20.0
    margin_right: float = 15.0
    margin_top: float = 25.0
    margin_bottom: float = 20.0
    line_spacing: float = 8.0
    x_height: float = 2.6


def _wrap(paragraph: str, max_w: float, measure) -> list[str]:
    lines, cur = [], ""
    for word in paragraph.split():
        trial = f"{cur} {word}" if cur else word
        if cur and measure(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def plan_pages(text: str, model: HandwritingModel, style: Style, spec: PageSpec = PageSpec()) -> list[dict]:
    max_w = spec.width - spec.margin_left - spec.margin_right
    measure = lambda s: model.estimate_width(s, style) * spec.x_height  # noqa: E731
    pages: list[dict] = []
    items: list[dict] = []
    y = spec.margin_top

    def new_page():
        nonlocal items, y
        items = []
        pages.append({"spec": asdict(spec), "items": items})
        y = spec.margin_top

    new_page()
    for para in text.replace("\r\n", "\n").split("\n"):
        para = para.strip()
        m = BOX_RE.match(para)
        if m:
            h, caption = float(m.group(1)), m.group(2).strip()
            need = h + (spec.line_spacing if caption else 0) + spec.line_spacing / 2
            if y + need > spec.height - spec.margin_bottom and items:
                new_page()
            items.append({"kind": "box", "x": spec.margin_left, "y": y, "w": max_w, "h": h})
            y += h + spec.line_spacing
            if caption:
                items.append({"kind": "line", "text": caption, "x": spec.margin_left + 4, "y": y})
                y += spec.line_spacing
            continue
        if not para:
            y += spec.line_spacing  # blank line between paragraphs
            continue
        for line in _wrap(para, max_w, measure):
            if y > spec.height - spec.margin_bottom:
                new_page()
            items.append({"kind": "line", "text": line, "x": spec.margin_left, "y": y})
            y += spec.line_spacing
    return [p for p in pages if p["items"]] or pages[:1]


def render_page(page: dict, model: HandwritingModel, style: Style, bias: float, seed: int) -> list[Polyline]:
    spec = PageSpec(**page["spec"])
    max_x = spec.width - spec.margin_right
    rng = np.random.default_rng(seed)
    out: list[Polyline] = []
    for it in page["items"]:
        if it["kind"] == "box":
            x, y, w, h = it["x"], it["y"], it["w"], it["h"]
            out.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]], dtype=np.float64))
            continue
        s3 = model.write_line(it["text"], style, bias, rng)
        polys = stroke3_to_polylines(s3)
        if not polys:
            continue
        polys = [p * spec.x_height for p in polys]
        x0 = min(p[:, 0].min() for p in polys)
        width = max(p[:, 0].max() for p in polys) - x0
        avail = max_x - it["x"]
        # Neural models do not know the page width; squeeze a line that overruns.
        sx = min(1.0, avail / width) if width > 0 else 1.0
        for p in polys:
            q = p.copy()
            q[:, 0] = (q[:, 0] - x0) * sx + it["x"]
            q[:, 1] += it["y"]
            out.append(q)
    return out

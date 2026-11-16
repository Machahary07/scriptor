"""Text to G-code in two stages: generate_strokes() then make_gcode()."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .gcode import GcodeConfig, write_gcode
from .models import Style, get_model
from .paths import join_paths, order_paths, simplify, toolpath_stats
from .planner import PageSpec, plan_pages, render_page
from .strokes import Polyline

SIGNATURE_WORDS = re.compile(r"\b(signature|signed|sign here|autograph|sd/-)\b", re.IGNORECASE)


class Refused(ValueError):
    """The request was refused on ethical grounds (e.g. it looks like a signature)."""


def check_request(text: str, writer_name: str | None = None) -> None:
    """Refuse text that looks like a signature request.

    Two cases are blocked: text that mentions signing, and very short text that is
    mostly the writer's own name, which is how a signature would be requested.
    """
    t = text.strip()
    if not t:
        raise Refused("text is empty")
    if SIGNATURE_WORDS.search(t):
        raise Refused("signatures are not generated")
    if writer_name:
        words = re.findall(r"[a-z]+", t.lower())
        name = set(re.findall(r"[a-z]+", writer_name.lower()))
        if words and len(words) <= 4 and sum(w in name for w in words) >= max(1, len(words) - 1):
            raise Refused("text looks like the writer's name or signature")


@dataclass
class StrokePage:
    layout: dict
    paths: list[Polyline]  # page mm, in written order


@dataclass
class GcodePage:
    paths: list[Polyline]  # simplified and ordered
    gcode: str
    stats: dict


def generate_strokes(text: str, style: Style, model_name: str = "font", bias: float = 1.0, seed: int = 0,
                     spec: PageSpec = PageSpec()) -> list[StrokePage]:
    model = get_model(model_name)
    layouts = plan_pages(text, model, style, spec)
    return [StrokePage(lay, render_page(lay, model, style, bias, seed + i)) for i, lay in enumerate(layouts)]


def make_gcode(paths: list[Polyline], page_h: float = 297.0, tag: str | None = None, order: bool = True,
               simplify_mm: float = 0.1, join_mm: float = 0.3) -> GcodePage:
    cfg = GcodeConfig(page_h=page_h)
    paths = [np.asarray(p, dtype=np.float64) for p in paths if len(p)]
    n_raw = len(paths)
    if join_mm > 0:
        paths = join_paths(paths, join_mm)
    paths = simplify(paths, simplify_mm)
    # The machine homes at X0 Y0, the bottom-left corner of the sheet; in page
    # coordinates (y down) that is (0, page_h).
    home = (0.0, page_h)
    written = toolpath_stats(paths, cfg.timing, home)
    if order:
        paths = order_paths(paths, home)
    stats = toolpath_stats(paths, cfg.timing, home).as_dict()
    stats["travel_mm_written_order"] = round(written.travel_mm, 3)
    stats["total_s_written_order"] = round(written.total_s, 3)
    stats["paths_before_join"] = n_raw
    stats["points"] = int(sum(len(p) for p in paths))
    gcode = write_gcode(paths, cfg, tag)
    stats["gcode_lines"] = gcode.count("\n")
    return GcodePage(paths, gcode, stats)


def generate(text: str, style: Style, model_name: str = "font", bias: float = 1.0, seed: int = 0,
             spec: PageSpec = PageSpec(), tag: str | None = None) -> list[GcodePage]:
    pages = generate_strokes(text, style, model_name, bias, seed, spec)
    n = len(pages)
    return [make_gcode(p.paths, spec.height, f"{tag} page={i + 1}/{n}" if tag else None) for i, p in enumerate(pages)]

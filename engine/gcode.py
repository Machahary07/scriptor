"""GRBL G-code writer for a CoreXY pen plotter.

Page coordinates have y growing downwards from the top-left corner. The machine
origin is the bottom-left corner of the sheet with y growing upwards, so y is
flipped here. Pen lift is driven by a servo on the spindle PWM output (M3 S<angle>
lowers, M5 raises), which needs a servo-enabled GRBL build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .paths import MachineTiming
from .strokes import Polyline


@dataclass
class GcodeConfig:
    page_h: float = 297.0
    pen_down_angle: int = 90
    timing: MachineTiming = field(default_factory=MachineTiming)
    decimals: int = 2


def _xy(x: float, y: float, cfg: GcodeConfig) -> str:
    d = cfg.decimals
    return f"X{x:.{d}f} Y{cfg.page_h - y:.{d}f}"


def write_gcode(paths: list[Polyline], cfg: GcodeConfig = GcodeConfig(), tag: str | None = None) -> str:
    t = cfg.timing
    out = []
    if tag:
        # Trace identifier: every Scriptor file names its job so a page can be traced.
        out.append(f"; scriptor {tag}")
    out += ["G21 ; mm", "G90 ; absolute", "M5 ; pen up", "G0 X0 Y0"]
    f_draw = int(t.draw_feed)
    f_travel = int(t.travel_feed)
    for p in paths:
        if len(p) == 0:
            continue
        out.append(f"G0 {_xy(*p[0], cfg)} F{f_travel}")
        out.append(f"M3 S{cfg.pen_down_angle} ; pen down")
        out.append(f"G4 P{t.pen_down_s:.2f}")
        pts = p[1:] if len(p) > 1 else p  # a single point is drawn as a dot
        for x, y in pts:
            out.append(f"G1 {_xy(x, y, cfg)} F{f_draw}")
        out.append("M5 ; pen up")
        out.append(f"G4 P{t.pen_up_s:.2f}")
    out.append(f"G0 X0 Y0 F{f_travel}")
    out.append("M2 ; end")
    return "\n".join(out) + "\n"

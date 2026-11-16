"""Single-stroke font generator with natural variation.

This is the prototype generator. It writes each character from a Hershey script
font [ref 14] and then varies it the way a hand does: letter size, slant,
spacing and baseline drift change slightly every time, and every stroke gets a
small smooth wobble. It needs no training, so it is used to build and test the
toolpath and simulator, and as the fallback when no neural weights exist. It
does not learn a person's letter shapes; it only takes slant, width and spacing
from their samples.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from HersheyFonts import HersheyFonts

from ..strokes import polylines_to_stroke3, resample
from .base import HandwritingModel, Style

FONT_NAME = "scripts"


@lru_cache(maxsize=1)
def _font():
    f = HersheyFonts()
    f.load_default_font(FONT_NAME)
    (_, ymin), (_, ymax) = next(iter(f.glyphs_for_text("x"))).draw_box
    base = f.render_options.base_line
    xh = ymax - ymin
    glyphs = {}
    for code in range(32, 127):
        ch = chr(code)
        g = next(iter(f.glyphs_for_text(ch)), None)
        if g is None:
            continue
        strokes = [
            np.array([((x - g.left_offset) / xh, (y - base) / xh) for x, y in s], dtype=np.float64)
            for s in g.strokes
        ]
        glyphs[ch] = (strokes, g.char_width / xh)
    return glyphs


def _wobble(poly: np.ndarray, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Add a smooth low-frequency displacement along the stroke."""
    if len(poly) < 2 or amp <= 0:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out = poly.copy()
    for axis in (0, 1):
        for freq in (1.3, 2.9):
            out[:, axis] += amp / freq * np.sin(freq * 2 * np.pi * s + rng.uniform(0, 2 * np.pi))
    return out


class FontModel(HandwritingModel):
    name = "font"
    label = "Varied single-stroke font"
    description = "Hershey script font with per-letter variation. No training needed; prototype and fallback."

    def write_line(self, text: str, style: Style, bias: float, rng: np.random.Generator) -> np.ndarray:
        glyphs = _font()
        # Bias works as in Graves' model: larger values give neater writing.
        m = style.messiness / (1.0 + max(bias, 0.0))
        polys = []
        x = 0.0
        drift = 0.0
        tilt = rng.normal(0.0, 0.006 * m)
        for ch in text:
            if ch not in glyphs:
                ch = "?"
            strokes, adv = glyphs[ch]
            if ch == " ":
                x += 0.8 * adv * style.word_gap * (1.0 + rng.normal(0, 0.12 * m))
                continue
            scale = 1.0 + rng.normal(0, 0.05 * m)
            slant = style.slant + rng.normal(0, 0.05 * m)
            drift = float(np.clip(drift + rng.normal(0, 0.025 * m), -0.18, 0.18))
            dy = drift + tilt * x
            shear = np.tan(slant)
            for s in strokes:
                p = resample(s, 0.08)
                p = p * np.array([scale * style.width, scale])
                p = _wobble(p, 0.02 * m, rng)
                p[:, 0] += x - p[:, 1] * shear
                p[:, 1] += dy
                polys.append(p)
            x += adv * style.width * style.spacing * scale * (1.0 + rng.normal(0, 0.05 * m))
        return polylines_to_stroke3(polys)

    def estimate_width(self, text: str, style: Style) -> float:
        glyphs = _font()
        w = 0.0
        for ch in text:
            adv = glyphs.get(ch, glyphs["?"])[1]
            w += adv * (0.8 * style.word_gap if ch == " " else style.width * style.spacing)
        return w

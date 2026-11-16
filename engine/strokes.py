"""Stroke representations and conversions.

Two forms are used throughout the engine:

* polylines: a list of (N, 2) float arrays, one per pen-down stroke, in absolute
  coordinates (y grows downwards, as on a page).
* stroke-3: an (N, 3) array of (dx, dy, p) rows [Ha & Eck, ref 10]. dx, dy are
  offsets from the previous point and p is 1 when the pen lifts after the point.
"""

from __future__ import annotations

import numpy as np

Polyline = np.ndarray


def polylines_to_stroke3(polys: list[Polyline]) -> np.ndarray:
    rows = []
    prev = np.zeros(2)
    for poly in polys:
        if len(poly) == 0:
            continue
        for i, pt in enumerate(poly):
            d = pt - prev
            rows.append((d[0], d[1], 1.0 if i == len(poly) - 1 else 0.0))
            prev = pt
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def stroke3_to_polylines(s3: np.ndarray, origin=(0.0, 0.0)) -> list[Polyline]:
    pts = np.cumsum(s3[:, :2], axis=0) + np.asarray(origin, dtype=np.float64)
    polys, start = [], 0
    for i, lift in enumerate(s3[:, 2]):
        if lift > 0.5:
            polys.append(pts[start : i + 1].copy())
            start = i + 1
    if start < len(pts):
        polys.append(pts[start:].copy())
    return [p for p in polys if len(p) > 0]


def bounds(polys: list[Polyline]) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) over all points."""
    allp = np.concatenate([p for p in polys if len(p)], axis=0)
    return (*allp.min(axis=0), *allp.max(axis=0))


def resample(poly: Polyline, spacing: float) -> Polyline:
    """Resample a polyline at even arc-length spacing, keeping both ends."""
    if len(poly) < 2:
        return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return poly[:1].copy()
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def estimate_slant(polys: list[Polyline]) -> float:
    """Dominant slant of near-vertical segments, in radians (positive leans right).

    Segments are weighted by length; only those within 45 degrees of vertical
    count, which is the usual way to measure slant in handwriting analysis.
    """
    num = den = 0.0
    for p in polys:
        if len(p) < 2:
            continue
        d = np.diff(p, axis=0)
        length = np.linalg.norm(d, axis=1)
        steep = np.abs(d[:, 1]) > np.abs(d[:, 0])
        for (dx, dy), w in zip(d[steep], length[steep]):
            if dy > 0:  # orient every segment upwards (y grows down)
                dx, dy = -dx, -dy
            num += np.arctan2(dx, -dy) * w
            den += w
    return float(num / den) if den else 0.0


def normalise_line(polys: list[Polyline]) -> tuple[list[Polyline], float]:
    """Remove slant, rest the line on y=0 and scale so the x-height is 1.

    The x-height is estimated as the spread between the 25th and 75th percentile
    of y values, a robust stand-in for the core zone of lower-case letters.
    Returns the normalised polylines and the scale factor that was applied.
    """
    slant = estimate_slant(polys)
    shear = np.array([[1.0, np.tan(slant)], [0.0, 1.0]])
    polys = [p @ shear.T for p in polys]
    ys = np.concatenate([p[:, 1] for p in polys])
    q25, q75 = np.percentile(ys, [25, 75])
    xh = max(q75 - q25, 1e-6) * 2.0
    base = np.percentile(ys, 85)
    x0 = min(p[:, 0].min() for p in polys)
    out = [np.stack([(p[:, 0] - x0) / xh, (p[:, 1] - base) / xh], axis=1) for p in polys]
    return out, 1.0 / xh

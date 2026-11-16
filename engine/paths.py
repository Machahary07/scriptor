"""Path processing: simplification, ordering and toolpath statistics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .strokes import Polyline


def rdp(poly: Polyline, eps: float) -> Polyline:
    """Ramer-Douglas-Peucker simplification [ref 12], iterative to avoid deep recursion."""
    n = len(poly)
    if n < 3:
        return poly.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = poly[a], poly[b]
        seg = q - p
        pts = poly[a + 1 : b]
        L = np.hypot(*seg)
        if L < 1e-12:
            d = np.linalg.norm(pts - p, axis=1)
        else:
            d = np.abs(seg[0] * (pts[:, 1] - p[1]) - seg[1] * (pts[:, 0] - p[0])) / L
        i = int(np.argmax(d))
        if d[i] > eps:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return poly[keep]


def simplify(paths: list[Polyline], eps: float = 0.1) -> list[Polyline]:
    return [rdp(p, eps) for p in paths if len(p)]


def join_paths(paths: list[Polyline], tol: float = 0.3) -> list[Polyline]:
    """Merge consecutive paths whose gap is under tol mm, so the pen stays down.

    Applied in written order, this joins the strokes of connected letters within a
    word; the short connecting move is drawn and saves a pen lift and a lower.
    """
    out: list[Polyline] = []
    for p in paths:
        gap = np.linalg.norm(out[-1][-1] - p[0]) if out else np.inf
        if gap <= tol:
            out[-1] = np.concatenate([out[-1], p[1:] if gap < 1e-9 else p])
        else:
            out.append(p)
    return out


def order_paths(paths: list[Polyline], start=(0.0, 0.0)) -> list[Polyline]:
    """Greedy nearest-neighbour ordering; a path may be drawn reversed."""
    if not paths:
        return []
    starts = np.array([p[0] for p in paths])
    ends = np.array([p[-1] for p in paths])
    remaining = np.ones(len(paths), dtype=bool)
    cur = np.asarray(start, dtype=np.float64)
    out = []
    for _ in range(len(paths)):
        ds = np.linalg.norm(starts - cur, axis=1)
        de = np.linalg.norm(ends - cur, axis=1)
        ds[~remaining] = np.inf
        de[~remaining] = np.inf
        i_s, i_e = int(np.argmin(ds)), int(np.argmin(de))
        if de[i_e] < ds[i_s]:
            i, p = i_e, paths[i_e][::-1]
        else:
            i, p = i_s, paths[i_s]
        out.append(p)
        remaining[i] = False
        cur = p[-1]
    return out


@dataclass
class MachineTiming:
    draw_feed: float = 3000.0  # mm/min
    travel_feed: float = 5000.0  # mm/min
    pen_down_s: float = 0.15
    pen_up_s: float = 0.10


@dataclass
class ToolpathStats:
    paths: int
    draw_mm: float
    travel_mm: float
    draw_s: float
    travel_s: float
    pen_s: float
    total_s: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["pen_share"] = self.pen_s / self.total_s if self.total_s else 0.0
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


def toolpath_stats(paths: list[Polyline], timing: MachineTiming = MachineTiming(), home=(0.0, 0.0)) -> ToolpathStats:
    """Constant-speed time model; matches the simulator's estimate."""
    draw = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in paths if len(p) > 1)
    travel, cur = 0.0, np.asarray(home, dtype=np.float64)
    for p in paths:
        travel += float(np.linalg.norm(p[0] - cur))
        cur = p[-1]
    travel += float(np.linalg.norm(cur - np.asarray(home)))
    draw_s = draw / timing.draw_feed * 60
    travel_s = travel / timing.travel_feed * 60
    pen_s = len(paths) * (timing.pen_down_s + timing.pen_up_s)
    return ToolpathStats(len(paths), draw, travel, draw_s, travel_s, pen_s, draw_s + travel_s + pen_s)

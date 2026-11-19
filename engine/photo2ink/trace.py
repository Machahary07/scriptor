"""Trace a photographed page into pen paths, without a model.

Clean the page (page.py), drop specks, rulings and the photo border, thin the
ink to one-pixel lines (Zhang-Suen), follow the lines into polylines and
scale them onto A4. The result is the writer's own page redrawn by the
plotter. It reproduces the page; stroke order and direction are not
recovered, which is what InkSight is for.

    python -m engine.photo2ink.trace photo.jpg -o out/traced     # page1.gcode, page1.svg, stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from ..paths import rdp
from .page import clean_page, load_photo

_NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # clockwise from north


def components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components as arrays of (row, col)."""
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        q, pix = deque([(r0, c0)]), []
        while q:
            r, c = q.popleft()
            pix.append((r, c))
            for dr, dc in _NB:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.asarray(pix))
    return out


def clean_ink(ink: np.ndarray, min_px: int = 10, border: int = 6) -> np.ndarray:
    """Keep handwriting: drop specks, blobs touching the photo border and long thin rulings."""
    H, W = ink.shape
    out = np.zeros_like(ink)
    for pix in components(ink):
        if len(pix) < min_px:
            continue
        r0, c0 = pix.min(0)
        r1, c1 = pix.max(0)
        if r0 < border or c0 < border or r1 >= H - border or c1 >= W - border:
            continue
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if (w > 0.4 * W and h < 0.03 * H) or (h > 0.4 * H and w < 0.03 * W):
            continue
        out[pix[:, 0], pix[:, 1]] = True
    return out


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to one-pixel-wide lines."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [np.roll(np.roll(img, -dr, 0), -dc, 1) for dr, dc in _NB]  # P2..P9
            B = sum(P)
            seq = P + [P[0]]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            p2, p4, p6, p8 = P[0], P[2], P[4], P[6]
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    skel = img[1:-1, 1:-1].astype(bool)
    for pix in components(mask):  # thinning can erase a small dot entirely; keep its centre
        if not skel[pix[:, 0], pix[:, 1]].any():
            r, c = pix[np.argmin(((pix - pix.mean(0)) ** 2).sum(1))]
            skel[r, c] = True
    return skel


def trace(skel: np.ndarray) -> list[np.ndarray]:
    """Follow skeleton pixels into polylines (x, y), splitting at junctions."""
    H, W = skel.shape
    pts = set(zip(*np.nonzero(skel)))

    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc) for dr, dc in _NB if (r + dr, c + dc) in pts]

    deg = {p: len(nbrs(p)) for p in pts}
    used = set()  # undirected edges already walked

    def walk(a, b):
        line = [a, b]
        used.add(frozenset((a, b)))
        while deg[line[-1]] == 2:
            nxt = [n for n in nbrs(line[-1]) if frozenset((line[-1], n)) not in used]
            if not nxt:
                break
            used.add(frozenset((line[-1], nxt[0])))
            line.append(nxt[0])
        return line

    lines = []
    for start in sorted(pts, key=lambda p: (deg[p] == 2, p)):  # ends and junctions first, then loops
        for n in nbrs(start):
            if frozenset((start, n)) not in used:
                lines.append(walk(start, n))
    def spur(l):  # a very short branch off a junction is a thinning artefact, not a pen stroke
        a, b = sorted((deg[l[0]], deg[l[-1]]))
        return len(l) < 4 and a == 1 and b >= 3

    lines = chain([l for l in lines if not spur(l)])
    lines += [[p, p] for p in pts if deg[p] == 0]  # dots
    return [np.array([(c, r) for r, c in l], dtype=np.float64) for l in lines]


def chain(lines: list[list[tuple]]) -> list[list[tuple]]:
    """Join segments that meet at a junction, so the pen stays down through it.

    From each end of the current path, continue with the unused segment whose end
    touches it (within one pixel) and that turns the least.
    """
    at: dict[tuple, list[int]] = {}
    for i, l in enumerate(lines):
        for p in (l[0], l[-1]):
            at.setdefault(p, []).append(i)
    used = [False] * len(lines)

    def extend(cur):
        while True:
            tail = cur[-1]
            d0 = np.subtract(tail, cur[max(0, len(cur) - 5)])
            best, best_cos = None, -2.0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for j in at.get((tail[0] + dr, tail[1] + dc), []):
                        if used[j]:
                            continue
                        seg = lines[j] if lines[j][0] == (tail[0] + dr, tail[1] + dc) else lines[j][::-1]
                        d1 = np.subtract(seg[min(4, len(seg) - 1)], seg[0])
                        n = np.linalg.norm(d0) * np.linalg.norm(d1)
                        cos = float(d0 @ d1 / n) if n else 0.0
                        if cos > best_cos:
                            best, best_cos = (j, seg), cos
            if best is None:
                return cur
            used[best[0]] = True
            cur.extend(best[1] if best[1][0] != tail else best[1][1:])

    out = []
    for i in sorted(range(len(lines)), key=lambda i: -len(lines[i])):
        if used[i]:
            continue
        used[i] = True
        cur = extend(list(lines[i]))
        out.append(extend(cur[::-1]))
    return out


def smooth(p: np.ndarray, k: int = 2) -> np.ndarray:
    """Moving average that keeps the ends, to remove pixel staircases."""
    if len(p) < 2 * k + 2:
        return p
    out = p.copy()
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    for d in (0, 1):
        out[k:-k, d] = np.convolve(p[:, d], kernel, mode="valid")
    return out


def page_paths(photo, width_mm: float = 175.0, left: float = 20.0, top: float = 20.0,
               page_h: float = 297.0) -> tuple[list[np.ndarray], dict]:
    page = clean_page(load_photo(photo))
    ink = clean_ink(page.ink)
    skel = thin(ink)
    lines = [rdp(smooth(l), 0.6) for l in trace(skel) if len(l) >= 2]
    allp = np.concatenate(lines)
    x0, y0 = allp.min(0)
    s = width_mm / max(np.ptp(allp[:, 0]), 1)
    s = min(s, (page_h - 2 * top) / max(np.ptp(allp[:, 1]), 1))
    paths = [(l - [x0, y0]) * s + [left, top] for l in lines]
    info = {"photo": Path(photo).name, "ink_px": int(ink.sum()), "skeleton_px": int(skel.sum()),
            "polylines": len(paths), "mm_per_px": round(float(s), 4), "deskew_deg": page.angle}
    return paths, info


def main(argv=None):
    from ..cli import to_svg
    from ..pipeline import make_gcode
    from ..planner import PageSpec

    ap = argparse.ArgumentParser(description="Trace a photographed page into plotter G-code.")
    ap.add_argument("photo")
    ap.add_argument("-o", "--out", default="out/traced")
    ap.add_argument("--width", type=float, default=175.0, help="width of the writing on paper (mm)")
    a = ap.parse_args(argv)
    spec = PageSpec()
    paths, info = page_paths(a.photo, a.width, page_h=spec.height)
    res = make_gcode(paths, spec.height, f"trace {info['photo']}", join_mm=0.4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "page1.gcode").write_text(res.gcode)
    (out / "page1.svg").write_text(to_svg(res.paths, spec))
    (out / "stats.json").write_text(json.dumps({**info, **res.stats}, indent=2))
    s = res.stats
    print(f"{info['photo']}: {s['paths']} paths, ink {s['draw_mm']:.0f} mm, travel {s['travel_mm']:.0f} mm, "
          f"time {s['total_s']:.0f} s -> {out / 'page1.gcode'}")


if __name__ == "__main__":
    main()

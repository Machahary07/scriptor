"""Words on a page: checks, page coordinates and grouping into lines.

Every word has two independent readings, docTR's OCR text (with a confidence)
and the text InkSight read while tracing it. A word is kept when its strokes
cover the ink in its box and its text is trustworthy (OCR is confident, or the
two readings agree). Kept words are grouped into lines, which become training
samples: {text, strokes: [[[x, y, pen], ...], ...]} in page pixels, y down,
pen = 1 on the last point of each stroke.
"""

from __future__ import annotations

import difflib
import string
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OCR_CONF_MIN = 0.6  # docTR confidence that is trusted on its own
AGREE_MIN = 0.6  # similarity of the two readings that counts as agreement
INK_MATCH_MIN = 0.3  # F1 between recovered strokes and the ink mask
LINE_KEEP_MIN = 0.5  # drop a line when fewer than half its words pass


@dataclass
class Word:
    page: int
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in cleaned-page pixels
    ocr_text: str
    ocr_conf: float
    ink_text: str = ""
    strokes: list[np.ndarray] = field(default_factory=list)  # page pixels
    ink_match: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flags

    @property
    def label(self) -> str:
        return self.ocr_text.strip() or self.ink_text.strip()

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def yc(self) -> float:
        return (self.box[1] + self.box[3]) / 2


def expand_box(box, width: int, height: int, frac: float = 0.1) -> tuple[int, int, int, int]:
    """Grow a word box by frac of its size on each side (as the InkSight demo does)."""
    x0, y0, x1, y1 = box
    ex, ey = (x1 - x0) * frac, (y1 - y0) * frac
    return (int(max(0, x0 - ex)), int(max(0, y0 - ey)), int(min(width, x1 + ex)), int(min(height, y1 + ey)))


def to_page(strokes: list[np.ndarray], ratio: float, dx: int, dy: int, x0: float, y0: float) -> list[np.ndarray]:
    """Undo scale_and_pad: 224-canvas pixels -> page pixels."""
    off = np.array([dx, dy], dtype=np.float64)
    org = np.array([x0, y0], dtype=np.float64)
    return [(s - off) / ratio + org for s in strokes]


def _ink_in_box(ink: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    m = ink[y0:y1, x0:x1].copy()
    if m.size:
        # Notebook rulings run across the whole box; they are not part of the word.
        m[m.mean(axis=1) > 0.6, :] = False
        m[:, m.mean(axis=0) > 0.8] = False
    return m


def ink_match(strokes: list[np.ndarray], ink: np.ndarray, box) -> float:
    """F1 between the ink pixels in the box and the pixels the strokes pass over."""
    m = _ink_in_box(ink, box)
    if not strokes or m.sum() == 0:
        return 0.0
    x0, y0 = int(round(box[0])), int(round(box[1]))
    h, w = m.shape
    width = max(2, int(round(h * 0.06)))
    drawn = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(drawn)
    for s in strokes:
        pts = [(float(x - x0), float(y - y0)) for x, y in s]
        if len(pts) == 1:
            pts = pts * 2
        d.line(pts, fill=255, width=width)
    tol = 2 * width + 1  # odd size for MaxFilter
    drawn_np = np.asarray(drawn) > 0
    drawn_near = np.asarray(drawn.filter(ImageFilter.MaxFilter(tol))) > 0
    ink_near = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(tol))) > 0
    recall = (m & drawn_near).sum() / m.sum()
    precision = (drawn_np & ink_near).sum() / max(drawn_np.sum(), 1)
    return float(0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision))


def similarity(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def judge(w: Word) -> Word:
    w.flags = []
    if not w.strokes:
        w.flags.append("no_strokes")
    elif w.ink_match < INK_MATCH_MIN:
        w.flags.append("strokes_miss_ink")
    if not w.label:
        w.flags.append("no_text")
    elif w.ocr_conf < OCR_CONF_MIN and similarity(w.ocr_text, w.ink_text) < AGREE_MIN:
        w.flags.append("unsure_text")
    return w


_PLAIN = set(string.ascii_letters + string.digits + " .,:;'\"!?()-#&/")


def plausible(text: str) -> bool:
    """Reject OCR nonsense: needs two letters or digits and mostly ordinary characters."""
    t = text.strip()
    if sum(c.isalnum() for c in t) < 2:
        return False
    return sum(c in _PLAIN for c in t) / len(t) >= 0.7


def group_lines(words: list[Word], gap_heights: float = 6.0) -> list[list[Word]]:
    """Cluster words of one page into lines by their vertical centre, left to right.

    A word joins a line when its centre lies within 0.6 median word heights of
    the line's centre. A horizontal gap of more than gap_heights word heights
    starts a new line (e.g. a note written beside the main text).
    """
    if not words:
        return []
    med_h = float(np.median([w.height for w in words]))
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.yc):
        if rows and abs(w.yc - np.mean([v.yc for v in rows[-1]])) < 0.6 * med_h:
            rows[-1].append(w)
        else:
            rows.append([w])
    lines = []
    for row in rows:
        row.sort(key=lambda w: w.box[0])
        cur = [row[0]]
        for w in row[1:]:
            if w.box[0] - cur[-1].box[2] > gap_heights * med_h:
                lines.append(cur)
                cur = []
            cur.append(w)
        lines.append(cur)
    return lines


def stroke_rows(strokes: list[np.ndarray]) -> list[list[list[float]]]:
    out = []
    for s in strokes:
        if len(s) == 0:
            continue
        rows = [[round(float(x), 1), round(float(y), 1), 0] for x, y in s]
        rows[-1][2] = 1
        out.append(rows)
    return out


def build_samples(words: list[Word]) -> tuple[list[dict], list[dict]]:
    """Group judged words into lines. Returns (samples kept, record of every line)."""
    samples, records = [], []
    for page in sorted({w.page for w in words}):
        for line in group_lines([w for w in words if w.page == page]):
            kept = [w for w in line if w.ok]
            text = " ".join(w.label for w in kept)
            rec = {"page": page, "words": len(line), "kept_words": len(kept), "text": text,
                   "all_text": " ".join(w.label for w in line)}
            if not kept:
                rec["dropped"] = "no word passed"
            elif len(kept) < LINE_KEEP_MIN * len(line):
                rec["dropped"] = "most words failed checks"
            elif not plausible(text):
                rec["dropped"] = "text looks like OCR noise"
            records.append(rec)
            if "dropped" in rec:
                continue
            strokes = [s for w in kept for s in w.strokes]
            xs = np.concatenate([s[:, 0] for s in strokes])
            ys = np.concatenate([s[:, 1] for s in strokes])
            samples.append({"text": text, "page": page,
                            "box": [round(float(v), 1) for v in (xs.min(), ys.min(), xs.max(), ys.max())],
                            "strokes": stroke_rows(strokes)})
    return samples, records

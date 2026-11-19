"""Page clean-up before OCR and ink recovery.

Order: crop to the paper, deskew, stretch contrast, binarise. Only numpy and
Pillow are used, so the same code runs in the InkSight environment and on a
laptop. The cleaned RGB image goes to OCR and InkSight; the binary ink mask is
used to check how well the recovered strokes cover the ink (layout.ink_match).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps

MAX_SIDE = 2000  # px; larger photos are scaled down first


@dataclass
class Page:
    rgb: Image.Image  # cleaned page
    ink: np.ndarray  # bool (H, W), True where there is ink
    angle: float  # degrees the photo was rotated by to deskew
    crop: tuple[int, int, int, int]  # paper box in the (scaled) photo


def load_photo(path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = MAX_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def _darkness(img: Image.Image) -> np.ndarray:
    # Minimum over channels: blue or black pen is dark in at least one channel,
    # while faint blue or grey rulings stay fairly bright in all of them.
    return np.asarray(img, dtype=np.float32).min(axis=2)


def binarise(img: Image.Image, radius: int = 15, k: float = 0.3) -> np.ndarray:
    """Adaptive threshold: ink is darker than (1 - k) times its local mean."""
    d = Image.fromarray(_darkness(img).astype(np.uint8))
    local = np.asarray(d.filter(ImageFilter.BoxBlur(radius)), dtype=np.float32)
    return np.asarray(d, dtype=np.float32) < local * (1 - k)


def paper_box(img: Image.Image, min_area: float = 0.4) -> tuple[int, int, int, int]:
    """Bounding box of the bright, unsaturated paper; the whole image if unsure."""
    a = np.asarray(img, dtype=np.float32)
    lo, hi = a.min(axis=2), a.max(axis=2)
    paper = (lo > 0.6 * np.percentile(lo, 95)) & (hi - lo < 45)
    rows = np.flatnonzero(paper.mean(axis=1) > 0.5)
    cols = np.flatnonzero(paper.mean(axis=0) > 0.5)
    full = (0, 0, img.width, img.height)
    if len(rows) == 0 or len(cols) == 0:
        return full
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return box if area >= min_area * img.width * img.height else full


def skew_angle(ink: np.ndarray, max_deg: float = 5.0, step: float = 0.25) -> float:
    """Angle (degrees) that makes ruled lines and text rows horizontal.

    Projection-profile method: rotate the ink mask and keep the angle whose
    row sums vary the most, i.e. where rows of ink line up sharply.
    """
    m = Image.fromarray(ink.astype(np.uint8) * 255)
    s = 800 / max(m.size)
    if s < 1:
        m = m.resize((round(m.width * s), round(m.height * s)))
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        rows = np.asarray(m.rotate(float(a), resample=Image.BILINEAR), dtype=np.float32).sum(axis=1)
        score = float(rows.var())
        if score > best_score:
            best, best_score = float(a), score
    return best


def clean_page(photo: Image.Image) -> Page:
    box = paper_box(photo)
    img = photo.crop(box)
    angle = skew_angle(binarise(img))
    if angle:
        fill = tuple(int(v) for v in np.median(np.asarray(img).reshape(-1, 3), axis=0))
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=fill)
    img = ImageOps.autocontrast(img, cutoff=1)
    return Page(img, binarise(img), angle, box)

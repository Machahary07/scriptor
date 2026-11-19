"""Preview of one page: the cleaned photo next to the recovered strokes.

Strokes are coloured by the order they were written (rainbow: first = violet,
last = red). Word boxes are green when the word was kept and red when flagged.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .layout import Word  # noqa: E402


def save_preview(path, rgb, words: list[Word], title: str = "") -> None:
    w, h = rgb.size
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 14 * h / w / 2 + 0.6), dpi=110)
    a.imshow(rgb)
    for wd in words:
        x0, y0, x1, y1 = wd.box
        a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.8,
                              ec="tab:green" if wd.ok else "tab:red"))
    a.set_title("photo (green = kept, red = flagged)")
    strokes = [s for wd in sorted(words, key=lambda v: (v.yc, v.box[0])) for s in wd.strokes if len(s) > 1]
    cmap = plt.get_cmap("rainbow")
    segs, cols = [], []
    for i, s in enumerate(strokes):
        segs.append(s)
        cols.append(cmap(i / max(len(strokes) - 1, 1)))
    b.add_collection(LineCollection(segs, colors=cols, linewidths=0.9))
    b.set_title(f"recovered strokes ({len(strokes)}), coloured by order")
    for ax in (a, b):
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

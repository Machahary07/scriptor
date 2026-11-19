"""Folder of page photos -> writer_samples.json, page previews and a report.

    python -m engine.photo2ink.run --photos /content/photos --out /content/out/photo2ink

Runs in the InkSight environment built by setup_env.sh (TensorFlow + docTR).
Writes, in --out:
    writer_samples.json   one entry per kept line: {text, page, box, strokes}
    words.json            every detected word with its readings, checks and flags
    previews/page_NN.png  photo vs recovered strokes
    cleaned/page_NN.jpg   the page after deskew / crop / contrast
    photo2ink_report.json counts and timings measured in this run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .inksight import FALLBACK_PROMPT, MODEL_ID, InkSight, configure_tf, scale_and_pad
from .layout import AGREE_MIN, INK_MATCH_MIN, LINE_KEEP_MIN, OCR_CONF_MIN, Word, build_samples, expand_box, \
    ink_match, judge, to_page
from .page import clean_page, load_photo
from .preview import save_preview

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIN_BOX_PX = 6  # boxes smaller than this are specks, not words


def _versions() -> dict:
    out = {}
    for name in ("tensorflow", "tensorflow_text", "tf_keras", "doctr", "huggingface_hub"):
        try:
            out[name] = __import__(name).__version__
        except Exception as e:  # report, never guess
            out[name] = f"unavailable ({type(e).__name__})"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args(argv)

    t_start = time.time()
    photos = sorted(p for p in a.photos.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not photos:
        print(f"no photos in {a.photos}", file=sys.stderr)
        return 2
    for d in ("previews", "cleaned"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    timing = {}

    t = time.time()
    pages = []
    for i, p in enumerate(photos):
        pg = clean_page(load_photo(p))
        pg.rgb.save(a.out / "cleaned" / f"page_{i + 1:02d}.jpg", quality=92)
        pages.append(pg)
        print(f"page {i + 1}: {p.name}  deskew {pg.angle:+.2f} deg  size {pg.rgb.size}")
    timing["clean_s"] = round(time.time() - t, 1)

    configure_tf()
    os.environ.setdefault("USE_TF", "1")  # docTR backend
    t = time.time()
    from doctr.models import ocr_predictor

    predictor = ocr_predictor(pretrained=True)
    words: list[Word] = []
    for i, pg in enumerate(pages):
        W, H = pg.rgb.size
        res = predictor([np.asarray(pg.rgb)])
        n0 = len(words)
        for block in res.pages[0].blocks:
            for line in block.lines:
                for wd in line.words:
                    (x0, y0), (x1, y1) = wd.geometry
                    if not wd.value.strip():
                        continue
                    words.append(Word(i, (x0 * W, y0 * H, x1 * W, y1 * H), wd.value, float(wd.confidence)))
        print(f"page {i + 1}: {len(words) - n0} words detected")
    timing["ocr_s"] = round(time.time() - t, 1)
    n_detected = len(words)
    words = [w for w in words if w.box[2] - w.box[0] >= MIN_BOX_PX and w.height >= MIN_BOX_PX]
    n_specks = n_detected - len(words)

    t = time.time()
    model = InkSight()
    timing["inksight_load_s"] = round(time.time() - t, 1)
    crops, info = [], []
    for w in words:
        pg = pages[w.page]
        box = expand_box(w.box, *pg.rgb.size)
        img, ratio, dx, dy = scale_and_pad(pg.rgb.crop(box))
        crops.append(img)
        info.append((ratio, dx, dy, box))
    t = time.time()
    results = model.run(crops, batch_size=a.batch)
    timing["inksight_s"] = round(time.time() - t, 1)

    for w, r, (ratio, dx, dy, box) in zip(words, results, info):
        w.strokes = [s for s in to_page(r["strokes"], ratio, dx, dy, box[0], box[1]) if len(s)]
        w.ink_text = r["text"]
        w.ink_match = ink_match(w.strokes, pages[w.page].ink, box)
        judge(w)

    samples, records = build_samples(words)
    for i, (pg, p) in enumerate(zip(pages, photos)):
        pw = [w for w in words if w.page == i]
        kept_lines = sum(1 for r in records if r["page"] == i and "dropped" not in r)
        save_preview(a.out / "previews" / f"page_{i + 1:02d}.png", pg.rgb, pw,
                     f"{p.name}: {sum(w.ok for w in pw)}/{len(pw)} words kept, {kept_lines} lines kept")

    (a.out / "writer_samples.json").write_text(json.dumps(samples))
    (a.out / "words.json").write_text(json.dumps([
        {"page": w.page + 1, "box": [round(v, 1) for v in w.box], "ocr_text": w.ocr_text,
         "ocr_conf": round(w.ocr_conf, 3), "inksight_text": w.ink_text, "ink_match": round(w.ink_match, 3),
         "strokes": len(w.strokes), "flags": w.flags} for w in words], indent=1))
    try:
        import tensorflow as tf

        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []
    report = {
        "photos": len(photos),
        "photo_files": [p.name for p in photos],
        "words_detected": n_detected,
        "words_skipped_as_specks": n_specks,
        "words_sent_to_inksight": len(words),
        "words_kept": sum(w.ok for w in words),
        "word_flags": dict(Counter(f for w in words for f in w.flags)),
        "fallback_prompt_used": sum(r["prompt"] == FALLBACK_PROMPT for r in results),
        "median_ink_match": round(float(np.median([w.ink_match for w in words])), 3) if words else None,
        "lines_total": len(records),
        "lines_kept": len(samples),
        "lines_dropped": dict(Counter(r["dropped"] for r in records if "dropped" in r)),
        "points_kept": sum(len(s) for x in samples for s in x["strokes"]),
        "thresholds": {"ocr_conf_min": OCR_CONF_MIN, "agree_min": AGREE_MIN, "ink_match_min": INK_MATCH_MIN,
                       "line_keep_min": LINE_KEEP_MIN},
        "inksight_model": MODEL_ID,
        "inksight_load_method": model.load_method,
        "ocr": "docTR ocr_predictor(pretrained=True)",
        "gpus": gpus,
        "versions": _versions(),
        "timing_s": {**timing, "total_s": round(time.time() - t_start, 1)},
    }
    (a.out / "photo2ink_report.json").write_text(json.dumps(report, indent=1))
    (a.out / "lines.json").write_text(json.dumps(records, indent=1))
    print(json.dumps({k: report[k] for k in ("photos", "words_detected", "words_kept", "word_flags",
                                             "lines_total", "lines_kept", "lines_dropped", "timing_s")}, indent=1))
    if not samples:
        print("no line passed the checks; see previews/ and words.json", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

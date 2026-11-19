"""Photo -> ink helpers (everything except the TensorFlow models) and the notes dataset / training loop."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from engine.photo2ink.inksight import detokenize, recognised_text, scale_and_pad, text_to_tokens
from engine.photo2ink.layout import Word, build_samples, group_lines, ink_match, judge, plausible, to_page
from engine.photo2ink.page import clean_page


def test_detokenize_strokes_and_text():
    out = "<extra_id_0> hello <ink_token_450> <ink_token_10> <ink_token_245> <ink_token_11> <ink_token_246> " \
          "<ink_token_450> <ink_token_5> <ink_token_230> <ink_token_7>"
    strokes = detokenize(text_to_tokens(out))
    assert [s.tolist() for s in strokes] == [[[10, 20], [11, 21]], [[5, 5]]]  # trailing x without y is skipped
    assert recognised_text(out) == "hello"
    with pytest.raises(ValueError):
        detokenize([451])


def test_scale_and_pad_maps_back_to_page():
    crop = Image.new("RGB", (100, 40), "white")
    img, ratio, dx, dy = scale_and_pad(crop)
    assert img.size == (224, 224) and dy > 0 and dx == 0
    # The crop's corner (100, 40) lands at (dx + 100 ratio, dy + 40 ratio) on the canvas.
    canvas_pt = np.array([[dx + 100 * ratio, dy + 40 * ratio]])
    (page_pt,) = to_page([canvas_pt], ratio, dx, dy, 300, 50)
    assert np.allclose(page_pt, [[400, 90]], atol=1e-6)


ZIGZAG = np.array([[20.0 + 16 * i, 40.0 if i % 2 else 60.0] for i in range(11)])


def test_ink_match_rewards_strokes_on_the_ink():
    img = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(img).line([tuple(p) for p in ZIGZAG], fill=255, width=3)
    ink = np.asarray(img) > 0
    ink[70, :] = True  # a ruling across the box is ignored
    box = (0, 20, 200, 80)
    assert ink_match([ZIGZAG], ink, box) > 0.8
    assert ink_match([ZIGZAG[:3]], ink, box) < 0.5  # covers only part of the word
    assert ink_match([ZIGZAG + [0, 15]], ink, box) < 0.3


def _word(page, x0, y0, text, conf=0.9, match=0.8):
    w = Word(page, (x0, y0, x0 + 50, y0 + 20), text, conf, ink_text=text)
    w.strokes = [np.array([[x0 + 5.0, y0 + 10.0], [x0 + 45.0, y0 + 12.0]])]
    w.ink_match = match
    return judge(w)


def test_lines_grouped_and_checked():
    words = [_word(0, 120, 102, "world"), _word(0, 10, 100, "hello"), _word(0, 10, 160, "next"),
             _word(0, 70, 160, "x", conf=0.1, match=0.1)]
    assert [[w.ocr_text for w in l] for l in group_lines(words)] == [["hello", "world"], ["next", "x"]]
    assert words[3].flags == ["strokes_miss_ink"]  # its text agrees with InkSight, its strokes do not
    samples, records = build_samples(words)
    assert [s["text"] for s in samples] == ["hello world", "next"]
    assert len(records) == 2 and all("dropped" not in r for r in records)
    stroke = samples[0]["strokes"][0]
    assert [p[2] for p in stroke] == [0, 1]  # pen lifts after the last point


def test_unsure_text_and_noise_are_dropped():
    w = _word(0, 0, 0, "abc", conf=0.2)
    w.ink_text = "xyz"
    assert "unsure_text" in judge(w).flags
    assert not plausible("~~ }{ ") and not plausible("a") and plausible("int n, i, key;")
    samples, records = build_samples([w])
    assert samples == [] and records[0]["dropped"] == "no word passed"


def test_clean_page_deskews():
    img = Image.new("RGB", (800, 600), (235, 235, 230))
    d = ImageDraw.Draw(img)
    for y in range(80, 560, 40):
        d.line([(60, y), (740, y)], fill=(30, 30, 120), width=3)
    page = clean_page(img.rotate(-2, fillcolor=(235, 235, 230)))
    assert abs(page.angle - 2) <= 0.5
    assert page.ink.mean() > 0.01


# ---- dataset and training ------------------------------------------------------

@pytest.fixture
def samples_json(tmp_path):
    from engine.models import Style
    from engine.models.font import FontModel
    from engine.strokes import stroke3_to_polylines

    rng = np.random.default_rng(0)
    texts = ["linear search", "int n, i, key;", "print element", "end of loop", "time taken",
             "return zero", "start clock", "found at"] * 3
    items = []
    for t in texts:
        polys = stroke3_to_polylines(FontModel().write_line(t, Style(), 0.5, rng))
        items.append({"text": t, "strokes": [[[x * 30 + 100, y * 30 + 400, 0] for x, y in p] for p in polys]})
    p = tmp_path / "writer_samples.json"
    p.write_text(json.dumps(items))
    return p


def test_dataset_split_and_augmentation(samples_json, tmp_path):
    from training.notes_dataset import build, load_lines

    stats = build(samples_json, tmp_path / "ds", copies=2)
    assert stats["lines_loaded"] == 24 and stats["val_lines"] == 2 and stats["train_lines"] == 22
    train, val, orig = (load_lines(tmp_path / "ds" / f) for f in ("train.npz", "val.npz", "train_orig.npz"))
    assert len(val) == 2 and len(orig) == 22 and len(train) == stats["train_lines_augmented"] <= 66
    assert train[0].s3.shape[1] == 3 and train[0].writer == "me"
    # augmented copies differ from their originals but keep the text
    assert train[22].text == orig[0].text and not np.allclose(train[22].s3, orig[0].s3)


def test_timeboxed_stages_and_lora(samples_json, tmp_path):
    import torch

    from engine.models.graves_net import GravesNet
    from engine.models.lora import apply_lora
    from training.data import norm_stats
    from training.notes_dataset import build, load_lines
    from training.notes_train import count_params, train_timeboxed

    torch.manual_seed(0)
    build(samples_json, tmp_path / "ds", copies=1)
    train, val = load_lines(tmp_path / "ds" / "train.npz"), load_lines(tmp_path / "ds" / "val.npz")
    norm = norm_stats(train)
    net = GravesNet(hidden=16, n_mix=3, style_dim=8)
    best = []
    a = train_timeboxed(net, train, val, norm, minutes=0.03, lr=1e-3, stage="A", device="cpu", batch=8,
                        eval_every_s=0.5, on_best=lambda v, s: best.append(v), log=lambda *_: None)
    assert a["steps"] > 0 and a["history"][-1]["step"] == a["steps"] and best
    apply_lora(net, r=2, alpha=4)
    trainable, total = count_params(net)
    assert 0 < trainable < total
    b = train_timeboxed(net, train, val, norm, minutes=0.02, lr=5e-4, stage="B", device="cpu", batch=8,
                        log=lambda *_: None, t_offset=a["wall_s"] / 60)
    assert abs(b["initial_val"] - a["final_val"]) < 1e-4  # LoRA starts exactly at the base model
    assert b["history"][0]["t_min"] == pytest.approx(a["wall_s"] / 60, abs=1e-3)


def test_trace_photo_without_a_model():
    from engine.photo2ink.trace import chain, clean_ink, thin, trace

    img = Image.new("L", (300, 120), 0)
    d = ImageDraw.Draw(img)
    d.line([(30, 60), (150, 60)], fill=255, width=5)  # a "T": crossing strokes
    d.line([(90, 30), (90, 100)], fill=255, width=5)
    d.ellipse((200, 55, 205, 60), fill=255)  # a dot
    ink = np.asarray(img) > 0
    ink[2, :] = True  # touches the border: dropped
    clean = clean_ink(ink)
    assert not clean[2].any() and clean[60, 100]
    skel = thin(clean)
    assert skel.sum() < clean.sum() / 3
    lines = trace(skel)
    assert len(lines) <= 4  # the T needs at least two pen paths; plus the dot
    xs = np.concatenate([l[:, 0] for l in lines])
    assert xs.min() < 40 and xs.max() > 195
    assert chain([[(0, 0), (0, 1)], [(0, 1), (0, 2)]]) == [[(0, 2), (0, 1), (0, 0)]] or \
        chain([[(0, 0), (0, 1)], [(0, 1), (0, 2)]]) == [[(0, 0), (0, 1), (0, 2)]]

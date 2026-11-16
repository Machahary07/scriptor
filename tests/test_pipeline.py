import numpy as np
import pytest

from engine.models import Style, get_model
from engine.pipeline import Refused, check_request, generate
from engine.planner import PageSpec, plan_pages
from engine.strokes import stroke3_to_polylines


def test_font_model_varies_each_time():
    m = get_model("font")
    rng = np.random.default_rng(0)
    a = m.write_line("hello", Style(), 1.0, rng)
    b = m.write_line("hello", Style(), 1.0, rng)
    assert a.shape[1] == 3
    assert not np.allclose(a[: min(len(a), len(b))], b[: min(len(a), len(b))])


def test_bias_makes_writing_neater():
    m = get_model("font")
    spread = []
    for bias in (0.0, 4.0):
        rng = np.random.default_rng(5)
        heights = [np.concatenate(stroke3_to_polylines(m.write_line("x", Style(), bias, rng)))[:, 1].min()
                   for _ in range(30)]
        spread.append(np.std(heights))
    assert spread[1] < spread[0]


def test_plan_wraps_inside_margins_and_reserves_boxes():
    spec = PageSpec()
    pages = plan_pages("word " * 400 + "\n[box 30 Fig. 1]\nend", get_model("font"), Style(), spec)
    assert len(pages) >= 1
    kinds = [it["kind"] for p in pages for it in p["items"]]
    assert "box" in kinds
    for p in pages:
        for it in p["items"]:
            assert spec.margin_top <= it["y"] <= spec.height


def test_generated_page_stays_on_sheet():
    pages = generate("The quick brown fox jumps over the lazy dog. " * 30, Style(), "font", seed=2, tag="job=t")
    for page in pages:
        pts = np.concatenate(page.paths)
        assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= 210
        assert pts[:, 1].min() >= 0 and pts[:, 1].max() <= 297
        assert page.stats["travel_mm"] <= page.stats["travel_mm_written_order"]
        assert page.gcode.startswith("; scriptor job=t")


@pytest.mark.parametrize("text", ["Signature", "sign here please", "Jeu Machahary", "J. Machahary"])
def test_signature_requests_are_refused(text):
    with pytest.raises(Refused):
        check_request(text, "Jeu Machahary")


def test_normal_text_mentioning_writer_is_allowed():
    check_request("Jeu Machahary measured the current through the resistor five times.", "Jeu Machahary")

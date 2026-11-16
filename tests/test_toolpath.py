import numpy as np
import pytest

from engine.gcode import GcodeConfig, write_gcode
from engine.paths import MachineTiming, join_paths, order_paths, rdp, toolpath_stats
from engine.strokes import polylines_to_stroke3, stroke3_to_polylines


def test_stroke3_round_trip():
    polys = [np.array([[1.0, 2.0], [3.0, 2.5], [4.0, 1.0]]), np.array([[6.0, 0.0], [7.0, 1.0]])]
    back = stroke3_to_polylines(polylines_to_stroke3(polys))
    assert len(back) == 2
    for a, b in zip(polys, back):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_rdp_removes_collinear_points_and_keeps_corners():
    line = np.array([[0, 0], [1, 0.01], [2, 0], [2, 2]], dtype=float)
    out = rdp(line, 0.1)
    np.testing.assert_allclose(out, [[0, 0], [2, 0], [2, 2]])


def test_order_paths_reverses_when_end_is_closer():
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[50.0, 0.0], [11.0, 0.0]])  # its end is next to a's end
    out = order_paths([b, a])
    np.testing.assert_allclose(out[0], a)
    np.testing.assert_allclose(out[1], b[::-1])


def test_order_paths_reduces_travel():
    rng = np.random.default_rng(1)
    paths = [rng.uniform(0, 200, size=(2, 2)) for _ in range(200)]
    before = toolpath_stats(paths).travel_mm
    after = toolpath_stats(order_paths(paths)).travel_mm
    assert after < 0.5 * before


def test_join_paths_merges_touching_strokes_only():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.1, 0.0], [2.0, 0.0]])
    c = np.array([[5.0, 0.0], [6.0, 0.0]])
    out = join_paths([a, b, c], tol=0.3)
    assert len(out) == 2 and len(out[0]) == 4


def test_plot_time_matches_report_formula():
    # 100 mm drawn, two paths, travel 0 -> (0,0) ... known geometry
    paths = [np.array([[0.0, 0.0], [50.0, 0.0]]), np.array([[50.0, 0.0], [100.0, 0.0]])]
    s = toolpath_stats(paths, MachineTiming())
    assert s.draw_mm == pytest.approx(100)
    assert s.travel_mm == pytest.approx(100)  # return home from x=100
    assert s.total_s == pytest.approx(100 / 3000 * 60 + 100 / 5000 * 60 + 2 * 0.25)


def test_gcode_structure_and_y_flip():
    g = write_gcode([np.array([[10.0, 20.0], [30.0, 20.0]])], GcodeConfig(page_h=297), tag="job=t")
    lines = g.strip().splitlines()
    assert lines[0] == "; scriptor job=t"
    assert "G0 X10.00 Y277.00 F5000" in lines
    assert "G1 X30.00 Y277.00 F3000" in lines
    i_down = next(i for i, l in enumerate(lines) if l.startswith("M3"))
    i_draw = next(i for i, l in enumerate(lines) if l.startswith("G1"))
    i_up = max(i for i, l in enumerate(lines) if l.startswith("M5"))
    assert i_down < i_draw < i_up
    assert lines[-2].startswith("G0 X0 Y0")

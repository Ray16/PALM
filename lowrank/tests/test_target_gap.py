"""Unit tests for the target-gap inversion (no GPU / no dataset needed).

These exercise the calibrator's math directly (predict / invert / range / clamp /
degenerate slope). The end-to-end calibration against real gaps is validated in
``experiments/target_gap.py``.

    python -m PALM.lowrank.tests.test_target_gap
"""

from __future__ import annotations

from PALM.lowrank.target_gap import GapCalibrator, invert_to_alpha


def _cal(b, a):
    return GapCalibrator(b=b, a=a, probe_alphas=[0.0, 0.5, 1.0])


def test_predict_endpoints():
    c = _cal(b=0.22, a=0.50)                       # esol-like: gap ≈ 0.22 + 0.50·α
    assert abs(c.predict(0.0) - 0.22) < 1e-9
    assert abs(c.predict(1.0) - 0.72) < 1e-9
    assert abs(c.predict(0.5) - 0.47) < 1e-9
    assert c.achievable_range == (0.22, 0.72)
    assert c.controllable


def test_invert_roundtrip_in_range():
    c = _cal(b=0.22, a=0.50)
    for target in (0.30, 0.47, 0.60):
        inv = c.invert(target)
        assert inv.in_range and inv.controllable
        assert 0.0 <= inv.alpha <= 1.0
        # predict(invert(target)) round-trips to target
        assert abs(c.predict(inv.alpha) - target) < 1e-6
        assert abs(inv.predicted_gap - target) < 1e-6


def test_invert_clamps_above_range():
    c = _cal(b=0.22, a=0.50)                       # max reachable = 0.72
    inv = c.invert(0.95)
    assert not inv.in_range
    assert inv.alpha == 1.0                        # clamped to hardest
    assert abs(inv.predicted_gap - 0.72) < 1e-9
    assert "OUTSIDE" in inv.message


def test_invert_clamps_below_range():
    c = _cal(b=0.22, a=0.50)                       # min reachable = 0.22
    inv = c.invert(0.05)
    assert not inv.in_range
    assert inv.alpha == 0.0                        # clamped to easiest
    assert abs(inv.predicted_gap - 0.22) < 1e-9


def test_negative_slope_range_and_invert():
    # a<0 must still work: range is [b+a, b], invert respects direction.
    c = _cal(b=0.80, a=-0.40)                      # gap: 0.80 (α0) -> 0.40 (α1)
    assert c.achievable_range == (0.40, 0.80)
    inv = c.invert(0.60)
    assert inv.in_range
    assert abs(c.predict(inv.alpha) - 0.60) < 1e-6
    assert 0.0 <= inv.alpha <= 1.0


def test_degenerate_slope_not_controllable():
    c = _cal(b=0.30, a=0.0)                        # flat: dial can't steer the gap
    assert not c.controllable
    inv = c.invert(0.50)
    assert not inv.controllable
    assert inv.alpha == 1.0                        # default to hardest
    assert "not controllable" in inv.message


def test_invert_to_alpha_helper():
    c = _cal(b=0.22, a=0.50)
    assert abs(invert_to_alpha(c, 0.47) - 0.5) < 1e-6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

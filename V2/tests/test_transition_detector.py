"""
tests/test_transition_detector.py
Unit tests for TransitionDetector temporal fall detection.
Run: pytest tests/test_transition_detector.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_detector():
    from transition_detector import TransitionDetector
    return TransitionDetector(
        window_size=15,
        min_fall_frames=3,
        walk_stability_frames=5,
    )


def test_stable_walk_returns_none():
    """15 frames of pure Walk -> no alert."""
    det = make_detector()
    for _ in range(15):
        level, conf = det.update(pred_class=2, confidence=0.85, geo_fall=False)
    assert level == "none"


def test_confirmed_fall_after_transition():
    """Walk for 10 frames then Fall for 5 frames -> confirmed."""
    det = make_detector()
    for _ in range(10):
        det.update(pred_class=2, confidence=0.85, geo_fall=False)
    for _ in range(5):
        level, conf = det.update(pred_class=0, confidence=0.75, geo_fall=False)
    assert level == "confirmed"


def test_possible_fall_on_first_fall_after_walk():
    """Walk for 10 frames then 1 Fall frame -> not yet confirmed."""
    det = make_detector()
    for _ in range(10):
        det.update(pred_class=2, confidence=0.85, geo_fall=False)
    level, conf = det.update(pred_class=0, confidence=0.72, geo_fall=False)
    assert level in ("possible", "none")


def test_geo_fall_fires_immediately():
    """Geometric rule fires Fall immediately regardless of class history."""
    det = make_detector()
    level, conf = det.update(pred_class=2, confidence=0.60, geo_fall=True)
    assert level in ("possible", "confirmed")


def test_no_fall_without_prior_walk():
    """Fall frames without prior Walk stability -> no alert."""
    det = make_detector()
    det.update(pred_class=2, confidence=0.80, geo_fall=False)
    det.update(pred_class=2, confidence=0.80, geo_fall=False)
    level, conf = det.update(pred_class=0, confidence=0.75, geo_fall=False)
    assert level == "none"


def test_reset_clears_state():
    """After reset, detector behaves as fresh."""
    det = make_detector()
    for _ in range(10):
        det.update(pred_class=2, confidence=0.85, geo_fall=False)
    for _ in range(5):
        det.update(pred_class=0, confidence=0.75, geo_fall=False)
    det.reset()
    for _ in range(5):
        level, conf = det.update(pred_class=2, confidence=0.85, geo_fall=False)
    assert level == "none"


def test_confidence_returned_is_fall_confidence():
    """When alert fires, confidence returned is from Fall frames."""
    det = make_detector()
    for _ in range(10):
        det.update(pred_class=2, confidence=0.85, geo_fall=False)
    fall_conf = 0.73
    for _ in range(5):
        level, conf = det.update(pred_class=0, confidence=fall_conf, geo_fall=False)
    assert abs(conf - fall_conf) < 0.01

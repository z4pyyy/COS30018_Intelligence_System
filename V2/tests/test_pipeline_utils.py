"""
tests/test_pipeline_utils.py
Unit tests for pipeline_utils shared utilities.
Run: pytest tests/test_pipeline_utils.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_pad_bbox_tall_box_gets_large_padding():
    """Tall box (standing person, aspect < 1.0) gets 25% padding."""
    from pipeline_utils import pad_bbox
    # 100x200 box = aspect 0.5 = tall = walking person
    x1, y1, x2, y2 = pad_bbox(100, 100, 200, 300, frame_w=640, frame_h=480)
    # Width was 100, 25% pad = 25 each side
    assert x1 == 75
    assert x2 == 225
    # Height was 200, 25% pad = 50 each side
    assert y1 == 50
    assert y2 == 350


def test_pad_bbox_wide_box_gets_small_padding():
    """Wide box (fallen person, aspect > 1.5) gets 10% padding."""
    from pipeline_utils import pad_bbox
    # 300x100 box = aspect 3.0 = wide = prone person
    x1, y1, x2, y2 = pad_bbox(100, 200, 400, 300, frame_w=640, frame_h=480)
    # Width was 300, 10% pad = 30 each side
    assert x1 == 70
    assert x2 == 430
    # Height was 100, 10% pad = 10 each side
    assert y1 == 190
    assert y2 == 310


def test_pad_bbox_clamps_to_frame_boundaries():
    """Padding never goes outside frame dimensions."""
    from pipeline_utils import pad_bbox
    x1, y1, x2, y2 = pad_bbox(5, 5, 50, 150, frame_w=640, frame_h=480)
    assert x1 >= 0
    assert y1 >= 0


def test_pad_bbox_clamps_to_frame_right_bottom():
    """Padding never exceeds frame width/height."""
    from pipeline_utils import pad_bbox
    x1, y1, x2, y2 = pad_bbox(600, 400, 638, 478, frame_w=640, frame_h=480)
    assert x2 <= 640
    assert y2 <= 480


def test_geometric_fall_rule_fires_on_wide_low_confidence():
    """Wide bbox + low keypoint visibility -> returns Fall."""
    from pipeline_utils import GeometricFallRule
    rule = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)
    result = rule.check(
        bbox=(50, 200, 500, 350),      # wide: 450x150 = aspect 3.0
        keypoint_visibilities=[0.2, 0.3, 0.1, 0.4]
    )
    assert result is True


def test_geometric_fall_rule_does_not_fire_on_tall_box():
    """Tall bbox (standing person) -> does not fire even with low confidence."""
    from pipeline_utils import GeometricFallRule
    rule = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)
    result = rule.check(
        bbox=(200, 50, 300, 400),      # tall: 100x350 = aspect 0.29
        keypoint_visibilities=[0.2, 0.3, 0.1, 0.4]
    )
    assert result is False


def test_geometric_fall_rule_does_not_fire_on_high_confidence():
    """Wide bbox but high keypoint visibility -> trust MLP."""
    from pipeline_utils import GeometricFallRule
    rule = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)
    result = rule.check(
        bbox=(50, 200, 500, 350),
        keypoint_visibilities=[0.8, 0.9, 0.85, 0.7]
    )
    assert result is False

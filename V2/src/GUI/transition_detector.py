"""
src/transition_detector.py
==========================
TransitionDetector: temporal fall detection based on Walk->Fall transition pattern.

Replaces the simple N-consecutive-frames TemporalSmoother.

Key difference:
  - Simple: requires N consecutive Fall frames
  - TransitionDetector: requires prior Walk stability + Fall signal onset
  - Also accepts geometric fall rule override

Class IDs (must match model output):
  0 = Fall, 1 = Sit, 2 = Walk
"""

import time
from collections import deque

CLASS_FALL = 0
CLASS_SIT  = 1
CLASS_WALK = 2


class TransitionDetector:
    """
    Detects Walk->Fall transition pattern in a rolling window.

    Alert levels:
        "none"      - no fall transition detected
        "possible"  - transition pattern detected, confidence < confirmed threshold
        "confirmed" - transition pattern + high confidence
    """

    def __init__(
        self,
        window_size: int = 15,
        min_fall_frames: int = 3,
        walk_stability_frames: int = 5,
        alert_hold_sec: float = 3.0,
        confirmed_conf_thresh: float = 0.70,
    ):
        self.window_size            = window_size
        self.min_fall_frames        = min_fall_frames
        self.walk_stability_frames  = walk_stability_frames
        self.alert_hold_sec         = alert_hold_sec
        self.confirmed_conf_thresh  = confirmed_conf_thresh

        self._window: deque = deque(maxlen=window_size)
        self._last_alert_t: float = -999.0
        self._last_fall_conf: float = 0.0

    def reset(self):
        self._window.clear()
        self._last_alert_t  = -999.0
        self._last_fall_conf = 0.0

    def update(
        self,
        pred_class: int,
        confidence: float,
        geo_fall: bool = False,
    ) -> tuple:
        self._window.append((pred_class, confidence))

        if geo_fall:
            self._last_alert_t   = time.time()
            self._last_fall_conf = max(confidence, 0.60)
            level = "confirmed" if self._last_fall_conf >= self.confirmed_conf_thresh else "possible"
            return level, self._last_fall_conf

        window = list(self._window)
        n      = len(window)

        if n < self.walk_stability_frames + self.min_fall_frames:
            return self._check_hold()

        early_window = window[:self.walk_stability_frames]
        walk_count   = sum(1 for cls, _ in early_window if cls in (CLASS_WALK, CLASS_SIT))

        if walk_count < self.walk_stability_frames:
            return self._check_hold()

        recent_window = window[-self.min_fall_frames:]
        fall_frames   = [(cls, conf) for cls, conf in recent_window if cls == CLASS_FALL]

        if len(fall_frames) < self.min_fall_frames:
            return self._check_hold()

        avg_fall_conf        = sum(c for _, c in fall_frames) / len(fall_frames)
        self._last_fall_conf = avg_fall_conf
        self._last_alert_t   = time.time()

        if avg_fall_conf >= self.confirmed_conf_thresh:
            return "confirmed", avg_fall_conf
        else:
            return "possible", avg_fall_conf

    def _check_hold(self) -> tuple:
        elapsed = time.time() - self._last_alert_t
        if elapsed < self.alert_hold_sec and self._last_fall_conf > 0:
            level = ("confirmed" if self._last_fall_conf >= self.confirmed_conf_thresh
                     else "possible")
            return level, self._last_fall_conf
        return "none", 0.0

    @property
    def consecutive_fall_frames(self) -> int:
        count = 0
        for cls, _ in reversed(self._window):
            if cls == CLASS_FALL:
                count += 1
            else:
                break
        return count

    @property
    def walk_stability_count(self) -> int:
        window = list(self._window)
        if len(window) < self.walk_stability_frames:
            return len([c for c, _ in window if c in (CLASS_WALK, CLASS_SIT)])
        early = window[:self.walk_stability_frames]
        return sum(1 for c, _ in early if c in (CLASS_WALK, CLASS_SIT))

    def get_transition_state(self) -> str:
        window = list(self._window)
        if len(window) < 5:
            return "stable"
        recent = [conf for cls, conf in window[-5:]]
        rising  = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        falling = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
        avg = sum(recent) / len(recent)
        if rising  and avg > 0.30:
            return "rising"
        if falling and avg > 0.30:
            return "recovering"
        return "stable"

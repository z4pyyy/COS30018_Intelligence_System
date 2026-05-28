"""
src/pipeline_utils.py
=====================
Shared constants, utilities, and inference helpers for the two-step
fall detection pipeline. All inference files import from here.

CRITICAL: BBOX_PADDING_TALL and BBOX_PADDING_WIDE must match the values
used during feature extraction (extract_pose_features.py). If you change
them here, re-extract features and retrain the MLP.
"""

from pathlib import Path
import pickle
import numpy as np

# =============================================================================
#  PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
#  MODEL PATHS (resolved to actual locations)
# =============================================================================

ONESTEP_MODEL_PATH  = PROJECT_ROOT / "src" / "models" / "techcare.pt"
STAGE1_MODEL_PATH   = PROJECT_ROOT / "V2" / "models" / "stage1_person.pt"
STAGE2_MODEL_PATH   = PROJECT_ROOT / "V2" / "models" / "stage2_pose_mlp_v2.pt"
STAGE2_SCALER_PATH  = PROJECT_ROOT / "V2" / "models" / "pose_scaler_v2.pkl"

# Fallback: try src/models for one-step
if not ONESTEP_MODEL_PATH.exists():
    _alt = PROJECT_ROOT / "V2" / "models" / "swook_techcare_fall_sit_walk.pt"
    if _alt.exists():
        ONESTEP_MODEL_PATH = _alt

# =============================================================================
#  INFERENCE THRESHOLDS
# =============================================================================

STAGE1_CONF     = 0.15
ONESTEP_CONF    = 0.35
IMG_SIZE        = 640
DEVICE          = 0

# =============================================================================
#  ALERT THRESHOLDS
# =============================================================================

CONFIRMED_THRESH = 0.70
POSSIBLE_THRESH  = 0.35

# =============================================================================
#  ADAPTIVE BBOX PADDING
# =============================================================================

ASPECT_THRESHOLD  = 1.2
BBOX_PADDING_TALL = 0.25
BBOX_PADDING_WIDE = 0.10


def pad_bbox(x1: int, y1: int, x2: int, y2: int,
             frame_w: int, frame_h: int) -> tuple:
    bw = x2 - x1
    bh = y2 - y1
    if bh == 0:
        return x1, y1, x2, y2

    aspect = bw / bh
    padding = BBOX_PADDING_TALL if aspect < ASPECT_THRESHOLD else BBOX_PADDING_WIDE

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0,       x1 - pad_x)
    y1 = max(0,       y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    return x1, y1, x2, y2


# =============================================================================
#  GEOMETRIC FALL RULE
# =============================================================================

class GeometricFallRule:
    """
    Fires Fall directly from bbox geometry when MediaPipe keypoint
    confidence is too low to trust the MLP.

    Wide bbox (aspect >= threshold) + low visibility = prone person.
    """

    def __init__(self,
                 aspect_threshold: float = 2.2,
                 visibility_threshold: float = 0.5):
        self.aspect_threshold     = aspect_threshold
        self.visibility_threshold = visibility_threshold

    def check(self, bbox: tuple, keypoint_visibilities: list) -> bool:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bh == 0:
            return False

        aspect = bw / bh
        if aspect < self.aspect_threshold:
            return False

        avg_visibility = (sum(keypoint_visibilities) / len(keypoint_visibilities)
                          if keypoint_visibilities else 0.0)
        return avg_visibility < self.visibility_threshold


# =============================================================================
#  CLASS MAPPINGS
# =============================================================================

CLASS_NAMES = {0: "Fall", 1: "Sit", 2: "Walk"}
FOLDER_TO_CLASS = {"fall": 0, "sit": 1, "walk": 2}

# =============================================================================
#  MEDIAPIPE POSE FEATURE EXTRACTION
# =============================================================================

FEATURE_COLS = [
    "spine_angle", "hip_height_norm", "shoulder_width_norm",
    "knee_bend_left", "knee_bend_right", "body_aspect_ratio",
    "head_height_norm", "ankle_hip_vert_dist",
]


def _angle_between(p1, p2, p3) -> float:
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def extract_pose_features(landmarks) -> tuple:
    """
    Extract 8 geometric features + 4 core visibility scores.

    Accepts either:
      - mp.solutions.pose result.pose_landmarks (has .landmark attribute)
      - mediapipe Tasks API landmark list (direct list of landmarks)

    Returns:
        (features: list[float], core_visibilities: list[float])
        Returns (None, None) if landmarks is None.
    """
    if landmarks is None:
        return None, None

    # Handle both APIs
    lm = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks

    l_sh = lm[11]; r_sh = lm[12]
    l_hi = lm[23]; r_hi = lm[24]
    l_kn = lm[25]; r_kn = lm[26]
    l_an = lm[27]; r_an = lm[28]
    nose  = lm[0]

    msx = (l_sh.x + r_sh.x) / 2;  msy = (l_sh.y + r_sh.y) / 2
    mhx = (l_hi.x + r_hi.x) / 2;  mhy = (l_hi.y + r_hi.y) / 2

    spine_angle = float(np.degrees(np.arctan2(abs(msx - mhx), abs(msy - mhy) + 1e-9)))

    hip_h = (l_hi.y + r_hi.y) / 2
    bh    = abs(msy - (l_an.y + r_an.y) / 2) + 1e-9
    sw    = abs(l_sh.x - r_sh.x) / bh

    kbl = _angle_between(l_hi, l_kn, l_an)
    kbr = _angle_between(r_hi, r_kn, r_an)

    xs  = [lm[i].x for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    ys  = [lm[i].y for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    bar = (max(xs) - min(xs)) / (max(ys) - min(ys) + 1e-9)

    ahv = abs((l_an.y + r_an.y) / 2 - hip_h) / (bh + 1e-9)

    features = [spine_angle, hip_h, sw, kbl, kbr, bar, nose.y, ahv]
    core_vis = [l_sh.visibility, r_sh.visibility, l_hi.visibility, r_hi.visibility]

    return features, core_vis


# =============================================================================
#  MODEL LOADER
# =============================================================================

def load_models(two_step: bool = True) -> dict:
    from ultralytics import YOLO

    models = {}

    if ONESTEP_MODEL_PATH.exists():
        models['onestep'] = YOLO(str(ONESTEP_MODEL_PATH))
        print(f"  [OK] One-step model: {ONESTEP_MODEL_PATH.name}")
    else:
        print(f"  [WARN] One-step model not found: {ONESTEP_MODEL_PATH}")

    if two_step:
        if STAGE1_MODEL_PATH.exists():
            models['stage1'] = YOLO(str(STAGE1_MODEL_PATH))
            print(f"  [OK] Stage 1 model : {STAGE1_MODEL_PATH.name}")
        else:
            print(f"  [WARN] Stage 1 model not found: {STAGE1_MODEL_PATH}")

        if STAGE2_MODEL_PATH.exists() and STAGE2_SCALER_PATH.exists():
            import torch
            import sys as _sys
            _sys.path.insert(0, str(PROJECT_ROOT / "V2" / "src"))
            from train_stage2_pose import PoseMLP
            mlp = PoseMLP()
            mlp.load_state_dict(torch.load(
                str(STAGE2_MODEL_PATH), map_location="cpu", weights_only=True
            ))
            mlp.eval()
            models['mlp'] = mlp

            with open(STAGE2_SCALER_PATH, "rb") as f:
                models['scaler'] = pickle.load(f)
            print(f"  [OK] Stage 2 MLP   : {STAGE2_MODEL_PATH.name}")
        else:
            print(f"  [WARN] Stage 2 model or scaler not found")

    return models

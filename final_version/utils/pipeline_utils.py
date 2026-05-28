"""
pipeline_utils.py  (V3)
========================
Shared constants, utilities, and CNN inference helpers for the
V3 two-step fall detection pipeline.

KEY CHANGE FROM V2:
  Stage 2 is now EfficientNet-B0 CNN operating directly on person
  crop pixels. MediaPipe and MLP are completely removed from Stage 2.
  The CNN sees the actual image — texture, body shape, floor contact,
  orientation — rather than 8-11 geometric scalar features.

  This resolves the fundamental limitation of the V2 MLP:
  static geometric features are ambiguous between Fall/Sit/Walk
  (44.2% Fall accuracy with perfect boxes using V2 MLP).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# =============================================================================
#  PATHS
# =============================================================================

FINAL_BASE         = Path(__file__).resolve().parent.parent
MODELS_DIR         = FINAL_BASE / "models"

ONESTEP_MODEL_PATH = MODELS_DIR / "swook_techcare_fall_sit_walk.pt"
STAGE1_MODEL_PATH  = MODELS_DIR / "stage1_person.pt"
STAGE2_CNN_PATH    = MODELS_DIR / "stage2_cnn.pt"
STAGE2_INFO_PATH   = MODELS_DIR / "stage2_cnn_info.json"

# =============================================================================
#  CLASS NAMES
# =============================================================================

CLASS_NAMES    = ["Fall", "Sit", "Walk"]
FOLDER_TO_CLASS = {"fall": 0, "sit": 1, "walk": 2}

# =============================================================================
#  INFERENCE THRESHOLDS
# =============================================================================

STAGE1_CONF      = 0.15    # low threshold — maximise person recall
ONESTEP_CONF     = 0.35
CONFIRMED_THRESH = 0.70
POSSIBLE_THRESH  = 0.35

CNN_CLASS_THRESH = {
    "Fall": 0.80,
    "Sit":  0.60,
    "Walk": 0.60,
}
IMG_SIZE         = 224     # EfficientNet-B0 input size
DEVICE           = 0       # GPU 0

# =============================================================================
#  ADAPTIVE BBOX PADDING
# =============================================================================

ASPECT_THRESHOLD  = 1.2
BBOX_PADDING_TALL = 0.25   # upright person (walking, standing)
BBOX_PADDING_WIDE = 0.10   # horizontal person (fallen, sitting on floor)


def pad_bbox(x1: int, y1: int, x2: int, y2: int,
             frame_w: int, frame_h: int) -> tuple:
    """
    Adaptive bbox padding based on aspect ratio.
    Tall boxes (aspect < 1.2) → 25% padding.
    Wide boxes (aspect >= 1.2) → 10% padding.
    Clamps to frame boundaries.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bh == 0:
        return x1, y1, x2, y2

    aspect  = bw / bh
    padding = BBOX_PADDING_TALL if aspect < ASPECT_THRESHOLD else BBOX_PADDING_WIDE

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0,       x1 - pad_x)
    y1 = max(0,       y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    return x1, y1, x2, y2

# =============================================================================
#  GEOMETRIC FALL RULE (kept as backstop — no MediaPipe needed)
# =============================================================================

class GeometricFallRule:
    """
    Fires Fall directly from bounding box geometry alone.
    Wide bbox (aspect >= threshold) = horizontal body.
    Used as a fast backstop when CNN confidence is low.

    In V3, this fires based on bbox aspect ratio only — no MediaPipe
    keypoint visibility check since MediaPipe is no longer in the pipeline.
    """

    def __init__(self, aspect_threshold: float = 1.8):
        self.aspect_threshold = aspect_threshold

    def check(self, bbox: tuple) -> bool:
        """
        Args:
            bbox: (x1, y1, x2, y2) in pixels
        Returns:
            True if bbox is wide enough to indicate prone/fallen person
        """
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bh == 0:
            return False
        aspect = bw / bh
        return aspect >= self.aspect_threshold

# =============================================================================
#  CNN INFERENCE
# =============================================================================

# ImageNet normalisation — must match training
_CNN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def build_efficientnet(n_classes: int = 3) -> nn.Module:
    """Build EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, n_classes),
    )
    return model


def load_cnn(model_path=None, info_path=None) -> tuple:
    """
    Load trained EfficientNet-B0 CNN classifier.

    Returns:
        (model, class_names, transform)
        model is on CUDA if available, eval mode.
    """
    model_path = Path(model_path or STAGE2_CNN_PATH)
    info_path  = Path(info_path  or STAGE2_INFO_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model not found: {model_path}\n"
            f"Run train_stage2_cnn.py first."
        )

    # Load class info
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        class_names = info.get("class_names", CLASS_NAMES)
    else:
        class_names = CLASS_NAMES
        print(f"  [WARN] Info file not found, using default class names: {CLASS_NAMES}")

    # Build and load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = build_efficientnet(n_classes=len(class_names))
    state  = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model  = model.to(device)
    model.eval()

    print(f"  [OK] CNN loaded: {model_path.name}  "
          f"classes={class_names}  device={device}")

    return model, class_names, _CNN_TRANSFORM, device


def predict_crop(model, class_names, transform, device,
                 crop_bgr: np.ndarray) -> tuple:
    """
    Classify a person crop using the CNN.

    Args:
        model:       loaded EfficientNet model (eval mode)
        class_names: list of class name strings
        transform:   torchvision transform
        device:      torch device
        crop_bgr:    BGR crop from OpenCV (H x W x 3)

    Returns:
        (class_name: str, confidence: float, class_id: int)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "NoDetection", 0.0, -1

    # BGR → RGB for torchvision
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor   = transform(crop_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

        sorted_ids = torch.argsort(probs, descending=True)
        for idx in sorted_ids:
            idx      = int(idx)
            name     = class_names[idx]
            conf_val = float(probs[idx])
            thresh   = CNN_CLASS_THRESH.get(name, 0.70)
            if conf_val >= thresh:
                return name, conf_val, idx

        best_id  = int(sorted_ids[0])
        return class_names[best_id], float(probs[best_id]), best_id

# =============================================================================
#  MODEL LOADER
# =============================================================================

def load_models(two_step: bool = True) -> dict:
    """
    Load all required models.

    Returns dict with keys:
        'onestep'      — YOLO one-step model
        'stage1'       — YOLO person detector
        'cnn'          — EfficientNet CNN classifier
        'class_names'  — list of class name strings
        'transform'    — torchvision transform for CNN
        'cnn_device'   — torch device CNN is on
    """
    from ultralytics import YOLO

    out = {}

    if ONESTEP_MODEL_PATH.exists():
        out["onestep"] = YOLO(str(ONESTEP_MODEL_PATH))
        print(f"  [OK] One-step : {ONESTEP_MODEL_PATH.name}")
    else:
        print(f"  [WARN] One-step model not found: {ONESTEP_MODEL_PATH}")

    if two_step:
        if STAGE1_MODEL_PATH.exists():
            out["stage1"] = YOLO(str(STAGE1_MODEL_PATH))
            print(f"  [OK] Stage 1  : {STAGE1_MODEL_PATH.name}")
        else:
            print(f"  [WARN] Stage 1 not found: {STAGE1_MODEL_PATH}")

        cnn, class_names, transform, cnn_device = load_cnn()
        out["cnn"]         = cnn
        out["class_names"] = class_names
        out["transform"]   = transform
        out["cnn_device"]  = cnn_device

    return out
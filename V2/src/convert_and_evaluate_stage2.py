"""
convert_and_evaluate_stage2.py
================================
Two-in-one script:

STEP 1 — Convert custom annotations to YOLO normalised format.
  Input:  annotations in "class_name x1 y1 x2 y2" format (absolute pixels)
  Output: YOLO format "class_id cx cy bw bh" (normalised 0-1)

STEP 2 — Evaluate Stage 2 MLP using perfect ground-truth bounding boxes.
  Skips Stage 1 person detector entirely.
  Uses ground-truth bbox + adaptive padding → MediaPipe → MLP → classify.
  Reports per-class accuracy to isolate whether Stage 1 or Stage 2 is the bottleneck.

Usage:
    python V2/src/convert_and_evaluate_stage2.py

Directory structure expected:
    ANNO_DIR/
    ├── fall/
    │   ├── fall001.txt   ("fall 1310 922 2580 2923")
    │   ├── fall001.jpg
    │   └── ...
    ├── sit/
    │   ├── sit001.txt
    │   ├── sit001.jpg
    │   └── ...
    └── walk/
        ├── walk001.txt
        ├── walk001.jpg
        └── ...

If your annotation files and images are in separate folders, update
ANNO_DIR and IMAGE_DIR paths below accordingly.
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

BASE = Path(__file__).resolve().parent.parent

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "pipeline_utils", str(BASE / "src" / "GUI" / "pipeline_utils.py")
)
_pu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pu)
GeometricFallRule = _pu.GeometricFallRule
CLASS_NAMES = _pu.CLASS_NAMES
FOLDER_TO_CLASS = _pu.FOLDER_TO_CLASS

# =============================================================================
#  CONFIGURATION — update these paths to match your setup
# =============================================================================

# Root folder containing fall/ sit/ walk/ subfolders with images + annotation txts
# If annotations and images are in the same folder per class, set both to same path
ANNO_DIR  = BASE / "dataset_stage1" / "fall_sit_walk_test_annotated" / "annotation"
IMAGE_DIR = BASE / "dataset_stage1" / "fall_sit_walk_test_annotated" / "annotated_images"

# Where to save converted YOLO labels
YOLO_OUT_DIR = BASE / "dataset_stage1" / "fall_sit_walk_test_yolo"

# Models
STAGE2_MODEL  = BASE / "models" / "stage2_pose_mlp_v3.pt"   # use v2 if v3 not ready
STAGE2_SCALER = BASE / "models" / "pose_scaler_v3.pkl"
POSE_MODEL    = BASE / "models" / "pose_landmarker_heavy.task"

# Adaptive padding — must match training
BBOX_PADDING_TALL = 0.25   # aspect < 1.2 (upright person)
BBOX_PADDING_WIDE = 0.10   # aspect >= 1.2 (prone/sitting person)
ASPECT_THRESHOLD  = 1.2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {"fall": 0, "sit": 1, "walk": 2}
GEO_RULE  = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)

# =============================================================================
#  STEP 1 — CONVERT ANNOTATIONS
# =============================================================================

def convert_annotations():
    """
    Convert class_name x1 y1 x2 y2 (absolute pixels) to
    YOLO normalised format: class_id cx cy bw bh
    """
    print("\n" + "="*60)
    print("  STEP 1 — Converting annotations to YOLO format")
    print("="*60)

    converted = 0
    skipped   = 0
    errors    = 0

    for folder_name in ["fall", "sit", "walk"]:
        anno_folder  = ANNO_DIR  / folder_name
        image_folder = IMAGE_DIR / folder_name
        yolo_folder  = YOLO_OUT_DIR / folder_name
        yolo_folder.mkdir(parents=True, exist_ok=True)

        if not anno_folder.exists():
            print(f"  [SKIP] {anno_folder} not found")
            continue

        txt_files = sorted(anno_folder.glob("*.txt"))
        print(f"\n  {folder_name}: {len(txt_files)} annotation files")

        for txt_path in txt_files:
            # Find matching image
            img_path = None
            for ext in IMG_EXTS:
                candidate = image_folder / (txt_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"    [WARN] No image found for {txt_path.name}")
                skipped += 1
                continue

            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [WARN] Could not read image: {img_path.name}")
                skipped += 1
                continue
            img_h, img_w = img.shape[:2]

            # Read and convert annotations
            lines = txt_path.read_text().strip().splitlines()
            yolo_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_name = parts[0].lower()
                if cls_name not in CLASS_MAP:
                    print(f"    [WARN] Unknown class '{cls_name}' in {txt_path.name}")
                    errors += 1
                    continue

                try:
                    x1 = int(parts[1])
                    y1 = int(parts[2])
                    x2 = int(parts[3])
                    y2 = int(parts[4])
                except ValueError:
                    print(f"    [WARN] Bad coordinates in {txt_path.name}: {line}")
                    errors += 1
                    continue

                # Clamp to image bounds
                x1 = max(0, min(img_w, x1))
                y1 = max(0, min(img_h, y1))
                x2 = max(0, min(img_w, x2))
                y2 = max(0, min(img_h, y2))

                if x2 <= x1 or y2 <= y1:
                    print(f"    [WARN] Invalid bbox in {txt_path.name}: {line}")
                    errors += 1
                    continue

                # Convert to YOLO normalised
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                cls_id = CLASS_MAP[cls_name]
                yolo_lines.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
                converted += 1

            # Save converted YOLO label
            out_path = yolo_folder / txt_path.name
            out_path.write_text("\n".join(yolo_lines))

    print(f"\n  Converted : {converted} bounding boxes")
    print(f"  Skipped   : {skipped} files (no matching image)")
    print(f"  Errors    : {errors} bad lines")
    print(f"  Output    : {YOLO_OUT_DIR}")


# =============================================================================
#  STEP 2 — EVALUATE STAGE 2 WITH PERFECT BOUNDING BOXES
# =============================================================================

def pad_bbox_adaptive(x1, y1, x2, y2, frame_w, frame_h):
    """Adaptive padding — tall box gets 25%, wide box gets 10%."""
    bw = x2 - x1
    bh = y2 - y1
    if bh == 0:
        return x1, y1, x2, y2
    aspect  = bw / bh
    padding = BBOX_PADDING_TALL if aspect < ASPECT_THRESHOLD else BBOX_PADDING_WIDE
    pad_x   = int(bw * padding)
    pad_y   = int(bh * padding)
    x1 = max(0,       x1 - pad_x)
    y1 = max(0,       y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    return x1, y1, x2, y2


def load_mlp(model_path, scaler_path):
    """Load MLP classifier and scaler. Falls back to v2 if v3 not found."""
    # Try v3 first, fall back to v2
    if not Path(model_path).exists():
        model_path  = str(BASE / "models" / "stage2_pose_mlp_v2.pt")
        scaler_path = str(BASE / "models" / "pose_scaler_v2.pkl")
        print(f"  [INFO] v3 model not found, using v2: {Path(model_path).name}")

    sys.path.insert(0, str(BASE / "src"))
    from train_stage2_pose import PoseMLP

    # Detect input size from model file
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    first_layer_key = [k for k in state.keys() if "weight" in k][0]
    n_features = state[first_layer_key].shape[1]
    print(f"  [INFO] MLP input features: {n_features}")

    mlp = PoseMLP(input_dim=n_features)
    mlp.load_state_dict(state)
    mlp.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return mlp, scaler, n_features


def load_feature_extractor():
    """Load the matching extract_features function based on feature count."""
    return _pu.extract_pose_features, len(_pu.FEATURE_COLS)


def evaluate_with_gt_boxes():
    """
    Evaluate Stage 2 MLP using ground-truth bounding boxes.
    Skips Stage 1 entirely — perfect localisation.
    """
    print("\n" + "="*60)
    print("  STEP 2 — Stage 2 Evaluation with Ground-Truth Boxes")
    print("  (Stage 1 person detector SKIPPED — using manual annotations)")
    print("="*60)

    # Load models
    print("\n  Loading models...")
    mlp, scaler, n_features = load_mlp(str(STAGE2_MODEL), str(STAGE2_SCALER))
    extract_features, feat_count = load_feature_extractor()

    if n_features != feat_count:
        print(f"  [WARN] MLP expects {n_features} features but extractor "
              f"produces {feat_count}. Using v2 model with v2 extractor.")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    print(f"  Models loaded.")

    results   = []
    method_counts = defaultdict(int)

    for folder_name, true_class_id in FOLDER_TO_CLASS.items():
        true_cls  = CLASS_NAMES[true_class_id]
        img_dir   = IMAGE_DIR / folder_name
        yolo_dir  = YOLO_OUT_DIR / folder_name

        if not img_dir.exists():
            print(f"\n  [SKIP] {img_dir}")
            continue

        images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        print(f"\n  {true_cls}: {len(images)} images")

        for img_path in images:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "load_error",
                })
                method_counts["load_error"] += 1
                continue

            h, w = img.shape[:2]

            # Load ground-truth YOLO annotation
            yolo_path = yolo_dir / (img_path.stem + ".txt")
            if not yolo_path.exists():
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "no_annotation",
                })
                method_counts["no_annotation"] += 1
                continue

            lines = yolo_path.read_text().strip().splitlines()
            if not lines:
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "empty_annotation",
                })
                method_counts["empty_annotation"] += 1
                continue

            # Use first (or largest) annotation box
            best_area = 0
            best_parts = None
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                bw_n = float(parts[3])
                bh_n = float(parts[4])
                area = bw_n * bh_n
                if area > best_area:
                    best_area  = area
                    best_parts = parts

            if best_parts is None:
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "bad_annotation",
                })
                method_counts["bad_annotation"] += 1
                continue

            # Convert YOLO normalised back to pixel coordinates
            cx_n = float(best_parts[1])
            cy_n = float(best_parts[2])
            bw_n = float(best_parts[3])
            bh_n = float(best_parts[4])

            x1 = int((cx_n - bw_n / 2) * w)
            y1 = int((cy_n - bh_n / 2) * h)
            x2 = int((cx_n + bw_n / 2) * w)
            y2 = int((cy_n + bh_n / 2) * h)

            # Adaptive padding
            x1, y1, x2, y2 = pad_bbox_adaptive(x1, y1, x2, y2, w, h)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "empty_crop",
                })
                method_counts["empty_crop"] += 1
                continue

            # MediaPipe pose
            rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "no_pose",
                })
                method_counts["no_pose"] += 1
                continue

            # Extract features
            features, core_vis = extract_features(result.pose_landmarks[0])

            # Geometric rule check
            if core_vis is not None:
                if GEO_RULE.check(
                    bbox=(x1, y1, x2, y2),
                    keypoint_visibilities=core_vis,
                ):
                    results.append({
                        "filename": img_path.name,
                        "true": true_cls,
                        "pred": "Fall",
                        "conf": 0.75,
                        "method": "geo_rule",
                    })
                    method_counts["geo_rule"] += 1
                    continue

            if features is None:
                results.append({
                    "filename": img_path.name,
                    "true": true_cls,
                    "pred": "NoDetection",
                    "conf": 0.0,
                    "method": "no_features",
                })
                method_counts["no_features"] += 1
                continue

            # MLP classification
            feat_norm = scaler.transform([features])
            with torch.no_grad():
                logits   = mlp(torch.tensor(feat_norm, dtype=torch.float32))
                probs    = torch.softmax(logits, dim=1)[0]
                pred_id  = int(probs.argmax())
                conf_val = float(probs[pred_id])

            results.append({
                "filename": img_path.name,
                "true": true_cls,
                "pred": CLASS_NAMES[pred_id],
                "conf": conf_val,
                "method": "pose_mlp",
            })
            method_counts["pose_mlp"] += 1

    landmarker.close()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STAGE 2 EVALUATION — GROUND TRUTH BOUNDING BOXES")
    print("  (No Stage 1 — perfect person localisation)")
    print("="*70)

    total   = len(results)
    correct = sum(1 for r in results if r["pred"] == r["true"])
    no_det  = sum(1 for r in results if r["pred"] == "NoDetection")

    if total == 0:
        print("\n  No images processed. Check ANNO_DIR and IMAGE_DIR paths.")
        return

    print(f"\n  Overall: {correct}/{total} = {correct/total:.1%}  "
          f"(NoDetection: {no_det})")

    labels = ["Fall", "Sit", "Walk"]
    print(f"\n  {'Class':<8} {'Total':>6} {'Correct':>8} {'Acc':>7}  "
          f"{'Misclassified As'}")
    print(f"  {'-'*65}")

    for cls in labels:
        cls_r   = [r for r in results if r["true"] == cls]
        correct_n = sum(1 for r in cls_r if r["pred"] == cls)
        total_n = len(cls_r)
        acc     = correct_n / total_n if total_n > 0 else 0

        misclass = defaultdict(int)
        for r in cls_r:
            if r["pred"] != cls:
                misclass[r["pred"]] += 1
        mis_str = ", ".join(
            f"{k}:{v}" for k, v in
            sorted(misclass.items(), key=lambda x: -x[1])
        )
        print(f"  {cls:<8} {total_n:>6} {correct_n:>8} {acc:>7.1%}  {mis_str}")

    print(f"\n  Method breakdown:")
    print(f"  {'Method':<20} {'Count':>6}")
    print(f"  {'-'*28}")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:<20} {count:>6}")

    # ── Comparison against Stage 1 results ───────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON — GT Boxes vs Stage 1 Detection")
    print(f"{'='*70}")
    print(f"""
  Metric              GT Boxes (Stage2 only)    Full Pipeline (S1+S2)
  ─────────────────────────────────────────────────────────────────────
  Overall accuracy    {correct/total:.1%}                      68.4%
  NoDetection rate    {no_det/total:.1%}                       9.8%

  If GT accuracy >> pipeline accuracy → Stage 1 is the bottleneck
  If GT accuracy ≈  pipeline accuracy → Stage 2 MLP is the bottleneck
""")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = BASE / "FALL_Detected_Manual_Test" / "confusion_matrices" / \
              "predictions_gt_boxes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved → {out_csv}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("COS30018 — Stage 2 Ground-Truth Box Evaluation")
    print(f"Annotation dir : {ANNO_DIR}")
    print(f"Image dir      : {IMAGE_DIR}")
    print(f"YOLO output    : {YOLO_OUT_DIR}")

    # Step 1: Convert annotations
    convert_annotations()

    # Step 2: Evaluate
    evaluate_with_gt_boxes()


if __name__ == "__main__":
    main()

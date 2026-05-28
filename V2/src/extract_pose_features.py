"""
extract_pose_features.py
Extracts MediaPipe Pose keypoints and computes geometric features
for Fall/Sit/Walk classification training.

v3: Adds 3 new features over v2 to address observed misclassification patterns:
    - face_vertical:       nose.y - mean(ears.y). Negative = face pointing up
                           (fallen on back). Addresses Fall->Sit misclassification
                           where fallen person's face-up orientation distinguishes
                           them from seated person.
    - shoulder_hip_vert:   Absolute vertical distance shoulder midpoint to hip
                           midpoint, normalised by body height. Large = spine
                           vertical (upright/sitting). Small = spine horizontal
                           (fallen). Directly discriminates sit-on-floor from fall.
    - ankle_visibility:    Mean visibility of both ankle landmarks [0.0-1.0].
                           Low = lower body not detected by MediaPipe. Flags
                           unreliable predictions caused by Stage 1 bbox cropping
                           off lower body, which artificially produces Sit-like
                           features (zero ankle distance = compressed body).

v2: Uses padded crop simulation to match inference domain.
     Training images are cropped using ground-truth YOLO bbox + 25% padding
     before feeding to MediaPipe, matching how Stage 1 detection + padding
     works at inference time.

Features extracted per person (11 total):
  0:  spine_angle          - angle of shoulder-hip vector from vertical (degrees)
  1:  hip_height_norm      - normalised hip y-position (0=top, 1=bottom)
  2:  shoulder_width_norm  - shoulder span / body height
  3:  knee_bend_left       - left knee angle in degrees
  4:  knee_bend_right      - right knee angle in degrees
  5:  body_aspect_ratio    - keypoint bbox width / height
  6:  head_height_norm     - nose y-position normalised to frame height
  7:  ankle_hip_vert_dist  - vertical distance ankles to hips / body height
  8:  face_vertical        - nose.y - mean(ear.y). Negative = face up (fallen)
  9:  shoulder_hip_vert    - abs vertical dist shoulders to hips / body height
  10: ankle_visibility     - mean visibility of ankle landmarks [0.0-1.0]

Usage:
    python V2/src/extract_pose_features.py --split both
    python V2/src/extract_pose_features.py --split train
    python V2/src/extract_pose_features.py --split test
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

CLASS_NAMES = {0: "Fall", 1: "Sit", 2: "Walk"}
FOLDER_TO_CLASS = {"fall": 0, "sit": 1, "walk": 2}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "datasets" / "pose_features"
MODEL_PATH = str(BASE / "models" / "pose_landmarker_heavy.task")

BBOX_PADDING = 0.25

FEATURE_HEADERS = [
    # v2 original features
    "spine_angle",
    "hip_height_norm",
    "shoulder_width_norm",
    "knee_bend_left",
    "knee_bend_right",
    "body_aspect_ratio",
    "head_height_norm",
    "ankle_hip_vert_dist",
    # v3 new features
    "face_vertical",
    "shoulder_hip_vert",
    "ankle_visibility",
]


def _create_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    return PoseLandmarker.create_from_options(options)


def pad_bbox(x1, y1, x2, y2, frame_w, frame_h, padding=BBOX_PADDING):
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    return x1, y1, x2, y2


def load_padded_crop(img_path, label_path, padding=BBOX_PADDING):
    """Load image, crop using ground-truth bbox + padding to simulate inference domain."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if label_path is None or not label_path.exists():
        return img

    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return img

    best_area = 0
    best_parts = None
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (
            float(parts[1]), float(parts[2]),
            float(parts[3]), float(parts[4]),
        )
        area = bw * bh
        if area > best_area:
            best_area = area
            best_parts = parts

    if best_parts is None:
        return img

    cx, cy, bw, bh = (
        float(best_parts[1]), float(best_parts[2]),
        float(best_parts[3]), float(best_parts[4]),
    )
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h, padding)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return img

    return img[y1:y2, x1:x2]


def angle_between(p1, p2, p3) -> float:
    """Angle at p2 in degrees formed by points p1-p2-p3."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def extract_features(landmarks) -> list:
    """
    Extract 11 geometric features from MediaPipe pose landmarks.

    Landmark indices used:
      0  = nose
      3  = left ear
      4  = right ear
      11 = left shoulder
      12 = right shoulder
      23 = left hip
      24 = right hip
      25 = left knee
      26 = right knee
      27 = left ankle
      28 = right ankle

    All x/y coordinates are normalised to [0,1] by MediaPipe
    where (0,0) is top-left and (1,1) is bottom-right.
    """
    lm = landmarks

    # Key landmarks
    nose  = lm[0]
    l_ear = lm[3];  r_ear = lm[4]
    l_sh  = lm[11]; r_sh  = lm[12]
    l_hi  = lm[23]; r_hi  = lm[24]
    l_kn  = lm[25]; r_kn  = lm[26]
    l_an  = lm[27]; r_an  = lm[28]

    # Midpoints
    mid_sh_x = (l_sh.x + r_sh.x) / 2
    mid_sh_y = (l_sh.y + r_sh.y) / 2
    mid_hi_x = (l_hi.x + r_hi.x) / 2
    mid_hi_y = (l_hi.y + r_hi.y) / 2
    mid_an_y = (l_an.y + r_an.y) / 2

    # ── Feature 0: spine_angle ──────────────────────────────────────────────
    # Angle of shoulder-hip vector from vertical.
    # 0° = perfectly upright spine. 90° = horizontal spine (fallen).
    dx = mid_sh_x - mid_hi_x
    dy = mid_sh_y - mid_hi_y
    spine_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))

    # ── Feature 1: hip_height_norm ──────────────────────────────────────────
    # Normalised y-position of hips. Low value = hips near top (standing).
    # High value = hips near bottom (sitting/fallen on floor).
    hip_height_norm = mid_hi_y

    # ── Feature 2: shoulder_width_norm ─────────────────────────────────────
    # Shoulder span relative to body height.
    # High = person is wide relative to height (horizontal/fallen).
    body_height = abs(mid_sh_y - mid_an_y) + 1e-9
    shoulder_width_norm = abs(l_sh.x - r_sh.x) / body_height

    # ── Features 3-4: knee_bend_left / right ───────────────────────────────
    # Angle at each knee. ~180° = straight leg (walking/standing/lying).
    # <120° = bent knee (sitting, crouching).
    knee_bend_left  = angle_between(l_hi, l_kn, l_an)
    knee_bend_right = angle_between(r_hi, r_kn, r_an)

    # ── Feature 5: body_aspect_ratio ───────────────────────────────────────
    # Width/height of bounding box enclosing key body landmarks.
    # >1.0 = body is wider than tall = horizontal = likely fallen.
    xs = [lm[i].x for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    ys = [lm[i].y for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    body_aspect_ratio = bbox_w / (bbox_h + 1e-9)

    # ── Feature 6: head_height_norm ────────────────────────────────────────
    # Normalised y-position of nose (0=top, 1=bottom of frame).
    head_height_norm = nose.y

    # ── Feature 7: ankle_hip_vert_dist ─────────────────────────────────────
    # Vertical distance from ankles to hips, normalised by body height.
    # Large = legs are extended below hips (standing/walking).
    # Small = legs folded up or horizontal (sitting/fallen).
    ankle_hip_vert = abs(mid_an_y - mid_hi_y) / (body_height + 1e-9)

    # ── Feature 8: face_vertical [NEW v3] ──────────────────────────────────
    # nose.y minus mean(ear.y). Uses image y-axis where 0=top, 1=bottom.
    # Negative = nose is ABOVE ears = face pointing UP = fallen on back.
    # Near zero = face pointing forward = normal upright orientation.
    # Positive = nose is BELOW ears = face pointing DOWN = prone fall or bowing.
    #
    # This directly addresses the Fall->Sit misclassification observed in
    # debug images: fallen persons lying on their back have clearly negative
    # face_vertical while seated persons always have near-zero or positive values.
    ear_y_mean = (l_ear.y + r_ear.y) / 2
    face_vertical = float(nose.y - ear_y_mean)

    # ── Feature 9: shoulder_hip_vert [NEW v3] ──────────────────────────────
    # Absolute vertical distance between shoulder midpoint and hip midpoint,
    # normalised by body height.
    # Large = spine is vertical = person is upright (standing or sitting on seat).
    # Small = spine is horizontal = person is lying down (fallen).
    #
    # Complements spine_angle by measuring actual vertical separation rather
    # than tilt angle. A seated person leaning forward has high spine_angle
    # but still has shoulders well above hips — this feature captures that.
    # Addresses sit050 case: cross-legged floor sitter vs fallen person.
    shoulder_hip_vert = abs(mid_sh_y - mid_hi_y) / (body_height + 1e-9)

    # ── Feature 10: ankle_visibility [NEW v3] ──────────────────────────────
    # Mean MediaPipe visibility score for both ankle landmarks [0.0-1.0].
    # Low visibility = MediaPipe could not confidently locate ankles.
    # This happens when Stage 1 bounding box crops off the lower body,
    # causing ankle_hip_vert_dist to collapse to near-zero and producing
    # false Sit-like features for walking persons.
    #
    # The MLP can use this as a reliability flag: if ankle_visibility < 0.3,
    # the lower-body features (knee_bend, ankle_hip_vert) are unreliable.
    ankle_visibility = float((l_an.visibility + r_an.visibility) / 2)

    return [
        # v2 features (0-7)
        spine_angle,
        hip_height_norm,
        shoulder_width_norm,
        knee_bend_left,
        knee_bend_right,
        body_aspect_ratio,
        head_height_norm,
        ankle_hip_vert,
        # v3 features (8-10)
        face_vertical,
        shoulder_hip_vert,
        ankle_visibility,
    ]


def process_yolo_split(
    landmarker, source_img_dir: Path, source_lbl_dir: Path, output_csv: Path
):
    """Process training data in YOLO format — uses padded crops from ground truth bbox."""
    rows = []
    failed = 0
    no_label = 0

    images = sorted(
        f for f in source_img_dir.iterdir() if f.suffix.lower() in IMG_EXTS
    )
    print(f"  Found {len(images)} images in {source_img_dir.name}")

    for i, img_path in enumerate(images):
        if (i + 1) % 500 == 0:
            print(f"  ... processed {i + 1}/{len(images)}")

        lbl_path = source_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            no_label += 1
            continue

        content = lbl_path.read_text().strip()
        if not content:
            continue

        first_line = content.splitlines()[0].strip().split()
        class_id = int(first_line[0])
        if class_id not in CLASS_NAMES:
            continue

        crop = load_padded_crop(img_path, lbl_path)
        if crop is None:
            failed += 1
            continue

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            features = extract_features(result.pose_landmarks[0])
            rows.append({
                "filename": img_path.name,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                **{h: f for h, f in zip(FEATURE_HEADERS, features)},
            })
        else:
            failed += 1

    _save_csv(rows, output_csv)
    print(f"  Saved {len(rows)} rows  ->  {output_csv}")
    print(f"  Failed (no pose): {failed}   No label: {no_label}")
    return rows


def process_folder_split(landmarker, source_dir: Path, output_csv: Path):
    """Process test data organised in class folders (fall/, sit/, walk/).
    No YOLO labels available — uses full images."""
    rows = []
    failed = 0

    for folder_name, class_id in FOLDER_TO_CLASS.items():
        folder = source_dir / folder_name
        if not folder.exists():
            print(f"  [SKIP] {folder}")
            continue
        images = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS
        )
        print(f"  {CLASS_NAMES[class_id]}: {len(images)} images")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                features = extract_features(result.pose_landmarks[0])
                rows.append({
                    "filename": img_path.name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    **{h: f for h, f in zip(FEATURE_HEADERS, features)},
                })
            else:
                failed += 1

    _save_csv(rows, output_csv)
    print(f"  Saved {len(rows)} rows  ->  {output_csv}")
    print(f"  Failed (no pose): {failed}")
    return rows


def _save_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("  WARNING: No rows extracted!")
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract pose features (v3 — 11 features) for Fall/Sit/Walk MLP"
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="both",
        help="Which split to process"
    )
    args = parser.parse_args()

    train_img_dir = BASE / "dataset_stage1" / "swook_techcare" / "train" / "images"
    train_lbl_dir = BASE / "dataset_stage1" / "swook_techcare" / "train" / "labels"
    test_dir      = BASE / "dataset_stage1" / "fall_sit_walk_test"

    print(f"Model      : {MODEL_PATH}")
    print(f"Padding    : {BBOX_PADDING}")
    print(f"Features   : {len(FEATURE_HEADERS)} ({', '.join(FEATURE_HEADERS)})")

    landmarker = _create_landmarker()

    if args.split in ("train", "both"):
        print("\nExtracting TRAIN features (YOLO format, padded crops)...")
        process_yolo_split(
            landmarker, train_img_dir, train_lbl_dir,
            OUTPUT_DIR / "train_features_v3.csv",
        )

    if args.split in ("test", "both"):
        print("\nExtracting TEST features (folder format, full images)...")
        process_folder_split(
            landmarker, test_dir,
            OUTPUT_DIR / "test_features_v3.csv",
        )

    landmarker.close()
    print("\nDone. Next step: python V2/src/train_stage2_pose.py --features v3")


if __name__ == "__main__":
    main()
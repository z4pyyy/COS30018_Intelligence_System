"""
extract_crops.py
================
Extracts padded person crops from swook_techcare dataset
for CNN Stage 2 classifier training.

Reads YOLO labels (0=Fall, 1=Sit, 2=Walk), crops each annotated
person region with adaptive padding, resizes to 224x224, and saves
into ImageFolder-compatible structure for PyTorch training.

Output structure:
    V3/datasets/stage2_crops/
    ├── train/
    │   ├── Fall/   (~85% of fall crops)
    │   ├── Sit/    (~85% of sit crops)
    │   └── Walk/   (~85% of walk crops)
    └── val/
        ├── Fall/   (~15% of fall crops)
        ├── Sit/    (~15% of sit crops)
        └── Walk/   (~15% of walk crops)

Usage:
    python V3/src/extract_crops.py
"""

import random
import shutil
from pathlib import Path

import cv2

# =============================================================================
#  PATHS
# =============================================================================

V2_BASE  = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V2")
V3_BASE  = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V3")

SRC_IMG  = V2_BASE / "dataset" / "swook_techcare" / "train" / "images"
SRC_LBL  = V2_BASE / "dataset" / "swook_techcare" / "train" / "labels"

OUT_DIR  = V3_BASE / "datasets" / "stage2_crops"

# =============================================================================
#  CONFIGURATION
# =============================================================================

CLASS_NAMES = {0: "Fall", 1: "Sit", 2: "Walk"}
IMG_SIZE    = 224       # EfficientNet-B0 input size
VAL_SPLIT   = 0.15      # 15% validation
RANDOM_SEED = 42

# Adaptive padding — same as pipeline_utils
ASPECT_THRESHOLD  = 1.2
BBOX_PADDING_TALL = 0.25
BBOX_PADDING_WIDE = 0.10

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================================================
#  HELPERS
# =============================================================================

def pad_bbox(x1, y1, x2, y2, frame_w, frame_h):
    bw = x2 - x1
    bh = y2 - y1
    if bh == 0:
        return x1, y1, x2, y2
    aspect  = bw / (bh + 1e-9)
    padding = BBOX_PADDING_TALL if aspect < ASPECT_THRESHOLD else BBOX_PADDING_WIDE
    pad_x   = int(bw * padding)
    pad_y   = int(bh * padding)
    x1 = max(0,       x1 - pad_x)
    y1 = max(0,       y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    return x1, y1, x2, y2


def yolo_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2

# =============================================================================
#  MAIN
# =============================================================================

def main():
    random.seed(RANDOM_SEED)

    # Create output directories
    for split in ["train", "val"]:
        for cls_name in CLASS_NAMES.values():
            (OUT_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    print(f"Source images : {SRC_IMG}")
    print(f"Source labels : {SRC_LBL}")
    print(f"Output        : {OUT_DIR}")
    print(f"Image size    : {IMG_SIZE}x{IMG_SIZE}")
    print(f"Val split     : {VAL_SPLIT:.0%}")
    print()

    # Collect all (image_path, class_id, bbox) tuples per class
    per_class = {0: [], 1: [], 2: []}

    label_files = sorted(SRC_LBL.glob("*.txt"))
    print(f"Processing {len(label_files)} label files...")

    skipped = 0
    for lbl_path in label_files:
        # Find matching image
        img_path = None
        for ext in IMG_EXTS:
            candidate = SRC_IMG / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        lines = lbl_path.read_text().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in CLASS_NAMES:
                continue
            cx, cy, bw, bh = (
                float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]),
            )
            per_class[cls_id].append((img_path, cx, cy, bw, bh))

    print(f"Skipped (no image): {skipped}")
    for cls_id, items in per_class.items():
        print(f"  {CLASS_NAMES[cls_id]}: {len(items)} annotations")

    # Extract and save crops
    print("\nExtracting crops...")
    stats = {"train": {0: 0, 1: 0, 2: 0}, "val": {0: 0, 1: 0, 2: 0}}
    errors = 0

    for cls_id, items in per_class.items():
        cls_name = CLASS_NAMES[cls_id]
        random.shuffle(items)

        n_val   = int(len(items) * VAL_SPLIT)
        val_set = set(range(len(items) - n_val, len(items)))

        for i, (img_path, cx, cy, bw, bh) in enumerate(items):
            split = "val" if i in val_set else "train"

            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue

            h, w = img.shape[:2]
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, w, h)
            x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                errors += 1
                continue

            # Resize to CNN input size
            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            # Save with unique filename
            out_name = f"{img_path.stem}_{i:05d}.jpg"
            out_path = OUT_DIR / split / cls_name / out_name
            cv2.imwrite(str(out_path), crop_resized)
            stats[split][cls_id] += 1

        if (cls_id + 1) % 1 == 0:
            print(f"  {cls_name}: done")

    # Summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Errors: {errors}")
    print(f"\n{'Split':<8} {'Fall':>8} {'Sit':>8} {'Walk':>8} {'Total':>8}")
    print(f"{'-'*40}")
    for split in ["train", "val"]:
        total = sum(stats[split].values())
        print(f"{split:<8} "
              f"{stats[split][0]:>8} "
              f"{stats[split][1]:>8} "
              f"{stats[split][2]:>8} "
              f"{total:>8}")

    print(f"\nNext: python V3/src/train_stage2_cnn.py")


if __name__ == "__main__":
    main()
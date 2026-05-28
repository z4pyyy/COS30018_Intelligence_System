"""
prepare_finetune_crops.py  (V3)
================================
Gathers wrong-prediction images, extracts Stage 1 person crops,
augments them 15x, and saves into stage2_crops fine-tune split.

Source:  V3/runs/evaluation/twostep_cnn/wrong_predictions/
Output:  V3/datasets/stage2_crops/finetune/{Fall,Sit,Walk}/

Folder → class mapping (what the TRUE label is):
  Fall_as_Sit   → Fall   (CNN confused Fall for Sit)
  Fall_as_Walk  → Fall
  Walk_as_Sit   → Walk   (CNN confused Walk for Sit)
  Walk_as_Fall  → Walk   ← NOTE: these are likely geo_rule misfires,
                            still add as hard negatives for Walk
  Sit_as_Walk   → Sit

Walk_as_Fall images are flagged separately so you can inspect them —
they're likely a code fix (geo_rule aspect threshold), not just training.

After running:
  - Check finetune/ counts in terminal output
  - Add finetune/ into your train_stage2_cnn.py dataset path
  - Fine-tune for 5-10 epochs only (not full retrain)

Usage:
    python V3/src/prepare_finetune_crops.py
"""

import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# =============================================================================
#  CONFIG
# =============================================================================

WRONG_PRED_DIR = Path(
    r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection"
    r"\V3\runs\evaluation\twostep_cnn\wrong_predictions"
)

V3_BASE   = Path(
    r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V3"
)
V2_BASE   = V3_BASE.parent / "V2"

OUT_DIR   = V3_BASE / "datasets" / "stage2_crops" / "finetune"
CROP_SIZE = 224
AUG_TIMES = 15        # augmentations per source image
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# True class for each wrong-prediction folder
FOLDER_TO_CLASS = {
    "Fall_as_Sit":  "Fall",
    "Fall_as_Walk": "Fall",
    "Walk_as_Sit":  "Walk",
    "Walk_as_Fall": "Walk",   # geo_rule suspect — flagged in output
    "Sit_as_Walk":  "Sit",
}

# Walk_as_Fall is suspected geo_rule misfire — warn but still include
GEO_RULE_SUSPECT = {"Walk_as_Fall"}

# =============================================================================
#  AUGMENTATION
# =============================================================================

def augment(crop: np.ndarray) -> list[np.ndarray]:
    """
    Returns AUG_TIMES augmented variants of a single crop.
    Augmentations chosen to match realistic fall detection variation:
      - Horizontal flip (falls happen both directions)
      - Brightness / contrast jitter (indoor lighting varies a lot)
      - Slight rotation (camera tilt, uneven floors)
      - Gaussian noise (compressed video frames)
      - Random crop + resize (simulate slight detection box shift)
    """
    results = []
    h, w = crop.shape[:2]

    for i in range(AUG_TIMES):
        img = crop.copy()

        # ── Horizontal flip (50%) ──────────────────────────────────────────
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # ── Brightness + contrast jitter ──────────────────────────────────
        alpha = random.uniform(0.7, 1.4)     # contrast
        beta  = random.randint(-30, 30)      # brightness
        img   = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # ── Rotation ±15° ─────────────────────────────────────────────────
        angle = random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)

        # ── Gaussian noise ────────────────────────────────────────────────
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(3, 12), img.shape)
            img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # ── Random crop + resize (simulate bbox shift ±10%) ───────────────
        if random.random() < 0.5:
            margin_x = int(w * 0.10)
            margin_y = int(h * 0.10)
            x1 = random.randint(0, margin_x)
            y1 = random.randint(0, margin_y)
            x2 = random.randint(w - margin_x, w)
            y2 = random.randint(h - margin_y, h)
            img = cv2.resize(img[y1:y2, x1:x2], (w, h),
                             interpolation=cv2.INTER_LINEAR)

        results.append(img)

    return results


# =============================================================================
#  CROP EXTRACTION  (Stage 1 → adaptive pad → 224×224)
# =============================================================================

def extract_crop(img_path: Path, stage1_model) -> np.ndarray | None:
    """
    Runs Stage 1 on the image, takes the largest person box,
    applies adaptive padding, returns 224×224 crop.
    Returns None if no person detected.
    """
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import pad_bbox, STAGE1_CONF

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    r = stage1_model.predict(
        source=img, conf=STAGE1_CONF,
        imgsz=640, device=0, verbose=False,
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None

    areas  = [(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.cpu().numpy()]
    best_i = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[best_i].cpu().numpy())
    x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    # ── Load Stage 1 ─────────────────────────────────────────────────────────
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import load_models
    from ultralytics import YOLO

    stage1_path = V2_BASE / "models" / "stage1_person.pt"
    print(f"Loading Stage 1: {stage1_path}")
    stage1 = YOLO(str(stage1_path))

    # ── Clear and recreate output dirs ────────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for cls in ["Fall", "Sit", "Walk"]:
        (OUT_DIR / cls).mkdir(parents=True)

    # ── Process each wrong-prediction folder ──────────────────────────────────
    stats        = defaultdict(lambda: {"source": 0, "augmented": 0, "no_detect": 0})
    geo_suspects = []

    for folder_name, true_cls in FOLDER_TO_CLASS.items():
        src_folder = WRONG_PRED_DIR / folder_name
        if not src_folder.exists():
            print(f"  [SKIP — not found] {folder_name}")
            continue

        images = sorted(f for f in src_folder.iterdir()
                        if f.suffix.lower() in IMG_EXTS)

        if folder_name in GEO_RULE_SUSPECT:
            geo_suspects.extend(images)

        print(f"\n  {folder_name} → {true_cls}  ({len(images)} images)")

        dst_folder = OUT_DIR / true_cls

        for img_path in images:
            crop = extract_crop(img_path, stage1)

            if crop is None:
                print(f"    [no detect] {img_path.name}")
                stats[folder_name]["no_detect"] += 1
                # Fall back: resize full image as crop
                raw = cv2.imread(str(img_path))
                if raw is not None:
                    crop = cv2.resize(raw, (CROP_SIZE, CROP_SIZE))
                else:
                    continue

            stats[folder_name]["source"] += 1

            # Save original crop (counts as 1 of the training samples)
            base_name = f"ft_{folder_name}_{img_path.stem}"
            cv2.imwrite(str(dst_folder / f"{base_name}_orig.jpg"), crop)

            # Save augmented variants
            augmented = augment(crop)
            for aug_i, aug_crop in enumerate(augmented):
                cv2.imwrite(
                    str(dst_folder / f"{base_name}_aug{aug_i:02d}.jpg"),
                    aug_crop,
                )

            stats[folder_name]["augmented"] += AUG_TIMES

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINE-TUNE CROP SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Folder':<22} {'Src':>5} {'No Det':>7} {'Aug':>7} {'Total':>7}")
    print(f"  {'-'*55}")

    class_totals = defaultdict(int)
    for folder_name, true_cls in FOLDER_TO_CLASS.items():
        s = stats[folder_name]
        total = s["source"] + s["augmented"]
        class_totals[true_cls] += total
        flag = " ⚠ geo_rule suspect" if folder_name in GEO_RULE_SUSPECT else ""
        print(f"  {folder_name:<22} {s['source']:>5} {s['no_detect']:>7} "
              f"{s['augmented']:>7} {total:>7}{flag}")

    print(f"\n  Per-class totals added to finetune/:")
    existing = {
        "Fall": 8079,
        "Sit":  6565,
        "Walk": 14140,
    }
    for cls in ["Fall", "Sit", "Walk"]:
        added  = class_totals[cls]
        before = existing[cls]
        after  = before + added
        pct    = added / before * 100
        print(f"    {cls:<6}  {before:>6} → {after:>6}  (+{added}, +{pct:.1f}%)")

    # ── Geo-rule suspect warning ──────────────────────────────────────────────
    if geo_suspects:
        print(f"\n  ⚠  Walk_as_Fall ({len(geo_suspects)} images):")
        print(f"     These walking people were predicted Fall at 100% confidence.")
        print(f"     This pattern = geo_rule misfiring on narrow upright crops.")
        print(f"     Recommended fix: in pipeline_utils.py, raise")
        print(f"     GeometricFallRule aspect_threshold from 1.8 → 2.2")
        print(f"     Then re-evaluate BEFORE fine-tuning to see if these disappear.")
        print(f"     Files:")
        for p in geo_suspects:
            print(f"       {p.name}")

    print(f"\n  Output → {OUT_DIR}")
    print(f"\n  Next steps:")
    print(f"    1. If Walk_as_Fall caused by geo_rule → fix threshold first,")
    print(f"       re-evaluate, then re-run this script with updated wrong_predictions/")
    print(f"    2. Add finetune/ to train_stage2_cnn.py dataset paths")
    print(f"    3. Fine-tune for 5-10 epochs only (not full retrain)")
    print(f"    4. Use lower LR: 1e-4 instead of default")


if __name__ == "__main__":
    main()
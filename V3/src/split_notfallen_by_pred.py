"""
split_notfallen_by_pred.py
===========================
Uses Pipeline 1 (GT box → CNN) predictions from the kaggle evaluation CSV
to pre-sort not-fallen images into Sit / Walk / Fall subfolders.

Source images : fall_dataset_kaggle/images/{train,val}/
Source labels  : fall_dataset_kaggle/labels/{train,val}/
CSV            : V3/runs/evaluation/kaggle/csvs/kaggle_gt_box_cnn.csv

Output:
  fall_sit_walk_kaggle/
  ├── fall/        ← gt=fallen images (all 285, for completeness)
  │   └── labels/
  ├── sit/         ← gt=not_fallen, CNN predicted Sit  (119 images)
  │   └── labels/
  └── walk/        ← gt=not_fallen, CNN predicted Walk (72 images)
      └── labels/

Note: 9 not-fallen images CNN predicted as Fall → copied to fall/ with a
      _FALSEALARM suffix so you can spot them during manual inspection.

After running:
  1. Open sit/ and walk/ and manually verify
     - Remove any obvious misclassifications
     - Move images between folders as needed
  2. Run evaluate_sit_walk_confusion.py to get the final confusion matrix
     (that script uses these verified folders as ground truth)

Usage:
    python V3/src/split_notfallen_by_pred.py
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

# =============================================================================
#  PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection")

KAGGLE_DIR   = PROJECT_ROOT / "fall_dataset_kaggle"
CSV_PATH     = PROJECT_ROOT / "V3" / "runs" / "evaluation" / "kaggle" / "csvs" / "kaggle_gt_box_cnn.csv"
OUT_DIR      = PROJECT_ROOT / "fall_sit_walk_kaggle"

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================================================
#  HELPERS
# =============================================================================

def find_image(filename: str) -> Path | None:
    """Search both train/ and val/ splits for the image file."""
    for split in ["train", "val"]:
        p = KAGGLE_DIR / "images" / split / filename
        if p.exists():
            return p
    return None


def find_label(filename: str) -> Path | None:
    """Search both train/ and val/ splits for the label file."""
    stem = Path(filename).stem
    for split in ["train", "val"]:
        p = KAGGLE_DIR / "labels" / split / (stem + ".txt")
        if p.exists():
            return p
    return None


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  split_notfallen_by_pred.py")
    print("  Pre-sorting not-fallen images by CNN prediction")
    print("=" * 65)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"\n  [ERROR] CSV not found: {CSV_PATH}")
        print(f"          Run evaluate_kaggle.py first.")
        return

    # ── Read CSV ──────────────────────────────────────────────────────────────
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"\n  Loaded {len(rows)} predictions from CSV")

    # ── Create output folders ─────────────────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for cls in ["fall", "sit", "walk"]:
        (OUT_DIR / cls / "images").mkdir(parents=True)
        (OUT_DIR / cls / "labels").mkdir(parents=True)

    # ── Sort and copy ─────────────────────────────────────────────────────────
    counts      = defaultdict(int)
    not_found   = []
    falsealarms = []   # not-fallen predicted as Fall

    for row in rows:
        filename  = row["filename"]
        gt_class  = row["gt_class"]   # "fallen" or "not_fallen"
        pred      = row["pred"]       # "Fall", "Sit", "Walk", "NoDetection"

        img_src   = find_image(filename)
        label_src = find_label(filename)

        if img_src is None:
            not_found.append(filename)
            continue

        # ── Determine destination folder ──────────────────────────────────────
        if gt_class == "fallen":
            dst_cls = "fall"
            dst_name = filename

        elif gt_class == "not_fallen":
            if pred == "Sit":
                dst_cls  = "sit"
                dst_name = filename
            elif pred == "Walk":
                dst_cls  = "walk"
                dst_name = filename
            elif pred == "Fall":
                # False alarm — put in fall/ but flag with suffix for easy spotting
                dst_cls  = "fall"
                stem     = Path(filename).stem
                ext      = Path(filename).suffix
                dst_name = f"{stem}_FALSEALARM{ext}"
                falsealarms.append(filename)
            else:
                # NoDetection — skip, can't classify
                counts["skipped_nodetection"] += 1
                continue
        else:
            continue

        # ── Copy image ────────────────────────────────────────────────────────
        dst_img = OUT_DIR / dst_cls / "images" / dst_name
        shutil.copy2(str(img_src), str(dst_img))
        counts[f"{dst_cls}_images"] += 1

        # ── Copy label (if exists) ────────────────────────────────────────────
        if label_src is not None:
            dst_stem  = Path(dst_name).stem
            dst_label = OUT_DIR / dst_cls / "labels" / (dst_stem + ".txt")
            shutil.copy2(str(label_src), str(dst_label))
            counts[f"{dst_cls}_labels"] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  OUTPUT SUMMARY")
    print(f"{'=' * 65}")
    print(f"\n  {'Folder':<10} {'Images':>8} {'Labels':>8}")
    print(f"  {'-'*30}")
    for cls in ["fall", "sit", "walk"]:
        imgs   = counts[f"{cls}_images"]
        labels = counts[f"{cls}_labels"]
        print(f"  {cls:<10} {imgs:>8} {labels:>8}")

    total = sum(counts[f"{c}_images"] for c in ["fall", "sit", "walk"])
    print(f"\n  Total images copied : {total}")
    print(f"  Skipped (NoDetect)  : {counts['skipped_nodetection']}")

    if falsealarms:
        print(f"\n  ⚠  False alarms in fall/ ({len(falsealarms)} images):")
        print(f"     These are not-fallen images the CNN predicted as Fall.")
        print(f"     Saved with _FALSEALARM suffix — easy to spot and remove.")
        for f in falsealarms:
            print(f"       {f}")

    if not_found:
        print(f"\n  ⚠  Images not found ({len(not_found)}):")
        for f in not_found[:10]:
            print(f"       {f}")
        if len(not_found) > 10:
            print(f"       ... and {len(not_found) - 10} more")

    print(f"\n{'=' * 65}")
    print(f"  OUTPUT → {OUT_DIR}")
    print(f"{'=' * 65}")
    print(f"\n  Next steps:")
    print(f"  1. Open sit/ and walk/ and manually verify")
    print(f"     - Move misclassified images to the correct folder")
    print(f"     - Delete images that are ambiguous or unusable")
    print(f"  2. Run evaluate_sit_walk_confusion.py for confusion matrix")


if __name__ == "__main__":
    main()
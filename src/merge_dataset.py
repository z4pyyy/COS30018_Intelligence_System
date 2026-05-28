"""
merge_datasets.py
Combines Fall_sit_walk_bbox (techcare, 4567 images) and better_Fall_sit_walk (swook, 8340 images)
into a single merged dataset ready for YOLOv8 training.

Both datasets share identical class IDs:
  0 = Fall, 1 = Sit, 2 = Walk/Stand
No label remapping needed — only the final data.yaml names field is updated.

Usage:
  python merge_datasets.py
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\datasets")

SOURCES = [
    {
        "name": "techcare",
        "root": BASE / "Fall_sit_walk_bbox",
        "splits": {
            "train": ("train/images", "train/labels"),
            "valid": ("valid/images", "valid/labels"),
            "test":  ("test/images",  "test/labels"),
        }
    },
    {
        "name": "swook",
        "root": BASE / "better_Fall_sit_walk",
        "splits": {
            "train": ("train/images", "train/labels"),
            "valid": ("valid/images", "valid/labels"),
            "test":  ("test/images",  "test/labels"),
        }
    },
]

OUTPUT = BASE / "merged_dataset"
CLASS_NAMES = ['Fall', 'Sit', 'Walk']   # stand → Walk (ID unchanged, name only)

# ── HELPERS ───────────────────────────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def file_hash(path: Path) -> str:
    """MD5 of first 64KB — fast near-duplicate check."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read(65536))
    return h.hexdigest()

def count_classes(label_dir: Path) -> Counter:
    counts = Counter()
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return counts

# ── MERGE ─────────────────────────────────────────────────────────────────────
def merge():
    # Roboflow test splits are NOT used — our self-collected images are the real test set.
    # We merge train + valid + test from both sources → all into training pool,
    # then hold out 15% as validation automatically via YOLO's val split during training.
    # Output structure: merged_dataset/train/images + labels

    out_img = OUTPUT / "train" / "images"
    out_lbl = OUTPUT / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    seen_hashes = {}       # hash → original filename (dedup)
    stats = defaultdict(int)
    class_counts = Counter()
    conflicts = 0
    duplicates = 0

    for src in SOURCES:
        root = src["root"]
        name = src["name"]
        print(f"\n── Processing source: {name} ({root.name}) ──")

        for split, (img_rel, lbl_rel) in src["splits"].items():
            img_dir = root / img_rel
            lbl_dir = root / lbl_rel

            if not img_dir.exists():
                print(f"  [SKIP] {split}/images not found: {img_dir}")
                continue

            images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
            print(f"  {split}: {len(images)} images found")

            for img_path in images:
                # Check label exists
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    stats["orphan_images"] += 1
                    continue

                # Skip empty label files (background-only images — not useful for 3-class task)
                content = lbl_path.read_text().strip()
                if not content:
                    stats["empty_labels"] += 1
                    continue

                # Dedup by file hash
                h = file_hash(img_path)
                if h in seen_hashes:
                    duplicates += 1
                    stats["duplicates"] += 1
                    continue
                seen_hashes[h] = img_path.name

                # Build output filename: {source}_{split}_{original_stem}{ext}
                new_stem = f"{name}_{split}_{img_path.stem}"
                new_img = out_img / (new_stem + img_path.suffix.lower())
                new_lbl = out_lbl / (new_stem + ".txt")

                # Handle filename collision (shouldn't happen with prefix, but be safe)
                if new_img.exists():
                    new_stem += "_x"
                    new_img = out_img / (new_stem + img_path.suffix.lower())
                    new_lbl = out_lbl / (new_stem + ".txt")
                    conflicts += 1

                shutil.copy2(img_path, new_img)
                shutil.copy2(lbl_path, new_lbl)

                # Count classes in this label
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1

                stats["copied"] += 1

    # ── Write data.yaml ───────────────────────────────────────────────────────
    yaml_content = f"""# Merged dataset: techcare (Fall_sit_walk_bbox) + swook (better_Fall_sit_walk)
# Total training images: {stats['copied']}
# Classes: {CLASS_NAMES}

path: {OUTPUT.as_posix()}
train: train/images
val: train/images   # YOLO will auto-split via fraction; override with your val set if needed

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (OUTPUT / "data.yaml").write_text(yaml_content)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("MERGE COMPLETE")
    print("="*55)
    print(f"  Output folder  : {OUTPUT}")
    print(f"  Images copied  : {stats['copied']}")
    print(f"  Duplicates skip: {duplicates}")
    print(f"  Orphan images  : {stats['orphan_images']}  (no matching label)")
    print(f"  Empty labels   : {stats['empty_labels']}  (skipped)")
    print(f"  Name conflicts : {conflicts}")
    print()
    print("Class distribution (annotations):")
    total_ann = sum(class_counts.values())
    for cid, cname in enumerate(CLASS_NAMES):
        count = class_counts.get(cid, 0)
        pct = count / total_ann * 100 if total_ann else 0
        bar = "█" * int(pct / 2)
        print(f"  [{cid}] {cname:<6}: {count:>6} annotations  ({pct:5.1f}%)  {bar}")
    print(f"  Total          : {total_ann} annotations")
    print()

    # Imbalance warning
    counts = [class_counts.get(i, 0) for i in range(len(CLASS_NAMES))]
    if max(counts) / max(min(counts), 1) > 3:
        dominant = CLASS_NAMES[counts.index(max(counts))]
        weakest  = CLASS_NAMES[counts.index(min(counts))]
        ratio    = max(counts) / max(min(counts), 1)
        print(f"  ⚠  Class imbalance: {dominant} is {ratio:.1f}x more than {weakest}")
        print(f"     → Training will use class_weights to compensate.")
    else:
        print("  ✅ Class balance looks healthy.")

    print("\nNext step:")
    print("  python src/train_onestep.py --data datasets/merged_dataset/data.yaml")

if __name__ == "__main__":
    merge()
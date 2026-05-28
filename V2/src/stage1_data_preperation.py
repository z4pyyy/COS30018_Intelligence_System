"""
stage1_data_preperation.py
==========================
Merges all person-detection datasets into a single nc:1 dataset for Stage 1 training.

Sources:
  1. PPP_closeup        — nc:1, class 0 = person (no remapping needed)
  2. pukyung_low_light  — nc:3, class 2 = person → remap to 0, drop bicycle/car
  3. swook_techcare     — nc:3, Fall/Sit/Walk → remap ALL to 0 (person)
  4. person_hard_neg    — raw images, no labels → generate empty label files

Output:
  datasets/stage1_person/
  ├── train/images/
  ├── train/labels/
  └── data.yaml  (nc:1, names:['person'])

Usage:
  cd C:\\z4pyyy\\Swinburne_Y2S2\\Intel_System\\COS30018_Fall-Detection
  python src/stage1_data_preperation.py
"""

import shutil
import hashlib
from pathlib import Path
from collections import Counter

# =============================================================================
#  PATHS
# =============================================================================

BASE   = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V2\dataset_stage1")
OUTPUT = BASE.parent / "datasets" / "stage1_person"

SOURCES = [
    {
        "name":         "ppp_closeup",
        "root":         BASE / "PPP_closeup",
        "splits":       ["train", "valid", "test"],
        "mode":         "person_only",
        "person_class": 0,
    },
    {
        "name":         "pukyung",
        "root":         BASE / "pukyung_low_light",
        "splits":       ["train", "valid", "test"],
        "mode":         "remap",         # nc:3, person is class 2
        "person_class": 2,               # only keep lines where class == 2
    },
    {
        "name":         "swook_techcare",
        "root":         BASE / "swook_techcare",
        "splits":       ["train"],       # only has train/
        "mode":         "all_to_person", # all classes → 0 (person)
        "person_class": None,            # remap everything regardless of class id
    },
    {
        "name":         "hard_neg",
        "root":         BASE / "person_hard_neg",
        "splits":       None,            # flat folder, no split structure
        "mode":         "hard_neg",      # raw images → empty label files
        "person_class": None,
    },
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================================================
#  HELPERS
# =============================================================================

def file_hash(path: Path) -> str:
    """MD5 of first 64KB for fast near-duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def remap_label(content: str, mode: str, person_class: int) -> str:
    """
    Transform label file content based on mode:
      person_only   → keep lines where class == person_class, rewrite as class 0
      remap         → same as person_only (drop non-person classes)
      all_to_person → rewrite ALL lines to class 0 regardless of original class
    Returns empty string if no valid lines remain.
    """
    out_lines = []
    for line in content.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])

        if mode == "all_to_person":
            parts[0] = "0"
            out_lines.append(" ".join(parts))

        elif mode in ("person_only", "remap"):
            if cls_id == person_class:
                parts[0] = "0"
                out_lines.append(" ".join(parts))
            # else: drop this annotation (it's bicycle/car/etc.)

    return "\n".join(out_lines)


def find_images(folder: Path):
    """Return all image files in a folder (non-recursive)."""
    return [f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in IMG_EXTS]


def find_label(img_path: Path, label_dir: Path) -> Path | None:
    """Return matching label file for an image, or None."""
    lbl = label_dir / (img_path.stem + ".txt")
    return lbl if lbl.exists() else None

# =============================================================================
#  MAIN MERGE
# =============================================================================

def merge():
    out_img = OUTPUT / "train" / "images"
    out_lbl = OUTPUT / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    seen_hashes = {}   # hash → first filename (dedup)
    stats       = Counter()

    for src in SOURCES:
        name         = src["name"]
        root         = src["root"]
        mode         = src["mode"]
        person_class = src["person_class"]
        splits       = src["splits"]

        if not root.exists():
            print(f"\n  [SKIP] Folder not found: {root}")
            print(f"         Update the path in SOURCES for '{name}'")
            continue

        print(f"\n── {name}  ({root.name})  mode={mode}")

        # ── Hard negatives: flat folder, no labels ────────────────────────────
        if mode == "hard_neg":
            images = find_images(root)
            # also check one level deep (train/ subfolder)
            if not images:
                for sub in root.iterdir():
                    if sub.is_dir():
                        images += find_images(sub)

            print(f"  images found: {len(images)}")
            for img_path in images:
                h = file_hash(img_path)
                if h in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes[h] = img_path.name

                new_stem = f"hardneg_{img_path.stem}"
                new_img  = out_img / (new_stem + img_path.suffix.lower())
                new_lbl  = out_lbl / (new_stem + ".txt")

                shutil.copy2(img_path, new_img)
                new_lbl.write_text("")   # empty label = background only
                stats["hard_neg"] += 1
                stats["copied"]   += 1
            continue

        # ── Normal datasets with split folders ───────────────────────────────
        for split in (splits or []):
            # Try both common structures
            img_dir = root / split / "images"
            if not img_dir.exists():
                img_dir = root / "images" / split
            if not img_dir.exists():
                img_dir = root / split   # flat split folder
            if not img_dir.exists():
                print(f"  [SKIP] {split}/images not found")
                continue

            # Label directory
            if "images" in str(img_dir):
                lbl_dir = img_dir.parent.parent / "labels" / split
                if not lbl_dir.exists():
                    lbl_dir = img_dir.parent / "labels"
            else:
                lbl_dir = root / split / "labels"

            images = find_images(img_dir)
            print(f"  {split}: {len(images)} images  (labels: {lbl_dir.name if lbl_dir.exists() else 'NOT FOUND'})")

            for img_path in images:
                # Dedup
                h = file_hash(img_path)
                if h in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes[h] = img_path.name

                # Find label
                lbl_path = find_label(img_path, lbl_dir) if lbl_dir.exists() else None

                # Skip images with no label file for non-hard-neg sources
                # (they have no annotations to learn from)
                if lbl_path is None:
                    stats["orphan"] += 1
                    continue

                raw_content  = lbl_path.read_text().strip()
                new_content  = remap_label(raw_content, mode, person_class)

                # Skip if remapping eliminated all annotations
                # (e.g. pukyung image with only bicycle/car, no person)
                if not new_content and mode in ("person_only", "remap"):
                    stats["no_person_in_image"] += 1
                    continue

                # Build output filename with source prefix to avoid collisions
                new_stem = f"{name}_{split}_{img_path.stem}"
                new_img  = out_img / (new_stem + img_path.suffix.lower())
                new_lbl  = out_lbl / (new_stem + ".txt")

                shutil.copy2(img_path, new_img)
                new_lbl.write_text(new_content)
                stats["copied"] += 1

    # ── Write data.yaml ───────────────────────────────────────────────────────
    yaml = f"""# Stage 1 Person Detector — merged dataset
# Sources: coco_person, ppp_closeup, pukyung (person class only),
#          swook_techcare (all classes → person), hard_neg (empty labels)

path: {OUTPUT.as_posix()}
train: train/images
val:   train/images   # YOLO auto-splits 15% for validation

nc: 1
names: ['person']
"""
    (OUTPUT / "data.yaml").write_text(yaml, encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("MERGE COMPLETE")
    print("=" * 58)
    print(f"  Output            : {OUTPUT}")
    print(f"  Images copied     : {stats['copied']}")
    print(f"    of which hard-neg : {stats['hard_neg']}")
    print(f"  Duplicates skipped: {stats['duplicates']}")
    print(f"  Orphan (no label) : {stats['orphan']}")
    print(f"  No person in image: {stats['no_person_in_image']}  (pukyung bicycle/car only)")
    print()

    # Quick class-distribution check
    lbl_dir = OUTPUT / "train" / "labels"
    all_classes   = Counter()
    empty_count   = 0
    for lbl in lbl_dir.glob("*.txt"):
        content = lbl.read_text().strip()
        if not content:
            empty_count += 1
        else:
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    all_classes[parts[0]] += 1

    print(f"  Label verification:")
    print(f"    Class 0 (person) annotations: {all_classes.get('0', 0)}")
    print(f"    Empty labels (hard-neg)      : {empty_count}")
    if any(k != "0" for k in all_classes):
        print(f"  WARNING: unexpected class IDs found: "
              f"{[k for k in all_classes if k != '0']}")
        print(f"  Check the remap logic above.")
    else:
        print(f"  All annotations are class 0 — remapping successful")

    hard_pct = empty_count / max(stats["copied"], 1) * 100
    print(f"\n  Hard-neg ratio    : {hard_pct:.1f}%  "
          f"({'OK' if hard_pct < 22 else 'HIGH — consider reducing'})")
    print(f"\nNext step:")
    print(f"  python src/train_stage1.py --data datasets/stage1_person/data.yaml")


if __name__ == "__main__":
    merge()
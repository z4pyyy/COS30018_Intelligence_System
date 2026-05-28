import os
import sys
from pathlib import Path
from datetime import datetime


WRONG_DIR = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\final_version\evaluation\kaggleImage_fall_sit_walk\images\wrong_all_pipelines")
DATASET_FALL_DIR = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\final_version\dataset\fall_sit_walk_kaggle\fall")


def mark_photo(path: Path):
    # create a small marker file next to the image to indicate it was processed
    marker = path.with_name(path.name + ".marked")
    try:
        with marker.open("a", encoding="utf-8") as f:
            f.write(f"marked: {datetime.utcnow().isoformat()}Z\n")
    except Exception:
        pass


def delete_matching_in_dataset(name: str) -> int:
    """Delete files that match the given filename (by basename) under DATASET_FALL_DIR.
    Returns number of deleted files."""
    deleted = 0
    if not DATASET_FALL_DIR.exists():
        return deleted
    for root, _, files in os.walk(DATASET_FALL_DIR):
        for f in files:
            if f == name:
                p = Path(root) / f
                try:
                    p.unlink()
                    deleted += 1
                except Exception:
                    pass
    return deleted


def main():
    if not WRONG_DIR.exists():
        print(f"Wrong-images directory not found: {WRONG_DIR}")
        sys.exit(1)

    total = 0
    total_deleted = 0
    for root, _, files in os.walk(WRONG_DIR):
        for f in files:
            p = Path(root) / f
            # skip marker files if rerun
            if p.suffix == ".marked":
                continue
            mark_photo(p)
            deleted = delete_matching_in_dataset(p.name)
            total += 1
            total_deleted += deleted
            print(f"Processed {p} -> deleted {deleted} matching file(s) in dataset")

    print(f"Done. Processed {total} wrong images. Total deleted files in dataset: {total_deleted}.")


if __name__ == "__main__":
    main()

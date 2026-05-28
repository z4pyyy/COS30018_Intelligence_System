"""
evaluate_kaggle.py  (V3)
=========================
Evaluates four pipeline configurations on the Kaggle fall detection dataset
to validate model performance on completely unseen external data.

Kaggle dataset format:
  images/train/*.jpg  — all images mixed, no subfolders
  images/val/*.jpg
  labels/train/*.txt  — YOLO format: class cx cy w h (normalised)
  labels/val/*.txt    — class 0 = fallen, class 1 = not fallen

Five pipelines evaluated:
    Pipeline 1 — Raw CNN                  (full image → CNN, zero localisation)
    Pipeline 2 — GT box → CNN crop        (Stage 2 in isolation, perfect boxes)
    Pipeline 3 — One-step YOLO            (swook_techcare_fall_sit_walk.pt)
    Pipeline 4 — Stage1 person.pt → CNN   (full two-step pipeline)
    Pipeline 5 — Stage1 person.pt → YOLO  (Stage1 crop → one-step YOLO)

Output format per pipeline:
  GT Class      Total   Fall   Sit   Walk   Correct   Acc
  fallen          X       X     X     X     Fall         %
  not fallen      X       X     X     X     Sit+Walk     %

"Correct" definition:
  fallen     → CNN predicts Fall
  not fallen → CNN predicts Sit OR Walk (either is acceptable)

Saved image folders per pipeline:
  fallen_correct/       gt=fallen,    pred=Fall        ✓
  fallen_wrong/         gt=fallen,    pred=Sit or Walk  ✗
  notfallen_correct/    gt=notfallen, pred=Sit or Walk  ✓
  notfallen_falsealarm/ gt=notfallen, pred=Fall         ✗  ← key diagnostic

Each saved image has an overlay banner:
  Top:    GT: fallen | PRED: Fall (85%)
  Bottom: method: cnn / geo_rule / gt_box+cnn / onestep

Usage:
    python V3/src/evaluate_kaggle.py
"""

import csv
import sys
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
#  PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection")
V3_BASE      = PROJECT_ROOT / "V3"
V2_BASE      = PROJECT_ROOT / "V2"

KAGGLE_DIR   = PROJECT_ROOT / "fall_dataset_kaggle"
OUT_DIR      = V3_BASE / "runs" / "evaluation" / "kaggle"

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kaggle class ids
KAGGLE_FALLEN     = 0
KAGGLE_NOT_FALLEN = 1

# Colour palette for overlay (BGR)
COLOUR = {
    "Fall":        (0,   0,   220),
    "Sit":         (0,   180, 180),
    "Walk":        (0,   200,   0),
    "NoDetection": (128, 128, 128),
    "fallen":      (0,   0,   220),
    "not_fallen":  (0,   200,   0),
}

# Sorted output subfolder per (gt_label, outcome)
SORT_FOLDER = {
    ("fallen",     "correct"):    "fallen_correct",
    ("fallen",     "wrong"):      "fallen_wrong",
    ("not_fallen", "correct"):    "notfallen_correct",
    ("not_fallen", "falsealarm"): "notfallen_falsealarm",
}

PIPELINE_DIRS = {
    "Pipeline 1 — Raw CNN":       "pipeline1_raw_cnn",
    "Pipeline 2 — GT box → CNN":  "pipeline2_gtbox_cnn",
    "Pipeline 3 — One-step YOLO": "pipeline3_onestep",
    "Pipeline 4 — Stage1 + CNN":  "pipeline4_twostep",
    "Pipeline 5 — Stage1 + YOLO": "pipeline5_stage1_yolo",
}

# =============================================================================
#  IMAGE SAVER
# =============================================================================

def get_outcome(gt_class_int: int, pred: str) -> tuple[str, str]:
    """
    Returns (gt_label, outcome).
    gt_label : "fallen" | "not_fallen"
    outcome  : "correct" | "wrong" | "falsealarm"
    """
    gt_label = "fallen" if gt_class_int == KAGGLE_FALLEN else "not_fallen"
    if gt_label == "fallen":
        outcome = "correct" if pred == "Fall" else "wrong"
    else:
        outcome = "correct" if pred in ("Sit", "Walk") else "falsealarm"
    return gt_label, outcome


def save_result_image(img: np.ndarray, img_name: str,
                      gt_class_int: int, pred: str, conf: float, method: str,
                      pipeline_out_dir: Path,
                      bbox: tuple | None = None):
    """
    Draws overlay banner + bbox on image and saves to sorted subfolder.

    Top banner  : GT: fallen/not_fallen  |  PRED: Fall/Sit/Walk (xx%)
    Bbox        : drawn if provided
    Bottom strip: method used
    """
    img_draw = img.copy()
    h, w = img_draw.shape[:2]

    gt_label, outcome = get_outcome(gt_class_int, pred)
    folder_name = SORT_FOLDER.get((gt_label, outcome), "other")
    save_dir = pipeline_out_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_draw, (x1, y1), (x2, y2),
                      COLOUR.get(pred, (200, 200, 200)), 2)

    # Top banner
    banner_h = 44
    banner   = np.zeros((banner_h, w, 3), dtype=np.uint8)
    gt_display = gt_label.replace("_", " ")
    cv2.putText(banner, f"GT: {gt_display}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLOUR.get(gt_label, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.putText(banner, f"PRED: {pred}  ({conf:.0%})",
                (w // 2, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLOUR.get(pred, (255, 255, 255)), 2, cv2.LINE_AA)
    img_draw = np.vstack([banner, img_draw])

    # Bottom method strip
    strip = np.zeros((26, w, 3), dtype=np.uint8)
    cv2.putText(strip, f"method: {method}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (190, 190, 190), 1, cv2.LINE_AA)
    img_draw = np.vstack([img_draw, strip])

    cv2.imwrite(str(save_dir / img_name), img_draw)

# =============================================================================
#  LABEL PARSING
# =============================================================================

def load_labels(label_path: Path) -> list[dict]:
    """
    Parse YOLO label file.
    Returns list of {class_id, cx, cy, w, h} dicts.
    Empty list if file missing or empty.
    """
    if not label_path.exists():
        return []
    entries = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                entries.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w":  float(parts[3]),
                    "h":  float(parts[4]),
                })
    return entries


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h) -> tuple[int, int, int, int]:
    """Convert normalised YOLO cx/cy/w/h to pixel x1/y1/x2/y2."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (
        max(0, x1), max(0, y1),
        min(img_w, x2), min(img_h, y2),
    )


def get_gt_class(label_entries: list[dict]) -> int | None:
    """
    Returns the dominant GT class for an image.
    If multiple labels exist, take the fallen (0) label if any — conservative.
    Returns None if no labels.
    """
    if not label_entries:
        return None
    class_ids = [e["class_id"] for e in label_entries]
    # If any annotation is fallen, treat image as fallen
    if KAGGLE_FALLEN in class_ids:
        return KAGGLE_FALLEN
    return KAGGLE_NOT_FALLEN


def get_largest_gt_box(label_entries: list[dict], img_w: int, img_h: int):
    """Returns xyxy of the largest annotated box (by area)."""
    if not label_entries:
        return None
    best = max(label_entries, key=lambda e: e["w"] * e["h"])
    return yolo_to_xyxy(best["cx"], best["cy"], best["w"], best["h"], img_w, img_h)


# =============================================================================
#  PIPELINE 1 — Raw CNN (full image, no localisation)
# =============================================================================

def run_raw_cnn(img: np.ndarray,
                cnn_model, class_names, transform, cnn_device) -> tuple[str, float]:
    """CNN on full image resized to 224×224. No bounding box at all.
    Returns (pred_class, confidence)."""
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import predict_crop
    cls_name, conf, _ = predict_crop(cnn_model, class_names, transform, cnn_device, img)
    return cls_name, conf


# =============================================================================
#  PIPELINE 2 — GT box → CNN
# =============================================================================

def run_gtbox_cnn(img: np.ndarray, label_entries: list[dict],
                  cnn_model, class_names, transform, cnn_device) -> tuple[str, float, tuple | None]:
    """Stage 2 CNN on GT bounding box crop. No Stage 1 involved.
    Returns (pred_class, confidence, raw_gt_bbox_xyxy)."""
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import pad_bbox, predict_crop

    h, w = img.shape[:2]
    raw_bbox = get_largest_gt_box(label_entries, w, h)
    if raw_bbox is None:
        return "NoDetection", 0.0, None

    x1, y1, x2, y2 = pad_bbox(*raw_bbox, w, h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "NoDetection", 0.0, raw_bbox

    cls_name, conf, _ = predict_crop(cnn_model, class_names, transform, cnn_device, crop)
    return cls_name, conf, raw_bbox


# =============================================================================
#  PIPELINE 3 — One-step YOLO
# =============================================================================

def run_onestep(img: np.ndarray, onestep_model) -> tuple[str, float, tuple | None]:
    """One-step YOLOv8 fall/sit/walk classifier.
    Returns (pred_class, confidence, bbox_xyxy)."""
    r = onestep_model.predict(
        source=img, imgsz=640, conf=0.25,
        device=0, verbose=False,
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return "NoDetection", 0.0, None

    confs  = r.boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    cls_id = int(r.boxes.cls[best_i].cpu().numpy())
    conf   = float(confs[best_i])
    bbox   = tuple(map(int, r.boxes.xyxy[best_i].cpu().numpy()))

    cls_name = r.names.get(cls_id, f"cls{cls_id}")
    name_map = {"fall": "Fall", "sit": "Sit", "walk": "Walk"}
    cls_name = name_map.get(cls_name.lower(), cls_name)

    return cls_name, conf, bbox


# =============================================================================
#  PIPELINE 4 — Stage1 person → CNN
# =============================================================================

def run_twostep(img: np.ndarray, stage1_model,
                cnn_model, class_names, transform, cnn_device,
                geo_rule) -> tuple[str, float, str, tuple | None]:
    """Full two-step pipeline: Stage 1 detect → adaptive pad → CNN classify.
    Returns (pred_class, confidence, method, detected_bbox_xyxy)."""
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import pad_bbox, predict_crop, STAGE1_CONF

    h, w = img.shape[:2]

    r = stage1_model.predict(
        source=img, conf=STAGE1_CONF,
        imgsz=640, device=0, verbose=False,
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return "NoDetection", 0.0, "no_person", None

    areas  = [(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.cpu().numpy()]
    best_i = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[best_i].cpu().numpy())
    raw_bbox = (x1, y1, x2, y2)
    x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "NoDetection", 0.0, "empty_crop", raw_bbox

    if geo_rule.check(bbox=(x1, y1, x2, y2)):
        return "Fall", 0.80, "geo_rule", raw_bbox

    cls_name, conf, _ = predict_crop(cnn_model, class_names, transform, cnn_device, crop)
    return cls_name, conf, "cnn", raw_bbox


# =============================================================================
#  PIPELINE 5 — Stage1 person → One-step YOLO
# =============================================================================

def run_stage1_yolo(img: np.ndarray, stage1_model, onestep_model,
                    geo_rule) -> tuple[str, float, str, tuple | None]:
    """Stage1 detect person crop → one-step YOLO classifies crop.
    Returns (pred_class, confidence, method, detected_bbox_xyxy)."""
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import pad_bbox, STAGE1_CONF

    h, w = img.shape[:2]

    r = stage1_model.predict(
        source=img, conf=STAGE1_CONF, imgsz=640, device=0, verbose=False,
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return "NoDetection", 0.0, "no_person", None

    areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in r.boxes.xyxy.cpu().numpy()]
    best_i = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[best_i].cpu().numpy())
    raw_bbox = (x1, y1, x2, y2)
    x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "NoDetection", 0.0, "empty_crop", raw_bbox

    if geo_rule.check(bbox=(x1, y1, x2, y2)):
        return "Fall", 0.80, "geo_rule", raw_bbox

    r2 = onestep_model.predict(
        source=crop, imgsz=640, conf=0.25, device=0, verbose=False,
    )[0]
    if r2.boxes is None or len(r2.boxes) == 0:
        return "NoDetection", 0.0, "yolo_no_detect_on_crop", raw_bbox

    confs2  = r2.boxes.conf.cpu().numpy()
    best_i2 = int(np.argmax(confs2))
    cls_id2 = int(r2.boxes.cls[best_i2].cpu().numpy())
    conf2   = float(confs2[best_i2])
    cls_name = r2.names.get(cls_id2, f"cls{cls_id2}")
    name_map = {"fall": "Fall", "sit": "Sit", "walk": "Walk"}
    return name_map.get(cls_name.lower(), cls_name), conf2, "stage1+yolo", raw_bbox


# =============================================================================
#  COLLECT IMAGES
# =============================================================================

def collect_image_label_pairs(split: str) -> list[dict]:
    """
    Returns list of {img_path, label_path, filename} for a split (train/val).
    Only includes images that have a corresponding label file.
    """
    img_dir   = KAGGLE_DIR / "images" / split
    label_dir = KAGGLE_DIR / "labels" / split

    if not img_dir.exists():
        print(f"  [SKIP] {img_dir} not found")
        return []

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = label_dir / (img_path.stem + ".txt")
        pairs.append({
            "img_path":   img_path,
            "label_path": label_path,
            "filename":   img_path.name,
        })
    return pairs


# =============================================================================
#  PRINT RESULTS
# =============================================================================

def print_pipeline_results(pipeline_name: str, results: list[dict]):
    """
    Prints the per-GT-class breakdown table.

    GT Class    Total   Fall   Sit   Walk   NoDetect   Correct    Acc
    fallen        X       X     X     X        X       Fall cnt    %
    not fallen    X       X     X     X        X      Sit+Walk cnt %
    """
    fallen_r     = [r for r in results if r["gt_class"] == KAGGLE_FALLEN]
    not_fallen_r = [r for r in results if r["gt_class"] == KAGGLE_NOT_FALLEN]

    print(f"\n{'─' * 75}")
    print(f"  {pipeline_name}")
    print(f"{'─' * 75}")
    print(f"  {'GT Class':<14} {'Total':>6} {'Fall':>6} {'Sit':>6} "
          f"{'Walk':>6} {'NoDet':>6} {'Correct':>8} {'Acc':>7}")
    print(f"  {'-' * 68}")

    total_correct = 0
    total_all     = 0

    for gt_label, group, correct_preds in [
        ("fallen",     fallen_r,     ["Fall"]),
        ("not fallen", not_fallen_r, ["Sit", "Walk"]),
    ]:
        total   = len(group)
        fall_n  = sum(1 for r in group if r["pred"] == "Fall")
        sit_n   = sum(1 for r in group if r["pred"] == "Sit")
        walk_n  = sum(1 for r in group if r["pred"] == "Walk")
        nodet_n = sum(1 for r in group if r["pred"] == "NoDetection")
        correct = sum(1 for r in group if r["pred"] in correct_preds)
        acc     = correct / total if total > 0 else 0.0

        total_correct += correct
        total_all     += total

        print(f"  {gt_label:<14} {total:>6} {fall_n:>6} {sit_n:>6} "
              f"{walk_n:>6} {nodet_n:>6} {correct:>8} {acc:>7.1%}")

    overall_acc = total_correct / total_all if total_all > 0 else 0.0
    print(f"  {'─' * 68}")
    print(f"  {'TOTAL':<14} {total_all:>6} "
          f"{'':>6} {'':>6} {'':>6} {'':>6} "
          f"{total_correct:>8} {overall_acc:>7.1%}")

    # Sit vs Walk breakdown for not-fallen (the diagnostic question)
    if not_fallen_r:
        sit_pct  = sum(1 for r in not_fallen_r if r["pred"] == "Sit")  / len(not_fallen_r)
        walk_pct = sum(1 for r in not_fallen_r if r["pred"] == "Walk") / len(not_fallen_r)
        fall_pct = sum(1 for r in not_fallen_r if r["pred"] == "Fall") / len(not_fallen_r)
        print(f"\n  Not-fallen breakdown: "
              f"Sit {sit_pct:.1%}  Walk {walk_pct:.1%}  "
              f"Fall (false alarm) {fall_pct:.1%}")

    return total_correct, total_all


def print_comparison_table(pipeline_results: dict):
    """Final side-by-side comparison of all three pipelines."""
    print(f"\n{'=' * 75}")
    print(f"  KAGGLE DATASET — FIVE PIPELINE COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Pipeline':<35} {'fallen Acc':>11} {'not-fallen Acc':>15} {'Overall':>9}")
    print(f"  {'-' * 73}")

    for name, results in pipeline_results.items():
        fallen_r     = [r for r in results if r["gt_class"] == KAGGLE_FALLEN]
        not_fallen_r = [r for r in results if r["gt_class"] == KAGGLE_NOT_FALLEN]

        fall_acc = (sum(1 for r in fallen_r if r["pred"] == "Fall") / len(fallen_r)
                    if fallen_r else 0.0)
        nf_acc   = (sum(1 for r in not_fallen_r if r["pred"] in ["Sit", "Walk"]) / len(not_fallen_r)
                    if not_fallen_r else 0.0)
        overall  = ((sum(1 for r in fallen_r if r["pred"] == "Fall") +
                     sum(1 for r in not_fallen_r if r["pred"] in ["Sit", "Walk"])) /
                    len(results) if results else 0.0)

        print(f"  {name:<35} {fall_acc:>11.1%} {nf_acc:>15.1%} {overall:>9.1%}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import load_models, GeometricFallRule

    print("=" * 75)
    print("  COS30018 — Kaggle Dataset Evaluation (Five Pipelines)")
    print("=" * 75)
    print(f"  Dataset : {KAGGLE_DIR}")
    print(f"  Output  : {OUT_DIR}")

    # ── Create output folder structure upfront ────────────────────────────────
    # Clear previous run so stale images don't accumulate
    img_out = OUT_DIR / "images"
    csv_out = OUT_DIR / "csvs"
    if img_out.exists():
        shutil.rmtree(img_out)

    subfolders = list(SORT_FOLDER.values())  # 4 sorted subfolders
    for pipeline_dir in PIPELINE_DIRS.values():
        for sub in subfolders:
            (img_out / pipeline_dir / sub).mkdir(parents=True, exist_ok=True)
    csv_out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output structure:")
    for pipeline_dir in PIPELINE_DIRS.values():
        print(f"    images/{pipeline_dir}/")
        for sub in subfolders:
            print(f"      {sub}/")

    # ── Load all models once ──────────────────────────────────────────────────
    print("\nLoading models...")
    m        = load_models(two_step=True)
    geo_rule = GeometricFallRule(aspect_threshold=2.2)
    print("  [OK] All models loaded")

    # ── Collect images from both splits ───────────────────────────────────────
    all_pairs = []
    for split in ["train", "val"]:
        pairs = collect_image_label_pairs(split)
        all_pairs.extend(pairs)
        print(f"  {split}: {len(pairs)} images")
    print(f"  Total  : {len(all_pairs)} images")

    if not all_pairs:
        print("\n  [ERROR] No images found. Check KAGGLE_DIR path.")
        return

    # ── Pipeline result stores ────────────────────────────────────────────────
    pipeline_results = {
        "Pipeline 1 — Raw CNN":       [],
        "Pipeline 2 — GT box → CNN":  [],
        "Pipeline 3 — One-step YOLO": [],
        "Pipeline 4 — Stage1 + CNN":  [],
        "Pipeline 5 — Stage1 + YOLO": [],
    }

    # Map pipeline name → output folder path
    pipeline_out_dirs = {
        name: img_out / PIPELINE_DIRS[name]
        for name in pipeline_results
    }

    no_label_count = 0

    print(f"\nRunning evaluation on {len(all_pairs)} images...")
    for i, pair in enumerate(all_pairs):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_pairs)}...")

        img = cv2.imread(str(pair["img_path"]))
        if img is None:
            continue

        labels   = load_labels(pair["label_path"])
        gt_class = get_gt_class(labels)

        if gt_class is None:
            no_label_count += 1
            continue

        fname = pair["filename"]

        # ── Pipeline 1: Raw CNN (full image, no localisation) ─────────────
        pred1, conf1 = run_raw_cnn(
            img, m["cnn"], m["class_names"], m["transform"], m["cnn_device"],
        )
        pipeline_results["Pipeline 1 — Raw CNN"].append({
            "filename": fname, "gt_class": gt_class,
            "pred": pred1, "conf": conf1, "method": "raw_cnn",
        })
        save_result_image(img, fname, gt_class, pred1, conf1,
                          "raw_cnn", pipeline_out_dirs["Pipeline 1 — Raw CNN"])

        # ── Pipeline 2: GT box → CNN ──────────────────────────────────────
        pred2, conf2, bbox2 = run_gtbox_cnn(
            img, labels,
            m["cnn"], m["class_names"], m["transform"], m["cnn_device"],
        )
        pipeline_results["Pipeline 2 — GT box → CNN"].append({
            "filename": fname, "gt_class": gt_class,
            "pred": pred2, "conf": conf2, "method": "gt_box+cnn",
        })
        save_result_image(img, fname, gt_class, pred2, conf2,
                          "gt_box+cnn", pipeline_out_dirs["Pipeline 2 — GT box → CNN"],
                          bbox=bbox2)

        # ── Pipeline 3: One-step YOLO ─────────────────────────────────────
        pred3, conf3, bbox3 = run_onestep(img, m["onestep"])
        pipeline_results["Pipeline 3 — One-step YOLO"].append({
            "filename": fname, "gt_class": gt_class,
            "pred": pred3, "conf": conf3, "method": "onestep",
        })
        save_result_image(img, fname, gt_class, pred3, conf3,
                          "onestep", pipeline_out_dirs["Pipeline 3 — One-step YOLO"],
                          bbox=bbox3)

        # ── Pipeline 4: Stage1 + CNN ──────────────────────────────────────
        pred4, conf4, method4, bbox4 = run_twostep(
            img, m["stage1"],
            m["cnn"], m["class_names"], m["transform"], m["cnn_device"],
            geo_rule,
        )
        pipeline_results["Pipeline 4 — Stage1 + CNN"].append({
            "filename": fname, "gt_class": gt_class,
            "pred": pred4, "conf": conf4, "method": method4,
        })
        save_result_image(img, fname, gt_class, pred4, conf4,
                          method4, pipeline_out_dirs["Pipeline 4 — Stage1 + CNN"],
                          bbox=bbox4)

        # ── Pipeline 5: Stage1 + YOLO ─────────────────────────────────────
        pred5, conf5, method5, bbox5 = run_stage1_yolo(
            img, m["stage1"], m["onestep"], geo_rule,
        )
        pipeline_results["Pipeline 5 — Stage1 + YOLO"].append({
            "filename": fname, "gt_class": gt_class,
            "pred": pred5, "conf": conf5, "method": method5,
        })
        save_result_image(img, fname, gt_class, pred5, conf5,
                          method5, pipeline_out_dirs["Pipeline 5 — Stage1 + YOLO"],
                          bbox=bbox5)

    if no_label_count:
        print(f"  [INFO] {no_label_count} images skipped (no label file)")

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  RESULTS PER PIPELINE")
    print(f"{'=' * 75}")
    print(f"  Correct definition:")
    print(f"    fallen     → predicted Fall")
    print(f"    not fallen → predicted Sit OR Walk")

    for name, results in pipeline_results.items():
        print_pipeline_results(name, results)

    print_comparison_table(pipeline_results)

    # ── Image folder summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  SAVED IMAGE COUNTS")
    print(f"{'=' * 75}")
    for pipeline_dir_name in PIPELINE_DIRS.values():
        print(f"\n  {pipeline_dir_name}/")
        for sub in subfolders:
            folder = img_out / pipeline_dir_name / sub
            count  = len(list(folder.glob("*")))
            print(f"    {sub:<30} {count:>5} images")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    gt_label_map = {KAGGLE_FALLEN: "fallen", KAGGLE_NOT_FALLEN: "not_fallen"}

    for name, results in pipeline_results.items():
        safe_name = name.split("—")[1].strip().replace(" ", "_").replace("+", "plus").lower()
        csv_path  = csv_out / f"kaggle_{safe_name}.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["filename", "gt_class", "pred", "conf", "method"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    **r,
                    "gt_class": gt_label_map.get(r["gt_class"], r["gt_class"]),
                    "conf": f"{r['conf']:.4f}",
                })
        print(f"\n  CSV → {csv_path.name}")

    print(f"\n  All outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
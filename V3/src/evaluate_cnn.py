"""
evaluate_twostep.py  (V3)
==========================
Evaluates the V3 two-step pipeline on the test set.

V3 change: Stage 2 is EfficientNet-B0 CNN — no MediaPipe, no MLP.
Stage 1 detects person bbox → adaptive pad → CNN classifies crop directly.

Generates:
  - Per-class accuracy table
  - Confusion matrices (counts + normalised)
  - Comparison table vs V2 MLP pipeline
  - CSV prediction log
  - wrong_predictions/ folder — misclassified images organised by error type
    Structure: wrong_predictions/{True}_as_{Pred}/
    Each image saved with overlay showing true label, predicted label,
    confidence, method used, and bounding box drawn on it.

Usage:
    python V3/src/evaluate_twostep.py
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO

# =============================================================================
#  PATHS
# =============================================================================

V3_BASE  = Path(__file__).resolve().parent.parent
V2_BASE  = V3_BASE.parent / "V2"

TEST_DIR = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V2\dataset\fall_sit_walk_test_raw")
OUT_DIR  = V3_BASE / "runs" / "evaluation" / "twostep_cnn"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Colour palette for overlay text  (BGR)
COLOUR = {
    "Fall": (0,   0,   220),   # red
    "Sit":  (0,   180, 180),   # yellow
    "Walk": (0,   200,  0),    # green
    "NoDetection": (128, 128, 128),
}

# =============================================================================
#  PIPELINE
# =============================================================================

def run_pipeline(stage1_model, cnn_model, class_names, transform,
                 cnn_device, img_path, geo_rule):
    """
    Run full V3 two-step pipeline on a single image.
    Returns (pred_class, confidence, method, bbox_xyxy_or_None)
    bbox is returned so the wrong-prediction saver can draw it.
    """
    import sys
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import pad_bbox, STAGE1_CONF

    img = cv2.imread(str(img_path))
    if img is None:
        return "NoDetection", 0.0, "load_error", None

    h, w = img.shape[:2]

    # Stage 1 — person detection
    r = stage1_model.predict(
        source=img, conf=STAGE1_CONF,
        imgsz=640, device=0, verbose=False,
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return "NoDetection", 0.0, "no_person", None

    # Use largest person box
    areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in r.boxes.xyxy.cpu().numpy()]
    best_i = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, r.boxes.xyxy[best_i].cpu().numpy())
    raw_bbox = (x1, y1, x2, y2)

    # Adaptive padding
    x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return "NoDetection", 0.0, "empty_crop", raw_bbox

    # Geometric fall rule — wide bbox = likely prone
    if geo_rule.check(bbox=(x1, y1, x2, y2)):
        return "Fall", 0.80, "geo_rule", raw_bbox

    # Stage 2 — CNN classification
    from pipeline_utils import predict_crop
    cls_name, conf, _ = predict_crop(
        cnn_model, class_names, transform, cnn_device, crop
    )
    return cls_name, conf, "cnn", raw_bbox

# =============================================================================
#  WRONG PREDICTION IMAGE SAVER
# =============================================================================

def save_wrong_prediction(img_path, true_cls, pred_cls, conf, method,
                           bbox, wrong_dir):
    """
    Saves a copy of the misclassified image with a diagnostic overlay into:
        wrong_dir / {true_cls}_as_{pred_cls} / {original_filename}

    Overlay drawn on the image:
      - Bounding box (if bbox available)
      - Top banner:  TRUE: {true_cls}  |  PRED: {pred_cls}  ({conf:.0%})
      - Bottom line: method used
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w = img.shape[:2]

    # ── Draw detection box ────────────────────────────────────────────────
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        box_colour = COLOUR.get(pred_cls, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), box_colour, 2)

    # ── Top banner ────────────────────────────────────────────────────────
    banner_h = 40
    banner   = np.zeros((banner_h, w, 3), dtype=np.uint8)

    true_colour = COLOUR.get(true_cls, (255, 255, 255))
    pred_colour = COLOUR.get(pred_cls, (255, 255, 255))

    true_text = f"TRUE: {true_cls}"
    pred_text = f"PRED: {pred_cls}  ({conf:.0%})"

    cv2.putText(banner, true_text,  (8,  28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, true_colour, 2, cv2.LINE_AA)
    cv2.putText(banner, pred_text,  (w // 2, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, pred_colour, 2, cv2.LINE_AA)

    img = np.vstack([banner, img])

    # ── Bottom strip — method ─────────────────────────────────────────────
    strip_h = 24
    strip   = np.zeros((strip_h, w, 3), dtype=np.uint8)
    cv2.putText(strip, f"method: {method}",
                (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    img = np.vstack([img, strip])

    # ── Save ──────────────────────────────────────────────────────────────
    folder_name = f"{true_cls}_as_{pred_cls}"
    save_folder = wrong_dir / folder_name
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / img_path.name
    cv2.imwrite(str(save_path), img)

# =============================================================================
#  EVALUATION
# =============================================================================

def evaluate():
    import sys
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import (
        load_models, GeometricFallRule,
        CLASS_NAMES, FOLDER_TO_CLASS,
    )

    print("="*60)
    print("  COS30018 — V3 Two-Step Pipeline Evaluation")
    print("  Stage 2: EfficientNet-B0 CNN (no MediaPipe)")
    print("="*60)

    print("\nLoading models...")
    m = load_models(two_step=True)
    geo_rule = GeometricFallRule(aspect_threshold=1.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wrong_dir = OUT_DIR / "wrong_predictions"

    # Clear previous wrong-prediction saves so stale images don't accumulate
    if wrong_dir.exists():
        shutil.rmtree(wrong_dir)
    wrong_dir.mkdir(parents=True)

    results = []

    for folder_name, true_class_id in FOLDER_TO_CLASS.items():
        true_cls = CLASS_NAMES[true_class_id]
        folder   = Path(TEST_DIR) / folder_name

        if not folder.exists():
            print(f"  [SKIP] {folder}")
            continue

        images = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)
        print(f"\n  {true_cls}: {len(images)} images")

        for img_path in images:
            pred_cls, conf, method, bbox = run_pipeline(
                m["stage1"], m["cnn"], m["class_names"],
                m["transform"], m["cnn_device"],
                img_path, geo_rule,
            )

            correct = (pred_cls == true_cls)

            # ── Save wrong predictions with overlay ───────────────────────
            if not correct:
                save_wrong_prediction(
                    img_path, true_cls, pred_cls, conf, method,
                    bbox, wrong_dir,
                )

            results.append({
                "filename": img_path.name,
                "true":     true_cls,
                "pred":     pred_cls,
                "conf":     f"{conf:.4f}",
                "method":   method,
                "correct":  correct,
                "bbox":     str(bbox),
            })

    # ── Print results ─────────────────────────────────────────────────────
    print_results(results, CLASS_NAMES)

    # ── Confusion matrices ────────────────────────────────────────────────
    plot_confusion(results, ["Fall", "Sit", "Walk"], OUT_DIR)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "predictions_twostep_cnn.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV           → {csv_path}")

    # ── Wrong prediction summary ──────────────────────────────────────────
    wrong_counts = defaultdict(int)
    for r in results:
        if not r["correct"]:
            wrong_counts[f"{r['true']}_as_{r['pred']}"] += 1

    print(f"\n  Wrong predictions saved → {wrong_dir}")
    print(f"  {'Folder':<30} {'Count':>6}")
    print(f"  {'-'*38}")
    for folder_name, count in sorted(wrong_counts.items(), key=lambda x: -x[1]):
        print(f"  {folder_name:<30} {count:>6}")


# =============================================================================
#  PRINT RESULTS
# =============================================================================

def print_results(results, class_names):
    labels  = ["Fall", "Sit", "Walk"]
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_det  = sum(1 for r in results if r["pred"] == "NoDetection")

    print(f"\n{'='*70}")
    print(f"  V3 TWO-STEP PIPELINE (CNN Stage 2) RESULTS")
    print(f"{'='*70}")
    print(f"  Overall: {correct}/{total} = {correct/total:.1%}  "
          f"(NoDetection: {no_det})")

    print(f"\n  {'Class':<8} {'Total':>6} {'Correct':>8} {'Acc':>7}  "
          f"{'Misclassified As'}")
    print(f"  {'-'*65}")

    for cls in labels:
        cls_r     = [r for r in results if r["true"] == cls]
        correct_n = sum(1 for r in cls_r if r["pred"] == cls)
        total_n   = len(cls_r)
        acc       = correct_n / total_n if total_n > 0 else 0

        misclass = defaultdict(int)
        for r in cls_r:
            if r["pred"] != cls:
                misclass[r["pred"]] += 1
        mis_str = ", ".join(
            f"{k}:{v}" for k, v in
            sorted(misclass.items(), key=lambda x: -x[1])
        )
        print(f"  {cls:<8} {total_n:>6} {correct_n:>8} {acc:>7.1%}  {mis_str}")

    # Method breakdown
    method_counts = defaultdict(int)
    for r in results:
        method_counts[r["method"]] += 1
    print(f"\n  Method breakdown:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"    {method:<20} {count}")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON — V2 MLP vs V3 CNN")
    print(f"{'='*70}")
    print(f"""
  Metric          V2 MLP (pose)    V3 CNN (pixels)   Delta
  ─────────────────────────────────────────────────────────
  Overall         68.4%            {correct/total:.1%}            {"+" if correct/total > 0.684 else ""}{(correct/total - 0.684)*100:.1f}%
  NoDetect rate    9.8%            {no_det/total:.1%}
  Fall recall     36.6%            ???%  (see table above)
  Sit recall      77.5%            ???%
  Walk recall     78.6%            ???%

  GT Box eval (Stage 2 only — no Stage 1):
    V2 MLP:  44.2% Fall accuracy with perfect boxes
    V3 CNN:  run convert_and_evaluate_stage2.py to compare
""")


# =============================================================================
#  CONFUSION MATRIX
# =============================================================================

def plot_confusion(results, labels, out_dir):
    cm     = np.zeros((3, 3), dtype=int)
    no_det = 0

    for r in results:
        ti = labels.index(r["true"]) if r["true"] in labels else -1
        if r["pred"] in labels:
            pi = labels.index(r["pred"])
            if ti >= 0:
                cm[ti][pi] += 1
        else:
            no_det += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("V3 CNN Two-Step (Counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    cm_norm = cm.astype(float)
    for i in range(len(labels)):
        row_sum = cm_norm[i].sum()
        if row_sum > 0:
            cm_norm[i] /= row_sum

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("V3 CNN Two-Step (Normalized)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    title = f"V3 Two-Step CNN Pipeline\n(NoDetection: {no_det})"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    save_path = out_dir / "confusion_twostep_cnn.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix  → {save_path}")


if __name__ == "__main__":
    evaluate()
"""
evaluate_confusion.py  (V3)
======================================
Runs five pipelines on the IS_Image/ dataset and produces
3-class confusion matrices (Fall / Sit / Walk).

Input folder structure (flat — images directly in class folders):
  IS_Image/
  ├── Fall/   ← GT = Fall
  ├── Sit/    ← GT = Sit
  └── Walk/   ← GT = Walk

Five pipelines evaluated:
  P1: Raw CNN         — full image resized to 224×224 → CNN (zero localisation)
  P2: Stage1 + CNN    — Stage1 person crop → CNN (real-world two-step)
  P3: Raw YOLO        — full image → one-step YOLO (its own detection+classify)
  P4: Stage1 + YOLO   — Stage1 person crop → one-step YOLO
  P5: Stage1 + CNN v2 — (reserved, same as P2 for now)

Key comparisons:
  P1 vs P2 → how much does localisation help CNN?
  P3 vs P4 → does YOLO benefit from pre-cropping?
  P2 vs P3 → CNN pipeline vs YOLO pipeline, real-world conditions
  P2 vs P4 → same Stage1 localisation, CNN vs YOLO classifier

Output:
  V3/runs/evaluation/ISImage_confusion/
  ├── images/<pipeline_dir>/{correct,wrong}/*.jpg
  ├── confusion_*.png
  └── results_summary.txt

Usage:
    python V3/src/evaluate_confusion.py
"""

import csv
import sys
import shutil
from pathlib import Path
from collections import defaultdict

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

DATA_DIR     = PROJECT_ROOT / "IS_Image"
OUT_DIR      = V3_BASE / "runs" / "evaluation" / "ISImage_confusion"

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS       = ["Fall", "Sit", "Walk"]

FOLDER_TO_CLASS = {
    "Fall": "Fall",
    "Sit":  "Sit",
    "Walk": "Walk",
}

COLOUR = {
    "Fall":        (0,   0,   220),
    "Sit":         (0,   180, 180),
    "Walk":        (0,   200,   0),
    "NoDetection": (128, 128, 128),
}

SORT_SUBFOLDERS = ["correct", "wrong"]

PIPELINE_DIRS = {
    "P1 — Raw CNN":        "p1_raw_cnn",
    "P2 — Stage1 + CNN":   "p2_stage1_cnn",
    "P3 — Raw YOLO":       "p3_raw_yolo",
    "P4 — Stage1 + YOLO":  "p4_stage1_yolo",
}

# =============================================================================
#  IMAGE SAVER
# =============================================================================

def save_result_image(img: np.ndarray, img_name: str,
                      true_cls: str, pred: str, conf: float, method: str,
                      pipeline_out_dir: Path,
                      bbox: tuple | None = None):
    img_draw = img.copy()
    _, w = img_draw.shape[:2]

    outcome = "correct" if pred == true_cls else "wrong"
    save_dir = pipeline_out_dir / outcome
    save_dir.mkdir(parents=True, exist_ok=True)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_draw, (x1, y1), (x2, y2),
                      COLOUR.get(pred, (200, 200, 200)), 2)

    banner_h = 44
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    cv2.putText(banner, f"GT: {true_cls}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLOUR.get(true_cls, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.putText(banner, f"PRED: {pred}  ({conf:.0%})",
                (w // 2, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLOUR.get(pred, (255, 255, 255)), 2, cv2.LINE_AA)
    img_draw = np.vstack([banner, img_draw])

    strip = np.zeros((26, w, 3), dtype=np.uint8)
    cv2.putText(strip, f"method: {method}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (190, 190, 190), 1, cv2.LINE_AA)
    img_draw = np.vstack([img_draw, strip])

    cv2.imwrite(str(save_dir / img_name), img_draw)


# =============================================================================
#  PIPELINE RUNNERS
# =============================================================================

def run_raw_cnn(img, cnn_model, class_names, transform, cnn_device):
    from pipeline_utils import predict_crop
    cls_name, conf, _ = predict_crop(cnn_model, class_names, transform, cnn_device, img)
    return cls_name, conf


def run_onestep(img, onestep_model):
    r = onestep_model.predict(
        source=img, imgsz=640, conf=0.25, device=0, verbose=False,
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
    return name_map.get(cls_name.lower(), cls_name), conf, bbox


def run_twostep(img, stage1_model, cnn_model, class_names, transform, cnn_device, geo_rule):
    from pipeline_utils import pad_bbox, predict_crop, STAGE1_CONF
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
    cls_name, conf, _ = predict_crop(cnn_model, class_names, transform, cnn_device, crop)
    return cls_name, conf, "cnn", raw_bbox


def run_stage1_yolo(img, stage1_model, onestep_model, geo_rule):
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
#  CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion(cm: np.ndarray, title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    cm_norm = cm.astype(float)
    for i in range(len(LABELS)):
        row_sum = cm_norm[i].sum()
        if row_sum > 0:
            cm_norm[i] /= row_sum

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
    axes[1].set_title(f"{title} (Normalised)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.name}")


def plot_all_pipelines(cms: dict, save_path: Path):
    n = len(cms)
    fig, axes = plt.subplots(n, 2, figsize=(14, n * 6))

    for row, (name, cm) in enumerate(cms.items()):
        short = name.split("—")[1].strip()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=axes[row][0])
        axes[row][0].set_title(f"{short} (Counts)")
        axes[row][0].set_xlabel("Predicted"); axes[row][0].set_ylabel("True")

        cm_norm = cm.astype(float)
        for i in range(len(LABELS)):
            s = cm_norm[i].sum()
            if s > 0:
                cm_norm[i] /= s

        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=axes[row][1])
        axes[row][1].set_title(f"{short} (Normalised)")
        axes[row][1].set_xlabel("Predicted"); axes[row][1].set_ylabel("True")

    fig.suptitle("IS_Image Dataset — Four Pipeline Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.name}")


# =============================================================================
#  PRINT TABLE
# =============================================================================

def print_results(pipeline_name: str, results: list[dict]):
    cm = np.zeros((3, 3), dtype=int)
    no_det = 0

    for r in results:
        ti = LABELS.index(r["true"]) if r["true"] in LABELS else -1
        if r["pred"] in LABELS:
            pi = LABELS.index(r["pred"])
            if ti >= 0:
                cm[ti][pi] += 1
        else:
            no_det += 1

    total   = len(results)
    correct = int(np.trace(cm))

    print(f"\n{'─' * 70}")
    print(f"  {pipeline_name}")
    print(f"{'─' * 70}")
    print(f"  Overall: {correct}/{total} = {correct/total:.1%}  "
          f"(NoDetection: {no_det})")
    print(f"\n  {'Class':<8} {'Total':>6} {'Correct':>8} {'Acc':>7}  "
          f"{'Misclassified As'}")
    print(f"  {'-'*60}")

    for i, cls in enumerate(LABELS):
        row_total   = cm[i].sum()
        row_correct = cm[i][i]
        acc         = row_correct / row_total if row_total > 0 else 0.0
        misclass    = {LABELS[j]: cm[i][j]
                       for j in range(3) if j != i and cm[i][j] > 0}
        mis_str = ", ".join(f"{k}:{v}" for k, v in
                            sorted(misclass.items(), key=lambda x: -x[1]))
        print(f"  {cls:<8} {row_total:>6} {row_correct:>8} {acc:>7.1%}  {mis_str}")

    return cm


def print_comparison(pipeline_results: dict):
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Pipeline':<28} {'Fall':>7} {'Sit':>7} {'Walk':>7} {'Overall':>9}")
    print(f"  {'-' * 62}")

    for name, results in pipeline_results.items():
        cm = np.zeros((3, 3), dtype=int)
        for r in results:
            ti = LABELS.index(r["true"]) if r["true"] in LABELS else -1
            if r["pred"] in LABELS:
                pi = LABELS.index(r["pred"])
                if ti >= 0:
                    cm[ti][pi] += 1

        per_cls = []
        for i in range(3):
            row_sum = cm[i].sum()
            per_cls.append(cm[i][i] / row_sum if row_sum > 0 else 0.0)

        total   = sum(cm[i].sum() for i in range(3))
        correct = int(np.trace(cm))
        overall = correct / total if total > 0 else 0.0

        short = name.split("—")[1].strip() if "—" in name else name
        print(f"  {short:<28} {per_cls[0]:>7.1%} {per_cls[1]:>7.1%} "
              f"{per_cls[2]:>7.1%} {overall:>9.1%}")

    print(f"\n  Key comparisons:")
    print(f"    P1 vs P2 → how much does localisation help CNN?")
    print(f"    P3 vs P4 → does YOLO benefit from pre-cropping?")
    print(f"    P2 vs P3 → CNN pipeline vs YOLO pipeline (real-world)")
    print(f"    P2 vs P4 → same Stage1 localisation, CNN vs YOLO classifier")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import load_models, GeometricFallRule

    print("=" * 70)
    print("  COS30018 — IS_Image Confusion Matrix (Four Pipelines)")
    print("=" * 70)
    print(f"  Data : {DATA_DIR}")
    print(f"  Out  : {OUT_DIR}")

    # ── Validate ──────────────────────────────────────────────────────────────
    for cls in FOLDER_TO_CLASS:
        d = DATA_DIR / cls
        if not d.exists():
            print(f"\n  [ERROR] Not found: {d}")
            return

    # ── Clear previous image outputs ──────────────────────────────────────────
    img_out = OUT_DIR / "images"
    if img_out.exists():
        shutil.rmtree(img_out)
    for pipeline_dir in PIPELINE_DIRS.values():
        for sub in SORT_SUBFOLDERS:
            (img_out / pipeline_dir / sub).mkdir(parents=True, exist_ok=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    m        = load_models(two_step=True)
    geo_rule = GeometricFallRule(aspect_threshold=2.2)
    print(f"  Pipelines:")
    print(f"    P1 Raw CNN            : full image → stage2_cnn.pt (no localisation)")
    print(f"    P2 Stage1 + CNN       : stage1_person.pt → stage2_cnn.pt")
    print(f"    P3 Raw YOLO           : full image → swook_techcare_fall_sit_walk.pt")
    print(f"    P4 Stage1 + YOLO      : stage1_person.pt → swook_techcare_fall_sit_walk.pt")

    # ── Collect images (flat structure: IS_Image/Fall/*.jpg) ──────────────────
    all_images = []
    for folder_name, true_cls in FOLDER_TO_CLASS.items():
        img_dir = DATA_DIR / folder_name
        images  = sorted(f for f in img_dir.iterdir()
                         if f.suffix.lower() in IMG_EXTS)
        for img_path in images:
            all_images.append({
                "img_path": img_path,
                "true_cls": true_cls,
            })

    class_counts = defaultdict(int)
    for item in all_images:
        class_counts[item["true_cls"]] += 1
    print(f"\n  Dataset:")
    for cls in LABELS:
        print(f"    {cls:<6} {class_counts[cls]:>5} images")
    print(f"    {'Total':<6} {len(all_images):>5} images")

    # ── Pipeline output dirs ──────────────────────────────────────────────────
    pipeline_out_dirs = {
        name: img_out / PIPELINE_DIRS[name]
        for name in PIPELINE_DIRS
    }

    # ── Run all four pipelines ────────────────────────────────────────────────
    pipeline_results = {
        "P1 — Raw CNN":        [],
        "P2 — Stage1 + CNN":   [],
        "P3 — Raw YOLO":       [],
        "P4 — Stage1 + YOLO":  [],
    }

    print(f"\nEvaluating {len(all_images)} images...")
    for i, item in enumerate(all_images):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_images)}...")

        img = cv2.imread(str(item["img_path"]))
        if img is None:
            continue

        true_cls = item["true_cls"]
        fname    = item["img_path"].name

        # P1 — Raw CNN (full image, no localisation)
        pred1, conf1 = run_raw_cnn(
            img, m["cnn"], m["class_names"], m["transform"], m["cnn_device"]
        )
        pipeline_results["P1 — Raw CNN"].append(
            {"true": true_cls, "pred": pred1, "conf": conf1,
             "method": "raw_cnn", "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred1, conf1,
                          "raw_cnn", pipeline_out_dirs["P1 — Raw CNN"])

        # P2 — Stage1 + CNN
        pred2, conf2, method2, bbox2 = run_twostep(
            img, m["stage1"], m["cnn"], m["class_names"],
            m["transform"], m["cnn_device"], geo_rule
        )
        pipeline_results["P2 — Stage1 + CNN"].append(
            {"true": true_cls, "pred": pred2, "conf": conf2,
             "method": method2, "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred2, conf2,
                          method2, pipeline_out_dirs["P2 — Stage1 + CNN"],
                          bbox=bbox2)

        # P3 — Raw YOLO (full image)
        pred3, conf3, bbox3 = run_onestep(img, m["onestep"])
        pipeline_results["P3 — Raw YOLO"].append(
            {"true": true_cls, "pred": pred3, "conf": conf3,
             "method": "onestep", "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred3, conf3,
                          "onestep", pipeline_out_dirs["P3 — Raw YOLO"],
                          bbox=bbox3)

        # P4 — Stage1 + YOLO
        pred4, conf4, method4, bbox4 = run_stage1_yolo(
            img, m["stage1"], m["onestep"], geo_rule
        )
        pipeline_results["P4 — Stage1 + YOLO"].append(
            {"true": true_cls, "pred": pred4, "conf": conf4,
             "method": method4, "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred4, conf4,
                          method4, pipeline_out_dirs["P4 — Stage1 + YOLO"],
                          bbox=bbox4)

    # ── Print results + save confusion matrices ──────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")

    cms = {}
    summary_lines = []

    for name, results in pipeline_results.items():
        cm = print_results(name, results)
        cms[name] = cm

        correct = int(np.trace(cm))
        total   = len(results)
        summary_lines.append(f"{name}: {correct}/{total} = {correct/total:.1%}")

        safe = name.split("—")[1].strip().replace(" ", "_").replace("+", "plus").lower()
        plot_confusion(cm, name, OUT_DIR / f"confusion_{safe}.png")

    plot_all_pipelines(cms, OUT_DIR / "confusion_all_pipelines.png")
    print_comparison(pipeline_results)

    # ── Image folder summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SAVED IMAGE COUNTS")
    print(f"{'=' * 70}")
    for pipeline_dir_name in PIPELINE_DIRS.values():
        print(f"\n  {pipeline_dir_name}/")
        for sub in SORT_SUBFOLDERS:
            folder = img_out / pipeline_dir_name / sub
            count  = len(list(folder.glob("*"))) if folder.exists() else 0
            print(f"    {sub:<30} {count:>5} images")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    csv_out = OUT_DIR / "csvs"
    csv_out.mkdir(parents=True, exist_ok=True)

    for name, results in pipeline_results.items():
        safe_name = name.split("—")[1].strip().replace(" ", "_").replace("+", "plus").lower()
        csv_path  = csv_out / f"isimage_{safe_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "true", "pred", "conf", "method"])
            writer.writeheader()
            for r in results:
                writer.writerow({**r, "conf": f"{r['conf']:.4f}"})
        print(f"\n  CSV -> {csv_path.name}")

    # ── Save text summary ─────────────────────────────────────────────────────
    summary_path = OUT_DIR / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("IS_Image Confusion Matrix — Summary\n")
        f.write("=" * 50 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\n  Summary -> {summary_path.name}")
    print(f"  All outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

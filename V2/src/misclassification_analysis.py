"""
misclassification_analysis.py
Runs both one-step (YOLOv8) and two-step (Pose MLP) models on the test set.
Copies each image into FALL_Detected_Manual_Test/{walk,sit,fall}/{correct,wrong}/
Generates confusion matrices and per-image analysis.

Usage:
    python V2/src/misclassification_analysis.py
"""
import csv
import shutil
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE.parent

sys.path.insert(0, str(BASE / "src" / "GUI"))
from pipeline_utils import (
    pad_bbox, extract_pose_features, GeometricFallRule,
    STAGE1_CONF, CLASS_NAMES, FOLDER_TO_CLASS,
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GEO_RULE = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)

TEST_DIR = BASE / "dataset_stage1" / "fall_sit_walk_test"
OUTPUT_BASE = BASE / "FALL_Detected_Manual_Test"

ONESTEP_MODEL = BASE / "models" / "swook_techcare_fall_sit_walk.pt"
STAGE1_MODEL = BASE / "models" / "stage1_person.pt"
STAGE2_MODEL = BASE / "models" / "stage2_pose_mlp_v2.pt"
STAGE2_SCALER = BASE / "models" / "pose_scaler_v2.pkl"
POSE_MODEL = BASE / "models" / "pose_landmarker_heavy.task"

DEBUG_SKELETON_DIR = OUTPUT_BASE / "debug_skeletons"

ONESTEP_NAMES = None


def draw_skeleton_on_frame(frame, pose_landmarks, cx1, cy1, cx2, cy2, color=(0, 220, 0)):
    """Draw MediaPipe skeleton on full frame, mapping crop-relative landmarks to frame coords."""
    if not pose_landmarks:
        return
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w <= 0 or crop_h <= 0:
        return
    h, w = frame.shape[:2]
    points = {}
    for i, lm in enumerate(pose_landmarks):
        px = max(0, min(w - 1, int(cx1 + lm.x * crop_w)))
        py = max(0, min(h - 1, int(cy1 + lm.y * crop_h)))
        points[i] = (px, py)
        if lm.visibility > 0.3:
            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            lm_a, lm_b = pose_landmarks[a], pose_landmarks[b]
            if lm_a.visibility > 0.3 and lm_b.visibility > 0.3:
                cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)


def save_debug_image(img_path, true_cls, pred_cls, conf, method,
                     bbox, padded_bbox, stage1_conf, pose_landmarks):
    """Save annotated debug image with bbox, skeleton, and text overlays."""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 100, 0), 2)
        cv2.putText(img, f"raw bbox s1conf={stage1_conf:.2f}", (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    if padded_bbox is not None:
        px1, py1, px2, py2 = padded_bbox
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 255), 2)
        bw = px2 - px1
        bh = py2 - py1
        aspect = bw / bh if bh > 0 else 0
        cv2.putText(img, f"padded {bw}x{bh} ar={aspect:.2f}", (px1, py2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if pose_landmarks:
            draw_skeleton_on_frame(img, pose_landmarks, px1, py1, px2, py2)

    lines = [
        f"TRUE: {true_cls}",
        f"PRED: {pred_cls} ({conf:.1%})",
        f"METHOD: {method}",
    ]
    for i, line in enumerate(lines):
        color = (0, 0, 255) if pred_cls != true_cls else (0, 200, 0)
        cv2.putText(img, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    folder_name = true_cls.lower()
    out_dir = DEBUG_SKELETON_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{img_path.stem}_debug.jpg"
    cv2.imwrite(str(out_dir / out_name), img)


def load_onestep():
    model = YOLO(str(ONESTEP_MODEL))
    global ONESTEP_NAMES
    ONESTEP_NAMES = model.names
    return model


def load_twostep():
    sys.path.insert(0, str(BASE / "src" / "GUI"))
    from train_stage2_pose import PoseMLP

    detector = YOLO(str(STAGE1_MODEL))
    mlp = PoseMLP()
    mlp.load_state_dict(torch.load(str(STAGE2_MODEL), map_location="cpu"))
    mlp.eval()
    with open(str(STAGE2_SCALER), "rb") as f:
        scaler = pickle.load(f)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    pose_landmarker = PoseLandmarker.create_from_options(options)
    return detector, mlp, scaler, pose_landmarker


def predict_onestep(model, img_path, conf=0.45):
    img = cv2.imread(str(img_path))
    if img is None:
        return "NoDetection", 0.0
    r = model.predict(source=img, conf=conf, imgsz=640, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return "NoDetection", 0.0
    best_conf = 0.0
    best_cls = "NoDetection"
    for box in r.boxes:
        cls_name = ONESTEP_NAMES[int(box.cls[0])]
        c = float(box.conf[0])
        if c > best_conf:
            best_conf = c
            best_cls = cls_name
    return best_cls, best_conf


def predict_twostep(detector, mlp, scaler, pose_landmarker, img_path, conf1=0.35):
    """Returns (pred_cls, conf, method, debug_info)."""
    no_debug = {"bbox": None, "padded_bbox": None, "stage1_conf": 0.0, "pose_landmarks": None}
    img = cv2.imread(str(img_path))
    if img is None:
        return "NoDetection", 0.0, "load_error", no_debug
    h, w = img.shape[:2]

    r = detector.predict(source=img, conf=conf1, imgsz=640, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return "NoDetection", 0.0, "no_person", no_debug

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.cpu().numpy()]
    best_i = int(np.argmax(areas))
    raw_box = tuple(map(int, r.boxes.xyxy[best_i].cpu().numpy()))
    s1_conf = float(r.boxes.conf[best_i])
    x1, y1, x2, y2 = pad_bbox(raw_box[0], raw_box[1], raw_box[2], raw_box[3], w, h)
    padded = (x1, y1, x2, y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "NoDetection", 0.0, "empty_crop", {
            "bbox": raw_box, "padded_bbox": padded,
            "stage1_conf": s1_conf, "pose_landmarks": None,
        }

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_landmarker.detect(mp_image)

    pose_lm = result.pose_landmarks[0] if result.pose_landmarks else None
    dbg = {"bbox": raw_box, "padded_bbox": padded,
           "stage1_conf": s1_conf, "pose_landmarks": pose_lm}

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return "NoDetection", 0.0, "no_pose", dbg

    features, core_vis = extract_pose_features(pose_lm)

    if core_vis is not None:
        if GEO_RULE.check(bbox=(x1, y1, x2, y2), keypoint_visibilities=core_vis):
            return "Fall", 0.75, "geo_rule", dbg

    if features is None:
        return "NoDetection", 0.0, "no_features", dbg

    feat_norm = scaler.transform([features])
    with torch.no_grad():
        logits = mlp(torch.tensor(feat_norm, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[0]
        pred_cls = int(probs.argmax())
        conf_val = float(probs[pred_cls])
    return CLASS_NAMES[pred_cls], conf_val, "pose_mlp", dbg


def build_confusion_matrix(results):
    labels = ["Fall", "Sit", "Walk"]
    cm = np.zeros((3, 3), dtype=int)
    no_det = 0
    for r in results:
        true_cls = r["true"]
        pred_cls = r["pred"]
        ti = labels.index(true_cls)
        if pred_cls in labels:
            pi = labels.index(pred_cls)
            cm[ti][pi] += 1
        else:
            no_det += 1
    return cm, labels, no_det


def plot_confusion(cm, labels, title, save_path, no_det=0):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_norm = cm.astype(float)
    for i in range(len(labels)):
        row_sum = cm_norm[i].sum()
        if row_sum > 0:
            cm_norm[i] /= row_sum
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    if no_det > 0:
        fig.suptitle(f"{title}\n(NoDetection count: {no_det})", fontsize=12)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def print_class_metrics(results, title):
    labels = ["Fall", "Sit", "Walk"]
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    total = len(results)
    correct = sum(1 for r in results if r["pred"] == r["true"])
    no_det = sum(1 for r in results if r["pred"] == "NoDetection")
    print(f"  Overall: {correct}/{total} = {correct/total:.1%}  (NoDetection: {no_det})")

    print(f"\n  {'Class':<8} {'Total':>6} {'Correct':>8} {'Acc':>7} {'Misclass As':>30}")
    print(f"  {'-'*65}")

    for cls in labels:
        cls_results = [r for r in results if r["true"] == cls]
        cls_correct = sum(1 for r in cls_results if r["pred"] == cls)
        cls_total = len(cls_results)
        acc = cls_correct / cls_total if cls_total > 0 else 0

        misclass = defaultdict(int)
        for r in cls_results:
            if r["pred"] != cls:
                misclass[r["pred"]] += 1
        mis_str = ", ".join(f"{k}:{v}" for k, v in sorted(misclass.items(), key=lambda x: -x[1]))
        print(f"  {cls:<8} {cls_total:>6} {cls_correct:>8} {acc:>7.1%} {mis_str:>30}")


def main():
    print("Loading models...")
    onestep = load_onestep()
    detector, mlp, scaler, pose_landmarker = load_twostep()

    onestep_results = []
    twostep_results = []

    for folder_name, true_id in FOLDER_TO_CLASS.items():
        true_cls = CLASS_NAMES[true_id]
        folder_path = TEST_DIR / folder_name
        if not folder_path.exists():
            print(f"  [SKIP] {folder_path}")
            continue

        images = sorted(f for f in folder_path.iterdir() if f.suffix.lower() in IMG_EXTS)
        print(f"\n  Processing {true_cls}: {len(images)} images")

        for img_path in images:
            pred_1s, conf_1s = predict_onestep(onestep, img_path)
            pred_2s, conf_2s, method, dbg = predict_twostep(detector, mlp, scaler, pose_landmarker, img_path)

            onestep_results.append({
                "filename": img_path.name,
                "true": true_cls,
                "pred": pred_1s,
                "conf": conf_1s,
            })
            twostep_results.append({
                "filename": img_path.name,
                "true": true_cls,
                "pred": pred_2s,
                "conf": conf_2s,
                "method": method,
            })

            is_2s_correct = (pred_2s == true_cls)
            if not is_2s_correct:
                save_debug_image(
                    img_path, true_cls, pred_2s, conf_2s, method,
                    dbg["bbox"], dbg["padded_bbox"], dbg["stage1_conf"],
                    dbg["pose_landmarks"],
                )

            for mode_name, pred_cls, conf in [("onestep", pred_1s, conf_1s), ("twostep", pred_2s, conf_2s)]:
                is_correct = (pred_cls == true_cls)
                verdict = "correct" if is_correct else "wrong"
                out_dir = OUTPUT_BASE / mode_name / folder_name / verdict
                out_dir.mkdir(parents=True, exist_ok=True)

                stem = img_path.stem
                conf_tag = f"_pred{pred_cls}_conf{int(conf*100)}"
                out_name = f"{stem}{conf_tag}{img_path.suffix}"
                shutil.copy2(str(img_path), str(out_dir / out_name))

    print_class_metrics(onestep_results, "ONE-STEP (YOLOv8) RESULTS")
    print_class_metrics(twostep_results, "TWO-STEP (Pose MLP) RESULTS")

    # Confusion matrices
    cm_dir = OUTPUT_BASE / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    cm_1s, labels, nodet_1s = build_confusion_matrix(onestep_results)
    plot_confusion(cm_1s, labels, "One-Step (YOLOv8)", cm_dir / "confusion_onestep.png", nodet_1s)

    cm_2s, labels, nodet_2s = build_confusion_matrix(twostep_results)
    plot_confusion(cm_2s, labels, "Two-Step (Pose MLP)", cm_dir / "confusion_twostep.png", nodet_2s)

    # Save detailed CSV logs
    for name, results in [("onestep", onestep_results), ("twostep", twostep_results)]:
        csv_path = cm_dir / f"predictions_{name}.csv"
        with open(str(csv_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\n  CSV -> {csv_path}")

    # Misclassification analysis summary
    print(f"\n{'='*70}")
    print("  MISCLASSIFICATION ANALYSIS")
    print(f"{'='*70}")

    for mode_name, results in [("One-Step", onestep_results), ("Two-Step", twostep_results)]:
        print(f"\n  --- {mode_name} ---")
        wrong = [r for r in results if r["pred"] != r["true"]]
        print(f"  Total misclassified: {len(wrong)}/{len(results)}")

        pairs = defaultdict(list)
        for r in wrong:
            pairs[(r["true"], r["pred"])].append(r["filename"])

        for (true, pred), files in sorted(pairs.items(), key=lambda x: -len(x[1])):
            print(f"\n  {true} -> {pred}: {len(files)} images")
            for f in files[:5]:
                print(f"    {f}")
            if len(files) > 5:
                print(f"    ... and {len(files)-5} more")

    # Two-step method breakdown
    print(f"\n{'='*70}")
    print("  TWO-STEP METHOD BREAKDOWN")
    print(f"{'='*70}")
    method_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    for r in twostep_results:
        m = r.get("method", "unknown")
        method_stats[m]["total"] += 1
        if r["pred"] == r["true"]:
            method_stats[m]["correct"] += 1
        else:
            method_stats[m]["wrong"] += 1

    print(f"\n  {'Method':<15} {'Total':>6} {'Correct':>8} {'Wrong':>6} {'Acc':>7}")
    print(f"  {'-'*45}")
    for m, s in sorted(method_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {m:<15} {s['total']:>6} {s['correct']:>8} {s['wrong']:>6} {acc:>7.1%}")

    # No-detection breakdown
    print(f"\n{'='*70}")
    print("  NO-DETECTION BREAKDOWN (Two-Step)")
    print(f"{'='*70}")
    nodet_methods = defaultdict(int)
    for r in twostep_results:
        if r["pred"] == "NoDetection":
            nodet_methods[r["method"]] += 1

    for method_name, label in [
        ("no_person", "Stage 1 missed"),
        ("no_pose", "MediaPipe failed"),
        ("empty_crop", "bad bbox"),
        ("load_error", "corrupt image"),
        ("no_features", "features extraction failed"),
    ]:
        count = nodet_methods.get(method_name, 0)
        print(f"  {method_name:<15} ({label}): {count:>4} images")

    # Compute aspect ratios for no_pose cases from debug images
    # Re-scan twostep results paired with original image paths
    no_pose_aspects = []
    for folder_name, true_id in FOLDER_TO_CLASS.items():
        folder_path = TEST_DIR / folder_name
        if not folder_path.exists():
            continue
        images = sorted(f for f in folder_path.iterdir() if f.suffix.lower() in IMG_EXTS)
        for img_path in images:
            matching = [r for r in twostep_results
                        if r["filename"] == img_path.name and r["method"] == "no_pose"]
            if not matching:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            r_det = detector.predict(source=img, conf=0.35, imgsz=640, device=0, verbose=False)[0]
            if r_det.boxes is None or len(r_det.boxes) == 0:
                continue
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in r_det.boxes.xyxy.cpu().numpy()]
            best_i = int(np.argmax(areas))
            bx1, by1, bx2, by2 = map(int, r_det.boxes.xyxy[best_i].cpu().numpy())
            bx1, by1, bx2, by2 = pad_bbox(bx1, by1, bx2, by2, iw, ih)
            bw = bx2 - bx1
            bh = by2 - by1
            if bh > 0:
                no_pose_aspects.append(bw / bh)

    if no_pose_aspects:
        print(f"\n  For no_pose cases — bbox aspect ratios:")
        print(f"    mean: {np.mean(no_pose_aspects):.2f}  "
              f"min: {np.min(no_pose_aspects):.2f}  "
              f"max: {np.max(no_pose_aspects):.2f}")
        wide = sum(1 for a in no_pose_aspects if a > 1.5)
        print(f"    wide > 1.5 (prone person): {wide}/{len(no_pose_aspects)}")
    else:
        print(f"\n  No no_pose cases found.")

    pose_landmarker.close()

    print(f"\n  Results saved to: {OUTPUT_BASE}")
    print(f"  Debug skeletons: {DEBUG_SKELETON_DIR}")
    print(f"  Confusion matrices: {cm_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()

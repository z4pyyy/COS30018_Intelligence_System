"""
evaluate_twostep.py
Evaluates the full two-step pipeline:
  Stage 1: YOLOv8n person detection
  Stage 2: MediaPipe pose -> MLP classification

Usage:
    python V2/src/evaluate_twostep.py
    python V2/src/evaluate_twostep.py --conf1 0.35
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline_utils import (
    pad_bbox, extract_pose_features, GeometricFallRule,
    STAGE1_CONF, CLASS_NAMES, FOLDER_TO_CLASS,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GEO_RULE = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)


class TwoStepPipeline:
    def __init__(self, stage1_path, stage2_path, scaler_path, pose_model_path, conf1=0.35):
        sys.path.insert(0, str(BASE / "src"))
        from train_stage2_pose import PoseMLP

        self.detector = YOLO(stage1_path)
        self.mlp = PoseMLP()
        self.mlp.load_state_dict(torch.load(stage2_path, map_location="cpu"))
        self.mlp.eval()
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.conf1 = conf1

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(options)

    def predict(self, img_path: Path) -> tuple:
        img = cv2.imread(str(img_path))
        if img is None:
            return -1, 0.0, "load_error"
        h, w = img.shape[:2]

        r = self.detector.predict(source=img, conf=self.conf1,
                                  imgsz=640, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return -1, 0.0, "no_person"

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.cpu().numpy()]
        best_i = int(np.argmax(areas))
        x1, y1, x2, y2 = map(int, r.boxes.xyxy[best_i].cpu().numpy())
        x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return -1, 0.0, "empty_crop"

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose_landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return -1, 0.0, "no_pose"

        features, core_vis = extract_pose_features(result.pose_landmarks[0])

        if core_vis is not None:
            if GEO_RULE.check(bbox=(x1, y1, x2, y2), keypoint_visibilities=core_vis):
                return 0, 0.75, "geo_rule"

        if features is None:
            return -1, 0.0, "no_pose"

        feat_norm = self.scaler.transform([features])
        with torch.no_grad():
            logits = self.mlp(torch.tensor(feat_norm, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)[0]
            pred_cls = int(probs.argmax())
            conf = float(probs[pred_cls])
        return pred_cls, conf, "pose"

    def close(self):
        self.pose_landmarker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default=str(BASE / "models" / "stage1_person.pt"))
    parser.add_argument("--stage2", default=str(BASE / "models" / "stage2_pose_mlp_v2.pt"))
    parser.add_argument("--scaler", default=str(BASE / "models" / "pose_scaler_v2.pkl"))
    parser.add_argument("--pose-model", default=str(BASE / "models" / "pose_landmarker_heavy.task"))
    parser.add_argument("--test-dir", default=str(BASE / "dataset_stage1" / "fall_sit_walk_test"))
    parser.add_argument("--conf1", type=float, default=0.35)
    parser.add_argument("--output", default=str(BASE / "runs" / "evaluation" / "twostep_results_v3"))
    args = parser.parse_args()

    pipeline = TwoStepPipeline(args.stage1, args.stage2, args.scaler,
                               args.pose_model, args.conf1)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    class_tp = defaultdict(int)
    class_fn = defaultdict(int)
    class_fp = defaultdict(int)
    class_total = defaultdict(int)
    stage_counts = Counter()

    for folder, true_id in FOLDER_TO_CLASS.items():
        folder_path = test_dir / folder
        if not folder_path.exists():
            continue
        images = sorted(f for f in folder_path.iterdir() if f.suffix.lower() in IMG_EXTS)
        print(f"  {CLASS_NAMES[true_id]}: {len(images)} images")

        for img_path in images:
            pred_id, conf, method = pipeline.predict(img_path)
            stage_counts[method] += 1
            is_correct = (pred_id == true_id)
            class_total[true_id] += 1
            if is_correct:
                class_tp[true_id] += 1
            else:
                class_fn[true_id] += 1
                if pred_id >= 0:
                    class_fp[pred_id] += 1
            pred_name = CLASS_NAMES.get(pred_id, "NoDetection")
            results.append({
                "filename": img_path.name,
                "true": CLASS_NAMES[true_id],
                "pred": pred_name,
                "conf": f"{conf:.4f}",
                "correct": is_correct,
                "method": method,
            })

    print("\n" + "=" * 65)
    print("TASK 2 — TWO-STEP PIPELINE EVALUATION RESULTS")
    print("=" * 65)
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\nOverall Accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"\nStage breakdown: {dict(stage_counts)}")
    no_det = stage_counts["no_person"] + stage_counts["no_pose"] + stage_counts["empty_crop"]
    print(f"No detection rate: {no_det}/{total} = {no_det / total:.1%}")

    print(f"\n{'Class':<10} {'Total':>6} {'Correct':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 65)
    for cid in sorted(CLASS_NAMES):
        tp = class_tp[cid]; fp = class_fp[cid]; fn = class_fn[cid]; tot = class_total[cid]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{CLASS_NAMES[cid]:<10} {tot:>6} {tp:>8} {tp / tot:>8.1%} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

    csv_path = out_dir / "prediction_log_twostep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nLog saved -> {csv_path}")

    # Confusion matrix
    y_true = [FOLDER_TO_CLASS[r["true"].lower()] for r in results if r["pred"] != "NoDetection"]
    y_pred = [next(k for k, v in CLASS_NAMES.items() if v == r["pred"])
              for r in results if r["pred"] != "NoDetection"]
    cm = np.zeros((3, 3), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()), ax=axes[0])
    axes[0].set_title("Two-Step Pipeline — Confusion Matrix (Counts)")
    cm_n = cm.astype(float)
    for i in range(3):
        if cm_n[i].sum() > 0:
            cm_n[i] /= cm_n[i].sum()
    sns.heatmap(cm_n, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()), ax=axes[1])
    axes[1].set_title("Two-Step Pipeline — Confusion Matrix (Normalized)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrix_twostep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix -> {out_dir / 'confusion_matrix_twostep.png'}")

    pipeline.close()


if __name__ == "__main__":
    main()

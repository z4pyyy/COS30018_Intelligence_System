"""
evaluate_sit_fall_walk.py  (V3)
======================================
Runs six pipelines on the IS_Image/ dataset and produces
3-class confusion matrices (Fall / Sit / Walk).

Input folder structure (flat — images directly in class folders):
  IS_Image/
  ├── Fall/   ← GT = Fall
  ├── Sit/    ← GT = Sit
  └── Walk/   ← GT = Walk

Six pipelines:
  P1: Raw CNN         — full image → EfficientNet-B0 (no localisation)
  P2: One-step YOLO   — full image → swook_techcare YOLO
  P3: Stage1 + CNN    — Stage1 person crop → CNN
  P4: Stage1 + YOLO   — Stage1 person crop → one-step YOLO
  P5: Raw MLP         — full image → MediaPipe pose → MLP
  P6: Stage1 + MLP    — Stage1 person crop → MediaPipe pose → MLP

Output:
  V3/runs/evaluation/ISImage_fallsitwalk/
  ├── images/<pipeline_dir>/{correct,wrong}/
  ├── images/wrong_all_pipelines/          ← wrong in ALL six pipelines
  ├── images/wrong_mlp_and_others/         ← wrong in MLP AND at least one other
  ├── confusion_*.png
  ├── csvs/
  └── results_summary.txt

Usage:
    python V3/src/evaluate_sit_fall_walk.py
"""

import csv
import sys
import pickle
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
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
DATA_DIR     = PROJECT_ROOT / "final_version" / "dataset" / "fall_sit_walk_ISImage"
OUT_DIR      = PROJECT_ROOT / "final_version" / "evaluation" / "ISImage_fallsitwalk"

MLP_MODEL_PATH  = V2_BASE / "models" / "stage2_pose_mlp_v2.pt"
MLP_SCALER_PATH = V2_BASE / "models" / "pose_scaler_v2.pkl"
POSE_TASK_PATH  = V2_BASE / "models" / "pose_landmarker_heavy.task"

IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_LABELS    = ["Fall", "Sit", "Walk"]
PRED_LABELS  = ["Fall", "Sit", "Walk", "NoDetection"]

FOLDER_TO_CLASS = {
    "fall": "Fall",
    "sit":  "Sit",
    "walk": "Walk",
}

COLOUR = {
    "Fall":        (0,   0,   220),
    "Sit":         (0,   180, 180),
    "Walk":        (0,   200,   0),
    "NoDetection": (128, 128, 128),
}

PIPELINE_DIRS = {
    "P1 — Raw CNN":        "p1_raw_cnn",
    "P2 — One-step YOLO":  "p2_onestep_yolo",
    "P3 — Stage1 + CNN":   "p3_stage1_cnn",
    "P4 — Stage1 + YOLO":  "p4_stage1_yolo",
    "P5 — Raw MLP":        "p5_raw_mlp",
    "P6 — Stage1 + MLP":   "p6_stage1_mlp",
}

MLP_PIPELINES = {"P5 — Raw MLP", "P6 — Stage1 + MLP"}

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


def save_flagged_image(img: np.ndarray, img_name: str,
                       true_cls: str, preds: dict,
                       out_dir: Path, banner_text: str):
    _, w = img.shape[:2]
    img_draw = img.copy()

    banner_h = 44
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    cv2.putText(banner, f"GT: {true_cls}  |  {banner_text}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2, cv2.LINE_AA)
    img_draw = np.vstack([banner, img_draw])

    strip_h = 22 * len(preds)
    strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
    for j, (pname, info) in enumerate(preds.items()):
        short = pname.split("—")[1].strip() if "—" in pname else pname
        text = f"{short}: {info['pred']} ({info['conf']:.0%})"
        cv2.putText(strip, text,
                    (8, 18 + j * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLOUR.get(info["pred"], (190, 190, 190)), 1, cv2.LINE_AA)
    img_draw = np.vstack([img_draw, strip])

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_name), img_draw)


# =============================================================================
#  MLP LOADER
# =============================================================================

def load_mlp():
    """Load V2 PoseMLP model + scaler + MediaPipe PoseLandmarker."""
    sys.path.insert(0, str(V2_BASE / "src"))
    from train_stage2_pose import PoseMLP

    mlp = PoseMLP(input_dim=8, hidden=128, n_classes=3)
    mlp.load_state_dict(torch.load(str(MLP_MODEL_PATH), map_location="cpu", weights_only=True))
    mlp.eval()

    with open(MLP_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_TASK_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    print(f"  [OK] MLP loaded  : {MLP_MODEL_PATH.name}")
    print(f"  [OK] Scaler      : {MLP_SCALER_PATH.name}")
    print(f"  [OK] PoseLandmark: {POSE_TASK_PATH.name}")

    return mlp, scaler, pose_landmarker


def extract_pose_features(landmarks):
    """Extract 8 geometric features from MediaPipe landmarks. Returns (features, core_vis) or (None, None)."""
    if landmarks is None:
        return None, None

    lm = landmarks

    l_sh = lm[11]; r_sh = lm[12]
    l_hi = lm[23]; r_hi = lm[24]
    l_kn = lm[25]; r_kn = lm[26]
    l_an = lm[27]; r_an = lm[28]
    nose = lm[0]

    msx = (l_sh.x + r_sh.x) / 2;  msy = (l_sh.y + r_sh.y) / 2
    mhx = (l_hi.x + r_hi.x) / 2;  mhy = (l_hi.y + r_hi.y) / 2

    spine_angle = float(np.degrees(np.arctan2(abs(msx - mhx), abs(msy - mhy) + 1e-9)))

    hip_h = (l_hi.y + r_hi.y) / 2
    bh = abs(msy - (l_an.y + r_an.y) / 2) + 1e-9
    sw = abs(l_sh.x - r_sh.x) / bh

    def _angle(a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    kbl = _angle(l_hi, l_kn, l_an)
    kbr = _angle(r_hi, r_kn, r_an)

    xs = [lm[i].x for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    ys = [lm[i].y for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    bar = (max(xs) - min(xs)) / (max(ys) - min(ys) + 1e-9)

    ahv = abs((l_an.y + r_an.y) / 2 - hip_h) / (bh + 1e-9)

    features = [spine_angle, hip_h, sw, kbl, kbr, bar, nose.y, ahv]
    core_vis = [l_sh.visibility, r_sh.visibility, l_hi.visibility, r_hi.visibility]

    return features, core_vis


def run_mlp_on_image(img_bgr, pose_landmarker, mlp, scaler):
    """Run MediaPipe pose → MLP on a BGR image. Returns (pred_class, confidence, method)."""
    import mediapipe as mp

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return "NoDetection", 0.0, "no_pose"

    landmarks = result.pose_landmarks[0]
    features, core_vis = extract_pose_features(landmarks)

    if features is None:
        return "NoDetection", 0.0, "no_features"

    min_vis = min(core_vis) if core_vis else 0.0
    if min_vis < 0.3:
        return "NoDetection", 0.0, "low_visibility"

    feat_arr = scaler.transform([features])
    feat_tensor = torch.tensor(feat_arr, dtype=torch.float32)

    with torch.no_grad():
        logits = mlp(feat_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(probs.argmax())
        conf = float(probs[pred_id])

    return GT_LABELS[pred_id], conf, "mlp"


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


def run_stage1_mlp(img, stage1_model, pose_landmarker, mlp, scaler, geo_rule):
    """Stage1 crop → MediaPipe pose → MLP."""
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

    pred, conf, method = run_mlp_on_image(crop, pose_landmarker, mlp, scaler)
    return pred, conf, method, raw_bbox


# =============================================================================
#  CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion(cm: np.ndarray, title: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=PRED_LABELS, yticklabels=GT_LABELS, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    cm_norm = cm.astype(float)
    for i in range(len(GT_LABELS)):
        row_sum = cm_norm[i].sum()
        if row_sum > 0:
            cm_norm[i] /= row_sum

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=PRED_LABELS, yticklabels=GT_LABELS, ax=axes[1])
    axes[1].set_title(f"{title} (Normalised)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.name}")


def plot_all_pipelines(cms: dict, save_path: Path):
    n = len(cms)
    fig, axes = plt.subplots(n, 2, figsize=(16, n * 6))

    for row, (name, cm) in enumerate(cms.items()):
        short = name.split("—")[1].strip()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=PRED_LABELS, yticklabels=GT_LABELS, ax=axes[row][0])
        axes[row][0].set_title(f"{short} (Counts)")
        axes[row][0].set_xlabel("Predicted"); axes[row][0].set_ylabel("True")

        cm_norm = cm.astype(float)
        for i in range(len(GT_LABELS)):
            s = cm_norm[i].sum()
            if s > 0:
                cm_norm[i] /= s

        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                    xticklabels=PRED_LABELS, yticklabels=GT_LABELS, ax=axes[row][1])
        axes[row][1].set_title(f"{short} (Normalised)")
        axes[row][1].set_xlabel("Predicted"); axes[row][1].set_ylabel("True")

    fig.suptitle("IS_Image Dataset — Six Pipeline Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.name}")


# =============================================================================
#  PRINT TABLE
# =============================================================================

def print_results(pipeline_name: str, results: list[dict]):
    n_gt   = len(GT_LABELS)
    n_pred = len(PRED_LABELS)
    cm = np.zeros((n_gt, n_pred), dtype=int)

    for r in results:
        ti = GT_LABELS.index(r["true"]) if r["true"] in GT_LABELS else -1
        if r["pred"] in PRED_LABELS:
            pi = PRED_LABELS.index(r["pred"])
            if ti >= 0:
                cm[ti][pi] += 1

    total   = len(results)
    correct = sum(cm[i][i] for i in range(n_gt))

    print(f"\n{'─' * 70}")
    print(f"  {pipeline_name}")
    print(f"{'─' * 70}")
    acc_str = f"{correct/total:.1%}" if total > 0 else "N/A"
    print(f"  Overall: {correct}/{total} = {acc_str}")
    print(f"\n  {'Class':<8} {'Total':>6} {'Correct':>8} {'Acc':>7}  "
          f"{'Misclassified As'}")
    print(f"  {'-'*60}")

    for i, cls in enumerate(GT_LABELS):
        row_total   = cm[i].sum()
        row_correct = cm[i][i]
        acc         = row_correct / row_total if row_total > 0 else 0.0
        misclass    = {PRED_LABELS[j]: cm[i][j]
                       for j in range(n_pred) if j != i and cm[i][j] > 0}
        mis_str = ", ".join(f"{k}:{v}" for k, v in
                            sorted(misclass.items(), key=lambda x: -x[1]))
        print(f"  {cls:<8} {row_total:>6} {row_correct:>8} {acc:>7.1%}  {mis_str}")

    return cm


def print_comparison(pipeline_results: dict):
    n_gt   = len(GT_LABELS)
    n_pred = len(PRED_LABELS)

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Pipeline':<28} {'Fall':>7} {'Sit':>7} {'Walk':>7} {'Overall':>9}")
    print(f"  {'-' * 62}")

    for name, results in pipeline_results.items():
        cm = np.zeros((n_gt, n_pred), dtype=int)
        for r in results:
            ti = GT_LABELS.index(r["true"]) if r["true"] in GT_LABELS else -1
            if r["pred"] in PRED_LABELS:
                pi = PRED_LABELS.index(r["pred"])
                if ti >= 0:
                    cm[ti][pi] += 1

        per_cls = []
        for i in range(n_gt):
            row_sum = cm[i].sum()
            per_cls.append(cm[i][i] / row_sum if row_sum > 0 else 0.0)

        total   = sum(cm[i].sum() for i in range(n_gt))
        correct = sum(cm[i][i] for i in range(n_gt))
        overall = correct / total if total > 0 else 0.0

        short = name.split("—")[1].strip() if "—" in name else name
        print(f"  {short:<28} {per_cls[0]:>7.1%} {per_cls[1]:>7.1%} "
              f"{per_cls[2]:>7.1%} {overall:>9.1%}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    sys.path.insert(0, str(V3_BASE / "src"))
    from pipeline_utils import load_models, GeometricFallRule

    print("=" * 70)
    print("  COS30018 — IS_Image Confusion Matrix (Six Pipelines)")
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
        for sub in ["correct", "wrong"]:
            (img_out / pipeline_dir / sub).mkdir(parents=True, exist_ok=True)
    wrong_all_dir = img_out / "wrong_all_pipelines"
    wrong_all_dir.mkdir(parents=True, exist_ok=True)
    wrong_mlp_dir = img_out / "wrong_mlp_and_others"
    wrong_mlp_dir.mkdir(parents=True, exist_ok=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    m        = load_models(two_step=True)
    geo_rule = GeometricFallRule(aspect_threshold=2.2)

    mlp, scaler, pose_landmarker = load_mlp()

    print(f"\n  Pipelines:")
    print(f"    P1 Raw CNN            : full image -> stage2_cnn.pt")
    print(f"    P2 One-step YOLO      : full image -> swook_techcare_fall_sit_walk.pt")
    print(f"    P3 Stage1 + CNN       : stage1_person.pt -> stage2_cnn.pt")
    print(f"    P4 Stage1 + YOLO      : stage1_person.pt -> swook_techcare_fall_sit_walk.pt")
    print(f"    P5 Raw MLP            : full image -> MediaPipe -> pose_mlp_v2.pt")
    print(f"    P6 Stage1 + MLP       : stage1_person.pt -> MediaPipe -> pose_mlp_v2.pt")

    # ── Collect images ──────────────────────────────────────────────────────
    all_images = []
    for folder_name, true_cls in FOLDER_TO_CLASS.items():
        # Support both flat (Fall/*.jpg) and nested (fall/images/*.jpg)
        img_dir = DATA_DIR / folder_name / "images"
        if not img_dir.exists():
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
    for cls in GT_LABELS:
        print(f"    {cls:<6} {class_counts[cls]:>5} images")
    print(f"    {'Total':<6} {len(all_images):>5} images")

    # ── Pipeline output dirs ──────────────────────────────────────────────────
    pipeline_out_dirs = {
        name: img_out / PIPELINE_DIRS[name]
        for name in PIPELINE_DIRS
    }

    # ── Run all six pipelines ─────────────────────────────────────────────────
    pipeline_results = {
        "P1 — Raw CNN":        [],
        "P2 — One-step YOLO":  [],
        "P3 — Stage1 + CNN":   [],
        "P4 — Stage1 + YOLO":  [],
        "P5 — Raw MLP":        [],
        "P6 — Stage1 + MLP":   [],
    }

    per_image_preds = []

    print(f"\nEvaluating {len(all_images)} images...")
    for i, item in enumerate(all_images):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_images)}...")

        img = cv2.imread(str(item["img_path"]))
        if img is None:
            continue

        true_cls = item["true_cls"]
        fname    = item["img_path"].name

        img_preds = {"img_path": item["img_path"], "true_cls": true_cls,
                     "fname": fname, "img": img, "preds": {}}

        # P1 — Raw CNN
        pred1, conf1 = run_raw_cnn(
            img, m["cnn"], m["class_names"], m["transform"], m["cnn_device"]
        )
        pipeline_results["P1 — Raw CNN"].append(
            {"true": true_cls, "pred": pred1, "conf": conf1,
             "method": "raw_cnn", "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred1, conf1,
                          "raw_cnn", pipeline_out_dirs["P1 — Raw CNN"])
        img_preds["preds"]["P1 — Raw CNN"] = {"pred": pred1, "conf": conf1}

        # P2 — One-step YOLO
        pred2, conf2, bbox2 = run_onestep(img, m["onestep"])
        pipeline_results["P2 — One-step YOLO"].append(
            {"true": true_cls, "pred": pred2, "conf": conf2,
             "method": "onestep", "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred2, conf2,
                          "onestep", pipeline_out_dirs["P2 — One-step YOLO"],
                          bbox=bbox2)
        img_preds["preds"]["P2 — One-step YOLO"] = {"pred": pred2, "conf": conf2}

        # P3 — Stage1 + CNN
        pred3, conf3, method3, bbox3 = run_twostep(
            img, m["stage1"], m["cnn"], m["class_names"],
            m["transform"], m["cnn_device"], geo_rule
        )
        pipeline_results["P3 — Stage1 + CNN"].append(
            {"true": true_cls, "pred": pred3, "conf": conf3,
             "method": method3, "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred3, conf3,
                          method3, pipeline_out_dirs["P3 — Stage1 + CNN"],
                          bbox=bbox3)
        img_preds["preds"]["P3 — Stage1 + CNN"] = {"pred": pred3, "conf": conf3}

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
        img_preds["preds"]["P4 — Stage1 + YOLO"] = {"pred": pred4, "conf": conf4}

        # P5 — Raw MLP (full image → MediaPipe → MLP)
        pred5, conf5, method5 = run_mlp_on_image(img, pose_landmarker, mlp, scaler)
        pipeline_results["P5 — Raw MLP"].append(
            {"true": true_cls, "pred": pred5, "conf": conf5,
             "method": method5, "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred5, conf5,
                          method5, pipeline_out_dirs["P5 — Raw MLP"])
        img_preds["preds"]["P5 — Raw MLP"] = {"pred": pred5, "conf": conf5}

        # P6 — Stage1 + MLP (Stage1 crop → MediaPipe → MLP)
        pred6, conf6, method6, bbox6 = run_stage1_mlp(
            img, m["stage1"], pose_landmarker, mlp, scaler, geo_rule
        )
        pipeline_results["P6 — Stage1 + MLP"].append(
            {"true": true_cls, "pred": pred6, "conf": conf6,
             "method": method6, "filename": fname}
        )
        save_result_image(img, fname, true_cls, pred6, conf6,
                          method6, pipeline_out_dirs["P6 — Stage1 + MLP"],
                          bbox=bbox6)
        img_preds["preds"]["P6 — Stage1 + MLP"] = {"pred": pred6, "conf": conf6}

        per_image_preds.append(img_preds)

    # ── Close MediaPipe ───────────────────────────────────────────────────────
    pose_landmarker.close()

    # ── Save wrong-in-all-pipelines images ────────────────────────────────────
    wrong_all_count = 0
    wrong_mlp_count = 0

    for entry in per_image_preds:
        preds    = entry["preds"]
        true_cls = entry["true_cls"]

        all_wrong = all(p["pred"] != true_cls for p in preds.values())
        if all_wrong:
            wrong_all_count += 1
            save_flagged_image(
                entry["img"], entry["fname"], true_cls, preds,
                wrong_all_dir, "WRONG IN ALL PIPELINES",
            )

        # MLP wrong AND at least one non-MLP pipeline also wrong
        mlp_wrong = any(
            preds[k]["pred"] != true_cls for k in MLP_PIPELINES if k in preds
        )
        non_mlp_wrong = any(
            preds[k]["pred"] != true_cls
            for k in preds if k not in MLP_PIPELINES
        )
        if mlp_wrong and non_mlp_wrong:
            wrong_mlp_count += 1
            save_flagged_image(
                entry["img"], entry["fname"], true_cls, preds,
                wrong_mlp_dir, "WRONG IN MLP + OTHER",
            )

    print(f"\n  Wrong in ALL pipelines:     {wrong_all_count}/{len(per_image_preds)} images")
    print(f"  Wrong in MLP + other:       {wrong_mlp_count}/{len(per_image_preds)} images")

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
        acc_pct = f"{correct/total:.1%}" if total > 0 else "N/A"
        summary_lines.append(f"{name}: {correct}/{total} = {acc_pct}")

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
        for sub in ["correct", "wrong"]:
            folder = img_out / pipeline_dir_name / sub
            count  = len(list(folder.glob("*"))) if folder.exists() else 0
            print(f"    {sub:<30} {count:>5} images")
    wac = len(list(wrong_all_dir.glob("*"))) if wrong_all_dir.exists() else 0
    wmc = len(list(wrong_mlp_dir.glob("*"))) if wrong_mlp_dir.exists() else 0
    print(f"\n  wrong_all_pipelines/         {wac:>5} images")
    print(f"  wrong_mlp_and_others/        {wmc:>5} images")

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
        print(f"  CSV -> {csv_path.name}")

    # Wrong-all CSV
    wrong_all_csv = csv_out / "isimage_wrong_all_pipelines.csv"
    with open(wrong_all_csv, "w", newline="") as f:
        fields = ["filename", "true_cls",
                   "p1_pred", "p1_conf", "p2_pred", "p2_conf",
                   "p3_pred", "p3_conf", "p4_pred", "p4_conf",
                   "p5_pred", "p5_conf", "p6_pred", "p6_conf"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in per_image_preds:
            all_wrong = all(p["pred"] != entry["true_cls"] for p in entry["preds"].values())
            if all_wrong:
                row = {"filename": entry["fname"], "true_cls": entry["true_cls"]}
                for j, (_, info) in enumerate(entry["preds"].items(), 1):
                    row[f"p{j}_pred"] = info["pred"]
                    row[f"p{j}_conf"] = f"{info['conf']:.4f}"
                writer.writerow(row)
    print(f"  CSV -> {wrong_all_csv.name}")

    # Wrong MLP+others CSV
    wrong_mlp_csv = csv_out / "isimage_wrong_mlp_and_others.csv"
    with open(wrong_mlp_csv, "w", newline="") as f:
        fields = ["filename", "true_cls",
                   "p1_pred", "p1_conf", "p2_pred", "p2_conf",
                   "p3_pred", "p3_conf", "p4_pred", "p4_conf",
                   "p5_pred", "p5_conf", "p6_pred", "p6_conf"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in per_image_preds:
            preds = entry["preds"]
            true_cls = entry["true_cls"]
            mlp_wrong = any(preds[k]["pred"] != true_cls for k in MLP_PIPELINES if k in preds)
            non_mlp_wrong = any(preds[k]["pred"] != true_cls for k in preds if k not in MLP_PIPELINES)
            if mlp_wrong and non_mlp_wrong:
                row = {"filename": entry["fname"], "true_cls": true_cls}
                for j, (_, info) in enumerate(preds.items(), 1):
                    row[f"p{j}_pred"] = info["pred"]
                    row[f"p{j}_conf"] = f"{info['conf']:.4f}"
                writer.writerow(row)
    print(f"  CSV -> {wrong_mlp_csv.name}")

    # ── Save text summary ─────────────────────────────────────────────────────
    summary_path = OUT_DIR / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("IS_Image Confusion Matrix — Summary\n")
        f.write("=" * 50 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nWrong in ALL pipelines: {wrong_all_count}/{len(per_image_preds)}\n")
        f.write(f"Wrong in MLP + other:  {wrong_mlp_count}/{len(per_image_preds)}\n")
    print(f"\n  Summary -> {summary_path.name}")
    print(f"  All outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

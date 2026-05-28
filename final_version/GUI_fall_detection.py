"""
gui_fall_detection.py  (V3 — Six Pipeline Update)
===================================================
PyQt5 live webcam fall detection GUI.

Supports six pipeline modes selectable from dropdown:
  P1: Raw CNN       — full image → EfficientNet-B0
  P2: One-step YOLO — full image → swook_techcare YOLO
  P3: Stage1 + CNN  — Stage1 crop → EfficientNet-B0  (recommended)
  P4: Stage1 + YOLO — Stage1 crop → one-step YOLO
  P5: Raw MLP       — full image → MediaPipe → MLP
  P6: Stage1 + MLP  — Stage1 crop → MediaPipe → MLP

TransitionDetector is available as an optional toggle per pipeline.

Usage:
    python V2/src/GUI/gui_fall_detection.py
"""

import sys
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QComboBox, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QGroupBox, QStatusBar, QCheckBox,
)
from PyQt5.QtCore  import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui   import QImage, QPixmap, QFont

try:
    from playsound import playsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

# =============================================================================
#  PATHS  (self-contained — works both as .py and as PyInstaller .exe)
# =============================================================================

if getattr(sys, 'frozen', False):
    FINAL_BASE = Path(sys._MEIPASS)
else:
    FINAL_BASE = Path(__file__).resolve().parent

MODELS_DIR   = FINAL_BASE / "models"

ONESTEP_PATH    = MODELS_DIR / "swook_techcare_fall_sit_walk.pt"
STAGE1_PATH     = MODELS_DIR / "stage1_person.pt"
CNN_PATH        = MODELS_DIR / "stage2_cnn.pt"
CNN_INFO_PATH   = MODELS_DIR / "stage2_cnn_info.json"
MLP_PATH        = MODELS_DIR / "stage2_pose_mlp_v2.pt"
SCALER_PATH     = MODELS_DIR / "pose_scaler_v2.pkl"
LANDMARKER_PATH = MODELS_DIR / "pose_landmarker_heavy.task"
ALERT_SOUND     = FINAL_BASE / "assets"  / "alert.wav"

# Pipeline utils (local)
sys.path.insert(0, str(FINAL_BASE / "utils"))
from pipeline_utils import (
    pad_bbox, predict_crop, load_cnn, GeometricFallRule,
    STAGE1_CONF, CLASS_NAMES,
)
from transition_detector import TransitionDetector

# =============================================================================
#  CONSTANTS
# =============================================================================

YOLO_CONF   = 0.35
YOLO_IMGSZ  = 640
DEVICE      = 0

ALERT_HOLD_SEC   = 3.0
CONFIRMED_THRESH = 0.70

CLS_FALL = "Fall"; CLS_SIT = "Sit"; CLS_WALK = "Walk"
BOX_COLS = {CLS_FALL: (0,40,200), CLS_SIT: (0,165,255), CLS_WALK: (60,200,80)}

PIPELINE_OPTIONS = [
    ("P3: Stage1 + CNN  (recommended)", "P3"),
    ("P1: Raw CNN",                      "P1"),
    ("P2: One-step YOLO",                "P2"),
    ("P4: Stage1 + YOLO",                "P4"),
    ("P5: Raw MLP",                      "P5"),
    ("P6: Stage1 + MLP",                 "P6"),
]

# =============================================================================
#  MLP HELPERS
# =============================================================================

def _angle_between(p1, p2, p3) -> float:
    v1 = np.array([p1.x-p2.x, p1.y-p2.y])
    v2 = np.array([p3.x-p2.x, p3.y-p2.y])
    cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a,-1,1))))


def extract_pose_features(landmarks):
    if landmarks is None:
        return None
    lm = landmarks.landmark
    l_sh=lm[11]; r_sh=lm[12]; l_hi=lm[23]; r_hi=lm[24]
    l_kn=lm[25]; r_kn=lm[26]; l_an=lm[27]; r_an=lm[28]; nose=lm[0]
    msx=(l_sh.x+r_sh.x)/2; msy=(l_sh.y+r_sh.y)/2
    spine_angle=float(np.degrees(np.arctan2(abs(msx-(l_hi.x+r_hi.x)/2),abs(msy-(l_hi.y+r_hi.y)/2)+1e-9)))
    hip_h=(l_hi.y+r_hi.y)/2
    bh=abs(msy-(l_an.y+r_an.y)/2)+1e-9
    sw=abs(l_sh.x-r_sh.x)/bh
    kbl=_angle_between(l_hi,l_kn,l_an); kbr=_angle_between(r_hi,r_kn,r_an)
    xs=[lm[i].x for i in [11,12,23,24,25,26,27,28]]
    ys=[lm[i].y for i in [11,12,23,24,25,26,27,28]]
    bar=(max(xs)-min(xs))/(max(ys)-min(ys)+1e-9)
    ahv=abs((l_an.y+r_an.y)/2-hip_h)/(bh+1e-9)
    return [spine_angle,hip_h,sw,kbl,kbr,bar,nose.y,ahv]


class PoseMLP(nn.Module):
    def __init__(self, input_dim=8, hidden=128, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x): return self.net(x)


SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
SKELETON_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def run_mlp_on_image(img_bgr, mlp, scaler, pose_model, mlp_device):
    if mlp is None:
        return "NoDetection", 0.0, None
    try:
        from mediapipe.tasks.python import vision as mp_vision
        rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp_vision.Image(image_format=mp_vision.ImageFormat.SRGB, data=rgb)
        result = pose_model.detect(mp_img)
        if not result.pose_landmarks:
            return "NoDetection", 0.0, None

        raw_lms = result.pose_landmarks[0]

        class _FakeLM:
            def __init__(self, x, y, z, v): self.x=x; self.y=y; self.z=z; self.visibility=v
        class _FakeLandmarks:
            def __init__(self, lms): self.landmark=[_FakeLM(l.x,l.y,l.z,l.visibility) for l in lms]

        fake = _FakeLandmarks(raw_lms)
        features = extract_pose_features(fake)
        if features is None:
            return "NoDetection", 0.0, None

        landmarks_out = [(l.x, l.y, l.visibility) for l in raw_lms]

        feat_norm = scaler.transform([features])
        with torch.no_grad():
            tensor = torch.tensor(feat_norm, dtype=torch.float32).to(mlp_device)
            probs  = torch.softmax(mlp(tensor), dim=1)[0]
            idx    = int(probs.argmax())
            return CLASS_NAMES[idx], float(probs[idx]), landmarks_out
    except Exception:
        return "NoDetection", 0.0, None

# =============================================================================
#  INFERENCE THREAD
# =============================================================================

class InferenceThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list, bool, float, str)
    # (annotated_frame, boxes, is_fall, fall_conf, pipeline_id)

    def __init__(self):
        super().__init__()
        self.camera_id      = 0
        self.pipeline_id    = "P3"
        self.conf           = YOLO_CONF
        self.temporal_on    = False
        self.running        = False
        self._models        = {}
        self._geo_rule      = GeometricFallRule(aspect_threshold=2.2)
        self._transition    = TransitionDetector(
            window_size=15, min_fall_frames=3,
            walk_stability_frames=5, alert_hold_sec=ALERT_HOLD_SEC,
            confirmed_conf_thresh=CONFIRMED_THRESH,
        )
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        from ultralytics import YOLO

        if ONESTEP_PATH.exists():
            self._models["onestep"] = YOLO(str(ONESTEP_PATH))
            print(f"  [OK] One-step : {ONESTEP_PATH.name}")

        if STAGE1_PATH.exists():
            self._models["stage1"] = YOLO(str(STAGE1_PATH))
            print(f"  [OK] Stage1   : {STAGE1_PATH.name}")

        if CNN_PATH.exists():
            cnn, class_names, transform, cnn_device = load_cnn(CNN_PATH, CNN_INFO_PATH)
            self._models["cnn"]         = cnn
            self._models["class_names"] = class_names
            self._models["transform"]   = transform
            self._models["cnn_device"]  = cnn_device

        if MLP_PATH.exists() and SCALER_PATH.exists() and LANDMARKER_PATH.exists():
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                mlp    = PoseMLP()
                mlp.load_state_dict(torch.load(str(MLP_PATH), map_location=device, weights_only=True))
                mlp    = mlp.to(device); mlp.eval()
                with open(SCALER_PATH, "rb") as f:
                    scaler = pickle.load(f)
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python.core import base_options as mp_base
                base_opts  = mp_base.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
                pose_opts  = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                self._models["mlp"]         = mlp
                self._models["scaler"]      = scaler
                self._models["pose_model"]  = mp_vision.PoseLandmarker.create_from_options(pose_opts)
                self._models["mlp_device"]  = device
                print(f"  [OK] MLP/pose loaded")
            except Exception as e:
                print(f"  [WARN] MLP load failed: {e}")
                self._models["mlp"] = None

    # ── Frame inference ───────────────────────────────────────────────────────

    def _run_frame(self, frame):
        m  = self._models
        gr = self._geo_rule
        pid = self.pipeline_id
        h, w = frame.shape[:2]

        if pid == "P1":
            from pipeline_utils import IMG_SIZE as CNN_IMG_SIZE
            resized = cv2.resize(frame, (CNN_IMG_SIZE, CNN_IMG_SIZE))
            cls, conf, _ = predict_crop(m["cnn"], m["class_names"], m["transform"], m["cnn_device"], resized)
            return [{"class_name": cls, "conf": conf, "xyxy": [0,0,w,h]}] if cls != "NoDetection" else []

        elif pid == "P2":
            r = m["onestep"].predict(source=frame, conf=self.conf, imgsz=YOLO_IMGSZ, device=DEVICE, verbose=False)[0]
            boxes = []
            if r.boxes is not None:
                for box in r.boxes:
                    name = m["onestep"].names[int(box.cls[0])]
                    name = {"fall":"Fall","sit":"Sit","walk":"Walk"}.get(name.lower(), name)
                    boxes.append({"class_name":name, "conf":float(box.conf[0]), "xyxy":box.xyxy[0].cpu().numpy().tolist()})
            return boxes

        elif pid in ("P3", "P4", "P6"):
            r = m["stage1"].predict(source=frame, conf=STAGE1_CONF, imgsz=YOLO_IMGSZ, device=DEVICE, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return []
            areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in r.boxes.xyxy.cpu().numpy()]
            best_i = int(np.argmax(areas))
            x1,y1,x2,y2 = map(int, r.boxes.xyxy[best_i].cpu().numpy())
            x1,y1,x2,y2 = pad_bbox(x1,y1,x2,y2,w,h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return []
            if gr.check(bbox=(x1,y1,x2,y2)):
                return [{"class_name":"Fall","conf":0.80,"xyxy":[x1,y1,x2,y2]}]

            if pid == "P3":
                cls, conf, _ = predict_crop(m["cnn"], m["class_names"], m["transform"], m["cnn_device"], crop)
                return [{"class_name":cls,"conf":conf,"xyxy":[x1,y1,x2,y2]}] if cls != "NoDetection" else []

            elif pid == "P4":
                r2 = m["onestep"].predict(source=crop, conf=self.conf, imgsz=YOLO_IMGSZ, device=DEVICE, verbose=False)[0]
                if r2.boxes is None or len(r2.boxes) == 0:
                    return []
                confs2 = r2.boxes.conf.cpu().numpy()
                bi2    = int(np.argmax(confs2))
                name   = m["onestep"].names[int(r2.boxes.cls[bi2].cpu().numpy())]
                name   = {"fall":"Fall","sit":"Sit","walk":"Walk"}.get(name.lower(), name)
                return [{"class_name":name,"conf":float(confs2[bi2]),"xyxy":[x1,y1,x2,y2]}]

            elif pid == "P6":
                cls, conf, lms = run_mlp_on_image(crop, m.get("mlp"), m.get("scaler"), m.get("pose_model"), m.get("mlp_device"))
                if cls == "NoDetection":
                    return []
                return [{"class_name":cls,"conf":conf,"xyxy":[x1,y1,x2,y2],
                         "landmarks":lms,"lm_offset":(x1,y1),"lm_size":(x2-x1,y2-y1)}]

        elif pid == "P5":
            cls, conf, lms = run_mlp_on_image(frame, m.get("mlp"), m.get("scaler"), m.get("pose_model"), m.get("mlp_device"))
            if cls == "NoDetection":
                return []
            return [{"class_name":cls,"conf":conf,"xyxy":[0,0,w,h],
                     "landmarks":lms,"lm_offset":(0,0),"lm_size":(w,h)}]

        return []

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                boxes = self._run_frame(frame)
            except Exception as e:
                print(f"[InferenceThread] Error: {e}")
                boxes = []

            is_fall   = any(b["class_name"] == CLS_FALL for b in boxes)
            fall_conf = max((b["conf"] for b in boxes if b["class_name"] == CLS_FALL), default=0.0)

            if self.temporal_on:
                cls_id  = 0 if is_fall else (1 if any(b["class_name"]==CLS_SIT  for b in boxes) else 2)
                geo_fired = self._geo_rule.check(bbox=tuple(map(int, boxes[0]["xyxy"]))) if boxes else False
                alert_lvl, sc = self._transition.update(cls_id, fall_conf, geo_fired)
                is_fall   = alert_lvl in ("confirmed", "possible")
                fall_conf = sc

            self.frame_ready.emit(frame, boxes, is_fall, fall_conf, self.pipeline_id)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def set_pipeline(self, pipeline_id):
        self.pipeline_id = pipeline_id
        self._transition.reset()

    def set_temporal(self, on: bool):
        self.temporal_on = on
        self._transition.reset()

# =============================================================================
#  MAIN WINDOW
# =============================================================================

class FallDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COS30018 — Fall Detection System  (V3 Six-Pipeline)")
        self.setMinimumSize(1300, 800)
        self._last_fall_t  = -999.0
        self._alert_active = False
        self._fall_count   = 0
        self._flash_state  = False
        self._setup_ui()
        self._inference = InferenceThread()
        self._inference.frame_ready.connect(self._on_frame)
        self._inference.start()

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Video feed
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: black;")
        main_layout.addWidget(self.video_label, stretch=3)

        # Controls panel
        ctrl  = QWidget(); ctrl.setMaximumWidth(290)
        clo   = QVBoxLayout(ctrl); main_layout.addWidget(ctrl)

        # Pipeline selector
        pg = QGroupBox("Detection Pipeline"); pl = QVBoxLayout(pg)
        self.pipeline_combo = QComboBox()
        for label, pid in PIPELINE_OPTIONS:
            self.pipeline_combo.addItem(label, pid)
        self.pipeline_combo.currentIndexChanged.connect(self._on_pipeline_change)
        pl.addWidget(self.pipeline_combo)
        self.temporal_check = QCheckBox("Enable TransitionDetector")
        self.temporal_check.setChecked(False)
        self.temporal_check.stateChanged.connect(self._on_temporal_change)
        pl.addWidget(self.temporal_check)
        clo.addWidget(pg)

        # Confidence slider
        cg = QGroupBox("Confidence Threshold"); cl = QVBoxLayout(cg)
        self.conf_label  = QLabel(f"{YOLO_CONF:.0%}")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90); self.conf_slider.setValue(int(YOLO_CONF*100))
        self.conf_slider.valueChanged.connect(self._on_conf_change)
        cl.addWidget(self.conf_label); cl.addWidget(self.conf_slider)
        clo.addWidget(cg)

        # Alert status
        ag = QGroupBox("Alert Status"); al = QVBoxLayout(ag)
        self.alert_label = QLabel("● MONITORING")
        self.alert_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.alert_label.setStyleSheet("color: #2E7D32;")
        al.addWidget(self.alert_label)
        self.fall_count_label = QLabel("Falls detected: 0")
        al.addWidget(self.fall_count_label)
        clo.addWidget(ag)

        # Current detection
        dg = QGroupBox("Current Detection"); dl = QVBoxLayout(dg)
        self.class_label = QLabel("Class: —")
        self.conf_val_label = QLabel("Confidence: —")
        self.pipeline_label = QLabel("Pipeline: P3")
        dl.addWidget(self.class_label)
        dl.addWidget(self.conf_val_label)
        dl.addWidget(self.pipeline_label)
        clo.addWidget(dg)

        clo.addStretch()

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready — P3: Stage1 + CNN")

        self._flash_timer = QTimer()
        self._flash_timer.timeout.connect(self._flash_tick)

    # ── Slot handlers ─────────────────────────────────────────────────────────

    def _on_pipeline_change(self, idx):
        pid   = self.pipeline_combo.itemData(idx)
        label = self.pipeline_combo.currentText()
        self._inference.set_pipeline(pid)
        self.pipeline_label.setText(f"Pipeline: {pid}")
        self.status_bar.showMessage(f"Pipeline: {label}")

    def _on_temporal_change(self, state):
        on = (state == Qt.Checked)
        self._inference.set_temporal(on)
        self.status_bar.showMessage(
            "TransitionDetector ON" if on else "TransitionDetector OFF"
        )

    def _on_conf_change(self, val):
        conf = val / 100.0
        self._inference.conf = conf
        self.conf_label.setText(f"{conf:.0%}")

    def _on_frame(self, frame, boxes, is_fall, fall_conf, pipeline_id):
        now = time.time()

        # Draw boxes or skeleton
        annotated = frame.copy()
        for b in boxes:
            cls  = b["class_name"]; conf = b["conf"]
            x1,y1,x2,y2 = map(int, b["xyxy"])
            color = BOX_COLS.get(cls, (200,200,200))

            if "landmarks" in b and b["landmarks"] is not None:
                lms = b["landmarks"]
                ox, oy = b["lm_offset"]
                lw_r, lh_r = b["lm_size"]
                vis_thresh = 0.3

                for i1, i2 in SKELETON_CONNECTIONS:
                    if i1 >= len(lms) or i2 >= len(lms):
                        continue
                    lx1, ly1, v1 = lms[i1]
                    lx2, ly2, v2 = lms[i2]
                    if v1 < vis_thresh or v2 < vis_thresh:
                        continue
                    pt1 = (int(lx1 * lw_r + ox), int(ly1 * lh_r + oy))
                    pt2 = (int(lx2 * lw_r + ox), int(ly2 * lh_r + oy))
                    cv2.line(annotated, pt1, pt2, color, 3, cv2.LINE_AA)

                for ji in SKELETON_JOINTS:
                    if ji >= len(lms):
                        continue
                    lx, ly, v = lms[ji]
                    if v < vis_thresh:
                        continue
                    pt = (int(lx * lw_r + ox), int(ly * lh_r + oy))
                    cv2.circle(annotated, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, pt, 5, color, 2, cv2.LINE_AA)

                nose_lm = lms[0] if len(lms) > 0 else None
                if nose_lm and nose_lm[2] >= vis_thresh:
                    tx = int(nose_lm[0] * lw_r + ox) - 20
                    ty = int(nose_lm[1] * lh_r + oy) - 15
                else:
                    tx, ty = x1, y1 - 10
                lbl = f"{cls} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (tx-2, ty-th-4), (tx+tw+4, ty+4), color, -1)
                cv2.putText(annotated, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2, cv2.LINE_AA)
            else:
                thick = 3 if cls == CLS_FALL else 2
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thick)
                lbl = f"{cls} {conf:.0%}"
                (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1,y1-lh-8), (x1+lw+6,y1), color, -1)
                cv2.putText(annotated, lbl, (x1+3,y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255,255,255), 2, cv2.LINE_AA)

        if is_fall:
            badge = "  ⚠ FALL DETECTED  "
            (bw,bh),_ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            bx = (annotated.shape[1] - bw) // 2
            cv2.rectangle(annotated, (bx-6,8), (bx+bw+6,bh+30), (0,40,200), -1)
            cv2.putText(annotated, badge, (bx,bh+20), cv2.FONT_HERSHEY_DUPLEX,
                        0.9, (255,255,255), 2, cv2.LINE_AA)

            if (now - self._last_fall_t) > ALERT_HOLD_SEC:
                self._fall_count += 1
                self._last_fall_t  = now
                self.fall_count_label.setText(f"Falls detected: {self._fall_count}")
                if HAS_SOUND and ALERT_SOUND.exists():
                    threading.Thread(target=playsound, args=(str(ALERT_SOUND),), daemon=True).start()
                self._alert_active = True
                self._flash_timer.start(300)

        if self._alert_active and (now - self._last_fall_t) > ALERT_HOLD_SEC:
            self._alert_active = False
            self._flash_timer.stop()
            self.alert_label.setText("● MONITORING")
            self.alert_label.setStyleSheet("color: #2E7D32;")

        # Update stats
        if boxes:
            top = max(boxes, key=lambda b: b["conf"])
            self.class_label.setText(f"Class: {top['class_name']}")
            self.conf_val_label.setText(f"Confidence: {top['conf']:.1%}")
        else:
            self.class_label.setText("Class: —")
            self.conf_val_label.setText("Confidence: —")

        # Display
        h, w, c = annotated.shape
        rgb    = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
        scaled = qt_img.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _flash_tick(self):
        self._flash_state = not self._flash_state
        if self._flash_state:
            self.alert_label.setText("⚠  FALL DETECTED")
            self.alert_label.setStyleSheet("color: #C0392B; font-weight: bold;")
        else:
            self.alert_label.setText("⚠  FALL DETECTED")
            self.alert_label.setStyleSheet("color: #E74C3C;")

    def closeEvent(self, event):
        self._inference.stop()
        super().closeEvent(event)


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FallDetectionGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
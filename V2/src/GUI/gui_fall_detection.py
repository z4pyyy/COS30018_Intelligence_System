"""
gui_fall_detection.py
PyQt5 live webcam fall detection GUI.
Supports one-step (YOLOv8) and two-step (YOLOv8 + MediaPipe MLP) modes.
Saves 6-second fall clips (3s before + 3s after) to V2/Fall_Detected_GUI/.

Usage:
    python V2/src/GUI/gui_fall_detection.py
"""
import sys
import time
import pickle
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                              QComboBox, QPushButton, QHBoxLayout, QVBoxLayout,
                              QSlider, QGroupBox, QStatusBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

BASE = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BASE.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline_utils import (
    pad_bbox, extract_pose_features, GeometricFallRule,
    STAGE1_CONF, CLASS_NAMES,
)
from transition_detector import TransitionDetector

ONESTEP_MODEL = str(BASE / "models" / "swook_techcare_fall_sit_walk.pt")
STAGE1_MODEL = str(BASE / "models" / "stage1_person.pt")
STAGE2_MODEL = str(BASE / "models" / "stage2_pose_mlp_v2.pt")
STAGE2_SCALER = str(BASE / "models" / "pose_scaler_v2.pkl")
POSE_MODEL_PATH = str(BASE / "models" / "pose_landmarker_heavy.task")
ALERT_SOUND = str(BASE / "assets" / "alert.wav")
FALL_SAVE_DIR = BASE / "Fall_Detected_GUI"

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]

BOX_COLORS = {"Fall": (0, 40, 200), "Sit": (0, 165, 255), "Walk": (60, 200, 80)}
ALERT_HOLD_S = 3.0
CLIP_PRE_S = 3.0
CLIP_POST_S = 3.0
ASSUMED_FPS = 15.0


def draw_skeleton_on_frame(frame, pose_landmarks, crop_x1, crop_y1, crop_x2, crop_y2, color=(0, 220, 0)):
    """Draw MediaPipe skeleton on full frame, mapping crop-relative landmarks to frame coords."""
    if not pose_landmarks:
        return
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w <= 0 or crop_h <= 0:
        return
    h, w = frame.shape[:2]
    points = {}
    for i, lm in enumerate(pose_landmarks):
        px = max(0, min(w - 1, int(crop_x1 + lm.x * crop_w)))
        py = max(0, min(h - 1, int(crop_y1 + lm.y * crop_h)))
        points[i] = (px, py)
        if lm.visibility > 0.3:
            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            lm_a, lm_b = pose_landmarks[a], pose_landmarks[b]
            if lm_a.visibility > 0.3 and lm_b.visibility > 0.3:
                cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)


class InferenceThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list, bool, float)

    def __init__(self):
        super().__init__()
        self.camera_id = 0
        self.mode = "one-step"
        self.conf = 0.45
        self.running = False
        self._geo_rule = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)
        self._load_models()

    def _load_models(self):
        self.onestep = YOLO(ONESTEP_MODEL) if Path(ONESTEP_MODEL).exists() else None
        self.stage1 = YOLO(STAGE1_MODEL) if Path(STAGE1_MODEL).exists() else None

        if Path(STAGE2_MODEL).exists() and Path(STAGE2_SCALER).exists():
            sys.path.insert(0, str(BASE / "src"))
            from train_stage2_pose import PoseMLP
            self.mlp = PoseMLP()
            self.mlp.load_state_dict(torch.load(STAGE2_MODEL, map_location="cpu"))
            self.mlp.eval()
            with open(STAGE2_SCALER, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.mlp = None
            self.scaler = None

        if Path(POSE_MODEL_PATH).exists():
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
            )
            self.pose_landmarker = PoseLandmarker.create_from_options(options)
        else:
            self.pose_landmarker = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            detections = []
            is_fall = False
            fall_conf = 0.0

            if self.mode == "one-step" and self.onestep:
                r = self.onestep.predict(source=frame, conf=self.conf,
                                         imgsz=640, device=0, verbose=False)[0]
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_name = self.onestep.names[int(box.cls[0])]
                        conf_val = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        detections.append({"class": cls_name, "conf": conf_val, "xyxy": xyxy})
                        if cls_name == "Fall" and conf_val > fall_conf:
                            is_fall = True
                            fall_conf = conf_val

            elif self.mode == "two-step" and self.stage1 and self.mlp and self.pose_landmarker:
                r1 = self.stage1.predict(source=frame, conf=STAGE1_CONF,
                                         imgsz=640, device=0, verbose=False)[0]
                if r1.boxes is not None:
                    h, w = frame.shape[:2]
                    for box in r1.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        result = self.pose_landmarker.detect(mp_image)

                        if result.pose_landmarks and len(result.pose_landmarks) > 0:
                            draw_skeleton_on_frame(
                                frame, result.pose_landmarks[0],
                                x1, y1, x2, y2,
                            )

                        feats, core_vis = extract_pose_features(
                            result.pose_landmarks[0] if result.pose_landmarks else None
                        )

                        method = "no_pose"
                        if core_vis is not None and self._geo_rule.check(
                            bbox=(x1, y1, x2, y2), keypoint_visibilities=core_vis
                        ):
                            detections.append({"class": "Fall", "conf": 0.75,
                                               "xyxy": [x1, y1, x2, y2],
                                               "method": "geo_rule"})
                            is_fall = True
                            fall_conf = max(fall_conf, 0.75)
                            continue

                        if feats is None:
                            detections.append({"class": "Person", "conf": float(box.conf[0]),
                                               "xyxy": box.xyxy[0].cpu().numpy().tolist(),
                                               "method": "no_pose"})
                            continue

                        method = "pose_mlp"
                        fn = self.scaler.transform([feats])
                        with torch.no_grad():
                            probs = torch.softmax(
                                self.mlp(torch.tensor(fn, dtype=torch.float32)), dim=1
                            )[0]
                            pred_id = int(probs.argmax())
                            conf_val = float(probs[pred_id])
                        cls_name = CLASS_NAMES[pred_id]
                        detections.append({"class": cls_name, "conf": conf_val,
                                           "xyxy": [x1, y1, x2, y2],
                                           "method": method})
                        if cls_name == "Fall" and conf_val > fall_conf:
                            is_fall = True
                            fall_conf = conf_val

            self.frame_ready.emit(frame, detections, is_fall, fall_conf)

        cap.release()
        if self.pose_landmarker:
            self.pose_landmarker.close()

    def stop(self):
        self.running = False
        self.wait()


class FallClipRecorder:
    """Records fall clips: dumps pre-buffer then captures post-fall frames."""

    def __init__(self, save_dir, pre_seconds=3.0, post_seconds=3.0, fps=15.0):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._pre_seconds = pre_seconds
        self._post_seconds = post_seconds
        self._fps = fps
        buf_size = int(pre_seconds * fps)
        self._buffer = deque(maxlen=max(buf_size, 1))
        self._recording = False
        self._writer = None
        self._record_until = 0.0
        self._lock = threading.Lock()

    def update_fps(self, fps):
        with self._lock:
            self._fps = fps
            buf_size = int(self._pre_seconds * fps)
            old = list(self._buffer)
            self._buffer = deque(old, maxlen=max(buf_size, 1))

    def feed(self, annotated_frame):
        with self._lock:
            if self._recording:
                self._writer.write(annotated_frame)
                if time.time() >= self._record_until:
                    self._writer.release()
                    self._writer = None
                    self._recording = False
            else:
                self._buffer.append(annotated_frame.copy())

    def trigger(self, mode, fall_conf):
        with self._lock:
            if self._recording:
                self._record_until = time.time() + self._post_seconds
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_{ts}_{mode}_conf{int(fall_conf*100)}pct.mp4"
            path = self._save_dir / filename
            h, w = 720, 1280
            if self._buffer:
                h, w = self._buffer[-1].shape[:2]
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(path), codec, self._fps, (w, h))
            for frame in self._buffer:
                fh, fw = frame.shape[:2]
                if fw != w or fh != h:
                    frame = cv2.resize(frame, (w, h))
                self._writer.write(frame)
            self._buffer.clear()
            self._recording = True
            self._record_until = time.time() + self._post_seconds

    def finalize(self):
        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None
            self._recording = False


class FallDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COS30018 — Fall Detection System")
        self.setMinimumSize(1280, 780)
        self._last_fall_t = -999.0
        self._alert_active = False
        self._fall_count = 0
        self._volume = 80
        self._frame_times = deque(maxlen=30)
        self._setup_ui()
        self._setup_audio()
        self._recorder = FallClipRecorder(
            FALL_SAVE_DIR, pre_seconds=CLIP_PRE_S,
            post_seconds=CLIP_POST_S, fps=ASSUMED_FPS,
        )
        self._inference = InferenceThread()
        self._inference.frame_ready.connect(self._on_frame)
        self._inference.start()

    def _setup_audio(self):
        self._player = QMediaPlayer()
        self._player.setVolume(self._volume)
        if Path(ALERT_SOUND).exists():
            self._player.setMedia(QMediaContent(QUrl.fromLocalFile(ALERT_SOUND)))

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: black;")
        main_layout.addWidget(self.video_label, stretch=3)

        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_panel.setMaximumWidth(280)
        main_layout.addWidget(ctrl_panel)

        model_group = QGroupBox("Detection Mode")
        model_inner = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["One-Step (YOLOv8)", "Two-Step (Pose-Based)"])
        self.model_combo.currentIndexChanged.connect(self._on_mode_change)
        model_inner.addWidget(self.model_combo)
        ctrl_layout.addWidget(model_group)

        conf_group = QGroupBox("Confidence Threshold")
        conf_inner = QVBoxLayout(conf_group)
        self.conf_label = QLabel("0.45")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(45)
        self.conf_slider.valueChanged.connect(self._on_conf_change)
        conf_inner.addWidget(self.conf_label)
        conf_inner.addWidget(self.conf_slider)
        ctrl_layout.addWidget(conf_group)

        vol_group = QGroupBox("Alert Volume")
        vol_inner = QVBoxLayout(vol_group)
        self.vol_label = QLabel(f"{self._volume}%")
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(self._volume)
        self.vol_slider.valueChanged.connect(self._on_vol_change)
        vol_inner.addWidget(self.vol_label)
        vol_inner.addWidget(self.vol_slider)
        ctrl_layout.addWidget(vol_group)

        alert_group = QGroupBox("Alert Status")
        alert_inner = QVBoxLayout(alert_group)
        self.alert_label = QLabel("MONITORING")
        self.alert_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.alert_label.setStyleSheet("color: #2E7D32;")
        alert_inner.addWidget(self.alert_label)
        self.fall_count_label = QLabel("Falls detected: 0")
        alert_inner.addWidget(self.fall_count_label)
        ctrl_layout.addWidget(alert_group)

        stats_group = QGroupBox("Current Detection")
        stats_inner = QVBoxLayout(stats_group)
        self.class_label = QLabel("Class: --")
        self.conf_val_label = QLabel("Confidence: --")
        stats_inner.addWidget(self.class_label)
        stats_inner.addWidget(self.conf_val_label)
        ctrl_layout.addWidget(stats_group)

        ctrl_layout.addStretch()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready")

        self._flash_timer = QTimer()
        self._flash_timer.timeout.connect(self._flash_tick)
        self._flash_state = False

    def _on_mode_change(self, idx):
        mode = "two-step" if idx == 1 else "one-step"
        self._inference.mode = mode
        self.status_bar.showMessage(f"Mode: {mode}")

    def _on_conf_change(self, val):
        conf = val / 100.0
        self._inference.conf = conf
        self.conf_label.setText(f"{conf:.2f}")

    def _on_vol_change(self, val):
        self._volume = val
        self._player.setVolume(val)
        self.vol_label.setText(f"{val}%")

    def _on_frame(self, frame, detections, is_fall, fall_conf):
        now = time.time()

        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                measured_fps = (len(self._frame_times) - 1) / elapsed
                self._recorder.update_fps(measured_fps)

        annotated = frame.copy()
        for d in detections:
            cls = d["class"]
            conf = d["conf"]
            x1, y1, x2, y2 = map(int, d["xyxy"])
            color = BOX_COLORS.get(cls, (200, 200, 200))
            thick = 3 if cls == "Fall" else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
            lbl = f"{cls} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(annotated, lbl, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            method = d.get("method")
            if method:
                cv2.putText(annotated, method, (x1 + 3, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        if is_fall:
            badge = "  FALL DETECTED  "
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            bx = (annotated.shape[1] - bw) // 2
            cv2.rectangle(annotated, (bx - 6, 8), (bx + bw + 6, bh + 30), (0, 40, 200), -1)
            cv2.putText(annotated, badge, (bx, bh + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if (now - self._last_fall_t) > ALERT_HOLD_S:
                self._fall_count += 1
                self._last_fall_t = now
                self.fall_count_label.setText(f"Falls detected: {self._fall_count}")
                if Path(ALERT_SOUND).exists() and self._volume > 0:
                    self._player.stop()
                    self._player.play()
                self._alert_active = True
                self._flash_timer.start(300)
                self._recorder.trigger(self._inference.mode, fall_conf)

        self._recorder.feed(annotated)

        if self._alert_active and (now - self._last_fall_t) > ALERT_HOLD_S:
            self._alert_active = False
            self._flash_timer.stop()
            self.alert_label.setText("MONITORING")
            self.alert_label.setStyleSheet("color: #2E7D32;")

        if detections:
            top = max(detections, key=lambda d: d["conf"])
            self.class_label.setText(f"Class: {top['class']}")
            self.conf_val_label.setText(f"Confidence: {top['conf']:.1%}")
        else:
            self.class_label.setText("Class: --")
            self.conf_val_label.setText("Confidence: --")

        h, w, c = annotated.shape
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        scaled = qt_img.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _flash_tick(self):
        self._flash_state = not self._flash_state
        if self._flash_state:
            self.alert_label.setText("!! FALL DETECTED !!")
            self.alert_label.setStyleSheet("color: #C0392B; font-weight: bold;")
        else:
            self.alert_label.setText("!! FALL DETECTED !!")
            self.alert_label.setStyleSheet("color: #E74C3C;")

    def closeEvent(self, event):
        self._inference.stop()
        self._recorder.finalize()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FallDetectionGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

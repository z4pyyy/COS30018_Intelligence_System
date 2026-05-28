"""
=============================================================================
  COS30018 — Automated Fall Detection
  Enhanced Live Visualizer v3
  - Temporal smoothing (N-frame persistence before alert fires)
  - Tiered confidence alerts (Possible Fall vs Confirmed Fall)
  - Optional two-model pipeline (person detector + fall classifier)
  - Transition state detection (early fall prediction proxy)
  - AUTO CLIP SAVING: 5s before + after each confirmed/possible fall event
      Confirmed → results_video/visualize_test/fall/
      Possible  → results_video/visualize_test/pos_fall/
=============================================================================

SINGLE MODEL MODE:
  python src/visualize_fall.py
  python src/visualize_fall.py evaluation_videos/fall_video.mp4
  python src/visualize_fall.py --camera

TWO MODEL MODE:
  python src/visualize_fall.py --two-step
  python src/visualize_fall.py --two-step --camera

CONTROLS:
  SPACE      pause / resume
  S          step one frame (while paused)
  +  /  -    speed up / slow down
  T          toggle temporal smoothing on/off
  M          toggle single/two-step mode (if both models loaded)
  R          restart video
  Q / ESC    quit
=============================================================================
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import timedelta
from collections import deque
import time as _time

warnings.filterwarnings("ignore")

MISSING = []
for pkg, imp in [
    ("opencv-python", "cv2"), ("numpy", "numpy"),
    ("ultralytics", "ultralytics"),
]:
    try:
        __import__(imp)
    except ImportError:
        MISSING.append(pkg)

if MISSING:
    print(f"\n[ERROR] Run:  pip install {' '.join(MISSING)}\n")
    sys.exit(1)

import cv2
import numpy as np
import torch
import pickle
import mediapipe as mp_lib
from ultralytics import YOLO

from pipeline_utils import (
    pad_bbox, extract_pose_features, GeometricFallRule,
    STAGE1_CONF, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH, STAGE2_SCALER_PATH,
    ONESTEP_MODEL_PATH, CONFIRMED_THRESH as _CONFIRMED_THRESH,
    PROJECT_ROOT,
)
from transition_detector import TransitionDetector

# =============================================================================
#  CONFIGURATION
# =============================================================================

FALL_MODEL_PATH   = str(ONESTEP_MODEL_PATH)
PERSON_MODEL_PATH = str(STAGE1_MODEL_PATH)
VIDEOS_DIR        = "src/videos/videos"

# ── Output clip directories ───────────────────────────────────────────────────
CLIP_ROOT     = Path("results_video/visualize_test")
CONFIRMED_DIR = CLIP_ROOT / "fall"
POSSIBLE_DIR  = CLIP_ROOT / "pos_fall"

# ── Clip settings ─────────────────────────────────────────────────────────────
CLIP_PRE_SEC      = 5.0    # seconds of footage BEFORE trigger frame
CLIP_POST_SEC     = 5.0    # seconds of footage AFTER trigger frame
CLIP_CODEC        = "mp4v"
CLIP_COOLDOWN_SEC = 8.0    # min gap between clips of the same type

# ── Inference ─────────────────────────────────────────────────────────────────
CONF_THRESHOLD    = 0.50
IMG_SIZE          = 640
DEVICE            = 0

# ── Tiered alert thresholds ───────────────────────────────────────────────────
CONFIRMED_THRESH  = 0.70
POSSIBLE_THRESH   = 0.35

# ── Temporal smoothing ────────────────────────────────────────────────────────
FALL_PERSISTENCE_FRAMES = 5
POSSIBLE_PERSISTENCE    = 3
ALERT_PERSIST_SEC       = 3.0

# ── Two-step ──────────────────────────────────────────────────────────────────
PERSON_CONF        = 0.45
PERSON_CLASS_NAME  = "person"
NON_PERSON_CLASSES = {"hand", "wildlife", "flower"}

# ── Display ───────────────────────────────────────────────────────────────────
CHART_HISTORY  = 200
CHART_W        = 420
PANEL_H        = 720
DEFAULT_DELAY  = 1
PAUSED_DELAY   = 50

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

COL_BG    = (18,  18,  18)
COL_LINE  = (50,  50,  50)
COL_TEXT  = (220, 220, 220)
COL_DIM   = (110, 110, 110)
COL_GREEN = (60,  200,  80)
COL_RED   = (40,   50, 210)
COL_AMBER = (0,   165, 255)
COL_BLUE  = (210, 120,  40)
COL_WHITE = (255, 255, 255)

CLASS_BOX_COLORS = {
    "Fall": (0,   40,  200),
    "Sit":  (60,  200,  80),
    "Walk": (255, 255, 255),
}

# =============================================================================
#  HELPERS
# =============================================================================

def fmt_time(seconds: float) -> str:
    td    = timedelta(seconds=int(seconds))
    total = int(td.total_seconds())
    h, r  = divmod(total, 3600)
    m, s  = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# =============================================================================
#  FRAME BUFFER
# =============================================================================

class FrameBuffer:
    """Rolling buffer of annotated frames + metadata for pre-event footage."""

    def __init__(self, max_frames: int):
        self.buf = deque(maxlen=max_frames)

    def push(self, annotated_frame: np.ndarray, meta: dict):
        self.buf.append((annotated_frame.copy(), meta))

    def get_all(self):
        return list(self.buf)

    def clear(self):
        self.buf.clear()


# =============================================================================
#  CLIP WRITER
# =============================================================================

class ClipWriter:
    """
    Writes fall event clips: pre-event buffer + trigger frame + post-event frames.
    Each clip gets a .txt sidecar log with per-frame metadata.
    The trigger frame is visually marked in the clip with a coloured bar.
    """

    def __init__(self, fps: float, frame_w: int, frame_h: int):
        self.fps       = fps
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        self.writer    = None
        self.active    = False
        self.post_frames_remaining = 0
        self.clip_path = None
        self.log_path  = None
        self.trigger_frame_idx = None

    def start(self, pre_buffer: list, trigger_meta: dict, level: str,
              out_dir: Path, clip_name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.clip_path = out_dir / f"{clip_name}.mp4"
        self.log_path  = out_dir / f"{clip_name}.txt"

        codec       = cv2.VideoWriter_fourcc(*CLIP_CODEC)
        self.writer = cv2.VideoWriter(
            str(self.clip_path), codec, self.fps, (self.frame_w, self.frame_h)
        )
        self.active                = True
        self.post_frames_remaining = int(CLIP_POST_SEC * self.fps)
        self.trigger_frame_idx     = trigger_meta["frame_idx"]

        # Write log header
        log_lines = [
            "=" * 60,
            "COS30018 Fall Detection — Event Clip Log",
            f"Level     : {level.upper()}",
            f"Clip      : {self.clip_path.name}",
            f"Trigger   : frame {self.trigger_frame_idx}  @ {fmt_time(trigger_meta['timestamp_s'])}",
            f"Confidence: {trigger_meta['fall_conf']:.1%}",
            f"Consecutive: {trigger_meta['consecutive']} frames",
            f"Pre-buffer: {len(pre_buffer)} frames ({len(pre_buffer)/self.fps:.1f}s)",
            f"Post      : {CLIP_POST_SEC}s",
            "=" * 60,
            f"{'Frame':>8}  {'Time':>8}  {'Section':>10}  {'Alert':>10}  {'Conf':>6}  Consec",
            "-" * 60,
        ]
        self.log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        print(f"\n  [CLIP] {level.upper()} → {self.clip_path.name}")
        print(f"         Trigger: frame {self.trigger_frame_idx}  "
              f"conf {trigger_meta['fall_conf']:.1%}  "
              f"consecutive {trigger_meta['consecutive']}")

        # Write pre-event buffer
        for pre_frame, pre_meta in pre_buffer:
            labeled = self._burn_overlay(pre_frame, pre_meta, "PRE", False)
            self.writer.write(labeled)
            self._append_log(pre_meta, "PRE")

        # Write trigger frame with special marker
        labeled = self._burn_overlay(trigger_meta["frame"], trigger_meta, "TRIGGER", True)
        self.writer.write(labeled)
        self._append_log(trigger_meta, "TRIGGER ***")

    def write_post(self, annotated_frame: np.ndarray, meta: dict) -> bool:
        if not self.active:
            return False
        labeled = self._burn_overlay(annotated_frame, meta, "POST", False)
        self.writer.write(labeled)
        self._append_log(meta, "POST")
        self.post_frames_remaining -= 1
        if self.post_frames_remaining <= 0:
            self.finish()
            return False
        return True

    def finish(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.active = False
        if self.clip_path and self.clip_path.exists():
            size_mb = self.clip_path.stat().st_size / (1024 ** 2)
            print(f"  [CLIP] Finalised → {self.clip_path.name}  ({size_mb:.1f} MB)")

    def _burn_overlay(self, frame: np.ndarray, meta: dict,
                      section: str, is_trigger: bool) -> np.ndarray:
        """Burn per-frame metadata bar onto the frame for the saved clip."""
        out  = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Bottom info bar
        cv2.rectangle(out, (0, h - 50), (w, h), (10, 10, 10), -1)
        cv2.putText(out,
                    f"Frame {meta['frame_idx']}  |  "
                    f"{fmt_time(meta['timestamp_s'])}  |  {section}",
                    (8, h - 32), font, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

        conf  = meta.get("fall_conf", 0.0)
        consec = meta.get("consecutive", 0)
        alert  = meta.get("alert_level", "none")
        c_col  = (40,50,210) if conf >= CONFIRMED_THRESH else (
                  (0,165,255) if conf >= POSSIBLE_THRESH else (160,160,160))

        cv2.putText(out,
                    f"Alert: {alert.upper()}  |  "
                    f"Conf: {conf:.1%}  |  "
                    f"Consecutive: {consec} / {FALL_PERSISTENCE_FRAMES} frames",
                    (8, h - 10), font, 0.46, c_col, 1, cv2.LINE_AA)

        # Trigger frame indicator — coloured bars top + bottom
        if is_trigger:
            cv2.rectangle(out, (0, 0), (w, 6), (40, 50, 210), -1)
            cv2.rectangle(out, (0, h - 52), (w, h - 50), (40, 50, 210), -1)
            label = "TRIGGER FRAME"
            (lw, lh), _ = cv2.getTextSize(label, font, 0.55, 2)
            cv2.putText(out, label, (w - lw - 10, 26),
                        font, 0.55, (40, 50, 210), 2, cv2.LINE_AA)

        return out

    def _append_log(self, meta: dict, section: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{meta['frame_idx']:>8}  "
                f"{fmt_time(meta['timestamp_s']):>8}  "
                f"{section:>10}  "
                f"{meta.get('alert_level','none'):>10}  "
                f"{meta.get('fall_conf',0.0):>5.1%}  "
                f"{meta.get('consecutive',0)}\n"
            )


# =============================================================================
#  TEMPORAL SMOOTHER
# =============================================================================

class TemporalSmoother:
    def __init__(self, persistence: int = FALL_PERSISTENCE_FRAMES):
        self.persistence      = persistence
        self.consecutive_fall = 0
        self.conf_history     = deque(maxlen=30)
        self.last_alert_t     = -999.0
        self.peak_conf        = 0.0

    def update(self, is_fall: bool, confidence: float) -> tuple:
        if is_fall:
            self.consecutive_fall += 1
            self.conf_history.append(confidence)
            self.peak_conf = max(self.peak_conf, confidence)
        else:
            self.consecutive_fall = 0
            self.peak_conf = 0.0

        if is_fall and self.consecutive_fall >= self.persistence:
            if confidence >= CONFIRMED_THRESH:
                level = "confirmed"
                self.last_alert_t = _time.time()
            elif confidence >= POSSIBLE_THRESH:
                level = "possible"
                self.last_alert_t = _time.time()
            else:
                level = "none"
        elif is_fall and self.consecutive_fall >= POSSIBLE_PERSISTENCE:
            level = "possible" if confidence >= POSSIBLE_THRESH else "none"
            if level == "possible":
                self.last_alert_t = _time.time()
        else:
            level = "none"

        if level == "none" and (_time.time() - self.last_alert_t) < ALERT_PERSIST_SEC:
            level = "possible"

        return level, confidence if is_fall else 0.0, self.consecutive_fall

    def get_transition_state(self) -> str:
        if len(self.conf_history) < 3:
            return "stable"
        recent  = list(self.conf_history)[-5:]
        if len(recent) < 3:
            return "stable"
        rising  = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        falling = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
        avg     = sum(recent) / len(recent)
        if rising  and avg > 0.30: return "rising"
        if falling and avg > 0.30: return "recovering"
        return "stable"


# =============================================================================
#  TWO-STEP PIPELINE
# =============================================================================

class TwoStepDetector:
    """Stage 1 YOLO person detector -> MediaPipe pose -> MLP classifier."""

    def __init__(self, person_model: YOLO, mlp, scaler):
        self.person_model = person_model
        self.mlp          = mlp
        self.scaler       = scaler
        self.geo_rule     = GeometricFallRule(aspect_threshold=1.5, visibility_threshold=0.5)
        self.pose_model   = mp_lib.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def predict(self, frame: np.ndarray, conf_thresh: float) -> list:
        results = []
        stage1  = self.person_model.predict(
            source=frame, conf=STAGE1_CONF,
            imgsz=IMG_SIZE, device=DEVICE, verbose=False
        )[0]

        person_boxes = []
        if stage1.boxes is not None:
            for box in stage1.boxes:
                cls_name = self.person_model.names[int(box.cls[0])].lower()
                if cls_name == PERSON_CLASS_NAME:
                    person_boxes.append({
                        "conf": float(box.conf[0]),
                        "xyxy": box.xyxy[0].cpu().numpy().tolist()
                    })

        if not person_boxes:
            return []

        h, w = frame.shape[:2]

        for pb in person_boxes:
            x1, y1, x2, y2 = [int(v) for v in pb["xyxy"]]
            x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, w, h)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop   = frame[y1:y2, x1:x2]
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self.pose_model.process(rgb)

            features, core_vis = extract_pose_features(
                result.pose_landmarks if result.pose_landmarks else None
            )

            geo_fired = False
            if core_vis is not None:
                geo_fired = self.geo_rule.check(
                    bbox=(x1, y1, x2, y2),
                    keypoint_visibilities=core_vis,
                )

            if geo_fired:
                results.append({
                    "class_name": "Fall", "conf": 0.75,
                    "xyxy": pb["xyxy"], "geo_rule": True,
                })
                continue

            if features is None:
                results.append({
                    "class_name": "Walk", "conf": pb["conf"],
                    "xyxy": pb["xyxy"], "geo_rule": False,
                })
                continue

            feat_norm = self.scaler.transform([features])
            with torch.no_grad():
                probs    = torch.softmax(self.mlp(
                    torch.tensor(feat_norm, dtype=torch.float32)), dim=1)[0]
                pred_id  = int(probs.argmax())
                conf_val = float(probs[pred_id])

            cls_name = ["Fall", "Sit", "Walk"][pred_id]
            results.append({
                "class_name": cls_name, "conf": conf_val,
                "xyxy": pb["xyxy"], "geo_rule": False,
            })

        return results

    def close(self):
        self.pose_model.close()


# =============================================================================
#  DETECTION LOGIC
# =============================================================================

def check_fall(detections: list) -> tuple:
    if not detections:
        return False, "none", 0.0, set()
    best_conf=0.0; best_cls="none"; fall_conf=0.0; is_fall=False; all_cls=set()
    for d in detections:
        cls=d["class_name"]; conf=d["conf"]; all_cls.add(cls)
        if conf > best_conf: best_conf=conf; best_cls=cls
        if cls == "Fall":
            is_fall=True
            if conf > fall_conf: fall_conf=conf
    return is_fall, best_cls, round(fall_conf if is_fall else best_conf,4), all_cls


# =============================================================================
#  FRAME ANNOTATION
# =============================================================================

def annotate_frame(frame, detections, alert_level, consecutive,
                   transition_state, two_step_mode):
    annotated = frame.copy()
    for d in detections:
        cls_name=d["class_name"]; conf=d["conf"]
        x1,y1,x2,y2=map(int,d["xyxy"])
        color=CLASS_BOX_COLORS.get(cls_name,COL_WHITE)
        thickness=3 if cls_name=="Fall" else 2
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,thickness)
        label=f"{cls_name} {conf:.0%}"
        (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.rectangle(annotated,(x1,y1-lh-8),(x1+lw+6,y1),color,-1)
        cv2.putText(annotated,label,(x1+3,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
        if cls_name=="Fall":
            tick=20
            for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(annotated,(cx,cy),(cx+dx*tick,cy),color,3)
                cv2.line(annotated,(cx,cy),(cx,cy+dy*tick),color,3)

    h,w=annotated.shape[:2]
    if alert_level=="confirmed":
        alpha=0.20+0.15*abs(np.sin(_time.time()*2*np.pi))
        overlay=annotated.copy()
        thick=max(int(min(h,w)*0.025),8)
        cv2.rectangle(overlay,(0,0),(w,h),CLASS_BOX_COLORS["Fall"],thick)
        cv2.addWeighted(overlay,alpha,annotated,1-alpha,0,annotated)
        badge="  FALL DETECTED  "
        (bw,bh),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_DUPLEX,0.80,2)
        bx=(w-bw)//2
        cv2.rectangle(annotated,(bx-6,8),(bx+bw+6,bh+28),CLASS_BOX_COLORS["Fall"],-1)
        cv2.putText(annotated,badge,(bx,bh+18),
                    cv2.FONT_HERSHEY_DUPLEX,0.80,COL_WHITE,2,cv2.LINE_AA)
    elif alert_level=="possible":
        thick=4; size=40
        for cx,cy,dx,dy in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
            cv2.line(annotated,(cx,cy),(cx+dx*size,cy),COL_AMBER,thick)
            cv2.line(annotated,(cx,cy),(cx,cy+dy*size),COL_AMBER,thick)
        badge="  POSSIBLE FALL  "
        (bw,bh),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_DUPLEX,0.65,1)
        bx=(w-bw)//2
        cv2.rectangle(annotated,(bx-4,8),(bx+bw+4,bh+24),COL_AMBER,-1)
        cv2.putText(annotated,badge,(bx,bh+16),
                    cv2.FONT_HERSHEY_DUPLEX,0.65,(0,0,0),1,cv2.LINE_AA)

    if transition_state=="rising" and alert_level=="none":
        ts_badge="  ^ ONSET DETECTED  "
        (tw,th),_=cv2.getTextSize(ts_badge,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
        cv2.rectangle(annotated,(8,h-th-20),(tw+16,h-8),(40,80,160),-1)
        cv2.putText(annotated,ts_badge,(12,h-12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.50,COL_WHITE,1,cv2.LINE_AA)

    mode_str="TWO-STEP" if two_step_mode else "ONE-STEP"
    cv2.putText(annotated,f"COS30018 Fall Detection  |  {mode_str}  |  YOLOv8",
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1,cv2.LINE_AA)
    return annotated


# =============================================================================
#  RIGHT PANEL
# =============================================================================

def build_right_panel(conf_history, fall_history, event_log,
                      alert_level, dominant_cls, fall_conf,
                      consecutive, transition_state,
                      timestamp_s, duration_s, event_count,
                      fps_actual, paused, speed,
                      temporal_enabled, two_step_mode, clips_saved):

    panel=np.full((PANEL_H,CHART_W,3),COL_BG,dtype=np.uint8); W=CHART_W

    def txt(text,x,y,scale=0.46,color=COL_TEXT,thickness=1,bold=False):
        font=cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel,text,(x,y),font,scale,color,thickness,cv2.LINE_AA)

    def hline(y,color=COL_LINE,thick=1):
        cv2.line(panel,(0,y),(W,y),color,thick)

    cv2.rectangle(panel,(0,0),(W,42),(30,30,30),-1)
    txt("COS30018 Fall Detection",10,18,0.50,COL_WHITE,bold=True)
    mode_col=(120,200,255) if two_step_mode else COL_DIM
    txt("TWO-STEP" if two_step_mode else "ONE-STEP",10,36,0.36,mode_col)
    status_col=COL_RED if alert_level=="confirmed" else (COL_AMBER if alert_level=="possible" else COL_GREEN)
    status_str=("● FALL" if alert_level=="confirmed" else "● POSSIBLE" if alert_level=="possible" else "● CLEAR")
    (sw,_),_=cv2.getTextSize(status_str,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
    txt(status_str,W-sw-10,26,0.50,status_col)
    hline(42,(60,60,60))

    y=64
    txt(f"{fmt_time(timestamp_s)}  /  {fmt_time(duration_s)}",10,y,0.44)
    if duration_s>0:
        pw=W-20; fv=int(pw*min(timestamp_s/duration_s,1.0))
        cv2.rectangle(panel,(10,y+8),(10+pw,y+14),(50,50,50),-1)
        cv2.rectangle(panel,(10,y+8),(10+fv,y+14),(130,130,130),-1)
    clip_col=COL_GREEN if clips_saved>0 else COL_DIM
    txt(f"Clips: {clips_saved}",W-70,y,0.36,clip_col)

    y=100; hline(y-8)
    txt("FALL CONFIDENCE",10,y,0.40,COL_DIM)
    mx,my,mw,mh=10,y+6,W-20,20
    cv2.rectangle(panel,(mx,my),(mx+mw,my+mh),(35,35,35),-1)
    p_end=mx+int(mw*CONFIRMED_THRESH)
    cv2.rectangle(panel,(mx,my),(p_end,my+mh),(20,50,60),-1)
    cv2.rectangle(panel,(p_end,my),(mx+mw,my+mh),(20,20,60),-1)
    fill=int(mw*min(fall_conf,1.0))
    if fall_conf>0:
        fill_col=COL_RED if fall_conf>=CONFIRMED_THRESH else COL_AMBER
        cv2.rectangle(panel,(mx,my),(mx+fill,my+mh),fill_col,-1)
    for thresh,label in [(POSSIBLE_THRESH,""),(CONFIRMED_THRESH,"70%")]:
        tx=mx+int(mw*thresh)
        cv2.line(panel,(tx,my-2),(tx,my+mh+2),COL_AMBER,1)
        if label: txt(label,tx-14,my-4,0.30,COL_AMBER)
    conf_str=f"{fall_conf:.0%}" if fall_conf>0 else "—"
    txt(conf_str,mx+mw-28,my+15,0.46,COL_WHITE,bold=True)

    y=148
    persist_pct=min(consecutive/FALL_PERSISTENCE_FRAMES,1.0)
    persist_col=COL_RED if persist_pct>=1.0 else (COL_AMBER if persist_pct>0 else COL_DIM)
    txt(f"Persistence: {consecutive}/{FALL_PERSISTENCE_FRAMES} frames",10,y,0.40,persist_col)
    txt(f"Temporal: {'ON' if temporal_enabled else 'OFF'}",W-80,y,0.36,
        COL_GREEN if temporal_enabled else COL_DIM)

    y=168
    ts_colors={"rising":COL_AMBER,"recovering":COL_GREEN,"stable":COL_DIM}
    ts_labels={"rising":"^ ONSET TREND","recovering":"v RECOVERING","stable":"— STABLE"}
    txt(f"Motion: {ts_labels.get(transition_state,'—')}",10,y,0.38,ts_colors.get(transition_state,COL_DIM))

    y=192; hline(y-6)
    card_w=(W-20)//4
    bar_col=COL_RED if fall_conf>=CONFIRMED_THRESH else (COL_AMBER if fall_conf>=POSSIBLE_THRESH else COL_DIM)
    cards=[("FALLS",str(event_count),COL_RED),("CLASS",dominant_cls[:5].upper(),bar_col),
           ("FPS",f"{fps_actual:.1f}",COL_DIM),("SPD",f"{speed:.1f}x",COL_DIM)]
    for i,(label,value,col) in enumerate(cards):
        cx=10+i*card_w
        cv2.rectangle(panel,(cx,y),(cx+card_w-4,y+44),(28,28,28),-1)
        cv2.rectangle(panel,(cx,y),(cx+card_w-4,y+44),(55,55,55),1)
        (vw,_),_=cv2.getTextSize(value,cv2.FONT_HERSHEY_DUPLEX,0.52,1)
        txt(value,cx+(card_w-vw)//2,y+24,0.52,col,bold=True)
        (lw,_),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.30,1)
        txt(label,cx+(card_w-lw)//2,y+40,0.30,COL_DIM)

    y=254; hline(y); y+=16
    if alert_level=="confirmed":
        badge="  FALL DETECTED  "
        (bw,bh),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_DUPLEX,0.55,1)
        bx=(W-bw)//2
        cv2.rectangle(panel,(bx-4,y-bh-4),(bx+bw+4,y+4),COL_RED,-1)
        txt(badge,bx,y,0.55,COL_WHITE,bold=True)
    elif alert_level=="possible":
        badge="  POSSIBLE FALL  "
        (bw,bh),_=cv2.getTextSize(badge,cv2.FONT_HERSHEY_DUPLEX,0.55,1)
        bx=(W-bw)//2
        cv2.rectangle(panel,(bx-4,y-bh-4),(bx+bw+4,y+4),COL_AMBER,-1)
        txt(badge,bx,y,0.55,(0,0,0),bold=True)
    else:
        txt(f"Monitoring — {dominant_cls}",10,y,0.44,COL_DIM)

    chart_top=290; chart_bottom=490; chart_h=chart_bottom-chart_top
    hline(chart_top-4)
    txt("FALL CONFIDENCE HISTORY",10,chart_top-8,0.36,COL_DIM)
    cv2.rectangle(panel,(8,chart_top),(W-8,chart_bottom),(26,26,26),-1)
    p_y=chart_bottom-int(chart_h*CONFIRMED_THRESH)
    cv2.rectangle(panel,(8,p_y),(W-8,chart_bottom),(15,30,40),-1)
    cv2.rectangle(panel,(8,chart_top),(W-8,p_y),(20,15,40),-1)
    cv2.line(panel,(8,p_y),(W-8,p_y),COL_RED,1); txt("70%",W-34,p_y-3,0.30,COL_RED)
    pos_y=chart_bottom-int(chart_h*POSSIBLE_THRESH)
    cv2.line(panel,(8,pos_y),(W-8,pos_y),COL_AMBER,1); txt("35%",W-34,pos_y-3,0.30,COL_AMBER)

    hist_list=list(conf_history); fall_list=list(fall_history); n=len(hist_list)
    if n>1:
        pw=W-16
        for i in range(1,n):
            x1p=8+int(pw*(i-1)/(CHART_HISTORY-1)); x2p=8+int(pw*i/(CHART_HISTORY-1))
            y1p=max(chart_top,min(chart_bottom,chart_bottom-int(chart_h*hist_list[i-1])))
            y2p=max(chart_top,min(chart_bottom,chart_bottom-int(chart_h*hist_list[i])))
            c=hist_list[i]
            lc=COL_RED if c>=CONFIRMED_THRESH else (COL_AMBER if c>=POSSIBLE_THRESH else COL_BLUE)
            cv2.line(panel,(x1p,y1p),(x2p,y2p),lc,2,cv2.LINE_AA)
        for i,(c,f) in enumerate(zip(hist_list,fall_list)):
            if f and c>=CONFIRMED_THRESH:
                xp=8+int(pw*i/(CHART_HISTORY-1))
                yp=max(chart_top,min(chart_bottom,chart_bottom-int(chart_h*c)))
                cv2.circle(panel,(xp,yp),3,COL_RED,-1)

    log_top=chart_bottom+10; hline(log_top-4)
    txt("FALL EVENT LOG",10,log_top+4,0.36,COL_DIM); log_top+=18
    for i,ev in enumerate(reversed(event_log[-8:])):
        ey=log_top+i*22
        if ey>PANEL_H-38: break
        col=COL_RED if ev["level"]=="confirmed" else COL_AMBER
        alpha=max(0.4,1.0-i*0.08)
        col_a=tuple(int(c*alpha) for c in col)
        cv2.circle(panel,(14,ey-4),4,col_a,-1)
        lvl_str="FALL" if ev["level"]=="confirmed" else "POSS"
        saved_str=" [SAVED]" if ev.get("clip_saved") else ""
        ev_str=f"{ev['time']}  [{lvl_str}]  {ev['conf']:.0%}{saved_str}"
        txt(ev_str,26,ey,0.36,tuple(int(c*alpha) for c in COL_TEXT))

    if not event_log:
        txt("No events detected",10,log_top,0.38,COL_DIM)

    footer_y=PANEL_H-42; hline(footer_y)
    cv2.rectangle(panel,(0,footer_y),(W,PANEL_H),(22,22,22),-1)
    txt("SPACE pause  S step  T temporal  M mode",6,footer_y+16,0.32,COL_DIM)
    txt("+/- speed  R restart  Q quit",6,footer_y+32,0.32,COL_DIM)
    if paused: txt("PAUSED",W-54,footer_y+24,0.44,COL_AMBER,bold=True)

    return panel


# =============================================================================
#  MAIN VISUALIZER
# =============================================================================

def run_visualizer(source, use_camera: bool = False, two_step: bool = False):

    CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
    POSSIBLE_DIR.mkdir(parents=True, exist_ok=True)

    fall_model_path = Path(FALL_MODEL_PATH)
    if not fall_model_path.exists():
        print(f"[ERROR] Fall model not found: {FALL_MODEL_PATH}"); sys.exit(1)

    print(f"Loading fall model: {FALL_MODEL_PATH}")
    fall_model = YOLO(str(fall_model_path))
    print(f"  Classes: {fall_model.names}")

    person_model=None; mlp=None; scaler=None; two_step_avail=False
    if two_step or Path(PERSON_MODEL_PATH).exists():
        try:
            print(f"Loading person model: {PERSON_MODEL_PATH}")
            person_model=YOLO(PERSON_MODEL_PATH); two_step_avail=True
            print(f"  Classes: {person_model.names}")

            if STAGE2_MODEL_PATH.exists() and STAGE2_SCALER_PATH.exists():
                sys.path.insert(0, str(PROJECT_ROOT / "V2" / "src"))
                from train_stage2_pose import PoseMLP
                mlp = PoseMLP()
                mlp.load_state_dict(torch.load(str(STAGE2_MODEL_PATH), map_location="cpu", weights_only=True))
                mlp.eval()
                with open(str(STAGE2_SCALER_PATH), "rb") as f:
                    scaler = pickle.load(f)
                print(f"  [OK] Stage 2 MLP: {STAGE2_MODEL_PATH.name}")
            else:
                print(f"  [WARN] Stage 2 model/scaler not found — two-step unavailable")
                two_step_avail = False
        except Exception as e:
            print(f"  [WARN] {e}")
            two_step_avail = False

    two_step_pipeline = TwoStepDetector(person_model, mlp, scaler) if two_step_avail else None
    two_step_mode     = two_step and two_step_avail

    if use_camera:
        cap=cv2.VideoCapture(0); source="Live Camera"
    else:
        cap=cv2.VideoCapture(str(source))
        print(f"Opening: {Path(str(source)).name}")

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_camera else 0
    duration_s   = total_frames / fps if total_frames > 0 else 0
    raw_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_h=PANEL_H; vid_w=int(raw_w*PANEL_H/raw_h); total_w=vid_w+CHART_W

    print(f"  {raw_w}x{raw_h}  |  {fps:.1f} fps")
    print(f"  Clip pre/post: {CLIP_PRE_SEC}s / {CLIP_POST_SEC}s  "
          f"|  cooldown: {CLIP_COOLDOWN_SEC}s")
    print(f"  Confirmed → {CONFIRMED_DIR}")
    print(f"  Possible  → {POSSIBLE_DIR}")
    print(f"  Mode: {'TWO-STEP' if two_step_mode else 'ONE-STEP'}")
    print("\nControls: SPACE=pause  T=temporal  M=mode  +/-=speed  Q=quit\n")

    win_name="COS30018 — Fall Detection"
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name,total_w,PANEL_H)

    pre_buf_max = int(CLIP_PRE_SEC * fps)
    pre_buffer  = FrameBuffer(pre_buf_max)

    confirmed_writer = ClipWriter(fps, raw_w, raw_h)
    possible_writer  = ClipWriter(fps, raw_w, raw_h)

    def _new_smoother():
        return TransitionDetector(
            window_size=15, min_fall_frames=3, walk_stability_frames=5,
            alert_hold_sec=ALERT_PERSIST_SEC, confirmed_conf_thresh=CONFIRMED_THRESH,
        )
    smoother     = _new_smoother()
    conf_history = deque([0.0]*CHART_HISTORY, maxlen=CHART_HISTORY)
    fall_history = deque([False]*CHART_HISTORY, maxlen=CHART_HISTORY)
    event_log    = []

    frame_idx=0; event_count=0; last_event_ts=-999.0
    paused=False; speed=1.0; temporal_on=True

    last_detections=[]; last_alert_level="none"
    last_dominant_cls="none"; last_fall_conf=0.0
    last_consecutive=0; last_transition="stable"

    last_confirmed_clip_t=-999.0; last_possible_clip_t=-999.0
    clips_saved=0

    source_stem = Path(str(source)).stem if not use_camera else "camera"

    fps_timer=_time.time(); fps_actual=fps; fps_frame_cnt=0
    frame = np.zeros((raw_h, raw_w, 3), np.uint8)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if use_camera: continue
                cap.set(cv2.CAP_PROP_POS_FRAMES,0); frame_idx=0
                ret,frame=cap.read()
                if not ret: break

            frame_idx+=1; timestamp_s=frame_idx/fps

            try:
                if two_step_mode and two_step_pipeline:
                    detections=two_step_pipeline.predict(frame,CONF_THRESHOLD)
                else:
                    result=fall_model.predict(
                        source=frame,conf=CONF_THRESHOLD,
                        imgsz=IMG_SIZE,device=DEVICE,verbose=False
                    )[0]
                    detections=[]
                    if result.boxes is not None:
                        for box in result.boxes:
                            detections.append({
                                "class_name": fall_model.names[int(box.cls[0])],
                                "conf":       float(box.conf[0]),
                                "xyxy":       box.xyxy[0].cpu().numpy().tolist(),
                            })
            except Exception:
                detections=last_detections

            is_fall,dom_cls,fall_conf,all_cls=check_fall(detections)

            geo_fired = any(d.get("geo_rule", False) for d in detections)
            pred_cls_id = 0 if is_fall else (1 if dom_cls == "Sit" else 2)

            if temporal_on:
                alert_level,sc=smoother.update(
                    pred_class=pred_cls_id, confidence=fall_conf, geo_fall=geo_fired,
                )
                consecutive=smoother.consecutive_fall_frames
                transition=smoother.get_transition_state()
            else:
                alert_level=("confirmed" if (is_fall and fall_conf>=CONFIRMED_THRESH)
                             else "possible" if is_fall else "none")
                consecutive=1 if is_fall else 0; transition="stable"

            # Annotate frame
            annotated=annotate_frame(frame,detections,alert_level,consecutive,transition,two_step_mode)

            # Frame metadata
            meta={
                "frame":       frame,
                "frame_idx":   frame_idx,
                "timestamp_s": timestamp_s,
                "alert_level": alert_level,
                "fall_conf":   fall_conf,
                "consecutive": consecutive,
            }

            # Always push to pre-event buffer
            pre_buffer.push(annotated, meta)

            now=_time.time()

            # Continue post-event recording
            if confirmed_writer.active:
                if not confirmed_writer.write_post(annotated, meta):
                    clips_saved+=1

            if possible_writer.active:
                if not possible_writer.write_post(annotated, meta):
                    clips_saved+=1

            # Trigger confirmed clip
            if (alert_level=="confirmed"
                    and not confirmed_writer.active
                    and (now-last_confirmed_clip_t) >= CLIP_COOLDOWN_SEC):
                ts_tag=fmt_time(timestamp_s).replace(":", "-")
                clip_name=(f"{source_stem}_confirmed_{ts_tag}"
                           f"_f{frame_idx:06d}_c{int(fall_conf*100):02d}pct")
                confirmed_writer.start(
                    pre_buffer.get_all(), meta, "confirmed", CONFIRMED_DIR, clip_name
                )
                last_confirmed_clip_t=now
                if event_log: event_log[-1]["clip_saved"]=True

            # Trigger possible clip (only if no confirmed clip active)
            elif (alert_level=="possible"
                    and not possible_writer.active
                    and not confirmed_writer.active
                    and (now-last_possible_clip_t) >= CLIP_COOLDOWN_SEC):
                ts_tag=fmt_time(timestamp_s).replace(":", "-")
                clip_name=(f"{source_stem}_possible_{ts_tag}"
                           f"_f{frame_idx:06d}_c{int(fall_conf*100):02d}pct")
                possible_writer.start(
                    pre_buffer.get_all(), meta, "possible", POSSIBLE_DIR, clip_name
                )
                last_possible_clip_t=now
                if event_log: event_log[-1]["clip_saved"]=True

            # Event log
            if alert_level in ("confirmed","possible"):
                if (timestamp_s-last_event_ts)>3.0:
                    event_count+=1; last_event_ts=timestamp_s
                    event_log.append({
                        "time":       fmt_time(timestamp_s),
                        "conf":       fall_conf,
                        "level":      alert_level,
                        "event":      event_count,
                        "clip_saved": False,
                    })
                else:
                    last_event_ts=timestamp_s

            conf_history.append(fall_conf); fall_history.append(is_fall)
            last_detections=detections; last_alert_level=alert_level
            last_dominant_cls=dom_cls; last_fall_conf=fall_conf
            last_consecutive=consecutive; last_transition=transition

            fps_frame_cnt+=1
            elapsed=_time.time()-fps_timer
            if elapsed>=0.5:
                fps_actual=fps_frame_cnt/elapsed
                fps_timer=_time.time(); fps_frame_cnt=0

        else:
            timestamp_s=frame_idx/fps if frame_idx>0 else 0

        left=annotate_frame(
            frame,last_detections,last_alert_level,
            last_consecutive,last_transition,two_step_mode
        )
        left=cv2.resize(left,(vid_w,vid_h))

        right=build_right_panel(
            conf_history,fall_history,event_log,
            last_alert_level,last_dominant_cls,last_fall_conf,
            last_consecutive,last_transition,
            timestamp_s,duration_s,event_count,
            fps_actual,paused,speed,
            temporal_on,two_step_mode,clips_saved
        )

        canvas=np.hstack([left,right])
        cv2.imshow(win_name,canvas)

        wait_ms=PAUSED_DELAY if paused else max(1,int(DEFAULT_DELAY/speed))
        key=cv2.waitKey(wait_ms)&0xFF

        if key in (ord('q'),ord('Q'),27): break
        elif key==ord(' '): paused=not paused
        elif key in (ord('t'),ord('T')):
            temporal_on=not temporal_on
            smoother=_new_smoother()
            print(f"Temporal: {'ON' if temporal_on else 'OFF'}")
        elif key in (ord('m'),ord('M')):
            if two_step_avail:
                two_step_mode=not two_step_mode
                smoother=_new_smoother()
                print(f"Mode: {'TWO-STEP' if two_step_mode else 'ONE-STEP'}")
        elif key in (ord('s'),ord('S')):
            if paused:
                ret,frame=cap.read()
                if ret: frame_idx+=1
        elif key in (ord('+'),ord('=')): speed=min(speed*1.5,8.0)
        elif key in (ord('-'),ord('_')): speed=max(speed/1.5,0.25)
        elif key in (ord('r'),ord('R')):
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); frame_idx=0
            event_count=0; last_event_ts=-999.0; event_log.clear()
            smoother=_new_smoother()
            conf_history.extend([0.0]*CHART_HISTORY)
            fall_history.extend([False]*CHART_HISTORY)
            pre_buffer.clear()

        if cv2.getWindowProperty(win_name,cv2.WND_PROP_VISIBLE)<1: break

    if confirmed_writer.active: confirmed_writer.finish(); clips_saved+=1
    if possible_writer.active:  possible_writer.finish();  clips_saved+=1

    if two_step_pipeline:
        two_step_pipeline.close()
    cap.release(); cv2.destroyAllWindows()
    print(f"\nSession ended. Events: {event_count}  |  Clips saved: {clips_saved}")
    print(f"  Confirmed → {CONFIRMED_DIR}")
    print(f"  Possible  → {POSSIBLE_DIR}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    parser=argparse.ArgumentParser(description="COS30018 Fall Detection v3")
    parser.add_argument("video",nargs="?",default=None)
    parser.add_argument("--camera",action="store_true")
    parser.add_argument("--two-step",action="store_true")
    args=parser.parse_args()

    if args.camera:
        run_visualizer(None,use_camera=True,two_step=args.two_step); return
    if args.video:
        source=Path(args.video)
        if not source.exists():
            print(f"[ERROR] File not found: {args.video}"); sys.exit(1)
        run_visualizer(source,two_step=args.two_step); return

    vdir=Path(VIDEOS_DIR)
    if not vdir.exists():
        print(f"[ERROR] Folder not found: {VIDEOS_DIR}"); sys.exit(1)
    videos=sorted([f for f in vdir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS])
    if not videos:
        print(f"[ERROR] No videos in {VIDEOS_DIR}"); sys.exit(1)

    print(f"Auto-selecting: {videos[0].name}")
    run_visualizer(videos[0],two_step=args.two_step)


if __name__ == "__main__":
    main()
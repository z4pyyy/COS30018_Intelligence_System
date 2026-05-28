# COS30018 — Automated Fall Detection System

Real-time fall detection using deep learning and pose estimation.
Supports one-step (YOLOv8) and two-step (YOLOv8 + MediaPipe + MLP) pipelines
with temporal smoothing via transition detection.

## Project Structure

```
COS30018_Fall-Detection/
├── V2/                              # Current version (V2 pipeline)
│   ├── src/
│   │   ├── pipeline_utils.py        # Shared constants, adaptive bbox, feature extraction
│   │   ├── transition_detector.py   # Temporal Walk->Fall transition detector
│   │   ├── visualize_fall.py        # Interactive live video visualizer
│   │   ├── test_video.py            # Batch video testing (all 4 modes)
│   │   ├── gui_fall_detection.py    # PyQt5 webcam GUI
│   │   ├── evaluate_twostep.py      # Static image evaluation (two-step)
│   │   ├── train_stage2_pose.py     # Train MLP classifier on pose features
│   │   ├── extract_pose_features.py # Extract pose features from images
│   │   └── stage1_data_preperation.py
│   ├── models/
│   │   ├── swook_techcare_fall_sit_walk.pt  # One-step YOLOv8 (Fall/Sit/Walk)
│   │   ├── stage1_person.pt                 # Stage 1 person detector
│   │   ├── stage2_pose_mlp.pt               # Stage 2 MLP classifier
│   │   └── pose_scaler_v2.pkl               # Feature scaler for MLP
│   └── docs/                        # Design docs and evaluation results
├── src/
│   ├── pipeline_utils.py            # Shared utilities (imported by V2)
│   ├── transition_detector.py       # TransitionDetector (imported by V2)
│   ├── train_onestep.py             # Train one-step YOLOv8 model
│   ├── evaluate.py                  # One-step model evaluation
│   ├── evaluate_test.py             # Test set evaluation
│   ├── merge_dataset.py             # Dataset merging utility
│   └── models/                      # One-step model weights
├── tests/
│   ├── test_pipeline_utils.py       # Unit tests for pipeline utilities
│   └── test_transition_detector.py  # Unit tests for transition detector
├── datasets/                        # Training/test datasets (not tracked)
└── results_video/                   # Video test outputs (generated)
```

## Pipelines

### One-Step Pipeline
Single YOLOv8 model directly classifies Fall / Sit / Walk from video frames.

```
Frame -> YOLOv8 (Fall/Sit/Walk) -> Detection
```

### Two-Step Pipeline (V2)
Person detection followed by pose-based classification with geometric fall rule.

```
Frame -> YOLOv8 (Person) -> Adaptive Bbox Padding -> MediaPipe Pose
      -> Geometric Fall Rule check
      -> MLP Classifier (8 pose features -> Fall/Sit/Walk)
```

Key improvements over one-step:
- **Adaptive bbox padding**: Tall boxes get 25% padding, wide boxes 10%
- **Geometric fall rule**: Fires Fall directly when bbox is wide + low keypoint visibility
- **Pose-based MLP**: 8 geometric features (spine angle, hip height, knee bend, etc.)

### Temporal Smoothing (TransitionDetector)
Replaces simple N-consecutive-frames approach with Walk->Fall transition detection.

- Requires 5 Walk-stable frames + 3 Fall frames before alerting
- Tiered alerts: `none` -> `possible` -> `confirmed`
- Reduces false positives from momentary misclassifications

## Setup

### Requirements
```
pip install ultralytics opencv-python mediapipe torch numpy pandas matplotlib seaborn openpyxl
```

For the GUI:
```
pip install PyQt5
```

### Models
Place model files in `V2/models/`:
- `swook_techcare_fall_sit_walk.pt` — one-step YOLOv8
- `stage1_person.pt` — person detector
- `stage2_pose_mlp.pt` — pose MLP classifier
- `pose_scaler_v2.pkl` — feature scaler

## Usage

### Live Video Visualizer
```bash
# One-step mode
python V2/src/visualize_fall.py

# Two-step mode
python V2/src/visualize_fall.py --two-step

# Webcam
python V2/src/visualize_fall.py --camera

# Specific video file
python V2/src/visualize_fall.py path/to/video.mp4
```

Controls: `SPACE` pause | `T` toggle temporal | `M` toggle mode | `+/-` speed | `Q` quit

### Batch Video Testing
```bash
# One-step baseline
python V2/src/test_video.py

# Two-step with temporal smoothing
python V2/src/test_video.py --two-step --temporal

# Compare all 4 modes
python V2/src/test_video.py --compare-all
```

Outputs per mode: annotated video, fall clips, detection Excel log, timeline/confidence plots, summary dashboard.

### PyQt5 GUI (Webcam)
```bash
python V2/src/gui_fall_detection.py
```

### Static Image Evaluation (Two-Step)
```bash
python V2/src/evaluate_twostep.py
```

### Training
```bash
# One-step YOLOv8
python src/train_onestep.py --data datasets/merged_dataset/data.yaml

# Stage 2 MLP
python V2/src/train_stage2_pose.py
```

### Tests
```bash
pytest tests/ -v
```

## Class Mapping

| ID | Class |
|----|-------|
| 0  | Fall  |
| 1  | Sit   |
| 2  | Walk  |

## Alert Levels

| Level     | Condition                                    |
|-----------|----------------------------------------------|
| none      | No fall transition detected                  |
| possible  | Transition pattern detected, confidence < 70%|
| confirmed | Transition pattern + confidence >= 70%       |

## COS30018 — Intelligent Systems, Swinburne University

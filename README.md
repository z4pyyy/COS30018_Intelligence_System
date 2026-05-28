# COS30018 — Automated Fall Detection System
# Group 3 
Member 1 (Leader) - TERENCE CHIN SENG WONG	104404059
Member 2          - ASHLEY HUI XING LEE	    104404457
Member 3          - ANDIA RONG HEE LIM	    104396978
Member 4          - EDWARD NGIE JIE LING	102765149
Member 5          - FRANDYA MEIDY FABIOLA	102765149



Swinburne University of Technology — Intelligent Systems

Real-time fall detection using deep learning (YOLOv8, EfficientNet-B0 CNN, MLP) and pose estimation (MediaPipe). Classifies video frames as **Fall**, **Sit**, or **Walk** using six pipeline variants with optional temporal smoothing.

---

## Quick Start — Running the Application

### Option 1: One-Click Launch (Recommended)

Double-click **`final_version/run.bat`**

This will:
1. Check Python is installed
2. Auto-install dependencies from `final_version/requirements.txt` on first run
3. Launch the PyQt5 GUI with webcam fall detection

### Option 2: Manual Launch

```bash
pip install -r final_version/requirements.txt
python final_version/GUI_fall_detection.py
```

### Requirements

- Python 3.10+
- Webcam (for live detection)
- Dependencies: PyTorch, OpenCV, Ultralytics (YOLOv8), MediaPipe, PyQt5, scikit-learn

---

## Project Structure

```
COS30018_Intelligence_System/
│
├── final_version/                  ★ FINAL DELIVERABLE — start here
│   ├── run.bat                     ★ One-click launcher
│   ├── GUI_fall_detection.py       ★ Main GUI application (6 pipelines)
│   ├── requirements.txt               Python dependencies
│   ├── models/                        All trained model weights
│   │   ├── stage2_cnn.pt              EfficientNet-B0 CNN (V3 Stage 2)
│   │   ├── stage1_person.pt           YOLOv8 person detector (Stage 1)
│   │   ├── swook_techcare_fall_sit_walk.pt  One-step YOLOv8
│   │   ├── stage2_pose_mlp_v2.pt      MLP classifier (V2 Stage 2)
│   │   ├── pose_scaler_v2.pkl         Feature scaler for MLP
│   │   └── pose_landmarker_heavy.task  MediaPipe pose model
│   ├── evaluation/                 ★ MODEL EVALUATION RESULTS
│   │   ├── ManualTestImage_confusionMatrix/
│   │   │   ├── confusion_*.png        Confusion matrices (all 6 pipelines)
│   │   │   ├── confusion_all_pipelines.png  ★ Side-by-side comparison
│   │   │   └── results_summary.txt    ★ Accuracy summary (P3 best: 96.4%)
│   │   ├── KaggleTestImage_confusionMatrix/
│   │   │   ├── confusion_*.png        Confusion matrices (Kaggle dataset)
│   │   │   └── results_summary.txt    Accuracy summary
│   │   ├── ISImage_fallsitwalk/       Per-image correct/wrong breakdown
│   │   ├── kaggle_fallsitwalk/        Per-image correct/wrong breakdown
│   │   └── Video_Evaluation_Comparison_EP+LL/
│   │       └── fall_*.mp4             Early prediction + low-light clips
│   ├── dataset/                       Test images (Fall/Sit/Walk)
│   ├── utils/                         Shared utilities
│   │   ├── pipeline_utils.py          Bbox padding, feature extraction
│   │   └── transition_detector.py     Temporal Walk→Fall smoothing
│   └── assets/                        Alert sound
│
├── V3/                             CNN-based pipeline (development)
│   ├── src/
│   │   ├── train_stage2_cnn.py        CNN training script
│   │   ├── evaluate_cnn.py            CNN evaluation with wrong-prediction analysis
│   │   ├── finetune_stage2_cnn.py     Fine-tuning script
│   │   └── extract_crops.py           Crop extraction for CNN training
│   ├── models/
│   │   ├── stage2_cnn.pt              Trained CNN weights
│   │   └── stage2_cnn_info.json       Model metadata
│   └── runs/evaluation/               CNN evaluation outputs
│       └── settings1/
│           ├── twostep_cnn/
│           │   ├── confusion_twostep_cnn.png   ★ CNN confusion matrix
│           │   ├── predictions_twostep_cnn.csv
│           │   └── wrong_predictions/          Misclassified images by error type
│           ├── ISImage_fallsitwalk/     6-pipeline confusion matrices
│           └── kaggle/                  Kaggle evaluation with per-image results
│
├── V2/                             MLP-based pipeline (development)
│   ├── src/
│   │   ├── train_stage2_pose.py       MLP training
│   │   ├── evaluate_twostep.py        Two-step evaluation
│   │   ├── extract_pose_features.py   Pose feature extraction
│   │   ├── visualize_fall.py          Interactive video visualizer
│   │   └── test_video.py             Batch video testing
│   ├── models/                        V2 model weights
│   └── runs/evaluation/              MLP evaluation outputs
│       ├── stage2_pose_confusion.png
│       └── twostep_results*/          Confusion matrices + prediction logs
│
├── src/                            V1 one-step pipeline
│   ├── train_onestep.py               YOLOv8 training
│   ├── evaluate.py                    One-step evaluation
│   └── runs/detect/                   Training run outputs
```

---

## Six Pipeline Modes

The GUI supports six pipeline modes selectable from a dropdown:

| Pipeline | Description | Stage 1 | Stage 2 | Best Accuracy |
|----------|-------------|---------|---------|---------------|
| **P1** | Raw CNN | None | EfficientNet-B0 on full image | 72.8% |
| **P2** | One-step YOLO | None | swook_techcare YOLOv8 | 75.1% |
| **P3** | Stage1 + CNN | YOLOv8 person detector | EfficientNet-B0 on crop | **96.4%** |
| **P4** | Stage1 + YOLO | YOLOv8 person detector | One-step YOLO on crop | 74.6% |
| **P5** | Raw MLP | None | MediaPipe → MLP | 66.9% |
| **P6** | Stage1 + MLP | YOLOv8 person detector | MediaPipe → MLP | 70.7% |

**P3 (Stage1 + CNN) is the recommended pipeline** with 96.4% accuracy on the manual test set.

---

## CNN Model Evaluation

### Where to Find Results

| What | Location |
|------|----------|
| **Best results (6 pipelines)** | `final_version/evaluation/ManualTestImage_confusionMatrix/` |
| **Accuracy summary** | `final_version/evaluation/ManualTestImage_confusionMatrix/results_summary.txt` |
| **All confusion matrices** | `final_version/evaluation/ManualTestImage_confusionMatrix/confusion_*.png` |
| **Combined comparison** | `final_version/evaluation/ManualTestImage_confusionMatrix/confusion_all_pipelines.png` |
| **Kaggle dataset results** | `final_version/evaluation/KaggleTestImage_confusionMatrix/` |
| **CNN-specific eval (V3)** | `V3/runs/evaluation/settings1/twostep_cnn/` |
| **CNN wrong predictions** | `V3/runs/evaluation/settings1/twostep_cnn/wrong_predictions/` |
| **Video evaluation outputs** | `Video_Evaluation/Video_Evaluation_EP+LL/` |

### How to Re-run Evaluation

```bash
# Evaluate all 6 pipelines on static images (final version)
python final_version/GUI_fall_detection.py

# Evaluate CNN two-step pipeline (V3)
python V3/src/evaluate_cnn.py

# Evaluate video (P3 vs P6 comparison)
python final_version/Evaluate_video.py

# Evaluate low-light vs baseline
python final_version/Evaluate_video_extension.py
```

---

## Development History

| Version | Stage 2 Classifier | Key Change |
|---------|-------------------|------------|
| **V1** | One-step YOLOv8 | Single model, direct classification |
| **V2** | MLP (8 pose features) | Two-step: person detect → MediaPipe → MLP |
| **V3** | EfficientNet-B0 CNN | Two-step: person detect → CNN on crop (no MediaPipe) |
| **Final** | All 6 pipelines | Combined GUI with P1–P6 + temporal smoothing |

---

## COS30018 — Intelligent Systems, Swinburne University
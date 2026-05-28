"""
COS30018 - Evaluate One-Step Model on Self-Collected Test Set
==============================================================
Evaluates a trained YOLOv8 model on test images sorted into class folders.
No YOLO label files needed — class is inferred from the folder name.

Expected folder structure:
    test_root/
    ├── fall/     ← images of fallen people
    ├── sit/      ← images of sitting people
    └── walk/     ← images of walking/standing people

Outputs:
    - Per-class precision, recall, F1
    - Overall accuracy
    - Confusion matrix (saved as PNG)
    - Per-image prediction log (CSV)
    - Failure case gallery (misclassified images saved separately)
    - Confidence distribution analysis

Usage:
    python evaluate_test.py --model path/to/best.pt --test-dir path/to/test/folder
    python evaluate_test.py --model best.pt --test-dir datasets/fall_sit_walk_test --conf 0.5
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("WARNING: matplotlib/seaborn not installed. Plots will be skipped.")


# Class mapping: folder name -> class ID (must match training data.yaml)
# Training used: 0=Fall, 1=Sit, 2=Walk
FOLDER_TO_CLASS = {
    'fall': 0,
    'sit': 1,
    'walk': 2,
}

CLASS_NAMES = {0: 'Fall', 1: 'Sit', 2: 'Walk'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def load_test_images(test_dir: str) -> list[dict]:
    """
    Load all test images from class-sorted folders.
    Returns list of dicts: {path, true_class_id, true_class_name, folder}
    """
    test_dir = Path(test_dir)
    images = []

    for folder_name, class_id in FOLDER_TO_CLASS.items():
        folder_path = test_dir / folder_name
        if not folder_path.exists():
            print(f"  WARNING: Folder '{folder_name}' not found in {test_dir}")
            continue

        folder_images = [
            f for f in sorted(folder_path.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for img_path in folder_images:
            images.append({
                'path': img_path,
                'true_class_id': class_id,
                'true_class_name': CLASS_NAMES[class_id],
                'folder': folder_name,
            })

        print(f"  {CLASS_NAMES[class_id]}: {len(folder_images)} images")

    return images


def two_step_predict(person_model, fall_model, img_path, conf, iou, imgsz, device):
    """Two-step: detect person, crop, classify fall/sit/walk."""
    import cv2
    frame = cv2.imread(str(img_path))
    if frame is None:
        return [], frame

    h, w = frame.shape[:2]
    p_result = person_model.predict(source=frame, conf=0.35, imgsz=imgsz, device=device, verbose=False)[0]

    person_boxes = []
    if p_result.boxes is not None:
        for box in p_result.boxes:
            cls_name = person_model.names[int(box.cls[0])].lower()
            if cls_name == 'person':
                person_boxes.append(box.xyxy[0].cpu().numpy().tolist())

    if not person_boxes:
        return [], frame

    best_cls = -1
    best_conf = 0.0
    best_name = 'No Detection'
    all_detections = []

    for xyxy in person_boxes:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        crop = frame[y1:y2, x1:x2]
        f_result = fall_model.predict(source=crop, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]

        if f_result.boxes is not None and len(f_result.boxes) > 0:
            confs = f_result.boxes.conf.cpu().numpy()
            classes = f_result.boxes.cls.cpu().numpy().astype(int)
            idx = confs.argmax()
            det_cls = classes[idx]
            det_conf = float(confs[idx])
            det_name = CLASS_NAMES.get(det_cls, f'Unknown({det_cls})')
            all_detections.append((det_cls, det_conf, det_name))
            if det_conf > best_conf:
                best_cls = det_cls
                best_conf = det_conf
                best_name = det_name

    return all_detections, frame


def evaluate(args):
    """Run evaluation on the self-collected test set."""

    mode_name = "Two-Step" if args.two_step else "One-Step"
    print("=" * 70)
    print(f"COS30018 - {mode_name} Fall Detection - Test Set Evaluation")
    print("=" * 70)

    # Load model
    print(f"\nFall Model: {args.model}")
    model = YOLO(args.model)

    person_model = None
    if args.two_step:
        print(f"Person Model: {args.person_model}")
        person_model = YOLO(args.person_model)

    # Load test images
    print(f"\nLoading test images from: {args.test_dir}")
    images = load_test_images(args.test_dir)

    if not images:
        print("ERROR: No images found. Check folder structure.")
        return

    print(f"\nTotal: {len(images)} test images")
    print(f"Confidence threshold: {args.conf}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / 'failure_cases'
    failures_dir.mkdir(exist_ok=True)
    success_dir = output_dir / 'success_cases'
    success_dir.mkdir(exist_ok=True)

    # Clear old cases
    for d in [failures_dir, success_dir]:
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    # Run predictions
    print("\nRunning inference...")

    results_log = []
    y_true = []
    y_pred = []
    confidences = []

    # Per-class tracking
    class_tp = defaultdict(int)      # True positives
    class_fp = defaultdict(int)      # False positives
    class_fn = defaultdict(int)      # False negatives
    class_total = defaultdict(int)   # Total ground truth per class

    no_detection_count = 0
    multi_class_count = 0

    for i, img_info in enumerate(images):
        img_path = img_info['path']
        true_class = img_info['true_class_id']
        true_name = img_info['true_class_name']

        # Run inference
        pred_class = -1
        pred_conf = 0.0
        pred_name = 'No Detection'
        num_detections = 0
        result = None

        if args.two_step and person_model:
            detections, frame = two_step_predict(
                person_model, model, img_path,
                args.conf, args.iou, args.imgsz, args.device
            )
            num_detections = len(detections)
            if num_detections > 0:
                best = max(detections, key=lambda d: d[1])
                pred_class, pred_conf, pred_name = best
                unique_classes = set(d[0] for d in detections)
                if len(unique_classes) > 1:
                    multi_class_count += 1
            else:
                no_detection_count += 1
            # Run one-step too for annotated image plot
            result = model.predict(
                source=str(img_path), conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, device=args.device, verbose=False,
            )[0]
        else:
            result = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            boxes = result.boxes
            num_detections = len(boxes)

            if num_detections > 0:
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                best_idx = confs.argmax()
                pred_class = classes[best_idx]
                pred_conf = confs[best_idx]
                pred_name = CLASS_NAMES.get(pred_class, f'Unknown({pred_class})')

                unique_classes = set(classes)
                if len(unique_classes) > 1:
                    multi_class_count += 1
            else:
                no_detection_count += 1

        # Record results
        is_correct = (pred_class == true_class)
        y_true.append(true_class)
        y_pred.append(pred_class)
        confidences.append(pred_conf)

        class_total[true_class] += 1

        if is_correct:
            class_tp[true_class] += 1
        else:
            class_fn[true_class] += 1
            if pred_class >= 0:
                class_fp[pred_class] += 1

        results_log.append({
            'filename': img_path.name,
            'folder': img_info['folder'],
            'true_class': true_name,
            'pred_class': pred_name,
            'confidence': f"{pred_conf:.4f}",
            'num_detections': num_detections,
            'correct': is_correct,
        })

        # Save annotated images for both failure and success cases
        if HAS_PLOTTING:
            annotated = result.plot()
            if not is_correct:
                save_name = f"TRUE_{true_name}_PRED_{pred_name}_{pred_conf:.2f}_{img_path.name}"
                save_dir = failures_dir
            else:
                save_name = f"TRUE_{true_name}_CONF_{pred_conf:.2f}_{img_path.name}"
                save_dir = success_dir

            plt.figure(figsize=(10, 8))
            plt.imshow(annotated[..., ::-1])  # BGR to RGB
            title = f"True: {true_name} | Pred: {pred_name} ({pred_conf:.2f})"
            if is_correct:
                title += " ✓"
            else:
                title += " ✗"
            plt.title(title)
            plt.axis('off')
            plt.savefig(save_dir / save_name.replace(img_path.suffix, '.png'),
                       bbox_inches='tight', dpi=100)
            plt.close()

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(images)} images...")

    # ========================================
    # Compute metrics
    # ========================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Overall accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"No Detection:    {no_detection_count} images ({no_detection_count/total:.1%})")
    print(f"Multi-class:     {multi_class_count} images (multiple classes in one image)")

    # Per-class metrics
    print(f"\n{'Class':<12} {'Total':>6} {'Correct':>8} {'Accuracy':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 70)

    class_metrics = {}
    for class_id in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[class_id]
        tp = class_tp[class_id]
        fp = class_fp[class_id]
        fn = class_fn[class_id]
        total_cls = class_total[class_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = tp / total_cls if total_cls > 0 else 0

        class_metrics[class_id] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': acc, 'tp': tp, 'fp': fp, 'fn': fn, 'total': total_cls
        }

        print(f"{name:<12} {total_cls:>6} {tp:>8} {acc:>9.1%} "
              f"{precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")

    # Macro averages
    macro_p = np.mean([m['precision'] for m in class_metrics.values()])
    macro_r = np.mean([m['recall'] for m in class_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
    print("-" * 70)
    print(f"{'Macro Avg':<12} {total:>6} {correct:>8} {accuracy:>9.1%} "
          f"{macro_p:>10.4f} {macro_r:>8.4f} {macro_f1:>8.4f}")

    # ========================================
    # Confusion Matrix
    # ========================================
    if HAS_PLOTTING:
        # Build confusion matrix (3x3 + "No Detection" row)
        num_classes = len(CLASS_NAMES)
        labels = list(range(num_classes))
        label_names = [CLASS_NAMES[i] for i in labels]

        # Standard 3x3 confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=int)
        no_det_per_class = defaultdict(int)

        for yt, yp in zip(y_true, y_pred):
            if yp >= 0 and yp < num_classes:
                cm[yt][yp] += 1
            else:
                no_det_per_class[yt] += 1

        # Plot confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix (Counts)')

        # Normalized
        cm_norm = cm.astype(float)
        for i in range(num_classes):
            row_sum = cm_norm[i].sum()
            if row_sum > 0:
                cm_norm[i] /= row_sum

        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix (Normalized)')

        plt.tight_layout()
        cm_path = output_dir / 'confusion_matrix_test.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved to: {cm_path}")

        # No-detection breakdown
        if sum(no_det_per_class.values()) > 0:
            print(f"\nNo-detection breakdown:")
            for cls_id, count in sorted(no_det_per_class.items()):
                print(f"  {CLASS_NAMES[cls_id]}: {count} images with no detection")

        # ========================================
        # Confidence Distribution
        # ========================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confidence for correct vs incorrect
        correct_confs = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt == yp and c > 0]
        incorrect_confs = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt != yp and c > 0]

        if correct_confs:
            axes[0].hist(correct_confs, bins=20, alpha=0.7, label=f'Correct (n={len(correct_confs)})',
                        color='green', edgecolor='black')
        if incorrect_confs:
            axes[0].hist(incorrect_confs, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect_confs)})',
                        color='red', edgecolor='black')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0].legend()

        # Confidence per class
        for cls_id in sorted(CLASS_NAMES.keys()):
            cls_confs = [c for c, yt in zip(confidences, y_true) if yt == cls_id and c > 0]
            if cls_confs:
                axes[1].hist(cls_confs, bins=20, alpha=0.5, label=CLASS_NAMES[cls_id],
                            edgecolor='black')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Confidence Distribution per True Class')
        axes[1].legend()

        plt.tight_layout()
        conf_path = output_dir / 'confidence_distribution_test.png'
        plt.savefig(conf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confidence distribution saved to: {conf_path}")

    # ========================================
    # Save prediction log CSV
    # ========================================
    csv_path = output_dir / 'prediction_log_test.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)
    print(f"Prediction log saved to: {csv_path}")

    # ========================================
    # Summary
    # ========================================
    failure_count = len(list(failures_dir.glob('*.png')))
    success_count = len(list(success_dir.glob('*.png')))
    print(f"Failure cases saved to: {failures_dir}/ ({failure_count} images)")
    print(f"Success cases saved to: {success_dir}/ ({success_count} images)")

    print("\n" + "=" * 70)
    print("SUMMARY FOR REPORT")
    print("=" * 70)
    print(f"Model:              YOLOv8m ({mode_name.lower()})")
    print(f"Test set:           {total} self-collected images")
    print(f"Overall accuracy:   {accuracy:.1%}")
    print(f"Macro Precision:    {macro_p:.4f}")
    print(f"Macro Recall:       {macro_r:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Confidence thresh:  {args.conf}")
    print(f"Optimal threshold:  0.533 (from F1 curve)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fall detection model on self-collected test set'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt)')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to test images root (with fall/sit/walk subfolders)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5, optimal from F1 curve)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--output', type=str, default='runs/evaluation/test_results',
                        help='Output directory for results')
    parser.add_argument('--two-step', action='store_true',
                        help='Use two-step detection (person detector + fall classifier)')
    parser.add_argument('--person-model', type=str, default='src/models/retrained.pt',
                        help='Path to person detection model (for two-step)')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
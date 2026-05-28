"""
COS30018 - Evaluate Fall Detection Model
==========================================
Evaluates a trained model on the self-collected test set.
Produces metrics required for the marking scheme:
  - mAP@0.5, mAP@0.5:0.95
  - Per-class precision, recall, F1
  - Confusion matrix
  - Condition breakdown (environment, viewpoint, distance)

Usage:
    python evaluate.py --model runs/onestep/train/weights/best.pt --data datasets/Fall_sit_walk_bbox/data.yaml
    python evaluate.py --model runs/onestep/train/weights/best.pt --test-images path/to/test/images
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def evaluate_on_test_set(args):
    """Run evaluation on the test set."""
    
    model = YOLO(args.model)
    
    print("=" * 60)
    print("COS30018 - Fall Detection Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data:  {args.data}")
    print("=" * 60)
    
    # Standard validation metrics
    metrics = model.val(
        data=args.data,
        split='test',           # Use test split from data.yaml
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,             # Generates confusion matrix, P/R curves
        save_json=True,         # Save predictions for further analysis
        verbose=True,
        project=args.output,
        name='evaluation',
        exist_ok=True,
    )
    
    # Print results
    class_names = ['Fall', 'Sit', 'Walk']
    
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    print("\nPER-CLASS METRICS:")
    print(f"{'Class':<10} {'AP@0.5':>8} {'AP@0.5:0.95':>12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    
    for i, name in enumerate(class_names):
        if i < len(metrics.box.ap50):
            p = metrics.box.p[i]
            r = metrics.box.r[i]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"{name:<10} {metrics.box.ap50[i]:>8.4f} {metrics.box.ap[i]:>12.4f} "
                  f"{p:>10.4f} {r:>8.4f} {f1:>8.4f}")
    
    print("=" * 60)
    print(f"\nPlots saved to: {args.output}/evaluation/")
    print("  - confusion_matrix.png")
    print("  - P_curve.png (Precision vs Confidence)")
    print("  - R_curve.png (Recall vs Confidence)")
    print("  - PR_curve.png (Precision-Recall)")
    print("  - F1_curve.png (F1 vs Confidence)")
    
    return metrics


def evaluate_by_condition(args):
    """
    Evaluate model performance broken down by condition.
    
    Requires a metadata CSV with columns:
        filename, environment, viewpoint, distance
    
    Example CSV:
        fall001.jpg, indoor, front, near
        walk015.jpg, outdoor, side, far
    """
    
    if not args.metadata:
        print("\nSkipping condition breakdown (no --metadata CSV provided)")
        print("To enable, create a CSV with columns: filename, environment, viewpoint, distance")
        return
    
    import csv
    
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    # Load metadata
    conditions = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename'].strip()
            conditions[filename] = {
                'environment': row.get('environment', 'unknown').strip().lower(),
                'viewpoint': row.get('viewpoint', 'unknown').strip().lower(),
                'distance': row.get('distance', 'unknown').strip().lower(),
            }
    
    print(f"\nLoaded metadata for {len(conditions)} images")
    
    # Run predictions on test images
    model = YOLO(args.model)
    test_images = Path(args.test_images) if args.test_images else None
    
    if not test_images or not test_images.exists():
        print("Provide --test-images path for condition breakdown")
        return
    
    # Collect results grouped by condition
    results_by_condition = {
        'environment': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'viewpoint': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'distance': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }
    
    # Run inference
    image_files = sorted(test_images.glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    results = model.predict(
        source=str(test_images),
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=False,
    )
    
    print(f"\nRan predictions on {len(results)} images")
    
    # Process results
    for result in results:
        filename = Path(result.path).name
        if filename not in conditions:
            continue
        
        meta = conditions[filename]
        
        # Check if prediction matches ground truth
        # (simplified - full mAP calculation per condition would require
        #  matching with ground truth labels which is more complex)
        has_detections = len(result.boxes) > 0
        
        for condition_type in ['environment', 'viewpoint', 'distance']:
            condition_value = meta[condition_type]
            results_by_condition[condition_type][condition_value]['total'] += 1
            if has_detections:
                results_by_condition[condition_type][condition_value]['correct'] += 1
    
    # Print condition breakdown
    print("\n" + "=" * 60)
    print("PERFORMANCE BY CONDITION")
    print("=" * 60)
    
    for condition_type, values in results_by_condition.items():
        print(f"\n{condition_type.upper()}:")
        print(f"  {'Value':<15} {'Images':>8} {'Detected':>10} {'Rate':>8}")
        print("  " + "-" * 45)
        for value, counts in sorted(values.items()):
            rate = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            print(f"  {value:<15} {counts['total']:>8} {counts['correct']:>10} {rate:>8.1%}")
    
    print("\n" + "=" * 60)
    print("NOTE: For full mAP per condition, use the YOLO val split approach")
    print("      by creating separate data.yaml files per condition subset.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='COS30018 Fall Detection Evaluation')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt)')
    parser.add_argument('--data', type=str,
                        default='datasets/Fall_sit_walk_bbox/data.yaml',
                        help='Path to dataset YAML')
    parser.add_argument('--test-images', type=str, default=None,
                        help='Path to test images directory (for condition breakdown)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to test metadata CSV (filename, environment, viewpoint, distance)')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='runs/evaluation',
                        help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_on_test_set(args)
    evaluate_by_condition(args)


if __name__ == '__main__':
    main()

"""
Convert YOLO polygon/segmentation labels to YOLO bounding box format.

Input format (polygon):  class_id x1 y1 x2 y2 x3 y3 ... xN yN
Output format (bbox):    class_id x_center y_center width height

All coordinates are normalized (0-1).

Usage:
    python convert_poly_to_bbox.py --input <input_labels_dir> --output <output_labels_dir>
    
Example:
    python convert_poly_to_bbox.py \
        --input  datasets/Fall_sit_walk/train/labels \
        --output datasets/Fall_sit_walk_bbox/train/labels
"""

import os
import argparse
from pathlib import Path


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """
    Convert polygon coordinates to bounding box.
    
    Args:
        coords: List of alternating x, y values [x1, y1, x2, y2, ..., xN, yN]
    
    Returns:
        Tuple of (x_center, y_center, width, height) in normalized coordinates
    """
    x_coords = coords[0::2]  # Every even index = x
    y_coords = coords[1::2]  # Every odd index = y
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def convert_label_file(input_path: Path, output_path: Path) -> dict:
    """
    Convert a single label file from polygon to bbox format.
    
    Returns:
        Dict with counts per class for statistics.
    """
    class_counts = {}
    bbox_lines = []
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = parts[0]
            coords = [float(v) for v in parts[1:]]
            
            # Validate: need at least 3 points (6 values) for a polygon
            if len(coords) < 6:
                # Already bbox format or invalid — check if it's 4 values (already bbox)
                if len(coords) == 4:
                    bbox_lines.append(line)  # Keep as-is
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    continue
                else:
                    print(f"  WARNING: Skipping malformed line in {input_path.name}: {line[:80]}...")
                    continue
            
            x_center, y_center, width, height = polygon_to_bbox(coords)
            
            # Clamp values to [0, 1] range
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            bbox_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            bbox_lines.append(bbox_line)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(bbox_lines) + '\n')
    
    return class_counts


def convert_dataset(input_dir: str, output_dir: str):
    """
    Convert all label files in a directory tree.
    Handles structure like:
        input_dir/train/labels/*.txt
        input_dir/valid/labels/*.txt
        input_dir/test/labels/*.txt
    OR flat:
        input_dir/*.txt
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all .txt label files
    label_files = list(input_path.rglob('*.txt'))
    # Exclude README files
    label_files = [f for f in label_files if 'README' not in f.name and 'classes' not in f.name]
    
    if not label_files:
        print(f"ERROR: No .txt label files found in {input_dir}")
        return
    
    print(f"Found {len(label_files)} label files to convert")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    total_counts = {}
    converted = 0
    errors = 0
    already_bbox = 0
    
    for label_file in sorted(label_files):
        # Preserve directory structure relative to input
        relative = label_file.relative_to(input_path)
        out_file = output_path / relative
        
        try:
            counts = convert_label_file(label_file, out_file)
            for cls, cnt in counts.items():
                total_counts[cls] = total_counts.get(cls, 0) + cnt
            converted += 1
        except Exception as e:
            print(f"  ERROR processing {label_file.name}: {e}")
            errors += 1
    
    # Print summary
    print("-" * 60)
    print(f"Converted: {converted} files")
    if errors:
        print(f"Errors:    {errors} files")
    print(f"\nClass distribution:")
    
    class_names = {
        '0': 'Fall',
        '1': 'Sit', 
        '2': 'Walk'
    }
    
    for cls_id in sorted(total_counts.keys()):
        name = class_names.get(cls_id, f'Unknown({cls_id})')
        print(f"  Class {cls_id} ({name}): {total_counts[cls_id]} annotations")
    
    total = sum(total_counts.values())
    print(f"  Total: {total} annotations across {converted} images")
    
    # Check for class imbalance
    if total_counts:
        max_count = max(total_counts.values())
        min_count = min(total_counts.values())
        if max_count > 3 * min_count:
            print(f"\n  ⚠ WARNING: Class imbalance detected (ratio {max_count/min_count:.1f}x)")
            print(f"  Consider using class weights during training")


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO polygon labels to bounding box format'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing polygon label files'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for bbox label files'
    )
    
    args = parser.parse_args()
    convert_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
"""
train_onestep.py  (updated for merged dataset)
Trains YOLOv8 one-step fall detection model.

Usage:
  python src/train_onestep.py --data datasets/merged_dataset/data.yaml
  python src/train_onestep.py --data datasets/merged_dataset/data.yaml --epochs 5 --name test_run
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="datasets/merged_dataset/data.yaml")
    p.add_argument("--model",  default="yolov8m.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--name",   default="onestep_merged")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of training data held out as validation")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"Model  : {args.model}")
    print(f"Data   : {args.data}")
    print(f"Epochs : {args.epochs}  |  Batch: {args.batch}  |  ImgSz: {args.imgsz}")
    print(f"Device : {args.device}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data        = args.data,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        name        = args.name,
        patience    = 25,           # early stopping — more patience for larger dataset
        val         = True,
        fraction    = 1 - args.val_split,   # use 85% for train, 15% auto-held for val

        # Augmentation — stronger than before to build viewpoint robustness
        hsv_h       = 0.015,        # hue shift
        hsv_s       = 0.7,          # saturation
        hsv_v       = 0.4,          # brightness/value  ← helps with low-light extension
        degrees     = 10,           # rotation
        translate   = 0.1,
        scale       = 0.6,          # scale jitter — helps with far/close distance variation
        shear       = 5,
        perspective = 0.0005,       # slight perspective warp — helps with viewpoint variation
        flipud      = 0.1,          # vertical flip — rare but helps overhead fall poses
        fliplr      = 0.5,
        mosaic      = 1.0,
        mixup       = 0.15,
        copy_paste  = 0.1,

        # Optimiser
        optimizer   = "AdamW",
        lr0         = 0.001,
        lrf         = 0.01,
        weight_decay= 0.0005,
        warmup_epochs = 3,

        # Logging
        plots       = True,
        save        = True,
        save_period = 10,           # checkpoint every 10 epochs
        verbose     = True,
    )

    best = Path("runs/detect") / args.name / "weights/best.pt"
    print(f"\n✅ Training complete.")
    print(f"   Best weights: {best}")
    print(f"\nNext — evaluate on your self-collected test set:")
    print(f"   python src/evaluate_test.py --model {best} --test-dir datasets/fall_sit_walk_test --conf 0.3")

if __name__ == "__main__":
    main()
"""
train_stage2_cnn.py
===================
Fine-tunes EfficientNet-B0 on extracted person crops for
Fall / Sit / Walk classification (Stage 2 of two-step pipeline).

Why EfficientNet-B0:
  - Pretrained on ImageNet — already understands body shapes, textures
  - 5.3M parameters — fast enough for real-time inference
  - Proven strong on medical/pose image classification tasks
  - Much stronger than MLP on visual ambiguity (Fall vs Sit on floor)

Training strategy:
  Phase 1 (epochs 1-10): Freeze backbone, train classifier head only
  Phase 2 (epochs 11-30): Unfreeze top layers, fine-tune end-to-end

Output:
  V3/models/stage2_cnn.pt       — best model weights
  V3/models/stage2_cnn_info.txt — class names and input size

Usage:
    python V3/src/train_stage2_cnn.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =============================================================================
#  PATHS
# =============================================================================

V3_BASE   = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V3")
DATA_DIR  = V3_BASE / "datasets" / "stage2_crops"
MODEL_OUT = V3_BASE / "models" / "stage2_cnn.pt"
INFO_OUT  = V3_BASE / "models" / "stage2_cnn_info.json"

# =============================================================================
#  CONFIGURATION
# =============================================================================

IMG_SIZE    = 224
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Phase 1 — head only
PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-3

# Phase 2 — fine-tune top layers
PHASE2_EPOCHS = 20
PHASE2_LR     = 1e-4

CLASS_NAMES = ["Fall", "Sit", "Walk"]   # must match folder names in stage2_crops

# =============================================================================
#  DATA
# =============================================================================

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def get_loaders():
    train_tf, val_tf = get_transforms()

    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} crops | Val: {len(val_ds)} crops")
    print(f"Class mapping: {train_ds.class_to_idx}")

    # Verify class order matches expected
    expected = {"Fall": 0, "Sit": 1, "Walk": 2}
    if train_ds.class_to_idx != expected:
        print(f"  [WARN] Class mapping differs from expected {expected}")
        print(f"         Actual: {train_ds.class_to_idx}")
        print(f"         Update CLASS_NAMES in pipeline_utils.py to match actual order")

    return train_loader, val_loader, train_ds.class_to_idx

# =============================================================================
#  MODEL
# =============================================================================

def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head for 3-class output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 3),
    )

    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_top(model, n_blocks=3):
    """Unfreeze the top N blocks of the EfficientNet backbone for fine-tuning."""
    # EfficientNet-B0 features has blocks 0-8
    # Unfreeze classifier + last n_blocks of features
    for param in model.classifier.parameters():
        param.requires_grad = True

    blocks = list(model.features.children())
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total:.1%})")

# =============================================================================
#  TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    # Per-class tracking
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            for i in range(3):
                mask = labels == i
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                class_total[i]   += mask.sum().item()

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            per_class[name] = class_correct[i] / class_total[i]
        else:
            per_class[name] = 0.0

    return total_loss / total, correct / total, per_class


def run_phase(model, train_loader, val_loader, criterion,
              n_epochs, lr, phase_name):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs,
    )

    best_val_acc = 0.0
    best_state   = None

    print(f"\n  {'Epoch':<8} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VaLoss':>8} {'VaAcc':>7} "
          f"{'Fall':>7} {'Sit':>7} {'Walk':>7}")
    print(f"  {'-'*70}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, per_cls = val_epoch(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  {epoch:<8} {tr_loss:>8.4f} {tr_acc:>7.1%} "
              f"{va_loss:>8.4f} {va_acc:>7.1%} "
              f"{per_cls['Fall']:>7.1%} {per_cls['Sit']:>7.1%} "
              f"{per_cls['Walk']:>7.1%}  ({elapsed:.0f}s)")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_acc, best_state

# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("="*60)
    print("  COS30018 — Stage 2 CNN Training (EfficientNet-B0)")
    print("="*60)
    print(f"  Device : {DEVICE}")
    print(f"  Data   : {DATA_DIR}")
    print()

    # Verify data exists
    if not (DATA_DIR / "train").exists():
        print("[ERROR] Training data not found.")
        print("        Run extract_crops.py first.")
        return

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = get_loaders()

    # Class weights for imbalanced dataset
    # Fall typically underrepresented vs Walk
    class_counts = [0, 0, 0]
    for _, label in train_loader.dataset:
        class_counts[label] += 1
    total = sum(class_counts)
    weights = torch.tensor(
        [total / (3 * c) if c > 0 else 1.0 for c in class_counts],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"\n  Class counts  : Fall={class_counts[0]}, "
          f"Sit={class_counts[1]}, Walk={class_counts[2]}")
    print(f"  Class weights : {weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    model     = build_model().to(DEVICE)

    # ── Phase 1: Head only ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 — Classifier head only ({PHASE1_EPOCHS} epochs, lr={PHASE1_LR})")
    print(f"{'─'*60}")
    freeze_backbone(model)
    unfreeze_top(model, n_blocks=0)

    best_acc_p1, best_state_p1 = run_phase(
        model, train_loader, val_loader, criterion,
        PHASE1_EPOCHS, PHASE1_LR, "Phase1",
    )
    print(f"\n  Phase 1 best val accuracy: {best_acc_p1:.1%}")

    # Load best Phase 1 weights before Phase 2
    model.load_state_dict(best_state_p1)

    # ── Phase 2: Fine-tune top blocks ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — Fine-tune top 3 blocks ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR})")
    print(f"{'─'*60}")
    unfreeze_top(model, n_blocks=3)

    best_acc_p2, best_state_p2 = run_phase(
        model, train_loader, val_loader, criterion,
        PHASE2_EPOCHS, PHASE2_LR, "Phase2",
    )
    print(f"\n  Phase 2 best val accuracy: {best_acc_p2:.1%}")

    # ── Save best overall model ───────────────────────────────────────────────
    best_state = best_state_p2 if best_acc_p2 >= best_acc_p1 else best_state_p1
    best_acc   = max(best_acc_p1, best_acc_p2)

    torch.save(best_state, str(MODEL_OUT))
    print(f"\n  Model saved → {MODEL_OUT}")

    # Save class mapping info
    info = {
        "class_names":  CLASS_NAMES,
        "class_to_idx": class_to_idx,
        "img_size":     IMG_SIZE,
        "best_val_acc": round(best_acc, 4),
        "architecture": "efficientnet_b0",
    }
    with open(INFO_OUT, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Info saved  → {INFO_OUT}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_acc:.1%}")
    print(f"{'='*60}")
    print(f"\n  Target for Stage 2:")
    print(f"    Fall recall  > 0.75")
    print(f"    Sit recall   > 0.75")
    print(f"    Walk recall  > 0.80")
    print(f"\n  Next: python V3/src/evaluate_twostep.py")


if __name__ == "__main__":
    main()
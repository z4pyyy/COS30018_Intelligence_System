"""
finetune_stage2_cnn.py  (V3)
==============================
Loads existing stage2_cnn.pt and fine-tunes on:
  - Original train/ crops  (unchanged)
  - finetune/ crops        (hard negatives — wrong predictions × 15 augmentations)

Why separate from train_stage2_cnn.py:
  - Fine-tuning hard negatives needs SHORT training (8 epochs, not 30)
  - Low learning rate (1e-4) to avoid overwriting what the model already knows
  - Full retrain regime would waste 2+ hours and risk hurting Sit/Walk recall

Training strategy:
  - Load existing stage2_cnn.pt weights
  - Unfreeze top 3 blocks + classifier (same as Phase 2 of original training)
  - ConcatDataset: train/ + finetune/ merged as one training set
  - Val set unchanged (val/ folder only — no finetune data in val)
  - 8 epochs, LR=1e-4, cosine annealing
  - Save best val accuracy model → stage2_cnn.pt (overwrites)
  - Also saves stage2_cnn_before_finetune.pt as backup

Usage:
    python V3/src/finetune_stage2_cnn.py
"""

import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models

# =============================================================================
#  PATHS
# =============================================================================

V3_BASE      = Path(r"C:\z4pyyy\Swinburne_Y2S2\Intel_System\COS30018_Fall-Detection\V3")
DATA_DIR     = V3_BASE / "datasets" / "stage2_crops"
FINETUNE_DIR = DATA_DIR / "finetune"
TRAIN_DIR    = DATA_DIR / "train"
VAL_DIR      = DATA_DIR / "val"

MODEL_PATH   = V3_BASE / "models" / "stage2_cnn.pt"
INFO_PATH    = V3_BASE / "models" / "stage2_cnn_info.json"
BACKUP_PATH  = V3_BASE / "models" / "stage2_cnn_before_finetune.pt"

# =============================================================================
#  CONFIGURATION
# =============================================================================

IMG_SIZE       = 224
BATCH_SIZE     = 32
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNE_EPOCHS = 8
FINETUNE_LR    = 1e-4
CLASS_NAMES    = ["Fall", "Sit", "Walk"]

# =============================================================================
#  DATA
# =============================================================================

def get_transforms():
    # Finetune gets same augmentation as original training
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

    # Original train set
    train_ds    = datasets.ImageFolder(str(TRAIN_DIR),    transform=train_tf)
    # Hard-negative finetune set — same transform pipeline
    finetune_ds = datasets.ImageFolder(str(FINETUNE_DIR), transform=train_tf)
    # Val set — unchanged, no finetune data leaks in
    val_ds      = datasets.ImageFolder(str(VAL_DIR),      transform=val_tf)

    # Verify class order matches between all three datasets
    expected = {"Fall": 0, "Sit": 1, "Walk": 2}
    for name, ds in [("train", train_ds), ("finetune", finetune_ds)]:
        if ds.class_to_idx != expected:
            print(f"  [WARN] {name} class mapping differs from expected {expected}")
            print(f"         Actual: {ds.class_to_idx}")
            print(f"         Check folder names in {name}/ match Fall/Sit/Walk exactly")

    # Merge: original train + finetune hard negatives
    combined_ds = ConcatDataset([train_ds, finetune_ds])

    train_loader = DataLoader(
        combined_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Count per-class for combined dataset
    # ConcatDataset doesn't expose class counts directly — compute from both
    ft_counts = [0, 0, 0]
    for _, label in finetune_ds:
        ft_counts[label] += 1

    tr_counts = [0, 0, 0]
    for _, label in train_ds:
        tr_counts[label] += 1

    combined_counts = [tr_counts[i] + ft_counts[i] for i in range(3)]

    print(f"\n  {'Class':<8} {'Train':>8} {'Finetune':>10} {'Combined':>10}")
    print(f"  {'-'*42}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<8} {tr_counts[i]:>8} {ft_counts[i]:>10} {combined_counts[i]:>10}")
    print(f"  {'Total':<8} {sum(tr_counts):>8} {sum(ft_counts):>10} {sum(combined_counts):>10}")
    print(f"\n  Val: {len(val_ds)} crops")

    return train_loader, val_loader, combined_counts, train_ds.class_to_idx


# =============================================================================
#  MODEL
# =============================================================================

def build_model():
    model = models.efficientnet_b0(weights=None)  # weights loaded from checkpoint
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 3),
    )
    return model


def unfreeze_top(model, n_blocks=3):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze top N backbone blocks
    blocks = list(model.features.children())
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")


# =============================================================================
#  TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0, 0, 0]
    class_total   = [0, 0, 0]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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

    per_class = {
        CLASS_NAMES[i]: class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0.0
        for i in range(3)
    }
    return total_loss / total, correct / total, per_class


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  COS30018 — Stage 2 CNN Fine-Tune (Hard Negatives)")
    print("=" * 65)
    print(f"  Device     : {DEVICE}")
    print(f"  Base model : {MODEL_PATH}")
    print(f"  Finetune   : {FINETUNE_DIR}")

    # ── Validate paths ────────────────────────────────────────────────────────
    for path, label in [(TRAIN_DIR, "train/"), (VAL_DIR, "val/"),
                        (FINETUNE_DIR, "finetune/"), (MODEL_PATH, "stage2_cnn.pt")]:
        if not path.exists():
            print(f"\n  [ERROR] Not found: {path}")
            print(f"          Cannot proceed without {label}")
            return

    # ── Backup existing model ─────────────────────────────────────────────────
    shutil.copy2(str(MODEL_PATH), str(BACKUP_PATH))
    print(f"\n  Backup saved → {BACKUP_PATH.name}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, combined_counts, class_to_idx = get_loaders()

    # Class weights from combined counts (Fall still most underrepresented)
    total   = sum(combined_counts)
    weights = torch.tensor(
        [total / (3 * c) if c > 0 else 1.0 for c in combined_counts],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"\n  Class weights: "
          f"Fall={weights[0]:.3f}  Sit={weights[1]:.3f}  Walk={weights[2]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state)
    print(f"\n  Loaded weights from stage2_cnn.pt")

    # Unfreeze top 3 blocks + classifier (same as Phase 2 of original training)
    unfreeze_top(model, n_blocks=3)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  Fine-tuning: {FINETUNE_EPOCHS} epochs, LR={FINETUNE_LR}")
    print(f"{'─' * 65}")
    print(f"  {'Epoch':<8} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VaLoss':>8} {'VaAcc':>7} "
          f"{'Fall':>7} {'Sit':>7} {'Walk':>7}")
    print(f"  {'-'*65}")

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, per_cls = val_epoch(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        marker  = " ◀ best" if va_acc > best_val_acc else ""
        print(f"  {epoch:<8} {tr_loss:>8.4f} {tr_acc:>7.1%} "
              f"{va_loss:>8.4f} {va_acc:>7.1%} "
              f"{per_cls['Fall']:>7.1%} {per_cls['Sit']:>7.1%} "
              f"{per_cls['Walk']:>7.1%}  ({elapsed:.0f}s){marker}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(best_state, str(MODEL_PATH))
    print(f"\n  Model saved  → {MODEL_PATH.name}  (val acc: {best_val_acc:.1%})")

    # Update info json
    if INFO_PATH.exists():
        with open(INFO_PATH) as f:
            info = json.load(f)
        info["best_val_acc_after_finetune"] = round(best_val_acc, 4)
        info["finetune_epochs"] = FINETUNE_EPOCHS
        info["finetune_lr"]     = FINETUNE_LR
        with open(INFO_PATH, "w") as f:
            json.dump(info, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  FINE-TUNE COMPLETE — best val acc: {best_val_acc:.1%}")
    print(f"{'=' * 65}")
    print(f"\n  If results are worse than backup:")
    print(f"    copy {BACKUP_PATH.name} → stage2_cnn.pt")
    print(f"\n  Next: python V3/src/evaluate_twostep.py")


if __name__ == "__main__":
    main()
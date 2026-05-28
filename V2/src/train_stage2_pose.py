"""
train_stage2_pose.py
Trains a lightweight MLP classifier on pose keypoint features.
Input: 8 geometric features from MediaPipe skeleton
Output: Fall (0) / Sit (1) / Walk (2)

Usage:
    python V2/src/train_stage2_pose.py
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

BASE = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "spine_angle", "hip_height_norm", "shoulder_width_norm",
    "knee_bend_left", "knee_bend_right", "body_aspect_ratio",
    "head_height_norm", "ankle_hip_vert_dist",
]
CLASS_NAMES = ["Fall", "Sit", "Walk"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseMLP(nn.Module):
    def __init__(self, input_dim=8, hidden=128, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    data_dir = BASE / "datasets" / "pose_features"
    model_dir = BASE / "models"
    eval_dir = BASE / "runs" / "evaluation"
    model_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train_features_v2.csv").dropna()
    test_df = pd.read_csv(data_dir / "test_features_v2.csv").dropna()

    print(f"Train: {len(train_df)} samples")
    print(f"  Per class: {dict(train_df['class_name'].value_counts())}")
    print(f"Test:  {len(test_df)} samples")
    print(f"  Per class: {dict(test_df['class_name'].value_counts())}")

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["class_id"].values.astype(np.int64)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["class_id"].values.astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open(model_dir / "pose_scaler_v2.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    counts = np.bincount(y_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)

    model = PoseMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_acc = 0.0
    for epoch in range(1, 151):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            acc = correct / total
            print(f"  Epoch {epoch:3d}  test acc: {acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), model_dir / "stage2_pose_mlp_v2.pt")

    # Final evaluation
    model.load_state_dict(torch.load(model_dir / "stage2_pose_mlp_v2.pt", map_location=DEVICE))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    print("\n" + "=" * 50)
    print("FINAL TEST SET RESULTS — Stage 2 Pose Classifier")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage 2 Pose Classifier — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(eval_dir / "stage2_pose_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Best accuracy: {best_acc:.3f}")
    print(f"Weights -> {model_dir / 'stage2_pose_mlp_v2.pt'}")
    print(f"Scaler  -> {model_dir / 'pose_scaler_v2.pkl'}")


if __name__ == "__main__":
    main()

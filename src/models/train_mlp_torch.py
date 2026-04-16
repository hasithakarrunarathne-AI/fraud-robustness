# src/attacks/train_mlp_torch.py

import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score


RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_data(processed_dir="data/processed"):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv")).values.astype(np.float32)
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv")).values.astype(np.float32)
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns").values.astype(np.float32)
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns").values.astype(np.float32)

    return X_train, X_test, y_train, y_test


class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "pr_auc": average_precision_score(y_test, probs),
    }

    return metrics


def train_model():
    set_seed()

    print("Using device:", DEVICE)

    X_train, X_test, y_train, y_test = load_processed_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train fraud count:", int(y_train.sum()))
    print("y_test fraud count :", int(y_test.sum()))

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    input_dim = X_train.shape[1]
    model = FraudMLP(input_dim).to(DEVICE)

    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count
    pos_weight = torch.tensor([non_fraud_count / fraud_count], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(X_train_tensor).squeeze(1)
        loss = criterion(logits, y_train_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

    metrics = evaluate_model(model, X_test, y_test)

    print("\nClean test performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    os.makedirs("results/saved_models", exist_ok=True)
    torch.save(model.state_dict(), "results/saved_models/mlp_torch.pth")
    joblib.dump({"input_dim": input_dim}, "results/saved_models/mlp_torch_meta.pkl")

    print("\nSaved model to results/saved_models/mlp_torch.pth")


if __name__ == "__main__":
    train_model()
# src/attacks/pgd_attack.py

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
THRESHOLD = 0.9

attack_name = "PGD"

os.makedirs("results/attacks/samples", exist_ok=True)

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
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def evaluate_model(model, X_data, y_data, threshold=THRESHOLD):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_data, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_data, preds),
        "precision": precision_score(y_data, preds, zero_division=0),
        "recall": recall_score(y_data, preds, zero_division=0),
        "f1": f1_score(y_data, preds, zero_division=0),
        "pr_auc": average_precision_score(y_data, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, probs, preds


def pgd_attack(model, X_batch, y_batch, epsilon, alpha, num_steps):
    model.eval()

    X_orig = X_batch.clone().detach().to(DEVICE)
    y_batch = y_batch.clone().detach().to(DEVICE)

    X_adv = X_orig.clone().detach()

    for _ in range(num_steps):
        X_adv.requires_grad_(True)

        logits = model(X_adv).squeeze(1)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, y_batch)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()

            perturbation = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = X_orig + perturbation

    return X_adv.detach()


def attack_fraud_samples_only(model, X_test, y_test, epsilon, alpha, num_steps):
    fraud_idx = np.where(y_test == 1)[0]

    X_adv_full = X_test.copy()

    if len(fraud_idx) == 0:
        return X_adv_full, fraud_idx

    X_fraud = torch.tensor(X_test[fraud_idx], dtype=torch.float32)
    y_fraud = torch.tensor(y_test[fraud_idx], dtype=torch.float32)

    adv_batches = []

    for start in range(0, len(X_fraud), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_X = X_fraud[start:end]
        batch_y = y_fraud[start:end]

        batch_adv = pgd_attack(
            model=model,
            X_batch=batch_X,
            y_batch=batch_y,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
        )
        adv_batches.append(batch_adv.cpu().numpy())

    X_fraud_adv = np.vstack(adv_batches)
    X_adv_full[fraud_idx] = X_fraud_adv

    return X_adv_full, fraud_idx


def compute_asr(clean_preds, adv_preds, y_true, attacked_idx):
    if len(attacked_idx) == 0:
        return 0.0, 0, 0

    y_attacked = y_true[attacked_idx]
    clean_attacked = clean_preds[attacked_idx]
    adv_attacked = adv_preds[attacked_idx]

    attempted_mask = (y_attacked == 1) & (clean_attacked == 1)
    success_mask = attempted_mask & (adv_attacked == 0)

    attempts = int(attempted_mask.sum())
    successes = int(success_mask.sum())

    asr = successes / attempts if attempts > 0 else 0.0

    return asr, successes, attempts


def run_pgd_attack():
    set_seed()

    print("Using device:", DEVICE)

    X_train, X_test, y_train, y_test = load_processed_data()

    print("X_test shape:", X_test.shape)
    print("Fraud count in y_test:", int(y_test.sum()))

    input_dim = X_train.shape[1]

    model = FraudMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load("results/saved_models/mlp_torch.pth", map_location=DEVICE))
    model.eval()

    print("\nLoaded model from results/saved_models/mlp_torch.pth")

    epsilons = [0.1, 0.15, 0.2] #[0.03, 0.05, 0.1] #[0.01, 0.03, 0.05, 0.1]
    alpha = 0.01 #0.02 # 0.005
    num_steps = 40 #20 #10

    all_results = []

    clean_metrics, clean_probs, clean_preds = evaluate_model(model, X_test, y_test, threshold=THRESHOLD)

    print("\nClean test performance:")
    for k, v in clean_metrics.items():
        if k in ["tn", "fp", "fn", "tp"]:
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")

    feature_names = pd.read_csv("data/processed/X_test.csv", nrows=1).columns.tolist()
    
    for epsilon in epsilons:
        safe_eps = str(epsilon).replace(".", "p")
        print(f"\nRunning PGD with epsilon = {epsilon}, alpha = {alpha}, steps = {num_steps}")

        X_test_adv, attacked_idx = attack_fraud_samples_only(
            model=model,
            X_test=X_test,
            y_test=y_test,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
        )

        adv_metrics, adv_probs, adv_preds = evaluate_model(model, X_test_adv, y_test, threshold=THRESHOLD)

        asr, successes, attempts = compute_asr(clean_preds, adv_preds, y_test, attacked_idx)

        result = {
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "num_steps": int(num_steps),

            "clean_accuracy": float(clean_metrics["accuracy"]),
            "clean_precision": float(clean_metrics["precision"]),
            "clean_recall": float(clean_metrics["recall"]),
            "clean_f1": float(clean_metrics["f1"]),
            "clean_pr_auc": float(clean_metrics["pr_auc"]),
            "clean_tn": int(clean_metrics["tn"]),
            "clean_fp": int(clean_metrics["fp"]),
            "clean_fn": int(clean_metrics["fn"]),
            "clean_tp": int(clean_metrics["tp"]),

            "adv_accuracy": float(adv_metrics["accuracy"]),
            "adv_precision": float(adv_metrics["precision"]),
            "adv_recall": float(adv_metrics["recall"]),
            "adv_f1": float(adv_metrics["f1"]),
            "adv_pr_auc": float(adv_metrics["pr_auc"]),
            "adv_tn": int(adv_metrics["tn"]),
            "adv_fp": int(adv_metrics["fp"]),
            "adv_fn": int(adv_metrics["fn"]),
            "adv_tp": int(adv_metrics["tp"]),

            "attacked_fraud_samples": int(len(attacked_idx)),
            "successful_attacks": int(successes),
            "attack_attempts": int(attempts),
            "asr": float(asr),
        }

        np.savez_compressed(
            f"results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.npz",
            X_adv=X_test_adv.astype(np.float32),
            y_test=y_test.astype(np.int32),
            attacked_idx=np.array(attacked_idx, dtype=np.int32),
            clean_probs=clean_probs.astype(np.float32),
            clean_preds=clean_preds.astype(np.int32),
            adv_probs=adv_probs.astype(np.float32),
            adv_preds=adv_preds.astype(np.int32),
            attack_name=np.array(attack_name),
            epsilon=np.float32(epsilon),
        )

        adv_df = pd.DataFrame(X_test_adv, columns=feature_names)

        adv_df.to_csv(
            f"results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.csv",
            index=False
        )

        print(f"Saved attacked samples to results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.npz")
        print(f"Saved attacked CSV to results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.csv")

        all_results.append(result)

        print(f"ASR: {asr:.4f}")
        print(f"Successful attacks: {successes}")
        print(f"Attack attempts: {attempts}")

        print("Adversarial test performance:")
        for k, v in adv_metrics.items():
            if k in ["tn", "fp", "fn", "tp"]:
                print(f"{k}: {int(v)}")
            else:
                print(f"{k}: {v:.4f}")

    os.makedirs("results/attacks", exist_ok=True)

    output_path = "results/attacks/pgd_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved PGD results to {output_path}")


if __name__ == "__main__":
    run_pgd_attack()
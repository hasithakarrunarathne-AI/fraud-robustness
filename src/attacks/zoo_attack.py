# src/attacks/zoo_attack.py

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

THRESHOLD = 0.9

# Keep this small enough so ZOO runtime is manageable
MAX_ATTACK_SAMPLES = 50 #100

# ZOO settings
MAX_ITERS = 120 #100 #80 #40
LEARNING_RATE = 0.15 #0.1 #0.05
FD_STEP = 0.001 #1e-3                 # finite difference step h
LAMBDA_L2 =0.0 # 0.001              # small penalty to keep perturbation smaller
COORDS_PER_ITER = 15 #12 # 8            # update only some features each iteration


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


def predict_prob(model, x_np):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_np.reshape(1, -1), dtype=torch.float32).to(DEVICE)
        logit = model(x_tensor).squeeze(1)
        prob = torch.sigmoid(logit).cpu().numpy()[0]
    return float(prob)


def zoo_objective(model, x_adv, x_orig):
    """
    Objective to minimize:
    fraud probability + small L2 distance penalty

    Since fraud class is label 1, attacker wants to reduce fraud probability.
    """
    fraud_prob = predict_prob(model, x_adv)
    l2_term = LAMBDA_L2 * np.sum((x_adv - x_orig) ** 2)
    return float(fraud_prob + l2_term)


def estimate_gradient(model, x_adv, x_orig, coord_indices, x_min, x_max):
    """
    Central-difference gradient estimate on selected coordinates only.
    """
    grad = np.zeros_like(x_adv, dtype=np.float32)
    query_count = 0

    for idx in coord_indices:
        x_plus = x_adv.copy()
        x_minus = x_adv.copy()

        x_plus[idx] = min(x_plus[idx] + FD_STEP, x_max[idx])
        x_minus[idx] = max(x_minus[idx] - FD_STEP, x_min[idx])

        f_plus = zoo_objective(model, x_plus, x_orig)
        f_minus = zoo_objective(model, x_minus, x_orig)

        grad[idx] = (f_plus - f_minus) / (2.0 * FD_STEP)
        query_count += 2

    return grad, query_count


def zoo_attack_one(model, x, y_true, x_min, x_max):
    """
    Attack one sample using practical ZOO.
    Only meaningful for fraud sample y=1 that is currently predicted as fraud.
    """
    x_orig = x.copy().astype(np.float32)
    x_adv = x.copy().astype(np.float32)

    original_prob = predict_prob(model, x_orig)
    original_pred = int(original_prob >= THRESHOLD)

    total_queries = 1

    result = {
        "success": False,
        "original_prob": float(original_prob),
        "original_pred": int(original_pred),
        "adv_prob": float(original_prob),
        "adv_pred": int(original_pred),
        "queries": int(total_queries),
        "l2_dist": 0.0,
        "linf_dist": 0.0,
        "x_adv": x_adv.copy(),
    }

    # Only attack fraud samples correctly detected on clean input
    if int(y_true) != 1 or original_pred != 1:
        return result

    n_features = x_adv.shape[0]
    best_x_adv = x_adv.copy()
    best_obj = zoo_objective(model, x_adv, x_orig)
    total_queries += 1

    for _ in range(MAX_ITERS):
        k = min(COORDS_PER_ITER, n_features)
        coord_indices = np.random.choice(n_features, size=k, replace=False)

        grad, used_queries = estimate_gradient(
            model=model,
            x_adv=x_adv,
            x_orig=x_orig,
            coord_indices=coord_indices,
            x_min=x_min,
            x_max=x_max,
        )
        total_queries += used_queries

        # Gradient descent on attacker objective
        x_adv = x_adv - LEARNING_RATE * grad
        x_adv = np.clip(x_adv, x_min, x_max)

        current_obj = zoo_objective(model, x_adv, x_orig)
        total_queries += 1

        if current_obj < best_obj:
            best_obj = current_obj
            best_x_adv = x_adv.copy()

        adv_prob = predict_prob(model, x_adv)
        adv_pred = int(adv_prob >= THRESHOLD)
        total_queries += 1

        if adv_pred == 0:
            result.update({
                "success": True,
                "adv_prob": float(adv_prob),
                "adv_pred": int(adv_pred),
                "queries": int(total_queries),
                "l2_dist": float(np.linalg.norm(x_adv - x_orig, ord=2)),
                "linf_dist": float(np.max(np.abs(x_adv - x_orig))),
                "x_adv": x_adv.copy(),
            })
            return result

    final_prob = predict_prob(model, best_x_adv)
    final_pred = int(final_prob >= THRESHOLD)
    total_queries += 1

    result.update({
        "success": bool(final_pred == 0),
        "adv_prob": float(final_prob),
        "adv_pred": int(final_pred),
        "queries": int(total_queries),
        "l2_dist": float(np.linalg.norm(best_x_adv - x_orig, ord=2)),
        "linf_dist": float(np.max(np.abs(best_x_adv - x_orig))),
        "x_adv": best_x_adv.copy(),
    })

    return result


def attack_fraud_samples_only(model, X_train, X_test, y_test, max_attack_samples=MAX_ATTACK_SAMPLES):
    """
    Attack only fraud samples in test set.
    To keep runtime practical, attack only first N fraud samples.
    """
    fraud_idx = np.where(y_test == 1)[0]

    X_adv_full = X_test.copy()

    if len(fraud_idx) == 0:
        return X_adv_full, fraud_idx, []

    # Feature-wise bounds from training data only
    x_min = X_train.min(axis=0).astype(np.float32)
    x_max = X_train.max(axis=0).astype(np.float32)

    selected_idx = fraud_idx[:max_attack_samples]
    sample_results = []

    for i, idx in enumerate(selected_idx, start=1):
        print(f"Attacking fraud sample {i}/{len(selected_idx)}")

        res = zoo_attack_one(
            model=model,
            x=X_test[idx],
            y_true=y_test[idx],
            x_min=x_min,
            x_max=x_max,
        )

        X_adv_full[idx] = res["x_adv"]
        sample_results.append({
            "test_index": int(idx),
            "success": bool(res["success"]),
            "original_prob": float(res["original_prob"]),
            "original_pred": int(res["original_pred"]),
            "adv_prob": float(res["adv_prob"]),
            "adv_pred": int(res["adv_pred"]),
            "queries": int(res["queries"]),
            "l2_dist": float(res["l2_dist"]),
            "linf_dist": float(res["linf_dist"]),
        })

    return X_adv_full, selected_idx, sample_results


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


def run_zoo_attack():
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

    clean_metrics, clean_probs, clean_preds = evaluate_model(model, X_test, y_test, threshold=THRESHOLD)

    print("\nClean test performance:")
    for k, v in clean_metrics.items():
        if k in ["tn", "fp", "fn", "tp"]:
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")

    print("\nRunning ZOO black-box attack...")
    print(f"Max attack samples: {MAX_ATTACK_SAMPLES}")
    print(f"Max iterations per sample: {MAX_ITERS}")
    print(f"Coords per iter: {COORDS_PER_ITER}")

    X_test_adv, attacked_idx, sample_results = attack_fraud_samples_only(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        max_attack_samples=MAX_ATTACK_SAMPLES,
    )

    adv_metrics, adv_probs, adv_preds = evaluate_model(model, X_test_adv, y_test, threshold=THRESHOLD)

    asr, successes, attempts = compute_asr(clean_preds, adv_preds, y_test, attacked_idx)

    avg_queries = float(np.mean([r["queries"] for r in sample_results])) if len(sample_results) > 0 else 0.0
    avg_l2 = float(np.mean([r["l2_dist"] for r in sample_results])) if len(sample_results) > 0 else 0.0
    avg_linf = float(np.mean([r["linf_dist"] for r in sample_results])) if len(sample_results) > 0 else 0.0

    result = {
        "attack_type": "ZOO_blackbox",
        "target_model": "MLP_torch",
        "max_attack_samples": int(len(attacked_idx)),
        "max_iters": int(MAX_ITERS),
        "learning_rate": float(LEARNING_RATE),
        "fd_step": float(FD_STEP),
        "lambda_l2": float(LAMBDA_L2),
        "coords_per_iter": int(COORDS_PER_ITER),

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

        "avg_queries_per_sample": float(avg_queries),
        "avg_l2_dist": float(avg_l2),
        "avg_linf_dist": float(avg_linf),
    }

    print(f"\nASR: {asr:.4f}")
    print(f"Successful attacks: {successes}")
    print(f"Attack attempts: {attempts}")
    print(f"Average queries per sample: {avg_queries:.2f}")

    print("\nAdversarial test performance:")
    for k, v in adv_metrics.items():
        if k in ["tn", "fp", "fn", "tp"]:
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")

    os.makedirs("results/attacks", exist_ok=True)

    json_path = "results/attacks/zoo_results.json"
    csv_path = "results/attacks/zoo_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": result,
            "sample_results": sample_results,
        }, f, indent=4)

    pd.DataFrame(sample_results).to_csv(csv_path, index=False)

    print(f"\nSaved ZOO results to {json_path}")
    print(f"Saved ZOO sample results to {csv_path}")


if __name__ == "__main__":
    run_zoo_attack()
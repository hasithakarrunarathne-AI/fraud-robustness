# src/defences/evaluate_defences.py

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.defences.noise_stability_defence import (
    RANDOM_STATE,
    DEVICE,
    THRESHOLD,
    load_processed_data,
    load_mlp_model,
)
from src.defences.ensemble_disagreement_defence import load_baseline_models
from src.defences.combined_defence import apply_combined_defence


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_defence_results(results_df):
    y_true = results_df["true_label"].values.astype(int)
    y_prob = results_df["target_prob"].values.astype(np.float32)
    y_pred = results_df["target_pred"].values.astype(int)
    review_flag = results_df["review_flag"].values.astype(int)

    non_review_mask = review_flag == 0
    review_mask = review_flag == 1

    summary = {
        "total_samples": int(len(results_df)),
        "reviewed_samples": int(review_mask.sum()),
        "non_reviewed_samples": int(non_review_mask.sum()),
        "review_rate": float(review_mask.mean()),
        "fraud_review_rate": float(review_mask[y_true == 1].mean()) if np.any(y_true == 1) else 0.0,
        "nonfraud_review_rate": float(review_mask[y_true == 0].mean()) if np.any(y_true == 0) else 0.0,
    }

    if non_review_mask.sum() > 0:
        y_true_nr = y_true[non_review_mask]
        y_prob_nr = y_prob[non_review_mask]
        y_pred_nr = y_pred[non_review_mask]

        cm = confusion_matrix(y_true_nr, y_pred_nr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        summary.update({
            "precision_non_reviewed": float(precision_score(y_true_nr, y_pred_nr, zero_division=0)),
            "recall_non_reviewed": float(recall_score(y_true_nr, y_pred_nr, zero_division=0)),
            "f1_non_reviewed": float(f1_score(y_true_nr, y_pred_nr, zero_division=0)),
            "pr_auc_non_reviewed": float(average_precision_score(y_true_nr, y_prob_nr)),
            "tn_non_reviewed": int(tn),
            "fp_non_reviewed": int(fp),
            "fn_non_reviewed": int(fn),
            "tp_non_reviewed": int(tp),
        })
    else:
        summary.update({
            "precision_non_reviewed": None,
            "recall_non_reviewed": None,
            "f1_non_reviewed": None,
            "pr_auc_non_reviewed": None,
            "tn_non_reviewed": None,
            "fp_non_reviewed": None,
            "fn_non_reviewed": None,
            "tp_non_reviewed": None,
        })

    summary["noise_gate_trigger_rate"] = float(results_df["noise_gate_triggered"].mean())
    summary["disagreement_gate_trigger_rate"] = float(results_df["disagreement_gate_triggered"].mean())

    return summary


def main():
    set_seed()

    print("Using device:", DEVICE)
    
    print("Loading processed data...")

    X_train_np, X_test_np, y_train_np, y_test_np = load_processed_data()

    # Keep original feature names for sklearn baseline models
    X_test_df = pd.read_csv("data/processed/X_test.csv")

    print("X_test shape:", X_test_np.shape)
    print("Fraud count in y_test:", int(y_test_np.sum()))

    print("\nLoading models...")
    mlp_model = load_mlp_model()
    baseline_models = load_baseline_models()

    print("Applying combined defence...")

    results_df = apply_combined_defence(
        mlp_model=mlp_model,
        baseline_models=baseline_models,
        X_test_np=X_test_np,
        X_test_df=X_test_df,
        y_test=y_test_np,
        threshold=THRESHOLD,
        noise_std=0.01,
        n_perturbations=20,
        max_flip_rate=0.30,
        max_prob_std_noise=0.10,
        max_prob_std_ensemble=0.20,
        max_prob_range_ensemble=0.60,
        min_majority_margin=1,
    )

    summary = evaluate_defence_results(results_df)

    print("\n=== Defence Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    os.makedirs("results/defences", exist_ok=True)

    csv_path = "results/defences/combined_defence_results.csv"
    json_path = "results/defences/combined_defence_summary.json"

    results_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\nSaved detailed defence results to {csv_path}")
    print(f"Saved defence summary to {json_path}")


if __name__ == "__main__":
    main()
# src/defences/combined_defence.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.defences.noise_stability_defence import (
    THRESHOLD,
    predict_probabilities,
    evaluate_noise_stability_for_sample,
)
from src.defences.ensemble_disagreement_defence import (
    evaluate_ensemble_disagreement_for_sample,
)


def apply_combined_defence(
    mlp_model,
    baseline_models,
    X_test_np,
    X_test_df,
    y_test,
    threshold=THRESHOLD,
    noise_std=0.01,
    n_perturbations=20,
    max_flip_rate=0.30,
    max_prob_std_noise=0.10,
    max_prob_std_ensemble=0.20,
    max_prob_range_ensemble=0.60,
    min_majority_margin=1,
):
    results = []

    mlp_probs = predict_probabilities(mlp_model, X_test_np)
    mlp_preds = (mlp_probs >= threshold).astype(int)

    total_samples = len(X_test_np)

    for i in range(total_samples):
        if i == 0:
            print(f"Starting combined defence on {total_samples} samples...")
        elif i % 1000 == 0:
            print(f"Processed {i} / {total_samples} samples...")

        target_prob = float(mlp_probs[i])
        target_pred = int(mlp_preds[i])

        noise_row = evaluate_noise_stability_for_sample(
            model=mlp_model,
            X_sample=X_test_np[i],
            threshold=threshold,
            noise_std=noise_std,
            n_perturbations=n_perturbations,
        )

        noise_triggered = (
            noise_row["flip_rate"] > max_flip_rate
            or noise_row["perturbed_prob_std"] > max_prob_std_noise
        )

        disagreement_row = evaluate_ensemble_disagreement_for_sample(
            models=baseline_models,
            X_row=X_test_df.iloc[[i]],
            threshold=threshold,
        )

        disagreement_triggered = (
            disagreement_row["ensemble_prob_std"] > max_prob_std_ensemble
            or disagreement_row["ensemble_prob_range"] > max_prob_range_ensemble
            or disagreement_row["majority_margin"] <= min_majority_margin
        )

        review_flag = int(noise_triggered or disagreement_triggered)
        final_action = "MANUAL_REVIEW" if review_flag == 1 else "ALLOW_NORMAL_PREDICTION"

        row = {
            "sample_index": i,
            "true_label": int(y_test[i]),
            "target_prob": target_prob,
            "target_pred": target_pred,
            "noise_flip_rate": float(noise_row["flip_rate"]),
            "noise_prob_std": float(noise_row["perturbed_prob_std"]),
            "noise_gate_triggered": int(noise_triggered),
            "fraud_votes": int(disagreement_row["fraud_votes"]),
            "nonfraud_votes": int(disagreement_row["nonfraud_votes"]),
            "ensemble_prob_std": float(disagreement_row["ensemble_prob_std"]),
            "ensemble_prob_range": float(disagreement_row["ensemble_prob_range"]),
            "majority_margin": int(disagreement_row["majority_margin"]),
            "disagreement_gate_triggered": int(disagreement_triggered),
            "review_flag": int(review_flag),
            "final_action": final_action,
        }

        for key, value in disagreement_row.items():
            if key not in row and key != "sample_index":
                row[key] = value

        results.append(row)

    print(f"Finished processing {total_samples} / {total_samples} samples.")
    return pd.DataFrame(results)
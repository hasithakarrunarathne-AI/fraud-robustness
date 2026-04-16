# src/defences/run_disagreement_defence_on_attacks.py

import os
import sys
import json
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.defences.noise_stability_defence import RANDOM_STATE
from src.defences.ensemble_disagreement_defence import (
    THRESHOLD,
    load_baseline_models,
    apply_ensemble_disagreement_defence,
)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_attack_files(sample_dir="results/attacks/samples"):
    #return sorted(glob.glob(os.path.join(sample_dir, "*.npz")))
    return sorted(glob.glob(os.path.join(sample_dir, "fgsm_eps_0p1.npz")))


def summarize_defence(npz_data, review_flag):
    y_test = npz_data["y_test"].astype(int)
    attacked_idx = npz_data["attacked_idx"].astype(int)
    clean_preds = npz_data["clean_preds"].astype(int)
    adv_preds = npz_data["adv_preds"].astype(int)
    adv_probs = npz_data["adv_probs"].astype(np.float32)

    attacked_mask = np.zeros(len(y_test), dtype=bool)
    attacked_mask[attacked_idx] = True

    attempted_mask = attacked_mask & (y_test == 1) & (clean_preds == 1)
    success_mask = attempted_mask & (adv_preds == 0)
    caught_mask = success_mask & (review_flag == 1)

    successful_attacks_before = int(success_mask.sum())
    successful_attacks_caught = int(caught_mask.sum())
    catch_rate = (
        successful_attacks_caught / successful_attacks_before
        if successful_attacks_before > 0 else 0.0
    )

    adv_precision = precision_score(y_test, adv_preds, zero_division=0)
    adv_recall = recall_score(y_test, adv_preds, zero_division=0)
    adv_f1 = f1_score(y_test, adv_preds, zero_division=0)
    adv_pr_auc = average_precision_score(y_test, adv_probs)

    defended_preds = adv_preds.copy()
    defended_preds[review_flag == 1] = 1

    defended_scores = adv_probs.copy()
    defended_scores[review_flag == 1] = np.maximum(defended_scores[review_flag == 1], 1.0)

    defended_precision = precision_score(y_test, defended_preds, zero_division=0)
    defended_recall = recall_score(y_test, defended_preds, zero_division=0)
    defended_f1 = f1_score(y_test, defended_preds, zero_division=0)
    defended_pr_auc = average_precision_score(y_test, defended_scores)

    return {
        "attack_name": str(npz_data["attack_name"]),
        "epsilon": float(npz_data["epsilon"]),
        "successful_attacks_before_defence": successful_attacks_before,
        "successful_attacks_caught_by_defence": successful_attacks_caught,
        "defence_catch_rate": float(catch_rate),
        "review_rate": float(review_flag.mean()),
        "adv_precision_before": float(adv_precision),
        "adv_recall_before": float(adv_recall),
        "adv_f1_before": float(adv_f1),
        "adv_pr_auc_before": float(adv_pr_auc),
        "defended_precision_after": float(defended_precision),
        "defended_recall_after": float(defended_recall),
        "defended_f1_after": float(defended_f1),
        "defended_pr_auc_after": float(defended_pr_auc),
    }


def main():
    set_seed()

    attack_files = load_attack_files()
    if not attack_files:
        print("No attacked sample files found in results/attacks/samples")
        return

    baseline_models = load_baseline_models()
    feature_names = pd.read_csv("data/processed/X_test.csv", nrows=1).columns.tolist()

    all_results = []

    for file_path in attack_files:
        print(f"Running disagreement defence on: {file_path}")
        data = np.load(file_path, allow_pickle=True)

        X_adv = data["X_adv"].astype(np.float32)
        X_adv_df = pd.DataFrame(X_adv, columns=feature_names)

        disagreement_df = apply_ensemble_disagreement_defence(
            models=baseline_models,
            X_data=X_adv_df,
            threshold=THRESHOLD,
            max_prob_std=0.20,
            max_prob_range=0.60,
            min_majority_margin=1,
        )

        review_flag = disagreement_df["disagreement_gate_triggered"].values.astype(int)
        summary = summarize_defence(data, review_flag)
        summary["defence_name"] = "EnsembleDisagreement"

        all_results.append(summary)

    os.makedirs("results/defences/on_attacks", exist_ok=True)

    out_csv = "results/defences/on_attacks/disagreement_defence_on_attacks.csv"
    out_json = "results/defences/on_attacks/disagreement_defence_on_attacks.json"

    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved to {out_csv}")
    print(f"Saved to {out_json}")


if __name__ == "__main__":
    main()
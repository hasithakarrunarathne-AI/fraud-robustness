# src/defences/run_combined_defence_on_zoo_attacks.py

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.defences.noise_stability_defence import (
    RANDOM_STATE,
    THRESHOLD,
    load_mlp_model,
)
from src.defences.ensemble_disagreement_defence import load_baseline_models
from src.defences.combined_defence import apply_combined_defence


# ---------------------------------------------------
# CHANGE THESE BEFORE EACH RUN
# ---------------------------------------------------
VARIANT_LABEL = "with_lr"   # "with_lr" or "without_lr"
INCLUDE_LR = True             # True or False

TARGET_ATTACK_FILES = [
    "results/attacks/samples/zoo_attack_logisticregression.npz",
    "results/attacks/samples/zoo_attack_svm.npz",
    "results/attacks/samples/zoo_attack_decisiontree.npz",
    "results/attacks/samples/zoo_attack_randomforest.npz",
    "results/attacks/samples/zoo_attack_mlp_torch.npz",
]

NOISE_STD = 0.01
N_PERTURBATIONS = 20
MAX_FLIP_RATE = 0.30
MAX_PROB_STD_NOISE = 0.10
MAX_PROB_STD_ENSEMBLE = 0.20
MAX_PROB_RANGE_ENSEMBLE = 0.60
MIN_MAJORITY_MARGIN = 1


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_attack_files():
    existing_files = []
    missing_files = []

    for file_path in TARGET_ATTACK_FILES:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)

    if missing_files:
        print("These configured ZOO attack files were not found:")
        for file_path in missing_files:
            print(f" - {file_path}")

    return existing_files


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

    attack_name = str(npz_data["attack_name"])
    target_model = str(npz_data["target_model"]) if "target_model" in npz_data.files else "Unknown"
    epsilon = float(npz_data["epsilon"]) if "epsilon" in npz_data.files else -1.0

    return {
        "attack_name": attack_name,
        "target_model": target_model,
        "epsilon": epsilon,
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


def build_detailed_review_df(npz_data, combined_df):
    y_test = npz_data["y_test"].astype(int)
    attacked_idx = npz_data["attacked_idx"].astype(int)
    clean_preds = npz_data["clean_preds"].astype(int)
    adv_preds = npz_data["adv_preds"].astype(int)
    adv_probs = npz_data["adv_probs"].astype(np.float32)

    n = len(y_test)

    attacked_mask = np.zeros(n, dtype=bool)
    attacked_mask[attacked_idx] = True

    attempted_mask = attacked_mask & (y_test == 1) & (clean_preds == 1)
    success_mask = attempted_mask & (adv_preds == 0)

    detailed_df = combined_df.copy()
    detailed_df["true_label"] = y_test
    detailed_df["clean_pred"] = clean_preds
    detailed_df["adv_pred"] = adv_preds
    detailed_df["adv_prob"] = adv_probs
    detailed_df["was_attacked"] = attacked_mask.astype(int)
    detailed_df["successful_attack_before_defence"] = success_mask.astype(int)
    detailed_df["caught_by_defence"] = (
        (detailed_df["successful_attack_before_defence"] == 1)
        & (detailed_df["review_flag"] == 1)
    ).astype(int)

    return detailed_df


def main():
    set_seed()

    attack_files = load_attack_files()
    if not attack_files:
        print("No matching ZOO attacked sample files found in results/attacks/samples")
        return

    print(f"Running ZOO combined defence variant: {VARIANT_LABEL}")
    print("Attack files selected for defence evaluation:")
    for file_path in attack_files:
        print(f" - {file_path}")

    mlp_model = load_mlp_model()
    baseline_models = load_baseline_models(include_lr=INCLUDE_LR)

    if "RandomForest" in baseline_models:
        baseline_models["RandomForest"].n_jobs = 1
        print("Set RandomForest n_jobs to 1 for defence evaluation.")

    feature_names = pd.read_csv("data/processed/X_test.csv", nrows=1).columns.tolist()

    all_results = []
    all_detailed = []

    for file_path in attack_files:
        print(f"\nRunning combined defence on ZOO file: {file_path}")
        data = np.load(file_path, allow_pickle=True)

        X_adv = data["X_adv"].astype(np.float32)
        y_test = data["y_test"].astype(int)
        X_adv_df = pd.DataFrame(X_adv, columns=feature_names)

        combined_df = apply_combined_defence(
            mlp_model=mlp_model,
            baseline_models=baseline_models,
            X_test_np=X_adv,
            X_test_df=X_adv_df,
            y_test=y_test,
            threshold=THRESHOLD,
            noise_std=NOISE_STD,
            n_perturbations=N_PERTURBATIONS,
            max_flip_rate=MAX_FLIP_RATE,
            max_prob_std_noise=MAX_PROB_STD_NOISE,
            max_prob_std_ensemble=MAX_PROB_STD_ENSEMBLE,
            max_prob_range_ensemble=MAX_PROB_RANGE_ENSEMBLE,
            min_majority_margin=MIN_MAJORITY_MARGIN,
        )

        review_flag = combined_df["review_flag"].values.astype(int)

        summary = summarize_defence(data, review_flag)
        summary["defence_name"] = "Combined"
        summary["variant_label"] = VARIANT_LABEL
        all_results.append(summary)

        detailed_df = build_detailed_review_df(data, combined_df)
        detailed_df["attack_name"] = str(data["attack_name"])
        detailed_df["target_model"] = str(data["target_model"]) if "target_model" in data.files else "Unknown"
        detailed_df["epsilon"] = float(data["epsilon"]) if "epsilon" in data.files else -1.0
        detailed_df["variant_label"] = VARIANT_LABEL
        all_detailed.append(detailed_df)

        print(
            f"Done: target_model={summary['target_model']} | "
            f"caught={summary['successful_attacks_caught_by_defence']} / "
            f"{summary['successful_attacks_before_defence']} | "
            f"review_rate={summary['review_rate']:.4f}"
        )

    all_results = sorted(
        all_results,
        key=lambda x: (str(x["attack_name"]), str(x["target_model"]))
    )

    final_details_df = pd.concat(all_detailed, ignore_index=True)
    final_details_df = final_details_df.sort_values(
        by=["attack_name", "target_model", "sample_index"]
    ).reset_index(drop=True)

    os.makedirs("results/defences/on_attacks", exist_ok=True)

    out_json = f"results/defences/on_attacks/combined_defence_on_zoo_attacks_{VARIANT_LABEL}.json"
    out_csv = f"results/defences/on_attacks/combined_defence_details_zoo_{VARIANT_LABEL}.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    final_details_df.to_csv(out_csv, index=False)

    print(f"\nSaved summary to {out_json}")
    print(f"Saved details to {out_csv}")


if __name__ == "__main__":
    main()
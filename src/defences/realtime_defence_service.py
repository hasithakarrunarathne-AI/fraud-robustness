import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.model_loader import DEVICE, MLP_THRESHOLD, ENSEMBLE_THRESHOLD
from src.defences.noise_stability_defence import evaluate_noise_stability_for_sample
from src.defences.ensemble_disagreement_defence import evaluate_ensemble_disagreement_for_sample


def predict_mlp_probability(mlp_model, x_row_np: np.ndarray) -> float:
    if x_row_np.ndim == 1:
        x_row_np = x_row_np.reshape(1, -1)

    with torch.no_grad():
        x_tensor = torch.tensor(x_row_np, dtype=torch.float32).to(DEVICE)
        logits = mlp_model(x_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    return float(probs[0])


def predict_mlp_probabilities(mlp_model, x_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
        logits = mlp_model(x_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs.astype(np.float32)


def run_noise_defence_for_transaction(
    mlp_model,
    x_row_np: np.ndarray,
    threshold: float = MLP_THRESHOLD,
    noise_std: float = 0.01,
    n_perturbations: int = 20,
    max_flip_rate: float = 0.30,
    max_prob_std: float = 0.10,
) -> Dict[str, object]:
    result = evaluate_noise_stability_for_sample(
        model=mlp_model,
        X_sample=x_row_np,
        threshold=threshold,
        noise_std=noise_std,
        n_perturbations=n_perturbations,
    )

    triggered = (
        result["flip_rate"] > max_flip_rate
        or result["perturbed_prob_std"] > max_prob_std
    )

    reason_parts = []
    if result["flip_rate"] > max_flip_rate:
        reason_parts.append("high_flip_rate")
    if result["perturbed_prob_std"] > max_prob_std:
        reason_parts.append("high_probability_variation")

    return {
        "base_prob": float(result["base_prob"]),
        "base_pred": int(result["base_pred"]),
        "flip_rate": float(result["flip_rate"]),
        "perturbed_prob_std": float(result["perturbed_prob_std"]),
        "perturbed_prob_mean": float(result["perturbed_prob_mean"]),
        "perturbed_prob_min": float(result["perturbed_prob_min"]),
        "perturbed_prob_max": float(result["perturbed_prob_max"]),
        "noise_gate_triggered": bool(triggered),
        "noise_gate_reason": ", ".join(reason_parts) if reason_parts else "stable_under_noise",
    }


def run_disagreement_defence_for_transaction(
    baseline_models,
    x_row_df: pd.DataFrame,
    threshold: float = ENSEMBLE_THRESHOLD,
    max_prob_std: float = 0.20,
    max_prob_range: float = 0.60,
    min_majority_margin: int = 1,
) -> Dict[str, object]:
    result = evaluate_ensemble_disagreement_for_sample(
        models=baseline_models,
        X_row=x_row_df,
        threshold=threshold,
    )

    triggered = (
        result["ensemble_prob_std"] > max_prob_std
        or result["ensemble_prob_range"] > max_prob_range
        or result["majority_margin"] <= min_majority_margin
    )

    reason_parts = []
    if result["ensemble_prob_std"] > max_prob_std:
        reason_parts.append("high_probability_std")
    if result["ensemble_prob_range"] > max_prob_range:
        reason_parts.append("high_probability_range")
    if result["majority_margin"] <= min_majority_margin:
        reason_parts.append("low_majority_margin")

    return {
        **result,
        "disagreement_gate_triggered": bool(triggered),
        "disagreement_gate_reason": ", ".join(reason_parts) if reason_parts else "models_mostly_agree",
    }


def build_defended_outputs(
    adv_preds: np.ndarray,
    adv_probs: np.ndarray,
    review_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    defended_preds = adv_preds.copy().astype(int)
    defended_probs = adv_probs.copy().astype(np.float32)

    defended_preds[review_flags == 1] = 1
    defended_probs[review_flags == 1] = np.maximum(defended_probs[review_flags == 1], 1.0)

    return defended_preds, defended_probs
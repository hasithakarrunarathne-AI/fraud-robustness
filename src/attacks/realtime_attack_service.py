import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.model_loader import DEVICE, MLP_THRESHOLD
from src.defences.realtime_defence_service import predict_mlp_probabilities


def _safe_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_scores))


def evaluate_subset_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": _safe_pr_auc(y_true, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_mlp_on_subset(
    mlp_model,
    x_df: pd.DataFrame,
    y_true: np.ndarray,
    threshold: float = MLP_THRESHOLD,
) -> tuple[Dict[str, float], np.ndarray, np.ndarray]:
    x_np = x_df.values.astype(np.float32)
    probs = predict_mlp_probabilities(mlp_model, x_np)
    preds = (probs >= threshold).astype(int)
    metrics = evaluate_subset_metrics(y_true, probs, preds)
    return metrics, probs, preds


def fgsm_attack_single(
    mlp_model,
    x_row_np: np.ndarray,
    y_value: float,
    epsilon: float,
) -> np.ndarray:
    x_tensor = torch.tensor(x_row_np.reshape(1, -1), dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor([y_value], dtype=torch.float32).to(DEVICE)

    x_adv = x_tensor.clone().detach()
    x_adv.requires_grad = True

    logits = mlp_model(x_adv).squeeze(1)
    loss = nn.BCEWithLogitsLoss()(logits, y_tensor)

    mlp_model.zero_grad()
    loss.backward()

    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach().cpu().numpy().reshape(-1).astype(np.float32)


def pgd_attack_single(
    mlp_model,
    x_row_np: np.ndarray,
    y_value: float,
    epsilon: float,
    alpha: float = 0.01,
    num_steps: int = 20,
) -> np.ndarray:
    x_orig = torch.tensor(x_row_np.reshape(1, -1), dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor([y_value], dtype=torch.float32).to(DEVICE)

    x_adv = x_orig.clone().detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)

        logits = mlp_model(x_adv).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(logits, y_tensor)

        mlp_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = x_orig + perturbation

    return x_adv.detach().cpu().numpy().reshape(-1).astype(np.float32)


def apply_attack_to_subset(
    mlp_model,
    x_subset_df: pd.DataFrame,
    y_subset: np.ndarray,
    attack_type: str,
    epsilon: float,
    alpha: float = 0.01,
    num_steps: int = 20,
    attack_fraud_only: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    x_adv_np = x_subset_df.values.astype(np.float32).copy()

    if attack_fraud_only:
        attacked_mask = (y_subset == 1)
    else:
        attacked_mask = np.ones(len(y_subset), dtype=bool)

    for i in range(len(x_adv_np)):
        if not attacked_mask[i]:
            continue

        if attack_type == "FGSM":
            x_adv_np[i] = fgsm_attack_single(
                mlp_model=mlp_model,
                x_row_np=x_adv_np[i],
                y_value=float(y_subset[i]),
                epsilon=epsilon,
            )
        elif attack_type == "PGD":
            x_adv_np[i] = pgd_attack_single(
                mlp_model=mlp_model,
                x_row_np=x_adv_np[i],
                y_value=float(y_subset[i]),
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
            )
        else:
            raise ValueError(f"Unsupported attack_type: {attack_type}")

    x_adv_df = pd.DataFrame(x_adv_np, columns=x_subset_df.columns)
    return x_adv_df, attacked_mask.astype(int)


def compute_asr(
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    y_true: np.ndarray,
    attacked_mask: np.ndarray,
) -> Dict[str, float]:
    attacked_mask = attacked_mask.astype(bool)

    attempted_mask = attacked_mask & (y_true == 1) & (clean_preds == 1)
    success_mask = attempted_mask & (adv_preds == 0)

    attempts = int(attempted_mask.sum())
    successes = int(success_mask.sum())
    asr = (successes / attempts) if attempts > 0 else 0.0

    return {
        "attack_attempts": attempts,
        "successful_attacks": successes,
        "asr": float(asr),
        "success_mask": success_mask.astype(int),
    }
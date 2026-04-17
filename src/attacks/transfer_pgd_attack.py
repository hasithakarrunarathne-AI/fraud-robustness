import os
import json
import random
import joblib
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

THRESHOLD_MLP = 0.9
BATCH_SIZE = 1024

attack_name = "TRANSFER_PGD"

os.makedirs("results/attacks/samples", exist_ok=True)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_data(processed_dir="data/processed"):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
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


# =========================================================
# MLP source attack helpers
# =========================================================
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


def generate_mlp_pgd_adversarial_examples(model, X_test_np, y_test, epsilon, alpha, num_steps):
    fraud_idx = np.where(y_test == 1)[0]

    X_adv_full = X_test_np.copy()

    if len(fraud_idx) == 0:
        return X_adv_full, fraud_idx

    X_fraud = torch.tensor(X_test_np[fraud_idx], dtype=torch.float32)
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


# =========================================================
# Model loading
# =========================================================
def load_source_mlp(input_dim):
    model = FraudMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load("results/saved_models/mlp_torch.pth", map_location=DEVICE))
    model.eval()
    return model


def load_target_models():
    models = {}

    models["LogisticRegression"] = joblib.load("results/saved_models/LogisticRegression.pkl")
    models["SVM"] = joblib.load("results/saved_models/SVM.pkl")
    models["DecisionTree"] = joblib.load("results/saved_models/DecisionTree.pkl")
    models["RandomForest"] = joblib.load("results/saved_models/RandomForest.pkl")

    mlp_sklearn_path = "results/saved_models/MLP.pkl"
    if os.path.exists(mlp_sklearn_path):
        models["MLP_sklearn"] = joblib.load(mlp_sklearn_path)

    return models


# =========================================================
# Evaluation helpers
# =========================================================
def get_sklearn_scores(model, X_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_data)

    preds = model.predict(X_data)
    return preds.astype(float)


def evaluate_sklearn_model(model, X_data, y_data):
    preds = model.predict(X_data)
    scores = get_sklearn_scores(model, X_data)

    cm = confusion_matrix(y_data, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_data, preds),
        "precision": precision_score(y_data, preds, zero_division=0),
        "recall": recall_score(y_data, preds, zero_division=0),
        "f1": f1_score(y_data, preds, zero_division=0),
        "pr_auc": average_precision_score(y_data, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, scores, preds


def evaluate_torch_mlp(model, X_data_np, y_data, threshold=THRESHOLD_MLP):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data_np, dtype=torch.float32).to(DEVICE)
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


# =========================================================
# Main transfer runner
# =========================================================
def run_transfer_pgd():
    set_seed()

    print("Using device:", DEVICE)

    X_train, X_test, y_train, y_test = load_processed_data()

    print("X_test shape:", X_test.shape)
    print("Fraud count in y_test:", int(y_test.sum()))

    input_dim = X_train.shape[1]

    source_mlp = load_source_mlp(input_dim=input_dim)
    print("\nLoaded source MLP from results/saved_models/mlp_torch.pth")

    target_models = load_target_models()
    print("Loaded target models:", list(target_models.keys()))

    include_source_mlp_as_target = True

    epsilons = [0.1, 0.15, 0.2]
    alpha = 0.01
    num_steps = 40

    all_results = []

    X_test_np = X_test.values.astype(np.float32)
    feature_names = X_test.columns.tolist()

    for epsilon in epsilons:
        safe_eps = str(epsilon).replace(".", "p")

        print(f"\n{'=' * 70}")
        print(f"Running transfer PGD with epsilon = {epsilon}, alpha = {alpha}, steps = {num_steps}")
        print(f"{'=' * 70}")

        X_test_adv_np, attacked_idx = generate_mlp_pgd_adversarial_examples(
            model=source_mlp,
            X_test_np=X_test_np,
            y_test=y_test,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
        )

        X_test_adv = pd.DataFrame(X_test_adv_np, columns=X_test.columns)

        source_clean_metrics, source_clean_probs, source_clean_preds = evaluate_torch_mlp(
            source_mlp, X_test_np, y_test
        )
        source_adv_metrics, source_adv_probs, source_adv_preds = evaluate_torch_mlp(
            source_mlp, X_test_adv_np, y_test
        )

        np.savez_compressed(
            f"results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.npz",
            X_adv=X_test_adv_np.astype(np.float32),
            y_test=y_test.astype(np.int32),
            attacked_idx=np.array(attacked_idx, dtype=np.int32),
            clean_probs=source_clean_probs.astype(np.float32),
            clean_preds=source_clean_preds.astype(np.int32),
            adv_probs=source_adv_probs.astype(np.float32),
            adv_preds=source_adv_preds.astype(np.int32),
            attack_name=np.array(attack_name),
            epsilon=np.float32(epsilon),
        )

        adv_df = pd.DataFrame(X_test_adv_np, columns=feature_names)
        adv_df.to_csv(
            f"results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.csv",
            index=False
        )

        print(f"Saved attacked samples to results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.npz")
        print(f"Saved attacked CSV to results/attacks/samples/{attack_name.lower()}_eps_{safe_eps}.csv")

        # sklearn target models
        for target_name, target_model in target_models.items():
            clean_metrics, clean_scores, clean_preds = evaluate_sklearn_model(target_model, X_test, y_test)
            adv_metrics, adv_scores, adv_preds = evaluate_sklearn_model(target_model, X_test_adv, y_test)

            asr, successes, attempts = compute_asr(clean_preds, adv_preds, y_test, attacked_idx)

            row = {
                "attack_type": "PGD_transfer",
                "source_model": "MLP_torch",
                "target_model": target_name,
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

            all_results.append(row)

            print(f"\nTarget model: {target_name}")
            print(f"Clean recall: {clean_metrics['recall']:.4f} | Adv recall: {adv_metrics['recall']:.4f}")
            print(f"Clean F1    : {clean_metrics['f1']:.4f} | Adv F1    : {adv_metrics['f1']:.4f}")
            print(f"ASR         : {asr:.4f}")

        # source torch MLP as target too
        if include_source_mlp_as_target:
            clean_metrics, clean_scores, clean_preds = evaluate_torch_mlp(source_mlp, X_test_np, y_test)
            adv_metrics, adv_scores, adv_preds = evaluate_torch_mlp(source_mlp, X_test_adv_np, y_test)

            asr, successes, attempts = compute_asr(clean_preds, adv_preds, y_test, attacked_idx)

            row = {
                "attack_type": "PGD_transfer",
                "source_model": "MLP_torch",
                "target_model": "MLP_torch",
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

            all_results.append(row)

            print(f"\nTarget model: MLP_torch")
            print(f"Clean recall: {clean_metrics['recall']:.4f} | Adv recall: {adv_metrics['recall']:.4f}")
            print(f"Clean F1    : {clean_metrics['f1']:.4f} | Adv F1    : {adv_metrics['f1']:.4f}")
            print(f"ASR         : {asr:.4f}")

    os.makedirs("results/attacks", exist_ok=True)

    json_path = "results/attacks/transfer_pgd_results.json"
    csv_path = "results/attacks/transfer_pgd_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    pd.DataFrame(all_results).to_csv(csv_path, index=False)

    print(f"\nSaved transfer PGD results to {json_path}")
    print(f"Saved transfer PGD results to {csv_path}")


if __name__ == "__main__":
    run_transfer_pgd()
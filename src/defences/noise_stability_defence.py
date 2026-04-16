# src/defences/noise_stability_defence.py

import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.9


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


def load_mlp_model(model_path="results/saved_models/mlp_torch.pth",
                   meta_path="results/saved_models/mlp_torch_meta.pkl"):
    meta = joblib.load(meta_path)
    input_dim = meta["input_dim"]

    model = FraudMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model


def predict_probabilities(model, X_data):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def add_gaussian_noise(X_sample, noise_std=0.01, n_perturbations=20):
    repeated = np.repeat(X_sample.reshape(1, -1), n_perturbations, axis=0)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=repeated.shape).astype(np.float32)
    return repeated + noise


def evaluate_noise_stability_for_sample(
    model,
    X_sample,
    threshold=THRESHOLD,
    noise_std=0.01,
    n_perturbations=20,
):
    base_prob = float(predict_probabilities(model, X_sample.reshape(1, -1))[0])
    base_pred = int(base_prob >= threshold)

    X_perturbed = add_gaussian_noise(
        X_sample,
        noise_std=noise_std,
        n_perturbations=n_perturbations,
    )

    perturbed_probs = predict_probabilities(model, X_perturbed)
    perturbed_preds = (perturbed_probs >= threshold).astype(int)

    flip_rate = float(np.mean(perturbed_preds != base_pred))
    prob_std = float(np.std(perturbed_probs))
    prob_mean = float(np.mean(perturbed_probs))
    prob_min = float(np.min(perturbed_probs))
    prob_max = float(np.max(perturbed_probs))

    return {
        "base_prob": base_prob,
        "base_pred": base_pred,
        "perturbed_prob_mean": prob_mean,
        "perturbed_prob_std": prob_std,
        "perturbed_prob_min": prob_min,
        "perturbed_prob_max": prob_max,
        "flip_rate": flip_rate,
    }


def apply_noise_stability_defence(
    model,
    X_data,
    threshold=THRESHOLD,
    noise_std=0.01,
    n_perturbations=20,
    max_flip_rate=0.30,
    max_prob_std=0.10,
):
    results = []

    for i in range(len(X_data)):
        row = evaluate_noise_stability_for_sample(
            model=model,
            X_sample=X_data[i],
            threshold=threshold,
            noise_std=noise_std,
            n_perturbations=n_perturbations,
        )

        triggered = (
            row["flip_rate"] > max_flip_rate
            or row["perturbed_prob_std"] > max_prob_std
        )

        row["noise_gate_triggered"] = int(triggered)
        row["noise_gate_reason"] = (
            "unstable_under_noise" if triggered else "stable_under_noise"
        )
        row["sample_index"] = i

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    set_seed()

    print("Running noise stability defence check...")

    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_mlp_model()

    df = apply_noise_stability_defence(
        model=model,
        X_data=X_test[:100],
        threshold=THRESHOLD,
        noise_std=0.01,
        n_perturbations=20,
        max_flip_rate=0.30,
        max_prob_std=0.10,
    )

    os.makedirs("results/defences", exist_ok=True)
    output_path = "results/defences/noise_stability_check.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved noise stability results to {output_path}")
# src/defences/ensemble_disagreement_defence.py

import os
import joblib
import numpy as np
import pandas as pd


THRESHOLD = 0.9


def load_processed_data(processed_dir="data/processed"):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")

    return X_train, X_test, y_train, y_test


def load_baseline_models(model_dir="results/saved_models"):
    models = {
        "LogisticRegression": joblib.load(os.path.join(model_dir, "LogisticRegression.pkl")),
        "SVM": joblib.load(os.path.join(model_dir, "SVM.pkl")),
        "DecisionTree": joblib.load(os.path.join(model_dir, "DecisionTree.pkl")),
        "RandomForest": joblib.load(os.path.join(model_dir, "RandomForest.pkl")),
    }
    return models


def get_model_probability(model, X_row):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_row)[0, 1])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X_row)[0])
        return 1.0 / (1.0 + np.exp(-score))

    pred = int(model.predict(X_row)[0])
    return float(pred)


def evaluate_ensemble_disagreement_for_sample(models, X_row, threshold=THRESHOLD):
    probabilities = {}
    predictions = {}

    for model_name, model in models.items():
        prob = get_model_probability(model, X_row)
        pred = int(prob >= threshold)

        probabilities[f"{model_name}_prob"] = prob
        predictions[f"{model_name}_pred"] = pred

    prob_values = np.array(list(probabilities.values()), dtype=np.float32)
    pred_values = np.array(list(predictions.values()), dtype=np.int32)

    fraud_votes = int(pred_values.sum())
    nonfraud_votes = int(len(pred_values) - fraud_votes)
    majority_margin = abs(fraud_votes - nonfraud_votes)

    row = {
        **probabilities,
        **predictions,
        "fraud_votes": fraud_votes,
        "nonfraud_votes": nonfraud_votes,
        "ensemble_prob_mean": float(np.mean(prob_values)),
        "ensemble_prob_std": float(np.std(prob_values)),
        "ensemble_prob_min": float(np.min(prob_values)),
        "ensemble_prob_max": float(np.max(prob_values)),
        "ensemble_prob_range": float(np.max(prob_values) - np.min(prob_values)),
        "majority_margin": int(majority_margin),
    }

    return row


def apply_ensemble_disagreement_defence(
    models,
    X_data,
    threshold=THRESHOLD,
    max_prob_std=0.20,
    max_prob_range=0.60,
    min_majority_margin=1,
):
    results = []

    total_samples = len(X_data)

    for i in range(total_samples):
        if i == 0:
            print(f"Starting disagreement defence on {total_samples} samples...")
        elif i % 1000 == 0:
            print(f"Processed {i} / {total_samples} samples...")
            
        X_row = X_data.iloc[[i]]
        row = evaluate_ensemble_disagreement_for_sample(
            models=models,
            X_row=X_row,
            threshold=threshold,
        )

        triggered = (
            row["ensemble_prob_std"] > max_prob_std
            or row["ensemble_prob_range"] > max_prob_range
            or row["majority_margin"] <= min_majority_margin
        )

        row["disagreement_gate_triggered"] = int(triggered)
        row["disagreement_gate_reason"] = (
            "high_model_disagreement" if triggered else "models_mostly_agree"
        )
        row["sample_index"] = i

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running ensemble disagreement defence check...")

    X_train, X_test, y_train, y_test = load_processed_data()
    models = load_baseline_models()

    df = apply_ensemble_disagreement_defence(
        models=models,
        X_data=X_test.iloc[:100],
        threshold=THRESHOLD,
        max_prob_std=0.20,
        max_prob_range=0.60,
        min_majority_margin=1,
    )

    os.makedirs("results/defences", exist_ok=True)
    output_path = "results/defences/ensemble_disagreement_check.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved ensemble disagreement results to {output_path}")
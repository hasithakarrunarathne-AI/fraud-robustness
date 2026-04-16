# src/models/train_baselines.py

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


RANDOM_STATE = 42


def load_processed_data(processed_dir="data/processed"):
    """
    Load preprocessed train/test datasets saved earlier.
    """
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using precision, recall, F1, and PR-AUC.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_scores),
        "tn": confusion_matrix(y_test, y_pred).ravel()[0],
        "fp": confusion_matrix(y_test, y_pred).ravel()[1],
        "fn": confusion_matrix(y_test, y_pred).ravel()[2],
        "tp": confusion_matrix(y_test, y_pred).ravel()[3],
    }

    return metrics


def get_baseline_models():
    """
    Define baseline models for fraud detection.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ),

        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1 #-1
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            random_state=RANDOM_STATE
        ),
    }

    return models


def save_results(results_df, results_dir="results/baseline_metrics"):
    """
    Save baseline metrics table as CSV.
    """
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "baseline_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Saved metrics to: {output_path}")


def save_models(trained_models, model_dir="results/saved_models"):
    """
    Save trained baseline models as .pkl files.
    """
    os.makedirs(model_dir, exist_ok=True)

    for model_name, model in trained_models.items():
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)

    print(f"Saved models to: {model_dir}")


def train_baselines():
    """
    Main training pipeline for baseline models.
    """
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train fraud count:", int(y_train.sum()))
    print("y_test fraud count :", int(y_test.sum()))
    print()

    models = get_baseline_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        metrics["model"] = model_name
        results.append(metrics)
        trained_models[model_name] = model

        print(
            f"{model_name} | "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"PR-AUC: {metrics['pr_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["model", "accuracy", "precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp"]
    ].sort_values(by="pr_auc", ascending=False).reset_index(drop=True)

    print("\nFinal baseline results:")
    print(results_df)

    save_results(results_df)
    save_models(trained_models)

    return results_df, trained_models


if __name__ == "__main__":
    train_baselines()
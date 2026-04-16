# src/models/plot_baselines.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)


def load_processed_data(processed_dir="data/processed"):
    """
    Load processed test data.
    """
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")
    return X_test, y_test


def load_results(results_path="results/baseline_metrics/baseline_results.csv"):
    """
    Load saved baseline results table.
    """
    return pd.read_csv(results_path)


def load_models(model_dir="results/saved_models"):
    """
    Load all saved baseline models.
    """
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    models = {}

    for file_name in model_files:
        model_name = file_name.replace(".pkl", "")
        model_path = os.path.join(model_dir, file_name)
        models[model_name] = joblib.load(model_path)

    return models


def make_output_dirs():
    """
    Create folders for saving figures.
    """
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/figures/confusion_matrices", exist_ok=True)


def plot_model_comparison(results_df):
    """
    Plot bar chart comparing baseline model metrics.
    """
    plot_df = results_df.set_index("model")[["accuracy", "precision", "recall", "f1", "pr_auc"]]

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Baseline Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = "results/figures/baseline_model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")


def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot and save confusion matrix for each model.
    """
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        save_path = f"results/figures/confusion_matrices/{model_name}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"Saved: {save_path}")


def plot_precision_recall_curves(models, X_test, y_test):
    """
    Plot and save precision-recall curves for all baseline models.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict(X_test)

        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_scores,
            name=model_name,
            ax=ax
        )

    ax.set_title("Precision-Recall Curves for Baseline Models")
    plt.tight_layout()

    save_path = "results/figures/precision_recall_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")


def main():
    print("Loading baseline results...")
    results_df = load_results()

    print("Loading processed test data...")
    X_test, y_test = load_processed_data()

    print("Loading trained models...")
    models = load_models()

    make_output_dirs()

    print("\nGenerating comparison bar chart...")
    plot_model_comparison(results_df)

    print("\nGenerating confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test)

    print("\nGenerating precision-recall curves...")
    plot_precision_recall_curves(models, X_test, y_test)

    print("\nAll baseline graphs generated successfully.")


if __name__ == "__main__":
    main()
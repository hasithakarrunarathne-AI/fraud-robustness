import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.model_loader import load_baseline_models, load_mlp_model, load_processed_test_data
from src.attacks.realtime_attack_service import (
    apply_attack_to_subset,
    compute_asr,
    evaluate_mlp_on_subset,
    evaluate_subset_metrics,
)
from src.defences.realtime_defence_service import (
    build_defended_outputs,
    predict_mlp_probability,
    run_disagreement_defence_for_transaction,
    run_noise_defence_for_transaction,
)


st.set_page_config(
    page_title="Subset Attack and Defence Demo",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def get_test_data():
    return load_processed_test_data()


@st.cache_resource
def get_mlp():
    return load_mlp_model()


@st.cache_resource
def get_baselines(include_lr: bool):
    return load_baseline_models(include_lr=include_lr)


def get_candidate_indices(y_test: pd.Series, source_type: str):
    if source_type == "All":
        return y_test.index.to_numpy()
    if source_type == "Fraud only":
        return y_test[y_test == 1].index.to_numpy()
    return y_test[y_test == 0].index.to_numpy()


def build_subset_indices(
    y_test: pd.Series,
    strategy: str,
    subset_source: str,
    subset_size: int,
    start_pos: int = 0,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    if strategy == "Sequential":
        candidate_indices = get_candidate_indices(y_test, subset_source)
        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        end_pos = min(start_pos + subset_size, len(candidate_indices))
        return candidate_indices[start_pos:end_pos]

    if strategy == "Random":
        candidate_indices = get_candidate_indices(y_test, subset_source)
        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        actual_size = min(subset_size, len(candidate_indices))
        return np.array(rng.choice(candidate_indices, size=actual_size, replace=False), dtype=int)

    if strategy == "Balanced Demo":
        fraud_indices = y_test[y_test == 1].index.to_numpy()
        normal_indices = y_test[y_test == 0].index.to_numpy()

        if len(fraud_indices) == 0 or len(normal_indices) == 0:
            return np.array([], dtype=int)

        fraud_target = min(subset_size // 2, len(fraud_indices))
        normal_target = min(subset_size - fraud_target, len(normal_indices))

        remaining = subset_size - (fraud_target + normal_target)
        if remaining > 0:
            normal_target = min(normal_target + remaining, len(normal_indices))

        selected_fraud = rng.choice(fraud_indices, size=fraud_target, replace=False)
        selected_normal = rng.choice(normal_indices, size=normal_target, replace=False)

        combined = np.concatenate([selected_fraud, selected_normal])
        rng.shuffle(combined)
        return combined.astype(int)

    return np.array([], dtype=int)


def metrics_rows_to_df(clean_metrics, adv_metrics, defended_metrics):
    rows = []
    for stage_name, stage_metrics in [
        ("Clean", clean_metrics),
        ("Attacked", adv_metrics),
        ("Defended", defended_metrics),
    ]:
        rows.append({
            "Stage": stage_name,
            "Precision": round(float(stage_metrics["precision"]), 4),
            "Recall": round(float(stage_metrics["recall"]), 4),
            "F1": round(float(stage_metrics["f1"]), 4),
            "PR-AUC": round(float(stage_metrics["pr_auc"]), 4) if not pd.isna(stage_metrics["pr_auc"]) else np.nan,
            "TP": int(stage_metrics["tp"]),
            "FN": int(stage_metrics["fn"]),
            "FP": int(stage_metrics["fp"]),
            "TN": int(stage_metrics["tn"]),
        })
    return pd.DataFrame(rows)


def plot_metric_comparison(clean_metrics, adv_metrics, defended_metrics):
    metric_names = ["precision", "recall", "f1", "pr_auc"]
    labels = ["Precision", "Recall", "F1", "PR-AUC"]

    clean_vals = [clean_metrics[m] if not pd.isna(clean_metrics[m]) else 0.0 for m in metric_names]
    adv_vals = [adv_metrics[m] if not pd.isna(adv_metrics[m]) else 0.0 for m in metric_names]
    defended_vals = [defended_metrics[m] if not pd.isna(defended_metrics[m]) else 0.0 for m in metric_names]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, clean_vals, width, label="Clean")
    ax.bar(x, adv_vals, width, label="Attacked")
    ax.bar(x + width, defended_vals, width, label="Defended")

    ax.set_title("Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def plot_conf_matrix(y_true: np.ndarray, preds: np.ndarray, title: str):
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

def plot_defence_review_breakdown(successful_attacks_before: int, caught_by_defence: int, total_review_flags: int):
    extra_review_flags = max(0, total_review_flags - caught_by_defence)

    labels = [
        "Successful\nAttacks",
        "Caught\nSuccessful Attacks",
        "Extra Review\nFlags",
        "Total Review\nFlags",
    ]

    values = [
        int(successful_attacks_before),
        int(caught_by_defence),
        int(extra_review_flags),
        int(total_review_flags),
    ]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(labels, values)

    ax.set_title("Defence Review Breakdown")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    max_value = max(values) if values else 0
    ax.set_ylim(0, max(1, max_value * 1.25))

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_review_rate(review_rate: float):
    review_percent = review_rate * 100

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.bar(["Review Rate"], [review_percent])

    ax.set_title("Manual Review Rate")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, max(10, review_percent * 1.5))
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{review_percent:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def run_subset_defence(
    mlp_model,
    baseline_models,
    x_adv_subset: pd.DataFrame,
    defence_type: str,
    noise_std: float,
    n_perturbations: int,
    max_flip_rate: float,
    max_prob_std_noise: float,
    max_prob_std_ensemble: float,
    max_prob_range_ensemble: float,
    min_majority_margin: int,
    disagreement_rule_mode: str,
    min_fraud_votes_for_review: int,
    mlp_threshold: float = 0.9,
):
    """
    Supports both:
    - Original disagreement logic
    - Improved disagreement logic
    """
    review_flags = []
    noise_trigger_count = 0
    disagreement_trigger_count = 0

    for i in range(len(x_adv_subset)):
        x_row_df = x_adv_subset.iloc[[i]].copy()
        x_row_np = x_row_df.values.astype(np.float32).reshape(-1)

        review_flag = 0

        # -------------------------------------------------
        # Noise branch
        # -------------------------------------------------
        if defence_type in ["Noise only", "Both"]:
            noise_result = run_noise_defence_for_transaction(
                mlp_model=mlp_model,
                x_row_np=x_row_np,
                noise_std=noise_std,
                n_perturbations=n_perturbations,
                max_flip_rate=max_flip_rate,
                max_prob_std=max_prob_std_noise,
            )

            if noise_result["noise_gate_triggered"]:
                review_flag = 1
                noise_trigger_count += 1

        # -------------------------------------------------
        # Disagreement branch
        # -------------------------------------------------
        if defence_type in ["Disagreement only", "Both"]:
            disagreement_result = run_disagreement_defence_for_transaction(
                baseline_models=baseline_models,
                x_row_df=x_row_df,
                max_prob_std=max_prob_std_ensemble,
                max_prob_range=max_prob_range_ensemble,
                min_majority_margin=min_majority_margin,
            )

            if disagreement_rule_mode == "Original":
                should_review_disagreement = disagreement_result["disagreement_gate_triggered"]
            else:
                mlp_prob = predict_mlp_probability(mlp_model, x_row_np)
                mlp_pred = int(mlp_prob >= mlp_threshold)

                should_review_disagreement = (
                    mlp_pred == 0
                    and disagreement_result["fraud_votes"] >= min_fraud_votes_for_review
                    and (
                        disagreement_result["ensemble_prob_std"] > max_prob_std_ensemble
                        or disagreement_result["ensemble_prob_range"] > max_prob_range_ensemble
                        or disagreement_result["majority_margin"] <= min_majority_margin
                    )
                )

            if should_review_disagreement:
                review_flag = 1
                disagreement_trigger_count += 1

        review_flags.append(review_flag)

    review_flags = np.array(review_flags, dtype=int)

    return {
        "review_flags": review_flags,
        "noise_trigger_count": int(noise_trigger_count),
        "disagreement_trigger_count": int(disagreement_trigger_count),
    }


# -------------------------------------------------
# Load data/models
# -------------------------------------------------
X_test, y_test = get_test_data()
mlp_model = get_mlp()

st.title("Subset Attack and Defence Demo")
st.caption(
    "Multiple transactions are scored by the fraud model, adversarial attack is simulated on the subset, and defence is then applied to evaluate robustness recovery and review burden."
)

include_lr = st.sidebar.checkbox("Include Logistic Regression in disagreement defence", value=True)
baseline_models = get_baselines(include_lr=include_lr)

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.subheader("Subset Selection")

subset_strategy = st.sidebar.selectbox(
    "Subset strategy",
    ["Sequential", "Random", "Balanced Demo"],
    index=2,
)

subset_source = st.sidebar.selectbox(
    "Subset source",
    ["All", "Fraud only", "Legitimate only"],
)

subset_size = st.sidebar.selectbox("Subset size", [25, 50, 100], index=1)

start_pos = 0
random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1,
)

if subset_strategy == "Sequential":
    candidate_indices = get_candidate_indices(y_test, subset_source)
    if len(candidate_indices) == 0:
        st.error("No samples found for the selected subset source.")
        st.stop()

    max_start = max(0, len(candidate_indices) - subset_size)
    start_pos = st.sidebar.number_input(
        "Start position in filtered pool",
        min_value=0,
        max_value=max_start,
        value=0,
        step=1,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Attack Settings")

attack_type = st.sidebar.selectbox("Attack type", ["FGSM", "PGD"])
epsilon = st.sidebar.selectbox("Epsilon", [0.01, 0.03, 0.05, 0.10, 0.15, 0.20], index=3)

alpha = 0.01
num_steps = 20
if attack_type == "PGD":
    alpha = st.sidebar.selectbox("PGD alpha", [0.005, 0.01, 0.02], index=1)
    num_steps = st.sidebar.selectbox("PGD steps", [10, 20, 40], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Defence Settings")

defence_type = st.sidebar.selectbox(
    "Defence type",
    ["Noise only", "Disagreement only", "Both"],
    index=1,
)

disagreement_rule_mode = st.sidebar.selectbox(
    "Disagreement rule mode",
    ["Original", "Improved"],
    index=1,
)

min_fraud_votes_for_review = st.sidebar.selectbox(
    "Min fraud votes for disagreement review",
    [1, 2, 3, 4],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Noise Defence Settings")

noise_std = st.sidebar.slider("Noise std", min_value=0.001, max_value=0.050, value=0.010, step=0.001)
n_perturbations = st.sidebar.slider("Perturbations", min_value=5, max_value=50, value=20, step=5)
max_flip_rate = st.sidebar.slider("Max flip rate", min_value=0.05, max_value=0.80, value=0.30, step=0.05)
max_prob_std_noise = st.sidebar.slider("Max prob std (noise)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Disagreement Thresholds")

max_prob_std_ensemble = st.sidebar.slider("Max prob std (ensemble)", min_value=0.05, max_value=0.50, value=0.20, step=0.01)
max_prob_range_ensemble = st.sidebar.slider("Max prob range (ensemble)", min_value=0.10, max_value=1.00, value=0.60, step=0.05)
min_majority_margin = st.sidebar.slider("Min majority margin", min_value=0, max_value=3, value=1, step=1)

# -------------------------------------------------
# Build subset
# -------------------------------------------------
subset_indices = build_subset_indices(
    y_test=y_test,
    strategy=subset_strategy,
    subset_source=subset_source,
    subset_size=subset_size,
    start_pos=int(start_pos),
    random_state=int(random_seed),
)

if len(subset_indices) == 0:
    st.error("Could not build a valid subset with the selected settings.")
    st.stop()

x_subset = X_test.loc[subset_indices].copy().reset_index(drop=True)
y_subset = y_test.loc[subset_indices].to_numpy().astype(int)

normal_count = int((y_subset == 0).sum())
fraud_count = int((y_subset == 1).sum())

if fraud_count == 0 or normal_count == 0:
    st.warning(
        "Selected subset contains only one class. For clearer confusion matrices and attack/defence behavior, use Random or Balanced Demo."
    )

# -------------------------------------------------
# Clean evaluation
# -------------------------------------------------
clean_metrics, clean_probs, clean_preds = evaluate_mlp_on_subset(
    mlp_model=mlp_model,
    x_df=x_subset,
    y_true=y_subset,
)

# -------------------------------------------------
# Attack
# -------------------------------------------------
x_adv_subset, attacked_mask = apply_attack_to_subset(
    mlp_model=mlp_model,
    x_subset_df=x_subset,
    y_subset=y_subset,
    attack_type=attack_type,
    epsilon=float(epsilon),
    alpha=float(alpha),
    num_steps=int(num_steps),
    attack_fraud_only=True,
)

adv_metrics, adv_probs, adv_preds = evaluate_mlp_on_subset(
    mlp_model=mlp_model,
    x_df=x_adv_subset,
    y_true=y_subset,
)

attack_summary = compute_asr(
    clean_preds=clean_preds,
    adv_preds=adv_preds,
    y_true=y_subset,
    attacked_mask=attacked_mask,
)

# -------------------------------------------------
# Defence
# -------------------------------------------------
defence_summary = run_subset_defence(
    mlp_model=mlp_model,
    baseline_models=baseline_models,
    x_adv_subset=x_adv_subset,
    defence_type=defence_type,
    noise_std=noise_std,
    n_perturbations=n_perturbations,
    max_flip_rate=max_flip_rate,
    max_prob_std_noise=max_prob_std_noise,
    max_prob_std_ensemble=max_prob_std_ensemble,
    max_prob_range_ensemble=max_prob_range_ensemble,
    min_majority_margin=min_majority_margin,
    disagreement_rule_mode=disagreement_rule_mode,
    min_fraud_votes_for_review=min_fraud_votes_for_review,
)

review_flags = defence_summary["review_flags"]
defended_preds, defended_probs = build_defended_outputs(
    adv_preds=adv_preds,
    adv_probs=adv_probs,
    review_flags=review_flags,
)

defended_metrics = evaluate_subset_metrics(
    y_true=y_subset,
    probs=defended_probs,
    preds=defended_preds,
)

success_mask = attack_summary["success_mask"].astype(int)
caught_by_defence = int(((success_mask == 1) & (review_flags == 1)).sum())
review_rate = float(review_flags.mean()) if len(review_flags) > 0 else 0.0

successful_attacks_before = int(attack_summary["successful_attacks"])
catch_rate = (
    caught_by_defence / successful_attacks_before
    if successful_attacks_before > 0 else 0.0
)

# -------------------------------------------------
# Top summary
# -------------------------------------------------
top1, top2, top3, top4, top5, top6 = st.columns(6)
top1.metric("Subset Size", len(x_subset))
top2.metric("Normal Count", normal_count)
top3.metric("Fraud Count", fraud_count)
top4.metric("Attack Attempts", int(attack_summary["attack_attempts"]))
top5.metric("Successful Attacks", int(attack_summary["successful_attacks"]))
top6.metric("ASR", f"{attack_summary['asr']:.4f}")

st.markdown("---")

info1, info2, info3, info4 = st.columns(4)
info1.metric("Review Rate", f"{review_rate:.2%}")
info2.metric("Caught Successful Attacks", caught_by_defence)
info3.metric("Catch Rate", f"{catch_rate:.2%}")
info4.metric("Defence Type", defence_type)

st.markdown("---")

rule1, rule2, rule3 = st.columns(3)
rule1.write(f"**Disagreement rule mode:** {disagreement_rule_mode}")
if disagreement_rule_mode == "Improved":
    rule2.write(f"**Min fraud votes required:** {min_fraud_votes_for_review}")
rule3.write(f"**Attack:** {attack_type} (ε={epsilon})")

# -------------------------------------------------
# Metrics table + chart
# -------------------------------------------------
metrics_df = metrics_rows_to_df(clean_metrics, adv_metrics, defended_metrics)

left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("Metrics Comparison")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Metric Comparison Chart")
    plot_metric_comparison(clean_metrics, adv_metrics, defended_metrics)

# -------------------------------------------------
# Defence summary
# -------------------------------------------------
st.markdown("---")
st.subheader("Defence Trigger Summary")

summary_left, summary_right = st.columns([1.25, 1.0])

with summary_left:
    chart1, chart2 = st.columns(2)

    with chart1:
        plot_defence_review_breakdown(
            successful_attacks_before=successful_attacks_before,
            caught_by_defence=caught_by_defence,
            total_review_flags=int(review_flags.sum()),
        )

    with chart2:
        plot_review_rate(review_rate=review_rate)

with summary_right:
    st.subheader("Interpretation")
    st.write("- **Clean** shows normal fraud detection performance.")
    st.write("- **Attacked** shows what happens after adversarial perturbation.")
    st.write("- **Defended** shows performance after review-based defence is applied.")
    st.write("- A useful defence should reduce missed fraud without causing excessive review burden.")

    extra_review_flags = max(0, int(review_flags.sum()) - caught_by_defence)

    st.info(
        f"Successful attacks: {successful_attacks_before} | "
        f"Caught successful attacks: {caught_by_defence} | "
        f"Extra review flags: {extra_review_flags} | "
        f"Total review flags: {int(review_flags.sum())} | "
        f"Review rate: {review_rate:.2%}"
    )


# -------------------------------------------------
# Three confusion matrices
# -------------------------------------------------
st.markdown("---")
st.subheader("Confusion Matrices")

cm1, cm2, cm3 = st.columns(3)

with cm1:
    plot_conf_matrix(y_subset, clean_preds, "Before Attack")

with cm2:
    plot_conf_matrix(y_subset, adv_preds, "After Attack")

with cm3:
    plot_conf_matrix(y_subset, defended_preds, "After Defence")

# -------------------------------------------------
# Preview
# -------------------------------------------------
st.markdown("---")
with st.expander("Subset Preview"):
    preview_df = x_subset.copy()
    preview_df["true_label"] = y_subset
    preview_df["clean_prob"] = clean_probs
    preview_df["clean_pred"] = clean_preds
    preview_df["adv_prob"] = adv_probs
    preview_df["adv_pred"] = adv_preds
    preview_df["review_flag"] = review_flags
    preview_df["defended_pred"] = defended_preds
    preview_df["successful_attack"] = success_mask

    show_cols = [
        "true_label",
        "clean_prob",
        "clean_pred",
        "adv_prob",
        "adv_pred",
        "review_flag",
        "defended_pred",
        "successful_attack",
    ]
    st.dataframe(preview_df[show_cols].head(30), use_container_width=True, height=450)

if len(np.unique(y_subset)) < 2:
    st.warning(
        "This subset has only one class, so some evaluation views are less meaningful. Use 'All' with Random or Balanced Demo for a mixed subset."
    )
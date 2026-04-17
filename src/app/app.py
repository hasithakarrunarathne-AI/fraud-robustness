import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# ---------------------------------------------------
# Project path setup
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_mlp_torch_v2 import FraudMLP, load_processed_data, evaluate_model, DEVICE


# ---------------------------------------------------
# Paths
# ---------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "results" / "saved_models" / "mlp_torch.pth"

FGSM_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "fgsm_results.json"
PGD_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "pgd_results.json"
TRANSFER_FGSM_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "transfer_fgsm_results.json"
TRANSFER_PGD_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "transfer_pgd_results.json"
ZOO_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "zoo_results.json"
ZOO_SAMPLE_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "zoo_sample_results.csv"

NOISE_DEFENCE_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "noise_defence_on_attacks.json"

DISAGREEMENT_WITH_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_on_attacks_with_lr.json"
DISAGREEMENT_WITHOUT_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_on_attacks_without_lr.json"
DISAGREEMENT_WITH_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_details_with_lr.csv"
DISAGREEMENT_WITHOUT_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_details_without_lr.csv"

COMBINED_WITH_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_on_attacks_with_lr.json"
COMBINED_WITHOUT_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_on_attacks_without_lr.json"
COMBINED_WITH_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_details_with_lr.csv"
COMBINED_WITHOUT_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_details_without_lr.csv"

NOISE_ZOO_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "noise_defence_on_zoo_attacks.json"
NOISE_ZOO_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "noise_defence_details_zoo.csv"

DISAGREEMENT_ZOO_WITH_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_on_zoo_attacks_with_lr.json"
DISAGREEMENT_ZOO_WITHOUT_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_on_zoo_attacks_without_lr.json"
DISAGREEMENT_ZOO_WITH_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_details_zoo_with_lr.csv"
DISAGREEMENT_ZOO_WITHOUT_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_details_zoo_without_lr.csv"

COMBINED_ZOO_WITH_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_on_zoo_attacks_with_lr.json"
COMBINED_ZOO_WITHOUT_LR_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_on_zoo_attacks_without_lr.json"
COMBINED_ZOO_WITH_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_details_zoo_with_lr.csv"
COMBINED_ZOO_WITHOUT_LR_DETAILS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_details_zoo_without_lr.csv"

THRESHOLD = 0.9


# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Robustness Demo",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
@st.cache_resource
def load_mlp_model_and_data():
    X_train, X_test, y_train, y_test = load_processed_data(
        processed_dir=str(PROJECT_ROOT / "data" / "processed")
    )
    input_dim = X_train.shape[1]

    model = FraudMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model, X_train, X_test, y_train, y_test


@st.cache_data
def get_clean_mlp_stats():
    model, X_train, X_test, y_train, y_test = load_mlp_model_and_data()
    metrics = evaluate_model(model, X_test, y_test)
    return metrics


@st.cache_data
def get_total_test_samples():
    _, X_test, _, y_test = load_processed_data(
        processed_dir=str(PROJECT_ROOT / "data" / "processed")
    )
    _ = y_test
    return len(X_test)


@st.cache_data
def load_attack_results(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    return pd.DataFrame(results)


def normalize_attack_labels(df):
    if df is None or df.empty or "attack_name" not in df.columns:
        return df

    label_map = {
        "FGSM": "FGSM",
        "PGD": "PGD",
        "TRANSFER_FGSM": "Transfer FGSM",
        "TRANSFER_PGD": "Transfer PGD",
        "ZOO": "ZOO",
    }

    df = df.copy()
    df["attack_name"] = df["attack_name"].replace(label_map)
    return df


@st.cache_data
def load_defence_results(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    df = normalize_attack_labels(df)
    return df


@st.cache_data
def load_csv_data(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    df = normalize_attack_labels(df)
    return df


@st.cache_data
def load_zoo_summary_results(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if isinstance(results, dict):
        summary = results.get("summary", {})
        if summary:
            return pd.DataFrame([summary])
        return None

    if isinstance(results, list):
        return pd.DataFrame(results)

    return None


def plot_line(df, x_col, y_col, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


def plot_bar(df, x_col, y_col, title, y_label):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)


def plot_grouped_bar(categories, clean_values, adv_values, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 3.8))

    x = range(len(categories))
    width = 0.35

    ax.bar([i - width / 2 for i in x], clean_values, width=width, label="Clean")
    ax.bar([i + width / 2 for i in x], adv_values, width=width, label="Adversarial")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Metric", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def plot_defence_catch_chart(row, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    categories = ["Successful Attacks", "Caught by Defence"]
    values = [
        int(row["successful_attacks_before_defence"]),
        int(row["successful_attacks_caught_by_defence"]),
    ]

    ax.bar(categories, values)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def plot_defence_review_chart(row, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    categories = ["Review Rate", "Catch Rate"]
    values = [
        float(row["review_rate"]),
        float(row["defence_catch_rate"]),
    ]

    ax.bar(categories, values)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Rate", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def plot_variant_rate_chart(with_row, without_row, title):
    fig, ax = plt.subplots(figsize=(7, 4))

    categories = ["Catch Rate", "Review Rate"]
    with_vals = [
        float(with_row["defence_catch_rate"]),
        float(with_row["review_rate"]),
    ]
    without_vals = [
        float(without_row["defence_catch_rate"]),
        float(without_row["review_rate"]),
    ]

    x = range(len(categories))
    width = 0.35

    ax.bar([i - width / 2 for i in x], with_vals, width=width, label="With LR")
    ax.bar([i + width / 2 for i in x], without_vals, width=width, label="Without LR")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def filter_by_attack_and_epsilon(df, selected_attack, selected_epsilon):
    if df is None or df.empty:
        return df

    return df[
        (df["attack_name"] == selected_attack)
        & (np.isclose(df["epsilon"].astype(float), float(selected_epsilon), atol=1e-8))
    ].copy()


def format_delta(clean_value, adv_value):
    delta = adv_value - clean_value
    return f"{adv_value:.4f}", f"{delta:.4f}"


def get_step3_attack_options(df):
    if df is None or df.empty or "attack_name" not in df.columns:
        return []

    allowed_attacks = ["FGSM", "PGD"]
    available_attacks = df["attack_name"].dropna().unique().tolist()

    return [attack for attack in allowed_attacks if attack in available_attacks]


# ---------------------------------------------------
# White-box attack section renderer
# ---------------------------------------------------
def render_attack_section(df, clean_metrics, attack_name):
    if df is None or df.empty:
        st.warning(f"{attack_name} results file not found. Run the {attack_name.lower()} script first.")
        return

    st.success(f"{attack_name} results loaded successfully.")

    max_row = df.loc[df["epsilon"].idxmax()]

    st.subheader(f"Key Comparison at Strongest {attack_name} Setting")

    comp1, comp2, comp3, comp4 = st.columns(4)

    recall_value, recall_delta = format_delta(clean_metrics["recall"], max_row["adv_recall"])
    f1_value, f1_delta = format_delta(clean_metrics["f1"], max_row["adv_f1"])
    prauc_value, prauc_delta = format_delta(clean_metrics["pr_auc"], max_row["adv_pr_auc"])

    comp1.metric("Recall", recall_value, recall_delta)
    comp2.metric("F1", f1_value, f1_delta)
    comp3.metric("PR-AUC", prauc_value, prauc_delta)
    comp4.metric("ASR", f"{max_row['asr']:.4f}")

    summary_text = (
        f"At the highest tested epsilon ({max_row['epsilon']}), the model's recall, F1, and PR-AUC "
        f"decrease compared with clean performance, while ASR rises to {max_row['asr']:.4f}."
    )

    if "alpha" in df.columns and "num_steps" in df.columns:
        summary_text += f" This run used alpha = {max_row['alpha']} and num_steps = {int(max_row['num_steps'])}."

    st.info(summary_text)

    st.subheader(f"{attack_name} Result Table")

    base_cols = [
        "epsilon",
        "adv_precision",
        "adv_recall",
        "adv_f1",
        "adv_pr_auc",
        "adv_tp",
        "adv_fn",
        "successful_attacks",
        "attack_attempts",
        "asr",
    ]

    extra_cols = []
    if "alpha" in df.columns:
        extra_cols.append("alpha")
    if "num_steps" in df.columns:
        extra_cols.append("num_steps")

    show_cols = ["epsilon"] + extra_cols + [c for c in base_cols if c != "epsilon"]

    pretty_df = df[show_cols].copy()
    st.dataframe(pretty_df, use_container_width=True, height=320)

    st.subheader(f"{attack_name} Trend Charts")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        plot_line(df, "epsilon", "adv_recall", f"{attack_name}: Epsilon vs Recall", "Recall")
        plot_line(df, "epsilon", "adv_pr_auc", f"{attack_name}: Epsilon vs PR-AUC", "PR-AUC")

    with chart_col2:
        plot_line(df, "epsilon", "adv_f1", f"{attack_name}: Epsilon vs F1", "F1")
        plot_line(df, "epsilon", "asr", f"{attack_name}: Epsilon vs ASR", "ASR")

    st.subheader("Research Interpretation")

    extra_line = ""
    if "alpha" in df.columns and "num_steps" in df.columns:
        extra_line = (
            f"\n**Attack parameters at strongest setting:** "
            f"alpha = **{max_row['alpha']}**, num_steps = **{int(max_row['num_steps'])}**"
        )

    st.markdown(
        f"""
**Observation:**  
As epsilon increases, adversarial impact becomes visible through reduced recall, reduced F1,
reduced PR-AUC, and increased false negatives.

**Strongest tested {attack_name} setting ({max_row['epsilon']}):**
- Adversarial Recall: **{max_row['adv_recall']:.4f}**
- Adversarial F1: **{max_row['adv_f1']:.4f}**
- Adversarial PR-AUC: **{max_row['adv_pr_auc']:.4f}**
- Successful Attacks: **{int(max_row['successful_attacks'])}**
- Attack Success Rate (ASR): **{max_row['asr']:.4f}**
{extra_line}

**Conclusion:**  
The tuned MLP shows measurable robustness degradation under {attack_name} attack, with stronger
attack settings causing more fraudulent transactions to escape detection.
"""
    )


# ---------------------------------------------------
# Transfer section renderer
# ---------------------------------------------------
def render_transfer_section(df, attack_label):
    if df is None or df.empty:
        st.warning(f"{attack_label} results file not found. Run the corresponding script first.")
        return

    st.success(f"{attack_label} results loaded successfully.")

    st.subheader(f"Filter {attack_label} Results")

    col1, col2 = st.columns(2)

    with col1:
        epsilon_options = sorted(df["epsilon"].unique().tolist())
        selected_epsilon = st.selectbox(
            "Select epsilon",
            epsilon_options,
            index=len(epsilon_options) - 1,
            key=f"{attack_label}_epsilon_select"
        )

    with col2:
        model_options = ["All"] + sorted(df["target_model"].unique().tolist())
        selected_model = st.selectbox(
            "Select target model",
            model_options,
            key=f"{attack_label}_model_select"
        )

    filtered_df = df[df["epsilon"] == selected_epsilon].copy()

    if selected_model != "All":
        filtered_df = filtered_df[filtered_df["target_model"] == selected_model].copy()

    strongest_row = filtered_df.loc[filtered_df["asr"].idxmax()].copy()
    strongest_row["recall_drop"] = strongest_row["adv_recall"] - strongest_row["clean_recall"]

    st.subheader("Key Transfer Finding")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strongest Target", strongest_row["target_model"])
    c2.metric("Highest ASR", f"{strongest_row['asr']:.4f}")
    c3.metric("Successful Attacks", int(strongest_row["successful_attacks"]))
    c4.metric(
        "Adversarial Recall",
        f"{strongest_row['adv_recall']:.4f}",
        f"{strongest_row['recall_drop']:.4f}"
    )

    summary_text = (
        f"At epsilon {selected_epsilon}, the strongest transfer effect is observed on "
        f"{strongest_row['target_model']} with ASR = {strongest_row['asr']:.4f}."
    )

    if "alpha" in filtered_df.columns and "num_steps" in filtered_df.columns:
        summary_text += (
            f" This run used alpha = {strongest_row['alpha']} and "
            f"num_steps = {int(strongest_row['num_steps'])}."
        )

    st.info(summary_text)

    st.subheader(f"{attack_label} Result Table")

    show_cols = [
        "target_model",
        "epsilon",
        "clean_recall",
        "adv_recall",
        "clean_f1",
        "adv_f1",
        "clean_pr_auc",
        "adv_pr_auc",
        "successful_attacks",
        "attack_attempts",
        "asr",
    ]

    extra_cols = []
    if "alpha" in filtered_df.columns:
        extra_cols.append("alpha")
    if "num_steps" in filtered_df.columns:
        extra_cols.append("num_steps")

    final_cols = ["target_model", "epsilon"] + extra_cols + [
        c for c in show_cols if c not in ["target_model", "epsilon"]
    ]

    st.dataframe(filtered_df[final_cols], use_container_width=True, height=350)

    if selected_model == "All":
        filtered_df["recall_drop"] = filtered_df["adv_recall"] - filtered_df["clean_recall"]

        st.subheader("Transfer Comparison Across Target Models")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            plot_bar(
                filtered_df,
                "target_model",
                "asr",
                f"{attack_label} ASR at epsilon={selected_epsilon}",
                "ASR",
            )
            plot_bar(
                filtered_df,
                "target_model",
                "adv_recall",
                f"{attack_label} Adversarial Recall at epsilon={selected_epsilon}",
                "Adversarial Recall",
            )

        with chart_col2:
            plot_bar(
                filtered_df,
                "target_model",
                "recall_drop",
                f"{attack_label} Recall Drop at epsilon={selected_epsilon}",
                "Recall Drop",
            )
            plot_bar(
                filtered_df,
                "target_model",
                "successful_attacks",
                f"{attack_label} Successful Attacks at epsilon={selected_epsilon}",
                "Successful Attacks",
            )

        st.subheader("Transfer Attack Interpretation")

        extra_line = ""
        if "alpha" in filtered_df.columns and "num_steps" in filtered_df.columns:
            extra_line = (
                f"\n**Attack parameters:** alpha = **{strongest_row['alpha']}**, "
                f"num_steps = **{int(strongest_row['num_steps'])}**"
            )

        st.markdown(
            f"""
**Observation:**  
The adversarial examples crafted on the PyTorch MLP do not affect all target models equally.
Some models show little or no transfer effect, while others show clearer degradation.

**Strongest transfer target at epsilon {selected_epsilon}:**
- Target Model: **{strongest_row['target_model']}**
- Adversarial Recall: **{strongest_row['adv_recall']:.4f}**
- Adversarial F1: **{strongest_row['adv_f1']:.4f}**
- Successful Attacks: **{int(strongest_row['successful_attacks'])}**
- ASR: **{strongest_row['asr']:.4f}**
{extra_line}

**Conclusion:**  
Transferability is model-dependent. This means adversarial samples crafted on one fraud model
can generalize unevenly across different fraud classifier families.
"""
        )
    else:
        row = filtered_df.iloc[0]

        st.subheader("Selected Target Model Interpretation")

        summary_text = (
            f"For target model {row['target_model']} at epsilon {row['epsilon']}, "
            f"ASR = {row['asr']:.4f}, adversarial recall = {row['adv_recall']:.4f}, "
            f"and adversarial F1 = {row['adv_f1']:.4f}."
        )

        if "alpha" in filtered_df.columns and "num_steps" in filtered_df.columns:
            summary_text += f" This run used alpha = {row['alpha']} and num_steps = {int(row['num_steps'])}."

        st.info(summary_text)


# ---------------------------------------------------
# ZOO section renderer
# ---------------------------------------------------
def render_zoo_section(summary_df, sample_df):
    if summary_df is None or summary_df.empty:
        st.warning("ZOO results file not found. Run the zoo_attack.py script first.")
        return

    st.success("ZOO results loaded successfully.")

    model_options = sorted(summary_df["target_model"].dropna().unique().tolist())
    selected_model = st.selectbox(
        "Select target model for ZOO",
        model_options,
        key="zoo_target_model_select"
    )

    selected_row = summary_df[summary_df["target_model"] == selected_model].iloc[0]

    st.subheader("Key Black-Box Finding")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Model", selected_row["target_model"])
    c2.metric("Successful Attacks", int(selected_row["successful_attacks"]))
    c3.metric("Attack Attempts", int(selected_row["attack_attempts"]))
    c4.metric("ASR", f"{selected_row['asr']:.4f}")

    st.info(
        f"ZOO is a target-model-specific black-box attack. For {selected_model}, "
        f"this run attacked {int(selected_row['attacked_fraud_samples'])} fraud samples with "
        f"max_iters = {int(selected_row['max_iters'])}, learning_rate = {selected_row['learning_rate']}, "
        f"and coords_per_iter = {int(selected_row['coords_per_iter'])}."
    )

    st.subheader("Selected Model Summary")

    summary_view = pd.DataFrame([{
        "Target Model": selected_row["target_model"],
        "Successful Attacks": int(selected_row["successful_attacks"]),
        "Attack Attempts": int(selected_row["attack_attempts"]),
        "ASR": round(float(selected_row["asr"]), 4),
        "Avg Queries / Sample": round(float(selected_row["avg_queries_per_sample"]), 2),
        "Avg L2 Dist": round(float(selected_row["avg_l2_dist"]), 4),
        "Avg Linf Dist": round(float(selected_row["avg_linf_dist"]), 4),
        "Adv Recall": round(float(selected_row["adv_recall"]), 4),
        "Adv F1": round(float(selected_row["adv_f1"]), 4),
        "Adv PR-AUC": round(float(selected_row["adv_pr_auc"]), 4),
    }])
    st.dataframe(summary_view, use_container_width=True, hide_index=True)

    st.subheader("ZOO Comparison Across Target Models")
    compare_cols = [
        "target_model",
        "successful_attacks",
        "attack_attempts",
        "asr",
        "avg_queries_per_sample",
        "adv_recall",
        "adv_f1",
        "adv_pr_auc",
    ]
    st.dataframe(summary_df[compare_cols], use_container_width=True, height=280)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_bar(
            summary_df,
            "target_model",
            "asr",
            "ZOO: ASR by Target Model",
            "ASR",
        )
        plot_bar(
            summary_df,
            "target_model",
            "successful_attacks",
            "ZOO: Successful Attacks by Target Model",
            "Successful Attacks",
        )

    with chart_col2:
        plot_grouped_bar(
            categories=["Recall", "F1", "PR-AUC"],
            clean_values=[
                selected_row["clean_recall"],
                selected_row["clean_f1"],
                selected_row["clean_pr_auc"],
            ],
            adv_values=[
                selected_row["adv_recall"],
                selected_row["adv_f1"],
                selected_row["adv_pr_auc"],
            ],
            title=f"ZOO: Clean vs Adversarial Metrics ({selected_model})",
            y_label="Score",
        )

    if sample_df is not None and not sample_df.empty:
        sample_df = sample_df.copy()
        model_sample_df = sample_df[sample_df["target_model"] == selected_model].copy()

        if not model_sample_df.empty:
            model_sample_df["prob_drop"] = (
                model_sample_df["original_prob"] - model_sample_df["adv_prob"]
            ).round(4)

            st.subheader("Selected Model Sample Results")

            success_df = model_sample_df[model_sample_df["success"] == True].copy()

            if not success_df.empty:
                show_cols = [
                    "test_index",
                    "success",
                    "original_prob",
                    "adv_prob",
                    "prob_drop",
                    "queries",
                    "l2_dist",
                    "linf_dist",
                ]
                st.dataframe(success_df[show_cols], use_container_width=True, height=220)
            else:
                st.info("No successful ZOO attack samples were recorded for this model.")

            top_drop_df = model_sample_df.sort_values("prob_drop", ascending=False).head(10).copy()
            show_cols = [
                "test_index",
                "success",
                "original_prob",
                "adv_prob",
                "prob_drop",
                "queries",
                "l2_dist",
                "linf_dist",
            ]
            st.subheader("Top Probability Drops")
            st.dataframe(top_drop_df[show_cols], use_container_width=True, height=260)

    st.subheader("Research Interpretation")
    st.markdown(
        f"""
**Observation:**  
ZOO is now evaluated as a target-model-specific black-box attack. This means each model has its own
attacked dataset, own successful attack count, and own query cost.

**Selected target model:**
- Target Model: **{selected_row['target_model']}**
- Successful Attacks: **{int(selected_row['successful_attacks'])}**
- Attack Attempts: **{int(selected_row['attack_attempts'])}**
- ASR: **{selected_row['asr']:.4f}**
- Avg Queries per Sample: **{selected_row['avg_queries_per_sample']:.2f}**

**Conclusion:**  
Under this query-based black-box setting, robustness is model-dependent. Some model families are harder
to fool than others under the same ZOO configuration.
"""
    )


# ---------------------------------------------------
# Generic defence section renderer
# ---------------------------------------------------
def render_defence_section(df, defence_label):
    if df is None or df.empty:
        st.warning(f"{defence_label} results file not found. Run the corresponding defence script first.")
        return

    st.success(f"{defence_label} results loaded successfully.")

    attack_options = get_step3_attack_options(df)
    if not attack_options:
        st.warning("No FGSM or PGD defence results found for Step 3.")
        return

    selected_attack = st.selectbox(
        f"Select attack type for {defence_label}",
        attack_options,
        key=f"{defence_label}_attack_select"
    )

    filtered_df = df[df["attack_name"] == selected_attack].copy()

    if "epsilon" in filtered_df.columns and not filtered_df["epsilon"].isna().all():
        epsilon_options = sorted(filtered_df["epsilon"].unique().tolist())
        selected_epsilon = st.selectbox(
            f"Select epsilon for {defence_label}",
            epsilon_options,
            index=len(epsilon_options) - 1,
            key=f"{defence_label}_epsilon_select"
        )
        filtered_df = filter_by_attack_and_epsilon(filtered_df, selected_attack, selected_epsilon)

    if filtered_df.empty:
        st.warning("No matching defence result found for the selected filters.")
        return

    row = filtered_df.iloc[0]
    total_samples = get_total_test_samples()
    review_count = int(round(float(row["review_rate"]) * total_samples))

    st.subheader(f"{defence_label} Key Result")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Successful Attacks Before", int(row["successful_attacks_before_defence"]))
    c2.metric("Caught by Defence", int(row["successful_attacks_caught_by_defence"]))
    c3.metric("Catch Rate", f"{row['defence_catch_rate']:.2%}")
    c4.metric("Review Rate", f"{row['review_rate']:.2%}")
    c5.metric("Manual Review Count", review_count)

    st.info(
        f"For {row['attack_name']} at epsilon {row['epsilon']:.3f}, the {defence_label} "
        f"caught {int(row['successful_attacks_caught_by_defence'])} out of "
        f"{int(row['successful_attacks_before_defence'])} successful attacks, while reviewing "
        f"{review_count} transactions ({row['review_rate']:.2%})."
    )

    st.subheader(f"{defence_label} Summary Table")

    show_cols = [
        "attack_name",
        "epsilon",
        "successful_attacks_before_defence",
        "successful_attacks_caught_by_defence",
        "defence_catch_rate",
        "review_rate",
        "adv_precision_before",
        "adv_recall_before",
        "adv_f1_before",
        "adv_pr_auc_before",
        "defended_precision_after",
        "defended_recall_after",
        "defended_f1_after",
        "defended_pr_auc_after",
    ]

    available_cols = [c for c in show_cols if c in filtered_df.columns]
    pretty_df = filtered_df[available_cols].copy()
    st.dataframe(pretty_df, use_container_width=True, height=220)

    st.subheader(f"{defence_label} Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_defence_catch_chart(row, f"{defence_label}: Successful Attacks vs Caught")

    with chart_col2:
        plot_defence_review_chart(row, f"{defence_label}: Catch Rate vs Review Rate")

    st.subheader("Research Interpretation")

    if int(row["successful_attacks_before_defence"]) == 0:
        conclusion_text = (
            "At this attack setting, there were no successful attacks before defence, "
            "so defence catch-rate interpretation is limited."
        )
    elif int(row["successful_attacks_caught_by_defence"]) == 0:
        conclusion_text = (
            f"The {defence_label} did not intercept successful attack cases at this setting, "
            f"so it provided no operational recovery here."
        )
    elif int(row["successful_attacks_caught_by_defence"]) == int(row["successful_attacks_before_defence"]):
        conclusion_text = (
            f"The {defence_label} intercepted all successful evasion cases at this setting, "
            f"while keeping the review rate relatively low."
        )
    else:
        conclusion_text = (
            f"The {defence_label} intercepted part of the successful evasion cases, "
            f"showing partial operational protection."
        )

    st.markdown(
        f"""
**Observation:**  
This section evaluates whether the defence mechanism can intercept successful adversarial evasions
and route suspicious transactions to manual review.

**Selected result:**
- Attack: **{row['attack_name']}**
- Epsilon: **{row['epsilon']:.3f}**
- Successful Attacks Before Defence: **{int(row['successful_attacks_before_defence'])}**
- Successful Attacks Caught by Defence: **{int(row['successful_attacks_caught_by_defence'])}**
- Defence Catch Rate: **{row['defence_catch_rate']:.2%}**
- Review Rate: **{row['review_rate']:.2%}**
- Manual Review Count: **{review_count}**

**Interpretation note:**  
The after-defence precision, F1, and PR-AUC values are shown for completeness, but they should be
interpreted carefully because this is a manual-review / abstain-style defence rather than a standard
binary classifier.

**Conclusion:**  
{conclusion_text}
"""
    )


# ---------------------------------------------------
# ZOO noise section renderer
# ---------------------------------------------------
def render_zoo_noise_section(zoo_df, zoo_details):
    if zoo_df is None or zoo_df.empty:
        st.warning("ZOO noise-defence results file not found.")
        return

    st.success("ZOO noise-defence results loaded successfully.")

    st.info(
        "ZOO is a target-model-specific black-box attack. Therefore, ZOO defence evaluation is shown "
        "per target model rather than per epsilon."
    )

    model_options = sorted(zoo_df["target_model"].dropna().unique().tolist())
    selected_model = st.selectbox(
        "Select target model for ZOO noise defence",
        model_options,
        key="zoo_noise_target_model_select"
    )

    row = zoo_df[zoo_df["target_model"] == selected_model].iloc[0]
    total_samples = get_total_test_samples()
    review_count = int(round(float(row["review_rate"]) * total_samples))

    st.subheader("ZOO Noise Stability Key Result")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Target Model", row["target_model"])
    c2.metric("Successful Attacks Before", int(row["successful_attacks_before_defence"]))
    c3.metric("Caught by Defence", int(row["successful_attacks_caught_by_defence"]))
    c4.metric("Catch Rate", f"{row['defence_catch_rate']:.2%}")
    c5.metric("Review Rate", f"{row['review_rate']:.2%}")

    st.info(
        f"For ZOO against {row['target_model']}, the noise stability defence caught "
        f"{int(row['successful_attacks_caught_by_defence'])} out of "
        f"{int(row['successful_attacks_before_defence'])} successful attacks while reviewing "
        f"{review_count} transactions."
    )

    show_cols = [
        "attack_name",
        "target_model",
        "successful_attacks_before_defence",
        "successful_attacks_caught_by_defence",
        "defence_catch_rate",
        "review_rate",
        "adv_precision_before",
        "adv_recall_before",
        "adv_f1_before",
        "adv_pr_auc_before",
        "defended_precision_after",
        "defended_recall_after",
        "defended_f1_after",
        "defended_pr_auc_after",
    ]
    st.dataframe(pd.DataFrame([row])[show_cols], use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        plot_defence_catch_chart(row, f"ZOO Noise Defence: {selected_model}")
    with chart_col2:
        plot_defence_review_chart(row, f"ZOO Noise Defence Rates: {selected_model}")

    if zoo_details is not None and not zoo_details.empty:
        detail_df = zoo_details[zoo_details["target_model"] == selected_model].copy()

        sample_view_choice = st.radio(
            "Select ZOO noise sample view",
            ["Caught attack samples", "All flagged review samples"],
            horizontal=True,
            key="zoo_noise_sample_view_choice"
        )

        if sample_view_choice == "Caught attack samples":
            detail_df = detail_df[detail_df["caught_by_defence"] == 1].copy()
        else:
            detail_df = detail_df[detail_df["noise_gate_triggered"] == 1].copy()

        if not detail_df.empty:
            show_cols = [
                "sample_index",
                "true_label",
                "clean_pred",
                "adv_pred",
                "adv_prob",
                "was_attacked",
                "successful_attack_before_defence",
                "noise_gate_triggered",
                "caught_by_defence",
                "flip_rate",
                "perturbed_prob_std",
                "perturbed_prob_mean",
                "perturbed_prob_min",
                "perturbed_prob_max",
            ]
            available_cols = [c for c in show_cols if c in detail_df.columns]
            st.subheader("Exact Sample View")
            st.dataframe(detail_df[available_cols], use_container_width=True, height=300)

    st.subheader("Research Interpretation")
    st.markdown(
        f"""
**Observation:**  
This section measures whether the noise-stability gate can intercept successful ZOO black-box evasion cases
for the selected target model.

**Selected target model:**
- Target Model: **{row['target_model']}**
- Successful Attacks Before Defence: **{int(row['successful_attacks_before_defence'])}**
- Successful Attacks Caught by Defence: **{int(row['successful_attacks_caught_by_defence'])}**
- Defence Catch Rate: **{row['defence_catch_rate']:.2%}**
- Review Rate: **{row['review_rate']:.2%}**

**Conclusion:**  
This result shows whether the noise-stability gate adds practical black-box protection for the selected
ZOO target model under a low or high review burden.
"""
    )


# ---------------------------------------------------
# Disagreement comparison renderer
# ---------------------------------------------------
def render_disagreement_comparison_section(with_lr_df, without_lr_df, with_lr_details, without_lr_details, total_samples):
    if with_lr_df is None or with_lr_df.empty:
        st.warning("Disagreement with-LR results file not found.")
        return

    if without_lr_df is None or without_lr_df.empty:
        st.warning("Disagreement without-LR results file not found.")
        return

    st.success("Both disagreement comparison files loaded successfully.")

    attack_options = get_step3_attack_options(with_lr_df)
    if not attack_options:
        st.warning("No FGSM or PGD disagreement-defence results found for Step 3.")
        return

    selected_attack = st.selectbox(
        "Select attack type for disagreement comparison",
        attack_options,
        key="disagreement_compare_attack_select"
    )

    with_lr_filtered = with_lr_df[with_lr_df["attack_name"] == selected_attack].copy()
    without_lr_filtered = without_lr_df[without_lr_df["attack_name"] == selected_attack].copy()

    epsilon_options = sorted(with_lr_filtered["epsilon"].unique().tolist())
    selected_epsilon = st.selectbox(
        "Select epsilon for disagreement comparison",
        epsilon_options,
        index=len(epsilon_options) - 1,
        key="disagreement_compare_epsilon_select"
    )

    with_lr_row = filter_by_attack_and_epsilon(with_lr_filtered, selected_attack, selected_epsilon).iloc[0]
    without_lr_row = filter_by_attack_and_epsilon(without_lr_filtered, selected_attack, selected_epsilon).iloc[0]

    with_lr_review_count = int(round(float(with_lr_row["review_rate"]) * total_samples))
    without_lr_review_count = int(round(float(without_lr_row["review_rate"]) * total_samples))

    st.subheader("Disagreement Defence Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### With LR")
        a1, a2, a3 = st.columns(3)
        a1.metric("Caught", f"{int(with_lr_row['successful_attacks_caught_by_defence'])} / {int(with_lr_row['successful_attacks_before_defence'])}")
        a2.metric("Catch Rate", f"{with_lr_row['defence_catch_rate']:.2%}")
        a3.metric("Review Rate", f"{with_lr_row['review_rate']:.2%}")

        b1, b2 = st.columns(2)
        b1.metric("Manual Review Count", with_lr_review_count)
        b2.metric("Recall After", f"{with_lr_row['defended_recall_after']:.4f}")

    with col2:
        st.markdown("### Without LR")
        c1, c2, c3 = st.columns(3)
        c1.metric("Caught", f"{int(without_lr_row['successful_attacks_caught_by_defence'])} / {int(without_lr_row['successful_attacks_before_defence'])}")
        c2.metric("Catch Rate", f"{without_lr_row['defence_catch_rate']:.2%}")
        c3.metric("Review Rate", f"{without_lr_row['review_rate']:.2%}")

        d1, d2 = st.columns(2)
        d1.metric("Manual Review Count", without_lr_review_count)
        d2.metric("Recall After", f"{without_lr_row['defended_recall_after']:.4f}")

    st.info(
        f"For {selected_attack} at epsilon {selected_epsilon:.3f}, disagreement with LR caught "
        f"{int(with_lr_row['successful_attacks_caught_by_defence'])} of "
        f"{int(with_lr_row['successful_attacks_before_defence'])} successful attacks with about "
        f"{with_lr_review_count} reviews, while disagreement without LR caught "
        f"{int(without_lr_row['successful_attacks_caught_by_defence'])} with about "
        f"{without_lr_review_count} reviews."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_variant_rate_chart(with_lr_row, without_lr_row, "Disagreement Defence: With LR vs Without LR")

    with chart_col2:
        compare_plot_df = pd.DataFrame({
            "Variant": ["With LR", "Without LR"],
            "Manual Review Count": [with_lr_review_count, without_lr_review_count],
        })
        plot_bar(compare_plot_df, "Variant", "Manual Review Count", "Manual Review Count Comparison", "Count")

    compare_df = pd.DataFrame([
        {
            "Variant": "With LR",
            "Attack": with_lr_row["attack_name"],
            "Epsilon": round(float(with_lr_row["epsilon"]), 4),
            "Successful Attacks Before": int(with_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(with_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(with_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(with_lr_row["review_rate"]), 4),
            "Manual Review Count": with_lr_review_count,
            "Recall After": round(float(with_lr_row["defended_recall_after"]), 4),
        },
        {
            "Variant": "Without LR",
            "Attack": without_lr_row["attack_name"],
            "Epsilon": round(float(without_lr_row["epsilon"]), 4),
            "Successful Attacks Before": int(without_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(without_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(without_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(without_lr_row["review_rate"]), 4),
            "Manual Review Count": without_lr_review_count,
            "Recall After": round(float(without_lr_row["defended_recall_after"]), 4),
        },
    ])

    st.subheader("Comparison Table")
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.subheader("Exact Sample View")

    sample_view_choice = st.radio(
        "Select sample view",
        ["Caught attack samples", "All flagged review samples"],
        horizontal=True,
        key="disagreement_sample_view_choice"
    )

    detail_variant = st.selectbox(
        "Select detail variant",
        ["With LR", "Without LR"],
        key="disagreement_detail_variant_select"
    )

    detail_df = with_lr_details if detail_variant == "With LR" else without_lr_details

    if detail_df is None or detail_df.empty:
        st.info("Detailed sample file not found for this variant.")
        return

    detail_df = filter_by_attack_and_epsilon(detail_df, selected_attack, selected_epsilon)

    if sample_view_choice == "Caught attack samples":
        detail_df = detail_df[detail_df["caught_by_defence"] == 1].copy()
    else:
        detail_df = detail_df[detail_df["disagreement_gate_triggered"] == 1].copy()

    if detail_df.empty:
        st.info("No matching sample rows found for the selected filters.")
        return

    show_cols = [
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "adv_prob",
        "was_attacked",
        "successful_attack_before_defence",
        "disagreement_gate_triggered",
        "caught_by_defence",
        "fraud_votes",
        "nonfraud_votes",
        "ensemble_prob_std",
        "ensemble_prob_range",
        "majority_margin",
    ]

    available_cols = [c for c in show_cols if c in detail_df.columns]
    st.dataframe(detail_df[available_cols], use_container_width=True, height=300)

    st.markdown(
        """
**Interpretation note:**  
- **Manual Review Count** means how many transactions were flagged by the disagreement gate for further checking.  
- **Caught attack samples** are the adversarially successful fraud cases that were later flagged by the defence.  
- **All flagged review samples** shows the wider suspicious set, which explains the trade-off between stronger protection and user friction.
"""
    )


# ---------------------------------------------------
# ZOO disagreement comparison renderer
# ---------------------------------------------------
def render_zoo_disagreement_comparison_section(with_lr_df, without_lr_df, with_lr_details, without_lr_details, total_samples):
    if with_lr_df is None or with_lr_df.empty:
        st.warning("ZOO disagreement with-LR results file not found.")
        return

    if without_lr_df is None or without_lr_df.empty:
        st.warning("ZOO disagreement without-LR results file not found.")
        return

    st.success("Both ZOO disagreement comparison files loaded successfully.")

    st.info(
        "ZOO is a target-model-specific black-box attack. Therefore, ZOO disagreement-defence comparison is shown "
        "per target model rather than per epsilon."
    )

    model_options = sorted(with_lr_df["target_model"].dropna().unique().tolist())
    selected_model = st.selectbox(
        "Select target model for ZOO disagreement comparison",
        model_options,
        key="zoo_disagreement_target_model_select"
    )

    with_lr_row = with_lr_df[with_lr_df["target_model"] == selected_model].iloc[0]
    without_lr_row = without_lr_df[without_lr_df["target_model"] == selected_model].iloc[0]

    with_lr_review_count = int(round(float(with_lr_row["review_rate"]) * total_samples))
    without_lr_review_count = int(round(float(without_lr_row["review_rate"]) * total_samples))

    st.subheader("ZOO Disagreement Defence Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### With LR")
        a1, a2, a3 = st.columns(3)
        a1.metric("Caught", f"{int(with_lr_row['successful_attacks_caught_by_defence'])} / {int(with_lr_row['successful_attacks_before_defence'])}")
        a2.metric("Catch Rate", f"{with_lr_row['defence_catch_rate']:.2%}")
        a3.metric("Review Rate", f"{with_lr_row['review_rate']:.2%}")

        b1, b2 = st.columns(2)
        b1.metric("Manual Review Count", with_lr_review_count)
        b2.metric("Recall After", f"{with_lr_row['defended_recall_after']:.4f}")

    with col2:
        st.markdown("### Without LR")
        c1, c2, c3 = st.columns(3)
        c1.metric("Caught", f"{int(without_lr_row['successful_attacks_caught_by_defence'])} / {int(without_lr_row['successful_attacks_before_defence'])}")
        c2.metric("Catch Rate", f"{without_lr_row['defence_catch_rate']:.2%}")
        c3.metric("Review Rate", f"{without_lr_row['review_rate']:.2%}")

        d1, d2 = st.columns(2)
        d1.metric("Manual Review Count", without_lr_review_count)
        d2.metric("Recall After", f"{without_lr_row['defended_recall_after']:.4f}")

    st.info(
        f"For ZOO against {selected_model}, disagreement with LR caught "
        f"{int(with_lr_row['successful_attacks_caught_by_defence'])} of "
        f"{int(with_lr_row['successful_attacks_before_defence'])} successful attacks with about "
        f"{with_lr_review_count} reviews, while disagreement without LR caught "
        f"{int(without_lr_row['successful_attacks_caught_by_defence'])} with about "
        f"{without_lr_review_count} reviews."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_variant_rate_chart(with_lr_row, without_lr_row, f"ZOO Disagreement Defence: {selected_model}")

    with chart_col2:
        compare_plot_df = pd.DataFrame({
            "Variant": ["With LR", "Without LR"],
            "Manual Review Count": [with_lr_review_count, without_lr_review_count],
        })
        plot_bar(compare_plot_df, "Variant", "Manual Review Count", "Manual Review Count Comparison", "Count")

    compare_df = pd.DataFrame([
        {
            "Variant": "With LR",
            "Attack": with_lr_row["attack_name"],
            "Target Model": with_lr_row["target_model"],
            "Successful Attacks Before": int(with_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(with_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(with_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(with_lr_row["review_rate"]), 4),
            "Manual Review Count": with_lr_review_count,
            "Recall After": round(float(with_lr_row["defended_recall_after"]), 4),
        },
        {
            "Variant": "Without LR",
            "Attack": without_lr_row["attack_name"],
            "Target Model": without_lr_row["target_model"],
            "Successful Attacks Before": int(without_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(without_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(without_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(without_lr_row["review_rate"]), 4),
            "Manual Review Count": without_lr_review_count,
            "Recall After": round(float(without_lr_row["defended_recall_after"]), 4),
        },
    ])

    st.subheader("Comparison Table")
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.subheader("Exact Sample View")

    sample_view_choice = st.radio(
        "Select ZOO disagreement sample view",
        ["Caught attack samples", "All flagged review samples"],
        horizontal=True,
        key="zoo_disagreement_sample_view_choice"
    )

    detail_variant = st.selectbox(
        "Select ZOO detail variant",
        ["With LR", "Without LR"],
        key="zoo_disagreement_detail_variant_select"
    )

    detail_df = with_lr_details if detail_variant == "With LR" else without_lr_details

    if detail_df is None or detail_df.empty:
        st.info("Detailed ZOO sample file not found for this variant.")
        return

    detail_df = detail_df[detail_df["target_model"] == selected_model].copy()

    if sample_view_choice == "Caught attack samples":
        detail_df = detail_df[detail_df["caught_by_defence"] == 1].copy()
    else:
        detail_df = detail_df[detail_df["disagreement_gate_triggered"] == 1].copy()

    if detail_df.empty:
        st.info("No matching ZOO sample rows found for the selected filters.")
        return

    show_cols = [
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "adv_prob",
        "was_attacked",
        "successful_attack_before_defence",
        "disagreement_gate_triggered",
        "caught_by_defence",
        "fraud_votes",
        "nonfraud_votes",
        "ensemble_prob_std",
        "ensemble_prob_range",
        "majority_margin",
    ]

    available_cols = [c for c in show_cols if c in detail_df.columns]
    st.dataframe(detail_df[available_cols], use_container_width=True, height=300)


# ---------------------------------------------------
# Combined comparison renderer
# ---------------------------------------------------
def render_combined_comparison_section(with_lr_df, without_lr_df, with_lr_details, without_lr_details, total_samples):
    if with_lr_df is None or with_lr_df.empty:
        st.warning("Combined with-LR results file not found.")
        return

    if without_lr_df is None or without_lr_df.empty:
        st.warning("Combined without-LR results file not found.")
        return

    st.success("Both combined comparison files loaded successfully.")

    attack_options = get_step3_attack_options(with_lr_df)
    if not attack_options:
        st.warning("No FGSM or PGD combined-defence results found for Step 3.")
        return

    selected_attack = st.selectbox(
        "Select attack type for combined comparison",
        attack_options,
        key="combined_compare_attack_select"
    )

    with_lr_filtered = with_lr_df[with_lr_df["attack_name"] == selected_attack].copy()
    without_lr_filtered = without_lr_df[without_lr_df["attack_name"] == selected_attack].copy()

    epsilon_options = sorted(with_lr_filtered["epsilon"].unique().tolist())
    selected_epsilon = st.selectbox(
        "Select epsilon for combined comparison",
        epsilon_options,
        index=len(epsilon_options) - 1,
        key="combined_compare_epsilon_select"
    )

    with_lr_row = filter_by_attack_and_epsilon(with_lr_filtered, selected_attack, selected_epsilon).iloc[0]
    without_lr_row = filter_by_attack_and_epsilon(without_lr_filtered, selected_attack, selected_epsilon).iloc[0]

    with_lr_review_count = int(round(float(with_lr_row["review_rate"]) * total_samples))
    without_lr_review_count = int(round(float(without_lr_row["review_rate"]) * total_samples))

    st.subheader("Combined Defence Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### With LR")
        a1, a2, a3 = st.columns(3)
        a1.metric("Caught", f"{int(with_lr_row['successful_attacks_caught_by_defence'])} / {int(with_lr_row['successful_attacks_before_defence'])}")
        a2.metric("Catch Rate", f"{with_lr_row['defence_catch_rate']:.2%}")
        a3.metric("Review Rate", f"{with_lr_row['review_rate']:.2%}")

        b1, b2 = st.columns(2)
        b1.metric("Manual Review Count", with_lr_review_count)
        b2.metric("Recall After", f"{with_lr_row['defended_recall_after']:.4f}")

    with col2:
        st.markdown("### Without LR")
        c1, c2, c3 = st.columns(3)
        c1.metric("Caught", f"{int(without_lr_row['successful_attacks_caught_by_defence'])} / {int(without_lr_row['successful_attacks_before_defence'])}")
        c2.metric("Catch Rate", f"{without_lr_row['defence_catch_rate']:.2%}")
        c3.metric("Review Rate", f"{without_lr_row['review_rate']:.2%}")

        d1, d2 = st.columns(2)
        d1.metric("Manual Review Count", without_lr_review_count)
        d2.metric("Recall After", f"{without_lr_row['defended_recall_after']:.4f}")

    st.info(
        f"For {selected_attack} at epsilon {selected_epsilon:.3f}, combined with LR caught "
        f"{int(with_lr_row['successful_attacks_caught_by_defence'])} of "
        f"{int(with_lr_row['successful_attacks_before_defence'])} successful attacks with about "
        f"{with_lr_review_count} reviews, while combined without LR caught "
        f"{int(without_lr_row['successful_attacks_caught_by_defence'])} with about "
        f"{without_lr_review_count} reviews."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_variant_rate_chart(with_lr_row, without_lr_row, "Combined Defence: With LR vs Without LR")

    with chart_col2:
        compare_plot_df = pd.DataFrame({
            "Variant": ["With LR", "Without LR"],
            "Manual Review Count": [with_lr_review_count, without_lr_review_count],
        })
        plot_bar(compare_plot_df, "Variant", "Manual Review Count", "Manual Review Count Comparison", "Count")

    compare_df = pd.DataFrame([
        {
            "Variant": "With LR",
            "Attack": with_lr_row["attack_name"],
            "Epsilon": round(float(with_lr_row["epsilon"]), 4),
            "Successful Attacks Before": int(with_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(with_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(with_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(with_lr_row["review_rate"]), 4),
            "Manual Review Count": with_lr_review_count,
            "Recall After": round(float(with_lr_row["defended_recall_after"]), 4),
        },
        {
            "Variant": "Without LR",
            "Attack": without_lr_row["attack_name"],
            "Epsilon": round(float(without_lr_row["epsilon"]), 4),
            "Successful Attacks Before": int(without_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(without_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(without_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(without_lr_row["review_rate"]), 4),
            "Manual Review Count": without_lr_review_count,
            "Recall After": round(float(without_lr_row["defended_recall_after"]), 4),
        },
    ])

    st.subheader("Comparison Table")
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.subheader("Exact Sample View")

    sample_view_choice = st.radio(
        "Select sample view",
        ["Caught attack samples", "All flagged review samples"],
        horizontal=True,
        key="combined_sample_view_choice"
    )

    detail_variant = st.selectbox(
        "Select detail variant",
        ["With LR", "Without LR"],
        key="combined_detail_variant_select"
    )

    detail_df = with_lr_details if detail_variant == "With LR" else without_lr_details

    if detail_df is None or detail_df.empty:
        st.info("Detailed sample file not found for this variant.")
        return

    detail_df = filter_by_attack_and_epsilon(detail_df, selected_attack, selected_epsilon)

    if sample_view_choice == "Caught attack samples":
        detail_df = detail_df[detail_df["caught_by_defence"] == 1].copy()
    else:
        detail_df = detail_df[detail_df["review_flag"] == 1].copy()

    if detail_df.empty:
        st.info("No matching sample rows found for the selected filters.")
        return

    show_cols = [
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "adv_prob",
        "was_attacked",
        "successful_attack_before_defence",
        "noise_gate_triggered",
        "disagreement_gate_triggered",
        "review_flag",
        "caught_by_defence",
        "fraud_votes",
        "nonfraud_votes",
        "ensemble_prob_std",
        "ensemble_prob_range",
        "majority_margin",
        "noise_flip_rate",
        "noise_prob_std",
    ]

    available_cols = [c for c in show_cols if c in detail_df.columns]
    st.dataframe(detail_df[available_cols], use_container_width=True, height=300)


# ---------------------------------------------------
# ZOO combined comparison renderer
# ---------------------------------------------------
def render_zoo_combined_comparison_section(with_lr_df, without_lr_df, with_lr_details, without_lr_details, total_samples):
    if with_lr_df is None or with_lr_df.empty:
        st.warning("ZOO combined with-LR results file not found.")
        return

    if without_lr_df is None or without_lr_df.empty:
        st.warning("ZOO combined without-LR results file not found.")
        return

    st.success("Both ZOO combined comparison files loaded successfully.")

    st.info(
        "ZOO is a target-model-specific black-box attack. Therefore, ZOO combined-defence comparison is shown "
        "per target model rather than per epsilon."
    )

    model_options = sorted(with_lr_df["target_model"].dropna().unique().tolist())
    selected_model = st.selectbox(
        "Select target model for ZOO combined comparison",
        model_options,
        key="zoo_combined_target_model_select"
    )

    with_lr_row = with_lr_df[with_lr_df["target_model"] == selected_model].iloc[0]
    without_lr_row = without_lr_df[without_lr_df["target_model"] == selected_model].iloc[0]

    with_lr_review_count = int(round(float(with_lr_row["review_rate"]) * total_samples))
    without_lr_review_count = int(round(float(without_lr_row["review_rate"]) * total_samples))

    st.subheader("ZOO Combined Defence Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### With LR")
        a1, a2, a3 = st.columns(3)
        a1.metric("Caught", f"{int(with_lr_row['successful_attacks_caught_by_defence'])} / {int(with_lr_row['successful_attacks_before_defence'])}")
        a2.metric("Catch Rate", f"{with_lr_row['defence_catch_rate']:.2%}")
        a3.metric("Review Rate", f"{with_lr_row['review_rate']:.2%}")

        b1, b2 = st.columns(2)
        b1.metric("Manual Review Count", with_lr_review_count)
        b2.metric("Recall After", f"{with_lr_row['defended_recall_after']:.4f}")

    with col2:
        st.markdown("### Without LR")
        c1, c2, c3 = st.columns(3)
        c1.metric("Caught", f"{int(without_lr_row['successful_attacks_caught_by_defence'])} / {int(without_lr_row['successful_attacks_before_defence'])}")
        c2.metric("Catch Rate", f"{without_lr_row['defence_catch_rate']:.2%}")
        c3.metric("Review Rate", f"{without_lr_row['review_rate']:.2%}")

        d1, d2 = st.columns(2)
        d1.metric("Manual Review Count", without_lr_review_count)
        d2.metric("Recall After", f"{without_lr_row['defended_recall_after']:.4f}")

    st.info(
        f"For ZOO against {selected_model}, combined with LR caught "
        f"{int(with_lr_row['successful_attacks_caught_by_defence'])} of "
        f"{int(with_lr_row['successful_attacks_before_defence'])} successful attacks with about "
        f"{with_lr_review_count} reviews, while combined without LR caught "
        f"{int(without_lr_row['successful_attacks_caught_by_defence'])} with about "
        f"{without_lr_review_count} reviews."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_variant_rate_chart(with_lr_row, without_lr_row, f"ZOO Combined Defence: {selected_model}")

    with chart_col2:
        compare_plot_df = pd.DataFrame({
            "Variant": ["With LR", "Without LR"],
            "Manual Review Count": [with_lr_review_count, without_lr_review_count],
        })
        plot_bar(compare_plot_df, "Variant", "Manual Review Count", "Manual Review Count Comparison", "Count")

    compare_df = pd.DataFrame([
        {
            "Variant": "With LR",
            "Attack": with_lr_row["attack_name"],
            "Target Model": with_lr_row["target_model"],
            "Successful Attacks Before": int(with_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(with_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(with_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(with_lr_row["review_rate"]), 4),
            "Manual Review Count": with_lr_review_count,
            "Recall After": round(float(with_lr_row["defended_recall_after"]), 4),
        },
        {
            "Variant": "Without LR",
            "Attack": without_lr_row["attack_name"],
            "Target Model": without_lr_row["target_model"],
            "Successful Attacks Before": int(without_lr_row["successful_attacks_before_defence"]),
            "Caught by Defence": int(without_lr_row["successful_attacks_caught_by_defence"]),
            "Catch Rate": round(float(without_lr_row["defence_catch_rate"]), 4),
            "Review Rate": round(float(without_lr_row["review_rate"]), 4),
            "Manual Review Count": without_lr_review_count,
            "Recall After": round(float(without_lr_row["defended_recall_after"]), 4),
        },
    ])

    st.subheader("Comparison Table")
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.subheader("Exact Sample View")

    sample_view_choice = st.radio(
        "Select ZOO combined sample view",
        ["Caught attack samples", "All flagged review samples"],
        horizontal=True,
        key="zoo_combined_sample_view_choice"
    )

    detail_variant = st.selectbox(
        "Select ZOO combined detail variant",
        ["With LR", "Without LR"],
        key="zoo_combined_detail_variant_select"
    )

    detail_df = with_lr_details if detail_variant == "With LR" else without_lr_details

    if detail_df is None or detail_df.empty:
        st.info("Detailed ZOO sample file not found for this variant.")
        return

    detail_df = detail_df[detail_df["target_model"] == selected_model].copy()

    if sample_view_choice == "Caught attack samples":
        detail_df = detail_df[detail_df["caught_by_defence"] == 1].copy()
    else:
        detail_df = detail_df[detail_df["review_flag"] == 1].copy()

    if detail_df.empty:
        st.info("No matching ZOO sample rows found for the selected filters.")
        return

    show_cols = [
        "sample_index",
        "true_label",
        "clean_pred",
        "adv_pred",
        "adv_prob",
        "was_attacked",
        "successful_attack_before_defence",
        "noise_gate_triggered",
        "disagreement_gate_triggered",
        "review_flag",
        "caught_by_defence",
        "fraud_votes",
        "nonfraud_votes",
        "ensemble_prob_std",
        "ensemble_prob_range",
        "majority_margin",
        "noise_flip_rate",
        "noise_prob_std",
    ]

    available_cols = [c for c in show_cols if c in detail_df.columns]
    st.dataframe(detail_df[available_cols], use_container_width=True, height=300)


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("Controls")
st.sidebar.write("**Model:** PyTorch MLP")
st.sidebar.write("**Attacks:** FGSM, PGD, Transfer FGSM, Transfer PGD, ZOO")
st.sidebar.write("**Defences:** Noise Stability, Ensemble Disagreement, Combined")
st.sidebar.write(f"**Decision Threshold:** {THRESHOLD}")
st.sidebar.write("---")
st.sidebar.info(
    "Demo flow:\n\n"
    "1. Review clean MLP statistics\n"
    "2. View adversarial attack results\n"
    "3. View defence effectiveness against attacked samples"
)


# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("Credit Card Fraud Robustness Demo")
st.caption(
    "Clean model performance → FGSM → PGD → Transfer FGSM → Transfer PGD → ZOO → Defence Effectiveness"
)


# ---------------------------------------------------
# Step 1: Clean model
# ---------------------------------------------------
st.header("Step 1 — Baseline Multi-Layer Perceptron (MLP) Performance")
st.caption("Clean test-set performance of the tuned PyTorch MLP fraud detection model.")

try:
    clean_metrics = get_clean_mlp_stats()

    row1 = st.columns(4)
    row1[0].metric("Precision", f"{clean_metrics['precision']:.4f}")
    row1[1].metric("Recall", f"{clean_metrics['recall']:.4f}")
    row1[2].metric("F1", f"{clean_metrics['f1']:.4f}")
    row1[3].metric("PR-AUC", f"{clean_metrics['pr_auc']:.4f}")

    row2 = st.columns(4)
    row2[0].metric("TP", int(clean_metrics["tp"]))
    row2[1].metric("FN", int(clean_metrics["fn"]))
    row2[2].metric("FP", int(clean_metrics["fp"]))
    row2[3].metric("TN", int(clean_metrics["tn"]))

except Exception as e:
    st.error(f"Could not load clean MLP stats: {e}")
    st.stop()


# ---------------------------------------------------
# Step 2: Attack tabs
# ---------------------------------------------------
st.header("Step 2 — Adversarial Robustness Results")

tab_fgsm, tab_pgd, tab_transfer_fgsm, tab_transfer_pgd, tab_zoo = st.tabs(
    ["FGSM", "PGD", "Transfer FGSM", "Transfer PGD", "ZOO"]
)

with tab_fgsm:
    st.subheader("FGSM Attack Analysis")
    st.write("This section shows the saved FGSM attack results for the tuned PyTorch MLP.")

    if "show_fgsm" not in st.session_state:
        st.session_state.show_fgsm = False

    if st.button("Show FGSM Results", key="fgsm_button", type="primary"):
        st.session_state.show_fgsm = True

    if st.session_state.show_fgsm:
        fgsm_df = load_attack_results(str(FGSM_RESULTS_PATH))
        render_attack_section(fgsm_df, clean_metrics, "FGSM")
    else:
        st.info("Click 'Show FGSM Results' to review FGSM robustness results.")

with tab_pgd:
    st.subheader("PGD Attack Analysis")
    st.write("This section shows the saved PGD attack results for the tuned PyTorch MLP.")

    if "show_pgd" not in st.session_state:
        st.session_state.show_pgd = False

    if st.button("Show PGD Results", key="pgd_button", type="primary"):
        st.session_state.show_pgd = True

    if st.session_state.show_pgd:
        pgd_df = load_attack_results(str(PGD_RESULTS_PATH))
        render_attack_section(pgd_df, clean_metrics, "PGD")
    else:
        st.info("Click 'Show PGD Results' to review PGD robustness results.")

with tab_transfer_fgsm:
    st.subheader("Transfer FGSM Analysis")
    st.write(
        "This section shows whether adversarial fraud samples crafted on the PyTorch MLP "
        "also degrade other trained fraud classifiers."
    )

    if "show_transfer_fgsm" not in st.session_state:
        st.session_state.show_transfer_fgsm = False

    if st.button("Show Transfer FGSM Results", key="transfer_fgsm_button", type="primary"):
        st.session_state.show_transfer_fgsm = True

    if st.session_state.show_transfer_fgsm:
        transfer_fgsm_df = load_attack_results(str(TRANSFER_FGSM_RESULTS_PATH))
        render_transfer_section(transfer_fgsm_df, "Transfer FGSM")
    else:
        st.info("Click 'Show Transfer FGSM Results' to review transfer robustness results.")

with tab_transfer_pgd:
    st.subheader("Transfer PGD Analysis")
    st.write(
        "This section shows whether PGD adversarial fraud samples crafted on the PyTorch MLP "
        "also degrade other trained fraud classifiers."
    )

    if "show_transfer_pgd" not in st.session_state:
        st.session_state.show_transfer_pgd = False

    if st.button("Show Transfer PGD Results", key="transfer_pgd_button", type="primary"):
        st.session_state.show_transfer_pgd = True

    if st.session_state.show_transfer_pgd:
        transfer_pgd_df = load_attack_results(str(TRANSFER_PGD_RESULTS_PATH))
        render_transfer_section(transfer_pgd_df, "Transfer PGD")
    else:
        st.info("Click 'Show Transfer PGD Results' to review transfer robustness results.")

with tab_zoo:
    st.subheader("ZOO Black-Box Attack Analysis")
    st.write(
        "This section shows target-model-specific ZOO black-box attack results. "
        "ZOO is displayed offline because query-based black-box attacks are computationally expensive."
    )

    if "show_zoo" not in st.session_state:
        st.session_state.show_zoo = False

    if st.button("Show ZOO Results", key="zoo_button", type="primary"):
        st.session_state.show_zoo = True

    if st.session_state.show_zoo:
        zoo_summary_df = load_zoo_summary_results(str(ZOO_RESULTS_PATH))
        zoo_sample_df = load_csv_data(str(ZOO_SAMPLE_RESULTS_PATH))
        render_zoo_section(zoo_summary_df, zoo_sample_df)
    else:
        st.info("Click 'Show ZOO Results' to review black-box robustness results.")


# ---------------------------------------------------
# Step 3: Defence tabs
# ---------------------------------------------------
st.header("Step 3 — Defence Effectiveness Against Attacked Samples")

tab_noise, tab_disagreement, tab_combined = st.tabs(
    ["Noise Stability", "Ensemble Disagreement", "Combined Defence"]
)

with tab_noise:
    st.subheader("Noise Stability Defence Analysis")
    st.write(
        "This section shows whether the noise-stability defence can intercept successful "
        "adversarial evasions and route them to manual review."
    )

    defence_mode = st.radio(
        "Select evaluation mode",
        ["White-box Attacks", "ZOO Black-box"],
        horizontal=True,
        key="noise_defence_mode_radio"
    )

    if "show_noise_defence" not in st.session_state:
        st.session_state.show_noise_defence = False

    if st.button("Show Noise Stability Defence Results", key="noise_defence_button", type="primary"):
        st.session_state.show_noise_defence = True

    if st.session_state.show_noise_defence:
        if defence_mode == "White-box Attacks":
            noise_df = load_defence_results(str(NOISE_DEFENCE_RESULTS_PATH))
            render_defence_section(noise_df, "Noise Stability")
        else:
            noise_zoo_df = load_defence_results(str(NOISE_ZOO_RESULTS_PATH))
            noise_zoo_details = load_csv_data(str(NOISE_ZOO_DETAILS_PATH))
            render_zoo_noise_section(noise_zoo_df, noise_zoo_details)
    else:
        st.info("Click 'Show Noise Stability Defence Results' to review defence effectiveness.")

with tab_disagreement:
    st.subheader("Ensemble Disagreement Defence Analysis")
    st.write(
        "This section compares disagreement defence with and without Logistic Regression in the ensemble, "
        "so the trade-off between stronger protection and lower review overhead can be demonstrated clearly."
    )

    defence_mode = st.radio(
        "Select evaluation mode",
        ["White-box Attacks", "ZOO Black-box"],
        horizontal=True,
        key="disagreement_defence_mode_radio"
    )

    if "show_disagreement_defence" not in st.session_state:
        st.session_state.show_disagreement_defence = False

    if st.button("Show Ensemble Disagreement Results", key="disagreement_defence_button", type="primary"):
        st.session_state.show_disagreement_defence = True

    if st.session_state.show_disagreement_defence:
        total_samples = get_total_test_samples()

        if defence_mode == "White-box Attacks":
            disagreement_with_lr_df = load_defence_results(str(DISAGREEMENT_WITH_LR_RESULTS_PATH))
            disagreement_without_lr_df = load_defence_results(str(DISAGREEMENT_WITHOUT_LR_RESULTS_PATH))

            disagreement_with_lr_details = load_csv_data(str(DISAGREEMENT_WITH_LR_DETAILS_PATH))
            disagreement_without_lr_details = load_csv_data(str(DISAGREEMENT_WITHOUT_LR_DETAILS_PATH))

            render_disagreement_comparison_section(
                disagreement_with_lr_df,
                disagreement_without_lr_df,
                disagreement_with_lr_details,
                disagreement_without_lr_details,
                total_samples,
            )
        else:
            disagreement_zoo_with_lr_df = load_defence_results(str(DISAGREEMENT_ZOO_WITH_LR_RESULTS_PATH))
            disagreement_zoo_without_lr_df = load_defence_results(str(DISAGREEMENT_ZOO_WITHOUT_LR_RESULTS_PATH))

            disagreement_zoo_with_lr_details = load_csv_data(str(DISAGREEMENT_ZOO_WITH_LR_DETAILS_PATH))
            disagreement_zoo_without_lr_details = load_csv_data(str(DISAGREEMENT_ZOO_WITHOUT_LR_DETAILS_PATH))

            render_zoo_disagreement_comparison_section(
                disagreement_zoo_with_lr_df,
                disagreement_zoo_without_lr_df,
                disagreement_zoo_with_lr_details,
                disagreement_zoo_without_lr_details,
                total_samples,
            )
    else:
        st.info("Click 'Show Ensemble Disagreement Results' to review defence effectiveness.")

with tab_combined:
    st.subheader("Combined Defence Analysis")
    st.write(
        "This section compares combined defence with and without Logistic Regression in the ensemble. "
        "The final combined policy routes a transaction to manual review if either defence gate flags it."
    )

    defence_mode = st.radio(
        "Select evaluation mode",
        ["White-box Attacks", "ZOO Black-box"],
        horizontal=True,
        key="combined_defence_mode_radio"
    )

    if "show_combined_defence" not in st.session_state:
        st.session_state.show_combined_defence = False

    if st.button("Show Combined Defence Results", key="combined_defence_button", type="primary"):
        st.session_state.show_combined_defence = True

    if st.session_state.show_combined_defence:
        total_samples = get_total_test_samples()

        if defence_mode == "White-box Attacks":
            combined_with_lr_df = load_defence_results(str(COMBINED_WITH_LR_RESULTS_PATH))
            combined_without_lr_df = load_defence_results(str(COMBINED_WITHOUT_LR_RESULTS_PATH))

            combined_with_lr_details = load_csv_data(str(COMBINED_WITH_LR_DETAILS_PATH))
            combined_without_lr_details = load_csv_data(str(COMBINED_WITHOUT_LR_DETAILS_PATH))

            render_combined_comparison_section(
                combined_with_lr_df,
                combined_without_lr_df,
                combined_with_lr_details,
                combined_without_lr_details,
                total_samples,
            )
        else:
            combined_zoo_with_lr_df = load_defence_results(str(COMBINED_ZOO_WITH_LR_RESULTS_PATH))
            combined_zoo_without_lr_df = load_defence_results(str(COMBINED_ZOO_WITHOUT_LR_RESULTS_PATH))

            combined_zoo_with_lr_details = load_csv_data(str(COMBINED_ZOO_WITH_LR_DETAILS_PATH))
            combined_zoo_without_lr_details = load_csv_data(str(COMBINED_ZOO_WITHOUT_LR_DETAILS_PATH))

            render_zoo_combined_comparison_section(
                combined_zoo_with_lr_df,
                combined_zoo_without_lr_df,
                combined_zoo_with_lr_details,
                combined_zoo_without_lr_details,
                total_samples,
            )
    else:
        st.info("Click 'Show Combined Defence Results' to review defence effectiveness.")
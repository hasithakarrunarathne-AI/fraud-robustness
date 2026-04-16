# src/app/app.py

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
# If needed, change above import to:
# from src.models.train_mlp_torch import FraudMLP, load_processed_data, evaluate_model, DEVICE


# ---------------------------------------------------
# Paths
# ---------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "results" / "saved_models" / "mlp_torch.pth"

FGSM_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "fgsm_results.json"
PGD_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "pgd_results.json"
TRANSFER_FGSM_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "transfer_fgsm_results.json"
TRANSFER_PGD_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "transfer_pgd_results.json"
ZOO_RESULTS_PATH = PROJECT_ROOT / "results" / "attacks" / "zoo_results.json"

NOISE_DEFENCE_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "noise_defence_on_attacks.json"
DISAGREEMENT_DEFENCE_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "disagreement_defence_on_attacks.json"
COMBINED_DEFENCE_RESULTS_PATH = PROJECT_ROOT / "results" / "defences" / "on_attacks" / "combined_defence_on_attacks.json"

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
def load_attack_results(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    return pd.DataFrame(results)


@st.cache_data
def load_zoo_results(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return None, None

    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = results.get("summary", {})
    sample_results = results.get("sample_results", [])

    summary_df = pd.DataFrame([summary]) if summary else None
    sample_df = pd.DataFrame(sample_results) if sample_results else None

    return summary_df, sample_df


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


def plot_line(df, x_col, y_col, title, y_label):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epsilon", fontsize=9)
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


def format_delta(clean_value, adv_value):
    delta = adv_value - clean_value
    return f"{adv_value:.4f}", f"{delta:.4f}"


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
def render_zoo_section(summary_df, sample_df, clean_metrics):
    if summary_df is None or summary_df.empty:
        st.warning("ZOO results file not found. Run the zoo_attack.py script first.")
        return

    st.success("ZOO results loaded successfully.")

    row = summary_df.iloc[0]

    st.subheader("Key Black-Box Finding")

    c1, c2, c3, c4 = st.columns(4)

    recall_value, recall_delta = format_delta(clean_metrics["recall"], row["adv_recall"])
    f1_value, f1_delta = format_delta(clean_metrics["f1"], row["adv_f1"])
    prauc_value, prauc_delta = format_delta(clean_metrics["pr_auc"], row["adv_pr_auc"])

    c1.metric("Recall", recall_value, recall_delta)
    c2.metric("F1", f1_value, f1_delta)
    c3.metric("PR-AUC", prauc_value, prauc_delta)
    c4.metric("ASR", f"{row['asr']:.4f}")

    st.info(
        f"Precomputed ZOO black-box results for the PyTorch MLP. "
        f"This run used max_iters = {int(row['max_iters'])}, "
        f"learning_rate = {row['learning_rate']}, "
        f"coords_per_iter = {int(row['coords_per_iter'])}, "
        f"and attacked {int(row['attacked_fraud_samples'])} fraud samples."
    )

    st.subheader("ZOO Summary")
    summary_view = pd.DataFrame([{
        "Successful Attacks": int(row["successful_attacks"]),
        "Attack Attempts": int(row["attack_attempts"]),
        "Avg Queries / Sample": round(float(row["avg_queries_per_sample"]), 2),
        "Avg L2 Dist": round(float(row["avg_l2_dist"]), 4),
        "Avg Linf Dist": round(float(row["avg_linf_dist"]), 4),
        "Adv TP": int(row["adv_tp"]),
        "Adv FN": int(row["adv_fn"]),
        "Adv Precision": round(float(row["adv_precision"]), 4),
    }])
    st.dataframe(summary_view, use_container_width=True, hide_index=True)

    st.subheader("ZOO Comparison Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        plot_grouped_bar(
            categories=["Recall", "F1", "PR-AUC"],
            clean_values=[
                row["clean_recall"],
                row["clean_f1"],
                row["clean_pr_auc"],
            ],
            adv_values=[
                row["adv_recall"],
                row["adv_f1"],
                row["adv_pr_auc"],
            ],
            title="ZOO: Clean vs Adversarial Metrics",
            y_label="Score",
        )

    with chart_col2:
        plot_grouped_bar(
            categories=["TP", "FN", "FP"],
            clean_values=[
                row["clean_tp"],
                row["clean_fn"],
                row["clean_fp"],
            ],
            adv_values=[
                row["adv_tp"],
                row["adv_fn"],
                row["adv_fp"],
            ],
            title="ZOO: Changed Confusion Counts",
            y_label="Count",
        )

    st.subheader("ZOO Summary Table")
    st.dataframe(summary_df, use_container_width=True)

    if sample_df is not None and not sample_df.empty:
        sample_df = sample_df.copy()
        sample_df["prob_drop"] = sample_df["original_prob"] - sample_df["adv_prob"]

        st.subheader("Successful ZOO Attacks")

        success_df = sample_df[sample_df["success"] == True].copy()

        if not success_df.empty:
            success_df["original_prob"] = success_df["original_prob"].round(4)
            success_df["adv_prob"] = success_df["adv_prob"].round(4)
            success_df["l2_dist"] = success_df["l2_dist"].round(4)
            success_df["linf_dist"] = success_df["linf_dist"].round(4)
            success_df["prob_drop"] = (
                success_df["original_prob"] - success_df["adv_prob"]
            ).round(4)

            show_cols = [
                "test_index",
                "original_prob",
                "adv_prob",
                "original_pred",
                "adv_pred",
                "queries",
                "l2_dist",
                "linf_dist",
            ]
            st.dataframe(success_df[show_cols], use_container_width=True, height=220)

            plot_bar(
                success_df.astype({"test_index": str}),
                "test_index",
                "prob_drop",
                "ZOO: Probability Drop for Successful Attacks",
                "Probability Drop",
            )
        else:
            st.info("No successful ZOO attacks were found in the saved file.")

        st.subheader("Top Probability Drops")

        top_drop_df = sample_df.sort_values("prob_drop", ascending=False).head(10).copy()

        top_drop_df["original_prob"] = top_drop_df["original_prob"].round(4)
        top_drop_df["adv_prob"] = top_drop_df["adv_prob"].round(4)
        top_drop_df["prob_drop"] = top_drop_df["prob_drop"].round(4)
        top_drop_df["l2_dist"] = top_drop_df["l2_dist"].round(4)
        top_drop_df["linf_dist"] = top_drop_df["linf_dist"].round(4)

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
        st.dataframe(top_drop_df[show_cols], use_container_width=True, height=300)

        plot_bar(
            top_drop_df.astype({"test_index": str}),
            "test_index",
            "prob_drop",
            "ZOO: Top Probability Drops",
            "Probability Drop",
        )

    st.subheader("Research Interpretation")
    st.markdown(
        f"""
**Observation:**  
The ZOO black-box attack is weaker than the white-box attacks, but it still causes measurable degradation.

**Saved ZOO result:**
- Adversarial Recall: **{row['adv_recall']:.4f}**
- Adversarial F1: **{row['adv_f1']:.4f}**
- Adversarial PR-AUC: **{row['adv_pr_auc']:.4f}**
- Successful Attacks: **{int(row['successful_attacks'])}**
- Attack Success Rate (ASR): **{row['asr']:.4f}**
- Average Queries per Sample: **{row['avg_queries_per_sample']:.2f}**

**Conclusion:**  
The PyTorch MLP shows limited but meaningful black-box vulnerability.  
Compared with white-box attacks, ZOO requires many more queries and larger perturbations to achieve evasion.
"""
    )


# ---------------------------------------------------
# Defence section renderer
# ---------------------------------------------------
def render_defence_section(df, defence_label):
    if df is None or df.empty:
        st.warning(f"{defence_label} results file not found. Run the corresponding defence script first.")
        return

    st.success(f"{defence_label} results loaded successfully.")

    attack_options = sorted(df["attack_name"].dropna().unique().tolist())
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
        filtered_df = filtered_df[filtered_df["epsilon"] == selected_epsilon].copy()

    if filtered_df.empty:
        st.warning("No matching defence result found for the selected filters.")
        return

    row = filtered_df.iloc[0]

    st.subheader(f"{defence_label} Key Result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Successful Attacks Before", int(row["successful_attacks_before_defence"]))
    c2.metric("Caught by Defence", int(row["successful_attacks_caught_by_defence"]))
    c3.metric("Catch Rate", f"{row['defence_catch_rate']:.2%}")
    c4.metric("Review Rate", f"{row['review_rate']:.2%}")

    st.info(
        f"For {row['attack_name']} at epsilon {row['epsilon']:.3f}, the {defence_label} "
        f"caught {int(row['successful_attacks_caught_by_defence'])} out of "
        f"{int(row['successful_attacks_before_defence'])} successful attacks, while reviewing "
        f"{row['review_rate']:.2%} of transactions."
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

**Interpretation note:**  
The after-defence precision, F1, and PR-AUC values are shown for completeness, but they should be
interpreted carefully because this is a manual-review / abstain-style defence rather than a standard
binary classifier.

**Conclusion:**  
{conclusion_text}
"""
    )


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
st.header("Step 1 — Clean Model Performance")

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
        "This section shows precomputed ZOO black-box attack results for the PyTorch MLP. "
        "ZOO is displayed offline because query-based black-box attacks are computationally expensive."
    )

    if "show_zoo" not in st.session_state:
        st.session_state.show_zoo = False

    if st.button("Show ZOO Results", key="zoo_button", type="primary"):
        st.session_state.show_zoo = True

    if st.session_state.show_zoo:
        zoo_summary_df, zoo_sample_df = load_zoo_results(str(ZOO_RESULTS_PATH))
        render_zoo_section(zoo_summary_df, zoo_sample_df, clean_metrics)
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

    if "show_noise_defence" not in st.session_state:
        st.session_state.show_noise_defence = False

    if st.button("Show Noise Stability Defence Results", key="noise_defence_button", type="primary"):
        st.session_state.show_noise_defence = True

    if st.session_state.show_noise_defence:
        noise_df = load_defence_results(str(NOISE_DEFENCE_RESULTS_PATH))
        render_defence_section(noise_df, "Noise Stability")
    else:
        st.info("Click 'Show Noise Stability Defence Results' to review defence effectiveness.")

with tab_disagreement:
    st.subheader("Ensemble Disagreement Defence Analysis")
    st.write(
        "This section shows whether the ensemble-disagreement defence can intercept successful "
        "adversarial evasions and route them to manual review."
    )

    if "show_disagreement_defence" not in st.session_state:
        st.session_state.show_disagreement_defence = False

    if st.button("Show Ensemble Disagreement Results", key="disagreement_defence_button", type="primary"):
        st.session_state.show_disagreement_defence = True

    if st.session_state.show_disagreement_defence:
        disagreement_df = load_defence_results(str(DISAGREEMENT_DEFENCE_RESULTS_PATH))
        render_defence_section(disagreement_df, "Ensemble Disagreement")
    else:
        st.info("Click 'Show Ensemble Disagreement Results' to review defence effectiveness.")

with tab_combined:
    st.subheader("Combined Defence Analysis")
    st.write(
        "This section shows the final per-transaction combined defence policy, where a transaction "
        "is routed to manual review if either defence gate flags it as suspicious."
    )

    if "show_combined_defence" not in st.session_state:
        st.session_state.show_combined_defence = False

    if st.button("Show Combined Defence Results", key="combined_defence_button", type="primary"):
        st.session_state.show_combined_defence = True

    if st.session_state.show_combined_defence:
        combined_df = load_defence_results(str(COMBINED_DEFENCE_RESULTS_PATH))
        render_defence_section(combined_df, "Combined Defence")
    else:
        st.info("Click 'Show Combined Defence Results' to review defence effectiveness.")
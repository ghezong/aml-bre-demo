"""AML Web App aligned with BRE Streamlit design language."""

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve project paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
AML_CORE_DIR = SRC_DIR / "aml_core"
PROJECT_ROOT = CURRENT_DIR.parents[2]

for p in [SRC_DIR, AML_CORE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aml_core.generate_sample_data import generate_transaction_data, generate_account_data, resolve_output_paths
from aml_core.ml_model import train_rule_model, predict_with_rule_model, save_model, load_model


DATA_DIR = PROJECT_ROOT / "aml_app" / "data"
MODELS_DIR = PROJECT_ROOT / "aml_app" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "aml_app" / "outputs"


def ensure_dirs():
    (DATA_DIR / "training").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "new").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "accounts").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def init_state():
    if "training_df" not in st.session_state:
        st.session_state.training_df = None
    if "new_df" not in st.session_state:
        st.session_state.new_df = None
    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = None
    if "scored_df" not in st.session_state:
        st.session_state.scored_df = None


def render_header():
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: #1f4e79; margin: 0; text-align: center;">🔎 AML Rule + ML Studio</h1>
            <h3 style="color: #666; margin: 5px 0 0 0; text-align: center;">General AML Pattern Detection with Learned Rule Parameters</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_generation_section():
    st.subheader("🧪 Synthetic Data Generation")
    col1, col2, col3 = st.columns(3)

    with col1:
        data_type = st.selectbox("Data Type", ["training", "new"], index=0)
    with col2:
        num_tx = st.number_input("Number of Transactions", min_value=100, max_value=1000000, value=5000, step=100)
    with col3:
        tx_file_name = st.text_input("Transactions File Name", value="training_rules.csv" if data_type == "training" else "new_rules.csv")

    if st.button("Generate Sample Data", type="primary"):
        tx_df = generate_transaction_data(num_records=int(num_tx), data_type=data_type)
        acct_df = generate_account_data(num_accounts=100)

        tx_path, acct_path = resolve_output_paths(
            project_root=PROJECT_ROOT,
            data_type=data_type,
            transactions_file=tx_file_name,
            accounts_file="sample_accounts.csv",
        )
        tx_df.to_csv(tx_path, index=False)
        acct_df.to_csv(acct_path, index=False)

        st.success(f"Saved transaction data to: {tx_path}")
        st.success(f"Saved account data to: {acct_path}")
        st.dataframe(tx_df.head(10), use_container_width=True)


def render_training_section():
    st.subheader("🏋️ Model Training")

    uploaded_training = st.file_uploader("Upload Training CSV (must include alert column)", type=["csv"], key="training_uploader")
    if uploaded_training is not None:
        st.session_state.training_df = pd.read_csv(uploaded_training)
        st.success(f"Loaded {len(st.session_state.training_df):,} training records")

    default_training_path = DATA_DIR / "training" / "training_rules.csv"
    if st.button("Load Default Training File") and default_training_path.exists():
        st.session_state.training_df = pd.read_csv(default_training_path)
        st.success(f"Loaded default training file: {default_training_path}")

    if st.session_state.training_df is not None:
        st.dataframe(st.session_state.training_df.head(10), use_container_width=True)
        if "alert" not in st.session_state.training_df.columns:
            st.error("Training dataset requires an alert column.")
            return

        if st.button("Train AML Model", type="primary"):
            model_bundle = train_rule_model(st.session_state.training_df)
            st.session_state.model_bundle = model_bundle

            st.success("Model trained successfully.")
            learned = model_bundle["learned_params"]

            st.markdown("**Learned Parameters**")
            st.write({"decision_threshold": learned["decision_threshold"]})

            coeff_df = pd.DataFrame(
                list(learned["coefficients"].items()),
                columns=["rule", "coefficient"],
            ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
            st.markdown("**Rule Coefficients**")
            st.dataframe(coeff_df, use_container_width=True)

            ref_df = pd.DataFrame(
                list(learned["rule_reference_levels_median_alerts"].items()),
                columns=["rule", "median_alert_level"],
            )
            st.markdown("**Rule Reference Levels (Median in Alerts)**")
            st.dataframe(ref_df, use_container_width=True)

            st.markdown("**Evaluation**")
            st.write("Confusion Matrix", model_bundle["evaluation"]["confusion_matrix"])
            st.code(model_bundle["evaluation"]["classification_report"])

            model_output = MODELS_DIR / "aml_model.pkl"
            save_model(model_bundle, model_output)
            st.info(f"Saved model to: {model_output}")


def render_scoring_section():
    st.subheader("🔍 Score New Transactions")

    if st.session_state.model_bundle is None:
        model_path = MODELS_DIR / "aml_model.pkl"
        if model_path.exists() and st.button("Load Saved Model"):
            st.session_state.model_bundle = load_model(model_path)
            st.success(f"Loaded model: {model_path}")

    uploaded_new = st.file_uploader("Upload New Transactions CSV", type=["csv"], key="new_uploader")
    if uploaded_new is not None:
        st.session_state.new_df = pd.read_csv(uploaded_new)

    default_new_path = DATA_DIR / "new" / "new_rules.csv"
    if st.button("Load Default New Data") and default_new_path.exists():
        st.session_state.new_df = pd.read_csv(default_new_path)
        st.success(f"Loaded default new file: {default_new_path}")

    if st.session_state.model_bundle is not None and st.session_state.new_df is not None:
        if st.button("Run Scoring", type="primary"):
            scored = predict_with_rule_model(st.session_state.model_bundle, st.session_state.new_df)
            st.session_state.scored_df = scored

            total = len(scored)
            alerts = int((scored["prediction"] == 1).sum())
            col1, col2 = st.columns(2)
            col1.metric("Total Transactions", f"{total:,}")
            col2.metric("Flagged Alerts", f"{alerts:,}")

            st.markdown("**Scored Data Preview**")
            st.dataframe(scored.head(30), use_container_width=True)

            flagged = scored[scored["prediction"] == 1]
            if not flagged.empty:
                st.markdown("**Dominant Rule Ranking (Flagged Only)**")
                ranking = flagged["top_rule_contributor"].value_counts().rename_axis("rule").reset_index(name="count")
                ranking["pct_flagged"] = (ranking["count"] / len(flagged) * 100).round(2)
                st.dataframe(ranking, use_container_width=True)

            out_path = OUTPUTS_DIR / "scored_predictions.csv"
            scored.to_csv(out_path, index=False)
            st.success(f"Scored output saved to: {out_path}")

            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Scored CSV",
                data=csv_bytes,
                file_name="scored_predictions.csv",
                mime="text/csv",
            )


def main():
    st.set_page_config(page_title="AML Rule + ML Studio", page_icon="🔎", layout="wide")
    ensure_dirs()
    init_state()
    render_header()

    tab1, tab2, tab3 = st.tabs([
        "Generate Data",
        "Train Model",
        "Score New Data",
    ])

    with tab1:
        render_generation_section()
    with tab2:
        render_training_section()
    with tab3:
        render_scoring_section()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --blue: #1769aa;
        --blue-soft: #eaf3fb;
        --ink: #243b53;
        --muted: #627d98;
        --line: #d9e2ec;
        --surface: #ffffff;
        --canvas: #f5f7fa;
        --success: #147d64;
        --warning: #b45309;
        --danger: #b42318;
    }

    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stSidebar"] {
        background: #102a43;
        border-right: 1px solid #0b1f33;
    }
    [data-testid="stSidebar"] * { color: #f0f4f8; }
    [data-testid="stSidebar"] .stRadio label { color: #d9e2ec; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { margin: 0.25rem 0; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }

    .shell { max-width: 1380px; margin: 0 auto; }
    .brand-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.45rem 0 1.1rem 0; border-bottom: 1px solid var(--line);
        margin-bottom: 1.45rem;
    }
    .brand-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 2.35rem; height: 2.35rem; background: var(--navy); color: white;
        font-weight: 800; letter-spacing: .03em; font-size: .76rem;
        border-radius: .35rem; margin-right: .8rem;
    }
    .brand-title { font-size: 1.18rem; font-weight: 700; color: var(--navy); line-height: 1.1; }
    .brand-subtitle { font-size: .78rem; color: var(--muted); margin-top: .18rem; }
    .status-pill {
        border: 1px solid #b7dfd2; background: #eefaf6; color: var(--success);
        border-radius: 999px; padding: .36rem .7rem; font-size: .75rem; font-weight: 650;
    }
    .section-title { color: var(--navy); font-size: 1.05rem; font-weight: 750; margin: .25rem 0 .18rem; }
    .section-caption { color: var(--muted); font-size: .83rem; margin-bottom: .8rem; }
    .metric-card {
        background: var(--surface); border: 1px solid var(--line); border-radius: .45rem;
        padding: 1rem 1.05rem; min-height: 6.7rem;
    }
    .metric-label { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .055em; font-weight: 700; }
    .metric-value { color: var(--navy); font-size: 1.55rem; font-weight: 750; margin-top: .3rem; }
    .metric-note { color: var(--muted); font-size: .75rem; margin-top: .18rem; }
    .result-panel {
        background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--blue);
        border-radius: .45rem; padding: 1.25rem 1.35rem; margin: .4rem 0 1rem;
    }
    .result-label { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; }
    .result-title { color: var(--navy); font-size: 1.45rem; font-weight: 760; margin: .2rem 0 .35rem; }
    .result-copy { color: var(--ink); font-size: .9rem; line-height: 1.5; }
    .risk-high { border-left-color: var(--danger); }
    .risk-medium { border-left-color: var(--warning); }
    .risk-low { border-left-color: var(--success); }
    .small-note { color: var(--muted); font-size: .78rem; line-height: 1.45; }
    .footer { color: var(--muted); border-top: 1px solid var(--line); margin-top: 2.5rem; padding-top: .9rem; font-size: .74rem; }
    div[data-testid="stForm"] { border: 1px solid var(--line); background: var(--surface); padding: 1rem 1.1rem .35rem; border-radius: .45rem; }
    .stButton > button { border-radius: .35rem; font-weight: 650; }
    .stDownloadButton > button { border-radius: .35rem; }
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); padding: .7rem .8rem; border-radius: .4rem; }
    @media (max-width: 768px) {
        .brand-row { align-items: flex-start; gap: .8rem; }
        .status-pill { margin-top: .25rem; white-space: nowrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model loading and input preparation
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "churn_model.joblib"
META_PATH = ROOT / "model_meta.json"

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    model_bundle = joblib.load(MODEL_PATH)
    metadata: dict[str, Any] = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model_bundle, metadata


def unpack_bundle(bundle: Any) -> tuple[Any, Any, list[str]]:
    """Support the dictionary artifact used in the lab and plain estimators."""
    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("estimator") or bundle.get("classifier")
        encoder = bundle.get("serving_encoder") or bundle.get("encoder") or bundle.get("preprocessor")
        columns = bundle.get("feature_columns") or bundle.get("columns") or []
    else:
        model, encoder, columns = bundle, None, []
    if model is None:
        raise ValueError("The model artifact does not contain a model object.")
    return model, encoder, list(columns)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        out["TotalCharges"] = out["TotalCharges"].fillna(
            pd.to_numeric(out.get("tenure", 0), errors="coerce").fillna(0)
            * pd.to_numeric(out.get("MonthlyCharges", 0), errors="coerce").fillna(0)
        )
    return out.reindex(columns=RAW_COLS)


def encode_for_model(raw: pd.DataFrame, encoder: Any, feature_columns: list[str]) -> Any:
    clean = clean_raw(raw)
    if encoder is not None and hasattr(encoder, "transform"):
        return encoder.transform(clean)

    encoded = pd.get_dummies(clean, drop_first=True)
    if feature_columns:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


def predict_probability(raw: pd.DataFrame, bundle: Any) -> np.ndarray:
    model, encoder, feature_columns = unpack_bundle(bundle)
    features = encode_for_model(raw, encoder, feature_columns)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    return np.asarray(model.predict(features), dtype=float)


def threshold_from(meta: dict[str, Any]) -> float:
    for key in (
        "threshold", "selected_threshold", "chosen_threshold", "cost_based_threshold",
        "cost_optimal_threshold", "optimal_threshold", "decision_threshold",
    ):
        value = meta.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    # Some notebook exports keep business settings inside a nested dictionary.
    for value in meta.values():
        if isinstance(value, dict):
            nested = threshold_from(value)
            if nested != 0.5:
                return nested
    return 0.5


def fmt_money(value: float) -> str:
    return f"PKR {value:,.0f}"


try:
    bundle, metadata = load_artifacts()
    model_object, _, _ = unpack_bundle(bundle)
    selected_threshold = threshold_from(metadata)
    artifact_ready = True
    artifact_error = ""
except Exception as exc:  # show a useful deployment message instead of a cryptic crash
    bundle, metadata, model_object = None, {}, None
    selected_threshold = 0.5
    artifact_ready = False
    artifact_error = str(exc)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-size:.72rem;letter-spacing:.08em;font-weight:750;color:#9fb3c8;'>RETENTION OPERATIONS</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.38rem;font-weight:750;margin:.35rem 0 1.3rem;'>Churn Risk<br>Advisor</div>", unsafe_allow_html=True)
    page = st.radio("Workspace", ["Customer assessment", "Batch scoring", "Model details"], label_visibility="collapsed")
    st.divider()
    if artifact_ready:
        st.markdown("<div class='small-note' style='color:#bcccdc;'>Model status</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;margin:.15rem 0 .9rem;'>Ready for scoring</div>", unsafe_allow_html=True)
    else:
        st.error("Model artifact unavailable")
        st.caption(artifact_error)
    st.markdown("<div class='small-note' style='color:#9fb3c8;margin-top:2rem;'>Decision support only. Review recommendations with the retention team before contacting a customer.</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="shell brand-row">
      <div style="display:flex;align-items:center;">
        <div class="brand-mark">CR</div>
        <div><div class="brand-title">Churn Risk Advisor</div><div class="brand-subtitle">Customer retention decision support</div></div>
      </div>
      <div class="status-pill">● &nbsp;Production model online</div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not artifact_ready:
    st.error("The application loaded, but the model artifact could not be opened. Confirm that churn_model.joblib and model_meta.json are in the repository root.")
    st.stop()


if page == "Customer assessment":
    st.markdown('<div class="section-title">Customer assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Enter the customer profile to estimate churn risk and support the next retention action.</div>', unsafe_allow_html=True)

    with st.form("customer_form"):
        st.markdown("#### Customer profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            partner = st.selectbox("Partner", ["Yes", "No"])
        with c2:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            phone = st.selectbox("Phone service", ["Yes", "No"])
        with c3:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        with c4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        st.markdown("#### Services and billing")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        with s2:
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with s3:
            tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        with s4:
            stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        b1, b2 = st.columns(2)
        with b1:
            stream_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        with b2:
            monthly = st.number_input("Monthly charges (PKR)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)

        submitted = st.form_submit_button("Assess customer", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech, "StreamingTV": stream_tv, "StreamingMovies": stream_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": float(tenure) * float(monthly),
        }])
        probability = float(predict_probability(row, bundle)[0])
        predicted = probability >= selected_threshold
        if probability >= max(selected_threshold, .70):
            level, title, css = "High", "Prioritise retention outreach", "risk-high"
            copy = "This profile is materially above the selected decision threshold. Review the account and consider a timely retention offer."
        elif probability >= selected_threshold:
            level, title, css = "Moderate", "Review for targeted outreach", "risk-medium"
            copy = "The customer is above the decision threshold. Use account context and recent interactions before selecting an offer."
        else:
            level, title, css = "Low", "Monitor through normal service", "risk-low"
            copy = "The estimated risk is below the decision threshold. No special intervention is indicated by this model alone."

        st.markdown("<div class='section-title' style='margin-top:1.65rem;'>Assessment result</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-panel {css}'><div class='result-label'>Recommended disposition · {level} risk</div><div class='result-title'>{title}</div><div class='result-copy'>{copy}</div></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated churn probability", f"{probability:.1%}")
        m2.metric("Decision threshold", f"{selected_threshold:.1%}")
        m3.metric("Model decision", "Outreach" if predicted else "Monitor")
        m4.metric("Potential avoided loss", fmt_money(6000 if predicted else 0))
        st.caption("The probability is a model estimate, not a guarantee. The threshold is configured from the business-cost analysis in the project notebook.")


elif page == "Batch scoring":
    st.markdown('<div class="section-title">Batch scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Upload a CSV with customer records using the same fields as the training data. The scored file can be downloaded for review.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Customer CSV", type=["csv"], help="Required columns: the customer profile fields used by the model.")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [col for col in RAW_COLS if col not in data.columns]
        if missing:
            st.error(f"The file is missing {len(missing)} required column(s): {', '.join(missing)}")
        else:
            probabilities = predict_probability(data[RAW_COLS], bundle)
            scored = data.copy()
            scored["churn_probability"] = probabilities
            scored["recommended_action"] = np.where(probabilities >= selected_threshold, "Review for outreach", "Monitor")
            st.success(f"Scored {len(scored):,} customer records.")
            a, b, c = st.columns(3)
            a.metric("Records", f"{len(scored):,}")
            b.metric("Above threshold", f"{(probabilities >= selected_threshold).sum():,}")
            c.metric("Average risk", f"{probabilities.mean():.1%}")
            st.dataframe(scored.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "churn_scored_customers.csv", "text/csv", type="primary")


else:
    st.markdown('<div class="section-title">Model details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Reference information for reviewers and the retention team.</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("Decision threshold", f"{selected_threshold:.1%}")
    d2.metric("Estimator", type(model_object).__name__)
    d3.metric("Input fields", len(RAW_COLS))
    st.markdown("#### Model card")
    if metadata:
        pretty = {str(k).replace("_", " ").title(): v for k, v in metadata.items() if not isinstance(v, (dict, list))}
        if pretty:
            st.dataframe(pd.DataFrame(list(pretty.items()), columns=["Item", "Value"]), use_container_width=True, hide_index=True)
    st.info("This tool supports prioritisation. It should not be used as the sole reason to deny service, change pricing, or make an irreversible customer decision.")


st.markdown('<div class="footer">Churn Risk Advisor · Applied AI Project 1 · Model outputs should be reviewed alongside customer context.</div>', unsafe_allow_html=True)

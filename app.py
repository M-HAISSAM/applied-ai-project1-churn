from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Churn Risk Advisor",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Visual system: aurora backdrop + frosted glass surfaces
# -----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

:root{
  --ink:#e8edff; --muted:#9aa6d1; --faint:#6b789f;
  --glass:rgba(255,255,255,.065); --glass-strong:rgba(255,255,255,.11);
  --edge:rgba(255,255,255,.15); --edge-soft:rgba(255,255,255,.08);
  --mint:#5ef2c0; --amber:#ffc766; --rose:#ff5c7a; --violet:#8b7bff;
  --radius:22px;
}
html, body, .stApp, [class*="css"]{font-family:'Figtree',system-ui,sans-serif;}
.stApp{
  color:var(--ink);
  background:
    radial-gradient(60rem 38rem at 8% -6%, rgba(94,242,192,.20), transparent 60%),
    radial-gradient(52rem 40rem at 96% 4%, rgba(139,123,255,.30), transparent 58%),
    radial-gradient(46rem 34rem at 50% 108%, rgba(255,92,122,.16), transparent 60%),
    #070b1a;
  background-attachment:fixed;
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"], footer, #MainMenu{display:none !important;}
.block-container{max-width:1240px; padding:2rem 1.4rem 4rem;}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif; letter-spacing:-.02em; color:var(--ink);}

/* masthead */
.mast{display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:2.2rem;}
.brand{display:flex; align-items:center; gap:.8rem;}
.logo{width:2.6rem; height:2.6rem; border-radius:50%;
  background:conic-gradient(from 210deg, var(--mint), var(--violet), var(--rose), var(--mint));
  box-shadow:0 0 28px rgba(139,123,255,.55); position:relative;}
.logo::after{content:""; position:absolute; inset:.55rem; border-radius:50%; background:#0b1026;
  box-shadow:inset 0 0 0 1px var(--edge);}
.brand b{font-family:'Bricolage Grotesque'; font-size:1.15rem; font-weight:700;}
.brand span{display:block; font-size:.78rem; color:var(--muted);}
.live{display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .85rem; border-radius:999px;
  background:var(--glass); border:1px solid var(--edge); backdrop-filter:blur(16px);
  font-size:.78rem; font-weight:600; color:var(--ink); white-space:nowrap;}
.live i{width:.5rem; height:.5rem; border-radius:50%; background:var(--mint); box-shadow:0 0 10px var(--mint);}
.hero h1{font-size:clamp(2rem,4.6vw,3.6rem); font-weight:800; line-height:1.02; margin:0 0 .8rem; max-width:16ch;}
.hero p{color:var(--muted); font-size:1.05rem; max-width:56ch; line-height:1.55; margin:0 0 1.8rem;}

/* glass surfaces */
.glass, div[data-testid="stForm"], [data-testid="stMetric"], .lever{
  background:linear-gradient(145deg, var(--glass-strong), var(--glass));
  border:1px solid var(--edge); border-radius:var(--radius);
  backdrop-filter:blur(24px) saturate(150%); -webkit-backdrop-filter:blur(24px) saturate(150%);
  box-shadow:0 18px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.18);
}
div[data-testid="stForm"]{padding:1.4rem 1.5rem .8rem;}
[data-testid="stMetric"]{padding:1rem 1.1rem;}
[data-testid="stMetricLabel"] p{color:var(--muted); font-size:.82rem;}
[data-testid="stMetricValue"]{font-family:'Bricolage Grotesque'; font-weight:700;}
.glass{padding:1.5rem;}
.cap{color:var(--muted); font-size:.88rem; line-height:1.5;}
.h-sec{font-family:'Bricolage Grotesque'; font-size:1.25rem; font-weight:700; margin:1.8rem 0 .2rem;}

/* tabs as a glass segmented control */
.stTabs [data-baseweb="tab-list"]{gap:.3rem; padding:.3rem; width:fit-content; max-width:100%; overflow-x:auto;
  background:var(--glass); border:1px solid var(--edge); border-radius:999px; backdrop-filter:blur(18px); margin-bottom:1.6rem;}
.stTabs [data-baseweb="tab"]{height:2.4rem; padding:0 1.1rem; border-radius:999px; color:var(--muted); font-weight:600; background:transparent;}
.stTabs [aria-selected="true"]{background:rgba(255,255,255,.14); color:var(--ink); box-shadow:inset 0 1px 0 rgba(255,255,255,.2);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}

/* inputs */
label, [data-testid="stWidgetLabel"] p{color:var(--muted) !important; font-size:.82rem !important; font-weight:600;}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"]{
  background:rgba(255,255,255,.05) !important; border-color:var(--edge-soft) !important; border-radius:14px !important;}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"]:focus-within{border-color:rgba(94,242,192,.6) !important;}
[data-baseweb="popover"] ul{background:#10163a !important; border:1px solid var(--edge); border-radius:14px;}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button{
  border-radius:14px; font-weight:700; border:1px solid var(--edge); min-height:3rem;
  background:var(--glass-strong); color:var(--ink); transition:transform .15s, box-shadow .15s;}
[data-testid="stFormSubmitButton"] > button[kind="primary"], .stDownloadButton > button[kind="primary"]{
  background:linear-gradient(120deg, var(--mint), #7fb7ff 55%, var(--violet)); color:#06101f; border:none;
  box-shadow:0 10px 30px rgba(94,242,192,.28);}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover{transform:translateY(-1px);}
[data-testid="stFileUploaderDropzone"]{background:var(--glass); border:1.5px dashed var(--edge); border-radius:var(--radius);}
[data-testid="stDataFrame"]{border-radius:16px; overflow:hidden; border:1px solid var(--edge-soft);}

/* result: the gauge */
.result{display:grid; grid-template-columns:minmax(260px,380px) 1fr; gap:1.6rem; align-items:center;}
.gauge{width:100%; max-width:380px; margin:0 auto; display:block; overflow:visible;}
.arc{fill:none; stroke-width:16; stroke-linecap:round;}
.arc-track{stroke:rgba(255,255,255,.10);}
.arc-fill{stroke-dasharray:282.74; animation:sweep 1.3s cubic-bezier(.2,.8,.2,1) both;}
@keyframes sweep{from{stroke-dashoffset:282.74;} to{stroke-dashoffset:var(--off);}}
.g-num{font-family:'Bricolage Grotesque'; font-weight:800; font-size:34px; fill:#e8edff; text-anchor:middle;}
.g-sub{font-size:9.5px; fill:#9aa6d1; text-anchor:middle;}
.tag{display:inline-block; padding:.28rem .75rem; border-radius:999px; font-size:.78rem; font-weight:700; border:1px solid currentColor;}
.t-high{color:var(--rose);} .t-med{color:var(--amber);} .t-low{color:var(--mint);}
.verdict{font-family:'Bricolage Grotesque'; font-size:clamp(1.5rem,3vw,2.1rem); font-weight:800; line-height:1.1; margin:.7rem 0 .5rem;}

/* levers */
.lever{display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.95rem 1.2rem; margin-bottom:.7rem; border-radius:18px;}
.lever b{display:block; font-weight:600;} .lever small{color:var(--muted);}
.delta{font-family:'Bricolage Grotesque'; font-weight:700; font-size:1.2rem; white-space:nowrap;}
.d-good{color:var(--mint);} .d-bad{color:var(--rose);}
.foot{margin-top:3rem; color:var(--faint); font-size:.78rem; text-align:center;}

@media (max-width:820px){
  .block-container{padding:1.2rem .9rem 3rem;}
  .mast{flex-direction:column; align-items:flex-start;}
  .result{grid-template-columns:1fr;}
  .lever{flex-direction:column; align-items:flex-start;}
}
@media (prefers-reduced-motion:reduce){ .arc-fill{animation:none;} .stButton > button{transition:none;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model loading and prediction (unchanged contract with the training notebook)
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
    bundle = joblib.load(MODEL_PATH)
    meta: dict[str, Any] = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return bundle, meta


def unpack_bundle(bundle: Any) -> tuple[Any, Any, list[str]]:
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
        fallback = (pd.to_numeric(out.get("tenure", 0), errors="coerce").fillna(0)
                    * pd.to_numeric(out.get("MonthlyCharges", 0), errors="coerce").fillna(0))
        out["TotalCharges"] = out["TotalCharges"].fillna(fallback)
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
    model, encoder, cols = unpack_bundle(bundle)
    features = encode_for_model(raw, encoder, cols)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    return np.asarray(model.predict(features), dtype=float)


def meta_float(meta: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if meta.get(key) is not None:
            try:
                return float(meta[key])
            except (TypeError, ValueError):
                pass
    for value in meta.values():
        if isinstance(value, dict):
            found = meta_float(value, keys, float("nan"))
            if found == found:
                return found
    return default


THRESHOLD_KEYS = ("threshold", "selected_threshold", "chosen_threshold", "cost_based_threshold",
                  "cost_optimal_threshold", "optimal_threshold", "decision_threshold")
VALUE_KEYS = ("retention_value", "avoided_loss", "value_saved_per_churner", "cost_false_negative")


def money(v: float) -> str:
    return f"PKR {v:,.0f}"


try:
    bundle, metadata = load_artifacts()
    model_object, _, _ = unpack_bundle(bundle)
    THRESHOLD = meta_float(metadata, THRESHOLD_KEYS, 0.5)
    SAVED_VALUE = meta_float(metadata, VALUE_KEYS, 6000.0)
    ready, err = True, ""
except Exception as exc:
    bundle, metadata, model_object, THRESHOLD, SAVED_VALUE = None, {}, None, 0.5, 6000.0
    ready, err = False, str(exc)


def risk_tier(p: float) -> tuple[str, str, str, str, str]:
    """level, css class, title, copy, colour"""
    if p >= max(THRESHOLD, 0.70):
        return ("High risk", "t-high", "Reach out this week",
                "This customer sits well above your decision threshold. Review the account and prepare a retention offer.",
                "#ff5c7a")
    if p >= THRESHOLD:
        return ("Elevated risk", "t-med", "Worth a targeted call",
                "Above the threshold. Check recent interactions before choosing an offer.", "#ffc766")
    return ("Low risk", "t-low", "No action needed",
            "Below the threshold. Keep serving them as usual and re-check if their plan changes.", "#5ef2c0")


def gauge_svg(p: float, color: str) -> str:
    import math
    length = math.pi * 90
    off = length * (1 - min(max(p, 0), 1))
    th = math.pi * (1 - min(max(THRESHOLD, 0), 1))
    x1, y1 = 110 + 76 * math.cos(th), 112 - 76 * math.sin(th)
    x2, y2 = 110 + 105 * math.cos(th), 112 - 105 * math.sin(th)
    return (
        f'<svg class="gauge" viewBox="0 0 220 138" role="img" aria-label="Churn probability {p:.0%}">'
        f'<defs><filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        f'<path class="arc arc-track" d="M20 112 A90 90 0 0 1 200 112"/>'
        f'<path class="arc arc-fill" style="--off:{off:.2f}" stroke="{color}" filter="url(#glow)" d="M20 112 A90 90 0 0 1 200 112"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e8edff" stroke-width="2.5" stroke-linecap="round"/>'
        f'<text class="g-num" x="110" y="104">{p:.0%}</text>'
        f'<text class="g-sub" x="110" y="124">chance of leaving</text></svg>'
    )


# -----------------------------------------------------------------------------
# Masthead
# -----------------------------------------------------------------------------
status = "Model online" if ready else "Model unavailable"
st.markdown(
    f'<div class="mast"><div class="brand"><div class="logo"></div>'
    f'<div><b>Churn Risk Advisor</b><span>Customer retention decision support</span></div></div>'
    f'<div class="live"><i></i>{status}</div></div>'
    f'<div class="hero"><h1>Know who is leaving before they do.</h1>'
    f'<p>Describe a customer and see their churn risk, then test which changes to their plan would keep them.</p></div>',
    unsafe_allow_html=True,
)

if not ready:
    st.error("The model could not be loaded. Confirm churn_model.joblib and model_meta.json are in the repository root.")
    st.caption(err)
    st.stop()

tab_one, tab_batch, tab_model = st.tabs(["Assess a customer", "Score a file", "About the model"])

# -----------------------------------------------------------------------------
# Tab 1: single customer
# -----------------------------------------------------------------------------
YNI = ["No", "Yes", "No internet service"]

with tab_one:
    with st.form("customer_form"):
        st.markdown("#### Customer")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = c3.selectbox("Partner", ["Yes", "No"])
        dependents = c4.selectbox("Dependents", ["No", "Yes"])

        st.markdown("#### Plan")
        p1, p2, p3, p4 = st.columns(4)
        tenure = p1.number_input("Tenure (months)", 0, 72, 12, 1)
        contract = p2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = p3.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        phone = p4.selectbox("Phone service", ["Yes", "No"])

        st.markdown("#### Add-ons")
        a1, a2, a3, a4 = st.columns(4)
        multiple = a1.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        security = a2.selectbox("Online security", YNI)
        backup = a3.selectbox("Online backup", YNI)
        protection = a4.selectbox("Device protection", YNI)
        a5, a6, a7, a8 = st.columns(4)
        tech = a5.selectbox("Tech support", YNI)
        stream_tv = a6.selectbox("Streaming TV", YNI)
        stream_movies = a7.selectbox("Streaming movies", YNI)
        paperless = a8.selectbox("Paperless billing", ["Yes", "No"])

        st.markdown("#### Billing")
        b1, b2 = st.columns(2)
        payment = b1.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = b2.number_input("Monthly charges (PKR)", 0.0, 500.0, 70.0, 1.0)
        submitted = st.form_submit_button("Check churn risk", type="primary", use_container_width=True)

    if submitted:
        st.session_state["row"] = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech, "StreamingTV": stream_tv, "StreamingMovies": stream_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": float(tenure) * float(monthly),
        }

    row = st.session_state.get("row")
    if row:
        prob = float(predict_probability(pd.DataFrame([row]), bundle)[0])
        level, tag_cls, title, copy, color = risk_tier(prob)
        outreach = prob >= THRESHOLD

        st.markdown('<div class="h-sec">Result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="glass result">{gauge_svg(prob, color)}<div>'
            f'<span class="tag {tag_cls}">{level}</span>'
            f'<div class="verdict">{title}</div><div class="cap">{copy}</div></div></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision threshold", f"{THRESHOLD:.0%}")
        m2.metric("Model decision", "Reach out" if outreach else "Monitor")
        m3.metric("Value protected if retained", money(SAVED_VALUE if outreach else 0))

        # What-if levers: re-score the same customer with one change at a time.
        levers: list[tuple[str, str, dict[str, Any]]] = []
        if row["Contract"] == "Month-to-month":
            levers += [("Move to a one-year contract", "Contract", {"Contract": "One year"}),
                       ("Move to a two-year contract", "Contract", {"Contract": "Two year"})]
        if row["InternetService"] != "No":
            if row["TechSupport"] == "No":
                levers.append(("Add tech support", "Services", {"TechSupport": "Yes"}))
            if row["OnlineSecurity"] == "No":
                levers.append(("Add online security", "Services", {"OnlineSecurity": "Yes"}))
        if row["PaymentMethod"] in ("Electronic check", "Mailed check"):
            levers.append(("Switch to automatic payment", "Billing", {"PaymentMethod": "Bank transfer (automatic)"}))

        if levers:
            variants = pd.DataFrame([{**row, **chg} for _, _, chg in levers])
            new_p = predict_probability(variants, bundle)
            ranked = sorted(zip(levers, new_p), key=lambda t: t[1])
            st.markdown('<div class="h-sec">What could change the outcome</div>', unsafe_allow_html=True)
            st.markdown('<div class="cap" style="margin-bottom:.9rem">Each option re-scores this customer with one change. These are model estimates, not promises.</div>', unsafe_allow_html=True)
            html = ""
            for (name, area, _), p_new in ranked:
                d = p_new - prob
                cls = "d-good" if d < 0 else "d-bad"
                html += (f'<div class="lever"><div><b>{name}</b><small>{area} · now {prob:.0%}, would be {p_new:.0%}</small></div>'
                         f'<div class="delta {cls}">{d*100:+.1f} pts</div></div>')
            st.markdown(html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 2: batch scoring
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="cap" style="margin-bottom:1rem">Upload a CSV with the same columns as the training data. You get every customer scored and ranked, ready to download.</div>', unsafe_allow_html=True)
    template = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 840.0,
    }])
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"), "churn_template.csv", "text/csv")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"That file could not be read as a CSV: {exc}")
            st.stop()
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f"Add {len(missing)} missing column(s) and upload again: {', '.join(missing)}")
        else:
            probs = predict_probability(data[RAW_COLS], bundle)
            scored = data.copy()
            scored["churn_probability"] = probs
            scored["recommended_action"] = np.where(probs >= THRESHOLD, "Reach out", "Monitor")
            scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Customers scored", f"{len(scored):,}")
            k2.metric("Above threshold", f"{int((probs >= THRESHOLD).sum()):,}")
            k3.metric("Average risk", f"{probs.mean():.1%}")
            bands = pd.cut(scored["churn_probability"], [-0.001, .25, .5, .75, 1.0],
                           labels=["0-25%", "25-50%", "50-75%", "75-100%"]).value_counts().sort_index()
            st.markdown('<div class="h-sec">Customers by risk band</div>', unsafe_allow_html=True)
            st.bar_chart(bands, color="#5ef2c0", height=240)
            st.markdown('<div class="h-sec">Highest risk first</div>', unsafe_allow_html=True)
            st.dataframe(scored.head(100), use_container_width=True, hide_index=True,
                         column_config={"churn_probability": st.column_config.ProgressColumn(
                             "Churn probability", min_value=0.0, max_value=1.0, format="percent")})
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"),
                               "churn_scored_customers.csv", "text/csv", type="primary")

# -----------------------------------------------------------------------------
# Tab 3: model details
# -----------------------------------------------------------------------------
with tab_model:
    d1, d2, d3 = st.columns(3)
    d1.metric("Decision threshold", f"{THRESHOLD:.1%}")
    d2.metric("Model type", type(model_object).__name__)
    d3.metric("Input fields", len(RAW_COLS))
    st.markdown('<div class="h-sec">Model card</div>', unsafe_allow_html=True)
    flat = {str(k).replace("_", " ").capitalize(): str(v) for k, v in metadata.items() if not isinstance(v, (dict, list))}
    if flat:
        st.dataframe(pd.DataFrame(list(flat.items()), columns=["Item", "Value"]), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="cap">No model metadata found.</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass cap" style="margin-top:1.2rem">This tool helps prioritise outreach. Do not use it as the only reason to deny service, change pricing, or make a decision that cannot be undone.</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">Churn Risk Advisor · Applied AI Project 1 · Review model outputs alongside customer context.</div>', unsafe_allow_html=True)

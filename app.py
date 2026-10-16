from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

import dataset_lab
from theme import SCALE, inject, kpi, panel, show, sty, topbar

try:
    import xgboost as xgb
except ImportError:  # only needed when the saved model is XGBoost
    xgb = None

st.set_page_config(page_title="Churn Risk Advisor | Mission Control", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

inject()

# ------------------------------------------------------------------ workspace switch: the churn advisor or the any-dataset lab
ROOT = Path(__file__).resolve().parent
with st.sidebar:
    st.markdown("## Workspace")
    mode = st.radio('Workspace', ['📊 Churn Risk Advisor', '🧪 Dataset Lab (any data)'], key='mode', label_visibility='collapsed')
    st.divider()
if mode.startswith('🧪'):
    dataset_lab.render(str(ROOT))
    st.stop()

# ------------------------------------------------------------------ constants
RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
NUM_COLS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CAT_COLS = [c for c in RAW_COLS if c not in NUM_COLS]
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
CONTRACTS = ['Month-to-month', 'One year', 'Two year']
INTERNET = ['Fiber optic', 'DSL', 'No']
YN = ['Yes', 'No']
VALID = {'gender': ['Female', 'Male'], 'Partner': YN, 'Dependents': YN, 'PhoneService': YN,
         'MultipleLines': ['No', 'Yes', 'No phone service'], 'InternetService': INTERNET,
         **{s: ['No', 'Yes', 'No internet service'] for s in SERVICES},
         'Contract': CONTRACTS, 'PaperlessBilling': YN, 'PaymentMethod': PAYMENTS}
FN_COST, FP_COST = 6000, 1000  # PKR, from the Week 2 cost analysis


# ------------------------------------------------------------------ model + serving encoder (lab Task 1.2)
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(ROOT / 'churn_model.joblib')
    mp = ROOT / 'model_meta.json'
    meta = json.loads(mp.read_text(encoding='utf-8')) if mp.exists() else {}
    return model, meta


try:
    model, meta = load_artifacts()
except Exception as exc:
    st.error("The model file could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.caption("Check that churn_model.joblib sits next to app.py and that requirements.txt pins the same "
               "scikit-learn and xgboost versions used in training.")
    st.stop()

FINAL = model[-1] if hasattr(model, 'steps') else model           # works for a Pipeline or a bare estimator
IS_XGB = hasattr(FINAL, 'get_booster') and xgb is not None
_names = getattr(model, 'feature_names_in_', None)
if _names is not None:
    COLS = [str(c) for c in _names]
elif 'feature_columns' in meta:
    COLS = [str(c) for c in meta['feature_columns']]
else:
    st.error("The model has no feature_names_in_ and model_meta.json has no feature_columns. Re-run Task 1.6.")
    st.stop()
THR0 = float(meta.get('threshold', 0.30))


def prepare_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Parity-tested serving encoder: no drop_first, reindexed to the exact training columns."""
    r = raw[RAW_COLS].copy()
    r[NUM_COLS] = r[NUM_COLS].apply(pd.to_numeric, errors='coerce')  # a numeric column must never be one-hot encoded
    X = pd.get_dummies(r)
    return X.reindex(columns=COLS, fill_value=0).astype(float)


def score_X(X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def score(raw: pd.DataFrame) -> np.ndarray:
    return score_X(prepare_input(raw))


def clean_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Make an uploaded or bundled file safe to score, and report what had to be fixed."""
    d = df.copy()
    issues: dict[str, int] = {}
    for c in CAT_COLS:
        d[c] = d[c].astype(str).str.strip()
    d['SeniorCitizen'] = d['SeniorCitizen'].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    for c in NUM_COLS:
        before = d[c].isna().sum()
        d[c] = pd.to_numeric(d[c], errors='coerce')
        bad = int(d[c].isna().sum() - before)
        if bad:
            issues[f'{c}: not a number'] = bad
    blank_tc = int(d['TotalCharges'].isna().sum())
    if blank_tc:
        issues['TotalCharges: blank, derived as tenure x monthly'] = blank_tc
    d['TotalCharges'] = d['TotalCharges'].fillna(d['tenure'] * d['MonthlyCharges'])
    d['SeniorCitizen'] = d['SeniorCitizen'].fillna(0).clip(0, 1).astype(int)
    for c, v in VALID.items():
        n = int((~d[c].isin(v)).sum())
        if n:
            issues[f'{c}: unknown category'] = n
    return d, issues


def to_model_space(X: pd.DataFrame):
    return model[:-1].transform(X) if hasattr(model, 'steps') else X


@st.cache_data(show_spinner=False)
def explain(rj: str) -> tuple[pd.Series | None, str]:
    X = prepare_input(pd.DataFrame([json.loads(rj)]))
    if IS_XGB:
        Z = pd.DataFrame(np.asarray(to_model_space(X), dtype=float), columns=COLS)
        c = FINAL.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
        return pd.Series(c[0, :-1], index=COLS), 'XGBoost tree contributions to the log-odds of churn'
    if hasattr(FINAL, 'coef_'):
        z = np.asarray(to_model_space(X), dtype=float)[0]
        return pd.Series(FINAL.coef_[0] * z, index=COLS), 'coefficient x scaled value (log-odds of churn)'
    return None, ''


def global_importance() -> pd.Series | None:
    if hasattr(FINAL, 'feature_importances_'):
        return pd.Series(FINAL.feature_importances_, index=COLS)
    if hasattr(FINAL, 'coef_'):
        return pd.Series(np.abs(FINAL.coef_[0]), index=COLS)
    return None


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame | None:
    p = ROOT / 'sample_customers.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if not all(c in df.columns for c in RAW_COLS):
        return None
    return clean_raw(df)[0]


@st.cache_data(show_spinner=False)
def row_parity(df: pd.DataFrame) -> float:
    """Score every row on its own and in one batch. Any gap means the encoder depends on what else is in the file."""
    batch = score(df)
    single = np.array([score(df.iloc[[i]])[0] for i in range(len(df))])
    return float(np.abs(batch - single).max())


def baseline_categories() -> dict[str, list[str]]:
    """Training dropped one category per field (drop_first). Each field must have exactly one."""
    return {c: [v for v in vals if f'{c}_{v}' not in COLS] for c, vals in VALID.items()}


def mfmt(key: str, f: str = '{:.3f}') -> str:
    v = meta.get(key)
    return f.format(v) if isinstance(v, (int, float)) else '—'


# ------------------------------------------------------------------ risk bands
def band_of(p: float, thr: float) -> tuple[str, str]:
    return ('HIGH', '#f43f5e') if p >= thr else ('WATCH', '#f59e0b') if p >= thr / 2 else ('LOW', '#10b981')


# ------------------------------------------------------------------ demo customers (Part 5: prepare the demo in advance)
DEFAULTS = dict(tenure=4, contract='Month-to-month', monthly=85.0, internet='Fiber optic', services=['StreamingTV'],
                phone='Yes', multi='No', payment='Electronic check', paperless='Yes', gender='Female',
                senior=False, partner=False, dependents=False)
PRESETS = {
    'Lab default (Task 2.3)': DEFAULTS,
    'High-risk demo': {**DEFAULTS, 'tenure': 2, 'monthly': 95.0, 'services': ['StreamingTV', 'StreamingMovies'],
                       'senior': True, 'multi': 'Yes'},
    'Loyal two-year customer': {**DEFAULTS, 'tenure': 60, 'contract': 'Two year', 'monthly': 65.0, 'internet': 'DSL',
                                'services': ['OnlineSecurity', 'OnlineBackup', 'TechSupport'],
                                'payment': 'Credit card (automatic)', 'paperless': 'No', 'partner': True, 'dependents': True},
    'Phone-only customer': {**DEFAULTS, 'tenure': 24, 'contract': 'One year', 'monthly': 20.0, 'internet': 'No',
                            'services': [], 'payment': 'Mailed check', 'paperless': 'No'},
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(f'w_{k}', list(v) if isinstance(v, list) else v)


def apply_preset() -> None:
    for k, v in PRESETS[st.session_state['preset']].items():
        st.session_state[f'w_{k}'] = list(v) if isinstance(v, list) else v


# ------------------------------------------------------------------ sidebar: one customer (lab Task 2.2)
with st.sidebar:
    st.markdown("## Customer profile")
    st.selectbox('Load a demo customer', list(PRESETS), key='preset', on_change=apply_preset)
    tenure = st.slider('Tenure (months)', 0, 72, key='w_tenure')
    contract = st.selectbox('Contract', CONTRACTS, key='w_contract')
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, step=0.5, key='w_monthly')
    internet = st.selectbox('Internet service', INTERNET, key='w_internet')
    services: list[str] = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES, key='w_services')
    phone = st.radio('Phone service', YN, horizontal=True, key='w_phone')
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True, key='w_multi')
    payment = st.selectbox('Payment method', PAYMENTS, key='w_payment')
    paperless = st.radio('Paperless billing', YN, horizontal=True, key='w_paperless')
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True, key='w_gender')
        senior = st.checkbox('Senior citizen', key='w_senior')
        partner = st.checkbox('Has partner', key='w_partner')
        dependents = st.checkbox('Has dependents', key='w_dependents')
    # valid-input logic: combinations that never occur in the training data
    if phone == 'No' and internet == 'No':
        st.warning('No phone and no internet means no service at all. The model never saw such a customer, so treat this score as meaningless.')
    elif internet == 'No' and monthly > 30:
        st.caption(f'Note: phone-only customers in the training data pay roughly 18 to 27 USD. {monthly:.0f} USD is outside that range.')
    elif phone == 'No' and internet != 'No' and monthly > 70:
        st.caption('Note: internet-only customers rarely pay above 70 USD in the training data.')
    st.caption(f'Total charges are derived: {tenure} x {monthly:.2f} = {tenure * monthly:,.2f} USD.')
    st.divider()
    threshold = st.slider('Contact threshold', 0.05, 0.95, float(min(max(round(THR0 * 20) / 20, .05), .95)), 0.05)
    st.caption("Moving the threshold changes the band and action. The probability stays the same.")

row = {'gender': gender, 'SeniorCitizen': int(senior), 'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': float(monthly), 'TotalCharges': float(tenure * monthly)}  # derived, never typed
for s in SERVICES:
    row[s] = 'No internet service' if internet == 'No' else 'Yes' if s in services else 'No'
customer = pd.DataFrame([row])
rj = json.dumps(row)
p = float(score(customer)[0])
band, bcol = band_of(p, threshold)
action = 'Contact now' if band == 'HIGH' else 'Monitor closely' if band == 'WATCH' else 'No action'

# ------------------------------------------------------------------ sample file + system checks (used by the top bar)
sample = load_sample()
sample_p = score(sample) if sample is not None else None
sdf = sample.assign(p_churn=sample_p) if sample is not None else None

ver_ok = meta.get('sklearn_version') in (None, sklearn.__version__)
xgb_ver = xgb.__version__ if xgb is not None else 'not installed'
xgb_ok = (not IS_XGB) or meta.get('xgboost_version') in (None, xgb_ver)
bases = baseline_categories()
enc_ok = all(len(v) == 1 for v in bases.values())
par_gap = row_parity(sample) if sample is not None else None
par_ok = par_gap is not None and par_gap < 1e-9
CHECKS = [('Model loaded', True, type(FINAL).__name__ + (' inside a Pipeline' if hasattr(model, 'steps') else '')),
          ('Feature columns match model', len(COLS) == int(getattr(model, 'n_features_in_', len(COLS))), f'{len(COLS)} columns'),
          ('scikit-learn matches training', ver_ok, f"runtime {sklearn.__version__}, trained {meta.get('sklearn_version', 'not recorded')}"),
          ('xgboost matches training', xgb_ok, f"runtime {xgb_ver}, trained {meta.get('xgboost_version', 'not recorded')}"),
          ('Encoder covers every training category', enc_ok, 'one baseline category per field' if enc_ok else
           'fields without exactly one baseline: ' + ', '.join(c for c, v in bases.items() if len(v) != 1)),
          ('Row-by-row parity on sample', par_ok, f'largest gap {par_gap:.1e}' if par_gap is not None else 'needs sample_customers.csv'),
          ('model_meta.json found', bool(meta), 'metrics and threshold' if meta else f'using default threshold {THR0:.2f}'),
          ('Sample file found', sample is not None, 'sample_customers.csv')]
critical_ok = all(ok for name, ok, _ in CHECKS[:6])

# ------------------------------------------------------------------ top bar
now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
topbar('Churn Risk Advisor', 'Retention mission control',
       [('STATUS', 'NOMINAL' if critical_ok else 'CHECK', 'ok' if critical_ok else 'bad'), ('MODEL', type(FINAL).__name__, ''),
        ('FEATURES', str(len(COLS)), ''), ('THRESHOLD', f'{threshold:.2f}', ''),
        ('PARITY', 'PASS' if par_ok else 'N/A' if par_gap is None else 'FAIL', 'ok' if par_ok else 'bad'),
        ('SKLEARN', sklearn.__version__, 'ok' if ver_ok else 'bad'), (now, '', '')])
if not critical_ok:
    st.warning('One or more system checks failed. Open the Model & checks tab before trusting any score.')

TAB = dict(zip(["ov", "wi", "rm", "ex", "sg", "rp", "fl", "rs", "md", "gd"], st.tabs(
    ["🏠 Overview", "🎛️ What-if lab", "🗺️ Risk maps", "🔎 Explainability", "👥 Segments", "💰 Retention planner", "📂 Fleet scoring",
     "📝 Customer report", "✅ Model & checks", "📘 Guide"])))


# ------------------------------------------------------------------ cached model-driven grids
@st.cache_data(show_spinner=False)
def grid_tm(rj: str, ctr: str):
    r = json.loads(rj)
    T, M = np.arange(0, 73, 4), np.arange(20, 121, 5)
    df = pd.DataFrame([{**r, 'Contract': ctr, 'tenure': int(t), 'MonthlyCharges': float(m), 'TotalCharges': float(t * m)} for m in M for t in T])
    return T, M, score(df).reshape(len(M), len(T))


@st.cache_data(show_spinner=False)
def grid_two(rj: str, a: str, av: tuple, b: str, bv: tuple):
    r = json.loads(rj)
    df = pd.DataFrame([{**r, a: x, b: y} for y in bv for x in av])
    if 'InternetService' in (a, b):
        df = df.assign(**{s: np.where(df['InternetService'] == 'No', 'No internet service',
                                      np.where(df[s] == 'No internet service', 'No', df[s])) for s in SERVICES})
    return score(df).reshape(len(bv), len(av))


@st.cache_data(show_spinner=False)
def curves(rj: str):
    r = json.loads(rj)
    T = np.arange(0, 73, 2)
    out = {}
    for c in CONTRACTS:
        out[c] = score(pd.DataFrame([{**r, 'Contract': c, 'tenure': int(t), 'TotalCharges': float(t * r['MonthlyCharges'])} for t in T]))
    return T, out


# ================================================================== OVERVIEW
with TAB['ov']:
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi('Churn probability', f'{p:.0%}', 'model estimate', bcol)
    with c2: kpi('Risk band', band, f'HIGH at {threshold:.0%}, WATCH at {threshold/2:.0%}', bcol)
    with c3: kpi('Recommended action', action, 'decision support only', bcol)
    with c4: kpi('Expected loss if ignored', f'PKR {p*FN_COST:,.0f}', f'{p:.0%} x PKR {FN_COST:,} per lost customer', '#f59e0b')
    left, right = st.columns([1, 1.25])
    with left, st.container(border=True):
        panel('Risk gauge')
        g = go.Figure(go.Indicator(mode='gauge+number', value=p * 100,
                                   number=dict(suffix='%', valueformat='.0f', font=dict(family='JetBrains Mono', size=46, color='#eef2ff')),
                                   gauge=dict(axis=dict(range=[0, 100], tickcolor='#a5b1cb'), bar=dict(color='#eef2ff', thickness=.16), bgcolor='rgba(0,0,0,0)', borderwidth=0,
                                              steps=[dict(range=[0, threshold * 50], color='rgba(5,150,105,.35)'), dict(range=[threshold * 50, threshold * 100], color='rgba(217,119,6,.35)'),
                                                     dict(range=[threshold * 100, 100], color='rgba(220,38,38,.38)')],
                                              threshold=dict(line=dict(color='#eef2ff', width=3), thickness=.9, value=threshold * 100))))
        show(sty(g, 300), 'ov_gauge')
        if sample_p is not None:
            pct = float((sample_p < p).mean())
            st.markdown(f"<div class='note'>Riskier than <b>{pct:.0%}</b> of the {len(sample_p)} customers in the sample file.</div>", unsafe_allow_html=True)
    with right, st.container(border=True):
        panel('Churn curve by contract, this customer profile')
        T, cv = curves(rj)
        fig = go.Figure()
        for (c, y), col in zip(cv.items(), ['#f43f5e', '#f59e0b', '#10b981']):
            fig.add_trace(go.Scatter(x=T, y=y * 100, name=c, mode='lines', line=dict(color=col, width=3)))
        fig.add_trace(go.Scatter(x=[tenure], y=[p * 100], name='This customer', mode='markers', marker=dict(size=15, color='#eef2ff', line=dict(color=bcol, width=3))))
        fig.add_hline(y=threshold * 100, line=dict(color='#8b5cf6', dash='dot'), annotation_text='threshold')
        show(sty(fig, 340, xaxis_title='Tenure (months)', yaxis_title='Churn probability (%)'), 'ov_curves')
    st.markdown("<div class='note'>Charts are model estimates from patterns in one dataset. They describe association, not guaranteed effects.</div>", unsafe_allow_html=True)

# ================================================================== WHAT-IF
changes = [('Contract', c) for c in ['One year', 'Two year'] if c != contract]
changes += [('PaymentMethod', m) for m in ['Bank transfer (automatic)', 'Credit card (automatic)']
            if payment in ('Electronic check', 'Mailed check') and m != payment]
if internet != 'No':
    changes += [(s, 'Yes') for s in ['TechSupport', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection'] if s not in services]
if paperless == 'Yes':
    changes.append(('PaperlessBilling', 'No'))
rows = []
for f, v in changes:
    alt = customer.copy(); alt[f] = v
    q = float(score(alt)[0])
    rows.append({'Change': f'{f}: {v}', 'New probability': q, 'Difference': q - p})
wi = pd.DataFrame(rows).sort_values('Difference').reset_index(drop=True) if rows else pd.DataFrame()


def greedy_plan(max_steps: int = 3) -> list[tuple[str, float]]:
    """Stack the single best change, then the next best given the first, and so on. One value per field."""
    cur, cur_p, used, steps = customer.copy(), p, set(), []
    for _ in range(max_steps):
        best = None
        for f, v in changes:
            if f in used:
                continue
            alt = cur.copy(); alt[f] = v
            q = float(score(alt)[0])
            if q < cur_p - 1e-6 and (best is None or q < best[2]):
                best = (f, v, q)
        if best is None:
            break
        cur[best[0]] = best[1]; used.add(best[0]); cur_p = best[2]
        steps.append((f'{best[0]}: {best[1]}', cur_p))
    return steps


with TAB['wi']:
    a, b = st.columns([1.3, 1])
    with a, st.container(border=True):
        panel('Change in churn probability, one change at a time')
        if wi.empty:
            st.info('No what-if options apply to this profile.')
        else:
            colors = ['#10b981' if d < 0 else '#f43f5e' for d in wi['Difference']]
            fig = go.Figure(go.Bar(x=wi['Difference'] * 100, y=wi['Change'], orientation='h', marker_color=colors,
                                   text=[f'{d*100:+.1f} pts' for d in wi['Difference']], textposition='outside', cliponaxis=False))
            fig.update_yaxes(autorange='reversed')
            show(sty(fig, 120 + 38 * len(wi), xaxis_title='Change in probability (percentage points)'), 'wi_bars')
    with b, st.container(border=True):
        panel('Scenario table')
        if not wi.empty:
            st.dataframe(wi.assign(**{'New probability': (wi['New probability'] * 100).round(1), 'Difference': (wi['Difference'] * 100).round(1)})
                         .rename(columns={'New probability': 'New %', 'Difference': 'Diff (pts)'}), hide_index=True)
    with st.container(border=True):
        panel('Stacked plan: best change first, then the next best on top of it')
        plan = greedy_plan()
        if not plan:
            st.info('No change, alone or combined, lowers the score for this profile.')
        else:
            labels = ['Today'] + [s for s, _ in plan] + ['After plan']
            vals = [p] + [q for _, q in plan]
            deltas = [p * 100] + [(vals[i + 1] - vals[i]) * 100 for i in range(len(plan))] + [vals[-1] * 100]
            fig = go.Figure(go.Waterfall(x=labels, y=deltas, measure=['absolute'] + ['relative'] * len(plan) + ['total'],
                                         text=[f'{d:+.1f}' if 0 < i < len(labels) - 1 else f'{d:.0f}%' for i, d in enumerate(deltas)],
                                         textposition='outside', connector=dict(line=dict(color='rgba(148,163,184,.4)')),
                                         decreasing=dict(marker=dict(color='#10b981')), increasing=dict(marker=dict(color='#f43f5e')),
                                         totals=dict(marker=dict(color='#8b5cf6'))))
            fig.add_hline(y=threshold * 100, line=dict(color='#f43f5e', dash='dot'), annotation_text='threshold')
            show(sty(fig, 340, yaxis_title='Churn probability (%)', showlegend=False), 'wi_waterfall')
            st.markdown(f"<div class='note'>Combined, these {len(plan)} changes move the score from <b>{p:.0%}</b> to <b>{plan[-1][1]:.0%}</b>. "
                        "Changes interact, so the combined drop is not the sum of the single drops above.</div>", unsafe_allow_html=True)
    st.caption('Associations learned from data, not guaranteed effects. A lower score does not prove that making the change '
               'would keep the customer. Only a controlled retention test can show that.')

# ================================================================== RISK MAPS
with TAB['rm']:
    with st.container(border=True):
        panel('Risk surface: tenure x monthly charges (all other fields as in the sidebar)')
        ctr = st.radio('Contract for the map', CONTRACTS, index=CONTRACTS.index(contract), horizontal=True)
        T, M, Z = grid_tm(rj, ctr)
        fig = go.Figure(go.Heatmap(x=T, y=M, z=Z * 100, colorscale=SCALE, zmin=0, zmax=100, colorbar=dict(title='%'),
                                   hovertemplate='tenure %{x} mo<br>$%{y}/mo<br>risk %{z:.0f}%<extra></extra>'))
        fig.add_trace(go.Contour(x=T, y=M, z=Z * 100, contours=dict(start=threshold * 100, end=threshold * 100, size=1, coloring='none'),
                                 line=dict(color='#eef2ff', width=2, dash='dash'), showscale=False, hoverinfo='skip', name='threshold'))
        fig.add_trace(go.Scatter(x=[tenure], y=[monthly], mode='markers', name='This customer', marker=dict(size=15, color='#eef2ff', symbol='x', line=dict(width=2))))
        show(sty(fig, 420, xaxis_title='Tenure (months)', yaxis_title='Monthly charges (USD)'), 'rm_surface')
        st.caption('The dashed line is the contact threshold: customers above and to the left of it would be contacted.')
    h1, h2 = st.columns(2)
    with h1, st.container(border=True):
        panel('Contract x payment method')
        Z2 = grid_two(rj, 'Contract', tuple(CONTRACTS), 'PaymentMethod', tuple(PAYMENTS))
        fig = go.Figure(go.Heatmap(x=CONTRACTS, y=PAYMENTS, z=Z2 * 100, colorscale=SCALE, zmin=0, zmax=100, text=np.round(Z2 * 100).astype(int), texttemplate='%{text}%'))
        show(sty(fig, 320), 'rm_pay')
    with h2, st.container(border=True):
        panel('Contract x internet service')
        Z3 = grid_two(rj, 'Contract', tuple(CONTRACTS), 'InternetService', tuple(INTERNET))
        fig = go.Figure(go.Heatmap(x=CONTRACTS, y=INTERNET, z=Z3 * 100, colorscale=SCALE, zmin=0, zmax=100, text=np.round(Z3 * 100).astype(int), texttemplate='%{text}%'))
        show(sty(fig, 320), 'rm_net')

# ================================================================== EXPLAINABILITY
cs, cs_label = explain(rj)
with TAB['ex']:
    e1, e2 = st.columns(2)
    with e1, st.container(border=True):
        panel('Why this score: top drivers for this customer')
        if cs is None:
            st.info(f'Per-customer explanations are available for XGBoost and Logistic Regression, not {type(FINAL).__name__}.')
        else:
            top = cs.reindex(cs.abs().sort_values(ascending=False).index).head(10)
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation='h', marker_color=['#f43f5e' if v > 0 else '#10b981' for v in top.values]))
            fig.update_yaxes(autorange='reversed')
            show(sty(fig, 400, xaxis_title='Contribution to log-odds of churn'), 'ex_local')
            st.caption(f'Red pushes toward churn, green pushes away. Values: {cs_label}. A field set to its baseline category '
                       '(for example Month-to-month) shows up as the absence of the other categories, such as Contract_Two year = 0.')
    with e2, st.container(border=True):
        panel('What the model relies on overall')
        gi = global_importance()
        if gi is None:
            st.info('This model type does not expose feature importances.')
        else:
            imp = gi.sort_values().tail(12)
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h', marker=dict(color=imp.values, colorscale=SCALE)))
            show(sty(fig, 400, xaxis_title='Relative importance'), 'ex_global')
            st.caption('Importance shows what the model uses, not what causes churn.')

# ================================================================== FLEET SCORING (runs before Segments and Planner so they can reuse it)
pool, pool_name = sdf, 'sample file'
with TAB['fl']:
    src = st.radio('Data source', ['Bundled sample file', 'Upload a CSV'], horizontal=True)
    data = None
    if src == 'Upload a CSV':
        f = st.file_uploader('Customer file', type='csv')
        st.caption('Needs the original Telco columns: ' + ', '.join(RAW_COLS) + '. customerID is optional.')
        if f is not None:
            try:
                data = pd.read_csv(f)
            except Exception as exc:
                st.error(f'This file could not be read as CSV: {exc}')
    else:
        data = pd.read_csv(ROOT / 'sample_customers.csv') if (ROOT / 'sample_customers.csv').exists() else None
        if data is None:
            st.warning('sample_customers.csv was not found next to app.py.')
    if data is not None:
        miss = [c for c in RAW_COLS if c not in data.columns]
        if miss:
            st.error(f'Missing columns: {miss}. Add them and upload again; nothing was scored.')
        elif data.empty:
            st.warning('The file has headers but no customer rows.')
        else:
            d, issues = clean_raw(data)
            if issues:
                with st.expander(f'⚠️ {sum(issues.values())} value(s) needed attention. Unknown categories are scored as the baseline category.', expanded=False):
                    st.dataframe(pd.DataFrame({'Issue': list(issues), 'Rows': list(issues.values())}), hide_index=True)
            d['p_churn'] = score(d)
            d['band'] = [band_of(x, threshold)[0] for x in d['p_churn']]
            d['contact'] = np.where(d['p_churn'] >= threshold, 'Yes', '')
            if src == 'Upload a CSV':
                pool, pool_name = d, f'uploaded file ({len(d)} rows)'
            n_hi = int((d['contact'] == 'Yes').sum())
            k1, k2, k3, k4 = st.columns(4)
            with k1: kpi('Customers scored', f'{len(d):,}', 'rows in file')
            with k2: kpi('Above threshold', f'{n_hi:,}', f'{n_hi/len(d):.0%} of file', '#f43f5e')
            with k3: kpi('Average risk', f'{d.p_churn.mean():.1%}', 'mean probability', '#f59e0b')
            with k4: kpi('Expected loss', f'PKR {d.p_churn.sum()*FN_COST:,.0f}', f'sum of p x PKR {FN_COST:,}', '#f59e0b')
            r1, r2 = st.columns([1.4, 1])
            with r1, st.container(border=True):
                panel('Distribution of churn probability')
                fig = go.Figure(go.Histogram(x=d.p_churn * 100, xbins=dict(start=0, end=100, size=5), marker=dict(color='#8b5cf6', line=dict(color='#0b1020', width=1))))
                fig.add_vline(x=threshold * 100, line=dict(color='#f43f5e', dash='dash'), annotation_text='threshold')
                show(sty(fig, 320, xaxis_title='Churn probability (%)', yaxis_title='Customers'), 'fl_hist')
            with r2, st.container(border=True):
                panel('Customers by band')
                cnt = d['band'].value_counts().reindex(['HIGH', 'WATCH', 'LOW']).fillna(0)
                fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.62, marker=dict(colors=['#f43f5e', '#f59e0b', '#10b981']), sort=False))
                show(sty(fig, 320), 'fl_pie')
            r3, r4 = st.columns(2)
            with r3, st.container(border=True):
                panel('Tenure vs monthly charges, colored by risk')
                fig = go.Figure(go.Scatter(x=d.tenure, y=d.MonthlyCharges, mode='markers', text=d.get('customerID'),
                                           marker=dict(size=11, color=d.p_churn * 100, colorscale=SCALE, cmin=0, cmax=100, colorbar=dict(title='%'), line=dict(width=1, color='#0b1020'))))
                show(sty(fig, 340, xaxis_title='Tenure (months)', yaxis_title='Monthly charges (USD)'), 'fl_scatter')
            with r4, st.container(border=True):
                panel('Average risk: contract x internet service')
                pv = d.pivot_table(index='InternetService', columns='Contract', values='p_churn', aggfunc='mean').reindex(index=INTERNET, columns=CONTRACTS)
                ct = d.pivot_table(index='InternetService', columns='Contract', values='p_churn', aggfunc='count').reindex(index=INTERNET, columns=CONTRACTS).fillna(0).astype(int)
                txt = [[f'{v*100:.0f}% (n={n})' if v == v else '' for v, n in zip(rv, rn)] for rv, rn in zip(pv.values, ct.values)]
                fig = go.Figure(go.Heatmap(x=CONTRACTS, y=INTERNET, z=pv.values * 100, colorscale=SCALE, zmin=0, zmax=100, text=txt, texttemplate='%{text}'))
                show(sty(fig, 340), 'fl_heat')
            with st.container(border=True):
                panel('Highest-risk customers')
                show_bands = st.multiselect('Show bands', ['HIGH', 'WATCH', 'LOW'], default=['HIGH', 'WATCH', 'LOW'])
                cols = [c for c in ['customerID', 'tenure', 'Contract', 'InternetService', 'PaymentMethod', 'MonthlyCharges'] if c in d.columns]
                view = d[d['band'].isin(show_bands)].sort_values('p_churn', ascending=False).head(50)
                tbl = view[cols + ['p_churn', 'band', 'contact']].assign(risk_pct=lambda x: (x.p_churn * 100).round(1))
                st.dataframe(tbl.drop(columns='p_churn'), hide_index=True,
                             column_config={'risk_pct': st.column_config.ProgressColumn('Risk', min_value=0, max_value=100, format='%.0f%%')})
            out = d.assign(p_churn=d['p_churn'].round(4))
            dl1, dl2 = st.columns(2)
            dl1.download_button('Download scored CSV', out.to_csv(index=False), 'scored_customers.csv', 'text/csv', type='primary')
            dl2.download_button(f'Download call list ({n_hi} customers)', out[out['contact'] == 'Yes'].sort_values('p_churn', ascending=False).to_csv(index=False),
                                'call_list.csv', 'text/csv', disabled=n_hi == 0)

pool_p = pool['p_churn'].to_numpy() if pool is not None else None

# ================================================================== SEGMENTS
with TAB['sg']:
    if pool is None:
        st.info('sample_customers.csv (or an uploaded file in Fleet scoring) is needed for segment analysis.')
    else:
        st.caption(f'Source: {pool_name}. Upload a file in Fleet scoring to analyse it here.')
        g = pool.assign(Tenure_band=pd.cut(pool.tenure, [-1, 12, 24, 48, 72], labels=['0-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']).astype(str),
                        Senior=pool.SeniorCitizen.map({0: 'No', 1: 'Yes'}))
        seg = g.groupby(['Contract', 'InternetService']).p_churn.agg(['mean', 'count']).reset_index()
        seg = seg[seg['count'] >= 3].sort_values('mean', ascending=False)
        if not seg.empty:
            t = seg.iloc[0]
            st.markdown(f"<div class='kpi' style='--c:#f43f5e'><span>Riskiest named segment (n of at least 3)</span><b style='font-size:1.4rem'>{t.Contract} + {t.InternetService}</b>"
                        f"<em>Average risk {t['mean']:.0%} across {int(t['count'])} customers. Suggested test: offer a one-year contract incentive or an autopay switch to a small group and compare with an untouched group.</em></div>", unsafe_allow_html=True)
        dims = ['Contract', 'InternetService', 'PaymentMethod', 'Tenure_band', 'Senior', 'PaperlessBilling']
        for chunk in (dims[:3], dims[3:]):
            for col, dim in zip(st.columns(3), chunk):
                with col, st.container(border=True):
                    panel(f'Average risk by {dim}')
                    a = g.groupby(dim).p_churn.agg(['mean', 'count']).reset_index().sort_values('mean')
                    fig = go.Figure(go.Bar(x=a['mean'] * 100, y=a[dim].astype(str), orientation='h', marker=dict(color=a['mean'] * 100, colorscale=SCALE, cmin=0, cmax=100),
                                           text=[f"{m*100:.0f}%  (n={c})" for m, c in zip(a['mean'], a['count'])], textposition='outside', cliponaxis=False))
                    fig.update_xaxes(range=[0, 115])
                    show(sty(fig, 270), f'sg_{dim}')
        with st.container(border=True):
            panel('Portfolio map: size is customers, colour is average risk')
            tm = px.treemap(g, path=[px.Constant('All customers'), 'Contract', 'InternetService'], color='p_churn', color_continuous_scale=SCALE, range_color=(0, 1))
            tm.update_traces(textinfo='label+value', textfont=dict(size=15))
            show(sty(tm, 380), 'sg_tree')
        st.caption(f'Based on {len(g)} customers. Small groups are noisy: treat these as leads to test, not conclusions.')

# ================================================================== RETENTION PLANNER
with TAB['rp']:
    if pool_p is None:
        st.info('sample_customers.csv (or an uploaded file in Fleet scoring) is needed for the planner.')
    else:
        a, b, c = st.columns(3)
        succ = a.slider('Offer success rate (assumption)', 0.05, 0.80, 0.30, 0.05)
        ocost = b.number_input('Cost of one retention offer (PKR)', 0, 20000, FP_COST, 100)
        val = c.number_input('Value of a retained customer (PKR)', 0, 100000, FN_COST, 500)
        ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
        out = []
        for t in ths:
            m = pool_p >= t
            out.append(dict(t=t, cost=float(((1 - pool_p[m]) * ocost).sum() + (pool_p[~m] * val).sum()),
                            net=float((pool_p[m] * succ * val).sum() - m.sum() * ocost), n=int(m.sum()),
                            caught=float(pool_p[m].sum() / max(pool_p.sum(), 1e-9))))
        o = pd.DataFrame(out)
        cur = o.iloc[(o.t - threshold).abs().argmin()]
        best = o.loc[o.net.idxmax()]
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi('Customers contacted', f'{int(cur.n)}', f'of {len(pool_p)} at threshold {threshold:.2f}')
        with k2: kpi('Offer spend', f'PKR {cur.n*ocost:,.0f}', f'expected churners reached: {cur.caught:.0%}', '#f59e0b')
        with k3: kpi('Expected net value', f'PKR {cur.net:,.0f}', 'saved value minus spend', '#10b981' if cur.net >= 0 else '#f43f5e')
        with k4: kpi('Best threshold', f'{best.t:.2f}', f'net PKR {best.net:,.0f}', '#8b5cf6')
        p1, p2 = st.columns(2)
        with p1, st.container(border=True):
            panel('Expected net value by threshold')
            fig = go.Figure(go.Scatter(x=o.t, y=o.net, mode='lines+markers', line=dict(color='#8b5cf6', width=3), fill='tozeroy', fillcolor='rgba(139,92,246,.18)'))
            fig.add_vline(x=threshold, line=dict(color='#eef2ff', dash='dot'), annotation_text='your threshold')
            show(sty(fig, 340, xaxis_title='Contact threshold', yaxis_title='PKR'), 'rp_net')
        with p2, st.container(border=True):
            panel('Expected misclassification cost by threshold')
            fig = go.Figure(go.Scatter(x=o.t, y=o.cost, mode='lines+markers', name='Expected cost', line=dict(color='#f59e0b', width=3)))
            mn = o.loc[o.cost.idxmin()]
            fig.add_trace(go.Scatter(x=[mn.t], y=[mn.cost], mode='markers', name=f'Minimum at {mn.t:.2f}', marker=dict(size=14, color='#10b981')))
            fig.add_vline(x=threshold, line=dict(color='#eef2ff', dash='dot'))
            show(sty(fig, 340, xaxis_title='Contact threshold', yaxis_title='PKR'), 'rp_cost')
        st.caption(f'Planning estimate on the {pool_name} ({len(pool_p)} customers), using model probabilities and the assumptions above. '
                   'Cost curve: each contacted non-churner costs one offer, each missed churner costs the customer value. '
                   'The success rate is a guess until a retention test measures it.')

# ================================================================== CUSTOMER REPORT
with TAB['rs']:
    bw = wi.iloc[0] if (not wi.empty and wi.iloc[0]['Difference'] < 0) else None
    lines = [f"CHURN RISK REPORT   {now}", "", f"Churn probability : {p:.1%}", f"Risk band         : {band} (threshold {threshold:.0%})",
             f"Recommended action: {action}", f"Expected loss     : PKR {p*FN_COST:,.0f}", ""]
    if cs is not None:
        top5 = cs.reindex(cs.abs().sort_values(ascending=False).index).head(5)
        lines += ["Main drivers (log-odds):"] + [f"  {'+' if v > 0 else '-'} {k}  ({v:+.2f})" for k, v in top5.items()] + [""]
    lines += [f"Best what-if      : {bw['Change']} -> {bw['New probability']:.0%} ({bw['Difference']*100:+.1f} pts)" if bw is not None else "Best what-if      : none lowers risk"]
    plan = greedy_plan()
    if len(plan) > 1:
        lines += [f"Stacked plan      : {' + '.join(s for s, _ in plan)} -> {plan[-1][1]:.0%}"]
    lines += ["", f"Model             : {type(FINAL).__name__} v{meta.get('version', '1.0')}, CV AUC {mfmt('cv_auc')} +/- {mfmt('cv_auc_std')}",
              "", "Profile: " + ', '.join(f'{k}={v}' for k, v in row.items()), "",
              "Decision support only. Associations, not causes. Review with the retention team."]
    r1, r2 = st.columns([1.2, 1])
    with r1, st.container(border=True):
        panel('Report')
        st.markdown(f"<div class='rep'>{chr(10).join(lines)}</div>", unsafe_allow_html=True)
        st.download_button('Download report (.txt)', '\n'.join(lines), 'customer_risk_report.txt', 'text/plain', type='primary')
    with r2, st.container(border=True):
        panel('Customer profile')
        st.dataframe(customer.T.rename(columns={0: 'Value'}).astype(str), height=520)

# ================================================================== MODEL & CHECKS
with TAB['md']:
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi('CV AUC', mfmt('cv_auc'), f"± {mfmt('cv_auc_std')}")
    with m2: kpi('Test AUC', mfmt('test_auc'), 'test set used once')
    with m3: kpi('Threshold', f'{THR0:.2f}', f'FN PKR {FN_COST:,}, FP PKR {FP_COST:,}', '#f59e0b')
    with m4: kpi('Features', str(len(COLS)), 'encoded columns', '#10b981')
    k1, k2 = st.columns(2)
    with k1, st.container(border=True):
        panel('System checks')
        for name, ok, det in CHECKS:
            st.markdown(f"{'🟢' if ok else '🟠'} **{name}** <span class='note'>{det}</span>", unsafe_allow_html=True)
    with k2, st.container(border=True):
        panel('Model card')
        st.markdown(f"**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.\n\n"
                    f"**Not for:** credit, pricing, or any decision that denies a service.\n\n"
                    f"**Data:** {meta.get('training_data', 'IBM Telco Customer Churn')}.\n\n"
                    f"**Model:** {type(FINAL).__name__}, {len(COLS)} encoded features. **Version:** {meta.get('version', '1.0')}.\n\n"
                    f"**Limitations:** one US dataset from one period, no Pakistan data, associations not causes, performance may drift as plans change.")
    with st.container(border=True):
        panel('Skew lab: this customer scored the Week 3 way vs the serving way (Task 1.3, live)')
        X_right = prepare_input(customer)
        X_wrong = pd.get_dummies(customer[RAW_COLS], drop_first=True).reindex(columns=COLS, fill_value=0).astype(float)
        p_wrong = float(score_X(X_wrong)[0])
        s1, s2, s3 = st.columns(3)
        with s1: kpi('Serving encoder', f'{p:.1%}', f'{int(X_right.drop(columns=[c for c in NUM_COLS if c in COLS]).iloc[0].sum())} category columns switched on', '#10b981')
        with s2: kpi('drop_first=True on one row', f'{p_wrong:.1%}', f'{int(X_wrong.drop(columns=[c for c in NUM_COLS if c in COLS]).iloc[0].sum())} category columns switched on', '#f43f5e')
        with s3: kpi('Silent error', f'{(p_wrong - p) * 100:+.1f} pts', 'no exception is raised', '#f59e0b')
        diff = pd.DataFrame({'serving': X_right.iloc[0], 'drop_first': X_wrong.iloc[0]})
        diff = diff[diff.serving != diff.drop_first]
        if not diff.empty:
            st.dataframe(diff.T.astype(int), hide_index=False)
        st.caption('With one row, every field has only one category, so drop_first drops it and the row collapses to every baseline '
                   '(Month-to-month, DSL, no add-ons...). Pick a two-year DSL customer with add-ons in the sidebar to see the gap grow.')
    with st.expander('Feature columns and baseline categories'):
        st.write(COLS)
        st.dataframe(pd.DataFrame({'Field': list(bases), 'Baseline (all dummies = 0)': [', '.join(v) for v in bases.values()]}), hide_index=True)

# ================================================================== GUIDE
with TAB['gd']:
    g1, g2 = st.columns(2)
    with g1, st.container(border=True):
        panel('How to use this dashboard')
        st.markdown("1. Describe a customer in the **sidebar**, or load a demo customer. Everything updates at once.\n2. Read the **Overview** for probability, band and action.\n"
                    "3. Use **What-if lab** to test plan changes on the same customer, one at a time and stacked.\n"
                    "4. Check **Explainability** to see what drives the score.\n5. Use **Fleet scoring** for many customers; an uploaded file also feeds **Segments** and **Retention planner**.\n"
                    "6. Save a one-page summary from **Customer report**.\n7. **Model & checks** proves the app scores exactly like the notebook.\n"
                    "8. Switch the sidebar **Workspace** to **Dataset Lab** to upload *any* CSV/Excel file, explore it, and train and compare "
                    "up to 8 ML algorithms on it (or cluster it with K-Means).")
        st.markdown(f"**Bands:** HIGH at or above {threshold:.0%}, WATCH from {threshold/2:.0%} to {threshold:.0%}, LOW below that. The threshold changes the band, never the probability.")
    with g2, st.container(border=True):
        panel('Glossary and limits')
        st.markdown("**Churn probability:** the model's estimate that this customer leaves.\n\n**Threshold:** the probability above which the team contacts a customer, set from the cost of missed churners versus wasted offers.\n\n"
                    "**Expected loss:** probability times the value of a lost customer.\n\n**What-if:** the model re-scored with one field changed. It shows association, not a proven effect.\n\n"
                    "**Parity:** the same customer gets the same score alone, in a batch, and in the notebook.\n\n"
                    "**Limits:** trained on one US telecom dataset. Not for credit, pricing, or any decision that denies a service.")

st.markdown("<div class='note' style='text-align:center;margin-top:1.5rem'>Churn Risk Advisor · Applied AI Project 1 · Decision support only. Review with the retention team before contacting a customer.</div>", unsafe_allow_html=True)

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Churn Risk Advisor | Mission Control", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root{--bg:#05070d;--ink:#e6edf7;--mut:#8ea0bd;--line:rgba(120,160,220,.16);--cy:#4fd1ff;--red:#fc3d21;--amb:#ffb84d;--grn:#3ddc97;--blu:#0b3d91}
html,body,.stApp{font-family:'Space Grotesk',system-ui,sans-serif}
.stApp{color:var(--ink);background:radial-gradient(70rem 40rem at 85% -10%,rgba(11,61,145,.45),transparent 60%),radial-gradient(50rem 30rem at -5% 100%,rgba(79,209,255,.10),transparent 60%),
linear-gradient(rgba(120,160,220,.045) 1px,transparent 1px) 0 0/44px 44px,linear-gradient(90deg,rgba(120,160,220,.045) 1px,transparent 1px) 0 0/44px 44px,#05070d;background-attachment:fixed}
[data-testid="stHeader"]{background:transparent}[data-testid="stToolbar"],footer,#MainMenu{display:none!important}
.block-container{max-width:1500px;padding:1rem 1.4rem 3rem}
h1,h2,h3,h4{font-family:'Space Grotesk';letter-spacing:-.01em}
[data-testid="stSidebar"]{background:rgba(8,14,28,.92);border-right:1px solid var(--line);backdrop-filter:blur(18px)}
[data-testid="stSidebar"] h2{font-size:1rem;letter-spacing:.08em;text-transform:uppercase;color:var(--cy)}
label,[data-testid="stWidgetLabel"] p{color:var(--mut)!important;font-size:.8rem!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"]{background:rgba(255,255,255,.04)!important;border-color:var(--line)!important;border-radius:10px!important}
[data-baseweb="popover"] ul{background:#0b1220!important}
.top{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding:.9rem 1.2rem;margin-bottom:1rem;border:1px solid var(--line);border-radius:16px;
background:linear-gradient(135deg,rgba(255,255,255,.07),rgba(255,255,255,.02));backdrop-filter:blur(20px);box-shadow:0 10px 40px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.1)}
.brand{display:flex;align-items:center;gap:.9rem}
.emb{width:2.7rem;height:2.7rem;border-radius:50%;background:radial-gradient(circle at 35% 30%,#4fd1ff,#0b3d91 60%,#05070d);position:relative;box-shadow:0 0 22px rgba(79,209,255,.5)}
.emb::before{content:"";position:absolute;inset:-5px -10px;border:1.5px solid var(--red);border-radius:50%;transform:rotate(-24deg)}
.brand b{font-size:1.2rem;font-weight:700}.brand small{display:block;color:var(--mut);font-family:'JetBrains Mono';font-size:.68rem;letter-spacing:.12em;text-transform:uppercase}
.chips{display:flex;gap:.5rem;flex-wrap:wrap}
.chip{font-family:'JetBrains Mono';font-size:.7rem;padding:.35rem .65rem;border-radius:8px;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--mut)}
.chip b{color:var(--ink);font-weight:600}.chip.ok b{color:var(--grn)}
.kpi{border:1px solid var(--line);border-radius:14px;padding:.85rem 1rem;background:linear-gradient(145deg,rgba(255,255,255,.07),rgba(255,255,255,.02));backdrop-filter:blur(16px);
box-shadow:inset 0 1px 0 rgba(255,255,255,.09);border-top:2px solid var(--c,var(--cy));margin-bottom:.6rem}
.kpi span{display:block;font-family:'JetBrains Mono';font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
.kpi b{display:block;font-family:'JetBrains Mono';font-size:1.7rem;font-weight:700;margin:.15rem 0;color:var(--c,var(--ink))}
.kpi em{font-style:normal;font-size:.76rem;color:var(--mut)}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;border-radius:16px!important;background:rgba(255,255,255,.03);backdrop-filter:blur(14px);padding:.3rem .4rem}
.ph{font-family:'JetBrains Mono';font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--cy);margin:.2rem 0 .1rem}
.note{color:var(--mut);font-size:.8rem;line-height:1.5}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;overflow-x:auto;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono';font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);padding:.6rem .9rem}
.stTabs [aria-selected="true"]{color:var(--cy)}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:600;border:1px solid var(--line)}
.stDownloadButton>button[kind="primary"]{background:linear-gradient(120deg,#0b3d91,#1f7ae0);border:none;color:#fff}
@media(max-width:820px){.block-container{padding:.6rem .7rem 2rem}.top{padding:.7rem}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ model + serving encoder (lab Task 1.2)
ROOT = Path(__file__).resolve().parent
RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
CONTRACTS = ['Month-to-month', 'One year', 'Two year']
INTERNET = ['Fiber optic', 'DSL', 'No']
VALID = {'gender': ['Female', 'Male'], 'Partner': ['Yes', 'No'], 'Dependents': ['Yes', 'No'], 'Contract': CONTRACTS,
         'InternetService': INTERNET, 'PaymentMethod': PAYMENTS, 'PaperlessBilling': ['Yes', 'No'],
         'PhoneService': ['Yes', 'No']}
FN_COST, FP_COST = 6000, 1000  # PKR, from the Week 2 cost analysis


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
    st.caption("Check that churn_model.joblib sits next to app.py and that requirements.txt pins the same scikit-learn and xgboost versions used in training.")
    st.stop()

COLS = [str(c) for c in (getattr(model, 'feature_names_in_', None) if getattr(model, 'feature_names_in_', None) is not None else meta['feature_columns'])]
THR0 = float(meta.get('threshold', 0.30))


def prepare_input(raw: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(raw[RAW_COLS])  # no drop_first at serving: this is the parity-tested encoder
    return X.reindex(columns=COLS, fill_value=0).astype(float)


def score(raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_input(raw))[:, 1]


def contributions(raw: pd.DataFrame) -> pd.Series:
    X = prepare_input(raw)
    c = model.get_booster().predict(xgb.DMatrix(X, feature_names=COLS), pred_contribs=True)
    return pd.Series(c[0, :-1], index=COLS)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame | None:
    p = ROOT / 'sample_customers.csv'
    return pd.read_csv(p) if p.exists() else None


def mfmt(key: str, f: str = '{:.3f}') -> str:
    v = meta.get(key)
    return f.format(v) if isinstance(v, (int, float)) else '—'


# ------------------------------------------------------------------ chart helpers
SCALE = [[0, '#0b2a4a'], [.3, '#1f7ae0'], [.55, '#4fd1ff'], [.75, '#ffb84d'], [1, '#fc3d21']]


def sty(fig: go.Figure, h: int = 330, **kw) -> go.Figure:
    fig.update_layout(height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Space Grotesk', color='#c3d0e6', size=12),
                      margin=dict(l=8, r=8, t=34, b=8), title_font=dict(size=13, color='#4fd1ff'),
                      legend=dict(orientation='h', y=-0.18), **kw)
    fig.update_xaxes(gridcolor='rgba(120,160,220,.10)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(120,160,220,.10)', zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={'displayModeBar': False})


def kpi(label: str, value: str, sub: str = '', color: str = '#4fd1ff') -> None:
    st.markdown(f"<div class='kpi' style='--c:{color}'><span>{label}</span><b>{value}</b><em>{sub}</em></div>", unsafe_allow_html=True)


def panel(title: str) -> None:
    st.markdown(f"<div class='ph'>{title}</div>", unsafe_allow_html=True)


def band_of(p: float, thr: float) -> tuple[str, str]:
    return ('HIGH', '#fc3d21') if p >= thr else ('WATCH', '#ffb84d') if p >= thr / 2 else ('LOW', '#3ddc97')


# ------------------------------------------------------------------ sidebar: one customer (lab Task 2.2)
with st.sidebar:
    st.markdown("## Customer profile")
    tenure = st.slider('Tenure (months)', 0, 72, 4)
    contract = st.selectbox('Contract', CONTRACTS)
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, 85.0, step=0.5)
    internet = st.selectbox('Internet service', INTERNET)
    services: list[str] = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES, default=['StreamingTV'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True)
    payment = st.selectbox('Payment method', PAYMENTS)
    paperless = st.radio('Paperless billing', ['Yes', 'No'], horizontal=True)
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        senior = st.checkbox('Senior citizen')
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')
    st.divider()
    threshold = st.slider('Contact threshold', 0.05, 0.95, float(min(max(THR0, .05), .95)), 0.05)
    st.caption("Moving the threshold changes the band and action. The probability stays the same.")

row = {'gender': gender, 'SeniorCitizen': int(senior), 'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': monthly, 'TotalCharges': tenure * monthly}  # derived, never typed
for s in SERVICES:
    row[s] = 'No internet service' if internet == 'No' else 'Yes' if s in services else 'No'
customer = pd.DataFrame([row])
p = float(score(customer)[0])
band, bcol = band_of(p, threshold)

# ------------------------------------------------------------------ top bar
now = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
ver_ok = meta.get('sklearn_version') in (None, sklearn.__version__)
st.markdown(
    f"<div class='top'><div class='brand'><div class='emb'></div><div><b>Churn Risk Advisor</b><small>Retention mission control</small></div></div>"
    f"<div class='chips'><div class='chip ok'>STATUS <b>NOMINAL</b></div><div class='chip'>MODEL <b>{type(model).__name__}</b></div>"
    f"<div class='chip'>FEATURES <b>{len(COLS)}</b></div><div class='chip'>THRESHOLD <b>{threshold:.2f}</b></div>"
    f"<div class='chip {'ok' if ver_ok else ''}'>SKLEARN <b>{sklearn.__version__}</b></div><div class='chip'>{now}</div></div></div>",
    unsafe_allow_html=True)

tabs = st.tabs(["Overview", "What-if lab", "Risk maps", "Explainability", "Fleet scoring", "Model & checks"])


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
        df = df.assign(**{s: np.where(df['InternetService'] == 'No', 'No internet service', np.where(df[s] == 'No internet service', 'No', df[s])) for s in SERVICES})
    return score(df).reshape(len(bv), len(av))


@st.cache_data(show_spinner=False)
def curves(rj: str):
    r = json.loads(rj)
    T = np.arange(0, 73, 2)
    out = {}
    for c in CONTRACTS:
        out[c] = score(pd.DataFrame([{**r, 'Contract': c, 'tenure': int(t), 'TotalCharges': float(t * r['MonthlyCharges'])} for t in T]))
    return T, out


rj = json.dumps(row)
sample = load_sample()
sample_p = score(sample[RAW_COLS]) if sample is not None and all(c in sample.columns for c in RAW_COLS) else None

# ================================================================== OVERVIEW
with tabs[0]:
    action = 'Contact now' if band == 'HIGH' else 'Monitor closely' if band == 'WATCH' else 'No action'
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi('Churn probability', f'{p:.0%}', 'model estimate', bcol)
    with c2: kpi('Risk band', band, f'HIGH at {threshold:.0%}, WATCH at {threshold/2:.0%}', bcol)
    with c3: kpi('Recommended action', action, 'decision support only', bcol)
    with c4: kpi('Expected loss if ignored', f'PKR {p*FN_COST:,.0f}', f'{p:.0%} x PKR {FN_COST:,} per lost customer', '#ffb84d')
    left, right = st.columns([1, 1.25])
    with left, st.container(border=True):
        panel('Risk gauge')
        g = go.Figure(go.Indicator(mode='gauge+number', value=p * 100,
                                   number=dict(suffix='%', valueformat='.0f', font=dict(family='JetBrains Mono', size=46, color='#fff')),
                                   gauge=dict(axis=dict(range=[0, 100], tickcolor='#6b7c99'), bar=dict(color='#ffffff', thickness=.16), bgcolor='rgba(0,0,0,0)', borderwidth=0,
                                              steps=[dict(range=[0, threshold * 50], color='rgba(61,220,151,.55)'), dict(range=[threshold * 50, threshold * 100], color='rgba(255,184,77,.6)'),
                                                     dict(range=[threshold * 100, 100], color='rgba(252,61,33,.65)')],
                                              threshold=dict(line=dict(color='#fff', width=3), thickness=.9, value=threshold * 100))))
        show(sty(g, 300))
        if sample_p is not None:
            pct = float((sample_p < p).mean())
            st.markdown(f"<div class='note'>Riskier than <b>{pct:.0%}</b> of the {len(sample_p)} customers in the sample file.</div>", unsafe_allow_html=True)
    with right, st.container(border=True):
        panel('Churn curve by contract, this customer profile')
        T, cv = curves(rj)
        fig = go.Figure()
        for (c, y), col in zip(cv.items(), ['#fc3d21', '#ffb84d', '#3ddc97']):
            fig.add_trace(go.Scatter(x=T, y=y * 100, name=c, mode='lines', line=dict(color=col, width=3)))
        fig.add_trace(go.Scatter(x=[tenure], y=[p * 100], name='This customer', mode='markers', marker=dict(size=14, color='#fff', line=dict(color=bcol, width=3))))
        fig.add_hline(y=threshold * 100, line=dict(color='#4fd1ff', dash='dot'), annotation_text='threshold')
        show(sty(fig, 340, xaxis_title='Tenure (months)', yaxis_title='Churn probability (%)'))
    st.markdown("<div class='note'>Charts are model estimates from patterns in one dataset. They describe association, not guaranteed effects.</div>", unsafe_allow_html=True)

# ================================================================== WHAT-IF
with tabs[1]:
    changes = [('Contract', c) for c in ['One year', 'Two year'] if c != contract]
    changes += [('PaymentMethod', m) for m in ['Bank transfer (automatic)', 'Credit card (automatic)'] if payment in ('Electronic check', 'Mailed check') and m != payment]
    if internet != 'No':
        changes += [(s, 'Yes') for s in ['TechSupport', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection'] if s not in services]
    if paperless == 'Yes':
        changes.append(('PaperlessBilling', 'No'))
    rows = []
    for f, v in changes:
        alt = customer.copy(); alt[f] = v
        q = float(score(alt)[0])
        rows.append({'Change': f'{f}: {v}', 'New probability': q, 'Difference': q - p})
    wi = pd.DataFrame(rows).sort_values('Difference') if rows else pd.DataFrame()
    a, b = st.columns([1.3, 1])
    with a, st.container(border=True):
        panel('Change in churn probability, one change at a time')
        if wi.empty:
            st.info('No what-if options apply to this profile.')
        else:
            colors = ['#3ddc97' if d < 0 else '#fc3d21' for d in wi['Difference']]
            fig = go.Figure(go.Bar(x=wi['Difference'] * 100, y=wi['Change'], orientation='h', marker_color=colors,
                                   text=[f'{d*100:+.1f} pts' for d in wi['Difference']], textposition='outside'))
            fig.update_yaxes(autorange='reversed')
            show(sty(fig, 120 + 38 * len(wi), xaxis_title='Change in probability (percentage points)'))
    with b, st.container(border=True):
        panel('Scenario table')
        if not wi.empty:
            st.dataframe(wi.assign(**{'New probability': (wi['New probability'] * 100).round(1), 'Difference': (wi['Difference'] * 100).round(1)})
                         .rename(columns={'New probability': 'New %', 'Difference': 'Diff (pts)'}), hide_index=True)
    st.caption('Associations learned from data, not guaranteed effects. A lower score does not prove that making the change would keep the customer. Only a controlled retention test can show that.')

# ================================================================== RISK MAPS
with tabs[2]:
    with st.container(border=True):
        panel('Risk surface: tenure x monthly charges (all other fields as in the sidebar)')
        ctr = st.radio('Contract for the map', CONTRACTS, index=CONTRACTS.index(contract), horizontal=True)
        T, M, Z = grid_tm(rj, ctr)
        fig = go.Figure(go.Heatmap(x=T, y=M, z=Z * 100, colorscale=SCALE, zmin=0, zmax=100, colorbar=dict(title='%'),
                                   hovertemplate='tenure %{x} mo<br>$%{y}/mo<br>risk %{z:.0f}%<extra></extra>'))
        fig.add_trace(go.Scatter(x=[tenure], y=[monthly], mode='markers', name='This customer', marker=dict(size=15, color='#fff', symbol='x', line=dict(width=2))))
        show(sty(fig, 420, xaxis_title='Tenure (months)', yaxis_title='Monthly charges (USD)'))
    h1, h2 = st.columns(2)
    with h1, st.container(border=True):
        panel('Contract x payment method')
        Z2 = grid_two(rj, 'Contract', tuple(CONTRACTS), 'PaymentMethod', tuple(PAYMENTS))
        fig = go.Figure(go.Heatmap(x=CONTRACTS, y=PAYMENTS, z=Z2 * 100, colorscale=SCALE, zmin=0, zmax=100, text=np.round(Z2 * 100).astype(int), texttemplate='%{text}%'))
        show(sty(fig, 320))
    with h2, st.container(border=True):
        panel('Contract x internet service')
        Z3 = grid_two(rj, 'Contract', tuple(CONTRACTS), 'InternetService', tuple(INTERNET))
        fig = go.Figure(go.Heatmap(x=CONTRACTS, y=INTERNET, z=Z3 * 100, colorscale=SCALE, zmin=0, zmax=100, text=np.round(Z3 * 100).astype(int), texttemplate='%{text}%'))
        show(sty(fig, 320))

# ================================================================== EXPLAINABILITY
with tabs[3]:
    e1, e2 = st.columns(2)
    with e1, st.container(border=True):
        panel('Why this score: top drivers for this customer')
        cs = contributions(customer)
        top = cs.reindex(cs.abs().sort_values(ascending=False).index).head(10)
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation='h', marker_color=['#fc3d21' if v > 0 else '#3ddc97' for v in top.values]))
        fig.update_yaxes(autorange='reversed')
        show(sty(fig, 400, xaxis_title='Contribution to log-odds of churn'))
        st.caption('Red pushes toward churn, green pushes away. Values come from XGBoost tree contributions.')
    with e2, st.container(border=True):
        panel('What the model relies on overall')
        imp = pd.Series(model.feature_importances_, index=COLS).sort_values().tail(12)
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h', marker=dict(color=imp.values, colorscale=SCALE)))
        show(sty(fig, 400, xaxis_title='Relative importance'))
        st.caption('Importance shows what the model uses, not what causes churn.')

# ================================================================== FLEET SCORING
with tabs[4]:
    src = st.radio('Data source', ['Bundled sample file', 'Upload a CSV'], horizontal=True)
    data = None
    if src == 'Upload a CSV':
        f = st.file_uploader('Customer file', type='csv')
        if f is not None:
            try:
                data = pd.read_csv(f)
            except Exception as exc:
                st.error(f'This file could not be read as CSV: {exc}')
    else:
        data = sample
        if data is None:
            st.warning('sample_customers.csv was not found next to app.py.')
    if data is not None:
        miss = [c for c in RAW_COLS if c not in data.columns]
        if miss:
            st.error(f'Missing columns: {miss}')
        else:
            d = data.copy()
            for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                d[c] = pd.to_numeric(d[c], errors='coerce')
            d['TotalCharges'] = d['TotalCharges'].fillna(d['tenure'] * d['MonthlyCharges']).fillna(0)
            bad = int(sum((~d[c].isin(v)).sum() for c, v in VALID.items()))
            if bad:
                st.warning(f'{bad} value(s) are outside the categories the model was trained on. Those fields are scored as if absent.')
            d['p_churn'] = score(d)
            d['band'] = [band_of(x, threshold)[0] for x in d['p_churn']]
            d['contact'] = np.where(d['p_churn'] >= threshold, 'Yes', '')
            n_hi = int((d['contact'] == 'Yes').sum())
            k1, k2, k3, k4 = st.columns(4)
            with k1: kpi('Customers scored', f'{len(d):,}', 'rows in file')
            with k2: kpi('Above threshold', f'{n_hi:,}', f'{n_hi/len(d):.0%} of file', '#fc3d21')
            with k3: kpi('Average risk', f'{d.p_churn.mean():.1%}', 'mean probability', '#ffb84d')
            with k4: kpi('Expected loss', f'PKR {d.p_churn.sum()*FN_COST:,.0f}', 'sum of p x PKR 6,000', '#ffb84d')
            r1, r2 = st.columns([1.4, 1])
            with r1, st.container(border=True):
                panel('Distribution of churn probability')
                fig = go.Figure(go.Histogram(x=d.p_churn * 100, nbinsx=20, marker=dict(color='#4fd1ff', line=dict(color='#05070d', width=1))))
                fig.add_vline(x=threshold * 100, line=dict(color='#fc3d21', dash='dash'), annotation_text='threshold')
                show(sty(fig, 320, xaxis_title='Churn probability (%)', yaxis_title='Customers'))
            with r2, st.container(border=True):
                panel('Customers by band')
                cnt = d['band'].value_counts().reindex(['HIGH', 'WATCH', 'LOW']).fillna(0)
                fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.62, marker=dict(colors=['#fc3d21', '#ffb84d', '#3ddc97']), sort=False))
                show(sty(fig, 320))
            r3, r4 = st.columns(2)
            with r3, st.container(border=True):
                panel('Tenure vs monthly charges, colored by risk')
                fig = go.Figure(go.Scatter(x=d.tenure, y=d.MonthlyCharges, mode='markers', text=d.get('customerID'),
                                           marker=dict(size=11, color=d.p_churn * 100, colorscale=SCALE, cmin=0, cmax=100, colorbar=dict(title='%'), line=dict(width=.5, color='#05070d'))))
                show(sty(fig, 340, xaxis_title='Tenure (months)', yaxis_title='Monthly charges (USD)'))
            with r4, st.container(border=True):
                panel('Average risk: contract x internet service')
                pv = d.pivot_table(index='InternetService', columns='Contract', values='p_churn', aggfunc='mean').reindex(index=INTERNET, columns=CONTRACTS)
                ct = d.pivot_table(index='InternetService', columns='Contract', values='p_churn', aggfunc='count').reindex(index=INTERNET, columns=CONTRACTS).fillna(0).astype(int)
                txt = [[f'{v*100:.0f}% (n={n})' if v == v else '' for v, n in zip(rv, rn)] for rv, rn in zip(pv.values, ct.values)]
                fig = go.Figure(go.Heatmap(x=CONTRACTS, y=INTERNET, z=pv.values * 100, colorscale=SCALE, zmin=0, zmax=100, text=txt, texttemplate='%{text}'))
                show(sty(fig, 340))
            with st.container(border=True):
                panel('Highest-risk customers')
                cols = [c for c in ['customerID', 'tenure', 'Contract', 'InternetService', 'PaymentMethod', 'MonthlyCharges'] if c in d.columns]
                tbl = d.sort_values('p_churn', ascending=False).head(50)[cols + ['p_churn', 'band', 'contact']].assign(risk_pct=lambda x: (x.p_churn * 100).round(1))
                st.dataframe(tbl.drop(columns='p_churn'), hide_index=True, column_config={'risk_pct': st.column_config.ProgressColumn('Risk', min_value=0, max_value=100, format='%.0f%%')})
            st.download_button('Download scored CSV', d.drop(columns=['band']).to_csv(index=False), 'scored_customers.csv', 'text/csv', type='primary')

# ================================================================== MODEL & CHECKS
with tabs[5]:
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi('CV AUC', mfmt('cv_auc'), f"± {mfmt('cv_auc_std')}")
    with m2: kpi('Test AUC', mfmt('test_auc'), 'test set used once')
    with m3: kpi('Threshold', f'{THR0:.2f}', f'FN PKR {FN_COST:,}, FP PKR {FP_COST:,}', '#ffb84d')
    with m4: kpi('Features', str(len(COLS)), 'encoded columns', '#3ddc97')
    k1, k2 = st.columns(2)
    with k1, st.container(border=True):
        panel('System checks')
        checks = [('Model loaded', True, type(model).__name__),
                  ('Feature columns match model', len(COLS) == int(getattr(model, 'n_features_in_', len(COLS))), f'{len(COLS)} columns'),
                  ('scikit-learn matches training', ver_ok, f"runtime {sklearn.__version__}, trained {meta.get('sklearn_version', 'not recorded')}"),
                  ('model_meta.json found', bool(meta), 'metrics and threshold' if meta else 'using default threshold 0.30'),
                  ('Sample file found', sample is not None, 'sample_customers.csv')]
        for name, ok, det in checks:
            st.markdown(f"{'🟢' if ok else '🟠'} **{name}** <span class='note'>{det}</span>", unsafe_allow_html=True)
    with k2, st.container(border=True):
        panel('Model card')
        st.markdown(f"**Intended use:** rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation.\n\n"
                    f"**Not for:** credit, pricing, or any decision that denies a service.\n\n"
                    f"**Data:** {meta.get('training_data', 'IBM Telco Customer Churn')}.\n\n"
                    f"**Model:** {type(model).__name__}, {len(COLS)} encoded features. **Version:** {meta.get('version', '1.0')}.\n\n"
                    f"**Limitations:** one US dataset from one period, no Pakistan data, associations not causes, performance may drift as plans change.")
    with st.expander('Feature columns'):
        st.write(COLS)
st.markdown("<div class='note' style='text-align:center;margin-top:1.5rem'>Churn Risk Advisor · Applied AI Project 1 · Decision support only. Review with the retention team before contacting a customer.</div>", unsafe_allow_html=True)

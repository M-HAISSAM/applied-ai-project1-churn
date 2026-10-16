"""Shared look and feel: one dark 'midnight' theme used by every page and every chart."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ palette
BG = '#0b1020'        # page
CARD = '#141c31'      # panels
CARD2 = '#1b2540'     # raised / hover
LINE = 'rgba(148,163,184,.20)'
INK = '#eef2ff'       # main text
MUT = '#a5b1cb'       # secondary text
ACC = '#8b5cf6'       # violet accent
ACC2 = '#22d3ee'      # cyan accent
RED = '#f43f5e'
AMB = '#f59e0b'
GRN = '#10b981'
BLUE = '#60a5fa'

SCALE = [[0, '#10b981'], [.35, '#facc15'], [.65, '#f97316'], [1, '#f43f5e']]     # low -> high risk
SEQ = [[0, '#1b2540'], [.5, '#8b5cf6'], [1, '#22d3ee']]                            # neutral sequential
DIV = [[0, '#f43f5e'], [.5, '#1b2540'], [1, '#22d3ee']]                            # -1 .. +1 (correlations)
PALETTE = ['#8b5cf6', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e', '#60a5fa', '#f472b6', '#a3e635', '#fb923c', '#e879f9']

CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{{--bg:{BG};--card:{CARD};--card2:{CARD2};--ink:{INK};--mut:{MUT};--line:{LINE};--acc:{ACC};--acc2:{ACC2};--red:{RED};--amb:{AMB};--grn:{GRN}}}
html,body,.stApp{{font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
.stApp{{color:var(--ink);background:radial-gradient(70rem 40rem at 95% -10%,rgba(139,92,246,.22),transparent 60%),
  radial-gradient(55rem 34rem at -8% 105%,rgba(34,211,238,.14),transparent 60%),var(--bg);background-attachment:fixed}}
[data-testid="stHeader"]{{background:transparent}}
/* hide the menu and deploy buttons, but NOT the toolbar itself: it holds the button that re-opens a collapsed sidebar */
[data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],[data-testid="stMainMenu"],footer,#MainMenu{{display:none!important}}
[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapsedControl"] button{{background:linear-gradient(135deg,{ACC},#6d28d9)!important;
  color:#fff!important;border-radius:10px!important;box-shadow:0 6px 18px -6px rgba(139,92,246,.9)}}
[data-testid="stExpandSidebarButton"] *,[data-testid="stSidebarCollapsedControl"] svg{{color:#fff!important;fill:#fff!important}}
/* workspace switch on the main page */
.st-key-mode [role="radiogroup"]{{gap:.5rem;flex-wrap:wrap}}
.st-key-mode label{{background:var(--card);border:1px solid rgba(139,92,246,.45);border-radius:12px;padding:.55rem 1rem!important;margin:0!important}}
.st-key-mode label[data-selected="true"],.st-key-mode label:has(input:checked){{background:linear-gradient(135deg,{ACC},#6d28d9);border-color:transparent;box-shadow:0 6px 18px -6px rgba(139,92,246,.8)}}
.st-key-mode label p{{font-weight:700!important;color:#fff!important;font-size:.95rem!important}}
.block-container{{max-width:1560px;padding:3.6rem 1.6rem 3rem}}

/* ---------- text: always readable, whatever theme Streamlit picked */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{{color:var(--ink)!important;letter-spacing:-.02em;font-weight:700}}
.stApp p,.stApp li,.stApp span,.stApp label,[data-testid="stMarkdownContainer"]{{color:var(--ink)}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,.stCaption{{color:var(--mut)!important}}
label,[data-testid="stWidgetLabel"] p{{color:#cbd5f5!important;font-size:.86rem!important;font-weight:600!important}}
.stApp a{{color:var(--acc2)}}
code{{color:#f0abfc!important;background:rgba(139,92,246,.15)!important}}

/* ---------- sidebar */
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#10172b,#0b1020);border-right:1px solid var(--line)}}
[data-testid="stSidebar"] h2{{font-size:.8rem!important;letter-spacing:.16em;text-transform:uppercase;color:var(--acc2)!important;font-weight:800}}
[data-testid="stSidebar"] [role="radiogroup"] label p{{color:var(--ink)!important}}

/* ---------- inputs */
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"],.stNumberInput input,.stTextInput input{{
  background:var(--card2)!important;color:var(--ink)!important;border-radius:10px!important;border-color:rgba(148,163,184,.28)!important}}
div[data-baseweb="select"] span,div[data-baseweb="select"] div{{color:var(--ink)}}
span[data-baseweb="tag"]{{background:rgba(139,92,246,.25)!important;border:1px solid rgba(139,92,246,.6)!important}}
span[data-baseweb="tag"] span{{color:#ede9fe!important}}
ul[role="listbox"],div[data-baseweb="popover"] li{{background:var(--card2)!important;color:var(--ink)!important}}
[data-testid="stFileUploaderDropzone"]{{background:linear-gradient(135deg,rgba(139,92,246,.14),rgba(34,211,238,.10))!important;
  border:2px dashed rgba(139,92,246,.6)!important;border-radius:14px!important}}
[data-testid="stFileUploaderDropzone"] *{{color:var(--ink)!important}}
[data-testid="stExpander"] details{{background:var(--card);border:1px solid var(--line)!important;border-radius:12px}}
[data-testid="stExpander"] summary p{{color:var(--ink)!important;font-weight:600}}

/* ---------- tabs: solid pills, the selected one glows */
.stTabs [role="tablist"]{{gap:.4rem;overflow-x:auto;padding:.35rem;background:var(--card);border:1px solid var(--line);border-radius:14px;margin-bottom:.8rem}}
.stTabs [role="tab"]{{height:auto;padding:.55rem 1rem!important;border-radius:10px;background:transparent;border:none;white-space:nowrap;margin:0!important}}
.stTabs [role="tab"] p{{color:var(--mut)!important;font-size:.86rem!important;font-weight:700!important}}
.stTabs [role="tab"]:hover{{background:var(--card2)}}
.stTabs [role="tab"]:hover p{{color:var(--ink)!important}}
.stTabs [role="tab"][aria-selected="true"]{{background:linear-gradient(135deg,{ACC},#6d28d9)!important;box-shadow:0 6px 18px -6px rgba(139,92,246,.8)}}
.stTabs [role="tab"][aria-selected="true"] p{{color:#fff!important}}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"],.stTabs [role="tablist"]>:not([role="tab"]){{display:none}}

/* ---------- cards */
.top,.kpi{{position:relative;overflow:hidden;background:linear-gradient(160deg,var(--card2) 0%,var(--card) 100%);border:1px solid var(--line);border-radius:16px;
  box-shadow:0 18px 40px -22px rgba(0,0,0,.8)}}
.top{{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding:1rem 1.3rem;margin-bottom:1.1rem;
  border-image:none;box-shadow:0 0 0 1px rgba(139,92,246,.25),0 18px 40px -22px rgba(0,0,0,.8)}}
.brand{{display:flex;align-items:center;gap:.95rem}}
.emb{{width:2.8rem;height:2.8rem;border-radius:14px;background:conic-gradient(from 210deg,{ACC},{ACC2},{GRN},{ACC});box-shadow:0 0 22px rgba(139,92,246,.55);
  display:flex;align-items:center;justify-content:center;font-size:1.35rem}}
.brand b{{font-size:1.35rem;font-weight:800;color:var(--ink);background:linear-gradient(90deg,#fff,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.brand small{{display:block;color:var(--acc2);font-family:'JetBrains Mono';font-size:.68rem;letter-spacing:.16em;text-transform:uppercase}}
.chips{{display:flex;gap:.5rem;flex-wrap:wrap}}
.chip{{font-family:'JetBrains Mono';font-size:.72rem;padding:.4rem .7rem;border-radius:9px;border:1px solid var(--line);background:rgba(11,16,32,.6);color:var(--mut)}}
.chip b{{color:var(--ink);font-weight:700}}.chip.ok b{{color:var(--grn)}}.chip.bad b{{color:var(--amb)}}
.kpi{{padding:.95rem 1.1rem;border-left:4px solid var(--c,var(--acc));margin-bottom:.7rem}}
.kpi::after{{content:"";position:absolute;right:-30px;top:-30px;width:110px;height:110px;border-radius:50%;background:var(--c,var(--acc));opacity:.12;filter:blur(8px)}}
.kpi span,.kpi b,.kpi em{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;word-break:normal}}
.kpi span{{display:block;font-family:'JetBrains Mono';font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;color:var(--mut)!important}}
.kpi b{{display:block;font-family:'JetBrains Mono';font-size:clamp(1.15rem,1.9vw,1.8rem);font-weight:700;margin:.2rem 0;color:var(--c,var(--ink))}}
.kpi em{{font-style:normal;font-size:.8rem;color:var(--mut)!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(>div>[data-testid="stVerticalBlock"]),
div[data-testid="stVerticalBlock"][class*="border"]{{background:linear-gradient(180deg,rgba(27,37,64,.75),rgba(20,28,49,.75));border-color:var(--line)!important;border-radius:16px!important}}
.ph{{font-family:'JetBrains Mono';font-size:.76rem;letter-spacing:.12em;text-transform:uppercase;color:var(--acc2)!important;font-weight:700;margin:.25rem 0 .25rem;
  display:flex;align-items:center;gap:.5rem}}
.ph::before{{content:"";width:8px;height:8px;border-radius:2px;background:var(--acc);box-shadow:0 0 10px var(--acc)}}
.note,.note *{{color:var(--mut)!important;font-size:.85rem;line-height:1.6}}
.note b{{color:var(--ink)!important}}
.insight{{background:linear-gradient(135deg,rgba(139,92,246,.16),rgba(34,211,238,.08));border:1px solid rgba(139,92,246,.4);border-radius:14px;padding:.9rem 1.1rem;margin:.3rem 0 .8rem}}
.insight li,.insight p{{color:var(--ink)!important;line-height:1.65}}
.insight b{{color:#c4b5fd!important}}

/* ---------- buttons */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{{border-radius:10px;font-weight:700;border:1px solid rgba(148,163,184,.3);background:var(--card2);color:var(--ink)}}
.stButton>button:hover,.stDownloadButton>button:hover{{border-color:var(--acc);color:#fff}}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"],.stFormSubmitButton>button{{
  background:linear-gradient(135deg,{ACC},#6d28d9);border:none;color:#fff;box-shadow:0 8px 20px -8px rgba(139,92,246,.9)}}
.stButton>button p,.stDownloadButton>button p,.stFormSubmitButton>button p{{color:inherit!important}}
.rep{{font-family:'JetBrains Mono';font-size:.82rem;line-height:1.7;white-space:pre-wrap;color:#dbe4ff;background:#0b1020;border:1px solid var(--line);border-radius:12px;padding:1rem}}
[data-testid="stAlert"] p{{color:inherit!important}}
@media(max-width:820px){{.block-container{{padding:3.4rem .7rem 2rem}}.top{{padding:.8rem}}.kpi b{{font-size:1.45rem}}}}
</style>"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ chart + card helpers
def sty(fig: go.Figure, h: int = 330, **kw) -> go.Figure:
    fig.update_layout(template='plotly_dark', height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Inter', color=INK, size=13), colorway=PALETTE,
                      margin=dict(l=8, r=8, t=34, b=8), title_font=dict(size=14, color=INK),
                      hoverlabel=dict(bgcolor=CARD2, bordercolor=ACC, font=dict(color=INK)),
                      legend=dict(orientation='h', y=-0.18, font=dict(color=INK)), **kw)
    fig.update_xaxes(gridcolor='rgba(148,163,184,.14)', zeroline=False, linecolor='rgba(148,163,184,.3)')
    fig.update_yaxes(gridcolor='rgba(148,163,184,.14)', zeroline=False, linecolor='rgba(148,163,184,.3)')
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, use_container_width=True,
                    config={'displaylogo': False, 'displayModeBar': 'hover', 'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'toImageButtonOptions': {'format': 'png', 'scale': 3}})


def kpi(label: str, value: str, sub: str = '', color: str = ACC) -> None:
    st.markdown(f"<div class='kpi' style='--c:{color}'><span>{label}</span><b>{value}</b><em>{sub}</em></div>", unsafe_allow_html=True)


def panel(title: str) -> None:
    st.markdown(f"<div class='ph'>{title}</div>", unsafe_allow_html=True)


def topbar(title: str, subtitle: str, chips: list[tuple[str, str, str]], icon: str = '📊') -> None:
    """chips: (label, value, state) where state is '', 'ok' or 'bad'."""
    c = ''.join(f"<div class='chip {s}'>{lab} <b>{v}</b></div>" if v else f"<div class='chip {s}'>{lab}</div>" for lab, v, s in chips)
    st.markdown(f"<div class='top'><div class='brand'><div class='emb'>{icon}</div><div><b>{title}</b><small>{subtitle}</small></div></div>"
                f"<div class='chips'>{c}</div></div>", unsafe_allow_html=True)

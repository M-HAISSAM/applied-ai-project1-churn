"""Shared look and feel: the 'Midnight Mono' theme (see THEME.md) used by every page and every chart."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------ palette (THEME.md §2)
BG = '#07090d'        # page
CARD = '#0f1218'      # panels
CARD2 = '#161a21'     # inputs / hover / raised
LINE = 'rgba(255,255,255,.09)'
LINE2 = 'rgba(255,255,255,.22)'
INK = '#f4f6fa'       # main text
MUT = '#8a93a6'       # secondary text
ACC = '#5b8def'       # steel-blue accent: the one accent
ACC2 = '#9cc3ff'      # ice blue: bars, second series
BLUE = '#7d8ca8'      # slate: neutral third series
WHITE = '#ffffff'     # highlighted bars, active pill
RED = '#f87171'
AMB = '#fbbf24'
GRN = '#34d399'
STATUS = (RED, AMB, GRN)  # only these tint a KPI value

SCALE = [[0, GRN], [.35, AMB], [.65, '#fb923c'], [1, RED]]                         # low -> high risk
SEQ = [[0, CARD], [.5, '#3a5fae'], [1, '#cfe1ff']]                                  # neutral sequential
DIV = [[0, RED], [.5, '#12161d'], [1, ACC2]]                                        # -1 .. +1 (correlations)
PALETTE = [ACC2, ACC, '#e8edf5', BLUE, '#3a5fae', '#c7d7f2', AMB, GRN, RED, '#a78bfa']

CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root{{--bg:{BG};--card:{CARD};--card2:{CARD2};--ink:{INK};--mut:{MUT};--line:{LINE};--line2:{LINE2};--acc:{ACC};--acc2:{ACC2};
  --red:{RED};--amb:{AMB};--grn:{GRN};--shadow:0 20px 40px -24px rgba(0,0,0,.9),inset 0 1px 0 rgba(255,255,255,.04)}}
html,body,.stApp{{font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}}
/* near-black canvas with faint diagonal light streaks and a dark bottom vignette */
.stApp{{color:var(--ink);background:
  linear-gradient(115deg,transparent 18%,rgba(190,210,255,.045) 24%,transparent 30%),
  linear-gradient(115deg,transparent 40%,rgba(190,210,255,.035) 47%,transparent 55%),
  linear-gradient(115deg,transparent 64%,rgba(190,210,255,.03) 70%,transparent 76%),
  radial-gradient(80rem 40rem at 50% 120%,rgba(0,0,0,.7),transparent 70%),var(--bg);background-attachment:fixed}}
[data-testid="stHeader"]{{background:transparent}}
/* hide the menu and deploy buttons, but NOT the toolbar itself: it holds the button that re-opens a collapsed sidebar */
[data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],[data-testid="stMainMenu"],footer,#MainMenu{{display:none!important}}
[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapsedControl"] button{{background:{ACC}!important;
  color:#fff!important;border-radius:10px!important;box-shadow:0 6px 18px -8px rgba(91,141,239,.8)}}
[data-testid="stExpandSidebarButton"] *,[data-testid="stSidebarCollapsedControl"] svg{{color:#fff!important;fill:#fff!important}}
.block-container{{max-width:1560px;padding:3.6rem 1.6rem 3rem}}

/* ---------- text: always readable, whatever theme Streamlit picked */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{{color:var(--ink)!important;letter-spacing:-.015em;font-weight:600}}
.stApp p,.stApp li,.stApp span,.stApp label,[data-testid="stMarkdownContainer"]{{color:var(--ink)}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,.stCaption{{color:var(--mut)!important}}
label,[data-testid="stWidgetLabel"] p{{color:var(--mut)!important;font-size:.86rem!important;font-weight:500!important}}
.stApp a{{color:var(--acc2)}}
code{{color:var(--acc2)!important;background:rgba(91,141,239,.12)!important}}

/* ---------- sidebar: plain items, the active one is a solid accent pill */
[data-testid="stSidebar"]{{background:#0a0c11;border-right:1px solid var(--line)}}
[data-testid="stSidebar"] h2{{font-size:1rem!important;color:var(--ink)!important;font-weight:600}}
[data-testid="stSidebar"] [role="radiogroup"] label{{border-radius:10px;padding:.4rem .7rem!important;margin:0 0 .15rem!important;width:100%}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){{background:{ACC}}}
[data-testid="stSidebar"] [role="radiogroup"] label p{{color:var(--ink)!important}}

/* ---------- pills: workspace switch. Inactive = outline, active = solid white */
.st-key-mode [role="radiogroup"]{{gap:.5rem;flex-wrap:wrap}}
.st-key-mode label{{background:transparent;border:1px solid var(--line2);border-radius:999px;padding:.5rem 1.1rem!important;margin:0!important}}
.st-key-mode label:hover{{border-color:var(--ink)}}
.st-key-mode label:has(input:checked){{background:{WHITE};border-color:{WHITE};box-shadow:0 0 18px -4px rgba(255,255,255,.45)}}
.st-key-mode label p{{font-weight:600!important;color:var(--ink)!important;font-size:.92rem!important}}
.st-key-mode label:has(input:checked) p{{color:{BG}!important}}
.st-key-mode [data-testid="stRadioOption"]>div>div:first-child{{display:none}}

/* ---------- inputs */
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"],.stNumberInput input,.stTextInput input{{
  background:var(--card2)!important;color:var(--ink)!important;border-radius:10px!important;border-color:var(--line)!important}}
div[data-baseweb="select"]>div:focus-within,div[data-baseweb="input"]:focus-within{{border-color:var(--acc)!important}}
div[data-baseweb="select"] span,div[data-baseweb="select"] div{{color:var(--ink)}}
span[data-baseweb="tag"]{{background:rgba(91,141,239,.18)!important;border:1px solid {ACC}!important;border-radius:999px!important}}
span[data-baseweb="tag"] span{{color:var(--ink)!important}}
ul[role="listbox"],div[data-baseweb="popover"] li{{background:var(--card2)!important;color:var(--ink)!important}}
[data-testid="stFileUploaderDropzone"]{{background:var(--card)!important;border:1px dashed var(--line2)!important;border-radius:16px!important}}
[data-testid="stFileUploaderDropzone"]:hover{{border-color:var(--acc)!important}}
[data-testid="stFileUploaderDropzone"] *{{color:var(--ink)!important}}
[data-testid="stExpander"] details{{background:var(--card);border:1px solid var(--line)!important;border-radius:16px}}
[data-testid="stExpander"] summary p{{color:var(--ink)!important;font-weight:600}}

/* ---------- tabs: outlined pills, the selected one is solid white */
.stTabs [role="tablist"]{{gap:.45rem;overflow-x:auto;padding:.2rem 0;background:transparent;border:none;margin-bottom:.9rem}}
.stTabs [role="tab"]{{height:auto;padding:.45rem 1.05rem!important;border-radius:999px;background:transparent;border:1px solid var(--line2);white-space:nowrap;margin:0!important}}
.stTabs [role="tab"] p{{color:var(--ink)!important;font-size:.86rem!important;font-weight:500!important}}
.stTabs [role="tab"]:hover{{border-color:var(--ink)}}
.stTabs [role="tab"][aria-selected="true"]{{background:{WHITE}!important;border-color:{WHITE};box-shadow:0 0 18px -4px rgba(255,255,255,.45)}}
.stTabs [role="tab"][aria-selected="true"] p{{color:{BG}!important;font-weight:600!important}}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"],.stTabs [role="tablist"]>:not([role="tab"]){{display:none}}

/* ---------- cards: flat black, hairline border, 16px radius */
.top,.kpi{{position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}}
.top{{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding:.9rem 1.2rem;margin-bottom:1.1rem}}
.brand{{display:flex;align-items:center;gap:.85rem}}
.emb{{width:2.4rem;height:2.4rem;border-radius:50%;background:var(--card2);border:1px solid var(--line2);display:flex;align-items:center;justify-content:center;font-size:1.1rem}}
.brand b{{font-size:1.4rem;font-weight:600;color:var(--ink);letter-spacing:-.015em}}
.brand small{{display:block;color:var(--mut);font-size:.8rem;margin-top:.05rem}}
.chips{{display:flex;gap:.45rem;flex-wrap:wrap}}
.chip{{font-size:.76rem;padding:.35rem .8rem;border-radius:999px;border:1px solid var(--line2);background:transparent;color:var(--mut)}}
.chip b{{color:var(--ink);font-weight:600}}.chip.ok b{{color:var(--grn)}}.chip.bad b{{color:var(--amb)}}
.kpi{{padding:1rem 1.15rem;margin-bottom:.7rem}}
.kpi span,.kpi b,.kpi em{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;word-break:normal}}
.kpi span{{display:flex;align-items:center;gap:.45rem;font-size:.9rem;font-weight:500;color:rgba(244,246,250,.85)!important}}
.kpi span::before{{content:"";flex:none;width:7px;height:7px;border-radius:50%;background:var(--c,var(--acc))}}
.kpi b{{display:block;font-size:clamp(1.5rem,2.3vw,2.3rem);font-weight:700;letter-spacing:-.02em;margin:.25rem 0 .1rem;color:var(--v,var(--ink))}}
.kpi em{{display:block;font-style:normal;font-size:.8rem;color:var(--mut)!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(>div>[data-testid="stVerticalBlock"]),
div[data-testid="stVerticalBlock"][class*="border"]{{background:var(--card);border-color:var(--line)!important;border-radius:16px!important;box-shadow:var(--shadow)}}
.ph{{font-size:1.02rem;font-weight:600;color:var(--ink)!important;margin:.2rem 0 .35rem;letter-spacing:-.01em}}
.note,.note *{{color:var(--mut)!important;font-size:.85rem;line-height:1.6}}
.note b{{color:var(--ink)!important}}
.insight{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:16px;padding:.9rem 1.1rem;margin:.3rem 0 .8rem}}
.insight li,.insight p{{color:var(--ink)!important;line-height:1.65}}
.insight b{{color:var(--acc2)!important}}

/* ---------- buttons: primary = solid accent, secondary = outline */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{{border-radius:10px;font-weight:600;border:1px solid var(--line2);background:transparent;color:var(--ink)}}
.stButton>button:hover,.stDownloadButton>button:hover{{border-color:var(--ink);color:#fff}}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"],.stFormSubmitButton>button{{
  background:{ACC};border:1px solid {ACC};color:#fff}}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover,.stFormSubmitButton>button:hover{{
  box-shadow:0 8px 22px -8px rgba(91,141,239,.9);border-color:{ACC2}}}
.stButton>button p,.stDownloadButton>button p,.stFormSubmitButton>button p{{color:inherit!important}}
.rep{{font-family:'JetBrains Mono';font-size:.82rem;line-height:1.7;white-space:pre-wrap;color:#dbe4f5;background:{BG};border:1px solid var(--line);border-radius:12px;padding:1rem}}
[data-testid="stAlert"] p{{color:inherit!important}}
@media(max-width:820px){{.block-container{{padding:3.4rem .7rem 2rem}}.top{{padding:.8rem}}.kpi b{{font-size:1.6rem}}}}
</style>"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ chart + card helpers (THEME.md §5-6)
def sty(fig: go.Figure, h: int = 330, **kw) -> go.Figure:
    fig.update_layout(template='plotly_dark', height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Inter', color=INK, size=13), colorway=PALETTE, barcornerradius=4,
                      margin=dict(l=8, r=8, t=34, b=8), title_font=dict(size=14, color=INK),
                      hoverlabel=dict(bgcolor=CARD2, bordercolor=ACC, font=dict(color=INK)),
                      legend={'orientation': 'h', 'y': -0.18, 'font': dict(color=INK), **kw.pop('legend', {})}, **kw)
    fig.update_xaxes(gridcolor='rgba(255,255,255,.06)', zeroline=False, linecolor=LINE, tickfont=dict(color=MUT, size=12))
    fig.update_yaxes(gridcolor='rgba(255,255,255,.06)', zeroline=False, linecolor=LINE, tickfont=dict(color=MUT, size=12))
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, width='stretch',
                    config={'displaylogo': False, 'displayModeBar': 'hover', 'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'toImageButtonOptions': {'format': 'png', 'scale': 3}})


def kpi(label: str, value: str, sub: str = '', color: str = ACC) -> None:
    """`color` is the dot before the label; it also tints the value when it is a status colour (RED/AMB/GRN)."""
    v = f';--v:{color}' if color in STATUS else ''
    st.markdown(f"<div class='kpi' style='--c:{color}{v}'><span>{label}</span><b>{value}</b><em>{sub}</em></div>", unsafe_allow_html=True)


def panel(title: str) -> None:
    st.markdown(f"<div class='ph'>{title}</div>", unsafe_allow_html=True)


def topbar(title: str, subtitle: str, chips: list[tuple[str, str, str]], icon: str = '📊') -> None:
    """chips: (label, value, state) where state is '', 'ok' or 'bad'."""
    c = ''.join(f"<div class='chip {s}'>{lab} <b>{v}</b></div>" if v else f"<div class='chip {s}'>{lab}</div>" for lab, v, s in chips)
    st.markdown(f"<div class='top'><div class='brand'><div class='emb'>{icon}</div><div><b>{title}</b><small>{subtitle}</small></div></div>"
                f"<div class='chips'>{c}</div></div>", unsafe_allow_html=True)

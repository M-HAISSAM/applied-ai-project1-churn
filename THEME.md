# Theme: "Midnight Mono"

The look and feel for every page of this project (Churn Risk Advisor and Dataset Lab).
It follows the reference admin dashboard: a near-black canvas, thin-bordered black cards,
crisp white type, **one** steel-blue accent, pill-shaped filters, and ice-blue or white bars.

All values live in [theme.py](theme.py) (Python constants + CSS) and [.streamlit/config.toml](.streamlit/config.toml).
**Never hardcode a hex colour in `app.py` or `dataset_lab.py`: import the token from `theme.py`.**

---

## 1. Principles
1. **Monochrome first.** Almost everything is black, grey or white. Colour is rare and means something.
2. **One accent.** Steel blue (`ACC`) marks the active or primary thing: the selected nav item, the primary button, the main data series.
3. **Semantic colours only for status.** Green, amber and red are used only for risk or quality states (low / watch / high, ok / warning / bad), never as decoration.
4. **Quiet surfaces, loud numbers.** Cards are flat black with a 1px hairline border. The big white number is the hero of each card.
5. **Soft light, no neon.** Glow is subtle: faint diagonal light streaks on the background, and a soft white glow on highlighted bars. No coloured gradients on cards.

## 2. Colour tokens

| Token (`theme.py`) | Hex / value | Use |
|---|---|---|
| `BG` | `#07090d` | Page canvas |
| `CARD` | `#0f1218` | Cards, panels, tab bar |
| `CARD2` | `#161a21` | Inputs, hover, raised surfaces |
| `LINE` | `rgba(255,255,255,.09)` | Card borders, dividers |
| `LINE2` | `rgba(255,255,255,.22)` | Outlined pills, input borders on hover |
| `INK` | `#f4f6fa` | Main text, KPI numbers |
| `MUT` | `#8a93a6` | Secondary text, axis labels, captions |
| `ACC` | `#5b8def` | **The accent:** active nav, primary buttons, main series |
| `ACC2` | `#9cc3ff` | Ice blue: bars, second series, links |
| `BLUE` | `#7d8ca8` | Slate: neutral third series, "other" categories |
| `WHITE` | `#ffffff` | Highlighted bars, active filter pill |
| `GRN` | `#34d399` | Status: good / low risk |
| `AMB` | `#fbbf24` | Status: watch / warning |
| `RED` | `#f87171` | Status: bad / high risk |

**Chart scales**
- `PALETTE` (categorical, in order): `#9cc3ff #5b8def #e8edf5 #7d8ca8 #3a5fae #c7d7f2 #fbbf24 #34d399 #f87171 #a78bfa`. The blues and greys come first, so most charts stay monochrome.
- `SEQ` (sequential): `#0f1218 → #3a5fae → #cfe1ff`
- `DIV` (diverging, for correlations): `#f87171 → #12161d → #9cc3ff`
- `SCALE` (risk, low → high): `#34d399 → #fbbf24 → #fb923c → #f87171`

## 3. Typography
- **Font:** Inter everywhere (400 / 500 / 600 / 700). JetBrains Mono only for code and the text report.
- **Numbers:** `font-variant-numeric: tabular-nums` so the digits line up.
- **Case:** sentence case for every title and label ("Revenue", "Data health score"). No all-caps headings.

| Element | Size | Weight | Colour |
|---|---|---|---|
| Page title (top bar) | 1.4rem | 600 | `INK` |
| Panel / card title | 1.02rem | 600 | `INK` |
| KPI label | .9rem | 500 | `INK` at 85% |
| KPI value | clamp(1.5rem, 2.3vw, 2.3rem) | 700 | `INK` (or a status colour, see §5) |
| KPI sub-text, captions | .8rem | 400 | `MUT` |
| Widget labels | .86rem | 500 | `MUT` |
| Body | .95rem | 400 | `INK` |

## 4. Surfaces and layout
- **Background:** `BG`, with two or three very faint diagonal light streaks (white at about 4–6% alpha, about 115°) and a dark vignette at the bottom.
- **Cards:** `CARD`, 1px `LINE` border, **16px radius**, no gradient, a soft drop shadow `0 20px 40px -24px rgba(0,0,0,.9)`, and a 1px inner top highlight `inset 0 1px 0 rgba(255,255,255,.04)`.
- **Spacing:** 16px gap between cards; 18–20px padding inside a card.
- **Grid:** a KPI row on top (3–6 cards), then charts in 2–3 columns, then tables.

## 5. Components

**Top bar.** A plain title on the left (no gradient text). The emblem is a 2.4rem circle with a hairline border, holding the icon. Status chips on the right are small **outlined pills**.

**Filter and segment pills** (workspace switch, tabs, horizontal radios). Every pill is fully rounded (`999px`).
- Inactive: transparent, 1px `LINE2` border, `INK` text.
- Active: solid `WHITE` background, `BG`-coloured text, a soft white glow.

**Sidebar.** Background `#0a0c11` with a `LINE` border on the right. Items are plain text with an icon. The active item is a **solid `ACC` pill** with white text.

**KPI card.**
- Layout: label at top left, big value below, sub-text under the value.
- Value colour: `INK` by default. It takes the status colour only when that colour is `GRN`, `AMB` or `RED`.
- Accent: any other accent colour shows only as a small 7px dot before the label.
- No coloured left border.

**Panel title** (`panel()`). Sentence case, `INK`, 600 weight. No prefix square, no uppercase mono.

**Buttons.**
- Primary: solid `ACC`, white text, 10px radius, faint blue glow on hover.
- Secondary: transparent with a `LINE2` border; on hover the border turns `INK`.

**Inputs.** `CARD2` background, `LINE` border, 10px radius. Focus: `ACC` border. Multiselect tags: `ACC` at 18% fill with an `ACC` border.

**Insight box.** `CARD` with a 3px `ACC` left border. Bold text is `ACC2`.

**Tables.** Native `st.dataframe` on `CARD`. Progress columns follow `primaryColor` (`ACC`).

## 6. Charts (Plotly, through `sty()` and `show()`)
- Transparent paper and plot background, so the card colour shows through.
- Grid lines `rgba(255,255,255,.06)`. No zero lines. Axis text `MUT`, 12px.
- **Bars:** `ACC2`, or `WHITE` for the highlighted or best bar. 4px corner radius (`barcornerradius`). Bar outline, if any, is `BG`.
- **Lines and areas:** a 2.5px line in `ACC2`, with an area fill of `ACC` at about 22% alpha. Markers are small white dots.
- **Reference lines** (thresholds, "your value"): `INK` dotted. A risk threshold may use `RED` dotted.
- **Donuts:** hole .62 and the total in the centre, coloured with `PALETTE`, or semantic colours for risk groups.
- **Hover label:** `CARD2` background, `ACC` border, `INK` text.
- **Legend:** horizontal, below the plot, `INK` text.

## 7. Do and don't
| Do | Don't |
|---|---|
| Import `ACC`, `RED` and the other tokens from `theme.py` | Write `'#f43f5e'` inline in a page file |
| Use one accent per view | Mix violet, cyan and green as decoration |
| Use white for the most important bar or number | Use gradient-filled cards or gradient text |
| Use sentence-case titles | Use ALL-CAPS mono headings |
| Use `width='stretch'` | Use `use_container_width` |

## 8. Where things live
- `theme.py`: tokens (§2), `CSS` (§3–5), `sty()` and `show()` for charts (§6), `kpi()`, `panel()`, `topbar()`.
- `.streamlit/config.toml`: `primaryColor = ACC`, `backgroundColor = BG`, `secondaryBackgroundColor = CARD2`, `textColor = INK`.
- When you change a token, change it in `theme.py` **and** in `config.toml`, and update this file.

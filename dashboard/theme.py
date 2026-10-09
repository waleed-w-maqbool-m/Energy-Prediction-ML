"""Visual identity for the dashboard: palette tokens (light/dark) and page CSS.

Series colours come from a CVD-validated categorical palette (blue = the model,
orange = what-if scenarios). Actuals are neutral ink so the forecast stays the
loudest thing on the chart.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Palette:
    mode: str
    ink: str
    muted: str
    grid: str
    surface: str
    actual: str
    model: str  # categorical slot 1
    scenario: str  # categorical slot 2
    reference: str  # naive baselines
    positive: str
    band_alpha: float = 0.16


LIGHT = Palette(
    mode="light", ink="#14202e", muted="#5b6675", grid="#e4e8ee", surface="#ffffff",
    actual="#3d4856", model="#2a78d6", scenario="#eb6834", reference="#9aa3af", positive="#006300",
)
DARK = Palette(
    mode="dark", ink="#e8edf3", muted="#9aa5b4", grid="#2b3441", surface="#0e1117",
    actual="#b4bdc9", model="#3987e5", scenario="#d95926", reference="#6c7685", positive="#4cc24c",
    band_alpha=0.3,
)


def palette() -> Palette:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:  # context unavailable (tests, bare mode)
        return LIGHT


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stApp { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3, .display { font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif !important;
  letter-spacing: -0.015em; }

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }

/* masthead */
.masthead h1 { font-size: clamp(1.9rem, 3.4vw, 2.7rem); font-weight: 700; margin: 0 0 .35rem 0;
  padding: 0; line-height: 1.08; }
.masthead p { font-size: 1.04rem; line-height: 1.55; max-width: 62ch; margin: 0; opacity: .78; }

/* scoreboard: plain typography separated by hairlines, not cards */
.scoreboard { display: flex; flex-wrap: wrap; margin: 1.6rem 0 .4rem 0;
  border-top: 1px solid var(--sb-line); border-bottom: 1px solid var(--sb-line); }
.scoreboard > div { flex: 1 1 190px; padding: .95rem 1.1rem .95rem 0; margin-right: 1.1rem; }
.scoreboard > div + div { border-left: 1px solid var(--sb-line); padding-left: 1.1rem; }
.scoreboard .n { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.05rem;
  line-height: 1.05; font-variant-numeric: tabular-nums; }
.scoreboard .l { font-size: .84rem; line-height: 1.35; opacity: .72; margin-top: .25rem; max-width: 26ch; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--sb-line); }
.stTabs [data-baseweb="tab"] { padding: .7rem 0; font-weight: 500; font-size: .98rem; }

/* result headline above the forecast chart */
.result { margin: .2rem 0 .6rem 0; }
.result .big { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.3rem;
  line-height: 1.1; font-variant-numeric: tabular-nums; }
.result .sub { font-size: .95rem; opacity: .75; margin-top: .15rem; }
.tag { display: inline-block; font-size: .78rem; font-weight: 600; padding: .12rem .5rem; border-radius: 3px;
  margin-right: .4rem; vertical-align: .12rem; }
.tag.scenario { background: rgba(235,104,52,.15); color: #c24a1c; }
.tag.model { background: rgba(42,120,214,.13); color: #1d5fae; }

.note { font-size: .88rem; line-height: 1.5; opacity: .75; max-width: 78ch; }
div[data-testid="stMetricValue"] { font-family: 'Bricolage Grotesque', sans-serif; }
.stDataFrame { font-variant-numeric: tabular-nums; }
:root { --sb-line: rgba(128,138,152,.28); }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

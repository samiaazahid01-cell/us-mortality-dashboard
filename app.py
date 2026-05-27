import streamlit as st
import pandas as pd
import google.generativeai as genai
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.warning("GEMINI_API_KEY not found in secrets. LLM-powered insights will be unavailable.")
import numpy as np
import io 
import base64
import plotly.io as pio

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Mortality Dashboard",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:    #070b14;
  --bg-secondary:  #0d1525;
  --bg-card:       rgba(15, 22, 40, 0.85);
  --bg-glass:      rgba(255,255,255,0.03);
  --border:        rgba(99,179,237,0.12);
  --border-glow:   rgba(99,179,237,0.35);
  --accent-blue:   #3b82f6;
  --accent-cyan:   #06b6d4;
  --accent-purple: #8b5cf6;
  --accent-rose:   #f43f5e;
  --accent-amber:  #f59e0b;
  --accent-emerald:#10b981;
  --text-primary:  #e2e8f0;
  --text-secondary:#94a3b8;
  --text-muted:    #475569;
  --glow-blue:     0 0 20px rgba(59,130,246,0.4), 0 0 60px rgba(59,130,246,0.15);
  --glow-cyan:     0 0 20px rgba(6,182,212,0.4),  0 0 60px rgba(6,182,212,0.15);
  --glow-purple:   0 0 20px rgba(139,92,246,0.4), 0 0 60px rgba(139,92,246,0.15);
  --glow-rose:     0 0 20px rgba(244,63,94,0.4),  0 0 60px rgba(244,63,94,0.15);
  --shadow-card:   0 25px 50px rgba(0,0,0,0.5), 0 10px 20px rgba(0,0,0,0.3);
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}
.stApp { background: var(--bg-primary) !important; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1600px; }

/* ── Animated background ── */
.stApp::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background: radial-gradient(ellipse 80% 60% at 10% -10%, rgba(59,130,246,0.08) 0%, transparent 60%),
              radial-gradient(ellipse 60% 50% at 90% 110%, rgba(139,92,246,0.07) 0%, transparent 60%),
              radial-gradient(ellipse 50% 40% at 50% 50%, rgba(6,182,212,0.04) 0%, transparent 70%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1020 0%, #070b14 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-logo {
  text-align: center; padding: 1.5rem 0 2rem;
  border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;
}
.sidebar-logo h1 {
  font-family: 'Syne', sans-serif;
  font-size: 1.15rem; font-weight: 800; letter-spacing: 0.05em;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0.5rem 0 0.25rem;
}
.sidebar-logo p { color: var(--text-muted); font-size: 0.7rem; margin: 0; letter-spacing: 0.1em; text-transform: uppercase; }
.sidebar-section { margin-bottom: 1.5rem; }
.sidebar-section-title {
  font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 0.6rem; padding-left: 0.25rem;
}

/* Selectbox / multiselect */
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: rgba(15,22,40,0.9) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--text-primary) !important;
  transition: border-color .25s, box-shadow .25s;
}
.stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
  border-color: var(--border-glow) !important;
  box-shadow: var(--glow-blue) !important;
}
div[data-baseweb="select"] span { color: var(--text-primary) !important; }

/* Slider */
.stSlider > div { padding: 0 0.25rem; }
.stSlider [data-testid="stSliderTrack"] { background: rgba(99,179,237,0.15) !important; }
.stSlider [data-testid="stSliderThumb"] { background: var(--accent-blue) !important; }

/* Button */
.stButton > button {
  width: 100%; font-family: 'Syne', sans-serif;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2)) !important;
  border: 1px solid rgba(99,179,237,0.3) !important;
  border-radius: 10px !important; color: #93c5fd !important;
  padding: 0.6rem 1rem !important; cursor: pointer;
  transition: all .3s ease;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(59,130,246,0.4), rgba(139,92,246,0.4)) !important;
  border-color: rgba(99,179,237,0.6) !important;
  box-shadow: var(--glow-blue) !important;
  transform: translateY(-1px);
}

/* ── Dashboard title ── */
.dashboard-header {
  text-align: center; padding: 2rem 0 2.5rem; position: relative;
}
.dashboard-header::after {
  content: '';
  display: block; height: 1px; width: 60%;
  margin: 1.5rem auto 0;
  background: linear-gradient(90deg, transparent, rgba(99,179,237,0.4), rgba(139,92,246,0.4), transparent);
}
.dashboard-header .eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.25em;
  text-transform: uppercase; color: var(--accent-cyan); margin-bottom: 0.5rem;
}
.dashboard-header h1 {
  font-family: 'Syne', sans-serif; font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 800; line-height: 1.1; margin: 0;
  background: linear-gradient(135deg, #e2e8f0 0%, #60a5fa 40%, #a78bfa 70%, #e2e8f0 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.dashboard-header .subtitle {
  color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.6rem;
  font-weight: 300; letter-spacing: 0.02em;
}

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1.2rem; margin-bottom: 2rem; }
.kpi-card {
  position: relative; overflow: hidden;
  background: var(--bg-card); border-radius: 16px;
  border: 1px solid var(--border);
  padding: 1.4rem 1.6rem;
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card);
  transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
  cursor: default;
}
.kpi-card::before {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(135deg, var(--card-glow-color, rgba(59,130,246,0.05)), transparent);
  border-radius: inherit; pointer-events: none;
}
.kpi-card::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--card-glow-color, rgba(59,130,246,0.5)), transparent);
}
.kpi-card:hover {
  transform: translateY(-4px);
  border-color: var(--card-border-color, rgba(59,130,246,0.4)) !important;
  box-shadow: var(--card-shadow, var(--glow-blue)), var(--shadow-card) !important;
}
.kpi-label {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 0.5rem;
}
.kpi-value {
  font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1;
  color: var(--text-primary); margin-bottom: 0.3rem;
}
.kpi-sub { font-size: 0.72rem; color: var(--text-secondary); }
.kpi-icon {
  position: absolute; right: 1.4rem; top: 1.2rem;
  font-size: 2rem; opacity: 0.15;
}
.kpi-badge {
  display: inline-block; font-size: 0.65rem; font-weight: 600; letter-spacing: 0.05em;
  padding: 0.15rem 0.5rem; border-radius: 999px;
  background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25);
}
.kpi-badge.warn { background: rgba(245,158,11,0.15); color: #fbbf24; border-color: rgba(245,158,11,0.25); }
.kpi-badge.danger { background: rgba(244,63,94,0.15); color: #fb7185; border-color: rgba(244,63,94,0.25); }

/* ── Section headers ── */
.section-header {
  display: flex; align-items: center; gap: 0.75rem;
  margin: 2.5rem 0 1.2rem;
}
.section-header .line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(99,179,237,0.3), transparent);
}
.section-header h2 {
  font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-secondary); white-space: nowrap; margin: 0;
}
.section-header .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent-blue);
  box-shadow: 0 0 8px var(--accent-blue);
}

/* ── Chart containers ── */
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem;
  backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card);
  margin-bottom: 1.2rem;
  transition: border-color .3s;
}
.chart-card:hover { border-color: var(--border-glow); }
.chart-title {
  font-family: 'Syne', sans-serif; font-size: 0.8rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-secondary); margin-bottom: 0.2rem;
}
.chart-subtitle { font-size: 0.72rem; color: var(--text-muted); margin-bottom: 1rem; }

/* ── Insight cards ── */
.insights-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; }
.insight-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.1rem 1.3rem;
  backdrop-filter: blur(20px); box-shadow: var(--shadow-card);
  transition: transform .25s, border-color .25s;
  position: relative; overflow: hidden;
}
.insight-card::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--insight-accent, var(--accent-blue));
  box-shadow: 0 0 10px var(--insight-accent, var(--accent-blue));
}
.insight-card:hover { transform: translateY(-3px); border-color: var(--border-glow); }
.insight-num {
  font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
  color: var(--text-muted); margin-bottom: 0.35rem; letter-spacing: 0.1em;
}
.insight-title {
  font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 700;
  color: var(--accent-cyan); margin-bottom: 0.4rem; letter-spacing: 0.03em;
}
.insight-body { font-size: 0.78rem; color: var(--text-secondary); line-height: 1.55; }
.insight-value {
  display: inline-block; margin-top: 0.4rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
  color: var(--text-primary); font-weight: 500;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
  font-size: 0.8rem !important; letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  padding: 0.8rem 1.6rem !important;
  background: transparent !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  transition: color .25s, border-color .25s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent-blue) !important;
  border-bottom-color: var(--accent-blue) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0 0 !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── Metric override ── */
[data-testid="stMetric"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }

/* ── Footer ── */
.dashboard-footer {
  text-align: center; padding: 2rem 0 1rem; margin-top: 3rem;
  border-top: 1px solid var(--border);
  color: var(--text-muted); font-size: 0.7rem; letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("NCHS_-_Leading_Causes_of_Death__United_States.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Age-adjusted Death Rate": "Death_Rate",
        "Cause Name": "Cause",
        "113 Cause Name": "Cause_Full"
    })
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")
    df["Death_Rate"] = pd.to_numeric(df["Death_Rate"], errors="coerce")
    df["Year"] = df["Year"].astype(int)
    return df

df_raw = load_data()

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_BG   = "rgba(0,0,0,0)"
PLOTLY_GRID = "rgba(99,179,237,0.07)"
PLOTLY_TEXT = "#94a3b8"
PLOTLY_FONT = dict(family="DM Sans", color=PLOTLY_TEXT)

CAUSE_COLORS = {
    "Heart disease":          "#3b82f6",
    "Cancer":                 "#8b5cf6",
    "All causes":             "#475569",
    "Unintentional injuries": "#f59e0b",
    "CLRD":                   "#06b6d4",
    "Stroke":                 "#10b981",
    "Alzheimer's disease":    "#f43f5e",
    "Diabetes":               "#fb923c",
    "Influenza and pneumonia":"#a78bfa",
    "Kidney disease":         "#34d399",
    "Suicide":                "#fbbf24",
}

def apply_theme(fig, height=400, show_legend=True, margin=None):
    m = margin or dict(l=10, r=10, t=40, b=10)
    fig.update_layout(
        height=height,
        paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        font=PLOTLY_FONT,
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(10,16,32,0.8)", bordercolor="rgba(99,179,237,0.2)",
            borderwidth=1, font=dict(size=11)
        ),
        margin=m,
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID,
                   tickfont=dict(color=PLOTLY_TEXT, size=10)),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID,
                   tickfont=dict(color=PLOTLY_TEXT, size=10)),
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div style="font-size:2.2rem">🔬</div>
      <h1>US MORTALITY<br>DASHBOARD</h1>
      <p>CDC · NCHS · 1999 – 2017</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📅 Year Range</div>', unsafe_allow_html=True)
    years = sorted(df_raw["Year"].unique())
    year_range = st.slider("", min_value=years[0], max_value=years[-1],
                           value=(years[0], years[-1]), label_visibility="collapsed")

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🗺 State</div>', unsafe_allow_html=True)
    states_all = sorted([s for s in df_raw["State"].unique() if s != "United States"])
    selected_state = st.selectbox("", ["United States"] + states_all, label_visibility="collapsed")

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🦠 Disease Focus</div>', unsafe_allow_html=True)
    causes_all = [c for c in df_raw["Cause"].unique() if c != "All causes"]
    selected_cause = st.selectbox(" ", causes_all,
                                  index=causes_all.index("Heart disease") if "Heart disease" in causes_all else 0,
                                  label_visibility="collapsed")

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    if st.button("↺  Reset All Filters"):
        st.rerun()

    st.markdown("""
    <hr style="margin:1.5rem 0 1rem">
    <div style="font-size:0.65rem;color:#475569;letter-spacing:0.06em;text-align:center">
      DATA SOURCE<br>
      <span style="color:#64748b">CDC National Center for<br>Health Statistics</span><br><br>
      RECORDS<br><span style="color:#64748b">10,868 observations</span><br>
      STATES<br><span style="color:#64748b">50 states + DC</span><br>
      CAUSES<br><span style="color:#64748b">10 disease categories</span>
    </div>""", unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
mask = (df_raw["Year"] >= year_range[0]) & (df_raw["Year"] <= year_range[1])
df = df_raw[mask].copy()

df_us    = df[df["State"] == "United States"].copy()
df_state = df[df["State"] == selected_state].copy()
df_states_only = df[df["State"] != "United States"].copy()

df_us_ac = df_us[df_us["Cause"] == "All causes"]
df_state_ac = df_state[df_state["Cause"] == "All causes"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
  <div class="eyebrow">▸ Exploratory Data Analysis · Public Health Intelligence ·  Gemini AI Assisted</div>
  <h1>US Mortality  AI Dashboard</h1>
  <div class="subtitle">CDC · NCHS Leading Causes of Death · {year_range[0]} – {year_range[1]} · {selected_state}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_us_deaths = int(df_us_ac["Deaths"].sum()) if not df_us_ac.empty else 0
avg_death_rate  = round(df_us_ac["Death_Rate"].mean(), 1) if not df_us_ac.empty else 0
state_deaths    = int(df_state_ac["Deaths"].sum()) if not df_state_ac.empty else 0

top_local_cause_df = df_state[df_state["Cause"] != "All causes"].groupby("Cause")["Deaths"].sum()
top_local_cause = top_local_cause_df.idxmax() if not top_local_cause_df.empty else "N/A"
top_local_short = top_local_cause[:16] + ("…" if len(top_local_cause) > 16 else "")

yrs_shown = year_range[1] - year_range[0] + 1

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card" style="--card-glow-color:rgba(59,130,246,0.12);--card-border-color:rgba(59,130,246,0.45);--card-shadow:0 0 30px rgba(59,130,246,0.3)">
    <div class="kpi-icon">💀</div>
    <div class="kpi-label">Total US Deaths</div>
    <div class="kpi-value">{total_us_deaths:,}</div>
    <div class="kpi-sub">All causes · {yrs_shown} years &nbsp;<span class="kpi-badge warn">National</span></div>
  </div>
  <div class="kpi-card" style="--card-glow-color:rgba(6,182,212,0.12);--card-border-color:rgba(6,182,212,0.45);--card-shadow:0 0 30px rgba(6,182,212,0.3)">
    <div class="kpi-icon">📊</div>
    <div class="kpi-label">Avg Death Rate</div>
    <div class="kpi-value">{avg_death_rate}</div>
    <div class="kpi-sub">Age-adjusted per 100k &nbsp;<span class="kpi-badge">Rate</span></div>
  </div>
  <div class="kpi-card" style="--card-glow-color:rgba(139,92,246,0.12);--card-border-color:rgba(139,92,246,0.45);--card-shadow:0 0 30px rgba(139,92,246,0.3)">
    <div class="kpi-icon">🗺</div>
    <div class="kpi-label">{selected_state} Deaths</div>
    <div class="kpi-value">{state_deaths:,}</div>
    <div class="kpi-sub">All causes · selected period &nbsp;<span class="kpi-badge">State</span></div>
  </div>
  <div class="kpi-card" style="--card-glow-color:rgba(244,63,94,0.12);--card-border-color:rgba(244,63,94,0.45);--card-shadow:0 0 30px rgba(244,63,94,0.3)">
    <div class="kpi-icon">🔴</div>
    <div class="kpi-label">Top Local Cause</div>
    <div class="kpi-value" style="font-size:1.2rem;line-height:1.3">{top_local_short}</div>
    <div class="kpi-sub">{selected_state} leading killer &nbsp;<span class="kpi-badge danger">Alert</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  📊  Overview & Trends  ",
    "  🗺  Maps & Analytics  ",
    "  💡  Insights  ",
    "  🤖  AI co pilot",
    "  ⚔️  state showdown",
    "  📄   report generator"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── Section: Distribution ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="dot"></div><h2>Mortality Distribution</h2><div class="line"></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Death Distribution by Cause</div><div class="chart-subtitle">Part-to-whole breakdown · All US · Selected period</div>', unsafe_allow_html=True)
        donut_df = df_us[df_us["Cause"] != "All causes"].groupby("Cause")["Deaths"].sum().reset_index().sort_values("Deaths", ascending=False)
        colors_donut = [CAUSE_COLORS.get(c, "#64748b") for c in donut_df["Cause"]]
        fig_donut = go.Figure(go.Pie(
            labels=donut_df["Cause"], values=donut_df["Deaths"],
            hole=0.58, marker=dict(colors=colors_donut, line=dict(color="#070b14", width=2)),
            textposition="outside", textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Deaths: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{int(donut_df['Deaths'].sum()/1e6):.0f}M</b><br><span style='font-size:10px'>DEATHS</span>",
            x=0.5, y=0.5, showarrow=False, align="center",
            font=dict(size=18, color="#e2e8f0", family="Syne")
        )
        apply_theme(fig_donut, height=380, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Ranked Causes of Mortality</div><div class="chart-subtitle">Cumulative deaths · Horizontal ranking</div>', unsafe_allow_html=True)
        bar_df = df_us[df_us["Cause"] != "All causes"].groupby("Cause")["Deaths"].sum().reset_index().sort_values("Deaths")
        colors_bar = [CAUSE_COLORS.get(c, "#64748b") for c in bar_df["Cause"]]
        fig_bar = go.Figure(go.Bar(
            x=bar_df["Deaths"], y=bar_df["Cause"], orientation="h",
            marker=dict(color=colors_bar,
                        line=dict(color="rgba(255,255,255,0.05)", width=0.5)),
            hovertemplate="<b>%{y}</b><br>Deaths: %{x:,}<extra></extra>",
            text=[f"{v/1e6:.2f}M" for v in bar_df["Deaths"]],
            textposition="outside", textfont=dict(size=10, color="#94a3b8"),
        ))
        apply_theme(fig_bar, height=380, show_legend=False, margin=dict(l=10, r=60, t=30, b=10))
        fig_bar.update_xaxes(showgrid=True)
        fig_bar.update_yaxes(showgrid=False)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section: Trends ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="dot" style="background:var(--accent-purple);box-shadow:0 0 8px var(--accent-purple)"></div>
      <h2>Historical Trends</h2><div class="line" style="background:linear-gradient(90deg,rgba(139,92,246,0.3),transparent)"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="chart-card"><div class="chart-title">Age-Adjusted Death Rate Over Time</div><div class="chart-subtitle">Multi-line by disease category · United States · per 100,000</div>', unsafe_allow_html=True)
    trend_df = df_us[df_us["Cause"] != "All causes"].groupby(["Year","Cause"])["Death_Rate"].mean().reset_index()
    fig_trend = go.Figure()
    for cause in sorted(trend_df["Cause"].unique()):
        d = trend_df[trend_df["Cause"] == cause]
        color = CAUSE_COLORS.get(cause, "#64748b")
        fig_trend.add_trace(go.Scatter(
            x=d["Year"], y=d["Death_Rate"], name=cause,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=5, color=color,
                        line=dict(color="#070b14", width=1.5)),
            hovertemplate=f"<b>{cause}</b><br>Year: %{{x}}<br>Rate: %{{y:.1f}}<extra></extra>",
            fill="tozeroy", fillcolor=color.replace(")", ",0.03)").replace("rgb","rgba") if "rgb" in color else f"rgba(99,179,237,0.02)",
        ))
    apply_theme(fig_trend, height=380)
    fig_trend.update_xaxes(dtick=2)
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section: Scatter + Heatmap ────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="dot" style="background:var(--accent-cyan);box-shadow:0 0 8px var(--accent-cyan)"></div>
      <h2>Deep Analysis</h2>
      <div class="line" style="background:linear-gradient(90deg,rgba(6,182,212,0.3),transparent)"></div>
    </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown('<div class="chart-card"><div class="chart-title">Deaths vs Age-Adjusted Death Rate</div><div class="chart-subtitle">Scatter plot · All causes · Bubble = relative magnitude</div>', unsafe_allow_html=True)
        scatter_df = df_us[df_us["Cause"] != "All causes"].copy()
        scatter_grp = scatter_df.groupby("Cause").agg(
            Deaths=("Deaths","mean"), Death_Rate=("Death_Rate","mean")
        ).reset_index()
        fig_scatter = px.scatter(
            scatter_grp, x="Deaths", y="Death_Rate", text="Cause",
            size="Deaths", size_max=60,
            color="Cause", color_discrete_map=CAUSE_COLORS,
            hover_data={"Cause":True,"Deaths":":.0f","Death_Rate":":.1f"},
        )
        fig_scatter.update_traces(
            textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            marker=dict(opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.1)"))
        )
        apply_theme(fig_scatter, height=380, show_legend=False)
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card"><div class="chart-title">State × Cause Mortality Heatmap</div><div class="chart-subtitle">Avg age-adjusted death rate intensity</div>', unsafe_allow_html=True)
        top_states_hm = ["California","Texas","Florida","New York","Pennsylvania",
                         "Ohio","Illinois","Michigan","Georgia","North Carolina",
                         "Virginia","Washington","Arizona","Tennessee","Indiana"]
        hm_causes = [c for c in CAUSE_COLORS if c != "All causes"]
        hm_df = df_states_only[
            (df_states_only["State"].isin(top_states_hm)) &
            (df_states_only["Cause"].isin(hm_causes))
        ].groupby(["State","Cause"])["Death_Rate"].mean().reset_index()
        hm_pivot = hm_df.pivot(index="State", columns="Cause", values="Death_Rate").fillna(0)
        fig_hm = go.Figure(go.Heatmap(
            z=hm_pivot.values, x=hm_pivot.columns, y=hm_pivot.index,
            colorscale=[[0,"#0d1525"],[0.25,"#1e3a5f"],[0.5,"#1d4ed8"],
                        [0.75,"#7c3aed"],[1.0,"#f43f5e"]],
            hovertemplate="<b>%{y}</b><br>%{x}<br>Rate: %{z:.1f}<extra></extra>",
            colorbar=dict(
                tickfont=dict(color="#94a3b8", size=9),
                title=dict(text="Rate", font=dict(color="#94a3b8", size=10)),
                bgcolor="rgba(10,16,32,0.8)",
                bordercolor="rgba(99,179,237,0.15)", borderwidth=1,
            )
        ))
        apply_theme(fig_hm, height=380, show_legend=False, margin=dict(l=90,r=10,t=30,b=80))
        fig_hm.update_xaxes(tickangle=-35, tickfont=dict(size=9))
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section: Top 5 States + Disease Comparison ───────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="dot" style="background:var(--accent-amber);box-shadow:0 0 8px var(--accent-amber)"></div>
      <h2>State & Disease Spotlight</h2>
      <div class="line" style="background:linear-gradient(90deg,rgba(245,158,11,0.3),transparent)"></div>
    </div>""", unsafe_allow_html=True)

    col5, col6 = st.columns([1, 1])

    with col5:
        st.markdown('<div class="chart-card"><div class="chart-title">Top 5 High-Mortality States</div><div class="chart-subtitle">Cumulative deaths — CA, FL, TX, NY, PA</div>', unsafe_allow_html=True)
        top5_states = ["California","Florida","Texas","New York","Pennsylvania"]
        t5_df = df_states_only[
            (df_states_only["State"].isin(top5_states)) &
            (df_states_only["Cause"] != "All causes")
        ].groupby(["State","Cause"])["Deaths"].sum().reset_index()
        palette5 = ["#3b82f6","#06b6d4","#8b5cf6","#f59e0b","#10b981"]
        fig_t5 = px.bar(
            t5_df, x="Deaths", y="State", color="Cause", orientation="h",
            color_discrete_map=CAUSE_COLORS, barmode="stack",
            hover_data={"Deaths":":.0f"},
        )
        apply_theme(fig_t5, height=380, margin=dict(l=10,r=20,t=30,b=10))
        fig_t5.update_xaxes(title_text="Cumulative Deaths")
        st.plotly_chart(fig_t5, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="chart-card"><div class="chart-title">Heart Disease vs Cancer vs Alzheimer\'s</div><div class="chart-subtitle">Annual deaths comparison · United States</div>', unsafe_allow_html=True)
        trio = ["Heart disease","Cancer","Alzheimer's disease"]
        trio_df = df_us[df_us["Cause"].isin(trio)].groupby(["Year","Cause"])["Deaths"].sum().reset_index()
        fig_trio = go.Figure()
        for cause in trio:
            d = trio_df[trio_df["Cause"] == cause]
            color = CAUSE_COLORS.get(cause, "#64748b")
            fig_trio.add_trace(go.Scatter(
                x=d["Year"], y=d["Deaths"], name=cause,
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=7, symbol="circle",
                            line=dict(color="#070b14", width=2)),
                hovertemplate=f"<b>{cause}</b><br>Year: %{{x}}<br>Deaths: %{{y:,}}<extra></extra>",
            ))
        apply_theme(fig_trio, height=380)
        fig_trio.update_xaxes(dtick=2)
        st.plotly_chart(fig_trio, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MAPS & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown("""
    <div class="section-header">
      <div class="dot" style="background:var(--accent-emerald);box-shadow:0 0 8px var(--accent-emerald)"></div>
      <h2>Geographic Intelligence</h2>
      <div class="line" style="background:linear-gradient(90deg,rgba(16,185,129,0.3),transparent)"></div>
    </div>""", unsafe_allow_html=True)

    # Map 1 — National disease distribution
    st.markdown('<div class="chart-card"><div class="chart-title">National Disease Distribution Map</div><div class="chart-subtitle">Interactive choropleth · Age-adjusted death rate per 100,000 · Select disease focus</div>', unsafe_allow_html=True)

    map1_df = df_states_only[df_states_only["Cause"] == selected_cause] \
        .groupby("State")["Death_Rate"].mean().reset_index()

    state_abbrev = {
        "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
        "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
        "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
        "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
        "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
        "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
        "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
        "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
        "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
        "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
        "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
    }
    map1_df["Code"] = map1_df["State"].map(state_abbrev)

    fig_map1 = go.Figure(go.Choropleth(
        locations=map1_df["Code"], locationmode="USA-states",
        z=map1_df["Death_Rate"],
        colorscale=[[0,"#0a1020"],[0.2,"#1e3a5f"],[0.45,"#1d4ed8"],
                    [0.7,"#7c3aed"],[0.85,"#db2777"],[1.0,"#f43f5e"]],
        marker_line_color="rgba(99,179,237,0.25)",
        marker_line_width=0.8,
        colorbar=dict(
            title=dict(text="Rate/100k", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#94a3b8", size=10),
            bgcolor="rgba(10,16,32,0.85)",
            bordercolor="rgba(99,179,237,0.2)", borderwidth=1,
            len=0.7, thickness=14,
        ),
        hovertemplate="<b>%{location}</b><br>Death Rate: %{z:.1f} per 100k<extra></extra>",
    ))
    fig_map1.update_geos(
        scope="usa", bgcolor="#070b14",
        showland=True, landcolor="#0d1525",
        showlakes=True, lakecolor="#070b14",
        showcoastlines=True, coastlinecolor="rgba(99,179,237,0.15)",
        showcountries=True, countrycolor="rgba(99,179,237,0.12)",
        showframe=False,
        showsubunits=True, subunitcolor="rgba(99,179,237,0.12)",
        projection_type="albers usa",
    )
    fig_map1.update_layout(
        height=480, paper_bgcolor="#070b14", geo_bgcolor="#070b14",
        font=PLOTLY_FONT, margin=dict(l=0,r=0,t=0,b=0),
    )
    st.plotly_chart(fig_map1, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    st.markdown('</div>', unsafe_allow_html=True)

    # Map 2 — Dominant disease per state
    col_m2, col_m3 = st.columns([1,1])

    with col_m2:
        st.markdown('<div class="chart-card"><div class="chart-title">Dominant Disease Map</div><div class="chart-subtitle">Leading cause of death per state · Color by disease</div>', unsafe_allow_html=True)

        dom_df = df_states_only[df_states_only["Cause"] != "All causes"] \
            .groupby(["State","Cause"])["Deaths"].sum().reset_index()
        dom_df = dom_df.loc[dom_df.groupby("State")["Deaths"].idxmax()]
        dom_df["Code"] = dom_df["State"].map(state_abbrev)

        causes_list = dom_df["Cause"].unique().tolist()
        cause_idx   = {c: i for i, c in enumerate(causes_list)}
        dom_df["CauseIdx"] = dom_df["Cause"].map(cause_idx)

        colorscale_dom = []
        for i, cause in enumerate(causes_list):
            v0 = i / len(causes_list)
            v1 = (i+1) / len(causes_list)
            col = CAUSE_COLORS.get(cause, "#64748b")
            colorscale_dom += [[v0, col], [v1, col]]

        fig_map2 = go.Figure(go.Choropleth(
            locations=dom_df["Code"], locationmode="USA-states",
            z=dom_df["CauseIdx"],
            colorscale=colorscale_dom,
            showscale=False,
            marker_line_color="rgba(255,255,255,0.15)", marker_line_width=0.8,
            customdata=dom_df[["Cause","Deaths"]].values,
            hovertemplate="<b>%{location}</b><br>Leading Cause: %{customdata[0]}<br>Total Deaths: %{customdata[1]:,}<extra></extra>",
        ))
        fig_map2.update_geos(
            scope="usa", bgcolor="#070b14",
            showland=True, landcolor="#0d1525",
            showlakes=True, lakecolor="#070b14",
            showcoastlines=True, coastlinecolor="rgba(99,179,237,0.15)",
            showcountries=True, countrycolor="rgba(99,179,237,0.12)",
            showsubunits=True, subunitcolor="rgba(99,179,237,0.12)",
            showframe=False, projection_type="albers usa",
        )

        # Legend annotations
        annotations = []
        for i, cause in enumerate(causes_list[:6]):
            col_val = CAUSE_COLORS.get(cause, "#64748b")
            annotations.append(dict(
                x=1.02, y=0.95 - i*0.15, xref="paper", yref="paper",
                text=f"<span style='color:{col_val}'>■</span> {cause[:20]}",
                showarrow=False, align="left",
                font=dict(size=9, color="#94a3b8"),
            ))

        fig_map2.update_layout(
            height=400, paper_bgcolor="#070b14", geo_bgcolor="#070b14",
            font=PLOTLY_FONT, margin=dict(l=0,r=120,t=0,b=0),
            annotations=annotations,
        )
        st.plotly_chart(fig_map2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m3:
        st.markdown('<div class="chart-card"><div class="chart-title">Regional Risk Hotspots</div><div class="chart-subtitle">Highest mortality zones · All-cause death rate ranking</div>', unsafe_allow_html=True)

        risk_df = df_states_only[df_states_only["Cause"] == "All causes"] \
            .groupby("State")["Death_Rate"].mean().reset_index() \
            .sort_values("Death_Rate", ascending=False).head(20)

        risk_colors = [
            "#f43f5e" if v > risk_df["Death_Rate"].quantile(0.8) else
            "#f59e0b" if v > risk_df["Death_Rate"].quantile(0.5) else
            "#3b82f6"
            for v in risk_df["Death_Rate"]
        ]
        fig_risk = go.Figure(go.Bar(
            x=risk_df["Death_Rate"], y=risk_df["State"],
            orientation="h",
            marker=dict(color=risk_colors,
                        line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
            hovertemplate="<b>%{y}</b><br>Death Rate: %{x:.1f}<extra></extra>",
            text=[f"{v:.1f}" for v in risk_df["Death_Rate"]],
            textposition="outside", textfont=dict(size=9, color="#94a3b8"),
        ))
        apply_theme(fig_risk, height=400, show_legend=False,
                    margin=dict(l=10,r=50,t=30,b=10))
        fig_risk.update_xaxes(title_text="Avg Age-Adjusted Rate")
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Year-over-Year Change ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="dot" style="background:var(--accent-rose);box-shadow:0 0 8px var(--accent-rose)"></div>
      <h2>Year-over-Year Analysis</h2>
      <div class="line" style="background:linear-gradient(90deg,rgba(244,63,94,0.3),transparent)"></div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1,1])

    with col_a:
        st.markdown('<div class="chart-card"><div class="chart-title">Annual Death Totals — US</div><div class="chart-subtitle">All-cause mortality trend with gradient fill</div>', unsafe_allow_html=True)
        annual_df = df_us[df_us["Cause"] == "All causes"].groupby("Year")["Deaths"].sum().reset_index()
        fig_annual = go.Figure()
        fig_annual.add_trace(go.Scatter(
            x=annual_df["Year"], y=annual_df["Deaths"],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=8, color="#3b82f6", line=dict(color="#070b14",width=2)),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.07)",
            hovertemplate="Year: %{x}<br>Deaths: %{y:,}<extra></extra>",
            name="All Causes"
        ))
        apply_theme(fig_annual, height=320, show_legend=False)
        fig_annual.update_xaxes(dtick=2)
        st.plotly_chart(fig_annual, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card"><div class="chart-title">Disease Focus — Annual Death Rate</div><div class="chart-subtitle">Selected disease · Year-by-year trajectory</div>', unsafe_allow_html=True)
        focus_df = df_us[df_us["Cause"] == selected_cause].groupby("Year")["Death_Rate"].mean().reset_index()
        focus_color = CAUSE_COLORS.get(selected_cause, "#8b5cf6")
        yoy = focus_df["Death_Rate"].pct_change() * 100
        colors_yoy = ["#f43f5e" if v > 0 else "#10b981" for v in yoy.fillna(0)]

        fig_focus = go.Figure(go.Bar(
            x=focus_df["Year"], y=focus_df["Death_Rate"],
            marker=dict(color=focus_color, opacity=0.85,
                        line=dict(color="rgba(0,0,0,0.3)",width=0.5)),
            hovertemplate="Year: %{x}<br>Rate: %{y:.1f}<extra></extra>",
        ))
        apply_theme(fig_focus, height=320, show_legend=False)
        fig_focus.update_xaxes(dtick=2)
        fig_focus.update_layout(title_text=f"")
        st.plotly_chart(fig_focus, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown("""
    <div class="section-header">
      <div class="dot" style="background:var(--accent-amber);box-shadow:0 0 8px var(--accent-amber)"></div>
      <h2>Analytical Intelligence · 18 Key Findings</h2>
      <div class="line" style="background:linear-gradient(90deg,rgba(245,158,11,0.3),transparent)"></div>
    </div>""", unsafe_allow_html=True)

    # --- Compute insights ---
    non_all = df_us[df_us["Cause"] != "All causes"]
    cause_totals  = non_all.groupby("Cause")["Deaths"].sum().sort_values(ascending=False)
    cause_rates   = non_all.groupby("Cause")["Death_Rate"].mean().sort_values(ascending=False)

    leading       = cause_totals.index[0]
    second        = cause_totals.index[1]
    lowest_cause  = cause_totals.index[-1]

    # Fastest growing
    growth_df = non_all.groupby(["Year","Cause"])["Death_Rate"].mean().reset_index()
    growth_pivoted = growth_df.pivot(index="Year",columns="Cause",values="Death_Rate")
    first_yr = growth_pivoted.iloc[0]
    last_yr  = growth_pivoted.iloc[-1]
    pct_change = ((last_yr - first_yr) / first_yr * 100).sort_values(ascending=False)
    fastest_growing = pct_change.index[0]
    fastest_pct = round(pct_change.iloc[0], 1)
    biggest_decline = pct_change.index[-1]
    biggest_decline_pct = round(pct_change.iloc[-1], 1)

    # State stats
    state_ac = df_states_only[df_states_only["Cause"]=="All causes"]
    state_deaths_sum = state_ac.groupby("State")["Deaths"].sum().sort_values(ascending=False)
    state_rate_mean  = state_ac.groupby("State")["Death_Rate"].mean().sort_values(ascending=False)
    highest_state    = state_deaths_sum.index[0]
    lowest_state     = state_deaths_sum.index[-1]
    highest_rate_st  = state_rate_mean.index[0]
    lowest_rate_st   = state_rate_mean.index[-1]

    top_cumulative   = state_deaths_sum.index[0]
    top_cum_val      = int(state_deaths_sum.iloc[0])

    # Peak year heart disease
    hd_df = df_us[df_us["Cause"]=="Heart disease"].groupby("Year")["Deaths"].sum()
    hd_peak_yr  = int(hd_df.idxmax())
    hd_peak_val = int(hd_df.max())

    # Peak Alzheimer's rate
    alz_df = df_us[df_us["Cause"]=="Alzheimer's disease"].groupby("Year")["Death_Rate"].mean()
    alz_latest = round(float(alz_df.iloc[-1]),1)
    alz_first  = round(float(alz_df.iloc[0]),1)
    alz_growth = round((alz_latest - alz_first)/alz_first*100, 1)

    # Highest-risk year
    yr_rate = df_us[df_us["Cause"]=="All causes"].groupby("Year")["Death_Rate"].mean()
    peak_rate_yr  = int(yr_rate.idxmax())
    peak_rate_val = round(float(yr_rate.max()),1)

    # Suicide trend
    suc_df = df_us[df_us["Cause"]=="Suicide"].groupby("Year")["Deaths"].sum()
    suc_first = int(suc_df.iloc[0])
    suc_last  = int(suc_df.iloc[-1])
    suc_pct   = round((suc_last-suc_first)/suc_first*100,1)

    # CLRD vs Stroke comparison
    clrd_total = int(cause_totals.get("CLRD",0))
    stroke_total = int(cause_totals.get("Stroke",0))

    insights = [
        {
            "title": "Leading Cause of Death",
            "body": f"{leading} is the single largest cause of mortality in the United States across the full dataset period, responsible for more deaths than any other tracked category.",
            "value": f"{int(cause_totals[leading]):,} cumulative deaths",
            "accent": "#3b82f6",
            "icon": "🔴"
        },
        {
            "title": "Second Leading Cause",
            "body": f"{second} ranks as the second most lethal condition nationally. Together, {leading} and {second} account for the vast majority of preventable deaths.",
            "value": f"{int(cause_totals[second]):,} cumulative deaths",
            "accent": "#8b5cf6",
            "icon": "🟣"
        },
        {
            "title": "Fastest Growing Disease",
            "body": f"Alzheimer's disease has shown the most dramatic upward trajectory, with age-adjusted death rates rising significantly. This mirrors an aging US population and improved diagnostic recognition.",
            "value": f"+{fastest_pct}% rate change since 1999",
            "accent": "#f43f5e",
            "icon": "📈"
        },
        {
            "title": "Highest Mortality State",
            "body": f"{highest_state} records the highest cumulative death count, largely driven by its large population size. On a per-capita rate basis, this does not necessarily indicate higher risk.",
            "value": f"{top_cum_val:,} total deaths",
            "accent": "#f59e0b",
            "icon": "🗺"
        },
        {
            "title": "Lowest Mortality State",
            "body": f"{lowest_state} consistently records the fewest total deaths across the dataset, which correlates with both its smaller population and generally favorable public health outcomes.",
            "value": "Lowest cumulative count",
            "accent": "#10b981",
            "icon": "✅"
        },
        {
            "title": "Highest Death Rate State",
            "body": f"{highest_rate_st} carries the highest age-adjusted death rate, meaning even when normalized for population size, residents face a statistically higher risk of death per year.",
            "value": f"{round(float(state_rate_mean.iloc[0]),1)} per 100,000",
            "accent": "#f43f5e",
            "icon": "⚠️"
        },
        {
            "title": "Biggest Mortality Decline",
            "body": f"{biggest_decline} has seen the most substantial decline in age-adjusted death rate over the study period, reflecting major advances in treatment protocols and preventive medicine.",
            "value": f"{biggest_decline_pct}% rate reduction",
            "accent": "#06b6d4",
            "icon": "📉"
        },
        {
            "title": "Heart Disease Peak Year",
            "body": f"Heart disease deaths reached their highest recorded level in {hd_peak_yr}. Despite this peak, improvements in cardiac care have resulted in declining death rates in recent years.",
            "value": f"{hd_peak_val:,} deaths in {hd_peak_yr}",
            "accent": "#3b82f6",
            "icon": "❤️"
        },
        {
            "title": "Alzheimer's Alarming Rise",
            "body": f"Alzheimer's disease death rate rose from {alz_first} to {alz_latest} per 100,000 — a {alz_growth}% increase. This makes it the fastest-accelerating public health threat in the dataset.",
            "value": f"{alz_growth}% increase since 1999",
            "accent": "#f43f5e",
            "icon": "🧠"
        },
        {
            "title": "Peak Overall Mortality Year",
            "body": f"{peak_rate_yr} recorded the highest national age-adjusted death rate across all causes. This year likely reflects a convergence of multiple disease burdens without the benefit of modern interventions.",
            "value": f"{peak_rate_val} per 100k · year {peak_rate_yr}",
            "accent": "#f59e0b",
            "icon": "📅"
        },
        {
            "title": "Suicide — Silent Epidemic",
            "body": f"Suicide deaths increased from {suc_first:,} in 1999 to {suc_last:,} in 2017 — a {suc_pct}% rise. This alarming trend reflects growing mental health crises, economic stress, and underinvestment in care.",
            "value": f"+{suc_pct}% over 19 years",
            "accent": "#a78bfa",
            "icon": "🚨"
        },
        {
            "title": "CLRD vs Stroke Burden",
            "body": f"Chronic Lower Respiratory Disease (CLRD) at {clrd_total:,} cumulative deaths outpaces stroke at {stroke_total:,}. Smoking prevalence is the primary driver of CLRD's persistently high mortality.",
            "value": f"CLRD {clrd_total:,} vs Stroke {stroke_total:,}",
            "accent": "#06b6d4",
            "icon": "🫁"
        },
        {
            "title": "Best Performing State",
            "body": f"{lowest_rate_st} consistently achieves the lowest age-adjusted death rate nationally. This likely reflects favorable demographics, higher income levels, robust healthcare access, and lifestyle factors.",
            "value": f"{round(float(state_rate_mean.iloc[-1]),1)} per 100k avg rate",
            "accent": "#10b981",
            "icon": "🏆"
        },
        {
            "title": "Top Cumulative State",
            "body": f"{top_cumulative} leads all states in total cumulative deaths, with {top_cum_val:,} recorded across the study period. Population size is the primary driver — California alone holds ~12% of US population.",
            "value": f"{top_cum_val:,} deaths",
            "accent": "#8b5cf6",
            "icon": "🥇"
        },
        {
            "title": "Unexpected Pattern: Diabetes",
            "body": f"Diabetes deaths display an unexpected plateau in the 2010s after decades of increase. This may reflect improved insulin access, weight management programs, and better glycemic control protocols.",
            "value": "Plateau observed post-2010",
            "accent": "#fb923c",
            "icon": "🔬"
        },
        {
            "title": "Hidden Insight: Unintentional Injuries",
            "body": f"Unintentional injuries (accidents) surged dramatically after 2012, largely fueled by the opioid epidemic. By 2017, they had become the third leading cause of death — overtaking CLRD.",
            "value": "Opioid crisis accelerant",
            "accent": "#fbbf24",
            "icon": "💊"
        },
        {
            "title": "Major Finding: Rate vs Volume Gap",
            "body": f"States like Mississippi and West Virginia show high death rates but moderate total volumes, while California shows massive volumes but below-average rates — proving population-adjusted metrics are essential.",
            "value": "Volume ≠ Risk",
            "accent": "#3b82f6",
            "icon": "🎯"
        },
        {
            "title": "Strategic Conclusion",
            "body": f"The US mortality landscape is shaped by two dominant forces: cardiovascular disease (still #1 but declining) and Alzheimer's (rapidly rising). Prevention investment in brain health and cardiac care will define outcomes for the next generation.",
            "value": "Policy-critical finding",
            "accent": "#f43f5e",
            "icon": "🧭"
        },
    ]

    # Render insight cards in a 3-column grid
    for row_start in range(0, len(insights), 3):
        cols = st.columns(3)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(insights):
                ins = insights[idx]
                with col:
                    st.markdown(f"""
                    <div class="insight-card" style="--insight-accent:{ins['accent']}">
                      <div class="insight-num">INSIGHT {idx+1:02d} &nbsp;{ins['icon']}</div>
                      <div class="insight-title">{ins['title']}</div>
                      <div class="insight-body">{ins['body']}</div>
                      <div class="insight-value">{ins['value']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Data table at bottom
    st.markdown("""
    <div class="section-header" style="margin-top:2.5rem">
      <div class="dot" style="background:var(--text-muted);box-shadow:none"></div>
      <h2>Raw Data Explorer</h2>
      <div class="line"></div>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋  View Filtered Dataset", expanded=False):
        show_df = df_state if selected_state != "United States" else df_us
        show_df = show_df[show_df["Cause"] != "All causes"].sort_values(["Year","Deaths"], ascending=[False,False])
        st.dataframe(
            show_df[["Year","State","Cause","Deaths","Death_Rate"]].reset_index(drop=True),
            use_container_width=True,
            height=300,
        )

with tab4:
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #bc77ff; font-family: 'Syne', sans-serif; font-size: 1.2rem; margin: 0; letter-spacing: 0.05em; text-transform: uppercase;">
                🤖 Public Health AI Co-Pilot
            </h2>
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0.2rem 0 0;">
                MORTALITY INTELLIGENCE CHAT • POWERED BY GEMINI 2.5 FLASH
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── AI Configuration & System Prompt ──────────────────────────────────────
    # Initializing systemic boundaries for the public health persona
    SYSTEM_INSTRUCTION = """
    You are an expert, highly professional Public Health Analyst and Data Scientist specializing in US Mortality Analytics.
    Your persona is strictly bounded:
    1. You must only answer queries related to healthcare, mortality statistics, public health trends, CDC data, or epidemiology.
    2. Analyze trends scientifically using epidemiological context (e.g., policy shifts, medical advancements, socio-economic crises like the opioid epidemic).
    3. If a user asks an out-of-scope query (e.g., general programming, entertainment, sports, or non-health math), politely decline by stating:
       "I am programmed strictly as a Public Health AI Assistant to analyze mortality statistics. Please keep your queries focused on healthcare data analytics."
    4. Keep your tone data-backed, clinical, yet highly accessible.
    """

    # ── Chat Session Initialization ───────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Mortality Intelligence Co-Pilot. Ask me anything about US death rates, state comparisons, or specific disease trends from 1999 to 2017."}
        ]

    # ── Display Existing Chat History ─────────────────────────────────────────
    # Custom CSS wrapper ensures it fits the glassmorphism theme flawlessly
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"""
                <div style="font-family: 'DM Sans', sans-serif; font-size: 0.85rem; line-height: 1.5; color: #e2e8f0;">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)

    # ── User Input & AI Execution Trigger ─────────────────────────────────────
    if user_query := st.chat_input("Ask about mortality rates, trends, or state comparisons..."):
        
        # Display user message instantly
        with st.chat_message("user"):
            st.markdown(f'<div style="font-family: \'DM Sans\', sans-serif; font-size: 0.85rem; color: #e2e8f0;">{user_query}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate response using Google Generative AI
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Check if API Key is configured via Secrets Management
            if "GEMINI_API_KEY" in st.secrets:
                try:
                    # Setting up the model configuration pipelines
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                    
                    # Packaging conversational context history for the model
                    chat_context = []
                    for msg in st.session_state.messages[:-1]:
                        role_label = "user" if msg["role"] == "user" else "model"
                        chat_context.append({"role": role_label, "parts": [msg["content"]]})
                    
                    # Fire API request with safety parameters
                    response = model.generate_content(
                        user_query,
                        generation_config={"temperature": 0.2} # Low temperature for high factual accuracy
                    )
                    
                    ai_response = response.text
                    
                    # Render response beautifully inside the container
                    message_placeholder.markdown(f"""
                        <div style="font-family: 'DM Sans', sans-serif; font-size: 0.85rem; line-height: 1.5; color: #e2e8f0;">
                            {ai_response}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Store response to persistence state
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    error_msg = f"An API processing anomaly occurred: {str(e)}"
                    message_placeholder.error(error_msg)
            else:
                # Fallback message if secrets management lacks the secure token
                warning_msg = "Gemini API token is offline. Please deploy via Streamlit Cloud and configure your system secrets securely."
                message_placeholder.warning(warning_msg)

with tab5:
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #00f2fe; font-family: 'Syne', sans-serif; font-size: 1.2rem; margin: 0; letter-spacing: 0.05em; text-transform: uppercase;">
                ⚔️ State vs State Showdown
            </h2>
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0.2rem 0 0;">
                COMPARATIVE ANALYTICS • BENCHMARKING REGIONAL HEALTH PERFORMANCE
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Safe Column Name Resolver
    cause_col = 'Cause Name' if 'Cause Name' in df.columns else 'Cause'
    rate_col = 'Age-adjusted Death Rate' if 'Age-adjusted Death Rate' in df.columns else 'Death_Rate'

    # 2. Safe Variable Resolver (NameError ko khatam karne ke liye)
    # Agar 'selected_disease' nahi mila, toh ye doosre possible names check karega, nahi toh default pehla disease select karega
    if 'selected_disease' in locals():
        chosen_disease = selected_disease
    elif 'disease_focus' in locals():
        chosen_disease = disease_focus
    else:
        chosen_disease = df[cause_col].unique()[0]

    # 3. State Selectors
    all_states = sorted(df['State'].unique())
    
    col_a, col_b = st.columns(2)
    with col_a:
        state_a = st.selectbox("🎯 Select Baseline State (A)", all_states, index=all_states.index("California") if "California" in all_states else 0, key="showdown_state_a")
    with col_b:
        state_b = st.selectbox("🚀 Select Comparison State (B)", all_states, index=all_states.index("Texas") if "Texas" in all_states else min(1, len(all_states)-1), key="showdown_state_b")

    # 4. Safe Filtering using the resolved variables
    df_a = df[(df['State'] == state_a) & (df[cause_col] == chosen_disease)].sort_values('Year')
    df_b = df[(df['State'] == state_b) & (df[cause_col] == chosen_disease)].sort_values('Year')

    if not df_a.empty and not df_b.empty:
        import plotly.graph_objects as go

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=df_a['Year'], y=df_a[rate_col], mode='lines+markers', name=state_a, line=dict(color='#00f2fe', width=3), marker=dict(size=6)))
        fig_comp.add_trace(go.Scatter(x=df_b['Year'], y=df_b[rate_col], mode='lines+markers', name=state_b, line=dict(color='#ff7e67', width=3), marker=dict(size=6)))

        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='DM Sans, sans-serif'),
            margin=dict(l=10, r=10, t=30, b=10), height=340, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickmode='linear', dtick=2),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        
        st.markdown('<div class="visual-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Metrics Calculation
        avg_a = df_a[rate_col].mean()
        avg_b = df_b[rate_col].mean()
        max_year_a = df_a.loc[df_a[rate_col].idxmax()]['Year']
        max_year_b = df_b.loc[df_b[rate_col].idxmax()]['Year']

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown(f"""
                <div style="background: rgba(0, 242, 254, 0.02); border: 1px solid rgba(0, 242, 254, 0.1); padding: 1.2rem; border-radius: 12px; backdrop-filter: blur(10px);">
                    <h4 style="color: #00f2fe; margin:0; font-family: 'Syne', sans-serif; font-size:0.9rem; text-transform: uppercase;">{state_a} Metrics</h4>
                    <p style="color:#94a3b8; font-family: 'DM Sans', sans-serif; font-size:0.8rem; margin:0.6rem 0 0;">• Avg Death Rate: <b style="color: #e2e8f0;">{round(avg_a, 1)}</b> per 100k</p>
                    <p style="color:#94a3b8; font-family: 'DM Sans', sans-serif; font-size:0.8rem; margin:0.2rem 0 0;">• Peak Crisis Year: <b style="color: #e2e8f0;">{max_year_a}</b></p>
                </div>
            """, unsafe_allow_html=True)
        with metric_col2:
            st.markdown(f"""
                <div style="background: rgba(255, 126, 103, 0.02); border: 1px solid rgba(255, 126, 103, 0.1); padding: 1.2rem; border-radius: 12px; backdrop-filter: blur(10px);">
                    <h4 style="color: #ff7e67; margin:0; font-family: 'Syne', sans-serif; font-size:0.9rem; text-transform: uppercase;">{state_b} Metrics</h4>
                    <p style="color:#94a3b8; font-family: 'DM Sans', sans-serif; font-size:0.8rem; margin:0.6rem 0 0;">• Avg Death Rate: <b style="color: #e2e8f0;">{round(avg_b, 1)}</b> per 100k</p>
                    <p style="color:#94a3b8; font-family: 'DM Sans', sans-serif; font-size:0.8rem; margin:0.2rem 0 0;">• Peak Crisis Year: <b style="color: #e2e8f0;">{max_year_b}</b></p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please choose differing states with valid parameters to load analytics.")

with tab6:
    st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #00e676; font-family: 'Syne', sans-serif; font-size: 1.2rem; margin: 0; letter-spacing: 0.05em; text-transform: uppercase;">
                📄 Custom Executive Report Builder
            </h2>
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0.2rem 0 0;">
                DYNAMIC REPORT COMPILER • COMPILED QUANTITATIVE INFRASTRUCTURE FOR EXPORT
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Column Name Resolvers
    cause_col = 'Cause Name' if 'Cause Name' in df.columns else 'Cause'
    rate_col = 'Age-adjusted Death Rate' if 'Age-adjusted Death Rate' in df.columns else 'Death_Rate'
    deaths_col = 'Number of Deaths' if 'Number of Deaths' in df.columns else 'Deaths'

    # 2. Interactive Report Customizer Widgets
    st.markdown("<p style='font-size: 0.85rem; color: #00e676; font-weight: bold; margin-bottom: 0.5rem;'>⚙️ PDF Report Configuration:</p>", unsafe_allow_html=True)
    
    all_states_rep = sorted(df['State'].unique())
    all_causes_rep = sorted(df[cause_col].unique())
    
    rep_col1, rep_col2, rep_col3 = st.columns(3)
    with rep_col1:
        rep_state = st.selectbox("📂 Focus Region / State", all_states_rep, index=all_states_rep.index("United States") if "United States" in all_states_rep else 0, key="pdf_widget_state")
    with rep_col2:
        rep_disease = st.selectbox("🔬 Primary Medical Cause", all_causes_rep, key="pdf_widget_disease")
    with rep_col3:
        min_yr, max_yr = int(df['Year'].min()), int(df['Year'].max())
        rep_years = st.slider("📅 Timeline Range", min_yr, max_yr, (min_yr, max_yr), key="pdf_widget_years")

    # 3. Dynamic Data Query Processing
    filtered_report_df = df[
        (df['State'] == rep_state) & 
        (df[cause_col] == rep_disease) & 
        (df['Year'] >= rep_years[0]) & 
        (df['Year'] <= rep_years[1])
    ].sort_values('Year')

    if not filtered_report_df.empty:
        # Calculations
        rep_total_deaths = filtered_report_df[deaths_col].sum()
        rep_avg_rate = filtered_report_df[rate_col].mean()
        rep_peak_row = filtered_report_df.loc[filtered_report_df[rate_col].idxmax()]
        
        start_rate = filtered_report_df.iloc[0][rate_col]
        end_rate = filtered_report_df.iloc[-1][rate_col]
        rate_diff = end_rate - start_rate
        trend_status = "an overall increase" if rate_diff > 0 else "a baseline decrease"

        # Display Live Preview Chart right inside the Streamlit Tab for the user
        import plotly.express as px
        fig_preview = px.line(
            filtered_report_df, x='Year', y=rate_col,
            title=f"Trend Vector: {rep_disease} ({rep_state})",
            template="plotly_dark"
        )
        fig_preview.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_preview, use_container_width=True)

        # 4. Generate Interactive Standalone HTML Report (Highly professional alternative to PDF)
        # It embeds charts flawlessly without needing system level image packages!
        chart_html = pio.to_html(fig_preview, include_plotlyjs='cdn', full_html=False)

        html_document = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; background-color: #f8fafc; color: #1e293b; }}
                .report-card {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border-top: 6px solid #0f172a; }}
                .header-bar {{ background-color: #0f172a; color: white; padding: 20px; border-radius: 6px; margin-bottom: 20px; }}
                .meta-table {{ width: 100%; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
                h2 {{ color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; }}
                .metric-box {{ background-color: #f1f5f9; padding: 15px; border-radius: 6px; font-weight: 500; line-height: 1.7; }}
                .footer {{ text-align: center; margin-top: 40px; font-size: 11px; color: #64748b; }}
            </style>
        </head>
        <body>
            <div class="report-card">
                <div class="header-bar">
                    <h1 style="margin:0; font-size:22px;">PUBLIC HEALTH EXECUTIVE BRIEFING</h1>
                    <p style="margin:5px 0 0 0; font-size:12px; color:#94a3b8;">AUTOMATED EDA MORTALITY COMPILER</p>
                </div>
                
                <table class="meta-table">
                    <tr>
                        <td><b>Target Vector:</b> {rep_disease}</td>
                        <td><b>Region Scope:</b> {rep_state}</td>
                    </tr>
                    <tr>
                        <td><b>Timeline Window:</b> {rep_years[0]} - {rep_years[1]}</td>
                        <td><b>Generated Date:</b> 2026</td>
                    </tr>
                </table>

                <h2>1. Core Metrics Summary</h2>
                <div class="metric-box">
                    • Cumulative Mortality Load: {rep_total_deaths:,} total recorded deaths.<br>
                    • Dynamic Timeline Baseline: {round(rep_avg_rate, 2)} average deaths per 100k population.<br>
                    • Peak Target Crisis Year: Year {int(rep_peak_row['Year'])} reached maximum severity of {rep_peak_row[rate_col]} per 100k.
                </div>

                <h2>2. Statistical Visual Trend</h2>
                <div style="margin: 20px 0; background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;">
                    {chart_html}
                </div>

                <h2>3. Analytical Narrative Interpretation</h2>
                <p>
                    During the initial evaluation milestone of {rep_years[0]}, the statistical age-adjusted death rate within the territorial boundaries of {rep_state} hit an operational reference index of <b>{start_rate}</b> per 100k population. By the standard terminal milestone interval of {rep_years[1]}, the verified metrics tracking indexes converged at <b>{end_rate}</b> per 100k.
                </p>
                <p>
                    This delta variation trajectory documents <b>{trend_status}</b> of approximately <b>{abs(round(rate_diff, 2))}</b> units per 100k citizens across the designated custom timeline matrix.
                </p>

                <div class="footer">
                    CONFIDENTIAL DATA REPOSITORY SYSTEM - GENERATED LOGISTICS SUMMARY
                </div>
            </div>
        </body>
        </html>
        """

        # 5. Premium Interactive HTML Download Trigger
        st.download_button(
            label="📥 Export Executive Report Bundle (.HTML)",
            data=html_document,
            file_name=f"Executive_Report_{rep_state}_{rep_disease.replace(' ', '_')}.html",
            mime="text/html"
        )
    else:
        st.warning("No records found matching these metrics. Modify sliders to reload configuration fields.")
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-footer">
  US MORTALITY DASHBOARD &nbsp;·&nbsp; CDC NCHS Leading Causes of Death 1999–2017
  &nbsp;·&nbsp; Built with Streamlit & Plotly &nbsp;·&nbsp; Exploratory Data Analysis Project &nbsp;·&nbsp; Gemini AI Assisted
</div>
""", unsafe_allow_html=True)

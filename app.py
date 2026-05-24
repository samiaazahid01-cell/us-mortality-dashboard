import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="US Mortality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ---------------- CSS (Premium Charcoal Matte UI) ----------------
st.markdown("""
<style>
/* Main App Background */
.stApp {
    background-color: #1c1c1c !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #181818 !important;
    border-right: 1px solid #2d2d2d;
}

/* Top Bright KPI Cards (Matching Dashboard UI Headers) */
.kpi-card-bright {
    background-color: #f3f4f6 !important;
    border-radius: 16px;
    padding: 20px;
    text-align: left;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    margin-bottom: 15px;
}
.kpi-title-bright { 
    font-size: 13px; 
    color: #6b7280; 
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-value-bright { 
    font-size: 28px; 
    font-weight: 800; 
    color: #111827; 
    margin-top: 5px;
}

/* Dark Container Panels for Graphs & Visuals */
.chart-container {
    background-color: #282828 !important;
    border: 1px solid #333333;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

/* ---- MODIFIED: Custom Title Center Alignment ---- */
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
    text-align: center; /* This centers the title */
    padding: 20px 0px 30px 0px;
    letter-spacing: -0.5px;
    width: 100%;
}

/* Tab Component Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #282828 !important;
    border: 1px solid #333333 !important;
    color: #9ca3af !important;
    border-radius: 8px 8px 0px 0px;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background-color: #38bdf8 !important;
    color: #111827 !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- APPLICATION TITLE ----------------
st.markdown('<div class="main-title">US Mortality Dashboard</div>', unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    return pd.read_csv("NCHS_-_Leading_Causes_of_Death__United_States.csv")

df = load_data()

# Separating National totals from individual state logs
df_us_total = df[df["State"] == "United States"]
df_states = df[df["State"] != "United States"]
df_clean = df_states[df_states["Cause Name"] != "All causes"]

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.markdown("### Controls")
year = st.sidebar.selectbox("Year", sorted(df_states["Year"].unique(), reverse=True))
state = st.sidebar.selectbox("State", sorted(df_states["State"].unique()))

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Configuration")
all_causes = sorted(df_clean["Cause Name"].unique())
map_cause = st.sidebar.selectbox("Disease Focus", all_causes)



# ---------------- DATA CALCULATIONS ----------------
# 1. National Level Computations
us_year = df_us_total[(df_us_total["Year"] == year) & (df_us_total["Cause Name"] == "All causes")]
us_deaths = int(us_year["Deaths"].values[0]) if not us_year.empty else 0
us_rate = us_year["Age-adjusted Death Rate"].values[0] if not us_year.empty else 0

# 2. Regional State Level Computations
filtered = df_clean[(df_clean["Year"] == year) & (df_clean["State"] == state)]
state_total = df_states[(df_states["Year"] == year) & (df_states["State"] == state) & (df_states["Cause Name"] == "All causes")]

state_deaths = int(state_total["Deaths"].values[0]) if not state_total.empty else 0
state_rate = state_total["Age-adjusted Death Rate"].values[0] if not state_total.empty else 0
top_cause = filtered.sort_values("Deaths", ascending=False).iloc[0]["Cause Name"] if not filtered.empty else "N/A"

# ---------------- NAVIGATION TABS ----------------
tab1, tab2 = st.tabs(["Overview", "Analytics & Maps"])

# ================= TAB 1: DASHBOARD OVERVIEW =================
with tab1:
    
    st.markdown('<div class="section-header">National & State Metrics</div>', unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f'<div class="kpi-card-bright"><div class="kpi-title-bright">Total US Deaths</div><div class="kpi-value-bright">{us_deaths:,}</div></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card-bright"><div class="kpi-title-bright">US Death Rate</div><div class="kpi-value-bright">{us_rate}</div></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card-bright"><div class="kpi-title-bright">{state} Deaths</div><div class="kpi-value-bright">{state_deaths:,}</div></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card-bright"><div class="kpi-title-bright">Top Local Cause</div><div class="kpi-value-bright" style="font-size:19px;">{top_cause[:20]}...</div></div>', unsafe_allow_html=True)

    # Visualization Panels
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Death Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(filtered, names="Cause Name", values="Deaths", hole=0.5)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Ranked Causes of Mortality</div>', unsafe_allow_html=True)
        fig = px.bar(filtered.sort_values("Deaths"), x="Deaths", y="Cause Name", orientation="h")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        fig.update_traces(marker_color='#38bdf8')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Timeline Row
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">Historical Trend Line ({state})</div>', unsafe_allow_html=True)
    trend = df_clean[df_clean["State"] == state]
    fig = px.line(trend, x="Year", y="Age-adjusted Death Rate", color="Cause Name")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: ADVANCED MAPS =================
with tab2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🗺️ National Distribution Map: {map_cause}</div>', unsafe_allow_html=True)
    
    map_data = df_clean[(df_clean["Year"] == year) & (df_clean["Cause Name"] == map_cause)]
    
    fig = px.choropleth(
        map_data,
        locations="State",
        locationmode="USA-states",
        color="Deaths",
        scope="usa",
        color_continuous_scale="Purples"
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabular Breakdown Leaderboard
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏆 Top 5 High-Mortality States (Cumulative Dataset)</div>', unsafe_allow_html=True)
    top5 = df_states.groupby("State")["Deaths"].sum().sort_values(ascending=False).head(5).reset_index()
    st.dataframe(top5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
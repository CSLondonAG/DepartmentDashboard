import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path

# --- Page Config & Custom CSS ---
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# A more refined, modern CSS with subtle shadows and a cleaner look
st.markdown("""
<style>
/* Main container padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px; /* Optional: Constrain width for a more focused layout */
    margin: auto;
}

/* Header and Subheader styling */
h1, h2, h3 {
    color: #0d47a1; /* A deep blue for headers */
    font-weight: 600;
}
.stMarkdown h3 {
    color: #455a64; /* Darker grey for subheaders */
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 5px;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

/* Custom metric tiles */
.metric-container {
    background-color: #f7f9fc; /* Light grey background for a clean look */
    border-left: 5px solid; /* Use the left border for color coding */
    border-radius: 8px;
    padding: 1rem 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: all 0.2s ease-in-out;
}
.metric-container:hover {
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.metric-title {
    font-size: 1rem;
    font-weight: 500;
    color: #616161;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e88e5; /* A vibrant blue for values */
}
.metric-container-red {
    border-left-color: #F44336; /* Red for low scores */
}
.metric-container-yellow {
    border-left-color: #FFC107; /* Yellow for warning */
}
.metric-container-green {
    border-left-color: #4CAF50; /* Green for good */
}

/* Chart container styling */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}
.vega-embed .chart-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #333;
}
</style>
""", unsafe_allow_html=True)


# --- Helpers & Renderers ---
def fmt_mmss(sec):
    if pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def fmt_hms(sec):
    if pd.isna(sec): return "–"
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"

def get_utilization_color(util):
    if util >= 0.50: return "#4CAF50"
    elif util >= 0.30: return "#FFC107"
    else: return "#F44336"

def get_sla_score_color(score):
    if score >= 80: return "#4CAF50"
    elif score >= 70: return "#FFC107"
    else: return "#F44336"

def render_custom_metric(container, title, value, tooltip, color):
    """Render a single metric tile with title, value, tooltip and colored border."""
    with container:
        st.markdown(
            f'<div class="metric-container" style="border-left-color:{color};" title="{tooltip}">'
            f'<div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# --- Load & preprocess data ---
BASE_DIR   = Path(__file__).parent
chat_path  = BASE_DIR / "chat.csv"
email_path = BASE_DIR / "email.csv"
if not chat_path.exists() or not email_path.exists():
    st.error("Please place chat.csv and email.csv beside this script.")
    st.stop()

df_items     = pd.read_csv("report_items.csv",    dayfirst=True, parse_dates=["Start DT","End DT"])
df_presence  = pd.read_csv("report_presence.csv", dayfirst=True, parse_dates=["Start DT","End DT"])
df_shifts    = pd.read_csv("shifts.csv")
chat_sla_df  = pd.read_csv(chat_path,  dayfirst=True, parse_dates=["Date/Time Opened"])
email_sla_df = pd.read_csv(email_path, dayfirst=True, parse_dates=["Date/Time Opened","Completion Date"])
for df in (df_items, df_presence, df_shifts, chat_sla_df, email_sla_df):
    df.columns = df.columns.str.strip()

# --- Sidebar: Date Range (chat.csv) ---
st.sidebar.header("Filter Options")
min_date = chat_sla_df["Date/Time Opened"].dt.date.min()
max_date = chat_sla_df["Date/Time Opened"].dt.date.max()
start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=6),
                                   min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End Date",   value=max_date,
                                   min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start must be on or before End")
    st.stop()

# --- Compute Core Metrics ---
mask      = ((df_items["Start DT"].dt.date >= start_date) &
             (df_items["Start DT"].dt.date <= end_date))
df_period = df_items[mask].copy()
df_period["Duration_sec"] = (df_period["End DT"] - df_period["Start DT"]).dt.total_seconds()
chat_df   = df_period[df_period["Service Channel: Developer Name"] == "sfdc_liveagent"]
email_df  = df_period[df_period["Service Channel: Developer Name"] == "casesChannel"]

chat_total  = len(chat_df)
email_total = len(email_df)
chat_aht    = chat_df["Duration_sec"].mean()  if chat_total  else None
email_aht   = email_df["Duration_sec"].mean() if email_total else None

# --- SLA slices & operational metrics ---
chat_sla_p  = chat_sla_df[
    (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
]
email_sla_p = email_sla_df[
    (email_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (email_sla_df["Date/Time Opened"].dt.date <= end_date)
]
avg_resp_hrs  = email_sla_p["Elapsed Time (Hours)"].mean() if len(email_sla_p) else 0
avg_resp_secs = avg_resp_hrs * 3600

answered_chats = chat_sla_p[chat_sla_p["Wait Time"].notna()]
avg_chat_wait  = answered_chats["Wait Time"].mean() if len(answered_chats) else 0

# --- Availability & Utilization ---
window_start = datetime.combine(start_date, datetime.min.time())
window_end   = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
chat_avail_secs = email_avail_secs = 0
for _, pres in df_presence.iterrows():
    s = max(pres["Start DT"], window_start)
    e = min(pres["End DT"],   window_end)
    delta = (e - s).total_seconds() if e > s else 0
    if pres["Service Presence Status: Developer Name"] in ("Available_Chat","Available_All"):
        chat_avail_secs += delta
    if pres["Service Presence Status: Developer Name"] in ("Available_Email_and_Web","Available_All"):
        email_avail_secs += delta

chat_handle_secs  = chat_df["Duration_sec"].sum()
email_handle_secs = email_df["Duration_sec"].sum()
chat_util  = chat_handle_secs  / chat_avail_secs  if chat_avail_secs  else 0
email_util = email_handle_secs / email_avail_secs if email_avail_secs else 0

# --- Build per-day SLA & weighted SLA ---
daily = []
for d in pd.date_range(start_date, end_date):
    dd = d.normalize()
    cd = chat_sla_p[chat_sla_p["Date/Time Opened"].dt.date == dd.date()]
    cw = cd[cd["Wait Time"].notna()]; v_c = len(cw)
    pct60 = (cw["Wait Time"] <= 60).sum()/v_c*100 if v_c else 0
    ar    = (cd["Abandoned After"] > 20).sum()/len(cd)*100 if len(cd) else 0
    sla_c = max(0, min(100, ((0.5*pct60 - 0.3*(cw["Wait Time"].mean()/60) - 0.2*ar)/56.25)*100))

    ed   = email_sla_p[email_sla_p["Date/Time Opened"].dt.date == dd.date()]
    v_e  = len(ed)
    pct1 = (ed["Elapsed Time (Hours)"] <= 1).sum()/v_e*100 if v_e else 0
    sla_e = max(0, min(100, ((0.6*pct1 - 0.4*ed["Elapsed Time (Hours)"].mean())/56.25)*100))

    daily.append({
        "Date":      dd,
        "Chat SLA":  sla_c,  "Chat Vol":  v_c,
        "Email SLA": sla_e,  "Email Vol": v_e
    })
df_daily = pd.DataFrame(daily)
df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"]*df_daily["Chat Vol"] +
    df_daily["Email SLA"]*df_daily["Email Vol"]
) / (df_daily["Chat Vol"] + df_daily["Email Vol"])
df_daily = df_daily.fillna(0) # Handle divisions by zero

chat_weighted  = (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum()  / df_daily["Chat Vol"].sum() if df_daily["Chat Vol"].sum() > 0 else 0
email_weighted = (df_daily["Email SLA"]*df_daily["Email Vol"]).sum() / df_daily["Email Vol"].sum() if df_daily["Email Vol"].sum() > 0 else 0
weighted_sla   = (
    (df_daily["Chat SLA"]*df_daily["Chat Vol"] +
     df_daily["Email SLA"]*df_daily["Email Vol"]).sum()
  / (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum()
) if (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum() > 0 else 0

# --- UI: Header, Metrics & Chart ---
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.divider()

# Core Metrics
st.subheader("Core Metrics")
c1, c2, c3, c4 = st.columns(4)
render_custom_metric(c1, "Total Chats",         chat_total,          "Total chat interactions",  "#4CAF50")
render_custom_metric(c2, "Total Emails",        email_total,         "Total email interactions", "#4CAF50")
render_custom_metric(c3, "Avg Chat AHT",        fmt_mmss(chat_aht),  "Average chat handle time", "#4CAF50")
render_custom_metric(c4, "Avg Email AHT",       fmt_mmss(email_aht), "Average email handle time","#4CAF50")

# Operational Metrics
st.subheader("Operational Metrics")
m1, m2, m3 = st.columns(3)
render_custom_metric(m1, "Chat Utilization",      f"{chat_util:.1%}",   "Agent-minute chat utilization",  get_utilization_color(chat_util))
render_custom_metric(m2, "Email Utilization",     f"{email_util:.1%}",  "Agent-minute email utilization", get_utilization_color(email_util))
render_custom_metric(m3, "Avg Email Resp Time",   fmt_hms(avg_resp_secs),"Average email response time",  get_sla_score_color(100 - (avg_resp_hrs * 100)))

# SLA Score Summary
st.subheader("🎯 SLA Score Summary")
s1, s2, s3 = st.columns(3)
render_custom_metric(s1, "Chat SLA Score",   f"{chat_weighted:.1f}",  "Weighted chat SLA",   get_sla_score_color(chat_weighted))
render_custom_metric(s2, "Email SLA Score",  f"{email_weighted:.1f}", "Weighted email SLA",  get_sla_score_color(email_weighted))
render_custom_metric(s3, "Weighted SLA",     f"{weighted_sla:.1f}",   "Overall weighted SLA",get_sla_score_color(weighted_sla))


# Weighted SLA Trend Chart
st.subheader("📈 Weighted SLA Trend")

trend = df_daily[["Date","Weighted SLA"]].sort_values("Date")
# Use a cleaner color scheme for the chart
chart = (
    alt.Chart(trend)
    .mark_line(point={"size": 100, "color": "#1e88e5"}, color="#42a5f5", strokeWidth=3)
    .encode(
        x=alt.X("Date:T", title="Date",
                axis=alt.Axis(format="%d %b", labelAngle=-45)),
        y=alt.Y("Weighted SLA:Q", title="Weighted SLA", scale=alt.Scale(domain=[0,105])),
        tooltip=[alt.Tooltip("Date:T", format="%d %b"), alt.Tooltip("Weighted SLA:Q", format=".1f")]
    )
    .interactive()
)
labels = chart.mark_text(dy=-15, color="#1e88e5").encode(text=alt.Text("Weighted SLA:Q", format=".1f"))
rule   = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(color="#d32f2f", strokeDash=[5,5]).encode(y="y:Q")
rule_lb= alt.Chart(pd.DataFrame({"y":[80]})).mark_text(align="left", color="#d32f2f", dy=-8, dx=5)\
           .encode(y="y:Q", text=alt.value("Target: 80%"))

st.altair_chart((chart+labels+rule+rule_lb).properties(width=700, height=350),
                use_container_width=True)

with st.expander("📊 View Daily Performance Data"):
    st.dataframe(df_daily.style.format({
        "Chat SLA": "{:.1f}", "Email SLA": "{:.1f}", "Weighted SLA": "{:.1f}"
    }))
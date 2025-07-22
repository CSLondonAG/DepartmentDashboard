import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path

# --- Page Config ---
st.set_page_config(page_title="Department Performance Dashboard", layout="wide")

# --- Custom CSS for Styling ---
st.markdown(
    """
<style>
.main .block-container {
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}
.stMetric {
    background-color: #f0f2f6;
    /* Removed border-left from here to allow dynamic control */
    padding: 15px;
    border-radius: 8px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}
.stMetric > div:first-child {
    font-size: 1.1em;
    font-weight: 600;
    color: #555;
}
.stMetric > div:nth-child(2) > div {
    font-size: 2.2em;
    font-weight: bold;
    color: #333;
}
h1, h2, h3, h4, h5, h6 {
    color: #2F80ED; /* A blue for headers */
}
hr {
    border-top: 1px solid #ddd;
}
.sidebar .sidebar-content {
    background-color: #e0f2f7; /* Light blue for sidebar */
}
</style>
""",
    unsafe_allow_html=True
)

# --- Paths & Existence Check ---
BASE_DIR   = Path(__file__).parent
chat_path  = BASE_DIR / "chat.csv"
email_path = BASE_DIR / "email.csv"

if not chat_path.exists() or not email_path.exists():
    st.error("Make sure chat.csv and email.csv are in the same folder as this script.")
    st.stop()

# --- Load Data (no caching) ---
def load_data():
    df_items = pd.read_csv("report_items.csv", dayfirst=True,
                           parse_dates=["Start DT","End DT"])
    df_presence = pd.read_csv("report_presence.csv", dayfirst=True,
                              parse_dates=["Start DT","End DT"])
    df_shifts = pd.read_csv("shifts.csv")

    chat_sla_df = pd.read_csv(chat_path, dayfirst=True,
                              parse_dates=["Date/Time Opened"])
    email_sla_df = pd.read_csv(email_path, dayfirst=True,
                               parse_dates=["Date/Time Opened","Completion Date"])

    for df in (df_items, df_presence, df_shifts, chat_sla_df, email_sla_df):
        df.columns = df.columns.str.strip()
    return df_items, df_presence, df_shifts, chat_sla_df, email_sla_df

df_items, df_presence, df_shifts, chat_sla_df, email_sla_df = load_data()

# --- Sidebar: Date Range from chat.csv only ---
st.sidebar.header("Filter Options")
min_date = chat_sla_df["Date/Time Opened"].dt.date.min()
max_date = chat_sla_df["Date/Time Opened"].dt.date.max()

start_date = st.sidebar.date_input(
    "Start Date", value=max_date - timedelta(days=6),
    min_value=min_date, max_value=max_date
)
end_date = st.sidebar.date_input(
    "End Date", value=max_date,
    min_value=min_date, max_value=max_date
)
if start_date > end_date:
    st.sidebar.error("Start date must be on or before End date")
    st.stop()

# --- Helper Formatters ---
def fmt_mmss(sec):
    if pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def fmt_hms(sec):
    if pd.isna(sec): return "–"
    h, remainder = divmod(int(sec), 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02}:{m:02}:{s:02}"

# --- Filter report_items for volumes & AHT ---
mask = (
    (df_items["Start DT"].dt.date >= start_date) &
    (df_items["Start DT"].dt.date <= end_date)
)
df_period = df_items[mask].copy()
df_period["Duration_sec"] = (df_period["End DT"] - df_period["Start DT"]).dt.total_seconds()

chat_df  = df_period[df_period["Service Channel: Developer Name"]=="sfdc_liveagent"]
email_df = df_period[df_period["Service Channel: Developer Name"]=="casesChannel"]

chat_total  = len(chat_df)
email_total = len(email_df)

chat_aht   = chat_df["Duration_sec"].mean()  if chat_total  else None
email_aht  = email_df["Duration_sec"].mean() if email_total else None

# --- SLA slices by Date/Time Opened ---
chat_sla_period  = chat_sla_df[
    (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
]
email_sla_period = email_sla_df[
    (email_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (email_sla_df["Date/Time Opened"].dt.date <= end_date)
]

# --- Compute Average Response Time for Email ---
avg_resp_hrs = email_sla_period["Elapsed Time (Hours)"].mean() if len(email_sla_period) else 0
avg_resp_secs = avg_resp_hrs * 3600

# --- Compute Average Wait Time for Answered Chats ---
answered_chats_df = chat_sla_period[chat_sla_period["Wait Time"].notna()]
avg_chat_wait_time = answered_chats_df["Wait Time"].mean() if len(answered_chats_df) > 0 else 0

# --- Compute Agent-minute Availability & Handling ---
# Define window
window_start = datetime.combine(start_date, datetime.min.time())
window_end   = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

# Chat availability
chat_avail_secs = 0
for _, row in df_presence.iterrows():
    if row["Service Presence Status: Developer Name"] in ("Available_Chat","Available_All"):
        start = max(row["Start DT"], window_start)
        end   = min(row["End DT"],   window_end)
        if end > start:
            chat_avail_secs += (end - start).total_seconds()

# Email availability
email_avail_secs = 0
for _, row in df_presence.iterrows():
    if row["Service Presence Status: Developer Name"] in ("Available_Email_and_Web","Available_All"):
        start = max(row["Start DT"], window_start)
        end   = min(row["End DT"],   window_end)
        if end > start:
            email_avail_secs += (end - start).total_seconds()

# Chat handling (sum of chat item durations)
chat_handle_secs = chat_df["Duration_sec"].sum()

# Email handling (sum of email item durations)
email_handle_secs = email_df["Duration_sec"].sum()

chat_util  = chat_handle_secs  / chat_avail_secs  if chat_avail_secs  else 0
email_util = email_handle_secs / email_avail_secs if email_avail_secs else 0

# --- Build daily SLA scores & volumes ---
days = pd.date_range(start_date, end_date).date
daily = []
for d in days:
    cd  = chat_sla_period[chat_sla_period["Date/Time Opened"].dt.date==d]
    cw  = cd[cd["Wait Time"].notna()]; v_c = len(cw)
    pct60 = (cw["Wait Time"]<=60).sum()/v_c*100 if v_c else 0
    avg_w = cw["Wait Time"].mean()/60 if v_c else 0
    ar    = (cd["Abandoned After"]>20).sum()/len(cd)*100 if len(cd) else 0
    sla_c = max(0, min(100, ((0.5*pct60 - 0.3*avg_w - 0.2*ar)/56.25)*100))

    ed  = email_sla_period[email_sla_period["Date/Time Opened"].dt.date==d]
    v_e = len(ed); pct1 = (ed["Elapsed Time (Hours)"]<=1).sum()/v_e*100 if v_e else 0
    avg_e = ed["Elapsed Time (Hours)"].mean() if v_e else 0
    sla_e = max(0, min(100, ((0.6*pct1 - 0.4*avg_e)/56.25)*100))

    daily.append({"Date": pd.to_datetime(d),
                  "Chat SLA": sla_c, "Chat Vol": v_c,
                  "Email SLA": sla_e, "Email Vol": v_e})
df_daily = pd.DataFrame(daily)

# --- Compute daily-weighted summary SLA ---
chat_weighted  = (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum()/df_daily["Chat Vol"].sum() if df_daily["Chat Vol"].sum() else 0
email_weighted = (df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/df_daily["Email Vol"].sum() if df_daily["Email Vol"].sum() else 0
weighted_sla   = ((df_daily["Chat SLA"]*df_daily["Chat Vol"]+df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/
                  (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum()) if (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum() else 0

# --- Helper for Conditional Metric Styling ---
def get_utilization_color(utilization_value):
    """Returns a hex color based on utilization percentage."""
    if utilization_value >= 0.50:
        return "#4CAF50" # Green
    elif 0.30 <= utilization_value < 0.50:
        return "#FFC107" # Amber
    else:
        return "#F44336" # Red

def get_email_resp_time_color(avg_seconds):
    """Returns a hex color based on average email response time."""
    if avg_seconds > 59 * 60: # Greater than 59 minutes
        return "#F44336" # Red
    else:
        return "#4CAF50" # Green

def get_chat_wait_time_color(avg_seconds):
    """Returns a hex color based on average chat wait time."""
    if avg_seconds > 30: # Greater than 30 seconds
        return "#F44336" # Red
    else:
        return "#4CAF50" # Green

def get_sla_score_color(score):
    """Returns a hex color based on SLA score."""
    if score >= 80:
        return "#4CAF50" # Green
    elif 70 <= score < 80:
        return "#FFC107" # Orange
    else: # score < 70
        return "#F44336" # Red

def render_custom_metric(col_object, label, value, help_text, border_color):
    """Renders a metric with custom border color and tooltip, allowing text resize and preventing wrap."""
    with col_object:
        st.markdown(f"""
        <div title="{help_text}" style="
            background-color: #f0f2f6;
            border-left: 5px solid {border_color};
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 10px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
        ">
            <div style="font-size: clamp(0.8em, 1.5vw, 1.1em); font-weight: 600; color: #555; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div>
            <div style="font-size: clamp(1.5em, 3.5vw, 2.2em); font-weight: bold; color: #333; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# --- UI: Header & KPI Tiles ---
st.title("Malawi CS Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.markdown("---")

# Core metrics
st.subheader("Core Metrics")
c1, c2, c3, c4 = st.columns(4)
render_custom_metric(c1, "Total Chats",         chat_total, "Total number of chat interactions", "#4CAF50")
render_custom_metric(c2, "Total Emails",        email_total, "Total number of email interactions", "#4CAF50")
render_custom_metric(c3, "Avg Chat AHT (mm:ss)", fmt_mmss(chat_aht), "Average Handle Time for chats", "#4CAF50")
render_custom_metric(c4, "Avg Email AHT (mm:ss)", fmt_mmss(email_aht), "Average Handle Time for emails", "#4CAF50")

# Additional metrics
st.markdown("---")
st.subheader("Operational Metrics")
m1, m2, m3, m4 = st.columns(4)
chat_util_color = get_utilization_color(chat_util)
render_custom_metric(m1, "Chat Utilization",       f"{chat_util:.1%}", "Percentage of time agents spend handling chats when available", chat_util_color)

email_util_color = get_utilization_color(email_util)
render_custom_metric(m2, "Email Utilization",      f"{email_util:.1%}", "Percentage of time agents spend handling emails when available", email_util_color)

email_resp_time_color = get_email_resp_time_color(avg_resp_secs)
render_custom_metric(m3, "Avg Email Resp Time",    fmt_hms(avg_resp_secs), "Average response time for emails", email_resp_time_color)

chat_wait_time_color = get_chat_wait_time_color(avg_chat_wait_time)
render_custom_metric(m4, "Avg Chat Wait Time (mm:ss)", fmt_mmss(avg_chat_wait_time), "Average wait time for answered chats", chat_wait_time_color)


st.markdown("---")
st.markdown("SLA Score Summary")
s1, s2, s3 = st.columns(3)
chat_sla_color = get_sla_score_color(chat_weighted)
render_custom_metric(s1, "Chat SLA Score",   f"{chat_weighted:.1f}", "Service Level Agreement score for chats", chat_sla_color)

email_sla_color = get_sla_score_color(email_weighted)
render_custom_metric(s2, "Email SLA Score",  f"{email_weighted:.1f}", "Service Level Agreement score for emails", email_sla_color)

weighted_sla_color = get_sla_score_color(weighted_sla)
render_custom_metric(s3, "Weighted SLA",     f"{weighted_sla:.1f}", "Overall weighted SLA score across all channels", weighted_sla_color)

# --- Weighted SLA Trend Chart ---
st.markdown("---")
st.subheader("Weighted SLA Trend")

trend = pd.DataFrame({
    "Date":        df_daily["Date"],
    "Weighted SLA": (df_daily["Chat SLA"]*df_daily["Chat Vol"] + df_daily["Email SLA"]*df_daily["Email Vol"]) /
                    (df_daily["Chat Vol"] + df_daily["Email Vol"])
})

# Convert start_date and end_date to datetime objects for consistent domain definition
# Add extra padding for the x-axis to ensure first/last points are not clipped
x_min_bound = datetime.combine(start_date, datetime.min.time()) - timedelta(days=0.5)
x_max_bound = datetime.combine(end_date, datetime.max.time()) + timedelta(days=0.5)

chart = (
    alt.Chart(trend)
    .mark_line(point={
        "filled": True,
        "fill": "white",
        "size": 80,
        "strokeWidth": 2,
        "stroke": "#2F80ED" # Point border color
    }, color="#2F80ED") # Line color
    .encode(
        x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%d %b", labelAngle=-45, tickCount='day'),
                scale=alt.Scale(domain=[x_min_bound, x_max_bound])),
        y=alt.Y("Weighted SLA:Q", title="Weighted SLA Score", scale=alt.Scale(domain=[0, 105])),
        tooltip=[
            alt.Tooltip("Date:T", format="%d %b"),
            alt.Tooltip("Weighted SLA:Q", format=".1f", title="SLA Score")
        ]
    ).properties(
        title="Daily Weighted SLA Performance"
    ).interactive()
)

labels  = chart.mark_text(dy=-10, color="#2F80ED").encode(text=alt.Text("Weighted SLA:Q", format=".1f"))
rule    = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(color="red", strokeDash=[5,5], size=2).encode(y="y:Q")
rule_lb = alt.Chart(pd.DataFrame({"y":[80]})).mark_text(align="left", color="red", dy=-8)\
            .encode(
                y="y:Q",
                text=alt.value("Target: 80%"),
                x=alt.value(0) # Align to the far left (pixel coordinate 0)
            )

st.altair_chart((chart+labels+rule+rule_lb).properties(width=700, height=350),
                use_container_width=True)

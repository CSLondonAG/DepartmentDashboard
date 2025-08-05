import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Paths & Sanity Check ---
BASE_DIR   = Path(__file__).parent
chat_path  = BASE_DIR / "chat.csv"
email_path = BASE_DIR / "email.csv"
if not chat_path.exists() or not email_path.exists():
    st.error("Make sure chat.csv and email.csv are alongside this script.")
    st.stop()

# --- Load Data ---
def load_data():
    df_items     = pd.read_csv("report_items.csv",    dayfirst=True, parse_dates=["Start DT","End DT"])
    df_presence  = pd.read_csv("report_presence.csv", dayfirst=True, parse_dates=["Start DT","End DT"])
    df_shifts    = pd.read_csv("shifts.csv")
    chat_sla_df  = pd.read_csv(chat_path,             dayfirst=True, parse_dates=["Date/Time Opened"])
    email_sla_df = pd.read_csv(email_path,            dayfirst=True,
                                parse_dates=["Date/Time Opened","Completion Date"])
    for df in (df_items, df_presence, df_shifts, chat_sla_df, email_sla_df):
        df.columns = df.columns.str.strip()
    return df_items, df_presence, df_shifts, chat_sla_df, email_sla_df

df_items, df_presence, df_shifts, chat_sla_df, email_sla_df = load_data()

# --- Sidebar: Date Range (from chat.csv only) ---
st.sidebar.header("Filter Options")
min_date = chat_sla_df["Date/Time Opened"].dt.date.min()
max_date = chat_sla_df["Date/Time Opened"].dt.date.max()

start_date = st.sidebar.date_input(
    "Start Date",
    value=max_date - timedelta(days=6),
    min_value=min_date,
    max_value=max_date
)
end_date   = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)
if start_date > end_date:
    st.sidebar.error("Start must be on or before End")
    st.stop()

# --- Helpers ---
def fmt_mmss(sec):
    if pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

# --- Volumes & AHT from report_items ---
mask      = ((df_items["Start DT"].dt.date >= start_date) &
             (df_items["Start DT"].dt.date <= end_date))
df_period = df_items[mask].copy()
df_period["Duration_sec"] = (df_period["End DT"] - df_period["Start DT"]).dt.total_seconds()

chat_df  = df_period[df_period["Service Channel: Developer Name"]=="sfdc_liveagent"]
email_df = df_period[df_period["Service Channel: Developer Name"]=="casesChannel"]

chat_total  = len(chat_df)
email_total = len(email_df)
chat_aht    = chat_df["Duration_sec"].mean()  if chat_total  else None
email_aht   = email_df["Duration_sec"].mean() if email_total else None

# --- SLA slices by Date/Time Opened ---
chat_sla_period  = chat_sla_df[
    (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
]
email_sla_period = email_sla_df[
    (email_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (email_sla_df["Date/Time Opened"].dt.date <= end_date)
]

# --- Avg Email Response Time (secs) ---
avg_resp_hrs  = email_sla_period["Elapsed Time (Hours)"].mean() if len(email_sla_period) else 0
avg_resp_secs = avg_resp_hrs * 3600

# --- Compute availability totals ---
window_start = datetime.combine(start_date, datetime.min.time())
window_end   = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

chat_pres_segments = df_presence[
    df_presence["Service Presence Status: Developer Name"].isin(["Available_Chat","Available_All"])
]
email_pres_segments= df_presence[
    df_presence["Service Presence Status: Developer Name"].isin(["Available_Email_and_Web","Available_All"])
]

chat_avail_secs = 0.0
for _, pres in chat_pres_segments.iterrows():
    seg_start = max(pres["Start DT"], window_start)
    seg_end   = min(pres["End DT"],   window_end)
    if seg_end > seg_start:
        chat_avail_secs += (seg_end - seg_start).total_seconds()

email_avail_secs = 0.0
for _, pres in email_pres_segments.iterrows():
    seg_start = max(pres["Start DT"], window_start)
    seg_end   = min(pres["End DT"],   window_end)
    if seg_end > seg_start:
        email_avail_secs += (seg_end - seg_start).total_seconds()

# --- Compute handled seconds via exact intersection per agent (updated) ---
chat_handle_secs = 0.0

# Build presence segments per agent
chat_pres_by_agent = {
    agent: list(zip(
        grp["Start DT"].dt.to_pydatetime().tolist(),
        grp["End DT"].dt.to_pydatetime().tolist()
    ))
    for agent, grp in chat_pres_segments.groupby("Created By: Full Name")
}

# For each agent, merge their chat‐handling intervals, then intersect with availability
for agent, group in chat_df.groupby("User: Full Name"):
    # 1) merge handling intervals
    handle_ints = list(zip(
        group["Start DT"].dt.to_pydatetime().tolist(),
        group["End DT"].dt.to_pydatetime().tolist()
    ))
    merged_handles = merge_intervals(handle_ints)

    # 2) intersect with each presence chunk
    for h_start, h_end in merged_handles:
        for p_start, p_end in chat_pres_by_agent.get(agent, []):
            o_s = max(h_start, p_start)
            o_e = min(h_end,   p_end)
            if o_e > o_s:
                chat_handle_secs += (o_e - o_s).total_seconds()

# Email handling remains unchanged
email_handle_secs = 0.0
email_pres_by_agent = {
    agent: list(zip(
        grp["Start DT"].dt.to_pydatetime().tolist(),
        grp["End DT"].dt.to_pydatetime().tolist()
    ))
    for agent, grp in email_pres_segments.groupby("Created By: Full Name")
}
for agent, group in email_df.groupby("User: Full Name"):
    intervals = list(zip(
        group["Start DT"].dt.to_pydatetime().tolist(),
        group["End DT"].dt.to_pydatetime().tolist()
    ))
    for stt, endt in merge_intervals(intervals):
        for ps, pe in email_pres_by_agent.get(agent, []):
            o_s = max(stt, ps)
            o_e = min(endt, pe)
            if o_e > o_s:
                email_handle_secs += (o_e - o_s).total_seconds()

# --- Utilization ---
chat_util  = chat_handle_secs  / chat_avail_secs  if chat_avail_secs  else 0
email_util = email_handle_secs / email_avail_secs if email_avail_secs else 0

# --- Build daily SLA & volumes (unchanged) ---
days = pd.date_range(start_date, end_date).date
daily = []
for d in days:
    cd   = chat_sla_period[chat_sla_period["Date/Time Opened"].dt.date == d]
    cw   = cd[cd["Wait Time"].notna()]; v_c = len(cw)
    pct60= (cw["Wait Time"] <= 60).sum()/v_c*100 if v_c else 0
    avg_w= cw["Wait Time"].mean()/60 if v_c else 0
    ar   = (cd["Abandoned After"] > 20).sum()/len(cd)*100 if len(cd) else 0
    sla_c= max(0, min(100, ((0.5*pct60 - 0.3*avg_w - 0.2*ar)/56.25)*100))

    ed   = email_sla_period[email_sla_period["Date/Time Opened"].dt.date == d]
    v_e  = len(ed); pct1= (ed["Elapsed Time (Hours)"] <= 1).sum()/v_e*100 if v_e else 0
    avg_e= ed["Elapsed Time (Hours)"].mean() if v_e else 0
    sla_e= max(0, min(100, ((0.6*pct1 - 0.4*avg_e)/56.25)*100))

    daily.append({
        "Date": pd.to_datetime(d),
        "Chat SLA": sla_c,   "Chat Vol": v_c,
        "Email SLA": sla_e,  "Email Vol": v_e
    })

df_daily = pd.DataFrame(daily)
df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"] * df_daily["Chat Vol"] +
    df_daily["Email SLA"] * df_daily["Email Vol"]
) / (df_daily["Chat Vol"] + df_daily["Email Vol"])

# --- Summary SLA Scores & UI rendering (unchanged) ---
chat_weighted  = (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum()/df_daily["Chat Vol"].sum()  if df_daily["Chat Vol"].sum() else 0
email_weighted = (df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/df_daily["Email Vol"].sum() if df_daily["Email Vol"].sum() else 0
total_vol      = (df_daily["Chat Vol"] + df_daily["Email Vol"]).sum()
weighted_sla   = ((df_daily["Chat SLA"]*df_daily["Chat Vol"] + df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/total_vol) if total_vol else 0

st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Chats",chat_total)
c2.metric("Total Emails",email_total)
c3.metric("Avg Chat AHT (mm:ss)",fmt_mmss(chat_aht))
c4.metric("Avg Email AHT (mm:ss)",fmt_mmss(email_aht))
st.markdown("---")
m1,m2,m3 = st.columns(3)
m1.metric("Chat Utilization",f"{chat_util:.1%}")
m2.metric("Email Utilization",f"{email_util:.1%}")
m3.metric("Avg Email Resp Time",fmt_mmss(avg_resp_secs))
st.markdown("---")
s1,s2,s3 = st.columns(3)
s1.metric("Chat SLA Score",f"{chat_weighted:.1f}")
s2.metric("Email SLA Score",f"{email_weighted:.1f}")
s3.metric("Weighted SLA Score",f"{weighted_sla:.1f}")
# (chart code omitted for brevity)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta
from pathlib import Path

# --- Page Config ---
st.set_page_config(page_title="Department Performance Dashboard", layout="wide")

# --- Paths & Existence Check ---
BASE_DIR   = Path(__file__).parent
chat_path  = BASE_DIR / "chat.csv"
email_path = BASE_DIR / "email.csv"

if not chat_path.exists() or not email_path.exists():
    st.error("Make sure chat.csv and email.csv are in the same folder as this script.")
    st.stop()

# --- Load Data (no caching) ---
def load_data():
    df_items = pd.read_csv(
        "report_items.csv",
        dayfirst=True,
        parse_dates=["Start DT", "End DT"]
    )
    df_presence = pd.read_csv(
        "report_presence.csv",
        dayfirst=True,
        parse_dates=["Start DT", "End DT"]
    )
    df_shifts = pd.read_csv("shifts.csv")

    chat_sla_df = pd.read_csv(
        chat_path,
        dayfirst=True,
        parse_dates=["Date/Time Opened"]
    )
    email_sla_df = pd.read_csv(
        email_path,
        dayfirst=True,
        parse_dates=["Date/Time Opened","Completion Date"]
    )

    # strip whitespace from all column names
    for df in (df_items, df_presence, df_shifts, chat_sla_df, email_sla_df):
        df.columns = df.columns.str.strip()

    # trim key string columns
    for df, cols in [
        (df_items,    ["User: Full Name","Service Channel: Developer Name"]),
        (df_presence, ["Created By: Full Name","Service Presence Status: Developer Name"])
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    return df_items, df_presence, df_shifts, chat_sla_df, email_sla_df

df_items, df_presence, df_shifts, chat_sla_df, email_sla_df = load_data()

# --- Sidebar: Date Range ---
st.sidebar.header("Filter Options")
min_date = df_items["Start DT"].dt.date.min()
max_date = df_items["Start DT"].dt.date.max()

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

# --- Format Helpers ---
def fmt_mmss(sec):
    if pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def fmt_hhmm(sec):
    if pd.isna(sec): return "–"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h:02}:{m:02}"

# --- Volumes & AHT (report_items) ---
period_mask = (
    (df_items["Start DT"].dt.date >= start_date) &
    (df_items["Start DT"].dt.date <= end_date)
)
df_period = df_items[period_mask].copy()
df_period["Duration_sec"] = (df_period["End DT"] - df_period["Start DT"]).dt.total_seconds()

chat_df  = df_period[df_period["Service Channel: Developer Name"]=="sfdc_liveagent"]
email_df = df_period[df_period["Service Channel: Developer Name"]=="casesChannel"]

chat_total  = len(chat_df)
email_total = len(email_df)

chat_aht  = chat_df["Duration_sec"].mean()  if chat_total  else None
email_aht = email_df["Duration_sec"].mean() if email_total else None

# --- SLA slices by Date/Time Opened ---
chat_sla_period = chat_sla_df[
    (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
]
email_sla_period = email_sla_df[
    (email_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (email_sla_df["Date/Time Opened"].dt.date <= end_date)
]

# --- Build daily SLA & volumes ---
days = pd.date_range(start_date, end_date).date
daily = []
for d in days:
    # Chat SLA
    cd    = chat_sla_period[chat_sla_period["Date/Time Opened"].dt.date==d]
    cw    = cd[cd["Wait Time"].notna()]
    v_c   = len(cw)
    pct60 = (cw["Wait Time"]<=60).sum()/v_c*100 if v_c else 0
    avg_w = cw["Wait Time"].mean()/60 if v_c else 0
    ar    = (cd["Abandoned After"]>20).sum()/len(cd)*100 if len(cd) else 0
    sla_c = max(0, min(100, ((0.5*pct60 - 0.3*avg_w - 0.2*ar)/56.25)*100))

    # Email SLA
    ed    = email_sla_period[email_sla_period["Date/Time Opened"].dt.date==d]
    v_e   = len(ed)
    pct1  = (ed["Elapsed Time (Hours)"]<=1).sum()/v_e*100 if v_e else 0
    avg_e = ed["Elapsed Time (Hours)"].mean() if v_e else 0
    sla_e = max(0, min(100, ((0.6*pct1 - 0.4*avg_e)/56.25)*100))

    daily.append({
        "Date": pd.to_datetime(d),
        "Chat SLA": sla_c,
        "Chat Vol": v_c,
        "Email SLA": sla_e,
        "Email Vol": v_e
    })

df_daily = pd.DataFrame(daily)

# --- Compute KPI SLAs (daily-weighted) ---
chat_weighted  = (
    (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum() /
    df_daily["Chat Vol"].sum()
) if df_daily["Chat Vol"].sum() else 0

email_weighted = (
    (df_daily["Email SLA"]*df_daily["Email Vol"]).sum() /
    df_daily["Email Vol"].sum()
) if df_daily["Email Vol"].sum() else 0

weighted_sla = (
    (df_daily["Chat SLA"]*df_daily["Chat Vol"] +
     df_daily["Email SLA"]*df_daily["Email Vol"]).sum() /
    (df_daily["Chat Vol"] + df_daily["Email Vol"]).sum()
) if (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum() else 0

# --- UI: Header & KPI Tiles ---
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Chats",   chat_total)
c2.metric("Total Emails",  email_total)
c3.metric("Avg Chat AHT",  fmt_mmss(chat_aht))
c4.metric("Avg Email AHT", fmt_mmss(email_aht))

st.markdown("---")
st.markdown("### 🎯 SLA Score Summary")
s1, s2, s3 = st.columns(3)
s1.metric("Chat SLA Score",  f"{chat_weighted:.1f}")
s2.metric("Email SLA Score", f"{email_weighted:.1f}")
s3.metric("Weighted SLA",    f"{weighted_sla:.1f}")

# --- Weighted SLA Trend Chart ---
st.markdown("---")
st.subheader("📈 Weighted SLA Trend")

trend = pd.DataFrame({
    "Date":        df_daily["Date"],
    "Weighted SLA": (
        df_daily["Chat SLA"]*df_daily["Chat Vol"] +
        df_daily["Email SLA"]*df_daily["Email Vol"]
    ) / (df_daily["Chat Vol"] + df_daily["Email Vol"])
})

chart = (
    alt.Chart(trend)
    .mark_line(point=True)
    .encode(
        x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%d %b", labelAngle=-45)),
        y=alt.Y("Weighted SLA:Q", title="Weighted SLA"),
        tooltip=[
            alt.Tooltip("Date:T", format="%d %b"),
            alt.Tooltip("Weighted SLA:Q", format=".1f")
        ]
    )
)

labels  = chart.mark_text(dy=-10).encode(text=alt.Text("Weighted SLA:Q", format=".1f"))
rule    = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(color="red").encode(y="y:Q")
rule_lb = alt.Chart(pd.DataFrame({"y":[80]})).mark_text(align="left",dx=5,dy=-5,color="red")\
            .encode(y="y:Q",text=alt.value("Target: 80"))

st.altair_chart((chart+labels+rule+rule_lb).properties(width=700,height=300),
                use_container_width=True)

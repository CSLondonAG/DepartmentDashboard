import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path

# --- Page Config & Custom CSS ---
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
}
.metric-container {
    padding: 12px;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    border: 2px solid transparent;
}
.metric-container-warning {
    border-color: #ff4d4d;
}
.metric-title {
    font-size: 1.1em;
    color: #333333;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.8em;
    font-weight: bold;
    color: #007bff;
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

def render_custom_metric(container, title, value, tooltip, color):
    container.markdown(f"""
        <div class="metric-container" style="border-color:{color}" title="{tooltip}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def get_utilization_color(util):
    if util >= 0.50: return "#4CAF50"
    elif util >= 0.30: return "#FFC107"
    else: return "#F44336"

def get_email_resp_time_color(sec):
    return "#F44336" if sec > 59*60 else "#4CAF50"

def get_sla_score_color(score):
    if score >= 80: return "#4CAF50"
    elif score >= 70: return "#FFC107"
    else: return "#F44336"

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

# --- Core Metrics: Volume & AHT ---
mask      = ((df_items["Start DT"].dt.date >= start_date) &
             (df_items["Start DT"].dt.date <= end_date))
df_period = df_items[mask].copy()
df_period["Duration_sec"] = (df_period["End DT"] - df_period["Start DT"]).dt.total_seconds()

chat_df   = df_period[df_period["Service Channel: Developer Name"]=="sfdc_liveagent"]
email_df  = df_period[df_period["Service Channel: Developer Name"]=="casesChannel"]

chat_total  = len(chat_df)
email_total = len(email_df)
chat_aht    = chat_df["Duration_sec"].mean()  if chat_total  else None
email_aht   = email_df["Duration_sec"].mean() if email_total else None

# --- SLA slices & response times ---
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

# --- Availability totals ---
window_start = datetime.combine(start_date, datetime.min.time())
window_end   = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

chat_pres_segments = df_presence[
    df_presence["Service Presence Status: Developer Name"].isin(["Available_Chat","Available_All"])
]
email_pres_segments = df_presence[
    df_presence["Service Presence Status: Developer Name"].isin(["Available_Email_and_Web","Available_All"])
]

chat_avail_secs = sum(
    max(0, (min(p["End DT"], window_end) - max(p["Start DT"], window_start)).total_seconds())
    for _, p in chat_pres_segments.iterrows()
)
email_avail_secs = sum(
    max(0, (min(p["End DT"], window_end) - max(p["Start DT"], window_start)).total_seconds())
    for _, p in email_pres_segments.iterrows()
)

# --- Exact‐intersection chat handling (NEW BLOCK) ---
def merge_intervals(ints):
    ints = sorted(ints, key=lambda x: x[0])
    out = []
    for s,e in ints:
        if not out or s > out[-1][1]:
            out.append([s,e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

chat_handle_secs = 0.0
# build per-agent presence
chat_pres_by_agent = {
    ag: list(zip(grp["Start DT"].dt.to_pydatetime(),
                 grp["End DT"].dt.to_pydatetime()))
    for ag,grp in chat_pres_segments.groupby("Created By: Full Name")
}
for ag,grp in chat_df.groupby("User: Full Name"):
    # merge their handle intervals
    ints = list(zip(grp["Start DT"].dt.to_pydatetime(),
                    grp["End DT"].dt.to_pydatetime()))
    for hs,he in merge_intervals(ints):
        for ps,pe in chat_pres_by_agent.get(ag, []):
            os_ = max(hs, ps); oe = min(he, pe)
            if oe > os_:
                chat_handle_secs += (oe-os_).total_seconds()

# email handling (unchanged)
email_handle_secs = email_df["Duration_sec"].sum()

# --- Utilization ---
chat_util  = chat_handle_secs  / chat_avail_secs  if chat_avail_secs  else 0
email_util = email_handle_secs / email_avail_secs if email_avail_secs else 0

# --- Build per-day SLA & volumes ---
daily = []
for d in pd.date_range(start_date, end_date):
    dd = d.normalize()
    # chat SLA & vol
    cd  = chat_sla_p[chat_sla_p["Date/Time Opened"].dt.date == dd.date()]
    cw  = cd[cd["Wait Time"].notna()]; v_c = len(chat_df[chat_df["Start DT"].dt.date==dd.date()])
    pct60 = (cw["Wait Time"]<=60).sum()/len(cw)*100 if len(cw) else 0
    avg_w  = cw["Wait Time"].mean()/60 if len(cw) else 0
    ar     = (cd["Abandoned After"]>20).sum()/len(cd)*100 if len(cd) else 0
    sla_c  = max(0, min(100, ((0.5*pct60 - 0.3*avg_w - 0.2*ar)/56.25)*100))
    # email SLA & vol
    ed  = email_sla_p[email_sla_p["Date/Time Opened"].dt.date == dd.date()]
    v_e = len(email_df[email_df["Start DT"].dt.date==dd.date()])
    pct1= (ed["Elapsed Time (Hours)"]<=1).sum()/len(ed)*100 if len(ed) else 0
    avg_e= ed["Elapsed Time (Hours)"].mean() if len(ed) else 0
    sla_e= max(0, min(100, ((0.6*pct1 - 0.4*avg_e)/56.25)*100))
    daily.append({"Date":dd, "Chat SLA":sla_c, "Chat Vol":v_c,
                  "Email SLA":sla_e,"Email Vol":v_e})
df_daily = pd.DataFrame(daily)
df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"]*df_daily["Chat Vol"] +
    df_daily["Email SLA"]*df_daily["Email Vol"]
) / (df_daily["Chat Vol"]+df_daily["Email Vol"])

# --- Summary SLA Scores ---
chat_weighted  = (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum()  / df_daily["Chat Vol"].sum()
email_weighted = (df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/ df_daily["Email Vol"].sum()
weighted_sla   = (
    (df_daily["Chat SLA"]*df_daily["Chat Vol"]+
     df_daily["Email SLA"]*df_daily["Email Vol"]).sum()
  / (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum()
)

# --- UI: Header & Metrics ---
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.markdown("---")

# Core Metrics
st.subheader("Core Metrics")
c1,c2,c3,c4=st.columns(4)
render_custom_metric(c1,"Total Chats",chat_total,"Total chat interactions","#4CAF50")
render_custom_metric(c2,"Total Emails",email_total,"Total email interactions","#4CAF50")
render_custom_metric(c3,"Avg Chat AHT (mm:ss)",fmt_mmss(chat_aht),"Average chat handle time","#4CAF50")
render_custom_metric(c4,"Avg Email AHT (mm:ss)",fmt_mmss(email_aht),"Average email handle time","#4CAF50")

# Operational Metrics
st.markdown("---")
st.subheader("Operational Metrics")
m1,m2,m3=st.columns(3)
render_custom_metric(m1,"Chat Utilization",f"{chat_util:.1%}","Agent-minute chat utilization",get_utilization_color(chat_util))
render_custom_metric(m2,"Email Utilization",f"{email_util:.1%}","Agent-minute email utilization",get_utilization_color(email_util))
render_custom_metric(m3,"Avg Chat Wait (sec)",f"{avg_chat_wait:.1f}","Average chat wait time","#4CAF50")

# SLA Score Summary
st.markdown("---")
st.subheader("🎯 SLA Score Summary")
s1,s2,s3=st.columns(3)
render_custom_metric(s1,"Chat SLA Score",f"{chat_weighted:.1f}","Weighted chat SLA",get_sla_score_color(chat_weighted))
render_custom_metric(s2,"Email SLA Score",f"{email_weighted:.1f}","Weighted email SLA",get_sla_score_color(email_weighted))
render_custom_metric(s3,"Weighted SLA Score",f"{weighted_sla:.1f}","Overall weighted SLA",get_sla_score_color(weighted_sla))

# Weighted SLA Trend Chart
st.markdown("---")
st.subheader("Weighted SLA Trend")
x_min = datetime.combine(start_date,datetime.min.time())-timedelta(days=0.5)
x_max = datetime.combine(end_date,  datetime.max.time())+timedelta(days=0.5)
trend = df_daily[["Date","Weighted SLA"]].sort_values("Date")
chart = (
    alt.Chart(trend)
    .mark_line(point=True,color="#2F80ED")
    .encode(
        x=alt.X("Date:T",axis=alt.Axis(format="%d %b",labelAngle=-45,tickCount="day"),
                scale=alt.Scale(domain=[x_min,x_max])),
        y=alt.Y("Weighted SLA:Q",scale=alt.Scale(domain=[0,105])),
        tooltip=[alt.Tooltip("Date:T",format="%d %b"), alt.Tooltip("Weighted SLA:Q",format=".1f")]
    )
)
labels = chart.mark_text(dy=-10,color="#2F80ED").encode(text=alt.Text("Weighted SLA:Q",format=".1f"))
rule   = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(color="red",strokeDash=[5,5]).encode(y="y:Q")
rule_lb= alt.Chart(pd.DataFrame({"y":[80]})).mark_text(align="left",color="red",dy=-8)\
            .encode(y="y:Q",text=alt.value("Target: 80%"))
st.altair_chart((chart+labels+rule+rule_lb).properties(width=700,height=350),use_container_width=True)

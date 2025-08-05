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

# Reverting CSS to a flexible height for metric containers
st.markdown("""
<style>
/* Main container styling with a light grey background */
.main .block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    background-color: #eef2f6;
}

/* Metric card styling with a gradient and hover effect */
.metric-container {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #f9f9f9, #ffffff);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    text-align: center;
    transition: all 0.3s ease-in-out;
    border-left: 5px solid transparent;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-container:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}

/* Specific metric card border colors */
.metric-container.success { border-color: #1abc9c; } /* Turquoise */
.metric-container.warning { border-color: #f39c12; } /* Orange */
.metric-container.danger  { border-color: #e74c3c; } /* Red */
.metric-container.info    { border-color: #3498db; } /* Blue for general stats */

/* Metric titles and values with a new font */
.metric-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.2em;
    font-weight: 600;
    color: #555;
    margin-bottom: 5px;
    line-height: 1.2em;
    overflow: hidden;
}
.metric-value {
    font-family: 'Montserrat', sans-serif;
    font-size: 2.5em;
    font-weight: 800;
    color: #444;
}

/* Header and subheader styling with a new font and color */
h1, h2, h3 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
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
    return f"{h:02}:{m:02}"

# Reverting renderer to use a single font size and no fixed height
def render_custom_metric(container, title, value, tooltip, border_class):
    container.markdown(f"""
        <div class="metric-container {border_class}" title="{tooltip}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# Updated color functions to return CSS classes
def get_utilization_color(util):
    if util >= 0.50: return "success"
    elif util >= 0.30: return "warning"
    else: return "danger"

def get_email_resp_time_color(sec):
    return "danger" if sec > 59*60 else "success"

def get_sla_score_color(score):
    if score >= 80: return "success"
    elif score >= 70: return "warning"
    else: return "danger"

def get_survey_score_color(score):
    if score >= 8: return "success"
    elif score >= 6: return "warning"
    else: return "danger"

def get_nps_color(nps_score):
    if nps_score >= 50: return "success"
    elif nps_score >= 0: return "warning"
    else: return "danger"

# --- Load & preprocess data ---
BASE_DIR   = Path(__file__).parent
chat_path  = BASE_DIR / "chat.csv"
email_path = BASE_DIR / "email.csv"
survey_path= BASE_DIR / "scored_surveys.csv"

if not chat_path.exists() or not email_path.exists():
    st.error("Please place chat.csv and email.csv beside this script.")
    st.stop()
if not survey_path.exists():
    st.error("Please place scored_surveys.csv beside this script.")
    st.stop()


df_items     = pd.read_csv("report_items.csv",    dayfirst=True, parse_dates=["Start DT","End DT"])
df_presence  = pd.read_csv("report_presence.csv", dayfirst=True, parse_dates=["Start DT","End DT"])
df_shifts    = pd.read_csv("shifts.csv")
chat_sla_df  = pd.read_csv(chat_path,  dayfirst=True, parse_dates=["Date/Time Opened"])
email_sla_df = pd.read_csv(email_path, dayfirst=True, parse_dates=["Date/Time Opened","Completion Date"])
df_surveys   = pd.read_csv(survey_path, dayfirst=True, parse_dates=["Survey Taker: Created Date"])

for df in (df_items, df_presence, df_shifts, chat_sla_df, email_sla_df, df_surveys):
    df.columns = df.columns.str.strip()

# --- NEW: Data Preprocessing for Survey Scores ---
# Define the two recommendation questions to be combined
rec_questions = [
    "How likely are you to recommend Premier Bet to a friend or colleague?",
    "How likely would you to to recommend Mercury Bet to a friend or colleague?"
]
combined_rec_title = "Combined Recommendation Score"
df_surveys.loc[df_surveys["Survey Question: Question Title"].isin(rec_questions),
               "Survey Question: Question Title"] = combined_rec_title

# Identify unique survey question titles for dynamic processing
survey_questions = df_surveys["Survey Question: Question Title"].dropna().unique()

# Create a sanitized dictionary to map original question titles to safe column names
survey_question_cols = {
    q: q.replace(':', '').replace(' ', '_').replace('?', '').replace('-', '_').strip()
    for q in survey_questions
}
# Define shorter labels for the survey questions
short_survey_labels = {
    "Combined Recommendation Score": "NPS",
    "Was the issue reported in the chat resolved by the agent?": "Chat Issue Resolved %",
    "Was the issue reported in the email resolved by the agent?": "Email Issue Resolved %"
}

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

    daily_entry = {"Date":dd, "Chat SLA":sla_c, "Chat Vol":v_c,
                   "Email SLA":sla_e,"Email Vol":v_e}

    for q in survey_questions:
        sd = df_surveys[(df_surveys["Survey Question: Question Title"] == q) &
                        (df_surveys["Survey Taker: Created Date"].dt.date == dd.date())]
        # Use the sanitized column name for the daily dataframe
        daily_entry[survey_question_cols[q]] = sd["Survey Score"].mean() if len(sd) > 0 else None

    daily.append(daily_entry)

df_daily = pd.DataFrame(daily)
df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"]*df_daily["Chat Vol"] +
    df_daily["Email SLA"]*df_daily["Email Vol"]
) / (df_daily["Chat Vol"]+df_daily["Email Vol"])

# --- Summary SLA Scores ---
chat_weighted  = (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum()  / df_daily["Chat Vol"].sum() if df_daily["Chat Vol"].sum() > 0 else 0
email_weighted = (df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/ df_daily["Email Vol"].sum() if df_daily["Email Vol"].sum() > 0 else 0
weighted_sla   = (
    (df_daily["Chat SLA"]*df_daily["Chat Vol"]+
     df_daily["Email SLA"]*df_daily["Email Vol"]).sum()
  / (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum()
) if (df_daily["Chat Vol"]+df_daily["Email Vol"]).sum() > 0 else 0

# --- Summary Survey Scores ---
survey_period = df_surveys[
    (df_surveys["Survey Taker: Created Date"].dt.date >= start_date) &
    (df_surveys["Survey Taker: Created Date"].dt.date <= end_date)
]
survey_summary_metrics = {}
for q in survey_questions:
    q_data = survey_period[survey_period["Survey Question: Question Title"] == q].dropna(subset=['Survey Score'])
    count = len(q_data)

    if q == combined_rec_title and count > 0:
        promoters = len(q_data[q_data['Survey Score'] > 8])
        detractors = len(q_data[q_data['Survey Score'] < 7])
        nps_score = ((promoters - detractors) / count) * 100
        survey_summary_metrics[q] = {"nps_score": nps_score, "count": count, "is_nps": True}
    else:
        avg_score = q_data["Survey Score"].mean() if count > 0 else None
        survey_summary_metrics[q] = {"avg_score": avg_score, "count": count, "is_yes_no": "Yes\nNo" in q_data["Survey Question: Choices"].unique() if not q_data.empty else False, "is_nps": False}

# --- UI: Header & Metrics ---
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.markdown("---")

# Core Metrics
st.subheader("Core Metrics")
cols = st.columns(4)
render_custom_metric(cols[0],"Total Chats",chat_total,"Total chat interactions","info")
render_custom_metric(cols[1],"Total Emails",email_total,"Total email interactions","info")
render_custom_metric(cols[2],"Avg Chat AHT (mm:ss)",fmt_mmss(chat_aht),"Average chat handle time","info")
render_custom_metric(cols[3],"Avg Email AHT (mm:ss)",fmt_mmss(email_aht),"Average email handle time","info")

# Operational Metrics
st.markdown("---")
st.subheader("Operational Metrics")
m1,m2,m3,m4 = st.columns(4)
render_custom_metric(m1,"Chat Utilization",f"{chat_util:.1%}","Agent-minute chat utilization",get_utilization_color(chat_util))
render_custom_metric(m2,"Email Utilization",f"{email_util:.1%}","Agent-minute email utilization",get_utilization_color(email_util))
render_custom_metric(m3,"Avg Chat Wait (sec)",f"{avg_chat_wait:.1f}","Average chat wait time","info")
render_custom_metric(m4,"Avg Email Resp Time (hh:mm)",fmt_hms(avg_resp_secs),"Average time to send the first response to an email",get_email_resp_time_color(avg_resp_secs))

# SLA Score Summary
st.markdown("---")
st.subheader("🎯 SLA Score Summary")
s1,s2,s3=st.columns(3)
render_custom_metric(s1,"Chat SLA Score",f"{chat_weighted:.1f}","Weighted chat SLA",get_sla_score_color(chat_weighted))
render_custom_metric(s2,"Email SLA Score",f"{email_weighted:.1f}","Weighted email SLA",get_sla_score_color(email_weighted))
render_custom_metric(s3,"Weighted SLA Score",f"{weighted_sla:.1f}","Overall weighted SLA",get_sla_score_color(weighted_sla))

# Customer Survey Scores (New Section)
st.markdown("---")
st.subheader("Customer Survey Scores")
cols_survey = st.columns(len(survey_questions))
for i, q in enumerate(survey_questions):
    metric = survey_summary_metrics[q]

    # Use the shorter, descriptive label
    title = short_survey_labels.get(q, q)
    tooltip = f"{q} ({metric['count']} responses)"

    if metric.get("is_nps"):
        value = f"{metric['nps_score']:.0f}" if metric["nps_score"] is not None else "N/A"
        color = get_nps_color(metric["nps_score"])
    elif metric.get("is_yes_no"):
        value = f"{metric['avg_score'] * 100:.1f}%" if metric["avg_score"] is not None else "N/A"
        color = get_survey_score_color(metric['avg_score'] * 10) if metric["avg_score"] is not None else "info"
    else: # Default numeric score
        value = f"{metric['avg_score']:.1f}" if metric["avg_score"] is not None else "N/A"
        color = get_survey_score_color(metric['avg_score']) if metric["avg_score"] is not None else "info"

    render_custom_metric(cols_survey[i], title, value, tooltip, color)


# Weighted SLA Trend Chart
st.markdown("---")
st.subheader("Weighted SLA Trend")
x_min = datetime.combine(start_date,datetime.min.time())-timedelta(days=0.5)
x_max = datetime.combine(end_date,  datetime.max.time())+timedelta(days=0.5)
trend = df_daily[["Date","Weighted SLA"]].sort_values("Date")
chart = (
    alt.Chart(trend)
    .mark_line(point=True, color="#3498db")
    .encode(
        x=alt.X("Date:T",axis=alt.Axis(format="%d %b",labelAngle=-45,tickCount="day"),
                scale=alt.Scale(domain=[x_min,x_max])),
        y=alt.Y("Weighted SLA:Q",scale=alt.Scale(domain=[0,105])),
        tooltip=[alt.Tooltip("Date:T",format="%d %b"), alt.Tooltip("Weighted SLA:Q",format=".1f")]
    )
)
labels = chart.mark_text(dy=-10,color="#3498db").encode(text=alt.Text("Weighted SLA:Q",format=".1f"))
rule   = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(color="#e74c3c",strokeDash=[5,5]).encode(y="y:Q")
rule_lb= alt.Chart(pd.DataFrame({"y":[80]})).mark_text(align="left",color="#e74c3c",dy=-8)\
            .encode(y="y:Q",text=alt.value("Target: 80%"))
st.altair_chart((chart+labels+rule+rule_lb).properties(width=700,height=350),use_container_width=True)

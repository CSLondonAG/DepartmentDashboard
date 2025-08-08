import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path

# --- Page Config & Custom CSS (Professional Design) ---
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid #2563eb;
        padding-bottom: 0.75rem;
    }
    
    .period-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--accent-color, #2563eb);
    }
    
    .metric-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        line-height: 1.2;
    }
    
    /* Status-specific colors */
    .metric-excellent { --accent-color: #10b981; }
    .metric-good { --accent-color: #3b82f6; }
    .metric-warning { --accent-color: #f59e0b; }
    .metric-danger { --accent-color: #ef4444; }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Remove default streamlit styling */
    .stMetric {
        background: none !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #e2e8f0 0%, #cbd5e1 50%, #e2e8f0 100%);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers & Renderers ---
def fmt_mmss(sec):
    if sec is None or pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def fmt_hms(sec):
    if sec is None or pd.isna(sec): return "–"
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"

def render_professional_metric(container, title, value, status_class="metric-good"):
    container.markdown(f"""
        <div class="metric-card {status_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def get_utilization_status(util):
    if util >= 0.50: return "metric-excellent"
    elif util >= 0.30: return "metric-warning"
    else: return "metric-danger"

def get_email_resp_time_status(sec):
    return "metric-danger" if (sec or 0) > 59*60 else "metric-excellent"

def get_sla_score_status(score):
    if score >= 80: return "metric-excellent"
    elif score >= 70: return "metric-warning"
    else: return "metric-danger"

# Interval helpers (unchanged)
def merge_intervals(ints):
    if not ints: return []
    ints = sorted(ints, key=lambda x: x[0])
    out = [list(ints[0])]
    for s, e in ints[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(s, e) for s, e in out]

def clip_to_window(s, e, wstart, wend):
    s2, e2 = max(s, wstart), min(e, wend)
    return (s2, e2) if e2 > s2 else None

def sum_secs(ints):
    return sum((e - s).total_seconds() for s, e in ints)

def intersect_sum(h_ints, a_ints):
    h = merge_intervals(h_ints)
    a = merge_intervals(a_ints)
    i = j = 0
    tot = 0.0
    while i < len(h) and j < len(a):
        hs, he = h[i]
        as_, ae = a[j]
        s, e = max(hs, as_), min(he, ae)
        if e > s:
            tot += (e - s).total_seconds()
        if he < ae: i += 1
        else:       j += 1
    return tot

# --- Load & preprocess data (unchanged) ---
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

# --- Sidebar: Date Range (from chat.csv) ---
st.sidebar.header("📅 Filter Options")
min_date = chat_sla_df["Date/Time Opened"].dt.date.min()
max_date = chat_sla_df["Date/Time Opened"].dt.date.max()
start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=6),
                                   min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End Date",   value=max_date,
                                   min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start must be on or before End")
    st.stop()

# --- Core Metrics: Volume & AHT from report_items (unchanged calculations) ---
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

# --- SLA slices & response times (unchanged calculations) ---
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

# --- Availability & Handling calculations (unchanged) ---
window_start = datetime.combine(start_date, datetime.min.time())
window_end   = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

presence_win = df_presence[(df_presence["Start DT"] < window_end) &
                           (df_presence["End DT"]   > window_start)].copy()

chat_only_map  = {}
email_only_map = {}
shared_map     = {}

for ag, grp in presence_win.groupby("Created By: Full Name"):
    co, eo, sh = [], [], []
    for _, r in grp.iterrows():
        seg = clip_to_window(r["Start DT"], r["End DT"], window_start, window_end)
        if not seg:
            continue
        st_name = r["Service Presence Status: Developer Name"]
        if st_name == "Available_Chat":
            co.append(seg)
        elif st_name == "Available_Email_and_Web":
            eo.append(seg)
        elif st_name == "Available_All":
            sh.append(seg)
    if co: chat_only_map[ag]  = merge_intervals(co)
    if eo: email_only_map[ag] = merge_intervals(eo)
    if sh: shared_map[ag]     = merge_intervals(sh)

# Handling intervals per agent (unchanged)
chat_handles_map, email_handles_map = {}, {}
for ag, grp in chat_df.groupby("User: Full Name"):
    ints = [clip_to_window(s, e, window_start, window_end) for s, e in zip(grp["Start DT"], grp["End DT"])]
    ints = [x for x in ints if x]
    if ints: chat_handles_map[ag] = merge_intervals(ints)

for ag, grp in email_df.groupby("User: Full Name"):
    ints = [clip_to_window(s, e, window_start, window_end) for s, e in zip(grp["Start DT"], grp["End DT"])]
    ints = [x for x in ints if x]
    if ints: email_handles_map[ag] = merge_intervals(ints)

# Compute utilization (unchanged calculations)
dept_chat_handle  = 0.0
dept_email_handle = 0.0
dept_chat_avail   = 0.0
dept_email_avail  = 0.0

agents = set(chat_only_map) | set(email_only_map) | set(shared_map) | set(chat_handles_map) | set(email_handles_map)

for ag in agents:
    co = chat_only_map.get(ag, [])
    eo = email_only_map.get(ag, [])
    sh = shared_map.get(ag, [])

    chat_av_union  = merge_intervals(co + sh)
    email_av_union = merge_intervals(eo + sh)

    chat_hand  = intersect_sum(chat_handles_map.get(ag, []),  chat_av_union)  if chat_av_union  else 0.0
    email_hand = intersect_sum(email_handles_map.get(ag, []), email_av_union) if email_av_union else 0.0

    co_secs = sum_secs(co)
    eo_secs = sum_secs(eo)
    sh_secs = sum_secs(sh)
    total_hand = chat_hand + email_hand

    if sh_secs > 0:
        if total_hand > 0:
            sh_to_chat  = sh_secs * (chat_hand / total_hand)
            sh_to_email = sh_secs * (email_hand / total_hand)
        else:
            sh_to_chat = sh_to_email = sh_secs / 2.0
    else:
        sh_to_chat = sh_to_email = 0.0

    chat_av = co_secs + sh_to_chat
    email_av = eo_secs + sh_to_email

    dept_chat_handle  += chat_hand
    dept_email_handle += email_hand
    dept_chat_avail   += chat_av
    dept_email_avail  += email_av

chat_util  = (dept_chat_handle  / dept_chat_avail)  if dept_chat_avail  else 0
email_util = (dept_email_handle / dept_email_avail) if dept_email_avail else 0

# --- Build per-day SLA & volumes (unchanged calculations) ---
daily = []
for d in pd.date_range(start_date, end_date):
    dd = d.normalize()

    cd   = chat_sla_p[chat_sla_p["Date/Time Opened"].dt.date == dd.date()]
    cw   = cd[cd["Wait Time"].notna()]
    pct60= (cw["Wait Time"] <= 60).sum() / len(cw) * 100 if len(cw) else 0
    avg_w= (cw["Wait Time"].mean() / 60) if len(cw) else 0
    ar   = (cd["Abandoned After"] > 20).sum() / len(cd) * 100 if len(cd) else 0
    sla_c= max(0, min(100, ((0.5 * pct60 - 0.3 * avg_w - 0.2 * ar) / 56.25) * 100))

    ed   = email_sla_p[email_sla_p["Date/Time Opened"].dt.date == dd.date()]
    pct1 = (ed["Elapsed Time (Hours)"] <= 1).sum() / len(ed) * 100 if len(ed) else 0
    avg_e= ed["Elapsed Time (Hours)"].mean() if len(ed) else 0
    sla_e= max(0, min(100, ((0.6 * pct1 - 0.4 * avg_e) / 56.25) * 100))

    v_c  = len(chat_df[chat_df["Start DT"].dt.date  == dd.date()])
    v_e  = len(email_df[email_df["Start DT"].dt.date == dd.date()])

    daily.append({
        "Date":      dd,
        "Chat SLA":  sla_c,  "Chat Vol":  v_c,
        "Email SLA": sla_e,  "Email Vol": v_e
    })

df_daily = pd.DataFrame(daily)

df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"] * df_daily["Chat Vol"] +
    df_daily["Email SLA"] * df_daily["Email Vol"]
) / (df_daily["Chat Vol"] + df_daily["Email Vol"])
df_daily["Weighted SLA"] = df_daily["Weighted SLA"].fillna(0)

# Summary SLA Scores (unchanged calculations)
chat_weighted  = (df_daily["Chat SLA"]  * df_daily["Chat Vol"]).sum()  / df_daily["Chat Vol"].sum()  if df_daily["Chat Vol"].sum()  else 0
email_weighted = (df_daily["Email SLA"] * df_daily["Email Vol"]).sum() / df_daily["Email Vol"].sum() if df_daily["Email Vol"].sum() else 0
total_vol      = (df_daily["Chat Vol"] + df_daily["Email Vol"]).sum()
weighted_sla   = ((df_daily["Chat SLA"] * df_daily["Chat Vol"] + df_daily["Email SLA"] * df_daily["Email Vol"]).sum() / total_vol) if total_vol else 0

# --- Professional UI Layout ---
st.markdown(f'<h1 class="dashboard-title">Department Performance Dashboard</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="period-subtitle">Reporting Period: {start_date.strftime("%B %d, %Y")} – {end_date.strftime("%B %d, %Y")}</p>', unsafe_allow_html=True)

# Core Metrics Section
st.markdown('<h2 class="section-header">Volume & Performance Metrics</h2>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_professional_metric(col1, "Total Chat Interactions", f"{chat_total:,}", "metric-good")
with col2:
    render_professional_metric(col2, "Total Email Interactions", f"{email_total:,}", "metric-good")
with col3:
    render_professional_metric(col3, "Average Chat Handle Time", fmt_mmss(chat_aht), "metric-good")
with col4:
    render_professional_metric(col4, "Average Email Handle Time", fmt_mmss(email_aht), "metric-good")

# Operational Metrics Section
st.markdown('<h2 class="section-header">Operational Efficiency</h2>', unsafe_allow_html=True)
col5, col6, col7 = st.columns(3)

with col5:
    render_professional_metric(col5, "Chat Utilization Rate", f"{chat_util:.1%}", get_utilization_status(chat_util))
with col6:
    render_professional_metric(col6, "Email Utilization Rate", f"{email_util:.1%}", get_utilization_status(email_util))
with col7:
    render_professional_metric(col7, "Average Email Response Time", fmt_hms(avg_resp_secs), get_email_resp_time_status(avg_resp_secs))

# SLA Performance Section
st.markdown('<h2 class="section-header">Service Level Agreement Performance</h2>', unsafe_allow_html=True)
col8, col9, col10 = st.columns(3)

with col8:
    render_professional_metric(col8, "Chat SLA Score", f"{chat_weighted:.1f}%", get_sla_score_status(chat_weighted))
with col9:
    render_professional_metric(col9, "Email SLA Score", f"{email_weighted:.1f}%", get_sla_score_status(email_weighted))
with col10:
    render_professional_metric(col10, "Overall Weighted SLA", f"{weighted_sla:.1f}%", get_sla_score_status(weighted_sla))

# Chart Section
st.markdown('<h2 class="section-header">Performance Trends</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="chart-title">Weighted SLA Performance Over Time</h3>', unsafe_allow_html=True)
    
    tick_dates = df_daily["Date"].tolist()
    x_min = datetime.combine(start_date, datetime.min.time()) - timedelta(days=0.5)
    x_max = datetime.combine(end_date,   datetime.max.time()) + timedelta(days=0.5)

    chart = (
        alt.Chart(df_daily)
        .mark_line(point=True, color="#2563eb", strokeWidth=3)
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                axis=alt.Axis(format="%d %b", labelAngle=-45, values=tick_dates, 
                             titleFontSize=12, labelFontSize=10),
                scale=alt.Scale(domain=[x_min, x_max])
            ),
            y=alt.Y("Weighted SLA:Q", title="SLA Score (%)", 
                   scale=alt.Scale(domain=[0,105]),
                   axis=alt.Axis(titleFontSize=12, labelFontSize=10)),
            tooltip=[
                alt.Tooltip("Date:T", format="%d %b %Y", title="Date"), 
                alt.Tooltip("Weighted SLA:Q", format=".1f", title="SLA Score")
            ]
        )
    )
    
    points = chart.mark_circle(color="#2563eb", size=80)
    labels = chart.mark_text(dy=-15, color="#1f2937", fontSize=10, fontWeight="bold").encode(
        text=alt.Text("Weighted SLA:Q", format=".1f")
    )
    
    rule = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(
        color="#ef4444", strokeDash=[8,4], strokeWidth=2
    ).encode(y="y:Q")
    
    rule_label = alt.Chart(pd.DataFrame({"y":[80], "x": [x_min]})).mark_text(
        align="left", color="#ef4444", dy=-10, fontSize=11, fontWeight="bold"
    ).encode(
        y="y:Q", 
        x="x:T",
        text=alt.value("Target: 80%")
    )

    final_chart = (chart + points + labels + rule + rule_label).properties(
        width=700, 
        height=400
    ).resolve_scale(color='independent')
    
    st.altair_chart(final_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

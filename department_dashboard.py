import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Optional

# =========================
# Page Config & Styling
# =========================
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main .block-container { padding-top: 2rem; }

.metric-container {
    padding: 12px;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    border: 2px solid transparent; /* colored border, no fill */
}
.metric-title { font-size: 1.1em; color: #333333; margin-bottom: 4px; }
.metric-value { font-size: 1.8em; font-weight: bold; color: #007bff; }
</style>
""", unsafe_allow_html=True)

# =========================
# SLA calibration constants
# =========================
CHAT_RESCALE_K       = 0.4167  # Chat: (chat_raw / 0.4167) * 80 -> perfect ≈ 96
EMAIL_RESCALE_K      = 0.50    # Email: (email_raw / 0.50) * 80 -> perfect ≈ 96
SLA_SCALE            = 80.0

# Chat target for “no penalty” wait (minutes)
CHAT_TARGET_WAIT_MIN = 1.0

# =========================
# Helpers
# =========================
def fmt_mmss(sec):
    if sec is None or pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def fmt_hms(sec):
    if sec is None or pd.isna(sec): return "–"
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
    return "#F44336" if (sec or 0) > 59*60 else "#4CAF50"

def get_sla_score_color(score):
    if score >= 80: return "#4CAF50"
    elif score >= 70: return "#FFC107"
    else: return "#F44336"

# --- Survey color helpers (your thresholds) ---
def get_csat_color_pct(v):
    if v is None or pd.isna(v): return "#9E9E9E"  # grey if missing
    return "#F44336" if v < 70 else "#4CAF50"

def get_nps_color(v):
    if v is None or pd.isna(v): return "#9E9E9E"
    return "#F44336" if v < 0 else "#4CAF50"

def get_fcr_color_pct(v):
    if v is None or pd.isna(v): return "#9E9E9E"
    return "#F44336" if v < 50 else "#4CAF50"

# Interval helpers
def merge_intervals(ints):
    """Merge overlapping intervals [(s,e),...] -> disjoint, sorted."""
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
    """Sum of intersections between two interval lists."""
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

# Survey helpers (question-level -> survey-level)
def _leading_int(x) -> Optional[int]:
    if pd.isna(x): return None
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None

def _bool_yes_no(x) -> Optional[bool]:
    if pd.isna(x): return None
    s = str(x).strip().lower()
    if s in ("yes","y","true","1"): return True
    if s in ("no","n","false","0"): return False
    return None

def _nps_from_0_10(series: pd.Series) -> Optional[float]:
    r = pd.to_numeric(series, errors="coerce").dropna()
    if r.empty: return None
    promoters  = (r >= 9).mean() * 100
    detractors = (r <= 6).mean() * 100
    return promoters - detractors  # -100..100

# =========================
# Load Data
# =========================
BASE_DIR   = Path(__file__).parent
chat_path  = BASE_DIR / "chat.csv"
email_path = BASE_DIR / "email.csv"
survey_path= BASE_DIR / "survey.csv"

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

# Optional survey (question-level)
survey = None
if survey_path.exists():
    survey_q = pd.read_csv(
        survey_path,
        dayfirst=True,
        parse_dates=["Survey Taker: Created Date"],
        low_memory=False
    )
    survey_q.columns = survey_q.columns.str.strip()

    # Ensure required columns exist
    req_cols = {"Survey Taker: ID", "Survey Taker: Created Date",
                "Survey Question: Survey", "Survey Question: Question Title", "Response"}
    if req_cols.issubset(set(survey_q.columns)):
        qtitle = survey_q["Survey Question: Question Title"].str.lower()
        is_nps  = qtitle.str.contains("recommend", na=False) | qtitle.str.contains("likely", na=False)
        is_csat = qtitle.str.contains("satisfied",  na=False)
        is_fcr  = qtitle.str.contains("resolved",   na=False)

        survey_q["NPS_raw"]   = survey_q["Response"].where(is_nps).apply(_leading_int)
        survey_q["CSAT_1_5"]  = survey_q["Response"].where(is_csat).apply(_leading_int)
        survey_q["FCR_bool"]  = survey_q["Response"].where(is_fcr).apply(_bool_yes_no)
        survey_q["Channel"]   = survey_q["Survey Question: Survey"].str.extract(r"(Email|Chat)", expand=False).fillna("Other")

        agg = {
            "Survey Taker: Created Date": "min",
            "Channel":   lambda s: s.dropna().iloc[0] if s.dropna().any() else "Other",
            "NPS_raw":   lambda s: pd.to_numeric(s, errors="coerce").dropna().max() if pd.to_numeric(s, errors="coerce").notna().any() else None,
            "CSAT_1_5":  lambda s: pd.to_numeric(s, errors="coerce").dropna().max() if pd.to_numeric(s, errors="coerce").notna().any() else None,
            "FCR_bool":  lambda s: s.dropna().iloc[0] if s.dropna().any() else None,
        }
        survey = survey_q.groupby("Survey Taker: ID", as_index=False).agg(agg)
        survey = survey.rename(columns={"Survey Taker: Created Date": "Survey Date"})
        survey["CSAT%"] = ((survey["CSAT_1_5"] - 1) / 4.0 * 100.0).clip(0, 100)
        survey["Survey Date"] = pd.to_datetime(survey["Survey Date"], errors="coerce")

# =========================
# Sidebar: Date Range (from chat.csv)
# =========================
st.sidebar.header("Filter Options")
# Quick maintenance: clear cache button
if st.sidebar.button("🧹 Clear cache & rerun"):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    finally:
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

min_date = chat_sla_df["Date/Time Opened"].dt.date.min()
max_date = chat_sla_df["Date/Time Opened"].dt.date.max()
start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=6),
                                   min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End Date",   value=max_date,
                                   min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start must be on or before End")
    st.stop()

# =========================
# Core Metrics: Volume & AHT (report_items)
# =========================
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

# =========================
# SLA slices & response times (chat.csv / email.csv)
# =========================
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

# =========================
# Availability & Handling (Proportional split of Available_All)
# =========================
window_start = datetime.combine(start_date, datetime.min.time())
window_end   = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

presence_win = df_presence[(df_presence["Start DT"] < window_end) &
                           (df_presence["End DT"]   > window_start)].copy()

chat_only_map  = {}  # Available_Chat
email_only_map = {}  # Available_Email_and_Web
shared_map     = {}  # Available_All

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

# Handling intervals per agent (clipped to window)
chat_handles_map, email_handles_map = {}, {}
for ag, grp in chat_df.groupby("User: Full Name"):
    ints = [clip_to_window(s, e, window_start, window_end) for s, e in zip(grp["Start DT"], grp["End DT"])]
    ints = [x for x in ints if x]
    if ints: chat_handles_map[ag] = merge_intervals(ints)

for ag, grp in email_df.groupby("User: Full Name"):
    ints = [clip_to_window(s, e, window_start, window_end) for s, e in zip(grp["Start DT"], grp["End DT"])]
    ints = [x for x in ints if x]
    if ints: email_handles_map[ag] = merge_intervals(ints)

# Numerators and proportional denominators
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

    chat_av   = co_secs + sh_to_chat
    email_av  = eo_secs + sh_to_email

    dept_chat_handle  += chat_hand
    dept_email_handle += email_hand
    dept_chat_avail   += chat_av
    dept_email_avail  += email_av

chat_util  = (dept_chat_handle  / dept_chat_avail)  if dept_chat_avail  else 0
email_util = (dept_email_handle / dept_email_avail) if dept_email_avail else 0

# =========================
# Build per-day SLA & volumes (one row per day)
# =========================
daily = []
for d in pd.date_range(start_date, end_date):
    dd = d.normalize()

    # --- Chat SLA (revised: only penalize wait ABOVE target; align numerator/denominator) ---
    cd = chat_sla_p[chat_sla_p["Date/Time Opened"].dt.date == dd.date()]
    cw = cd[cd["Wait Time"].notna()]  # answered chats with wait times

    # % answered ≤60s = answered ≤60 / TOTAL chats (fraction 0–1)
    frac_answer_60s = ((cw["Wait Time"] <= 60).sum() / len(cd)) if len(cd) else 0.0

    # Average wait (minutes) across answered chats only
    avg_wait_min    = (cw["Wait Time"].mean() / 60.0) if len(cw) else 0.0

    # Abandon rate includes only those abandoned AFTER 20s (fraction of total)
    abandon_frac    = ((cd["Abandoned After"] > 20).sum() / len(cd)) if len(cd) else 0.0

    # Only penalize excess wait above target
    excess_wait_min = max(avg_wait_min - CHAT_TARGET_WAIT_MIN, 0.0)

    chat_raw = 0.5 * frac_answer_60s - 0.3 * excess_wait_min - 0.2 * abandon_frac
    sla_c    = max(0.0, min(100.0, (chat_raw / CHAT_RESCALE_K) * SLA_SCALE))

    # --- Email SLA (revised: only penalize avg above 1 hour) ---
    ed = email_sla_p[email_sla_p["Date/Time Opened"].dt.date == dd.date()]
    frac_le_1hr = ((ed["Elapsed Time (Hours)"] <= 1).sum() / len(ed)) if len(ed) else 0.0
    avg_resp_hr = (ed["Elapsed Time (Hours)"].mean()) if len(ed) else 0.0

    excess = max(avg_resp_hr - 1.0, 0.0)
    email_raw = 0.6 * frac_le_1hr - 0.4 * excess
    sla_e     = max(0.0, min(100.0, (email_raw / EMAIL_RESCALE_K) * SLA_SCALE))

    # Volumes from report_items (for weighting)
    v_c = len(chat_df[chat_df["Start DT"].dt.date  == dd.date()])
    v_e = len(email_df[email_df["Start DT"].dt.date == dd.date()])

    daily.append({
        "Date":      dd,
        "Chat SLA":  sla_c,  "Chat Vol":  v_c,
        "Email SLA": sla_e,  "Email Vol": v_e
    })

df_daily = pd.DataFrame(daily)

# Weighted SLA per day & fill NaNs so every date renders
df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"] * df_daily["Chat Vol"] +
    df_daily["Email SLA"] * df_daily["Email Vol"]
) / (df_daily["Chat Vol"] + df_daily["Email Vol"])
df_daily["Weighted SLA"] = df_daily["Weighted SLA"].fillna(0)

# Summary SLA Scores (volume-weighted across selected days)
chat_weighted  = (df_daily["Chat SLA"]  * df_daily["Chat Vol"]).sum()  / df_daily["Chat Vol"].sum()  if df_daily["Chat Vol"].sum()  else 0
email_weighted = (df_daily["Email SLA"] * df_daily["Email Vol"]).sum() / df_daily["Email Vol"].sum() if df_daily["Email Vol"].sum() else 0
total_vol      = (df_daily["Chat Vol"] + df_daily["Email Vol"]).sum()
weighted_sla   = ((df_daily["Chat SLA"] * df_daily["Chat Vol"] + df_daily["Email SLA"] * df_daily["Email Vol"]).sum() / total_vol) if total_vol else 0

# =========================
# Header & KPI Tiles
# =========================
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.markdown("---")

# Core Metrics
st.subheader("Core Metrics")
c1, c2, c3, c4 = st.columns(4)
render_custom_metric(c1, "💬 Total Chats",            chat_total,           "Total chat interactions",          "#4CAF50")
render_custom_metric(c2, "✉️ Total Emails",           email_total,          "Total email interactions",         "#4CAF50")
render_custom_metric(c3, "⏳ Avg Chat Handle Time",    fmt_mmss(chat_aht),   "Average chat handle time",         "#4CAF50")
render_custom_metric(c4, "⏳ Avg Email Handle Time",   fmt_mmss(email_aht),  "Average email handle time",        "#4CAF50")

# Operational Metrics
st.markdown("---")
st.subheader("Operational Metrics")
m1, m2, m3 = st.columns(3)
render_custom_metric(m1, "📈 Chat Utilization",     f"{chat_util:.1%}",     "Handled∩Available / proportional availability", get_utilization_color(chat_util))
render_custom_metric(m2, "📈 Email Utilization",    f"{email_util:.1%}",    "Handled∩Available / proportional availability", get_utilization_color(email_util))
render_custom_metric(m3, "⏱️ Avg Email Resp Time",  fmt_hms(avg_resp_secs), "Average email response time",           get_email_resp_time_color(avg_resp_secs))

# SLA Score Summary
st.markdown("---")
st.subheader("🎯 SLA Score Summary")
s1, s2, s3 = st.columns(3)
render_custom_metric(s1, "Chat SLA Score",     f"{chat_weighted:.1f}",  "Daily-volume weighted chat SLA",  get_sla_score_color(chat_weighted))
render_custom_metric(s2, "Email SLA Score",    f"{email_weighted:.1f}", "Daily-volume weighted email SLA", get_sla_score_color(email_weighted))
render_custom_metric(s3, "Weighted SLA Score", f"{weighted_sla:.1f}",   "Volume-weighted blended SLA",     get_sla_score_color(weighted_sla))

# =========================
# Weighted SLA Trend Chart
# =========================
st.markdown("---")
st.subheader("Weighted SLA Trend")

tick_dates = df_daily["Date"].tolist()
x_min = datetime.combine(start_date, datetime.min.time()) - timedelta(days=0.5)
x_max = datetime.combine(end_date,   datetime.max.time()) + timedelta(days=0.5)

chart = (
    alt.Chart(df_daily)
    .mark_line(point=True, color="#2F80ED")
    .encode(
        x=alt.X(
            "Date:T",
            title="Date",
            axis=alt.Axis(format="%d %b", labelAngle=-45, values=tick_dates),
            scale=alt.Scale(domain=[x_min, x_max])
        ),
        y=alt.Y("Weighted SLA:Q", title="Weighted SLA Score", scale=alt.Scale(domain=[0,105])),
        tooltip=[alt.Tooltip("Date:T", format="%d %b"), alt.Tooltip("Weighted SLA:Q", format=".1f")]
    )
)
labels = chart.mark_text(dy=-10, color="#2F80ED").encode(text=alt.Text("Weighted SLA:Q", format=".1f"))
rule   = alt.Chart(pd.DataFrame({"y":[85]})).mark_rule(color="red", strokeDash=[5,5]).encode(y="y:Q")
rule_lb= alt.Chart(pd.DataFrame({"y":[85]})).mark_text(align="left", color="red", dy=-8)\
            .encode(y="y:Q", text=alt.value("Target: 85%"))

st.altair_chart((chart + labels + rule + rule_lb).properties(width=700, height=350),
                use_container_width=True)

# =========================
# Customer Feedback Section
# =========================
if survey is not None:
    survey_period = survey[
        (survey["Survey Date"].dt.date >= start_date) &
        (survey["Survey Date"].dt.date <= end_date)
    ].copy()

    if not survey_period.empty:
        st.markdown("---")
        st.subheader("Customer Feedback")

        # --- Overall KPIs ---
        total_surveys = len(survey_period)
        csat_overall  = survey_period["CSAT%"].mean() if survey_period["CSAT%"].notna().any() else None
        nps_overall   = _nps_from_0_10(survey_period["NPS_raw"]) if survey_period["NPS_raw"].notna().any() else None
        fcr_overall   = (survey_period["FCR_bool"] == True).mean() * 100 if survey_period["FCR_bool"].notna().any() else None

        k1, k2, k3, k4 = st.columns(4)
        render_custom_metric(k1, "🗳️ Surveys", f"{total_surveys:,}", "Total surveys in range", "#4CAF50")
        render_custom_metric(
            k2, "😊 CSAT (avg %)",
            f"{csat_overall:.1f}%" if csat_overall is not None else "–",
            "Average CSAT normalized to 0–100%",
            get_csat_color_pct(csat_overall)
        )
        render_custom_metric(
            k3, "⭐ NPS",
            f"{nps_overall:.1f}" if nps_overall is not None else "–",
            "NPS: %Promoters − %Detractors",
            get_nps_color(nps_overall)
        )
        render_custom_metric(
            k4, "🎯 FCR",
            f"{fcr_overall:.1f}%" if fcr_overall is not None else "–",
            "First Contact Resolution rate",
            get_fcr_color_pct(fcr_overall)
        )

        # --- Per-channel tiles (CSAT color threshold applied) ---
        survey_period["ChanSimple"] = survey_period["Channel"].map(
            lambda x: "Email" if "Email" in str(x) else "Chat" if "Chat" in str(x) else "Other"
        )
        by_chan = (
            survey_period
            .groupby("ChanSimple")
            .agg(
                Surveys=("CSAT%","size"),
                CSAT_pct=("CSAT%","mean"),
                NPS=("NPS_raw", _nps_from_0_10),
                FCR_pct=("FCR_bool", lambda s: (s==True).mean()*100 if s.notna().any() else None)
            )
            .reset_index()
        )
        if not by_chan.empty:
            cols = st.columns(len(by_chan))
            for i, row in enumerate(by_chan.itertuples(index=False)):
                chan = row.ChanSimple
                tip  = f"{int(row.Surveys)} surveys"
                if row.NPS is not None:
                    tip += f" • NPS {row.NPS:.1f}"
                if row.FCR_pct is not None:
                    tip += f" • FCR {row.FCR_pct:.1f}%"
                render_custom_metric(
                    cols[i],
                    f"{chan}: CSAT%",
                    f"{(row.CSAT_pct or 0):.1f}%",
                    tip,
                    get_csat_color_pct(row.CSAT_pct)
                )

        # --- CSAT & NPS Trend (dual-axis lines) ---
        st.subheader("CSAT & NPS Trend Analysis")
        monthly_kpis = (
            survey_period
            .assign(month=survey_period["Survey Date"].dt.to_period('M'))
            .groupby('month', as_index=False)
            .agg(
                csat=('CSAT%', 'mean'),
                nps=('NPS_raw', _nps_from_0_10)
            )
        )
        if not monthly_kpis.empty:
            monthly_kpis['month'] = monthly_kpis['month'].dt.to_timestamp()

            # Create a base chart for layering
            base = alt.Chart(monthly_kpis).encode(
                x=alt.X('month', axis=alt.Axis(format="%b", title="Month")),
            )

            # CSAT bars on the left Y-axis
            csat_bars = base.mark_bar(opacity=0.7, color='#28a745').encode(
                y=alt.Y('csat', axis=alt.Axis(title='CSAT')),
                tooltip=[alt.Tooltip('month', format="%b %Y"), 'csat']
            )

            # NPS line on the right Y-axis
            nps_line = base.mark_line(color='#ffc107', point=True).encode(
                y=alt.Y('nps', axis=alt.Axis(title='NPS')),
                tooltip=[alt.Tooltip('month', format="%b %Y"), 'nps']
            )

            # Layer the charts and resolve the scales to be independent
            csat_nps_chart = alt.layer(csat_bars, nps_line).resolve_scale(
                y='independent'
            ).properties(
                title='CSAT vs NPS Trend'
            )

            st.altair_chart(csat_nps_chart, use_container_width=True)

    else:
        st.info("No survey responses in the selected date range.")
else:
    st.info("Survey data not found (survey.csv). Add it next to the app to see CSAT/NPS/FCR.")

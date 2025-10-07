import streamlit as st

def merge_intervals(intervals):
    """Merge overlapping/adjacent [start, end] datetime intervals.
    Accepts tuples or lists; returns a list of tuples (start, end).
    """
    if not intervals:
        return []
    # Ensure all intervals are lists (mutable) and valid (end > start)
    norm = []
    for s, e in intervals:
        if s is None or e is None:
            continue
        # coerce to pandas.Timestamp if possible (no import cost if already)
        try:
            import pandas as pd
            s = pd.to_datetime(s)
            e = pd.to_datetime(e)
        except Exception:
            pass
        if e > s:
            norm.append([s, e])
    if not norm:
        return []
    norm.sort(key=lambda x: x[0])
    merged = [norm[0]]
    for s, e in norm[1:]:
        if s <= merged[-1][1]:
            # extend the right edge
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(iv) for iv in merged]


import pandas as pd
import altair as alt
from datetime import datetime, timedelta, date
from pathlib import Path
import re
import unicodedata
from typing import Optional
import datetime as _dt

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
CHAT_RESCALE_K       = 0.4167  # Chat: (raw / 0.4167) * 80
EMAIL_RESCALE_K      = 0.50    # Email: (raw / 0.50) * 80
SLA_SCALE            = 80.0
CHAT_TARGET_WAIT_MIN = 1.0     # “no penalty” wait threshold in minutes

# =========================
# Helpers
# =========================
def fmt_mmss(sec):
    if sec is None or pd.isna(sec): return "–"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"

def fmt_hhmm(sec):
    if sec is None or pd.isna(sec): return "–"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h:02}:{m:02}"

def fmt_hms(sec):
    if sec is None or pd.isna(sec): return "–"
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"

def fmt_hhmm_dt(dt):
    return "—" if dt is None or (isinstance(dt, float) and pd.isna(dt)) else dt.strftime("%H:%M")

def fmt_minutes_clean(x):
    if x is None or pd.isna(x): return "—"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}".rstrip('0').rstrip('.')

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

# Survey color helpers
def get_csat_color_pct(v):
    if v is None or pd.isna(v): return "#9E9E9E"
    return "#F44336" if v < 70 else "#4CAF50"

def get_nps_color(v):
    if v is None or pd.isna(v): return "#9E9E9E"
    return "#F44336" if v < 0 else "#4CAF50"

def get_fcr_color_pct(v):
    if v is None or pd.isna(v): return "#9E9E9E"
    return "#F44336" if v < 50 else "#4CAF50"

# Interval helpers
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
# Name/Status normalizers
# =========================
def _norm_person_key(s: str) -> str:
    """Lowercase, strip accents, keep alnum tokens, sort tokens so 'A B' == 'B A'."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return " ".join(sorted(toks))

def _norm_status_key(s: str) -> str:
    """Lowercase, strip accents, collapse non-letters to underscores."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z]+", "_", s.lower()).strip("_")
    return s

# =========================
# Wide -> Tidy shifts.csv normalizer (matrix to tidy)
# =========================
def normalize_wide_shifts(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide shifts matrix into a tidy dataframe with:
        Agent | Date | Shift Start | Shift End
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Agent", "Date", "Shift Start", "Shift End"])

    cols = list(df_raw.columns)

    # Identify date columns
    date_cols = []
    date_map = {}
    for c in cols:
        d = pd.to_datetime(c, errors="coerce", dayfirst=True)
        if isinstance(d, pd.Timestamp) and not pd.isna(d):
            date_cols.append(c)
            date_map[c] = d.date()

    if not date_cols:
        return pd.DataFrame(columns=["Agent", "Date", "Shift Start", "Shift End"])

    # Assume first non-date column is the agent name
    name_col = next((c for c in cols if c not in date_cols), cols[0])

    long = (
        df_raw.melt(id_vars=[name_col], value_vars=date_cols,
                    var_name="DateStr", value_name="Range")
             .rename(columns={name_col: "Agent"})
    )
    long["Date"] = long["DateStr"].map(date_map)
    long["Range"] = long["Range"].astype(str).str.strip()

    off_mask = long["Range"].str.fullmatch(r"(?i)\s*(off|off day|offday|na|nan|-|—|–)?\s*")
    long = long[~off_mask.fillna(True)]

    rgx = re.compile(r"^\s*(\d{1,2}:\d{2}\s*[AaPp][Mm])\s*[-–—]\s*(\d{1,2}:\d{2}\s*[AaPp][Mm])\s*$")

    def _parse_range(s, d):
        m = rgx.match(s or "")
        if not m:
            return (pd.NaT, pd.NaT)
        t1, t2 = m.group(1), m.group(2)
        start_dt = pd.to_datetime(f"{d} {t1}", format="%Y-%m-%d %I:%M %p", errors="coerce")
        end_dt   = pd.to_datetime(f"{d} {t2}", format="%Y-%m-%d %I:%M %p", errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt) and end_dt <= start_dt:
            end_dt = end_dt + pd.Timedelta(days=1)
        return (start_dt, end_dt)

    parsed = long.apply(lambda r: pd.Series(_parse_range(r["Range"], r["Date"])), axis=1)
    long["Shift Start"] = pd.to_datetime(parsed[0], errors="coerce")
    long["Shift End"]   = pd.to_datetime(parsed[1], errors="coerce")

    tidy = long.dropna(subset=["Shift Start", "Shift End"])[["Agent", "Date", "Shift Start", "Shift End"]]
    return tidy.reset_index(drop=True)

# =========================
# File paths (case-insensitive) + cache-busting
# =========================
BASE_DIR = Path(__file__).parent

def resolve_path_case_insensitive(filename: str) -> Path:
    """Return a Path to a file in BASE_DIR matching filename, case-insensitively."""
    p = BASE_DIR / filename
    if p.exists():
        return p
    low = filename.lower()
    for child in BASE_DIR.iterdir():
        if child.is_file() and child.name.lower() == low:
            return child
    return p  # may not exist; caller should check

def file_signature(p: Path):
    """Tuple that changes when the file changes (mtime, size)."""
    stat = p.stat()
    return (int(stat.st_mtime), int(stat.st_size))

@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, sig: tuple, **read_kwargs):
    # sig is unused inside; it's only to bust the cache when the file changes
    return pd.read_csv(path_str, **read_kwargs)

# Resolve data files
chat_path   = resolve_path_case_insensitive("chat.csv")
email_path  = resolve_path_case_insensitive("email.csv")
survey_path = resolve_path_case_insensitive("survey.csv")
items_path  = resolve_path_case_insensitive("report_items.csv")
pres_path   = resolve_path_case_insensitive("report_presence.csv")
shifts_path = resolve_path_case_insensitive("shifts.csv")

missing = [p.name for p in (chat_path, email_path, items_path, pres_path, shifts_path) if not p.exists()]
if missing:
    st.error(f"Missing required files in the app folder: {', '.join(missing)}")
    st.stop()

# Manual refresh control
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload data now"):
    st.cache_data.clear()
    st.rerun()

# =========================
# Load Data (cached, cache-busted)
# =========================
df_items     = load_csv_cached(str(items_path),  file_signature(items_path),  dayfirst=True, parse_dates=["Start DT","End DT"])
df_presence  = load_csv_cached(str(pres_path),   file_signature(pres_path),   dayfirst=True, parse_dates=["Start DT","End DT"])
df_shifts_raw= load_csv_cached(str(shifts_path), file_signature(shifts_path))
chat_sla_df  = load_csv_cached(str(chat_path),   file_signature(chat_path),   dayfirst=True, parse_dates=["Date/Time Opened"])
email_sla_df = load_csv_cached(str(email_path),  file_signature(email_path),  dayfirst=True, parse_dates=["Date/Time Opened","Completion Date"])

# Optional survey (question-level)
survey = None
if survey_path.exists():
    survey_q = load_csv_cached(str(survey_path), file_signature(survey_path),
                               dayfirst=True, parse_dates=["Survey Taker: Created Date"], low_memory=False)
    survey_q.columns = survey_q.columns.str.strip()

# Normalize columns
for df in (df_items, df_presence, df_shifts_raw, chat_sla_df, email_sla_df):
    df.columns = df.columns.str.strip()

# Re-coerce critical datetime columns (defensive)
for df_ in (df_items, df_presence):
    for col in ("Start DT", "End DT"):
        df_[col] = pd.to_datetime(df_[col], errors="coerce", dayfirst=True, utc=False)

# Normalize shifts
df_shifts = normalize_wide_shifts(df_shifts_raw)

# =========================
# Sidebar: Date Range (from chat.csv)
# =========================
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

# Show which files were loaded (helpful on Cloud)
def fmt_sig(p: Path):
    mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime)
    return f"{p.name} • {p.stat().st_size:,} bytes • {mtime:%Y-%m-%d %H:%M:%S}"

with st.expander("ℹ️ Data files loaded"):
    lines = [
        f"• {fmt_sig(chat_path)}",
        f"• {fmt_sig(email_path)}",
        f"• {fmt_sig(items_path)}",
        f"• {fmt_sig(pres_path)}",
        f"• {fmt_sig(shifts_path)}",
    ]
    if survey_path.exists():
        lines.append(f"• {fmt_sig(survey_path)}")
    st.write("\n".join(lines))

# =========================
# Core Metrics: Volume & AHT (report_items) — robust time filtering
# =========================
ts_start = pd.Timestamp(start_date)  # inclusive
ts_end   = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # exclusive

mask = (df_items["Start DT"] >= ts_start) & (df_items["Start DT"] < ts_end)
df_period = df_items.loc[mask].copy()

# Safe duration calculation
df_period = df_period.dropna(subset=["Start DT", "End DT"])
start_dt = pd.to_datetime(df_period["Start DT"], errors="coerce")
end_dt   = pd.to_datetime(df_period["End DT"],   errors="coerce")
dur_td = end_dt - start_dt
df_period["Duration_sec"] = dur_td.dt.total_seconds()

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
# Availability & Handling (Proportional split for overall utilization tiles)
# =========================
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

# Numerators and proportional denominators for utilization tiles
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

    # --- Chat SLA ---
    cd = chat_sla_p[chat_sla_p["Date/Time Opened"].dt.date == dd.date()]
    cw = cd[cd["Wait Time"].notna()]  # answered chats with wait times

    frac_answer_60s = ((cw["Wait Time"] <= 60).sum() / len(cd)) if len(cd) else 0.0  # fraction 0–1
    avg_wait_min    = (cw["Wait Time"].mean() / 60.0) if len(cw) else 0.0
    abandon_frac    = ((cd["Abandoned After"] > 20).sum() / len(cd)) if len(cd) else 0.0

    excess_wait_min = max(avg_wait_min - CHAT_TARGET_WAIT_MIN, 0.0)
    chat_raw = 0.5 * frac_answer_60s - 0.3 * excess_wait_min - 0.2 * abandon_frac
    sla_c    = max(0.0, min(100.0, (chat_raw / CHAT_RESCALE_K) * SLA_SCALE))

    # --- Email SLA ---
    ed = email_sla_p[email_sla_p["Date/Time Opened"].dt.date == dd.date()]
    frac_le_1hr = ((ed["Elapsed Time (Hours)"] <= 1).sum() / len(ed)) if len(ed) else 0.0
    avg_resp_hr = (ed["Elapsed Time (Hours)"].mean()) if len(ed) else 0.0
    excess = max(avg_resp_hr - 1.0, 0.0)
    email_raw = 0.6 * frac_le_1hr - 0.4 * excess
    sla_e     = max(0.0, min(100.0, (email_raw / EMAIL_RESCALE_K) * SLA_SCALE))

    v_c = len(chat_df[chat_df["Start DT"].dt.date  == dd.date()])
    v_e = len(email_df[email_df["Start DT"].dt.date == dd.date()])

    daily.append({
        "Date":      dd,
        "Chat SLA":  sla_c,  "Chat Vol":  v_c,
        "Email SLA": sla_e,  "Email Vol": v_e
    })

df_daily = pd.DataFrame(daily)

# Weighted SLA per day
df_daily["Weighted SLA"] = (
    df_daily["Chat SLA"] * df_daily["Chat Vol"] +
    df_daily["Email SLA"] * df_daily["Email Vol"]
) / (df_daily["Chat Vol"] + df_daily["Email Vol"])
df_daily["Weighted SLA"] = df_daily["Weighted SLA"].fillna(0)

# Summary SLA Scores
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

trend_chart = (
    alt.Chart(df_daily)
    .mark_line(point=True, color="#2F80ED")
    .encode(
        x=alt.X("Date:T", title="Date",
                axis=alt.Axis(format="%d %b", labelAngle=-45, values=tick_dates),
                scale=alt.Scale(domain=[x_min, x_max])),
        y=alt.Y("Weighted SLA:Q", title="Weighted SLA Score", scale=alt.Scale(domain=[0,105])),
        tooltip=[alt.Tooltip("Date:T", format="%d %b"), alt.Tooltip("Weighted SLA:Q", format=".1f")]
    )
)
labels = trend_chart.mark_text(dy=-10, color="#2F80ED").encode(text=alt.Text("Weighted SLA:Q", format=".1f"))
rule   = alt.Chart(pd.DataFrame({"y":[85]})).mark_rule(color="red", strokeDash=[5,5]).encode(y="y:Q")
rule_lb= alt.Chart(pd.DataFrame({"y":[85]})).mark_text(align="left", color="red", dy=-8)\
            .encode(y="y:Q", text=alt.value("Target: 85%"))
st.altair_chart((trend_chart + labels + rule + rule_lb).properties(width=700, height=350),
                use_container_width=True)

# =========================
# Customer Feedback Section (CSAT / NPS / FCR)
# =========================
if survey_path.exists():
    # Prepare survey records if they exist
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

    survey_period = survey[
        (survey["Survey Date"].dt.date >= start_date) &
        (survey["Survey Date"].dt.date <= end_date)
    ].copy()

    if not survey_period.empty:
        st.markdown("---")
        st.subheader("Customer Feedback")

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

        # ---------- CSAT & NPS Trend: day-level x-axis ----------
        daily_survey = (
            survey_period
            .assign(Date=survey_period["Survey Date"].dt.normalize())
            .groupby("Date", as_index=False)
            .agg(
                CSAT_pct=("CSAT%","mean"),
                NPS=("NPS_raw", _nps_from_0_10),
                Surveys=("CSAT%","size")
            )
            .sort_values("Date")
            .drop_duplicates(subset=["Date"])
        )

        if not daily_survey.empty:
            st.subheader("CSAT & NPS Trend Analysis")

            base = alt.Chart(daily_survey)
            x_enc = alt.X(
                "yearmonthdate(Date):T",
                title="Date",
                axis=alt.Axis(format="%d %b", labelAngle=45),
                scale=alt.Scale(nice="day")
            )
            hover = alt.selection_point(on="mouseover", fields=["Date"], nearest=True, empty=False)

            csat_line = base.mark_line(strokeWidth=3, color="#2563eb").encode(
                x=x_enc,
                y=alt.Y("CSAT_pct:Q", title="CSAT (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("Date:T", title="Date", format="%d %b"),
                         alt.Tooltip("CSAT_pct:Q", title="CSAT (%)", format=".1f"),
                         alt.Tooltip("Surveys:Q", title="Surveys", format=".0f")]
            )
            csat_points = base.mark_point(filled=True, color="#2563eb").encode(
                x=x_enc,
                y=alt.Y("CSAT_pct:Q", axis=None, scale=alt.Scale(domain=[0,100])),
                size=alt.condition(hover, alt.value(120), alt.value(60)),
                tooltip=[alt.Tooltip("Date:T", title="Date", format="%d %b"),
                         alt.Tooltip("CSAT_pct:Q", title="CSAT (%)", format=".1f")]
            ).add_params(hover)

            nps_line = base.mark_line(strokeWidth=3, color="#dc2626", strokeDash=[5, 5]).encode(
                x=x_enc,
                y=alt.Y("NPS:Q", title="NPS Score",
                        axis=alt.Axis(orient="right"),
                        scale=alt.Scale(domain=[-100, 100])),
                tooltip=[alt.Tooltip("Date:T", title="Date", format="%d %b"),
                         alt.Tooltip("NPS:Q", title="NPS Score", format=".0f"),
                         alt.Tooltip("Surveys:Q", title="Surveys", format=".0f")]
            )
            nps_points = base.mark_point(filled=True, color="#dc2626", shape="square").encode(
                x=x_enc,
                y=alt.Y("NPS:Q", axis=None, scale=alt.Scale(domain=[-100,100])),
                size=alt.condition(hover, alt.value(120), alt.value(60)),
                tooltip=[alt.Tooltip("Date:T", title="Date", format="%d %b"),
                         alt.Tooltip("NPS:Q", title="NPS Score", format=".0f")]
            ).add_params(hover)
            nps_zero_rule = alt.Chart(pd.DataFrame({"zero":[0]})).mark_rule(
                strokeDash=[6,4], color="#9ca3af", opacity=0.6
            ).encode(y=alt.Y("zero:Q", axis=None, scale=alt.Scale(domain=[-100,100])))

            trend = alt.layer(csat_line, nps_line, csat_points, nps_points, nps_zero_rule)\
                .resolve_scale(y="independent")\
                .properties(width=800, height=400, title=alt.TitleParams(
                    text="CSAT & NPS Trend Analysis", font="Arial", fontSize=18, anchor="start", color="#111827"
                ))\
                .configure_axis(grid=True, gridColor="#e5e7eb", gridDash=[2,3],
                                labelFont="Arial", titleFont="Arial",
                                labelColor="#374151", titleColor="#111827")\
                .configure_view(stroke="#d1d5db", fill="white")\
                .configure_legend(orient="top-right", titleFont="Arial", labelFont="Arial")\
                .interactive()
            st.altair_chart(trend, use_container_width=True)
    else:
        st.info("No survey responses in the selected date range.")
else:
    st.info("Survey data not found (survey.csv). Add it next to the app to see CSAT/NPS/FCR.")

# =========================
# 🌍 Chats by Country (volume)
# =========================
st.markdown("---")
st.subheader("🌍 Chats by Country (volume)")

def _detect_country_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns if "country" in c.lower()]
    if not cols:
        cols = [c for c in df.columns if "geo" in c.lower()]
    return cols[0] if cols else None

country_col = _detect_country_col(chat_sla_df)
if not country_col:
    st.info("No country column found in chat.csv (e.g., 'Country', 'Visitor Country').")
else:
    chat_country = chat_sla_df[
        (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
        (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
    ].copy()

    chat_country[country_col] = (
        chat_country[country_col]
        .astype(str).str.strip()
        .replace({"": None, "nan": None, "none": None, "None": None})
        .fillna("Unknown").str.title()
    )

    counts = (
        chat_country.groupby(country_col, dropna=False)
        .size()
        .reset_index(name="Chats")
        .sort_values("Chats", ascending=False)
        .reset_index(drop=True)
    )

    top_n = st.sidebar.slider("Pie chart: top countries", min_value=3, max_value=12, value=8, step=1)
    if len(counts) > top_n:
        top = counts.head(top_n)
        others_total = counts["Chats"].iloc[top_n:].sum()
        counts = pd.concat(
            [top, pd.DataFrame({country_col: ["Other"], "Chats": [others_total]})],
            ignore_index=True
        )

    total_chats = int(counts["Chats"].sum()) if len(counts) else 0
    counts["Share"] = counts["Chats"] / total_chats if total_chats else 0.0

    pie = (
        alt.Chart(counts)
        .mark_arc(outerRadius=140, innerRadius=60)
        .encode(
            theta=alt.Theta("Chats:Q", stack=True),
            color=alt.Color(f"{country_col}:N", legend=alt.Legend(title="Country")),
            tooltip=[
                alt.Tooltip(f"{country_col}:N", title="Country"),
                alt.Tooltip("Chats:Q", title="Chats", format=","),
                alt.Tooltip("Share:Q", title="Share", format=".1%")
            ],
        )
        .properties(width=480, height=360, title="Chat Volume by Country")
    )
    labels = (
        alt.Chart(counts)
        .mark_text(radius=105, size=11)
        .encode(theta=alt.Theta("Chats:Q", stack=True),
                text=alt.Text("Chats:Q", format=","))
    )
    st.altair_chart(pie + labels, use_container_width=True)

    with st.expander("View country breakdown table"):
        st.dataframe(
            counts.rename(columns={country_col: "Country"})[["Country", "Chats", "Share"]]
                  .style.format({"Chats": "{:,}", "Share": "{:.1%}"}),
            use_container_width=True
        )

# =========================
# ⏱️ Hourly Weighted SLA (selected day) + Available Minutes + Logged-in Agents
# =========================
st.markdown("---")
st.subheader("⏱️ Hourly Weighted SLA (selected day)")

hourly_date = st.date_input(
    "Select a day for hourly view",
    value=end_date,
    min_value=min_date,
    max_value=max_date
)

def _clamp01(x):
    if pd.isna(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))

def _merge_intervals(ints):
    if not ints: return []
    ints = sorted(ints, key=lambda x: x[0])
    out = [list(ints[0])]
    for s, e in ints[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(s, e) for s, e in out]

def _clip(seg_s, seg_e, w_s, w_e):
    s, e = max(seg_s, w_s), min(seg_e, w_e)
    return (s, e) if e > s else None

def _sum_secs(ints):
    return sum((e - s).total_seconds() for s, e in ints)

def compute_hourly_available_minutes_and_logged_in(sel_date: datetime.date) -> pd.DataFrame:
    day_start = datetime.combine(sel_date, datetime.min.time())
    day_end   = day_start + timedelta(days=1)

    pres_day = df_presence[(df_presence["Start DT"] < day_end) &
                           (df_presence["End DT"]   > day_start)].copy()

    hours = pd.date_range(day_start, day_end, freq="h", inclusive="left").to_pydatetime().tolist()
    avail_secs_per_hour  = {h: 0.0 for h in hours}
    logged_sets_per_hour = {h: set() for h in hours}

    AVAILABLE_STATUSES = {"Available_Chat", "Available_Email_and_Web", "Available_All"}

    if not pres_day.empty:
        for _, r in pres_day.iterrows():
            seg = _clip(r["Start DT"].to_pydatetime(), r["End DT"].to_pydatetime(), day_start, day_end)
            if not seg:
                continue
            stt, end = seg
            agent  = str(r["Created By: Full Name"])
            status = str(r["Service Presence Status: Developer Name"])

            h = stt.replace(minute=0, second=0, microsecond=0)
            while h < end:
                hour_start = h
                hour_end   = h + timedelta(hours=1)
                o = _clip(stt, end, hour_start, hour_end)
                if o:
                    o_s, o_e = o
                    dur = (o_e - o_s).total_seconds()
                    logged_sets_per_hour[hour_start].add(agent)
                    if status in AVAILABLE_STATUSES:
                        avail_secs_per_hour[hour_start] += dur
                h = hour_end

    return pd.DataFrame({
        "Hour": hours,
        "Avail (min)":  [avail_secs_per_hour[h] / 60.0 for h in hours],
        "Logged In Agents": [len(logged_sets_per_hour[h]) for h in hours],
    })

def compute_hourly_sla_for_date(sel_date: datetime.date) -> pd.DataFrame:
    day_start = datetime.combine(sel_date, datetime.min.time())
    day_end   = day_start + timedelta(days=1)

    chat_day  = chat_sla_df[(chat_sla_df["Date/Time Opened"] >= day_start) &
                            (chat_sla_df["Date/Time Opened"] <  day_end)].copy()
    email_day = email_sla_df[(email_sla_df["Date/Time Opened"] >= day_start) &
                             (email_sla_df["Date/Time Opened"] <  day_end)].copy()
    chat_day["Hour"]  = chat_day["Date/Time Opened"].dt.floor("h")
    email_day["Hour"] = email_day["Date/Time Opened"].dt.floor("h")

    items_day = df_items[(df_items["Start DT"] >= day_start) &
                         (df_items["Start DT"] <  day_end)].copy()
    items_day["Hour"] = items_day["Start DT"].dt.floor("h")

    chat_vol_hour = (
        items_day[items_day["Service Channel: Developer Name"] == "sfdc_liveagent"]
        .groupby("Hour").size().rename("Chat Vol")
    )
    email_vol_hour = (
        items_day[items_day["Service Channel: Developer Name"] == "casesChannel"]
        .groupby("Hour").size().rename("Email Vol")
    )

    hours = pd.date_range(day_start, day_end, freq="h", inclusive="left")
    out = pd.DataFrame({"Hour": hours})

    def _chat_sla_for_group(g: pd.DataFrame) -> float:
        if g.empty: return None
        total = len(g)
        answered = g[g["Wait Time"].notna()]
        frac_60 = _clamp01((answered["Wait Time"] <= 60).sum() / total) if total else 0.0
        avg_wait_min = (answered["Wait Time"].mean() / 60.0) if len(answered) else 0.0
        abandon_frac = _clamp01((g["Abandoned After"] > 20).sum() / total) if total else 0.0
        excess_wait  = max(avg_wait_min - CHAT_TARGET_WAIT_MIN, 0.0)
        raw = 0.5 * frac_60 - 0.3 * excess_wait - 0.2 * abandon_frac
        return max(0.0, min(100.0, (raw / CHAT_RESCALE_K) * SLA_SCALE))

    chat_hour_sla = (
        chat_day.groupby("Hour")
        .apply(_chat_sla_for_group, include_groups=False)
        .rename("Chat SLA")
    )

    def _email_sla_for_group(g: pd.DataFrame) -> float:
        if g.empty: return None
        total = len(g)
        frac_le_1hr = _clamp01((g["Elapsed Time (Hours)"] <= 1).sum() / total) if total else 0.0
        avg_resp_hr = g["Elapsed Time (Hours)"].mean() if total else 0.0
        excess = max((avg_resp_hr or 0.0) - 1.0, 0.0)
        raw = 0.6 * frac_le_1hr - 0.4 * excess
        return max(0.0, min(100.0, (raw / EMAIL_RESCALE_K) * SLA_SCALE))

    email_hour_sla = (
        email_day.groupby("Hour")
        .apply(_email_sla_for_group, include_groups=False)
        .rename("Email SLA")
    )

    out = (
        out.set_index("Hour")
           .join([chat_hour_sla, email_hour_sla, chat_vol_hour, email_vol_hour])
           .reset_index()
    )

    avail_df = compute_hourly_available_minutes_and_logged_in(sel_date)
    out = out.merge(avail_df, on="Hour", how="left")

    def _weighted(row):
        cv = row.get("Chat Vol", 0) or 0
        ev = row.get("Email Vol", 0) or 0
        denom = cv + ev
        if denom == 0: return None
        cs = row.get("Chat SLA", 0) or 0
        es = row.get("Email SLA", 0) or 0
        return (cs * cv + es * ev) / denom

    out["Weighted SLA"] = out.apply(_weighted, axis=1)
    out["HourLabel"] = out["Hour"].dt.strftime("%H:%M")
    return out

df_hourly = compute_hourly_sla_for_date(hourly_date)

if df_hourly[["Chat Vol","Email Vol"]].fillna(0).sum().sum() == 0:
    st.info("No activity for the selected day.")
else:
    show_breakdown = st.checkbox("Show Chat & Email lines", value=False)
    show_avail     = st.checkbox("Overlay available minutes (bars)", value=True)

    # -------- Two-axis composition --------
    # Left axis owner: Weighted SLA (and optional Chat/Email lines share it without axes)
    weighted_line = (
        alt.Chart(df_hourly)
        .mark_line(point=True, color="#2F80ED")
        .encode(
            x=alt.X("Hour:T", title="Hour", axis=alt.Axis(format="%H:%M", labelAngle=-45)),
            y=alt.Y("Weighted SLA:Q", title="Weighted SLA",
                    scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(orient="left")),
            tooltip=[
                alt.Tooltip("Hour:T", title="Hour", format="%H:%M"),
                alt.Tooltip("Weighted SLA:Q", format=".1f"),
                alt.Tooltip("Chat Vol:Q", title="Chat Vol", format=".0f"),
                alt.Tooltip("Email Vol:Q", title="Email Vol", format=".0f"),
                alt.Tooltip("Avail (min):Q", title="Available (min)", format=".0f"),
                alt.Tooltip("Logged In Agents:Q", title="Logged In Agents", format=".0f"),
            ],
        )
    )
    left_layers = [weighted_line]

    if show_breakdown:
        chat_line = (
            alt.Chart(df_hourly)
            .mark_line(point=True, color="#0EA5E9")
            .encode(
                x=alt.X("Hour:T", axis=alt.Axis(format="%H:%M", labelAngle=-45)),
                y=alt.Y("Chat SLA:Q", title=None, axis=None, scale=alt.Scale(domain=[0,105])),
                tooltip=[alt.Tooltip("Hour:T", format="%H:%M"),
                         alt.Tooltip("Chat SLA:Q", format=".1f")],
            )
        )
        email_line = (
            alt.Chart(df_hourly)
            .mark_line(point=True, color="#EF4444", strokeDash=[5,3])
            .encode(
                x=alt.X("Hour:T", axis=alt.Axis(format="%H:%M", labelAngle=-45)),
                y=alt.Y("Email SLA:Q", title=None, axis=None, scale=alt.Scale(domain=[0,105])),
                tooltip=[alt.Tooltip("Hour:T", format="%H:%M"),
                         alt.Tooltip("Email SLA:Q", format=".1f")],
            )
        )
        left_layers += [chat_line, email_line]

    target_val = 85
    rule = alt.Chart(pd.DataFrame({"y":[target_val]})).mark_rule(color="red", strokeDash=[5,5]).encode(y="y:Q")
    rule_lb = alt.Chart(pd.DataFrame({"y":[target_val]})).mark_text(align="left", dy=-8, color="red")\
        .encode(y="y:Q", text=alt.value(f"Target: {target_val}%"))
    left_chart = alt.layer(*left_layers, rule, rule_lb).resolve_scale(y="shared")

    # Right axis owner: Available minutes bars
    right_chart = (
        alt.Chart(df_hourly)
        .mark_bar(opacity=0.28, color="#8B5CF6")
        .encode(
            x=alt.X("Hour:T", axis=alt.Axis(format="%H:%M", labelAngle=-45)),
            y=alt.Y("Avail (min):Q", title="Available Minutes",
                    axis=alt.Axis(orient="right"), scale=alt.Scale(nice=True)),
            tooltip=[alt.Tooltip("Hour:T", title="Hour", format="%H:%M"),
                     alt.Tooltip("Avail (min):Q", title="Available (min)", format=".0f")],
        )
    ) if show_avail else None

    if right_chart is not None:
        combined_top = alt.layer(left_chart, right_chart).resolve_scale(y="independent")
    else:
        combined_top = left_chart

    combined_top = combined_top.properties(width=900, height=360, title=f"Hourly Weighted SLA — {hourly_date:%d %b %Y}")\
        .configure_axis(grid=True, gridColor="#e5e7eb", gridDash=[2,3])\
        .configure_view(stroke="#d1d5db", fill="white")

    st.altair_chart(combined_top, use_container_width=True)

    # Logged-in agents (separate big chart)
    max_agents_val = pd.to_numeric(df_hourly["Logged In Agents"], errors="coerce").max()
    max_agents = int((0 if pd.isna(max_agents_val) else max_agents_val) + 1)

    agents_chart = (
        alt.Chart(df_hourly)
        .mark_bar(color="#6B7280", opacity=0.70)
        .encode(
            x=alt.X("Hour:T", title="Hour", axis=alt.Axis(format="%H:%M", labelAngle=-45, labelLimit=140)),
            y=alt.Y("Logged In Agents:Q", title="Logged In Agents", scale=alt.Scale(domain=[0, max_agents])),
            tooltip=[alt.Tooltip("Hour:T", title="Hour", format="%H:%M"),
                     alt.Tooltip("Logged In Agents:Q", format=".0f")],
        )
        .properties(width=900, height=220, title="Logged In Agents per Hour")
        .configure_axis(grid=True, gridColor="#e5e7eb", gridDash=[2,3])
        .configure_view(stroke="#d1d5db", fill="white")
    )
    st.altair_chart(agents_chart, use_container_width=True)

    with st.expander("View hourly table"):
        show_cols = ["Hour","Chat SLA","Chat Vol","Email SLA","Email Vol","Avail (min)","Logged In Agents","Weighted SLA"]
        st.dataframe(
            df_hourly[show_cols].style.format({
                "Chat SLA": "{:.1f}",
                "Email SLA": "{:.1f}",
                "Weighted SLA": "{:.1f}",
                "Avail (min)": "{:.0f}",
                "Logged In Agents": "{:.0f}",
            }),
            use_container_width=True
        )

# =========================
# 👥 Daily Schedule Summary (end_date)
# =========================
st.markdown("---")
st.subheader(f"👥 Daily Schedule Summary — {end_date:%d %b %Y}")

def build_daily_schedule(df_shifts_tidy: pd.DataFrame, df_presence: pd.DataFrame, day: datetime.date) -> pd.DataFrame:
    """
    For each agent scheduled overlapping `day` (from normalized shifts.csv), show:
      - Scheduled Shift Start / End (HH:MM)
      - Lunch Start / End (from presence where status contains 'lunch')
      - Total Shift (scheduled hh:mm), Logged-in/Available (hh:mm) within scheduled,
        Adherence %, Availability %,
      - Login / Logout (AVAILABLE statuses only, full day),
      - Late/Early mins (based on ANY presence),
      plus hidden helpers for styling.
    """
    if df_shifts_tidy is None or df_shifts_tidy.empty:
        return pd.DataFrame(columns=[
            "Agent","Shift Start","Lunch Start","Lunch End","Shift End","Total Shift",
            "Logged-in (hh:mm)","Available (hh:mm)","Adherence %","Availability %",
            "Login","Logout","Late Start (min)","Early Finish (min)",
            "_shift_start_dt","_lunch_start_dt"
        ])

    day_start = datetime.combine(day, datetime.min.time())
    day_end   = day_start + timedelta(days=1)

    # Consider any shift that overlaps the selected day
    sched = df_shifts_tidy[
        (df_shifts_tidy["Shift Start"] < day_end) &
        (df_shifts_tidy["Shift End"]   > day_start)
    ].copy()

    if sched.empty:
        return pd.DataFrame(columns=[
            "Agent","Shift Start","Lunch Start","Lunch End","Shift End","Total Shift",
            "Logged-in (hh:mm)","Available (hh:mm)","Adherence %","Availability %",
            "Login","Logout","Late Start (min)","Early Finish (min)",
            "_shift_start_dt","_lunch_start_dt"
        ])

    # Presence overlapping the day (normalize agent and status)
    pres_day = df_presence[(df_presence["Start DT"] < day_end) &
                           (df_presence["End DT"]   > day_start)].copy()
    pres_day["__status_norm"] = pres_day["Service Presence Status: Developer Name"].apply(_norm_status_key)
    pres_day["__agent_key"]   = pres_day["Created By: Full Name"].apply(_norm_person_key)

    AVAILABLE_STATUSES = {"available_chat","available_email_and_web","available_all"}

    rows = []
    for _, r in sched.iterrows():
        agent = str(r["Agent"])
        agent_key = _norm_person_key(agent)

        s_sched = r["Shift Start"]; e_sched = r["Shift End"]
        sched_clip_s = max(s_sched, day_start)
        sched_clip_e = min(e_sched, day_end)
        if sched_clip_e <= sched_clip_s:
            continue

        sched_secs = (sched_clip_e - sched_clip_s).total_seconds()

        # Agent presence for the day
        pa = pres_day[pres_day["__agent_key"] == agent_key]

        def _clip_to_sched(s, e):
            cs, ce = max(s, sched_clip_s), min(e, sched_clip_e)
            return (cs, ce) if ce > cs else None

        # Logged-in intervals (ANY presence) — for adherence and late/early
        logged_ints = []
        for _, pr in pa.iterrows():
            seg = _clip_to_sched(pr["Start DT"].to_pydatetime(), pr["End DT"].to_pydatetime())
            if seg: logged_ints.append(seg)
        logged_ints = merge_intervals(logged_ints)
        logged_secs = sum((e - s).total_seconds() for s, e in logged_ints)

        # Available intervals within scheduled (for availability %)
        avail_ints = []
        for _, pr in pa[pa["__status_norm"].isin(AVAILABLE_STATUSES)].iterrows():
            seg = _clip_to_sched(pr["Start DT"].to_pydatetime(), pr["End DT"].to_pydatetime())
            if seg: avail_ints.append(seg)
        avail_ints = merge_intervals(avail_ints)
        avail_secs = sum((e - s).total_seconds() for s, e in avail_ints)

        # Lunch from presence = any status containing 'lunch'
        lunch_rows = pa[pa["__status_norm"].str.contains("lunch", na=False)]
        lunch_ints = []
        for _, lr in lunch_rows.iterrows():
            seg = _clip_to_sched(lr["Start DT"].to_pydatetime(), lr["End DT"].to_pydatetime())
            if seg: lunch_ints.append(seg)
        lunch_ints = merge_intervals(lunch_ints)
        lunch_start = min((s for s, _ in lunch_ints), default=None)
        lunch_end   = max((e for _, e in lunch_ints), default=None)

        # Login/Logout from AVAILABLE statuses over the full day (not clipped to schedule)
        avail_day_ints = []
        for _, pr in pa[pa["__status_norm"].isin(AVAILABLE_STATUSES)].iterrows():
            cs, ce = max(pr["Start DT"], day_start), min(pr["End DT"], day_end)
            if ce > cs:
                avail_day_ints.append((cs.to_pydatetime(), ce.to_pydatetime()))
        avail_day_ints = merge_intervals(avail_day_ints)
        login_avail  = min((s for s, _ in avail_day_ints), default=None)
        logout_avail = max((e for _, e in avail_day_ints), default=None)

        # First/Last presence across the day (ANY presence) for late/early mins
        all_pres_ints = []
        for _, pr in pa.iterrows():
            cs, ce = max(pr["Start DT"], day_start), min(pr["End DT"], day_end)
            if ce > cs:
                all_pres_ints.append((cs.to_pydatetime(), ce.to_pydatetime()))
        all_pres_ints = merge_intervals(all_pres_ints)
        first_login_any = min((s for s, _ in all_pres_ints), default=None)
        last_logout_any = max((e for _, e in all_pres_ints), default=None)

        # Late start / early finish vs scheduled (mins) using ANY presence
        late_start_min   = round(max(((first_login_any - sched_clip_s).total_seconds()/60.0), 0.0), 1) if first_login_any else None
        early_finish_min = round(max(((sched_clip_e - last_logout_any).total_seconds()/60.0), 0.0), 1)  if last_logout_any else None

        adher_pct = (logged_secs / sched_secs * 100.0) if sched_secs > 0 else None
        avail_pct = (avail_secs  / sched_secs * 100.0) if sched_secs > 0 else None

        rows.append({
            "Agent":               agent,
            "Shift Start":         sched_clip_s.strftime("%H:%M"),
            "Lunch Start":         ("—" if lunch_start is None else lunch_start.strftime("%H:%M")),
            "Lunch End":           ("—" if lunch_end   is None else lunch_end.strftime("%H:%M")),
            "Shift End":           sched_clip_e.strftime("%H:%M"),
            "Total Shift":         fmt_hhmm(sched_secs),          # HH:MM
            "Logged-in (hh:mm)":   fmt_hhmm(logged_secs),         # HH:MM
            "Available (hh:mm)":   fmt_hhmm(avail_secs),          # HH:MM
            "Adherence %":         (round(adher_pct, 1) if adher_pct is not None else None),
            "Availability %":      (round(avail_pct, 1) if avail_pct is not None else None),
            "Login":               fmt_hhmm_dt(login_avail),
            "Logout":              fmt_hhmm_dt(logout_avail),
            "Late Start (min)":    fmt_minutes_clean(late_start_min),
            "Early Finish (min)":  fmt_minutes_clean(early_finish_min),

            # hidden helper columns for styling
            "_shift_start_dt":     sched_clip_s,
            "_lunch_start_dt":     lunch_start
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_key = pd.to_datetime(out["Shift Start"], format="%H:%M", errors="coerce")
    out = out.assign(_sort=sort_key).sort_values(["_sort","Agent"]).drop(columns=["_sort"]).reset_index(drop=True)
    return out

schedule_df = build_daily_schedule(df_shifts, df_presence, end_date)

if schedule_df.empty:
    st.info(f"No scheduled agents found for {end_date:%d %b %Y}.")
else:
    # Compute lunch-window violations using hidden datetime helpers
    df = schedule_df.copy()

    # Hours delta from shift start to lunch start (NaN if no lunch)
    delta_hours = (
        (pd.to_datetime(df["_lunch_start_dt"]) - pd.to_datetime(df["_shift_start_dt"]))
        .dt.total_seconds() / 3600
    )

    # Violation: lunch < 3h OR > 5h from shift start (only when lunch exists)
    viol_idx = df.index[delta_hours.notna() & ((delta_hours < 3.0) | (delta_hours > 5.0))]

    # Drop helper columns from what we display
    display_cols = [c for c in df.columns if c not in {"_shift_start_dt", "_lunch_start_dt"}]
    disp = df[display_cols].copy()

    # Row-wise styling: make Lunch cells red when violated
    lunch_cols = {"Lunch Start", "Lunch End"}
    def _style_row(row):
        is_violation = row.name in viol_idx
        styles = []
        for col in disp.columns:
            if is_violation and col in lunch_cols:
                styles.append("color: red; font-weight: 600;")
            else:
                styles.append("")
        return styles

    left, right = st.columns([4,1])
    with left:
        st.dataframe(
            disp.style.apply(_style_row, axis=1),
            use_container_width=True
        )
    with right:
        st.metric("Scheduled agents", f"{len(disp):,}")
        # Sum total shift seconds from HH:MM strings
        def _hhmm_to_sec(s):
            if not isinstance(s, str) or ":" not in s:
                return 0
            h, m = s.split(":")[:2]
            return int(h)*3600 + int(m)*60
        total_secs = sum(_hhmm_to_sec(x) for x in disp["Total Shift"])
        st.metric("Total scheduled time", fmt_hms(total_secs))

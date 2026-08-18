"""
Department Performance Dashboard
================================

Revision notes (this version fixes 27 issues found in a full calculation review):

CRITICAL
  1.  Chat & Email Utilization now share ONE definition (proportional split of
      shared 'Available_All' time). Previously chat used full availability and
      email used a proportional share, making the two tiles differ by ~2x for
      identical workloads.
  2.  FCR is now averaged over surveys that actually answered the FCR question
      (was counting unanswered surveys as failures).
  3.  shifts.csv is now OPTIONAL and is actually used -> Schedule Adherence tile.
  4.  email 'Completion Date' is now used as a fallback/derivation for elapsed
      time; survey 'Channel' now drives a per-channel feedback breakdown.
  5.  Rows with a missing end timestamp are clamped to the export horizon rather
      than silently dropped, and all exclusions are reported in a data-quality
      panel. Volume counts no longer lose in-progress work items.

SLA SCORING
  6.  Scores now reach a true 100 at perfect performance (was capped at 96).
  7.  Wait-time / response-time penalties are bounded ratios instead of
      unbounded linear terms (a single stuck contact no longer zeroes the day).
  8.  Abandoned chats are counted ONCE (answer-rate denominator is now answered
      chats; abandonment is captured solely by its own penalty term).
  9.  Per-respondent CSAT/NPS use the mean of their answers, not the maximum.
  10. Survey questions are mapped from an explicit config, with a warning for
      unmapped question titles, instead of overlapping keyword matching.
  11. The CSAT response scale is declared and validated instead of assumed 1-5.
  12. Daily NPS is suppressed below a minimum sample size.

ROBUSTNESS / CORRECTNESS
  13. Zero-volume days are NaN (a gap in the trend) rather than a real 0.
  14. Daily SLA is weighted by the same population it was computed from.
  15. Missed-chat de-dup no longer crashes on a missing button column.
  16. De-dup skips rows with no usable customer key instead of bucketing them.
  17. De-dup uses a true window-from-first-kept anchor.
  18. The hourly view no longer raises TypeError on a day with only one channel.
  19. The date picker no longer crashes when the data spans fewer than 7 days.
  20. Wait Time / Abandoned After / Elapsed Time are coerced to numeric.
  21. Work items straddling the window boundary are included in utilisation.

HYGIENE
  22. ~120 lines of dead code removed; three interval-merge implementations
      collapsed into one. Daily SLA components are now surfaced in a table.
  23. Country names keep their canonical casing.
  24. Two-letter country codes require an exact token match; unmapped chat
      buttons are surfaced so mapping gaps are visible.
  25. SLA trend x-axis padding is symmetric.
  26. Streamlit chart width API is consistent (with a compatibility shim).
  27. Email colour threshold aligned to 60 min; dark-mode-safe KPI tiles;
      empty-data tiles show a dash; timezone handling is explicit.
"""

import re
import unicodedata
import datetime as _dt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# =============================================================================
# Page config & styling
# =============================================================================
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# FIX #27: theme-aware tiles. The previous CSS hard-coded a white background and
# dark text, which made the whole KPI row unreadable in Streamlit's dark theme.
st.markdown(
    """
<style>
.main .block-container { padding-top: 2rem; }

.metric-container {
    padding: 12px;
    border-radius: 8px;
    background-color: var(--background-color, transparent);
    box-shadow: 0 2px 4px rgba(0,0,0,0.10);
    text-align: center;
    border: 2px solid transparent;
}
.metric-title { font-size: 1.05em; opacity: 0.85; margin-bottom: 4px; }
.metric-value { font-size: 1.8em; font-weight: 700; }
.metric-sub   { font-size: 0.80em; opacity: 0.65; margin-top: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CONFIGURATION
# -----------------------------------------------------------------------------
# Everything that used to be a magic number now lives here so the scoring model
# can be tuned without hunting through the code.
# =============================================================================

# ---- Chat SLA -----------------------------------------------------------
CHAT_ANSWER_TARGET_SEC = 60.0   # "answered quickly" threshold
CHAT_ABANDON_AFTER_SEC = 20.0   # a chat counts as abandoned beyond this wait
CHAT_TARGET_WAIT_MIN = 1.0      # no wait penalty at or below this average
CHAT_WAIT_TOLERANCE_MIN = 4.0   # FIX #7: average wait this far ABOVE target = full penalty

W_CHAT_ANSWER = 0.50            # reward: share of answered chats within target
W_CHAT_WAIT = 0.30              # penalty: average wait above target
W_CHAT_ABANDON = 0.20           # penalty: share of chats abandoned

# ---- Email SLA ----------------------------------------------------------
EMAIL_TARGET_HRS = 1.0          # no response penalty at or below this average
# The penalty statistic is the 85th percentile of response time expressed in
# multiples of each case's own target — "85% of emails were answered within this
# many times their target".
#
# It replaces the mean, which a handful of stuck cases could drag high enough to
# max out the penalty on its own: across 89 days of real data the mean floored
# the score at zero on 25% of them while the hit rate was still moving. p85 is
# as stable day to day as the mean (average swing 1.26 vs 1.27 multiples) but
# floors on only 6%. p95+ is far too jumpy (swing 6.8) and p75 correlates so
# closely with the hit rate that the penalty stops adding information.
EMAIL_PENALTY_PERCENTILE = 0.85

# ---- Reopened cases are measured differently -----------------------------
# For a case routed only once, 'Elapsed Time (Hours)' matches time-to-first-
# close within 0.02h — it genuinely is a first-response time, judged against the
# case's own entitlement target.
#
# For a reopened case it tracks the LAST close instead (0.12h away, vs 2.37h
# from the first), so it is a RESOLUTION time spanning every response cycle and
# the customer's own reply time in between. Judging that against a 1-hour
# response target measures the wrong thing. It gets its own resolution target,
# and the two populations are scored separately then blended by volume.
EMAIL_REOPENED_TARGET_HRS = 12.0

# Tolerance in MULTIPLES OF TARGET, not hours, so a 4-hour entitlement gets
# proportionate slack instead of being judged on an absolute hour count.
# 6.0 => full penalty once p85 reaches 7x target.
EMAIL_RESP_TOLERANCE_TARGETS = 6.0

W_EMAIL_IN_TARGET = 0.60        # reward: share replied within target
W_EMAIL_RESP = 0.40             # penalty: average response above target

# FIX #6: a flawless day must score exactly 100. The rescale divisor is the
# maximum achievable raw score, so raw==max maps to SLA_SCALE.
SLA_SCALE = 100.0
CHAT_RESCALE_K = W_CHAT_ANSWER      # 0.50 -> perfect chat day == 100
EMAIL_RESCALE_K = W_EMAIL_IN_TARGET  # 0.60 -> perfect email day == 100

SLA_TARGET = 85.0               # target line drawn on the trend chart

# ---- Presence statuses --------------------------------------------------
STATUS_CHAT_ONLY = "Available_Chat"
STATUS_EMAIL_ONLY = "Available_Email_and_Web"
STATUS_SHARED = "Available_All"
AVAILABLE_STATUSES = {STATUS_CHAT_ONLY, STATUS_EMAIL_ONLY, STATUS_SHARED}

CHAT_DEVNAME = "sfdc_liveagent"
EMAIL_DEVNAME = "casesChannel"

# ---- Excluded people ----------------------------------------------------
# Names listed here are removed from every metric: their work items drop out of
# volume, AHT and utilisation, and their presence time drops out of the
# availability denominator and schedule adherence.
#
# Matching is accent- and word-order-insensitive ("Ferreira, Thammy" == "Thammy
# Ferreira"), but it still needs the name to appear in the data. A configured
# name that matches nothing is reported in the data-quality panel rather than
# passing silently — a typo here would otherwise look like a working filter.
#
# NOTE: this cannot be applied to the email SLA. The email milestone export has
# no case-owner column, so there is no way to tell whose cases they are. Add a
# 'Case Owner' column to that export and set EMAIL_OWNER_COLUMN below.
EXCLUDED_OWNERS = [
    "Adam Goodridge",
    "David Balish",
    "Hamond Manouan",
    "Thammy Ferreira",
    "Felipe Schmid",
]

# Candidate owner columns, tried in order, in chat.csv and the email export.
# Leave as-is; the first one present is used.
OWNER_COLUMN_CANDIDATES = [
    "Case Owner", "Case Owner: Full Name", "Owner", "Owner: Full Name",
    "Assigned To", "Agent", "Agent: Full Name", "User: Full Name",
    "Created By: Full Name", "Owner Name",
]

# Case identifier in email.csv, joined to 'Work Item: Name' in report_items.csv
# for the exact volume/SLA reconciliation.
EMAIL_CASE_KEY = "Case Number"

# A case can carry several entitlement milestones. "lenient" keeps the longest
# target (the wider entitlement actually promised); "strictest" keeps the shortest.
EMAIL_MILESTONE_PREFERENCE = "lenient"

# Status values that mean the case is finished. Used to tell a genuinely OPEN
# case (no reply yet -> a breach once past target) apart from a case that was
# closed while its response milestone was never completed (unmeasurable ->
# excluded from the score, but reported).
EMAIL_CLOSED_STATUSES = {"resolved", "closed", "completed"}

# ---- Survey -------------------------------------------------------------
# FIX #10: explicit question mapping. Keyword matching previously let a single
# question land in two categories ("How satisfied were you that your issue was
# resolved?" scored as both CSAT and FCR) and matched "likely" in non-NPS
# questions. Patterns are tried in order; first match wins; a question matching
# none is reported so new survey questions do not silently vanish.
SURVEY_QUESTION_PATTERNS = [
    ("NPS",  r"\brecommend\b"),
    ("FCR",  r"\bresolved by the agent\b"
             r"|\bfirst contact\b"
             r"|\bresolved (?:on|at) first\b"
             r"|\bwas (?:the|your) (?:issue|query|problem|enquiry|inquiry|case|request|question)\b.*\bresolv"
             r"|\bresolve[d]? your (?:issue|query|problem|enquiry|inquiry|case|request|question)\b"),
    ("CSAT", r"\bsatisf(?:ied|action)\b|\bhow would you rate\b"),
]

# FIX #11: declared, validated response scales (was hard-coded 1-5 with a
# .clip() that silently hid out-of-range values).
CSAT_SCALE_MIN, CSAT_SCALE_MAX = 1, 5
NPS_SCALE_MIN, NPS_SCALE_MAX = 0, 10
NPS_MIN_SAMPLE = 10             # FIX #12: suppress daily NPS below this many responses

# ---- Display thresholds -------------------------------------------------
UTIL_GOOD, UTIL_WARN = 0.50, 0.30
SLA_GOOD, SLA_WARN = 80.0, 70.0
CSAT_GOOD_PCT, FCR_GOOD_PCT = 70.0, 50.0
EMAIL_RESP_RED_SEC = EMAIL_TARGET_HRS * 3600  # FIX #27: was 59 min, target is 60

COLOR_GOOD, COLOR_WARN, COLOR_BAD, COLOR_NONE = "#4CAF50", "#FFC107", "#F44336", "#9E9E9E"

# FIX #27: all timestamps are naive local time. If the Salesforce export is in
# UTC, set this to shift every timestamp into the reporting timezone. The hourly
# heatmap and hourly SLA chart are meaningless if this is wrong.
SOURCE_UTC_OFFSET_HOURS = 0.0


# =============================================================================
# Small helpers
# =============================================================================
def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def fmt_mmss(sec) -> str:
    if sec is None or pd.isna(sec):
        return "—"
    m, s = divmod(int(sec), 60)
    return f"{m:02}:{s:02}"


def fmt_hhmm(sec) -> str:
    if sec is None or pd.isna(sec):
        return "—"
    sec = int(round(sec))
    return f"{sec // 3600:02}:{(sec % 3600) // 60:02}"


def fmt_hms(sec) -> str:
    if sec is None or pd.isna(sec):
        return "—"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


def fmt_pct(v, dp: int = 1) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.{dp}f}%"


def render_custom_metric(container, title, value, tooltip, color, sub: str = ""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    container.markdown(
        f"""
        <div class="metric-container" style="border-color:{color}" title="{tooltip}">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_utilization_color(util):
    if util is None or pd.isna(util):
        return COLOR_NONE
    return COLOR_GOOD if util >= UTIL_GOOD else (COLOR_WARN if util >= UTIL_WARN else COLOR_BAD)


def get_email_resp_time_color(sec):
    if sec is None or pd.isna(sec):
        return COLOR_NONE
    return COLOR_BAD if sec > EMAIL_RESP_RED_SEC else COLOR_GOOD


def get_sla_score_color(score):
    if score is None or pd.isna(score):
        return COLOR_NONE
    return COLOR_GOOD if score >= SLA_GOOD else (COLOR_WARN if score >= SLA_WARN else COLOR_BAD)


def get_csat_color_pct(v):
    if v is None or pd.isna(v):
        return COLOR_NONE
    return COLOR_BAD if v < CSAT_GOOD_PCT else COLOR_GOOD


def get_nps_color(v):
    if v is None or pd.isna(v):
        return COLOR_NONE
    return COLOR_BAD if v < 0 else COLOR_GOOD


def get_fcr_color_pct(v):
    if v is None or pd.isna(v):
        return COLOR_NONE
    return COLOR_BAD if v < FCR_GOOD_PCT else COLOR_GOOD


def altair_chart(chart, **kwargs):
    """FIX #26: one chart-width call site, tolerant of both Streamlit APIs."""
    try:
        st.altair_chart(chart, width="stretch", **kwargs)
    except TypeError:
        st.altair_chart(chart, use_container_width=True, **kwargs)


# =============================================================================
# Interval maths
# FIX #22: this is now the ONLY implementation. The original file carried three
# near-identical mergers (merge_intervals / coalesce_intervals / _merge_intervals)
# with subtly different null handling, plus unused interval_seconds/total_overlap.
# =============================================================================
Interval = Tuple[pd.Timestamp, pd.Timestamp]


def merge_intervals(intervals) -> List[Interval]:
    """Merge overlapping/adjacent [start, end] intervals. Invalid rows are dropped."""
    if not intervals:
        return []

    norm = []
    for pair in intervals:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        s, e = pd.to_datetime(pair[0], errors="coerce"), pd.to_datetime(pair[1], errors="coerce")
        if pd.notna(s) and pd.notna(e) and e > s:
            norm.append([s, e])

    if not norm:
        return []

    norm.sort(key=lambda x: x[0])
    merged = [norm[0]]
    for s, e in norm[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def clip_to_window(s, e, wstart, wend) -> Optional[Interval]:
    if pd.isna(s) or pd.isna(e):
        return None
    s2, e2 = max(s, wstart), min(e, wend)
    return (s2, e2) if e2 > s2 else None


def sum_secs(intervals) -> float:
    return float(sum((e - s).total_seconds() for s, e in intervals))


def intersect_sum(a_ints, b_ints) -> float:
    """Total seconds where the two interval sets overlap."""
    a, b = merge_intervals(a_ints), merge_intervals(b_ints)
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += (e - s).total_seconds()
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def _norm_person_key(s: str) -> str:
    """Lowercase, strip accents, sort tokens so 'A B' == 'B A'.

    FIX #22: this used to be dead code. It is now used to join shifts.csv agent
    names to presence 'Created By: Full Name' for the adherence metric.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(sorted(re.findall(r"[a-z0-9]+", s.lower())))


# =============================================================================
# SLA scoring
# FIX #7 / #8: bounded penalties, single-counted abandonment. These two functions
# are the ONLY place the formulas live, so the daily and hourly views can never
# drift apart (previously the logic was duplicated).
# =============================================================================
def chat_sla_from_slice(g: pd.DataFrame) -> Optional[float]:
    """Chat SLA score 0-100 for a slice of chat.csv. None if the slice is empty."""
    if g is None or len(g) == 0:
        return None

    total = len(g)
    answered = g[g["Wait Time"].notna()]
    n_answered = len(answered)

    # FIX #8: denominator is ANSWERED chats. Previously this divided by all
    # chats, so abandonment suppressed this term AND was penalised again below.
    frac_in_target = (
        float((answered["Wait Time"] <= CHAT_ANSWER_TARGET_SEC).sum()) / n_answered
        if n_answered else 0.0
    )
    avg_wait_min = (answered["Wait Time"].mean() / 60.0) if n_answered else 0.0
    abandon_frac = float((g["Abandoned After"] > CHAT_ABANDON_AFTER_SEC).sum()) / total

    # FIX #7: bounded. Was `0.3 * excess_minutes`, i.e. -57.6 points per minute,
    # so a day could floor at 0 from a single long wait.
    excess = max(avg_wait_min - CHAT_TARGET_WAIT_MIN, 0.0)
    wait_penalty = min(excess / CHAT_WAIT_TOLERANCE_MIN, 1.0) if CHAT_WAIT_TOLERANCE_MIN > 0 else (1.0 if excess else 0.0)

    raw = (W_CHAT_ANSWER * frac_in_target
           - W_CHAT_WAIT * wait_penalty
           - W_CHAT_ABANDON * abandon_frac)
    return clamp((raw / CHAT_RESCALE_K) * SLA_SCALE)


def email_scorable(g: pd.DataFrame) -> pd.DataFrame:
    """The subset of an email slice that counts toward the SLA.

    Drops emails with no usable response time, and unanswered emails that are
    not yet past target ("not yet due"). Everything remaining is scored on
    'Effective Elapsed', so the reward term and the penalty term always refer to
    the same set of emails.
    """
    if g is None or len(g) == 0:
        return g
    if "Counts Toward SLA" not in g.columns:
        return g
    return g[g["Counts Toward SLA"]]


def _email_score_core(elapsed: pd.Series, target: pd.Series) -> Optional[float]:
    """Score one homogeneous population against its own target column."""
    total = len(elapsed)
    if not total:
        return None

    frac_in_target = float((elapsed <= target).sum()) / total

    # Penalty statistic: p85 of elapsed in multiples of target. A few very slow
    # cases can no longer set the score by themselves, while a genuine shift in
    # the bulk of the distribution still moves it.
    ratio = (elapsed / target).replace([float("inf"), float("-inf")], pd.NA).dropna()
    stat = float(ratio.quantile(EMAIL_PENALTY_PERCENTILE)) if len(ratio) else 1.0
    excess = max(stat - 1.0, 0.0)
    resp_penalty = (min(excess / EMAIL_RESP_TOLERANCE_TARGETS, 1.0)
                    if EMAIL_RESP_TOLERANCE_TARGETS > 0 else (1.0 if excess else 0.0))

    raw = W_EMAIL_IN_TARGET * frac_in_target - W_EMAIL_RESP * resp_penalty
    return clamp((raw / EMAIL_RESCALE_K) * SLA_SCALE)


def email_sla_parts(g: pd.DataFrame):
    """(first-contact score, n), (reopened score, n) for a slice."""
    g = email_scorable(g)
    if g is None or len(g) == 0:
        return (None, 0), (None, 0)
    elapsed = g["Effective Elapsed"] if "Effective Elapsed" in g.columns else g["Elapsed Time (Hours)"]
    target = g["Effective Target"] if "Effective Target" in g.columns else (
        g["Target Hrs"] if "Target Hrs" in g.columns else pd.Series(EMAIL_TARGET_HRS, index=g.index))
    if "Reopened" not in g.columns:
        return (_email_score_core(elapsed, target), len(g)), (None, 0)
    m = g["Reopened"].astype(bool)
    first = (_email_score_core(elapsed[~m], target[~m]), int((~m).sum()))
    reop = (_email_score_core(elapsed[m], target[m]), int(m.sum()))
    return first, reop


def email_sla_from_slice(g: pd.DataFrame) -> Optional[float]:
    """Email SLA 0-100: first-contact response and reopened resolution, blended by volume."""
    (sf, nf), (sr, nr) = email_sla_parts(g)
    num = den = 0.0
    for s, n in ((sf, nf), (sr, nr)):
        if s is not None and n:
            num += s * n
            den += n
    return (num / den) if den else None


# =============================================================================
# Missed-chat de-duplication
# =============================================================================
def exclude_repeat_missed_chats(
    df: pd.DataFrame,
    time_col: str = "Date/Time Opened",
    email_col: str = "Visitor Email",
    ip_col: str = "Visitor IP Address",
    button_col: str = "Chat Button: Developer Name",
    abandoned_col: str = "Abandoned After",
    window_minutes: int = 20,
) -> Tuple[pd.DataFrame, dict]:
    """
    Collapse repeat 'missed' chats from the same customer on the same chat button.

    FIX #17: a kept record anchors the window, so a customer retrying every 15
    minutes for two hours collapses to ONE record. The previous version compared
    each row to the immediately preceding row regardless of whether that row was
    kept, which is a minimum-gap rule, not a window rule.

    FIX #15: no longer raises AttributeError when the button column is absent.
    FIX #16: rows with no usable customer key are left alone rather than being
    bucketed together under the string "nan" (which deleted real missed chats on
    pandas 2.x and silently did nothing on pandas 3.x).

    Returns (dataframe, stats).
    """
    stats = {"collapsed": 0, "unkeyed": 0, "missed_in": 0, "missed_out": 0}

    if df is None or df.empty or abandoned_col not in df.columns:
        return df, stats

    dfx = df.copy()
    dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce", dayfirst=True)

    is_missed = dfx[abandoned_col].notna()
    missed = dfx[is_missed].copy()
    non_missed = dfx[~is_missed].copy()
    stats["missed_in"] = len(missed)

    if missed.empty:
        return dfx, stats

    # Customer key: email, else IP. Rows with neither are NOT de-duplicated.
    def _col_or_na(frame, col):
        if col in frame.columns:
            return frame[col]
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")

    key = _col_or_na(missed, email_col).astype("object")
    key = key.where(key.notna() & (key.astype(str).str.strip() != ""), _col_or_na(missed, ip_col))
    key = key.where(key.notna() & (key.astype(str).str.strip() != ""), pd.NA)

    # FIX #15: safe access whether or not the button column exists.
    if button_col in missed.columns:
        button = missed[button_col].astype(str)
    else:
        button = pd.Series([""] * len(missed), index=missed.index)

    missed["_cust_key"] = key
    missed["_button_key"] = button

    keyed = missed[missed["_cust_key"].notna()].copy()
    unkeyed = missed[missed["_cust_key"].isna()].copy()
    stats["unkeyed"] = len(unkeyed)

    kept_idx = []
    if not keyed.empty:
        keyed = keyed.sort_values(["_button_key", "_cust_key", time_col])
        window = pd.Timedelta(minutes=float(window_minutes))
        # FIX #17: anchor on the last KEPT timestamp within each group.
        for _, grp in keyed.groupby(["_button_key", "_cust_key"], sort=False):
            anchor = None
            for idx, ts in grp[time_col].items():
                if pd.isna(ts):
                    kept_idx.append(idx)
                    continue
                if anchor is None or (ts - anchor) >= window:
                    kept_idx.append(idx)
                    anchor = ts

    missed_dedup = pd.concat(
        [keyed.loc[kept_idx] if kept_idx else keyed.iloc[0:0], unkeyed]
    )
    missed_dedup = missed_dedup.drop(columns=["_cust_key", "_button_key"], errors="ignore")

    stats["missed_out"] = len(missed_dedup)
    stats["collapsed"] = stats["missed_in"] - stats["missed_out"]

    out = pd.concat([non_missed, missed_dedup], ignore_index=True).sort_values(time_col)
    return out.reset_index(drop=True), stats


# =============================================================================
# Wide -> tidy shifts normaliser
# =============================================================================
def normalize_wide_shifts(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide shifts matrix into tidy: Agent | Date | Shift Start | Shift End."""
    empty = pd.DataFrame(columns=["Agent", "Date", "Shift Start", "Shift End"])
    if df_raw is None or df_raw.empty:
        return empty

    cols = list(df_raw.columns)
    date_cols, date_map = [], {}
    for c in cols:
        d = pd.to_datetime(c, errors="coerce", dayfirst=True)
        if isinstance(d, pd.Timestamp) and not pd.isna(d):
            date_cols.append(c)
            date_map[c] = d.date()

    if not date_cols:
        return empty

    name_col = next((c for c in cols if c not in date_cols), cols[0])

    long = (
        df_raw.melt(id_vars=[name_col], value_vars=date_cols,
                    var_name="DateStr", value_name="Range")
        .rename(columns={name_col: "Agent"})
    )
    long["Date"] = long["DateStr"].map(date_map)
    long["Range"] = long["Range"].astype(str).str.strip()

    off_mask = long["Range"].str.fullmatch(r"(?i)\s*(off|off day|offday|na|nan|none|-|—|–)?\s*")
    long = long[~off_mask.fillna(True)]
    if long.empty:
        return empty

    rgx = re.compile(r"^\s*(\d{1,2}:\d{2}\s*[AaPp][Mm])\s*[-—–]\s*(\d{1,2}:\d{2}\s*[AaPp][Mm])\s*$")

    def _parse_range(s, d):
        m = rgx.match(s if isinstance(s, str) else "")
        if not m:
            return (pd.NaT, pd.NaT)
        start_dt = pd.to_datetime(f"{d} {m.group(1)}", format="%Y-%m-%d %I:%M %p", errors="coerce")
        end_dt = pd.to_datetime(f"{d} {m.group(2)}", format="%Y-%m-%d %I:%M %p", errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt) and end_dt <= start_dt:
            end_dt = end_dt + pd.Timedelta(days=1)
        return (start_dt, end_dt)

    parsed = long.apply(lambda r: pd.Series(_parse_range(r["Range"], r["Date"])), axis=1)
    long["Shift Start"] = pd.to_datetime(parsed[0], errors="coerce")
    long["Shift End"] = pd.to_datetime(parsed[1], errors="coerce")

    tidy = long.dropna(subset=["Shift Start", "Shift End"])[["Agent", "Date", "Shift Start", "Shift End"]]
    return tidy.reset_index(drop=True)


# =============================================================================
# File loading
# =============================================================================
BASE_DIR = Path(__file__).parent


def resolve_path_case_insensitive(filename: str) -> Path:
    p = BASE_DIR / filename
    if p.exists():
        return p
    low = filename.lower()
    for child in BASE_DIR.iterdir():
        if child.is_file() and child.name.lower() == low:
            return child
    return p


def file_signature(p: Path):
    stat = p.stat()
    return (int(stat.st_mtime), int(stat.st_size))


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, sig: tuple, **read_kwargs):
    return pd.read_csv(path_str, **read_kwargs)


chat_path = resolve_path_case_insensitive("chat.csv")
email_path = resolve_path_case_insensitive("email.csv")
survey_path = resolve_path_case_insensitive("survey.csv")
items_path = resolve_path_case_insensitive("report_items.csv")
pres_path = resolve_path_case_insensitive("report_presence.csv")
shifts_path = resolve_path_case_insensitive("shifts.csv")

# FIX #3: shifts.csv is no longer required. It used to block startup while
# contributing nothing; it is now optional AND actually used (adherence tile).
REQUIRED = (chat_path, email_path, items_path, pres_path)
missing = [p.name for p in REQUIRED if not p.exists()]
if missing:
    st.error(f"Missing required files in the app folder: {', '.join(missing)}")
    st.stop()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload data now"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Column-name canonicalisation.
#
# Salesforce report exports rename columns between runs — 'End DT' has also
# arrived as 'End Dt'. Passing a missing name to read_csv(parse_dates=...)
# raises before any of the app's own error handling can report it, so dates are
# parsed AFTER load and column names are matched ignoring case, spaces and
# punctuation. A column that still cannot be found is named explicitly.
# ---------------------------------------------------------------------------
def _colkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def canonicalise_columns(frame: pd.DataFrame, wanted, label: str) -> pd.DataFrame:
    frame.columns = frame.columns.str.strip()
    lookup = {}
    for c in frame.columns:
        lookup.setdefault(_colkey(c), c)
    renames, missing_cols = {}, []
    for want in wanted:
        if want in frame.columns:
            continue
        found = lookup.get(_colkey(want))
        if found is not None:
            renames[found] = want
        else:
            missing_cols.append(want)
    if renames:
        frame = frame.rename(columns=renames)
        _col_renames.setdefault(label, []).extend(f"{k} → {v}" for k, v in renames.items())
    if missing_cols:
        _col_missing[label] = missing_cols
    return frame


def parse_dt(frame: pd.DataFrame, cols):
    """Coerce datetime columns. Excel error strings ('#VALUE!') become NaT."""
    for c in cols:
        if c in frame.columns:
            frame[c] = pd.to_datetime(frame[c], errors="coerce", dayfirst=True)
    return frame


_col_renames: dict = {}
_col_missing: dict = {}

df_items = canonicalise_columns(
    load_csv_cached(str(items_path), file_signature(items_path)),
    ["Work Item: Name", "Service Channel: Developer Name", "Handle Time",
     "Queue: Name", "User: Full Name", "Start DT", "End DT"], "report_items.csv")
df_presence = canonicalise_columns(
    load_csv_cached(str(pres_path), file_signature(pres_path)),
    ["Created By: Full Name", "Service Presence Status: Developer Name",
     "Start DT", "End DT"], "report_presence.csv")
chat_sla_df = canonicalise_columns(
    load_csv_cached(str(chat_path), file_signature(chat_path)),
    ["Date/Time Opened", "Wait Time", "Abandoned After",
     "Chat Button: Developer Name"], "chat.csv")
email_sla_df = canonicalise_columns(
    load_csv_cached(str(email_path), file_signature(email_path)),
    ["Date/Time Opened", "Completion Date", "Elapsed Time (Hours)",
     "Target Response (Hours)", "Status", "Case Origin", EMAIL_CASE_KEY], "email.csv")

df_items = parse_dt(df_items, ["Start DT", "End DT"])
df_presence = parse_dt(df_presence, ["Start DT", "End DT"])
chat_sla_df = parse_dt(chat_sla_df, ["Date/Time Opened"])
email_sla_df = parse_dt(email_sla_df, ["Date/Time Opened", "Completion Date"])

df_shifts = pd.DataFrame(columns=["Agent", "Date", "Shift Start", "Shift End"])
if shifts_path.exists():
    df_shifts_raw = load_csv_cached(str(shifts_path), file_signature(shifts_path))
    df_shifts_raw.columns = df_shifts_raw.columns.str.strip()
    df_shifts = normalize_wide_shifts(df_shifts_raw)

survey_q = None
if survey_path.exists():
    survey_q = canonicalise_columns(
        load_csv_cached(str(survey_path), file_signature(survey_path), low_memory=False),
        ["Survey Taker: ID", "Survey Taker: Created Date", "Survey Question: Question Title",
         "Survey Question: Survey", "Response"], "survey.csv")
    survey_q = parse_dt(survey_q, ["Survey Taker: Created Date"])

# FIX #27: explicit timezone shift, applied once, before anything is bucketed.
if SOURCE_UTC_OFFSET_HOURS:
    _shift = pd.Timedelta(hours=SOURCE_UTC_OFFSET_HOURS)
    for _df, _cols in ((df_items, ["Start DT", "End DT"]),
                       (df_presence, ["Start DT", "End DT"]),
                       (chat_sla_df, ["Date/Time Opened"]),
                       (email_sla_df, ["Date/Time Opened", "Completion Date"])):
        for _c in _cols:
            if _c in _df.columns:
                _df[_c] = _df[_c] + _shift

# ---------------------------------------------------------------------------
# FIX #20: coerce the numeric SLA columns. If any of these exports as a duration
# string ("00:01:23") the comparisons below raise TypeError; worse, a partially
# string column becomes object dtype and .mean() silently returns nonsense.
# ---------------------------------------------------------------------------
dq: dict = {}


def _coerce_numeric(frame: pd.DataFrame, col: str, label: str):
    if col not in frame.columns:
        dq[f"missing_col_{label}"] = col
        frame[col] = pd.NA
        return
    before = frame[col].notna().sum()
    frame[col] = pd.to_numeric(frame[col], errors="coerce")
    after = frame[col].notna().sum()
    if after < before:
        dq[f"uncoercible_{label}"] = int(before - after)


# ---------------------------------------------------------------------------
# Apply the excluded-owner list.
#
# Done before any metric is computed so the exclusion reaches volume, AHT,
# utilisation, availability and adherence consistently. Row-level: if an
# excluded person was one of several people who touched a case, the case
# survives through the other agents' touches but that person's handle time and
# availability do not count.
# ---------------------------------------------------------------------------
_excluded_keys = {_norm_person_key(n): n for n in EXCLUDED_OWNERS if str(n).strip()}
_excluded_hits: dict = {n: 0 for n in _excluded_keys.values()}


def _apply_owner_exclusion(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop rows owned by an excluded person. Returns the filtered frame."""
    if not _excluded_keys or frame is None or frame.empty:
        return frame
    col = next((c for c in OWNER_COLUMN_CANDIDATES if c in frame.columns), None)
    if col is None:
        dq.setdefault("exclusion_no_owner_col", []).append(label)
        return frame
    keys = frame[col].map(_norm_person_key)
    drop = keys.isin(_excluded_keys)
    if drop.any():
        for k, cnt in keys[drop].value_counts().items():
            _excluded_hits[_excluded_keys[k]] += int(cnt)
        dq[f"excluded_rows_{label}"] = int(drop.sum())
    return frame.loc[~drop].copy()


df_items = _apply_owner_exclusion(df_items, "items")
df_presence = _apply_owner_exclusion(df_presence, "presence")
chat_sla_df = _apply_owner_exclusion(chat_sla_df, "chat")
email_sla_df = _apply_owner_exclusion(email_sla_df, "email")
if not df_shifts.empty and _excluded_keys and "Agent" in df_shifts.columns:
    _sk = df_shifts["Agent"].map(_norm_person_key)
    dq["excluded_rows_shifts"] = int(_sk.isin(_excluded_keys).sum())
    df_shifts = df_shifts.loc[~_sk.isin(_excluded_keys)].copy()

dq["excluded_hits"] = _excluded_hits
dq["excluded_unmatched"] = [n for n, c in _excluded_hits.items() if c == 0]

_coerce_numeric(chat_sla_df, "Wait Time", "wait_time")
_coerce_numeric(chat_sla_df, "Abandoned After", "abandoned_after")
_coerce_numeric(email_sla_df, "Elapsed Time (Hours)", "elapsed_hours")
_coerce_numeric(df_items, "Handle Time", "handle_time")

# ---------------------------------------------------------------------------
# Per-case SLA target.
#
# email.csv carries 'Target Response (Hours)' per case (1hr for most, but some
# are 4hr or 6hr). Judging every case against a single global 1hr target marks
# the slower-target cases as breaches when they are comfortably inside their own
# entitlement. Each case is now scored against its own target, falling back to
# EMAIL_TARGET_HRS where the column is absent or blank.
# ---------------------------------------------------------------------------
if "Target Response (Hours)" in email_sla_df.columns:
    email_sla_df["Target Hrs"] = pd.to_numeric(
        email_sla_df["Target Response (Hours)"], errors="coerce")
else:
    email_sla_df["Target Hrs"] = pd.NA
_missing_target = email_sla_df["Target Hrs"].isna() | (email_sla_df["Target Hrs"] <= 0)
email_sla_df.loc[_missing_target, "Target Hrs"] = EMAIL_TARGET_HRS
email_sla_df["Target Hrs"] = email_sla_df["Target Hrs"].astype(float)
dq["email_target_defaulted"] = int(_missing_target.sum())
dq["email_target_non_default"] = int((email_sla_df["Target Hrs"] != EMAIL_TARGET_HRS).sum())

# ---------------------------------------------------------------------------
# One row per case.
#
# A case can carry more than one entitlement milestone, producing several rows
# with identical timestamps but different targets (e.g. both a 1 Hour and a
# 6 Hour milestone). Left as-is the case is scored twice, against two different
# targets. Keep a single row per case, using the MOST LENIENT (longest) target
# on the basis that the wider entitlement is the one actually promised.
# ---------------------------------------------------------------------------
if EMAIL_CASE_KEY in email_sla_df.columns:
    email_sla_df[EMAIL_CASE_KEY] = email_sla_df[EMAIL_CASE_KEY].astype(str).str.strip()
    _before = len(email_sla_df)
    email_sla_df = (
        email_sla_df.sort_values("Target Hrs", ascending=(EMAIL_MILESTONE_PREFERENCE == "strictest"))
        .drop_duplicates(subset=[EMAIL_CASE_KEY], keep="first")
        .sort_index()
    )
    dq["email_milestone_dupes"] = int(_before - len(email_sla_df))
else:
    dq["email_no_case_key"] = True

# FIX #4: 'Completion Date' was parsed and never used. Use it to fill any missing
# elapsed time, so a blank pre-computed column no longer silently drops emails
# from every SLA calculation.
if "Completion Date" in email_sla_df.columns:
    _derived = (email_sla_df["Completion Date"] - email_sla_df["Date/Time Opened"]).dt.total_seconds() / 3600.0
    _fillable = email_sla_df["Elapsed Time (Hours)"].isna() & _derived.notna() & (_derived >= 0)
    dq["elapsed_derived_from_completion"] = int(_fillable.sum())
    email_sla_df.loc[_fillable, "Elapsed Time (Hours)"] = _derived[_fillable]

    _disagree = (
        email_sla_df["Elapsed Time (Hours)"].notna() & _derived.notna()
        & ((email_sla_df["Elapsed Time (Hours)"] - _derived).abs() > 0.5)
    )
    dq["elapsed_disagrees_with_completion"] = int(_disagree.sum())

# ---------------------------------------------------------------------------
# FIX #5: open intervals. A NaT end timestamp used to silently delete the row
# from availability (NaT > ts is False) and from volume. Clamp instead, to the
# latest end timestamp seen anywhere in the export ("as of").
# ---------------------------------------------------------------------------
_ends = [s.max() for s in (df_items["End DT"], df_presence["End DT"],
                           email_sla_df.get("Completion Date", pd.Series(dtype="datetime64[ns]")))
         if s.notna().any()]
_starts = [s.max() for s in (df_items["Start DT"], df_presence["Start DT"],
                             chat_sla_df["Date/Time Opened"], email_sla_df["Date/Time Opened"])
           if s.notna().any()]
DATA_ASOF = max(_ends + _starts) if (_ends or _starts) else pd.Timestamp.now()

# ---------------------------------------------------------------------------
# Work items ASSIGNED but never ACCEPTED.
#
# Some exports include routing attempts the agent never accepted (declined or
# timed out). They have no Accept/Close time, so Start DT / End DT are blank or
# carry an Excel error string, and there is no handle time. They cannot be
# scored or timed — but the rate is a real operational signal, so it is counted
# and shown rather than silently dropped.
# ---------------------------------------------------------------------------
_unaccepted = df_items["Start DT"].isna()
if "Handle Time" in df_items.columns:
    _unaccepted = _unaccepted | df_items["Handle Time"].isna()
dq["items_unaccepted"] = int(_unaccepted.sum())
if dq["items_unaccepted"]:
    _ua = df_items.loc[_unaccepted, "Service Channel: Developer Name"]
    dq["items_unaccepted_chat"] = int((_ua == CHAT_DEVNAME).sum())
    dq["items_unaccepted_email"] = int((_ua == EMAIL_DEVNAME).sum())
    _tot_chat = int((df_items["Service Channel: Developer Name"] == CHAT_DEVNAME).sum())
    _tot_email = int((df_items["Service Channel: Developer Name"] == EMAIL_DEVNAME).sum())
    dq["items_unaccepted_chat_pct"] = (dq["items_unaccepted_chat"] / _tot_chat * 100) if _tot_chat else 0.0
    dq["items_unaccepted_email_pct"] = (dq["items_unaccepted_email"] / _tot_email * 100) if _tot_email else 0.0

for _df, _label in ((df_items, "items"), (df_presence, "presence")):
    _open = _df["End DT"].isna() & _df["Start DT"].notna()
    dq[f"open_{_label}"] = int(_open.sum())
    _df["End DT Clamped"] = _df["End DT"].where(~_open, DATA_ASOF)
    _df["Is Open"] = _open
    dq[f"no_start_{_label}"] = int(_df["Start DT"].isna().sum())

# ---------------------------------------------------------------------------
# Email response timestamp, effective elapsed time, and "counts toward SLA".
#
# An email has a response time from either source: an explicit Completion Date,
# or Date/Time Opened + Elapsed Time (Hours). Deriving one canonical column lets
# the SLA use the OPENED-OR-RESPONDED window (see email_window_slice) and lets
# the two halves of the score agree about which emails exist.
#
# Unanswered emails: previously an email with no reply counted as a failure in
# the "replied <=1hr" term but was skipped by .mean() in the average-response
# term — the two halves of the same score disagreed. Now an unanswered email
# counts as a breach ONLY once it is already past target; one that arrived more
# recently than the target is "not yet due" and is excluded from both terms.
# ---------------------------------------------------------------------------
if "Completion Date" not in email_sla_df.columns:
    email_sla_df["Completion Date"] = pd.NaT

_from_elapsed = email_sla_df["Date/Time Opened"] + pd.to_timedelta(
    email_sla_df["Elapsed Time (Hours)"], unit="h", errors="coerce")
email_sla_df["Response DT"] = email_sla_df["Completion Date"].fillna(_from_elapsed)

email_sla_df["Is Answered"] = email_sla_df["Response DT"].notna()
_age_hrs = (DATA_ASOF - email_sla_df["Date/Time Opened"]).dt.total_seconds() / 3600.0

# Answered -> the real elapsed time. Unanswered -> how long it has been waiting.
email_sla_df["Effective Elapsed"] = email_sla_df["Elapsed Time (Hours)"].where(
    email_sla_df["Is Answered"], _age_hrs)
# A case that is already CLOSED but has no completion time for its response
# milestone cannot be measured — there is no response timestamp to compare
# against the target, and ageing it to "now" would be meaningless because the
# case is not actually still waiting. Excluded from the score and reported.
if "Status" in email_sla_df.columns:
    _closed = email_sla_df["Status"].astype(str).str.strip().str.lower().isin(EMAIL_CLOSED_STATUSES)
else:
    _closed = pd.Series(False, index=email_sla_df.index)
email_sla_df["Unmeasurable"] = ~email_sla_df["Is Answered"] & _closed
dq["email_unmeasurable"] = int(email_sla_df["Unmeasurable"].sum())

# "Not yet due" is judged against the case's OWN target, not a global 1hr.
email_sla_df["Counts Toward SLA"] = (
    email_sla_df["Effective Elapsed"].notna()
    & ~email_sla_df["Unmeasurable"]
    & (email_sla_df["Is Answered"] | (_age_hrs > email_sla_df["Target Hrs"]))
)

_still_open = ~email_sla_df["Is Answered"] & ~email_sla_df["Unmeasurable"]
dq["email_unanswered_breach"] = int((_still_open & (_age_hrs > email_sla_df["Target Hrs"])).sum())
dq["email_unanswered_not_due"] = int((_still_open & (_age_hrs <= email_sla_df["Target Hrs"])).sum())
dq["email_no_response_time"] = int((~email_sla_df["Counts Toward SLA"]
                                    & email_sla_df["Is Answered"]).sum())


# ---------------------------------------------------------------------------
# Work-item rows -> CASES.
#
# report_items.csv has one row per routing event, not per contact. An email case
# that is re-opened or handed on is routed again and produces another row, so
# counting rows overstates email volume (1.24x on real data; chats are 1.00x
# because a chat is one continuous conversation).
#
# Rolling up to the case gives: a true volume count, and a handle time that is
# the TOTAL effort across every touch of that case — including re-opens handled
# on a later day, and touches by different agents.
#
# Row duration prefers 'Handle Time' (exact seconds). Start DT / End DT are
# built from minute-resolution fields, so deriving duration from them carries a
# median error of ~18s per row (up to 59s). Individual rows matter for AHT even
# though the error cancels out in the mean.
# ---------------------------------------------------------------------------
CASE_KEY = "Work Item: Name"
HAS_CASE_KEY = CASE_KEY in df_items.columns

_row_secs_from_ts = (df_items["End DT Clamped"] - df_items["Start DT"]).dt.total_seconds()
if "Handle Time" in df_items.columns and df_items["Handle Time"].notna().any():
    df_items["Row Secs"] = df_items["Handle Time"].where(
        df_items["Handle Time"].notna() & (df_items["Handle Time"] >= 0), _row_secs_from_ts)
    dq["handle_time_used"] = int(df_items["Handle Time"].notna().sum())
else:
    df_items["Row Secs"] = _row_secs_from_ts
    dq["handle_time_used"] = 0

if HAS_CASE_KEY:
    df_cases = (
        df_items.dropna(subset=["Start DT"])
        .groupby(["Service Channel: Developer Name", CASE_KEY], as_index=False)
        .agg(
            First_Start=("Start DT", "min"),
            Last_End=("End DT Clamped", "max"),
            Touches=("Row Secs", "size"),
            Total_Secs=("Row Secs", "sum"),
            Agents=("User: Full Name", "nunique"),
            Any_Open=("Is Open", "any"),
        )
    )
    dq["cases_rolled_up"] = int(len(df_items) - len(df_cases))
else:
    # No case identifier: fall back to one case per row so the app still runs.
    df_cases = df_items.dropna(subset=["Start DT"]).assign(
        First_Start=lambda d: d["Start DT"], Last_End=lambda d: d["End DT Clamped"],
        Touches=1, Total_Secs=lambda d: d["Row Secs"],
        Agents=1, Any_Open=lambda d: d["Is Open"],
    )[["Service Channel: Developer Name", "First_Start", "Last_End",
       "Touches", "Total_Secs", "Agents", "Any_Open"]]
    dq["cases_rolled_up"] = 0
    dq["no_case_key"] = True


# ---------------------------------------------------------------------------
# Attach the routing touch count to each email case.
#
# 'Elapsed Time (Hours)' runs from ORIGINAL case open to milestone completion,
# so for a case that was reopened it accumulates every response cycle instead of
# resetting. That is a Salesforce entitlement setting (the response milestone is
# 'No Recurrence'; 'Independent' would create a fresh milestone record per
# reopen). Until that changes, the touch count from report_items is the only way
# to tell which cases are affected — so it is attached here and reported, rather
# than the distortion being left invisible.
# ---------------------------------------------------------------------------
if HAS_CASE_KEY and EMAIL_CASE_KEY in email_sla_df.columns:
    _email_cases = df_cases[df_cases["Service Channel: Developer Name"] == EMAIL_DEVNAME]
    _touch_map = pd.Series(
        _email_cases["Touches"].values,
        index=_email_cases[CASE_KEY].astype(str).str.strip().values,
    )
    _touch_map = _touch_map[~_touch_map.index.duplicated()]
    _keys = email_sla_df[EMAIL_CASE_KEY].astype(str).str.strip()
    email_sla_df["Touches"] = _keys.map(_touch_map)
    dq["email_no_routing_record"] = int(email_sla_df["Touches"].isna().sum())
    # A case with no routing record was never re-routed, so it cannot have been
    # reopened through Omni-Channel — treat it as a single response cycle.
    email_sla_df["Touches"] = email_sla_df["Touches"].fillna(1).astype(int)
    email_sla_df["Reopened"] = email_sla_df["Touches"] > 1
else:
    email_sla_df["Touches"] = 1
    email_sla_df["Reopened"] = False

# First contact keeps its entitlement response target; a reopened case is judged
# against the resolution target, because that is what its elapsed figure is.
email_sla_df["Effective Target"] = email_sla_df["Target Hrs"].where(
    ~email_sla_df["Reopened"], EMAIL_REOPENED_TARGET_HRS)


def email_window_slice(df: pd.DataFrame, w_start, w_end) -> pd.DataFrame:
    """Emails OPENED **or** RESPONDED TO within [w_start, w_end).

    An email opened before the window but answered inside it is this window's
    work; an email opened inside the window but answered after it is also this
    window's work. Each row is attributed to exactly ONE timestamp inside the
    window — its open time if it was opened here, otherwise its response time —
    so daily buckets partition the period slice instead of double-counting.
    """
    opened_in = (df["Date/Time Opened"] >= w_start) & (df["Date/Time Opened"] < w_end)
    resp_in = df["Response DT"].notna() & (df["Response DT"] >= w_start) & (df["Response DT"] < w_end)
    keep = opened_in | resp_in
    sel = df.loc[keep].copy()
    sel["SLA Timestamp"] = sel["Date/Time Opened"].where(opened_in.loc[keep], sel["Response DT"])
    return sel


# =============================================================================
# Sidebar: date range
# =============================================================================
st.sidebar.header("Filter Options")

# FIX #19 (and a correctness improvement): the selectable range now spans every
# source, not just chat.csv. Previously a period with email/work-item activity
# but no chats was unreachable.
# Take min/max on the DATETIME series, then convert. Calling .dt.date first
# turns NaT into a float nan, giving a column of mixed date/float values whose
# .min() raises TypeError — reachable as soon as an export contains an
# unparseable timestamp (e.g. an Excel '#VALUE!' string).
_all_dates = [
    chat_sla_df["Date/Time Opened"],
    email_sla_df["Date/Time Opened"],
    df_items["Start DT"],
]
_mins = [s.min().date() for s in _all_dates if s.notna().any()]
_maxs = [s.max().date() for s in _all_dates if s.notna().any()]
if not _mins:
    st.error("No usable dates found in chat.csv, email.csv or report_items.csv.")
    st.stop()
min_date, max_date = min(_mins), max(_maxs)

# FIX #19: clamp the default so a dataset spanning < 7 days does not raise
# StreamlitAPIException on load.
default_start = max(min_date, max_date - timedelta(days=6))

start_date = st.sidebar.date_input("Start Date", value=default_start,
                                   min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date,
                                 min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start must be on or before End")
    st.stop()

dedup_window_minutes = int(st.sidebar.number_input(
    "Missed de-dup window (mins)", min_value=5, max_value=60, value=20, step=5,
    help="Repeat missed chats from the same visitor on the same button within "
         "this window of the first counted attempt collapse into one record.",
))

chat_sla_df, dedup_stats = exclude_repeat_missed_chats(
    chat_sla_df, window_minutes=dedup_window_minutes
)

# =============================================================================
# Data quality panel  (FIX #5: exclusions are now visible instead of silent)
# =============================================================================
with st.expander("ℹ️ Data files & quality"):
    def fmt_sig(p: Path):
        mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime)
        return f"{p.name} • {p.stat().st_size:,} bytes • {mtime:%Y-%m-%d %H:%M:%S}"

    _files = [chat_path, email_path, items_path, pres_path]
    if shifts_path.exists():
        _files.append(shifts_path)
    if survey_path.exists():
        _files.append(survey_path)
    st.write("\n".join(f"• {fmt_sig(p)}" for p in _files))

    st.markdown(f"**Export horizon (as-of):** {DATA_ASOF:%Y-%m-%d %H:%M:%S}")

    notes = []
    if dedup_stats.get("collapsed"):
        notes.append(f"Missed-chat de-dup collapsed **{dedup_stats['collapsed']}** of "
                     f"{dedup_stats['missed_in']} missed chats.")
    if dedup_stats.get("unkeyed"):
        notes.append(f"**{dedup_stats['unkeyed']}** missed chats had no email or IP and "
                     f"were left un-deduplicated (they are not assumed to be the same visitor).")
    for lbl, singular, plural in (("items", "work item", "work items"),
                                  ("presence", "presence record", "presence records")):
        _n = dq.get(f"open_{lbl}", 0)
        if _n:
            notes.append(f"**{_n}** {singular if _n == 1 else plural} had no end timestamp and "
                         f"{'was' if _n == 1 else 'were'} clamped to the export horizon "
                         f"(previously dropped entirely).")
        _n = dq.get(f"no_start_{lbl}", 0)
        if _n:
            notes.append(f"**{_n}** {singular if _n == 1 else plural} had no start timestamp and "
                         f"{'was' if _n == 1 else 'were'} excluded.")
    for key, msg in (
        ("uncoercible_wait_time", "chat 'Wait Time' values were not numeric and became blank"),
        ("uncoercible_abandoned_after", "chat 'Abandoned After' values were not numeric and became blank"),
        ("uncoercible_elapsed_hours", "email 'Elapsed Time (Hours)' values were not numeric and became blank"),
    ):
        if dq.get(key):
            notes.append(f"⚠️ **{dq[key]}** {msg}.")
    if dq.get("elapsed_derived_from_completion"):
        notes.append(f"**{dq['elapsed_derived_from_completion']}** emails had a blank elapsed time, "
                     f"derived from Completion Date − Date/Time Opened.")
    for _lbl, _rn in _col_renames.items():
        notes.append(f"`{_lbl}`: column name(s) auto-matched — {', '.join(f'`{r}`' for r in _rn)}. "
                     f"The export renamed these; the dashboard adapted, but worth keeping consistent.")
    for _lbl, _mc in _col_missing.items():
        notes.append(f"⚠️ `{_lbl}` is missing expected column(s): "
                     + ", ".join(f"`{c}`" for c in _mc) + ".")
    if dq.get("items_unaccepted"):
        notes.append(
            f"**{dq['items_unaccepted']:,}** work item(s) were assigned but never accepted "
            f"(no accept/close time, no handle time) — "
            f"{dq.get('items_unaccepted_chat', 0):,} chat ({dq.get('items_unaccepted_chat_pct', 0):.2f}% of chat) "
            f"and {dq.get('items_unaccepted_email', 0):,} email ({dq.get('items_unaccepted_email_pct', 0):.2f}% of email). "
            f"They cannot be timed so they are excluded from AHT and utilisation, but the rate is "
            f"worth watching — these are routing attempts an agent declined or let time out."
        )
    if _excluded_keys:
        _hits = {n: c for n, c in dq.get("excluded_hits", {}).items() if c}
        if _hits:
            notes.append("Excluded owners removed from every metric: "
                         + ", ".join(f"**{n}** ({c:,} rows)" for n, c in sorted(_hits.items())) + ".")
        _un = dq.get("excluded_unmatched") or []
        if _un:
            notes.append("⚠️ These names are in `EXCLUDED_OWNERS` but **match no rows in any file**, so "
                         "they are having no effect — check the spelling against the agent names in "
                         "report_items.csv, or they may simply not appear in this data: "
                         + ", ".join(f"**{n}**" for n in _un) + ".")
        if dq.get("exclusion_no_owner_col"):
            notes.append("⚠️ No owner column found in: "
                         + ", ".join(f"`{s}`" for s in dq["exclusion_no_owner_col"])
                         + ". Rows from these sources **cannot be filtered by owner** — for the email "
                           "milestone export, add a *Case Owner* column to make the exclusion apply "
                           "to the Email SLA as well as to volume and utilisation.")
    if dq.get("cases_rolled_up"):
        notes.append(f"**{dq['cases_rolled_up']:,}** report_items rows were repeat routings of a case "
                     f"already counted. Volume counts distinct cases; handle time sums every touch.")
    if dq.get("no_case_key"):
        notes.append("⚠️ report_items.csv has no 'Work Item: Name' column, so rows could not be rolled "
                     "up into cases — volume counts routing events and will overstate contacts.")
    if dq.get("handle_time_used"):
        notes.append(f"Handle time taken from the exact 'Handle Time' column for "
                     f"**{dq['handle_time_used']:,}** rows (Start/End DT are minute-resolution).")
    if dq.get("email_milestone_dupes"):
        notes.append(f"**{dq['email_milestone_dupes']:,}** duplicate milestone row(s) collapsed so each "
                     f"case is scored once, keeping the "
                     f"{'longest' if EMAIL_MILESTONE_PREFERENCE == 'lenient' else 'shortest'} target.")
    if dq.get("email_unmeasurable"):
        notes.append(f"**{dq['email_unmeasurable']:,}** case(s) are closed but their response milestone "
                     f"never completed (no completion time). Excluded from the SLA as unmeasurable — "
                     f"they are neither a confirmed hit nor a datable miss.")
    if dq.get("email_no_case_key"):
        notes.append(f"ℹ️ email.csv has no '{EMAIL_CASE_KEY}' column, so it cannot be reconciled "
                     f"case-by-case against report_items.csv.")
    if dq.get("email_target_non_default"):
        notes.append(f"**{dq['email_target_non_default']:,}** email case(s) have a response target "
                     f"other than {EMAIL_TARGET_HRS:.0f}hr and are scored against their own target.")
    if dq.get("email_target_defaulted"):
        notes.append(f"**{dq['email_target_defaulted']:,}** email case(s) had no usable "
                     f"'Target Response (Hours)' and defaulted to {EMAIL_TARGET_HRS:.0f}hr.")
    if dq.get("email_unanswered_breach"):
        notes.append(f"**{dq['email_unanswered_breach']}** email(s) have no response and are already "
                     f"past the {EMAIL_TARGET_HRS:.0f}hr target — counted as SLA breaches, using "
                     f"time-waiting-so-far as their elapsed time.")
    if dq.get("email_unanswered_not_due"):
        notes.append(f"**{dq['email_unanswered_not_due']}** email(s) have no response but are not yet "
                     f"past target — excluded from the SLA as 'not yet due'.")
    if dq.get("email_no_response_time"):
        notes.append(f"⚠️ **{dq['email_no_response_time']}** email(s) look answered but have no usable "
                     f"response time and cannot be scored.")
    if dq.get("elapsed_disagrees_with_completion"):
        notes.append(f"⚠️ **{dq['elapsed_disagrees_with_completion']}** emails have an 'Elapsed Time (Hours)' "
                     f"that disagrees with Completion Date by more than 30 minutes.")
    if not shifts_path.exists():
        notes.append("shifts.csv not found — Schedule Adherence is unavailable (optional).")

    st.markdown("\n\n".join(f"- {n}" for n in notes) if notes else "No data-quality issues detected.")


# =============================================================================
# Period slices
# =============================================================================
ts_start = pd.Timestamp(start_date)
ts_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
window_start = ts_start.to_pydatetime()
window_end = ts_end.to_pydatetime()

# Volume basis: items STARTING in the window.
mask_started = (df_items["Start DT"] >= ts_start) & (df_items["Start DT"] < ts_end)
df_period = df_items.loc[mask_started].copy()

# FIX #21: interval basis is OVERLAP with the window. Filtering on Start DT alone
# excluded work that began just before the window while still counting the
# agent's availability, biasing utilisation low at every window boundary.
mask_overlap = (df_items["Start DT"] < ts_end) & (df_items["End DT Clamped"] > ts_start)
df_overlap = df_items.loc[mask_overlap].copy()

# ---------------------------------------------------------------------------
# Volume and AHT are CASE-level. A case belongs to the period in which it was
# FIRST handled; its handle time is the total across every touch, including
# re-opens that happen later. That is what "all time spent on this case" means.
# ---------------------------------------------------------------------------
cases_period = df_cases[(df_cases["First_Start"] >= ts_start) & (df_cases["First_Start"] < ts_end)]
chat_cases = cases_period[cases_period["Service Channel: Developer Name"] == CHAT_DEVNAME]
email_cases = cases_period[cases_period["Service Channel: Developer Name"] == EMAIL_DEVNAME]

# Row-level slices are still needed for the utilisation intervals.
chat_df = df_period[df_period["Service Channel: Developer Name"] == CHAT_DEVNAME]
email_df = df_period[df_period["Service Channel: Developer Name"] == EMAIL_DEVNAME]

chat_total = len(chat_cases)
email_total = len(email_cases)
chat_aht = chat_cases["Total_Secs"].mean() if chat_total else None
email_aht = email_cases["Total_Secs"].mean() if email_total else None

# Re-work: how much of the volume is a case coming back round again.
chat_touches = float(chat_cases["Touches"].mean()) if chat_total else None
email_touches = float(email_cases["Touches"].mean()) if email_total else None
chat_reopened = int((chat_cases["Touches"] > 1).sum())
email_reopened = int((email_cases["Touches"] > 1).sum())
email_multi_agent = int((email_cases["Agents"] > 1).sum())
chat_open_n = int(chat_cases["Any_Open"].sum())
email_open_n = int(email_cases["Any_Open"].sum())

# Per-touch AHT, kept so the two definitions can be compared.
chat_aht_touch = (chat_cases["Total_Secs"].sum() / chat_cases["Touches"].sum()) if chat_total else None
email_aht_touch = (email_cases["Total_Secs"].sum() / email_cases["Touches"].sum()) if email_total else None

chat_sla_p = chat_sla_df[
    (chat_sla_df["Date/Time Opened"] >= ts_start) & (chat_sla_df["Date/Time Opened"] < ts_end)
].copy()
# Email SLA population: opened OR responded to within the period.
email_sla_p = email_window_slice(email_sla_df, ts_start, ts_end)
email_scored_p = email_scorable(email_sla_p)

# Arrivals-only slice, for the contact-volume heatmap (that chart is about when
# contacts ARRIVE, so it must not include emails that merely closed here).
email_opened_p = email_sla_df[
    (email_sla_df["Date/Time Opened"] >= ts_start) & (email_sla_df["Date/Time Opened"] < ts_end)
].copy()

avg_resp_hrs = email_scored_p["Effective Elapsed"].mean() if len(email_scored_p) else None
# FIX #27: no emails now shows a dash, not a perfect 00:00:00.
avg_resp_secs = None if (avg_resp_hrs is None or pd.isna(avg_resp_hrs)) else avg_resp_hrs * 3600


# =============================================================================
# Availability & utilisation
# FIX #1: ONE definition for both channels. Shared 'Available_All' time is split
# between chat and email in proportion to the work actually handled on each.
# =============================================================================
presence_win = df_presence[
    (df_presence["Start DT"] < ts_end) & (df_presence["End DT Clamped"] > ts_start)
].copy()

chat_only_map, email_only_map, shared_map = {}, {}, {}
for ag, grp in presence_win.groupby("Created By: Full Name"):
    co, eo, sh = [], [], []
    for _, r in grp.iterrows():
        seg = clip_to_window(r["Start DT"], r["End DT Clamped"], window_start, window_end)
        if not seg:
            continue
        status = str(r["Service Presence Status: Developer Name"]).strip()
        if status == STATUS_CHAT_ONLY:
            co.append(seg)
        elif status == STATUS_EMAIL_ONLY:
            eo.append(seg)
        elif status == STATUS_SHARED:
            sh.append(seg)
    if co:
        chat_only_map[ag] = merge_intervals(co)
    if eo:
        email_only_map[ag] = merge_intervals(eo)
    if sh:
        shared_map[ag] = merge_intervals(sh)

# Handling intervals, built from the OVERLAP slice (FIX #21).
chat_handles_map, email_handles_map = {}, {}
for devname, target in ((CHAT_DEVNAME, chat_handles_map), (EMAIL_DEVNAME, email_handles_map)):
    sub = df_overlap[df_overlap["Service Channel: Developer Name"] == devname]
    for ag, grp in sub.groupby("User: Full Name"):
        ints = [clip_to_window(s, e, window_start, window_end)
                for s, e in zip(grp["Start DT"], grp["End DT Clamped"])]
        ints = [x for x in ints if x]
        if ints:
            target[ag] = merge_intervals(ints)

dept_chat_handle = dept_email_handle = 0.0
dept_chat_avail = dept_email_avail = 0.0
util_rows = []

agents = (set(chat_only_map) | set(email_only_map) | set(shared_map)
          | set(chat_handles_map) | set(email_handles_map))

for ag in agents:
    co = chat_only_map.get(ag, [])
    eo = email_only_map.get(ag, [])
    sh = shared_map.get(ag, [])

    chat_av_union = merge_intervals(co + sh)
    email_av_union = merge_intervals(eo + sh)

    chat_hand = intersect_sum(chat_handles_map.get(ag, []), chat_av_union) if chat_av_union else 0.0
    email_hand = intersect_sum(email_handles_map.get(ag, []), email_av_union) if email_av_union else 0.0

    co_secs, eo_secs, sh_secs = sum_secs(co), sum_secs(eo), sum_secs(sh)
    total_hand = chat_hand + email_hand

    if sh_secs > 0:
        if total_hand > 0:
            sh_to_chat = sh_secs * (chat_hand / total_hand)
            sh_to_email = sh_secs * (email_hand / total_hand)
        else:
            sh_to_chat = sh_to_email = sh_secs / 2.0
    else:
        sh_to_chat = sh_to_email = 0.0

    chat_av = co_secs + sh_to_chat
    email_av = eo_secs + sh_to_email

    dept_chat_handle += chat_hand
    dept_email_handle += email_hand
    dept_chat_avail += chat_av
    dept_email_avail += email_av

    util_rows.append({
        "Agent": ag,
        "Chat handled (h)": chat_hand / 3600,
        "Chat available (h)": chat_av / 3600,
        "Chat util": (chat_hand / chat_av) if chat_av else None,
        "Email handled (h)": email_hand / 3600,
        "Email available (h)": email_av / 3600,
        "Email util": (email_hand / email_av) if email_av else None,
        "_avail_secs": co_secs + eo_secs + sh_secs,
        "_key": _norm_person_key(ag),
    })

chat_util = (dept_chat_handle / dept_chat_avail) if dept_chat_avail else None
email_util = (dept_email_handle / dept_email_avail) if dept_email_avail else None

df_util = pd.DataFrame(util_rows)

# ---------------------------------------------------------------------------
# FIX #3: Schedule adherence — shifts.csv finally does something.
# ---------------------------------------------------------------------------
adherence = None
rostered_secs = available_secs = 0.0
if not df_shifts.empty and not df_util.empty:
    shift_by_key: dict = {}
    for _, r in df_shifts.iterrows():
        seg = clip_to_window(r["Shift Start"], r["Shift End"], window_start, window_end)
        if seg:
            shift_by_key.setdefault(_norm_person_key(r["Agent"]), []).append(seg)
    shift_by_key = {k: merge_intervals(v) for k, v in shift_by_key.items()}

    avail_by_key = dict(zip(df_util["_key"], df_util["_avail_secs"]))
    rostered_secs = float(sum(sum_secs(v) for v in shift_by_key.values()))
    available_secs = float(sum(avail_by_key.get(k, 0.0) for k in shift_by_key))
    adherence = (available_secs / rostered_secs) if rostered_secs else None

    unmatched = sorted(set(shift_by_key) - set(avail_by_key))
    dq["shift_agents_unmatched"] = len(unmatched)
    dq["shift_agents_unmatched_names"] = unmatched[:10]


# =============================================================================
# Per-day SLA
# =============================================================================
daily = []
for d in pd.date_range(start_date, end_date):
    day = d.date()

    cd = chat_sla_p[chat_sla_p["Date/Time Opened"].dt.date == day]
    # Attributed to exactly one day inside the period, so daily weights sum to
    # the period weight even though the slice is an opened-OR-responded union.
    ed = email_sla_p[email_sla_p["SLA Timestamp"].dt.date == day]
    ed_scored = email_scorable(ed)

    sla_c = chat_sla_from_slice(cd)
    sla_e = email_sla_from_slice(ed)

    _ans = cd[cd["Wait Time"].notna()]
    daily.append({
        "Date": d.normalize(),
        "Chat SLA": sla_c,
        "Email SLA": sla_e,
        # Weight by the population each score was computed from...
        "Chat SLA Wt": len(cd),
        "Email SLA Wt": len(ed_scored),
        # ...and show the report_items work-item volume alongside it, so the two
        # populations can be compared rather than silently conflated.
        "Chat Vol": int((chat_cases["First_Start"].dt.date == day).sum()),
        "Email Vol": int((email_cases["First_Start"].dt.date == day).sum()),
        "Email Not Due": int(len(ed) - len(ed_scored)),
        "Chat ≤60s %": round(float((_ans["Wait Time"] <= CHAT_ANSWER_TARGET_SEC).mean() * 100), 1) if len(_ans) else None,
        "Chat Avg Wait (s)": round(float(_ans["Wait Time"].mean()), 1) if len(_ans) else None,
        "Chat Abandon %": round(float((cd["Abandoned After"] > CHAT_ABANDON_AFTER_SEC).mean() * 100), 1) if len(cd) else None,
        "Email in target %": round(float((ed_scored["Effective Elapsed"] <= ed_scored["Effective Target"]).mean() * 100), 1) if len(ed_scored) else None,
        "Email Avg Resp (h)": round(float(ed_scored["Effective Elapsed"].mean()), 3) if len(ed_scored) else None,
    })

df_daily = pd.DataFrame(daily)

# FIX #13: a day with no contacts is a GAP, not a zero. The old fillna(0) drew
# weekends and closure days as genuine 0-score days and dragged the trend down.
_cw = df_daily["Chat SLA Wt"].fillna(0)
_ew = df_daily["Email SLA Wt"].fillna(0)
_num = df_daily["Chat SLA"].fillna(0) * _cw + df_daily["Email SLA"].fillna(0) * _ew
_den = _cw.where(df_daily["Chat SLA"].notna(), 0) + _ew.where(df_daily["Email SLA"].notna(), 0)
# Mask the denominator BEFORE dividing. Dividing first and masking after raises
# ZeroDivisionError on pandas 3.x for a day with no contacts at all.
df_daily["Weighted SLA"] = _num / _den.where(_den > 0)

_ok_c = df_daily["Chat SLA"].notna()
_ok_e = df_daily["Email SLA"].notna()
chat_weighted = ((df_daily.loc[_ok_c, "Chat SLA"] * df_daily.loc[_ok_c, "Chat SLA Wt"]).sum()
                 / df_daily.loc[_ok_c, "Chat SLA Wt"].sum()) if df_daily.loc[_ok_c, "Chat SLA Wt"].sum() else None
email_weighted = ((df_daily.loc[_ok_e, "Email SLA"] * df_daily.loc[_ok_e, "Email SLA Wt"]).sum()
                  / df_daily.loc[_ok_e, "Email SLA Wt"].sum()) if df_daily.loc[_ok_e, "Email SLA Wt"].sum() else None
_tot_wt = _den.sum()
weighted_sla = (_num.sum() / _tot_wt) if _tot_wt else None

# Period-level components
_cw_p = chat_sla_p[chat_sla_p["Wait Time"].notna()]
n_chat_total = len(chat_sla_p)
n_chat_answered = len(_cw_p)
n_chat_in_target = int((_cw_p["Wait Time"] <= CHAT_ANSWER_TARGET_SEC).sum())
n_chat_abandoned = int((chat_sla_p["Abandoned After"] > CHAT_ABANDON_AFTER_SEC).sum())

sla_comp_chat_in_target_pct = (n_chat_in_target / n_chat_answered * 100) if n_chat_answered else None
sla_comp_chat_avg_wait_secs = _cw_p["Wait Time"].mean() if n_chat_answered else None
sla_comp_chat_abandon_pct = (n_chat_abandoned / n_chat_total * 100) if n_chat_total else None

n_email_total = len(email_scored_p)
n_email_in_window = len(email_sla_p)
n_email_not_due = n_email_in_window - n_email_total
n_email_in_target = int((email_scored_p["Effective Elapsed"]
                         <= email_scored_p["Effective Target"]).sum()) if n_email_total else 0
sla_comp_email_in_target_pct = (n_email_in_target / n_email_total * 100) if n_email_total else None
sla_comp_email_avg_resp_hrs = email_scored_p["Effective Elapsed"].mean() if n_email_total else None

# Composition of the email SLA population, for the tooltip and the QA panel.
if n_email_total:
    _opened_here = ((email_scored_p["Date/Time Opened"] >= ts_start)
                    & (email_scored_p["Date/Time Opened"] < ts_end))
    n_email_opened_here = int(_opened_here.sum())
    n_email_carried_in = int((~_opened_here).sum())
    n_email_answered_after = int((_opened_here
                                  & (email_scored_p["Response DT"] >= ts_end)).sum())
    n_email_unanswered = int((~email_scored_p["Is Answered"]).sum())
else:
    n_email_opened_here = n_email_carried_in = n_email_answered_after = n_email_unanswered = 0


# =============================================================================
# Header & KPI tiles
# =============================================================================
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} — {end_date:%d %b %Y}")
st.markdown("---")

st.subheader("Core Metrics")
c1, c2, c3, c4 = st.columns(4)
render_custom_metric(c1, "💬 Total Chats", f"{chat_total:,}",
                     "Distinct chat cases first handled in the period. Counted once even if "
                     "the case was routed more than once.",
                     COLOR_GOOD,
                     f"{chat_reopened:,} re-handled" if chat_reopened else "")
render_custom_metric(c2, "✉️ Total Emails", f"{email_total:,}",
                     "Distinct email cases first handled in the period. A case that is "
                     "re-opened and routed again counts ONCE, not once per touch.",
                     COLOR_GOOD,
                     f"{email_reopened:,} re-opened" if email_reopened else "")
render_custom_metric(c3, "⏳ Avg Chat Handle Time", fmt_mmss(chat_aht),
                     "Total handle time per chat case, summed across every touch",
                     COLOR_GOOD,
                     f"{chat_touches:.2f} touches/case" if chat_touches else "")
render_custom_metric(c4, "⏳ Avg Email Handle Time", fmt_mmss(email_aht),
                     "Total handle time per email case, summed across EVERY touch — including "
                     "re-opens handled later and by different agents",
                     COLOR_GOOD,
                     f"{email_touches:.2f} touches/case" if email_touches else "")

# Re-work is invisible when volume counts rows; surfaced explicitly instead.
if email_total and email_reopened:
    _single = email_cases.loc[email_cases["Touches"] == 1, "Total_Secs"].mean()
    _multi = email_cases.loc[email_cases["Touches"] > 1, "Total_Secs"].mean()
    st.caption(
        f"🔁 **{email_reopened:,} of {email_total:,} email cases ({email_reopened/email_total:.0%}) "
        f"were handled more than once**, {email_multi_agent:,} of them by more than one agent. "
        f"Those cases average **{fmt_mmss(_multi)}** of total handle time versus "
        f"**{fmt_mmss(_single)}** for cases resolved in a single touch. "
        f"Counting routing rows instead of cases would report "
        f"{int(email_cases['Touches'].sum()):,} emails instead of {email_total:,}."
    )

st.markdown("---")
st.subheader("Operational Metrics")

_util_tooltip = ("Time handling contacts ∩ time available, divided by availability. "
                 "Shared 'Available_All' time is split between chat and email in "
                 "proportion to the work handled on each — the SAME method for both tiles.")

_n_op_cols = 4 if adherence is not None else 3
_op = st.columns(_n_op_cols)
render_custom_metric(_op[0], "📈 Chat Utilization",
                     f"{chat_util:.1%}" if chat_util is not None else "—",
                     _util_tooltip, get_utilization_color(chat_util),
                     f"{dept_chat_handle/3600:,.1f}h of {dept_chat_avail/3600:,.1f}h")
render_custom_metric(_op[1], "📈 Email Utilization",
                     f"{email_util:.1%}" if email_util is not None else "—",
                     _util_tooltip, get_utilization_color(email_util),
                     f"{dept_email_handle/3600:,.1f}h of {dept_email_avail/3600:,.1f}h")
render_custom_metric(_op[2], "⏱️ Avg Email Resp Time", fmt_hms(avg_resp_secs),
                     "Mean email response time across the period",
                     get_email_resp_time_color(avg_resp_secs),
                     f"across {n_email_total:,} emails" if n_email_total else "")
if adherence is not None:
    render_custom_metric(_op[3], "🗓️ Schedule Adherence", f"{adherence:.1%}",
                         "Available time as a share of rostered shift time (from shifts.csv)",
                         get_utilization_color(adherence),
                         f"{available_secs/3600:,.1f}h of {rostered_secs/3600:,.1f}h rostered")

with st.expander("Per-agent utilisation"):
    if df_util.empty:
        st.info("No presence or handling data in this period.")
    else:
        _show = df_util.drop(columns=["_avail_secs", "_key"]).sort_values("Agent")
        st.dataframe(
            _show.style.format({
                "Chat handled (h)": "{:,.2f}", "Chat available (h)": "{:,.2f}",
                "Email handled (h)": "{:,.2f}", "Email available (h)": "{:,.2f}",
                "Chat util": "{:.1%}", "Email util": "{:.1%}",
            }, na_rep="—"),
            width="stretch",
        )
        if dq.get("shift_agents_unmatched"):
            st.caption(f"⚠️ {dq['shift_agents_unmatched']} agent(s) in shifts.csv had no matching "
                       f"presence record: {', '.join(dq['shift_agents_unmatched_names'])}")

# ---- SLA summary --------------------------------------------------------
st.markdown("---")
st.subheader("🎯 SLA Score Summary")
s1, s2, s3 = st.columns(3)
render_custom_metric(s1, "Chat SLA Score",
                     f"{chat_weighted:.1f}" if chat_weighted is not None else "—",
                     "Daily chat SLA, weighted by daily chat volume", get_sla_score_color(chat_weighted),
                     f"0–100, {SLA_TARGET:.0f} = target")
render_custom_metric(s2, "Email SLA Score",
                     f"{email_weighted:.1f}" if email_weighted is not None else "—",
                     "Daily email SLA, weighted by daily email volume", get_sla_score_color(email_weighted),
                     f"0–100, {SLA_TARGET:.0f} = target")
render_custom_metric(s3, "Weighted SLA Score",
                     f"{weighted_sla:.1f}" if weighted_sla is not None else "—",
                     "Volume-weighted blend of chat and email", get_sla_score_color(weighted_sla),
                     f"0–100, {SLA_TARGET:.0f} = target")

st.markdown("#### Score breakdown")
_bc1, _bc2 = st.columns(2)

with _bc1:
    st.markdown("**💬 Chat SLA components**")
    _cc1, _cc2, _cc3 = st.columns(3)
    _cc1.metric(
        "Answered ≤60s", fmt_pct(sla_comp_chat_in_target_pct),
        # FIX #27: the numbers in the tooltip now reconcile with the percentage.
        help=f"{n_chat_in_target:,} of {n_chat_answered:,} answered chats waited ≤60s "
             f"(weight +{W_CHAT_ANSWER}). Abandoned chats are excluded here and counted "
             f"once, in the abandon term.",
    )
    _cc2.metric(
        "Avg Wait Time", fmt_mmss(sla_comp_chat_avg_wait_secs),
        help=f"No penalty at or below {CHAT_TARGET_WAIT_MIN:.0f}:00. Full penalty "
             f"(−{W_CHAT_WAIT}) at {CHAT_TARGET_WAIT_MIN + CHAT_WAIT_TOLERANCE_MIN:.0f}:00, "
             f"scaled linearly between. Based on {n_chat_answered:,} answered chats.",
    )
    _cc3.metric(
        "Abandon Rate", fmt_pct(sla_comp_chat_abandon_pct),
        help=f"{n_chat_abandoned:,} of {n_chat_total:,} chats abandoned after "
             f"{CHAT_ABANDON_AFTER_SEC:.0f}s (weight −{W_CHAT_ABANDON})",
    )

with _bc2:
    st.markdown("**✉️ Email SLA components**")
    _ec1, _ec2 = st.columns(2)
    _ec1.metric(
        "Within target", fmt_pct(sla_comp_email_in_target_pct),
        help=f"{n_email_in_target:,} of {n_email_total:,} scored emails met their target "
             f"(weight +{W_EMAIL_IN_TARGET}). First-contact cases are measured against their "
             f"entitlement response target ({EMAIL_TARGET_HRS:.0f}hr for most); reopened cases "
             f"against the {EMAIL_REOPENED_TARGET_HRS:.0f}hr resolution target, because their "
             f"elapsed figure spans every response cycle. Population = emails opened OR responded "
             f"to in the period: {n_email_opened_here:,} opened here ({n_email_answered_after:,} "
             f"answered after the period ends) and {n_email_carried_in:,} carried in from before. "
             f"{n_email_not_due:,} unanswered email(s) are not yet past target and are excluded.",
    )
    # Report the actual penalty statistic (p85 of response time in multiples of
    # target) so the number driving the penalty is visible, not just the mean.
    if n_email_total:
        _ratio = (email_scored_p["Effective Elapsed"] / email_scored_p["Effective Target"]).dropna()
        _p85 = float(_ratio.quantile(EMAIL_PENALTY_PERCENTILE)) if len(_ratio) else 1.0
        _pen_frac = min(max(_p85 - 1.0, 0.0) / EMAIL_RESP_TOLERANCE_TARGETS, 1.0) \
            if EMAIL_RESP_TOLERANCE_TARGETS else 0.0
        _p85_hrs = _p85 * float(email_scored_p["Effective Target"].median())
        st.caption(
            f"Penalty statistic: **p{EMAIL_PENALTY_PERCENTILE * 100:.0f} = {_p85:.2f}× target** "
            f"(≈{_p85_hrs:.1f}h at the median {float(email_scored_p['Effective Target'].median()):.0f}h target) — "
            f"{EMAIL_PENALTY_PERCENTILE * 100:.0f}% of emails were answered within that. "
            f"Penalty is at **{_pen_frac:.0%} of maximum**, reaching 100% at "
            f"{1 + EMAIL_RESP_TOLERANCE_TARGETS:.0f}× target."
        )
        if _pen_frac >= 1.0:
            st.caption(
                f"⚠️ The penalty is maxed out, so further deterioration in response times will not "
                f"move the score — only the {sla_comp_email_in_target_pct:.0f}% hit rate will. "
                f"Raise `EMAIL_RESP_TOLERANCE_TARGETS` if you want headroom below this level."
            )

    _ec2.metric(
        "Avg Response Time", fmt_hhmm(sla_comp_email_avg_resp_hrs * 3600
                                      if sla_comp_email_avg_resp_hrs is not None
                                      and not pd.isna(sla_comp_email_avg_resp_hrs) else None),
        help=f"Shown as the mean for reference, but the SLA penalty uses the "
             f"p{EMAIL_PENALTY_PERCENTILE * 100:.0f} of response time in multiples of each case's "
             f"own target — the mean was too easily dominated by a few stuck cases. No penalty at "
             f"or below 1× target; full penalty (−{W_EMAIL_RESP}) at "
             f"{1 + EMAIL_RESP_TOLERANCE_TARGETS:.0f}× target, scaled linearly between. "
             f"Across {n_email_total:,} emails.",
    )

# ---------------------------------------------------------------------------
# First contact vs reopened.
#
# These are two different measurements, not one metric split in half:
#   - first contact -> 'Elapsed' IS the first-response time (matches time-to-
#     first-close within 0.02h), judged against the entitlement response target.
#   - reopened      -> 'Elapsed' tracks the LAST close, so it is a RESOLUTION
#     time spanning every cycle plus the customer's own reply time, judged
#     against EMAIL_REOPENED_TARGET_HRS.
# The headline blends the two scores by case volume.
# ---------------------------------------------------------------------------
if n_email_total and "Reopened" in email_scored_p.columns:
    (_sla_first, _n_first), (_sla_reop, _n_reop) = email_sla_parts(email_scored_p)
    _first = email_scored_p[~email_scored_p["Reopened"].astype(bool)]
    _reop = email_scored_p[email_scored_p["Reopened"].astype(bool)]

    def _row(label, g, score, measure, tgt_label):
        if not len(g):
            return None
        return {
            "Population": label,
            "Measures": measure,
            "Target": tgt_label,
            "Cases": len(g),
            "Median (h)": float(g["Effective Elapsed"].median()),
            "% within target": float((g["Effective Elapsed"] <= g["Effective Target"]).mean() * 100),
            "SLA": score,
        }

    _rows = [r for r in (
        _row("First contact", _first, _sla_first, "First response time",
             f"{EMAIL_TARGET_HRS:.0f}h (entitlement)"),
        _row("Reopened", _reop, _sla_reop, "Resolution time",
             f"{EMAIL_REOPENED_TARGET_HRS:.0f}h"),
    ) if r]

    if _rows:
        _split = pd.DataFrame(_rows)
        _hdr = ("📬 First contact vs reopened — blended into the headline by volume"
                if _n_reop else "📬 First contact vs reopened")
        with st.expander(_hdr, expanded=bool(_n_reop)):
            st.dataframe(
                _split.style.format({"Cases": "{:,}", "Median (h)": "{:.2f}",
                                     "% within target": "{:.1f}", "SLA": "{:.1f}"}, na_rep="—"),
                width="stretch", hide_index=True,
            )
            if _n_reop:
                _share = _n_reop / (_n_first + _n_reop) * 100
                st.markdown(
                    f"""
The two rows measure **different things**, so they carry different targets.

For a case routed once, `Elapsed Time (Hours)` matches time-to-first-close almost exactly — it is a
genuine **first-response** time. For a reopened case it tracks the **last** close instead, so it
spans every response cycle *and* the customer's own reply time in between. Scoring that against a
1-hour response target would measure something the team cannot control, so reopened cases are
judged against a **{EMAIL_REOPENED_TARGET_HRS:.0f}-hour resolution** target instead.

Reopened cases are **{_share:.1f}%** of scored volume, so the headline Email SLA of
**{email_weighted:.1f}** is mostly the first-contact figure.

*This is a workaround.* The underlying cause is that the response milestone is set to
**No Recurrence** in Salesforce, so one milestone record covers the whole case life. Setting it to
**Independent** creates a fresh record per reopen, at which point reopened cases can be measured on
response time like everything else and this split becomes unnecessary.
"""
                )
            if dq.get("email_no_routing_record"):
                st.caption(
                    f"{dq['email_no_routing_record']:,} case(s) have no routing record in "
                    f"report_items.csv and are counted as first contact — a case never routed "
                    f"cannot have been re-routed."
                )

# ---------------------------------------------------------------------------
# Email SLA split by Case Origin.
#
# Different origins can behave like completely different services — a combined
# score averages them into a number that describes neither. Worth checking
# before reading the headline figure.
# ---------------------------------------------------------------------------
if n_email_total and "Case Origin" in email_scored_p.columns and email_scored_p["Case Origin"].nunique() > 1:
    _orig_rows = []
    for _o, _g in email_scored_p.groupby("Case Origin"):
        _orig_rows.append({
            "Case Origin": str(_o),
            "Cases": len(_g),
            "Median (h)": float(_g["Effective Elapsed"].median()),
            "Mean (h)": float(_g["Effective Elapsed"].mean()),
            "p90 (h)": float(_g["Effective Elapsed"].quantile(0.90)),
            "% within target": float((_g["Effective Elapsed"] <= _g["Target Hrs"]).mean() * 100),
            "SLA": email_sla_from_slice(_g),
        })
    _orig = pd.DataFrame(_orig_rows).sort_values("Cases", ascending=False)

    _spread = _orig["SLA"].max() - _orig["SLA"].min()
    with st.expander(f"📬 Email SLA by case origin"
                     f"{'  ⚠️ origins differ by ' + f'{_spread:.0f}' + ' points' if _spread >= 20 else ''}",
                     expanded=bool(_spread >= 40)):
        st.dataframe(
            _orig.style.format({"Cases": "{:,}", "Median (h)": "{:.2f}", "Mean (h)": "{:.2f}",
                                "p90 (h)": "{:.2f}", "% within target": "{:.1f}", "SLA": "{:.1f}"},
                               na_rep="—"),
            width="stretch", hide_index=True,
        )
        if _spread >= 20:
            _best = _orig.loc[_orig["SLA"].idxmax()]
            _worst = _orig.loc[_orig["SLA"].idxmin()]
            st.warning(
                f"**These origins are not one service.** “{_best['Case Origin']}” scores "
                f"{_best['SLA']:.1f} on {_best['Cases']:,} cases while “{_worst['Case Origin']}” "
                f"scores {_worst['SLA']:.1f} on {_worst['Cases']:,} — median "
                f"{_best['Median (h)']:.2f}h against {_worst['Median (h)']:.2f}h. The combined "
                f"figure of {email_weighted:.1f} describes neither. Consider tracking them "
                f"separately, or setting different targets per origin."
            )

# FIX #14: surface the two populations so a divergence is visible.
_wt_chat, _wt_email = int(df_daily["Chat SLA Wt"].sum()), int(df_daily["Email SLA Wt"].sum())

# ---------------------------------------------------------------------------
# Volume vs SLA-population reconciliation.
#
# The volume tiles come from report_items.csv (cases an agent handled) while the
# SLA comes from chat.csv / email.csv. Neither file is a subset of the other, so
# the two counts legitimately differ — but a LARGE positive gap on email means
# cases were routed and then never appeared in email.csv, which in a
# resolved-only export means they are still open. Those cases are by definition
# past target, so excluding them flatters the score. This panel makes the size
# of that blind spot explicit instead of leaving it to be discovered.
# ---------------------------------------------------------------------------
with st.expander("🧮 Volume vs SLA population — why these two numbers differ"):
    _rec = pd.DataFrame([
        {"Channel": "Chat",
         "Cases handled (tile)": chat_total,
         "Contacts scored for SLA": _wt_chat,
         "Difference": _wt_chat - chat_total},
        {"Channel": "Email",
         "Cases handled (tile)": email_total,
         "Contacts scored for SLA": _wt_email,
         "Difference": _wt_email - email_total},
    ])
    st.dataframe(_rec.style.format({c: "{:,}" for c in _rec.columns if c != "Channel"}),
                 width="stretch", hide_index=True)

    st.markdown(
        f"""
The two columns count different things and **neither file is a subset of the other**:

- **Cases handled** — distinct cases in `report_items.csv` first picked up by an agent in this
  period. A case appears here only if it was routed through Omni-Channel.
- **Contacts scored** — rows in `chat.csv` / `email.csv` for this period. For email that is
  {n_email_opened_here:,} opened in the period plus {n_email_carried_in:,} opened earlier but
  answered inside it.

Nothing is being silently discarded: every one of the {_wt_email:,} emails in scope was scored.
"""
    )

    # ----- Exact reconciliation, when email.csv carries the Case Number -----
    if EMAIL_CASE_KEY in email_sla_df.columns and HAS_CASE_KEY:
        _routed = set(
            df_cases.loc[df_cases["Service Channel: Developer Name"] == EMAIL_DEVNAME, CASE_KEY]
            .astype(str).str.strip()
        ) if CASE_KEY in df_cases.columns else set()
        _routed_period = set(
            cases_period.loc[cases_period["Service Channel: Developer Name"] == EMAIL_DEVNAME, CASE_KEY]
            .astype(str).str.strip()
        ) if CASE_KEY in cases_period.columns else set()
        _in_export = set(email_sla_df[EMAIL_CASE_KEY].astype(str).str.strip())

        _matched = _routed_period & _in_export
        _missing = _routed_period - _in_export
        _opened_here_keys = set(
            email_sla_df.loc[(email_sla_df["Date/Time Opened"] >= ts_start)
                             & (email_sla_df["Date/Time Opened"] < ts_end),
                             EMAIL_CASE_KEY].astype(str).str.strip())
        _never_routed = _opened_here_keys - _routed

        st.markdown("**Exact reconciliation** (joined on Case Number ↔ Work Item: Name)")
        st.dataframe(pd.DataFrame([
            {"Measure": "Cases routed to an agent this period", "Cases": len(_routed_period)},
            {"Measure": "…also present in the email export", "Cases": len(_matched)},
            {"Measure": "…MISSING from the email export", "Cases": len(_missing)},
            {"Measure": "Cases in the export, opened this period, never routed",
             "Cases": len(_never_routed)},
        ]).style.format({"Cases": "{:,}"}), width="stretch", hide_index=True)

        if _routed_period and len(_missing) > max(10, 0.02 * len(_routed_period)):
            st.warning(
                f"**{len(_missing):,} cases ({len(_missing)/len(_routed_period):.1%}) were routed to "
                f"an agent this period but do not appear in the email export at all.**\n\n"
                f"The export is filtered to closed cases, so these are cases still open when it ran. "
                f"Each is already past its response target, so excluding them makes the Email SLA "
                f"look better than reality. Re-export **without the status filter** to close this gap "
                f"— the dashboard already knows how to age an open case and count it as a breach "
                f"once it passes target."
            )
        if _never_routed:
            st.caption(
                f"The {len(_never_routed):,} never-routed cases are the opposite effect — resolved "
                f"without passing through Omni-Channel, so they reach the email export but never "
                f"reach report_items.csv. They partly offset the figure above, which is why the raw "
                f"difference between the two columns understates both flows."
            )
    else:
        _email_gap = email_total - n_email_opened_here
        if email_total and _email_gap > max(20, 0.05 * email_total):
            st.warning(
                f"**{_email_gap:,} email cases were handled in this period but never appear in "
                f"email.csv** ({_email_gap / email_total:.0%} of the tile). Add a **Case Number** "
                f"column to the email export to reconcile these exactly instead of by difference."
            )

# FIX #22: the daily component columns used to be computed and never shown.
with st.expander("Daily SLA detail"):
    # Both populations are shown side by side: "SLA Wt" is what the score was
    # computed on, "Vol" is the report_items work-item count for the same day.
    _detail = df_daily[[
        "Date", "Chat SLA", "Chat SLA Wt", "Chat Vol", "Chat ≤60s %", "Chat Avg Wait (s)",
        "Chat Abandon %", "Email SLA", "Email SLA Wt", "Email Vol", "Email Not Due",
        "Email in target %", "Email Avg Resp (h)", "Weighted SLA",
    ]].copy()
    _detail["Date"] = _detail["Date"].dt.strftime("%a %d %b")
    _detail = _detail.rename(columns={
        "Chat SLA Wt": "Chat scored", "Chat Vol": "Chat items",
        "Email SLA Wt": "Email scored", "Email Vol": "Email items",
        "Email Not Due": "Email not due",
    })
    st.dataframe(
        _detail.style.format({
            "Chat SLA": "{:.1f}", "Email SLA": "{:.1f}", "Weighted SLA": "{:.1f}",
            "Chat ≤60s %": "{:.1f}", "Chat Abandon %": "{:.1f}",
            "Email in target %": "{:.1f}", "Chat Avg Wait (s)": "{:.0f}", "Email Avg Resp (h)": "{:.2f}",
        }, na_rep="—"),
        width="stretch",
    )
    st.caption(
        "**scored** = contacts the SLA was calculated on (chats opened that day; emails "
        "opened *or* responded to that day, excluding any not yet past target). "
        "**items** = work items from report_items.csv started that day. "
        "The score is weighted by *scored*; *items* is shown so you can see when the two diverge."
    )


# =============================================================================
# Weighted SLA trend
# =============================================================================
st.markdown("---")
st.subheader("Weighted SLA Trend")

_trend_df = df_daily[["Date", "Chat SLA", "Email SLA", "Weighted SLA", "Chat SLA Wt", "Email SLA Wt"]].copy()

# FIX #25: symmetric padding. datetime.max.time() overshot the right edge by
# ~36 hours because it is 23:59:59.999999, not midnight.
x_min = pd.Timestamp(start_date) - pd.Timedelta(hours=12)
x_max = pd.Timestamp(end_date) + pd.Timedelta(hours=12)

_plot = _trend_df.dropna(subset=["Weighted SLA"])
if _plot.empty:
    st.info("No contacts in the selected period, so there is no SLA to plot.")
else:
    trend_chart = (
        alt.Chart(_plot)
        .mark_line(point=True, color="#2F80ED")
        .encode(
            x=alt.X("Date:T", title="Date",
                    axis=alt.Axis(format="%d %b", labelAngle=-45, values=_plot["Date"].tolist()),
                    scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("Weighted SLA:Q", title="Weighted SLA Score", scale=alt.Scale(domain=[0, 105])),
            tooltip=[
                alt.Tooltip("Date:T", format="%d %b"),
                alt.Tooltip("Weighted SLA:Q", format=".1f"),
                alt.Tooltip("Chat SLA:Q", format=".1f"),
                alt.Tooltip("Email SLA:Q", format=".1f"),
                alt.Tooltip("Chat SLA Wt:Q", title="Chats", format=",.0f"),
                alt.Tooltip("Email SLA Wt:Q", title="Emails", format=",.0f"),
            ],
        )
    )
    labels = trend_chart.mark_text(dy=-12, color="#2F80ED").encode(
        text=alt.Text("Weighted SLA:Q", format=".1f"))
    rule = alt.Chart(pd.DataFrame({"y": [SLA_TARGET]})).mark_rule(
        color="red", strokeDash=[5, 5]).encode(y="y:Q")
    rule_lb = alt.Chart(pd.DataFrame({"y": [SLA_TARGET]})).mark_text(
        align="left", color="red", dy=-8).encode(
        y="y:Q", text=alt.value(f"Target: {SLA_TARGET:.0f}"))

    altair_chart((trend_chart + labels + rule + rule_lb).properties(height=350))

    _gaps = int(_trend_df["Weighted SLA"].isna().sum())
    if _gaps:
        st.caption(f"{_gaps} day(s) in this period had no contacts and are shown as gaps, "
                   f"not as zero scores.")


# =============================================================================
# Contact volume heatmap
# =============================================================================
st.markdown("---")
st.subheader("📅 Contact Volume Heatmap")

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_ORDER = [f"{h:02d}:00" for h in range(24)]

_hm_parts = []
for _src, _label in ((chat_sla_p, "Chat"), (email_opened_p, "Email")):
    if len(_src):
        _p = _src[["Date/Time Opened"]].copy()
        _p["Channel"] = _label
        _hm_parts.append(_p)

_hm = pd.concat(_hm_parts, ignore_index=True) if _hm_parts else pd.DataFrame(
    columns=["Date/Time Opened", "Channel"])
if not _hm.empty:
    _hm["Date/Time Opened"] = pd.to_datetime(_hm["Date/Time Opened"], errors="coerce")
    _hm = _hm.dropna(subset=["Date/Time Opened"])
    _hm["DayOfWeek"] = _hm["Date/Time Opened"].dt.day_name()
    _hm["Hour"] = _hm["Date/Time Opened"].dt.strftime("%H:00")


def _build_heatmap(df_src: pd.DataFrame, title: str, color_scheme: str = "blues"):
    from itertools import product

    # FIX: day ordering is now derived per tab, so the chat tab no longer shows
    # empty weekend rows just because email had weekend traffic.
    days_present = [d for d in DOW_ORDER if d in set(df_src["DayOfWeek"].unique())]
    grid = df_src.groupby(["DayOfWeek", "Hour"]).size().reset_index(name="Volume")
    all_combos = pd.DataFrame(list(product(days_present, HOUR_ORDER)), columns=["DayOfWeek", "Hour"])
    grid = all_combos.merge(grid, on=["DayOfWeek", "Hour"], how="left").fillna({"Volume": 0})
    grid["Volume"] = grid["Volume"].astype(int)

    return (
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("Hour:O", title="Hour of Day", sort=HOUR_ORDER,
                    axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
            y=alt.Y("DayOfWeek:O", title=None, sort=days_present),
            color=alt.Color("Volume:Q", title="Contacts",
                            scale=alt.Scale(scheme=color_scheme, zero=True),
                            legend=alt.Legend(orient="right")),
            tooltip=[alt.Tooltip("DayOfWeek:O", title="Day"),
                     alt.Tooltip("Hour:O", title="Hour"),
                     alt.Tooltip("Volume:Q", title="Contacts")],
        )
        .properties(height=220, title=title)
    )


_tab_all, _tab_chat, _tab_email = st.tabs(["Combined", "Chat only", "Email only"])
for _tab, _sub, _title, _scheme in (
    (_tab_all, _hm, "All Contacts (Chat + Email)", "blues"),
    (_tab_chat, _hm[_hm["Channel"] == "Chat"] if not _hm.empty else _hm,
     "Chat Contacts (incl. missed)", "greens"),
    (_tab_email, _hm[_hm["Channel"] == "Email"] if not _hm.empty else _hm,
     "Email Contacts", "oranges"),
):
    with _tab:
        if _sub.empty:
            st.info("No data in this period.")
        else:
            altair_chart(_build_heatmap(_sub, _title, _scheme))


# =============================================================================
# Customer feedback (CSAT / NPS / FCR)
# =============================================================================
def _leading_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None


def _bool_yes_no(x) -> Optional[bool]:
    """Parse a yes/no response.

    Tolerates decorated answers ("Yes - it was resolved", "No, I had to call
    back"), which an exact-match-only version silently turned into None and
    therefore dropped from the FCR denominator.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    # Leading token only: "not resolved" must NOT read as "no" (there is no word
    # boundary between "no" and "t"), while "No, I had to call back" must.
    m = re.match(r"^(yes|no|y|n|true|false)\b", s)
    if m:
        return m.group(1) in ("yes", "y", "true")
    return None


def _nps_from_scores(series: pd.Series) -> Optional[float]:
    r = pd.to_numeric(series, errors="coerce").dropna()
    if r.empty:
        return None
    return float((r >= 9).mean() * 100 - (r <= 6).mean() * 100)


def _nps_guarded(series: pd.Series) -> Optional[float]:
    """FIX #12: NPS from a handful of responses is ±100 noise. Suppress it."""
    r = pd.to_numeric(series, errors="coerce").dropna()
    if len(r) < NPS_MIN_SAMPLE:
        return None
    return _nps_from_scores(r)


st.markdown("---")

if survey_q is None:
    st.info("Survey data not found (survey.csv). Add it next to the app to see CSAT/NPS/FCR.")
else:
    st.subheader("Customer Feedback")

    _qtitle = survey_q["Survey Question: Question Title"].astype(str).str.lower()

    # FIX #10: first-match-wins classification from explicit config.
    _metric = pd.Series([None] * len(survey_q), index=survey_q.index, dtype="object")
    for _name, _pat in SURVEY_QUESTION_PATTERNS:
        _hit = _qtitle.str.contains(_pat, na=False, regex=True) & _metric.isna()
        _metric[_hit] = _name
    survey_q["_metric"] = _metric

    _unmapped = sorted(survey_q.loc[survey_q["_metric"].isna(),
                                    "Survey Question: Question Title"].astype(str).unique())

    survey_q["NPS_raw"] = survey_q["Response"].where(survey_q["_metric"] == "NPS").apply(_leading_int)
    survey_q["CSAT_raw"] = survey_q["Response"].where(survey_q["_metric"] == "CSAT").apply(_leading_int)
    survey_q["FCR_bool"] = survey_q["Response"].where(survey_q["_metric"] == "FCR").apply(_bool_yes_no)

    # FIX #4: Channel is now actually used (per-channel breakdown below).
    # Fall back to the question title, since channel-specific wordings such as
    # "Was the issue reported in the chat resolved by the agent?" name the
    # channel even when the survey name does not.
    _ch_survey = (survey_q["Survey Question: Survey"].astype(str)
                  .str.extract(r"(?i)\b(Email|Chat)\b", expand=False).str.capitalize())
    _ch_title = (survey_q["Survey Question: Question Title"].astype(str)
                 .str.extract(r"(?i)\b(Email|Chat)\b", expand=False).str.capitalize())
    survey_q["Channel"] = _ch_survey.fillna(_ch_title).fillna("Other")

    # FIX #11: validate the declared scales instead of assuming them.
    _scale_warnings = []
    _csat_vals = pd.to_numeric(survey_q["CSAT_raw"], errors="coerce").dropna()
    if not _csat_vals.empty and (_csat_vals.min() < CSAT_SCALE_MIN or _csat_vals.max() > CSAT_SCALE_MAX):
        _scale_warnings.append(
            f"CSAT responses range {_csat_vals.min():.0f}–{_csat_vals.max():.0f} but the "
            f"configured scale is {CSAT_SCALE_MIN}–{CSAT_SCALE_MAX}. Update CSAT_SCALE_MIN/MAX "
            f"— percentages are wrong until you do."
        )
    _nps_vals = pd.to_numeric(survey_q["NPS_raw"], errors="coerce").dropna()
    if not _nps_vals.empty and (_nps_vals.min() < NPS_SCALE_MIN or _nps_vals.max() > NPS_SCALE_MAX):
        _scale_warnings.append(
            f"NPS responses range {_nps_vals.min():.0f}–{_nps_vals.max():.0f} but the "
            f"configured scale is {NPS_SCALE_MIN}–{NPS_SCALE_MAX}."
        )

    # FIX #9: mean, not max. Taking max scored a respondent who answered 2 and 5
    # as a 5.
    def _mean_or_none(s):
        v = pd.to_numeric(s, errors="coerce").dropna()
        return float(v.mean()) if len(v) else None

    def _first_or_none(s):
        v = s.dropna()
        return v.iloc[0] if len(v) else None

    def _first_channel(s):
        """Prefer a real channel over the 'Other' placeholder.

        A respondent's CSAT question may name no channel while their FCR
        question does; taking the literal first value would label them 'Other'.
        """
        v = s.dropna()
        real = v[v != "Other"]
        if len(real):
            return real.iloc[0]
        return v.iloc[0] if len(v) else "Other"

    survey = survey_q.groupby("Survey Taker: ID", as_index=False).agg(
        **{
            "Survey Date": ("Survey Taker: Created Date", "min"),
            "Channel": ("Channel", _first_channel),
            "NPS_raw": ("NPS_raw", _mean_or_none),
            "CSAT_raw": ("CSAT_raw", _mean_or_none),
            "FCR_bool": ("FCR_bool", _first_or_none),
        }
    )

    _span = max(CSAT_SCALE_MAX - CSAT_SCALE_MIN, 1)
    survey["CSAT%"] = (survey["CSAT_raw"] - CSAT_SCALE_MIN) / _span * 100.0
    survey["Survey Date"] = pd.to_datetime(survey["Survey Date"], errors="coerce")

    _undated = int(survey["Survey Date"].isna().sum())
    survey_period = survey[
        (survey["Survey Date"] >= ts_start) & (survey["Survey Date"] < ts_end)
    ].copy()

    for _w in _scale_warnings:
        st.warning(_w)
    if _unmapped:
        st.warning(
            "These survey questions matched no CSAT/NPS/FCR pattern and are excluded — "
            "add them to SURVEY_QUESTION_PATTERNS if they should count: "
            + "; ".join(f"“{q}”" for q in _unmapped[:6])
            + (f" (+{len(_unmapped) - 6} more)" if len(_unmapped) > 6 else "")
        )
    if _undated:
        st.caption(f"{_undated} survey(s) had an unparseable date and are excluded.")

    # ---------------------------------------------------------------------
    # Question-mapping diagnostic.
    # A silently mis-mapped question is the easiest way for CSAT/NPS/FCR to go
    # quietly wrong, and the "unmapped" warning above only catches questions
    # that match NOTHING — it cannot catch a question mapped to the WRONG
    # metric. This table shows every distinct question title, what it mapped
    # to, and how many answers were parsed, so both failure modes are visible.
    # ---------------------------------------------------------------------
    with st.expander("🔍 Survey question mapping — check this if CSAT/NPS/FCR look wrong"):
        _rows = []
        for _title, _grp in survey_q.groupby("Survey Question: Question Title", dropna=False):
            _m = _grp["_metric"].dropna()
            _m = _m.iloc[0] if len(_m) else None
            if _m == "NPS":
                _parsed = int(_grp["NPS_raw"].notna().sum())
            elif _m == "CSAT":
                _parsed = int(_grp["CSAT_raw"].notna().sum())
            elif _m == "FCR":
                _parsed = int(_grp["FCR_bool"].notna().sum())
            else:
                _parsed = 0
            _sample = _grp["Response"].dropna()
            _rows.append({
                "Question": str(_title),
                "Mapped to": _m if _m else "— excluded —",
                "Rows": len(_grp),
                "Answers parsed": _parsed,
                "Example response": str(_sample.iloc[0]) if len(_sample) else "",
            })
        _map_df = pd.DataFrame(_rows).sort_values(["Mapped to", "Rows"], ascending=[True, False])
        st.dataframe(_map_df, width="stretch", hide_index=True)

        # A question that mapped to a metric but parsed almost nothing means the
        # RESPONSE format is unexpected, not the question title.
        _bad = _map_df[(_map_df["Mapped to"] != "— excluded —")
                       & (_map_df["Answers parsed"] < _map_df["Rows"] * 0.5)]
        if not _bad.empty:
            st.warning(
                "These questions were mapped but fewer than half their responses could be "
                "parsed — check the response format (FCR expects Yes/No; CSAT and NPS expect "
                "a leading number): "
                + "; ".join(f"“{r['Question']}” ({r['Answers parsed']}/{r['Rows']} parsed)"
                            for _, r in _bad.iterrows())
            )

    if survey_period.empty:
        st.info("No survey responses in the selected date range.")
    else:
        total_surveys = len(survey_period)
        csat_overall = survey_period["CSAT%"].mean() if survey_period["CSAT%"].notna().any() else None
        nps_overall = (_nps_from_scores(survey_period["NPS_raw"])
                       if survey_period["NPS_raw"].notna().any() else None)

        # FIX #2: denominator is respondents who ANSWERED the FCR question.
        _fcr_answered = survey_period["FCR_bool"].dropna()
        fcr_overall = float((_fcr_answered == True).mean() * 100) if len(_fcr_answered) else None

        n_csat = int(survey_period["CSAT%"].notna().sum())
        n_nps = int(survey_period["NPS_raw"].notna().sum())
        n_fcr = int(len(_fcr_answered))

        k1, k2, k3, k4 = st.columns(4)
        render_custom_metric(k1, "🗳️ Surveys", f"{total_surveys:,}",
                             "Distinct survey respondents in range", COLOR_GOOD)
        render_custom_metric(k2, "😊 CSAT (avg %)",
                             fmt_pct(csat_overall),
                             f"Mean CSAT normalised from the {CSAT_SCALE_MIN}–{CSAT_SCALE_MAX} scale to 0–100%",
                             get_csat_color_pct(csat_overall), f"from {n_csat:,} responses")
        render_custom_metric(k3, "⭐ NPS",
                             f"{nps_overall:.1f}" if nps_overall is not None else "—",
                             "NPS: %Promoters (9–10) − %Detractors (0–6)",
                             get_nps_color(nps_overall), f"from {n_nps:,} responses")
        render_custom_metric(k4, "🎯 FCR", fmt_pct(fcr_overall),
                             "First Contact Resolution, over respondents who answered the FCR question",
                             get_fcr_color_pct(fcr_overall), f"from {n_fcr:,} responses")

        # ---- FIX #4: per-channel breakdown (Channel was previously unused) ----
        _by_ch = []
        for _ch, _grp in survey_period.groupby("Channel"):
            _f = _grp["FCR_bool"].dropna()
            _by_ch.append({
                "Channel": _ch,
                "Surveys": len(_grp),
                "CSAT %": _grp["CSAT%"].mean() if _grp["CSAT%"].notna().any() else None,
                "NPS": _nps_from_scores(_grp["NPS_raw"]) if _grp["NPS_raw"].notna().any() else None,
                "FCR %": float((_f == True).mean() * 100) if len(_f) else None,
            })
        if len(_by_ch) > 1:
            st.markdown("**Feedback by channel**")
            st.dataframe(
                pd.DataFrame(_by_ch).sort_values("Surveys", ascending=False)
                .style.format({"CSAT %": "{:.1f}", "NPS": "{:.1f}", "FCR %": "{:.1f}",
                               "Surveys": "{:,}"}, na_rep="—"),
                width="stretch",
            )

        # ---- CSAT & NPS trend ----
        daily_survey = (
            survey_period
            .assign(Date=survey_period["Survey Date"].dt.normalize())
            .groupby("Date", as_index=False)
            .agg(CSAT_pct=("CSAT%", "mean"),
                 NPS=("NPS_raw", _nps_guarded),
                 Surveys=("Survey Date", "size"))
            .sort_values("Date")
        )

        if not daily_survey.empty:
            st.subheader("CSAT & NPS Trend Analysis")

            base = alt.Chart(daily_survey)
            x_enc = alt.X("yearmonthdate(Date):T", title="Date",
                          axis=alt.Axis(format="%d %b", labelAngle=45))
            hover = alt.selection_point(on="mouseover", fields=["Date"], nearest=True, empty=False)

            csat_line = base.mark_line(strokeWidth=3, color="#2563eb").encode(
                x=x_enc,
                y=alt.Y("CSAT_pct:Q", title="CSAT (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("Date:T", format="%d %b"),
                         alt.Tooltip("CSAT_pct:Q", title="CSAT (%)", format=".1f"),
                         alt.Tooltip("Surveys:Q", title="Surveys", format=".0f")],
            )
            csat_points = base.mark_point(filled=True, color="#2563eb").encode(
                x=x_enc,
                y=alt.Y("CSAT_pct:Q", axis=None, scale=alt.Scale(domain=[0, 100])),
                size=alt.condition(hover, alt.value(120), alt.value(60)),
                tooltip=[alt.Tooltip("Date:T", format="%d %b"),
                         alt.Tooltip("CSAT_pct:Q", title="CSAT (%)", format=".1f")],
            ).add_params(hover)
            nps_line = base.mark_line(strokeWidth=3, color="#dc2626", strokeDash=[5, 5]).encode(
                x=x_enc,
                y=alt.Y("NPS:Q", title="NPS Score", axis=alt.Axis(orient="right"),
                        scale=alt.Scale(domain=[-100, 100])),
                tooltip=[alt.Tooltip("Date:T", format="%d %b"),
                         alt.Tooltip("NPS:Q", title="NPS Score", format=".0f"),
                         alt.Tooltip("Surveys:Q", title="Surveys", format=".0f")],
            )
            nps_points = base.mark_point(filled=True, color="#dc2626", shape="square").encode(
                x=x_enc,
                y=alt.Y("NPS:Q", axis=None, scale=alt.Scale(domain=[-100, 100])),
                size=alt.condition(hover, alt.value(120), alt.value(60)),
                tooltip=[alt.Tooltip("Date:T", format="%d %b"),
                         alt.Tooltip("NPS:Q", title="NPS Score", format=".0f")],
            ).add_params(hover)
            nps_zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
                strokeDash=[6, 4], color="#9ca3af", opacity=0.6
            ).encode(y=alt.Y("zero:Q", axis=None, scale=alt.Scale(domain=[-100, 100])))

            altair_chart(
                alt.layer(csat_line, nps_line, csat_points, nps_points, nps_zero)
                .resolve_scale(y="independent")
                .properties(height=400)
                .configure_axis(grid=True, gridColor="#e5e7eb", gridDash=[2, 3])
                .configure_legend(orient="top-right")
                .interactive()
            )

            _suppressed = int(daily_survey["NPS"].isna().sum())
            if _suppressed:
                st.caption(f"NPS is hidden on {_suppressed} day(s) with fewer than "
                           f"{NPS_MIN_SAMPLE} responses, where the figure would be noise.")


# =============================================================================
# Chats by country
# =============================================================================
st.markdown("---")
st.subheader("🌍 Chats by Country (volume)")

DEVNAME_COL = "Chat Button: Developer Name"

COUNTRY_ALIASES = {
    "Angola": ["angola", "ao"],
    "Botswana": ["botswana", "bw"],
    "Cameroon": ["cameroon", "cm"],
    "Congo (DRC)": ["congo drc", "drc", "dr congo", "democratic republic of congo", "rdc"],
    "Côte d'Ivoire": ["cote d'ivoire", "côte d'ivoire", "ivory coast", "ci"],
    "Ghana": ["ghana", "gh"],
    "Kenya": ["kenya", "ke"],
    "Malawi": ["malawi", "mw"],
    "Mozambique": ["mozambique", "mocambique", "moçambique", "mz"],
    "Nigeria": ["nigeria", "ng"],
    "Rwanda": ["rwanda", "rw"],
    "Sierra Leone": ["sierra leone", "sl"],
    "South Africa": ["south africa", "sa"],
    "Tanzania": ["tanzania", "tz"],
    "Uganda": ["uganda", "ug"],
    "Zambia": ["zambia", "zm"],
    "Zimbabwe": ["zimbabwe", "zw"],
    "Eswatini": ["eswatini", "swaziland", "sz"],
    "Lesotho": ["lesotho", "ls"],
    "São Tomé and Príncipe": ["sao tome and principe", "sao tome", "são tomé", "stp"],
    "Gabon": ["gabon", "ga"],
    "Liberia": ["liberia", "lr"],
    "Senegal": ["senegal", "sn"],
    "Burkina Faso": ["burkina faso", "burkina", "bf"],
    "Mali": ["mali", "ml"],
    "Benin": ["benin", "bj"],
    "Niger": ["niger", "ne"],
    "Guinea": ["guinea", "gn"],
    "Guinea-Bissau": ["guinea bissau", "gw"],
    "The Gambia": ["gambia", "gm"],
    "Namibia": ["namibia", "na"],
    "Ethiopia": ["ethiopia", "et"],
    "Somalia": ["somalia", "so"],
    "Morocco": ["morocco", "ma"],
    "Tunisia": ["tunisia", "tn"],
    "Algeria": ["algeria", "dz"],
}

# FIX #24: tokens that look like country codes but usually are not.
CODE_STOPWORDS = {"na", "en", "ci", "so", "ne", "ma", "no", "id", "ok"}
NAME_STOPWORDS = {
    "website", "web", "center", "centre", "support", "service", "services",
    "customer", "cs", "live", "chat", "button", "btn", "queue", "portal",
    "mobile", "app", "online", "site", "vip", "en", "pt", "fr", "eng", "por",
}


def _country_from_devname(devname: str) -> str:
    """
    Map a chat-button developer name to a country.

    FIX #24: bare two-letter codes must be a standalone token AND must not be a
    known false friend (a button containing "na" for 'not applicable' used to be
    reported as Namibia). Multi-word aliases are matched first, and longer
    aliases before shorter ones, so matching no longer depends on dict order.
    """
    if not isinstance(devname, str) or not devname.strip():
        return "Unknown"

    s_norm = re.sub(r"[\-_.:/|]+", " ", devname.strip()).lower()
    tokens = [t for t in re.split(r"\s+", s_norm) if t and t not in NAME_STOPWORDS]
    s_clean = " ".join(tokens)
    token_set = set(tokens)

    candidates = []
    for canon, pats in COUNTRY_ALIASES.items():
        for alias in pats:
            candidates.append((len(alias), canon, alias))
    candidates.sort(reverse=True)  # longest alias first

    for _, canon, alias in candidates:
        if len(alias) <= 2:
            if alias in CODE_STOPWORDS:
                continue
            if alias in token_set:
                return canon
        else:
            if re.search(r"(?:^|\b)" + re.escape(alias) + r"(?:\b|$)", s_clean):
                return canon
    return "Unknown"


if DEVNAME_COL not in chat_sla_p.columns:
    st.info("Column 'Chat Button: Developer Name' not found in chat.csv.")
elif chat_sla_p.empty:
    st.info("No chats in the selected period.")
else:
    dfc = chat_sla_p.copy()
    # FIX #23: no .str.title() — it turned "Congo (DRC)" into "Congo (Drc)" and
    # "São Tomé and Príncipe" into "São Tomé And Príncipe".
    dfc["Country"] = dfc[DEVNAME_COL].apply(_country_from_devname)

    counts = (dfc.groupby("Country", dropna=False).size().reset_index(name="Chats")
              .sort_values("Chats", ascending=False).reset_index(drop=True))

    if counts["Chats"].sum() == 0:
        st.info("No chats in the selected period.")
    else:
        top_n = st.sidebar.slider("Pie chart: top countries", 3, 12, 8, 1)
        min_share = st.sidebar.slider("Label slices ≥ this share", 0.0, 0.1, 0.02, 0.01)

        grand_total = int(counts["Chats"].sum())
        plot_counts = counts.copy()
        if len(plot_counts) > top_n:
            top = plot_counts.head(top_n)
            others = int(plot_counts["Chats"].iloc[top_n:].sum())
            plot_counts = pd.concat(
                [top, pd.DataFrame({"Country": ["Other"], "Chats": [others]})], ignore_index=True)
        plot_counts["Share"] = plot_counts["Chats"] / grand_total

        pie = (
            alt.Chart(plot_counts)
            .mark_arc(outerRadius=180, innerRadius=100, stroke="#f8f9fa", strokeWidth=2)
            .encode(
                theta=alt.Theta("Chats:Q", stack=True),
                color=alt.Color("Country:N", legend=alt.Legend(title="Country"),
                                scale=alt.Scale(scheme="tableau10")),
                order=alt.Order("Chats:Q", sort="descending"),
                tooltip=[alt.Tooltip("Country:N", title="Country"),
                         alt.Tooltip("Chats:Q", title="Chats", format=","),
                         alt.Tooltip("Share:Q", title="Share", format=".1%")],
            )
            .properties(width=550, height=450,
                        title=alt.TitleParams("Chat Volume by Country (from Chat Button)",
                                              anchor="middle", fontSize=16))
        )

        label_df = plot_counts[plot_counts["Share"] >= min_share].copy()
        label_df["Label"] = label_df.apply(lambda r: f"{r['Country']} ({r['Share']:.0%})", axis=1)
        labels = (
            alt.Chart(label_df)
            .mark_text(radius=205, size=12)
            .encode(theta=alt.Theta("Chats:Q", stack=True), text="Label:N",
                    order=alt.Order("Chats:Q", sort="descending"), color=alt.value("black"))
        )
        center_text = alt.Chart(
            pd.DataFrame({"text": [f"Total: {grand_total:,}"], "x": [0], "y": [0]})
        ).mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold").encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="text:N")

        altair_chart(alt.layer(pie, labels, center_text).configure_view(stroke=None))

        with st.expander("View country breakdown table"):
            _tbl = counts.copy()
            _tbl["Share"] = _tbl["Chats"] / grand_total
            st.dataframe(_tbl.style.format({"Chats": "{:,}", "Share": "{:.1%}"}), width="stretch")

            # FIX #24: make mapping gaps visible instead of burying them in "Unknown".
            _unknown = dfc.loc[dfc["Country"] == "Unknown", DEVNAME_COL]
            if len(_unknown):
                _names = _unknown.value_counts()
                st.markdown(f"**{len(_unknown):,} chats did not map to a country.** "
                            f"Unmapped chat buttons:")
                st.dataframe(_names.rename("Chats").reset_index().rename(
                    columns={"index": DEVNAME_COL}), width="stretch")


# =============================================================================
# Hourly weighted SLA
# =============================================================================
st.markdown("---")
st.subheader("⏱️ Hourly Weighted SLA (selected day)")

hourly_date = st.date_input("Select a day for hourly view", value=end_date,
                            min_value=min_date, max_value=max_date)


def compute_hourly_availability(sel_date) -> pd.DataFrame:
    day_start = pd.Timestamp(sel_date)
    day_end = day_start + pd.Timedelta(days=1)

    pres_day = df_presence[(df_presence["Start DT"] < day_end)
                           & (df_presence["End DT Clamped"] > day_start)]

    hours = pd.date_range(day_start, day_end, freq="h", inclusive="left")
    avail = {h: 0.0 for h in hours}
    logged = {h: set() for h in hours}

    for _, r in pres_day.iterrows():
        seg = clip_to_window(r["Start DT"], r["End DT Clamped"], day_start, day_end)
        if not seg:
            continue
        seg_s, seg_e = seg
        agent = str(r["Created By: Full Name"])
        status = str(r["Service Presence Status: Developer Name"]).strip()

        h = seg_s.floor("h")
        while h < seg_e:
            nxt = h + pd.Timedelta(hours=1)
            o = clip_to_window(seg_s, seg_e, h, nxt)
            if o and h in avail:
                logged[h].add(agent)
                if status in AVAILABLE_STATUSES:
                    avail[h] += (o[1] - o[0]).total_seconds()
            h = nxt

    return pd.DataFrame({
        "Hour": list(hours),
        "Avail (min)": [avail[h] / 60.0 for h in hours],
        "Logged In Agents": [len(logged[h]) for h in hours],
    })


def compute_hourly_sla(sel_date) -> pd.DataFrame:
    day_start = pd.Timestamp(sel_date)
    day_end = day_start + pd.Timedelta(days=1)
    hours = pd.date_range(day_start, day_end, freq="h", inclusive="left")
    out = pd.DataFrame({"Hour": hours})

    chat_day = chat_sla_df[(chat_sla_df["Date/Time Opened"] >= day_start)
                           & (chat_sla_df["Date/Time Opened"] < day_end)].copy()
    # Same opened-OR-responded union as the period view, attributed to one
    # timestamp inside the day so hourly buckets don't double-count.
    email_day = email_window_slice(email_sla_df, day_start, day_end)

    # FIX #18: build the per-hour series explicitly. The previous
    # `groupby(...).apply(...).rename("Chat SLA")` raised TypeError whenever the
    # slice was empty (an empty groupby.apply returns a DataFrame, and
    # DataFrame.rename(str) is invalid), so any day with only one channel of
    # activity crashed the whole page. This also avoids the pandas 2.2+
    # groupby.apply deprecation warning.
    chat_scores, email_scores, chat_wts, email_wts = [], [], [], []
    for h in hours:
        nxt = h + pd.Timedelta(hours=1)
        cslice = chat_day[(chat_day["Date/Time Opened"] >= h) & (chat_day["Date/Time Opened"] < nxt)]
        eslice = email_day[(email_day["SLA Timestamp"] >= h) & (email_day["SLA Timestamp"] < nxt)]
        chat_scores.append(chat_sla_from_slice(cslice))
        email_scores.append(email_sla_from_slice(eslice))
        # SLA weights come from the same slices that produced the scores (FIX #14).
        chat_wts.append(len(cslice))
        email_wts.append(len(email_scorable(eslice)))

    out["Chat SLA"] = chat_scores
    out["Email SLA"] = email_scores
    out["Chat SLA Wt"] = chat_wts
    out["Email SLA Wt"] = email_wts

    items_day = df_items[(df_items["Start DT"] >= day_start) & (df_items["Start DT"] < day_end)].copy()
    if not items_day.empty:
        items_day["Hour"] = items_day["Start DT"].dt.floor("h")
        cv = (items_day[items_day["Service Channel: Developer Name"] == CHAT_DEVNAME]
              .groupby("Hour").size().rename("Chat Vol"))
        ev = (items_day[items_day["Service Channel: Developer Name"] == EMAIL_DEVNAME]
              .groupby("Hour").size().rename("Email Vol"))
        out = out.merge(cv.reset_index(), on="Hour", how="left").merge(
            ev.reset_index(), on="Hour", how="left")
    else:
        out["Chat Vol"] = 0
        out["Email Vol"] = 0
    out[["Chat Vol", "Email Vol"]] = out[["Chat Vol", "Email Vol"]].fillna(0).astype(int)

    out = out.merge(compute_hourly_availability(sel_date), on="Hour", how="left")

    # FIX #13 (hourly): an hour with no contacts is a gap, not a zero.
    cw = out["Chat SLA Wt"].where(out["Chat SLA"].notna(), 0).fillna(0)
    ew = out["Email SLA Wt"].where(out["Email SLA"].notna(), 0).fillna(0)
    num = out["Chat SLA"].fillna(0) * cw + out["Email SLA"].fillna(0) * ew
    den = cw + ew
    # Mask the denominator BEFORE dividing (see the daily calculation above).
    out["Weighted SLA"] = num / den.where(den > 0)

    return out


df_hourly = compute_hourly_sla(hourly_date)

if df_hourly["Weighted SLA"].notna().sum() == 0 and df_hourly["Avail (min)"].sum() == 0:
    st.info("No activity for the selected day.")
else:
    show_breakdown = st.checkbox("Show Chat & Email lines", value=False)
    show_avail = st.checkbox("Overlay available minutes (bars)", value=True)

    _hplot = df_hourly.dropna(subset=["Weighted SLA"])

    layers = []
    if not _hplot.empty:
        weighted_line = (
            alt.Chart(_hplot)
            .mark_line(point=True, color="#2F80ED")
            .encode(
                x=alt.X("Hour:T", title="Hour", axis=alt.Axis(format="%H:%M", labelAngle=-45)),
                y=alt.Y("Weighted SLA:Q", title="Weighted SLA",
                        scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(orient="left")),
                tooltip=[alt.Tooltip("Hour:T", title="Hour", format="%H:%M"),
                         alt.Tooltip("Weighted SLA:Q", format=".1f"),
                         alt.Tooltip("Chat SLA Wt:Q", title="Chats", format=",.0f"),
                         alt.Tooltip("Email SLA Wt:Q", title="Emails", format=",.0f")],
            )
        )
        layers.append(weighted_line)

        if show_breakdown:
            for _col, _colour, _wt in (("Chat SLA", "#4CAF50", "Chat SLA Wt"),
                                       ("Email SLA", "#F44336", "Email SLA Wt")):
                _sub = df_hourly.dropna(subset=[_col])
                if _sub.empty:
                    continue
                layers.append(
                    alt.Chart(_sub).mark_line(point=True, color=_colour).encode(
                        x=alt.X("Hour:T", axis=None),
                        y=alt.Y(f"{_col}:Q", scale=alt.Scale(domain=[0, 105]), axis=None),
                        tooltip=[alt.Tooltip("Hour:T", format="%H:%M"),
                                 alt.Tooltip(f"{_col}:Q", title=_col, format=".1f"),
                                 alt.Tooltip(f"{_wt}:Q", title="Contacts", format=",.0f")],
                    )
                )

    if show_avail and df_hourly["Avail (min)"].sum() > 0:
        layers.append(
            alt.Chart(df_hourly).mark_bar(opacity=0.3, color="#FFC107").encode(
                x=alt.X("Hour:T", axis=None),
                y=alt.Y("Avail (min):Q", title="Available (min)", axis=alt.Axis(orient="right")),
                tooltip=[alt.Tooltip("Hour:T", format="%H:%M"),
                         alt.Tooltip("Avail (min):Q", title="Available (min)", format=",.0f"),
                         alt.Tooltip("Logged In Agents:Q", title="Logged In Agents", format=",.0f")],
            )
        )

    if layers:
        altair_chart(alt.layer(*layers).resolve_scale(y="independent").properties(height=350))
        st.caption("Available minutes are summed across agents, so an hour with 3 agents "
                   "fully available reads 180.")
    else:
        st.info("No activity for the selected day.")

"""
Department Performance Dashboard
================================

Revision notes (this version fixes 27 issues found in a full calculation review):

ENHANCED MESSAGING
  30. Loads Agent Work and Messaging Session Metrics as separate sources joined
      at session level, preventing transfers from duplicating response metrics.
  31. Adds Messaging volume, handled/abandoned sessions, AHT, transfers,
      first-response performance, speed to answer and message counts.
  32. Legacy Chat and Enhanced Messaging share the observed two-slot real-time
      conversation capacity; their channel-level metrics remain separate.
  33. Adds daily Messaging detail, heatmap coverage and data-quality checks.
  34. Weighted SLA now transitions from legacy Chat through 31 October 2026 to
      Enhanced Messaging from 1 November, with Email included throughout.

CAPACITY MODEL
  28. Chat utilisation now models two concurrent chat slots per agent. One
      active chat consumes 50% of chat capacity, two consume 100%, and any
      additional overlap is capped rather than pushing utilisation above the
      configured operational capacity.
  29. Email utilisation now uses residual capacity. Email-only presence gives
      one full email slot; during 'Available_All', email capacity is available
      only while there are no active chats. Email work recorded during a chat
      is reported explicitly instead of inflating utilisation.

CRITICAL
  1.  Utilization is capacity-based. Chat uses all chat-eligible availability;
      email uses email-only availability plus the capacity left by chats during
      shared 'Available_All' time.
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

import html as _html
import re
import unicodedata
import datetime as _dt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from data.ingest import clean_email_frame, normalise_case_number, reopen_timing_available
from metrics.adherence import (
    COVERAGE_MIN_PCT as ADH_COVERAGE_MIN_PCT, TARGET_PCT as ADH_TARGET_PCT, compute_adherence,
    shifts_long_from_wide,
)
from metrics.cx import (
    csat_summary, csat_trend, in_window as cx_in_window, nps_summary, operational_fcr,
    prepare_survey, survey_fcr_summary, unmapped_questions,
)
from metrics.workload import UtilisationLine, email_handling, messaging_diagnostics
from metrics.backlog import (
    NEW_BANDS, NEW_TARGET_HOURS, REOPENED_BANDS, REOPENED_TARGET_HOURS, ageing as backlog_ageing,
    build_populations, reconcile as backlog_reconcile,
)
from metrics.breakdowns import (
    MIN_CONTACTS as BD_MIN_CONTACTS, adaptive_grain, add_time_dims, attainment_trend,
    combine_outcomes, dimension_table, first_queue_by_case,
)
from metrics.reconcile import email_cohort_reconciliation
from metrics.service_attainment import (
    COVERAGE_MIN_PCT, EMAIL_TARGET_BUSINESS_HOURS, MESSAGING_RESPONSE_COL,
    MESSAGING_TARGET_SEC, TARGET_PCT, attainment_series, build_overall, classify_email,
    classify_messaging, fmt_pct_floor, fmt_ratio, in_window, reopened_summary,
)

# =============================================================================
# Page config & styling
# =============================================================================
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI THEME: "liquid glass". Presentation only — warm blurred gradient backdrop,
# frosted translucent panels, light typography. Pairs with .streamlit/config.toml
# (dark base) so native widgets and dataframes render with light text.
st.markdown(
    """
<style>
/* Brand font: Proxima Nova, served from static/fonts (see .streamlit/config.toml) */
@font-face { font-family: 'Proxima Nova'; font-weight: 300; font-style: normal; font-display: swap;
    src: url('app/static/fonts/ProximaNova-light.woff2') format('woff2'); }
@font-face { font-family: 'Proxima Nova'; font-weight: 400; font-style: normal; font-display: swap;
    src: url('app/static/fonts/ProximaNova-regular.woff2') format('woff2'); }
@font-face { font-family: 'Proxima Nova'; font-weight: 500; font-style: normal; font-display: swap;
    src: url('app/static/fonts/ProximaNova-medium.woff2') format('woff2'); }
@font-face { font-family: 'Proxima Nova'; font-weight: 600; font-style: normal; font-display: swap;
    src: url('app/static/fonts/ProximaNova-semibold.woff2') format('woff2'); }
@font-face { font-family: 'Proxima Nova'; font-weight: 700; font-style: normal; font-display: swap;
    src: url('app/static/fonts/ProximaNova-bold.woff2') format('woff2'); }

:root {
    /* Premier Bet palette (Editec/GOAT brand guidelines) */
    --pb-bet-green: #009A44;
    --pb-bright-green: #00FF66;
    --pb-dark-green: #003806;
    --pb-dark-grey: #111111;
    --pb-bet-yellow: #FAE100;
    --pb-bet-orange: #FF9300;
    --pb-white: #FFFFFF;

    --glass-bg: linear-gradient(145deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.04) 100%);
    --glass-tint: rgba(0, 56, 6, 0.38);
    --glass-border: rgba(255,255,255,0.24);
    --glass-highlight: rgba(255,255,255,0.45);
    --glass-shadow: 0 26px 50px -20px rgba(0, 20, 2, 0.70);
    --glass-blur: blur(22px) saturate(150%);
    --text-main: #FFFFFF;
    --text-soft: rgba(255,255,255,0.80);
    --text-faint: rgba(255,255,255,0.60);
    --radius-lg: 28px;
    --radius-md: 20px;
}

/* ---------- Backdrop ---------- */
.stApp {
    background:
        radial-gradient(ellipse 45% 38% at 88% 6%,  rgba(0,255,102,0.38) 0%, transparent 62%),
        radial-gradient(ellipse 50% 40% at 12% 18%, rgba(0,154,68,0.95) 0%, transparent 62%),
        radial-gradient(ellipse 60% 50% at 75% 72%, rgba(0,154,68,0.90) 0%, transparent 66%),
        radial-gradient(ellipse 55% 55% at 8% 92%,  rgba(0,56,6,0.85) 0%, transparent 62%),
        radial-gradient(ellipse 40% 35% at 45% 42%, rgba(0,255,102,0.16) 0%, transparent 70%),
        linear-gradient(160deg, #009A44 0%, #007A36 45%, #003806 100%);
    background-attachment: fixed;
    color: var(--text-main);
    font-family: 'Proxima Nova', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.main .block-container, [data-testid="stMainBlockContainer"] { padding-top: 2rem; }

/* ---------- Typography ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp p, .stApp li, .stApp label, .stApp summary {
    font-family: 'Proxima Nova', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--text-main);
    font-weight: 500;
    letter-spacing: -0.01em;
}
.stApp h3 { font-size: 1.4rem; padding-top: 0.4rem; }
.stApp h3 [data-testid="stIconMaterial"], .stApp h4 [data-testid="stIconMaterial"] {
    color: var(--pb-bright-green); vertical-align: -0.12em; margin-right: 0.15em;
}
.stApp h4 { font-size: 1.1rem; color: var(--text-soft); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-soft) !important; }
.stApp hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.38), transparent);
    margin: 2.2rem 0 1.2rem;
}

/* ---------- Shared glass surface ---------- */
.glass-tile, .glass-hero,
[data-testid="stMetric"],
[data-testid="stExpander"] details,
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: var(--glass-bg), var(--glass-tint);
    border: 1px solid var(--glass-border);
    box-shadow: inset 0 1px 0 var(--glass-highlight),
                inset 0 -1px 0 rgba(255,255,255,0.06),
                var(--glass-shadow);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
}

/* ---------- Hero header ---------- */
.glass-hero {
    position: relative;
    overflow: hidden;
    border-radius: 34px;
    padding: 26px 30px 24px;
    margin: 0 0 1.4rem;
    background:
        radial-gradient(ellipse 60% 95% at 88% 40%, rgba(0,255,102,0.30) 0%, transparent 70%),
        radial-gradient(ellipse 55% 80% at 40% 0%,  rgba(255,255,255,0.14) 0%, transparent 70%),
        linear-gradient(120deg, rgba(0,56,6,0.78) 0%, rgba(0,154,68,0.45) 100%);
    color: var(--text-main);
}
.gh-top { display: flex; justify-content: space-between; align-items: flex-start; }
.gh-arrow {
    width: 54px; height: 54px; border-radius: 50%;
    background: var(--pb-bet-yellow); color: var(--pb-dark-grey);
    display: grid; place-items: center; font-size: 1.25rem; font-weight: 700;
    box-shadow: 0 10px 30px -8px rgba(0,0,0,0.45);
}
.gh-kicker { margin-top: 18px; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--pb-bet-yellow); }
.gh-title  { font-size: 2.2rem; font-weight: 500; line-height: 1.1; margin: 4px 0 18px; letter-spacing: -0.02em; }
.gh-bottom { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
.gh-period { font-size: 1.15rem; font-weight: 400; color: var(--text-main); }
.gh-track  { display: flex; align-items: center; gap: 0; }
.gh-track i {
    display: block; width: 10px; height: 10px; border-radius: 50%;
    background: rgba(255,255,255,0.92);
}
.gh-track i + i { margin-left: 26px; position: relative; }
.gh-track i + i::before {
    content: ""; position: absolute; right: 100%; top: 50%;
    width: 26px; height: 1px; background: rgba(255,255,255,0.55);
}
.gh-track i.mid { width: 20px; height: 20px; background: rgba(0,255,102,0.55); }
.gh-track i.end { width: 32px; height: 32px; background: var(--pb-bright-green); box-shadow: 0 0 18px rgba(0,255,102,0.7); }

/* ---------- KPI tiles ---------- */
.glass-tile {
    --accent: #D6D3D1;
    position: relative;
    isolation: isolate;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 14px;
    min-height: 176px;
    margin-bottom: 1rem;
    padding: 18px 20px 18px;
    border-radius: var(--radius-lg);
    color: var(--text-main);
    transition: transform .25s ease, box-shadow .25s ease;
}
.glass-tile::before {
    content: "";
    position: absolute; inset: 0; z-index: -1;
    background: radial-gradient(circle at 100% 0%,
                color-mix(in srgb, var(--accent) 38%, transparent) 0%, transparent 58%);
    opacity: 0.75;
}
.glass-tile:hover {
    transform: translateY(-3px);
    box-shadow: inset 0 1px 0 var(--glass-highlight), 0 32px 60px -22px rgba(0,20,2,0.80);
}
.gt-head   { display: flex; align-items: center; gap: 12px; }
.gt-icon {
    flex: none;
    width: 44px; height: 44px; border-radius: 50%;
    display: grid; place-items: center;
    font-size: 1.1rem;
    border: 1px solid rgba(255,255,255,0.38);
    background: rgba(255,255,255,0.06);
    color: var(--pb-white);
}
.gt-icon svg { display: block; }
.gh-arrow svg { display: block; }
.gt-title  { flex: 1; font-size: 0.98rem; font-weight: 500; line-height: 1.18; }
.gt-status {
    flex: none; align-self: flex-start; margin-top: 4px;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 22%, transparent),
                0 0 16px var(--accent);
}
.gt-value {
    margin-top: auto;
    font-size: 2.4rem; font-weight: 400; line-height: 1;
    letter-spacing: -0.02em;
    color: color-mix(in srgb, var(--accent) 30%, #FFFFFF);
    font-variant-numeric: tabular-nums;
}
.gt-unit { font-size: 0.48em; opacity: 0.75; margin-left: 2px; }
.gt-sub  { font-size: 0.80rem; color: var(--text-faint); line-height: 1.3; }

/* ---------- Native st.metric ---------- */
[data-testid="stMetric"] { border-radius: var(--radius-md); padding: 12px 12px 10px; }
[data-testid="stMetricLabel"] { color: var(--text-soft); }
[data-testid="stMetricValue"] { font-weight: 400; font-size: 1.45rem; color: var(--text-main); }
[data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"] div {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    word-break: normal !important; overflow-wrap: normal !important;
    line-height: 1.2;
}
[data-testid="stMetricLabel"] p { font-size: 0.82rem; }
[data-testid="stMetric"] { height: 100%; }

/* ---------- Expanders ---------- */
[data-testid="stExpander"] details { border-radius: var(--radius-md); overflow: hidden; }
[data-testid="stExpander"] summary { color: var(--text-main); font-weight: 500; }
[data-testid="stExpander"] summary:hover { background: rgba(255,255,255,0.06); }

/* ---------- Charts ---------- */
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
    border-radius: 24px;
    padding: 16px 6px 8px;
    box-sizing: border-box;
    overflow: hidden;
}
#vg-tooltip-element {
    background: rgba(0,56,6,0.90) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(14px);
    font-family: 'Proxima Nova', system-ui, sans-serif !important;
}

/* ---------- Tabs (pill segmented control) ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; padding: 6px; width: fit-content;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid var(--glass-border);
    backdrop-filter: var(--glass-blur);
}
.stTabs [data-baseweb="tab"] {
    height: auto; padding: 7px 18px;
    border-radius: 999px;
    background: transparent;
    color: var(--text-soft);
}
.stTabs [aria-selected="true"] { background: var(--pb-bet-yellow); color: var(--pb-dark-grey); }
.stTabs [aria-selected="true"] p { color: var(--pb-dark-grey); font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: rgba(0,56,6,0.55) !important;
    backdrop-filter: blur(30px) saturate(140%);
    -webkit-backdrop-filter: blur(30px) saturate(140%);
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"] > div:first-child { background: transparent; }

/* ---------- Inputs & buttons ---------- */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    border-color: var(--glass-border) !important;
}
.stApp input { color: var(--text-main) !important; }
.stButton > button {
    border-radius: 999px;
    border: none;
    background: var(--pb-bet-yellow);
    color: var(--pb-dark-grey);
    font-weight: 600;
    box-shadow: 0 10px 24px -10px rgba(0,0,0,0.55);
    transition: filter .2s ease, transform .2s ease;
}
.stButton > button p { color: var(--pb-dark-grey); font-weight: 600; }
.stButton > button:hover { filter: brightness(1.06); transform: translateY(-1px); color: var(--pb-dark-grey); }
.stButton > button:focus:not(:active) { color: var(--pb-dark-grey); }

/* ---------- Tables & alerts ---------- */
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid var(--glass-border); }
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
    border-radius: 18px;
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.25); border-radius: 999px; }
::-webkit-scrollbar-track { background: transparent; }
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
MESSAGING_DEVNAME = "sfdc_livemessage"
EMAIL_DEVNAME = "casesChannel"

# Enhanced Messaging uses the same real-time conversation availability pool as
# legacy Chat in this org. The supplied presence and Agent Work exports show a
# maximum of two simultaneous assignments, matching the existing Chat capacity.
MESSAGING_FIRST_RESPONSE_TARGET_SEC = 60.0
MESSAGING_SLA_START_DATE = datetime(2026, 11, 1).date()
MESSAGING_CHANNEL_FIELD = "Messaging Channel: Channel Name"
MESSAGING_CHANNEL_FIELD_CANDIDATES = (
    MESSAGING_CHANNEL_FIELD, "Messaging Channel Name", "Channel Name", "Country")
MESSAGING_TARGET_RESPONSE_MIN = 1.0
MESSAGING_RESPONSE_TOLERANCE_MIN = 4.0
W_MESSAGING_IN_TARGET = 0.50
W_MESSAGING_RESPONSE = 0.30
W_MESSAGING_ABANDON = 0.20
MESSAGING_RESCALE_K = W_MESSAGING_IN_TARGET

# One agent can actively work up to two chats at once. Chat utilisation is
# therefore measured against two chat slots: one active chat consumes 50% of
# the agent's chat capacity and two active chats consume 100%.
CHAT_CONCURRENCY_CAPACITY = 2

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

# Premier Bet palette: Bright Green / Bet Yellow / Bet Orange; neutral = soft white.
COLOR_GOOD, COLOR_WARN, COLOR_BAD, COLOR_NONE = "#00FF66", "#FAE100", "#FF9300", "#CFE3D6"

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


# UI: thin line icons (24px grid, white stroke) to match the glass theme.
_ICON_PATHS = {
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/>',
    "timer": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M9 2h6"/>',
    "messages": '<path d="M14 9a2 2 0 0 1-2 2H6l-3 3V4a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2z"/>'
                '<path d="M18 9h1a2 2 0 0 1 2 2v10l-3-3h-6a2 2 0 0 1-2-2v-1"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "transfer": '<path d="M4 7h16l-4-4M20 17H4l4 4"/>',
    "zap": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "reply": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 6 6v5"/>',
    "gauge": '<path d="M4 18a9 9 0 1 1 16 0"/><path d="m12 14 4-5"/><circle cx="12" cy="14" r="1.2"/>',
    "send": '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/>',
    "pie": '<circle cx="12" cy="12" r="9"/><path d="M12 3v9l6.4 6.4"/>',
    "bars": '<path d="M4 20h16M7 16v-4M12 16V7M17 16v-6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/>'
                '<path d="M3 10h18M8 3v4M16 3v4M9 15l2 2 4-4"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "clipboard": '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V3h6v1M9 11h6M9 15h4"/>',
    "smile": '<circle cx="12" cy="12" r="9"/><path d="M8.5 14a4 4 0 0 0 7 0M9 9.5h.01M15 9.5h.01"/>',
    "star": '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    "shield": '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 6-6"/>',
    "arrow": '<path d="M7 17 17 7M8 7h9v9"/>',
    "spark": '<path d="M12 3c.6 4.5 1.5 5.4 6 6-4.5.6-5.4 1.5-6 6-.6-4.5-1.5-5.4-6-6 4.5-.6 5.4-1.5 6-6z"/>',
}

# Tile label -> icon. Unlisted labels fall back to "spark".
_TILE_ICONS = {
    "Total Chats": "chat", "Total Emails": "mail",
    "Avg Chat Handle Time": "timer", "Avg Email Handle Time": "timer",
    "Messaging Sessions": "messages", "Sessions Handled": "check",
    "Messaging AHT": "timer", "Transferred Sessions": "transfer",
    "First Response ≤60s": "zap", "Median First Response": "reply",
    "Avg Speed to Answer": "gauge", "Messages Sent": "send",
    "Conversation Utilization": "pie", "Email Utilization": "bars",
    "Avg Email Resp Time": "clock", "Schedule Adherence": "calendar",
    "Chat SLA Score": "target", "Messaging SLA Score": "target",
    "Email SLA Score": "target", "Weighted SLA Score": "layers",
    "Surveys": "clipboard", "CSAT (avg %)": "smile", "NPS": "star", "FCR": "shield",
}


def _icon_svg(name: str, size: int = 20) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS.get(name, _ICON_PATHS["spark"])}</svg>')


def render_custom_metric(container, title, value, tooltip, color, sub: str = ""):
    # Presentation only: same inputs as before, rendered as a glass tile. Any
    # leading emoji in the title is dropped in favour of a line icon chosen by
    # label; a trailing '%' on the value is set as a small unit; `color`
    # drives the status glow.
    label = re.sub(r"^[^\w\s]+\s*", "", str(title)).strip()
    icon = _icon_svg(_TILE_ICONS.get(label, "spark"))
    value = str(value)
    if value.endswith("%") and len(value) > 1:
        value = f'{value[:-1]}<span class="gt-unit">%</span>'
    sub_html = f'<div class="gt-sub">{sub}</div>' if sub else ""
    container.markdown(
        f'<div class="glass-tile" style="--accent:{color}" '
        f'title="{_html.escape(str(tooltip), quote=True)}">'
        f'<div class="gt-head"><div class="gt-icon">{icon}</div>'
        f'<div class="gt-title">{label}</div><div class="gt-status"></div></div>'
        f'<div class="gt-value">{value}</div>{sub_html}</div>',
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
    try:  # UI: let the glass panel show through behind every chart
        chart = chart.properties(background="transparent")
        _cfg = chart.config
        if _cfg is alt.Undefined or getattr(_cfg, "axis", alt.Undefined) is alt.Undefined:
            chart = chart.configure_axis(
                gridColor="rgba(255,255,255,0.14)", domainColor="rgba(255,255,255,0.30)",
                tickColor="rgba(255,255,255,0.30)", labelColor="#FFFFFF",
                titleColor="rgba(255,255,255,0.80)",
                labelFont="Proxima Nova", titleFont="Proxima Nova")
    except Exception:
        pass
    try:
        st.altair_chart(chart, width="stretch", **kwargs)
    except TypeError:
        st.altair_chart(chart, use_container_width=True, **kwargs)


def date_axis(dates, fmt: str = "%d %b", angle: int = -45, max_ticks: int = 62, **kw):
    """Axis with ONE tick per distinct calendar day.

    Vega-Lite's automatic time ticks sit every ~12 hours on a short daily series, so a date-only
    label repeats ("13 Nov, 13 Nov, 14 Nov, ..."). Pinning the tick values to the distinct days
    avoids that. Long series fall back to an automatic, de-duplicated tick count.
    """
    try:
        days = sorted(pd.to_datetime(pd.Series(list(dates))).dt.normalize().dropna().unique())
    except Exception:
        days = []
    if days and len(days) <= max_ticks:
        return alt.Axis(format=fmt, labelAngle=angle, values=[pd.Timestamp(d).to_pydatetime() for d in days], **kw)
    return alt.Axis(format=fmt, labelAngle=angle, tickCount=12, **kw)


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


def concurrent_slot_seconds(work_intervals, available_intervals,
                            capacity: int = CHAT_CONCURRENCY_CAPACITY) -> float:
    """Return used slot-seconds inside availability, capped at ``capacity``.

    Unlike ``merge_intervals``, this deliberately preserves overlapping work
    items. With a capacity of two, one active chat for an hour contributes one
    slot-hour, while two simultaneous chats contribute two slot-hours. Any
    third overlapping chat is ignored for utilisation rather than allowing the
    KPI to exceed the configured operational capacity.
    """
    if capacity <= 0:
        raise ValueError("capacity must be greater than zero")

    available = merge_intervals(available_intervals)
    if not work_intervals or not available:
        return 0.0

    events = []
    for pair in work_intervals:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        start = pd.to_datetime(pair[0], errors="coerce")
        end = pd.to_datetime(pair[1], errors="coerce")
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue

        for av_start, av_end in available:
            overlap_start = max(start, av_start)
            overlap_end = min(end, av_end)
            if overlap_end > overlap_start:
                events.append((overlap_start, 1))
                events.append((overlap_end, -1))

    if not events:
        return 0.0

    # Process end events before start events at the same timestamp. There is no
    # elapsed time between them, but this keeps the active count intuitive.
    events.sort(key=lambda event: (event[0], event[1]))
    active = 0
    previous = None
    slot_seconds = 0.0

    for timestamp, change in events:
        if previous is not None and timestamp > previous:
            elapsed = (timestamp - previous).total_seconds()
            slot_seconds += elapsed * min(active, capacity)
        active += change
        previous = timestamp

    return float(slot_seconds)


def residual_capacity_usage(primary_work_intervals, secondary_work_intervals,
                            shared_intervals,
                            primary_capacity: int = CHAT_CONCURRENCY_CAPACITY):
    """Return capacity, used capacity, and excess demand seconds for shared time.

    ``primary_work_intervals`` are chats. Under the email-routing rule, shared
    status is email-eligible only while there are no active chats; one or more
    chats leave zero email capacity. Overlapping email rows are treated as one
    active email workload.
    """
    if primary_capacity <= 0:
        raise ValueError("primary_capacity must be greater than zero")

    shared = merge_intervals(shared_intervals)
    if not shared:
        return 0.0, 0.0, 0.0

    capacity_seconds = 0.0
    used_seconds = 0.0
    raw_secondary_seconds = 0.0

    for shared_start, shared_end in shared:
        # timestamp -> [chat delta, email delta]
        events = {shared_start: [0, 0], shared_end: [0, 0]}

        for intervals, event_index in (
            (primary_work_intervals, 0),
            (secondary_work_intervals, 1),
        ):
            for pair in intervals or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                start = pd.to_datetime(pair[0], errors="coerce")
                end = pd.to_datetime(pair[1], errors="coerce")
                if pd.isna(start) or pd.isna(end) or end <= start:
                    continue
                overlap_start = max(start, shared_start)
                overlap_end = min(end, shared_end)
                if overlap_end <= overlap_start:
                    continue
                events.setdefault(overlap_start, [0, 0])[event_index] += 1
                events.setdefault(overlap_end, [0, 0])[event_index] -= 1

        active_chat = active_email = 0
        previous = shared_start
        for timestamp in sorted(events):
            if timestamp > previous:
                elapsed = (timestamp - previous).total_seconds()
                residual_fraction = 1.0 if active_chat == 0 else 0.0
                capacity_seconds += elapsed * residual_fraction
                if active_email > 0:
                    used_seconds += elapsed * residual_fraction
                    raw_secondary_seconds += elapsed
            chat_delta, email_delta = events[timestamp]
            active_chat += chat_delta
            active_email += email_delta
            previous = timestamp

    excess_seconds = max(0.0, raw_secondary_seconds - used_seconds)
    return float(capacity_seconds), float(used_seconds), float(excess_seconds)


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
    if "encoding" in read_kwargs:
        return pd.read_csv(path_str, **read_kwargs)
    try:
        return pd.read_csv(path_str, encoding="utf-8-sig", **read_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path_str, encoding="cp1252", **read_kwargs)


chat_path = resolve_path_case_insensitive("chat.csv")
email_path = resolve_path_case_insensitive("email.csv")
survey_path = resolve_path_case_insensitive("survey.csv")
items_path = resolve_path_case_insensitive("report_items.csv")
messaging_items_path = resolve_path_case_insensitive("report_messaging_agent_work.csv")
messaging_metrics_path = resolve_path_case_insensitive("report_messaging_metrics.csv")
pres_path = resolve_path_case_insensitive("report_presence.csv")
shifts_path = resolve_path_case_insensitive("shifts.csv")

# FIX #3: shifts.csv is no longer required. It used to block startup while
# contributing nothing; it is now optional AND actually used (adherence tile).
REQUIRED = (chat_path, email_path, items_path, messaging_items_path,
            messaging_metrics_path, pres_path)
missing = [p.name for p in REQUIRED if not p.exists()]
if missing:
    st.error(f"Missing required files in the app folder: {', '.join(missing)}")
    st.stop()

st.sidebar.markdown("---")
if st.sidebar.button(":material/refresh: Reload data now"):
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


def parse_dt(frame: pd.DataFrame, cols, *, dayfirst: bool = True):
    """Parse Salesforce dates using UK day-first order by default.

    Ambiguous values such as 12/10/2026 mean 12 October, not 10 December.
    Mixed-format parsing supports differing timestamp precision and ISO dates.
    Excel error strings become NaT.
    """
    for c in cols:
        if c in frame.columns:
            frame[c] = pd.to_datetime(
                frame[c], errors="coerce", format="mixed", dayfirst=dayfirst
            )
    return frame


_col_renames: dict = {}
_col_missing: dict = {}

df_items = canonicalise_columns(
    load_csv_cached(str(items_path), file_signature(items_path)),
    ["Work Item: Name", "Service Channel: Developer Name", "Handle Time",
     "Queue: Name", "User: Full Name", "Start DT", "End DT"], "report_items.csv")
if "Work Item: Name" in df_items.columns:
    # same canonical key as email.csv Case Number (zero-padding differs between exports)
    df_items["Work Item: Name"] = normalise_case_number(df_items["Work Item: Name"])
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
     "Status", "Case Origin", EMAIL_CASE_KEY, "Re-opened", "Reopened Time", "Country"],
    "email.csv")
# New-schema ingestion: drops the duplicated header row, normalises zero-padded Case Number,
# reads the authoritative `Re-opened` flag, and degrades (with messages) on the legacy schema.
email_ingest = clean_email_frame(email_sla_df)
email_sla_df = email_ingest.frame
for _c in ("Date/Time Opened", "Completion Date", "Reopened Time"):
    if _c not in email_sla_df.columns:
        email_sla_df[_c] = pd.NaT
for _c in ("Elapsed Time (Hours)", "Status"):
    if _c not in email_sla_df.columns:
        email_sla_df[_c] = pd.NA
messaging_items_df = canonicalise_columns(
    load_csv_cached(str(messaging_items_path), file_signature(messaging_items_path)),
    ["Agent Work ID", "Messaging Session Name", "User: Full Name",
     "Service Channel: Developer Name", "Queue: Name", "Assign Date",
     "Accept Date", "Close Date", "Handle Time", "Speed To Answer", "Status"],
    "report_messaging_agent_work.csv")
messaging_metrics_long = canonicalise_columns(
    load_csv_cached(str(messaging_metrics_path), file_signature(messaging_metrics_path)),
    ["Messaging Session ID", "Messaging Session Name", "Start Time", "End Time",
     "Messaging Session Metric Type", "Messaging Session Metric Value",
     MESSAGING_CHANNEL_FIELD],
    "report_messaging_metrics.csv")

# Salesforce orgs and custom report types expose Channel Name under slightly
# different labels. Normalise the common variants without making the field
# mandatory for the rest of the Messaging dashboard.
_messaging_channel_source = next(
    (c for c in MESSAGING_CHANNEL_FIELD_CANDIDATES if c in messaging_metrics_long.columns),
    None,
)
if _messaging_channel_source and _messaging_channel_source != MESSAGING_CHANNEL_FIELD:
    messaging_metrics_long = messaging_metrics_long.rename(
        columns={_messaging_channel_source: MESSAGING_CHANNEL_FIELD})
    _messaging_channel_source = MESSAGING_CHANNEL_FIELD
if _messaging_channel_source:
    # A recognised alias satisfies the requirement, so do not leave the
    # preferred Salesforce label showing as a false missing-column warning.
    _missing = _col_missing.get("report_messaging_metrics.csv", [])
    _missing = [c for c in _missing if c != MESSAGING_CHANNEL_FIELD]
    if _missing:
        _col_missing["report_messaging_metrics.csv"] = _missing
    else:
        _col_missing.pop("report_messaging_metrics.csv", None)

df_items = parse_dt(df_items, ["Start DT", "End DT"], dayfirst=True)
df_presence = parse_dt(df_presence, ["Start DT", "End DT"])
chat_sla_df = parse_dt(chat_sla_df, ["Date/Time Opened"])
email_sla_df = parse_dt(email_sla_df, ["Date/Time Opened", "Completion Date"])
messaging_items_df = parse_dt(
    messaging_items_df, ["Assign Date", "Accept Date", "Close Date"])
messaging_metrics_long = parse_dt(
    messaging_metrics_long, ["Start Time", "End Time"])

# Normalise Agent Work to the interval names used by the utilisation engine.
# An unaccepted routing attempt deliberately has no Start DT and cannot consume
# handling capacity, but remains available for abandonment/routing analysis.
messaging_items_df["Start DT"] = messaging_items_df["Accept Date"]
messaging_items_df["End DT"] = messaging_items_df["Close Date"]

df_shifts = pd.DataFrame(columns=["Agent", "Date", "Kind", "Shift Start", "Shift End"])
if shifts_path.exists():
    df_shifts_raw = load_csv_cached(str(shifts_path), file_signature(shifts_path))
    df_shifts_raw.columns = df_shifts_raw.columns.str.strip()
    df_shifts = shifts_long_from_wide(df_shifts_raw)

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
                       (email_sla_df, ["Date/Time Opened", "Completion Date"]),
                       (messaging_items_df,
                        ["Assign Date", "Accept Date", "Close Date", "Start DT", "End DT"]),
                       (messaging_metrics_long, ["Start Time", "End Time"])):
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
messaging_items_df = _apply_owner_exclusion(messaging_items_df, "messaging items")
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
_coerce_numeric(messaging_items_df, "Handle Time", "messaging_handle_time")
_coerce_numeric(messaging_items_df, "Speed To Answer", "messaging_speed_to_answer")
_coerce_numeric(
    messaging_metrics_long, "Messaging Session Metric Value", "messaging_metric_value")

# ---------------------------------------------------------------------------
# Enhanced Messaging: one session row joined to its Agent Work roll-up.
#
# The Salesforce metrics export is long-form (one row per metric type). Pivot it
# once and keep all session-level calculations on this table. Never join the
# metric values directly to Agent Work rows: transferred sessions have several
# assignments and doing so would duplicate message counts and response times.
# ---------------------------------------------------------------------------
_metric_dupes = messaging_metrics_long.duplicated(
    subset=["Messaging Session Name", "Messaging Session Metric Type"], keep=False)
dq["messaging_duplicate_metric_rows"] = int(_metric_dupes.sum())

_messaging_base_columns = [
    "Messaging Session Name", "Messaging Session ID", "Start Time", "End Time"
]
if _messaging_channel_source:
    _messaging_base_columns.append(MESSAGING_CHANNEL_FIELD)

_messaging_session_base = (
    messaging_metrics_long.sort_values("Start Time")
    .drop_duplicates(subset=["Messaging Session Name"], keep="first")
    [_messaging_base_columns]
)
_messaging_metric_wide = messaging_metrics_long.pivot_table(
    index="Messaging Session Name",
    columns="Messaging Session Metric Type",
    values="Messaging Session Metric Value",
    aggfunc="first",
).reset_index()
_messaging_metric_wide.columns.name = None

messaging_items_df["Human Agent"] = messaging_items_df["User: Full Name"].where(
    messaging_items_df["User: Full Name"].astype(str).str.strip().str.casefold()
    != "automated process"
)
messaging_items_df["Is Closed Assignment"] = (
    messaging_items_df["Status"].astype(str).str.strip().str.casefold() == "closed")
messaging_items_df["Is Abandoned Assignment"] = (
    messaging_items_df["Status"].astype(str).str.strip().str.casefold() == "abandoned")
messaging_items_df["Is Canceled Assignment"] = (
    messaging_items_df["Status"].astype(str).str.strip().str.casefold() == "canceled")
messaging_items_df["Is Unavailable Assignment"] = (
    messaging_items_df["Status"].astype(str).str.strip().str.casefold() == "unavailable")
messaging_items_df["Is Human Closed Assignment"] = (
    messaging_items_df["Is Closed Assignment"] & messaging_items_df["Human Agent"].notna())

_messaging_work_by_session = (
    messaging_items_df.groupby("Messaging Session Name", as_index=False)
    .agg(
        First_Assign=("Assign Date", "min"),
        First_Accept=("Accept Date", "min"),
        Last_Close=("Close Date", "max"),
        Assignments=("Agent Work ID", "size"),
        Accepted_Assignments=("Accept Date", "count"),
        Total_Handle_Secs=("Handle Time", lambda s: s.sum(min_count=1)),
        Avg_Speed_To_Answer=("Speed To Answer", "mean"),
        Human_Agents=("Human Agent", "nunique"),
        Any_Closed=("Is Closed Assignment", "any"),
        Any_Abandoned=("Is Abandoned Assignment", "any"),
        Any_Canceled=("Is Canceled Assignment", "any"),
        Any_Human_Closed=("Is Human Closed Assignment", "any"),
        Any_Unavailable=("Is Unavailable Assignment", "any"),
    )
)

messaging_sessions = (
    _messaging_session_base
    .merge(_messaging_metric_wide, on="Messaging Session Name", how="left")
    .merge(_messaging_work_by_session, on="Messaging Session Name", how="left")
)
for _c in ("Assignments", "Accepted_Assignments", "Human_Agents"):
    messaging_sessions[_c] = messaging_sessions[_c].fillna(0).astype(int)
for _c in ("Any_Closed", "Any_Abandoned", "Any_Canceled", "Any_Human_Closed",
           "Any_Unavailable"):
    messaging_sessions[_c] = messaging_sessions[_c].fillna(False).astype(bool)

dq["messaging_sessions"] = int(len(messaging_sessions))
dq["messaging_sessions_without_agent_work"] = int(
    (messaging_sessions["Assignments"] == 0).sum())

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
_ends = [s.max() for s in (df_items["End DT"], messaging_items_df["End DT"],
                           messaging_metrics_long["End Time"], df_presence["End DT"],
                           email_sla_df.get("Completion Date", pd.Series(dtype="datetime64[ns]")))
         if s.notna().any()]
_starts = [s.max() for s in (df_items["Start DT"], messaging_items_df["Assign Date"],
                             messaging_metrics_long["Start Time"], df_presence["Start DT"],
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

_messaging_unaccepted = messaging_items_df["Start DT"].isna()
if "Handle Time" in messaging_items_df.columns:
    _messaging_unaccepted = _messaging_unaccepted | messaging_items_df["Handle Time"].isna()
dq["messaging_items_unaccepted"] = int(_messaging_unaccepted.sum())
dq["messaging_items_total"] = int(len(messaging_items_df))
dq["messaging_items_unaccepted_pct"] = (
    dq["messaging_items_unaccepted"] / dq["messaging_items_total"] * 100
    if dq["messaging_items_total"] else 0.0
)

for _df, _label in ((df_items, "items"),
                    (messaging_items_df, "messaging items"),
                    (df_presence, "presence")):
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
# Cast the fallback to the column's own datetime dtype: filling datetime64[us] with a differently
# typed datetime array raises Pandas4Warning today and will raise in a future pandas.
# (Legacy diagnostic only: the headline attainment uses Completion Date, never Elapsed Time.)
_from_elapsed = pd.to_datetime(_from_elapsed).astype(email_sla_df["Completion Date"].dtype)
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
    # A case with no routing record was never re-routed: treat it as a single touch.
    email_sla_df["Touches"] = email_sla_df["Touches"].fillna(1).astype(int)
else:
    email_sla_df["Touches"] = 1
# `Reopened` is the authoritative `Re-opened` export flag (set in data.ingest). It is NEVER
# inferred from routing touches (the old `Touches > 1` proxy was banned and removed).
email_sla_df["Effective Target"] = email_sla_df["Target Hrs"]


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
    messaging_sessions["Start Time"],
]
_mins = [s.min().date() for s in _all_dates if s.notna().any()]
_maxs = [s.max().date() for s in _all_dates if s.notna().any()]
if not _mins:
    st.error("No usable dates found in the Chat, Email, work-item or Messaging exports.")
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
with st.expander(":material/info: Data files & quality"):
    def fmt_sig(p: Path):
        mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime)
        return f"{p.name} • {p.stat().st_size:,} bytes • {mtime:%Y-%m-%d %H:%M:%S}"

    _files = [chat_path, email_path, items_path, messaging_items_path,
              messaging_metrics_path, pres_path]
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
                                  ("messaging items", "Messaging work item", "Messaging work items"),
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
        ("uncoercible_messaging_handle_time", "Messaging 'Handle Time' values were not numeric and became blank"),
        ("uncoercible_messaging_speed_to_answer", "Messaging 'Speed To Answer' values were not numeric and became blank"),
        ("uncoercible_messaging_metric_value", "Messaging metric values were not numeric and became blank"),
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
    if dq.get("messaging_items_unaccepted"):
        notes.append(
            f"**{dq['messaging_items_unaccepted']:,}** of "
            f"**{dq.get('messaging_items_total', 0):,}** Messaging routing attempts "
            f"({dq.get('messaging_items_unaccepted_pct', 0):.1f}%) were not accepted. "
            "They remain in session abandonment analysis but are excluded from AHT and utilization."
        )
    if dq.get("messaging_sessions_without_agent_work"):
        notes.append(
            f"**{dq['messaging_sessions_without_agent_work']:,}** Messaging session(s) have metrics "
            "but no matching Agent Work row. They remain in total session volume but are unclassified "
            "for human handling and abandonment."
        )
    if dq.get("messaging_duplicate_metric_rows"):
        notes.append(
            f"⚠️ **{dq['messaging_duplicate_metric_rows']:,}** Messaging metric row(s) repeat a "
            "session/metric pair; the first value is used."
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

# Enhanced Messaging sessions are attributed to the day they started. Metric
# rows are available only after a session ends, so this is a historical view,
# not a live queue monitor.
messaging_period = messaging_sessions[
    (messaging_sessions["Start Time"] >= ts_start)
    & (messaging_sessions["Start Time"] < ts_end)
    & (messaging_sessions["Start Time"] >= pd.Timestamp(MESSAGING_SLA_START_DATE))
].copy()
_messaging_period_names = set(messaging_period["Messaging Session Name"])
messaging_work_period = messaging_items_df[
    messaging_items_df["Messaging Session Name"].isin(_messaging_period_names)
].copy()


def _positive_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric, strictly-positive Messaging metric values.

    Salesforce emits zero when a response or message does not exist. Treating
    that as a genuine zero-second response would incorrectly improve the SLA.
    """
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[values > 0]


messaging_total = int(len(messaging_period))
messaging_handled = int(messaging_period["Any_Closed"].sum())
messaging_abandoned = int(
    ((~messaging_period["Any_Closed"]) & messaging_period["Any_Abandoned"]).sum())
messaging_unclassified = int((messaging_period["Assignments"] == 0).sum())
messaging_transfers = int((messaging_period["Human_Agents"] > 1).sum())
messaging_assignments = int(messaging_period["Assignments"].sum())

_messaging_handle = pd.to_numeric(
    messaging_period.loc[messaging_period["Any_Closed"], "Total_Handle_Secs"],
    errors="coerce",
).dropna()
messaging_aht = _messaging_handle.mean() if len(_messaging_handle) else None

_messaging_first_response = _positive_metric(
    messaging_period, "Service Rep First Response Time")
messaging_first_response_mean = (
    _messaging_first_response.mean() if len(_messaging_first_response) else None)
messaging_first_response_median = (
    _messaging_first_response.median() if len(_messaging_first_response) else None)
messaging_first_response_in_target = int(
    (_messaging_first_response <= MESSAGING_FIRST_RESPONSE_TARGET_SEC).sum())
messaging_first_response_target_pct = (
    messaging_first_response_in_target / len(_messaging_first_response) * 100
    if len(_messaging_first_response) else None)

_messaging_accept_to_response = _positive_metric(
    messaging_period, "Service Rep Accept To First Response Time")
messaging_accept_to_response_median = (
    _messaging_accept_to_response.median() if len(_messaging_accept_to_response) else None)

_messaging_speed = pd.to_numeric(
    messaging_work_period["Speed To Answer"], errors="coerce").dropna()
messaging_speed_to_answer = _messaging_speed.mean() if len(_messaging_speed) else None

_agent_messages = pd.to_numeric(
    messaging_period.get("Agent Message Count", pd.Series(dtype="float64")),
    errors="coerce",
).fillna(0)
_customer_messages = pd.to_numeric(
    messaging_period.get("End User Message Count", pd.Series(dtype="float64")),
    errors="coerce",
).fillna(0)
messaging_agent_messages = int(_agent_messages.sum())
messaging_customer_messages = int(_customer_messages.sum())

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
chat_cases = cases_period[
    (cases_period["Service Channel: Developer Name"] == CHAT_DEVNAME)
    & (cases_period["First_Start"] < pd.Timestamp(MESSAGING_SLA_START_DATE))
]
email_cases = cases_period[cases_period["Service Channel: Developer Name"] == EMAIL_DEVNAME]

# Row-level slices are still needed for the utilisation intervals.
chat_df = df_period[
    (df_period["Service Channel: Developer Name"] == CHAT_DEVNAME)
    & (df_period["Start DT"] < pd.Timestamp(MESSAGING_SLA_START_DATE))
]
email_df = df_period[df_period["Service Channel: Developer Name"] == EMAIL_DEVNAME]

chat_total = len(chat_cases)
email_total = len(email_cases)
has_chat_data = chat_total > 0
chat_aht = chat_cases["Total_Secs"].mean() if chat_total else None
email_aht = email_cases["Total_Secs"].mean() if email_total else None

# Re-work: how much of the volume is a case coming back round again.
chat_touches = float(chat_cases["Touches"].mean()) if chat_total else None
email_touches = float(email_cases["Touches"].mean()) if email_total else None
# Legacy Live Chat has no reopen flag: this is only "handled more than once".
chat_multi_touch = int((chat_cases["Touches"] > 1).sum())
# Email reopening comes ONLY from the authoritative `Re-opened` flag (joined on case number),
# never from routing touches. Without the flag (legacy export) it is 0 and not displayed.
if email_ingest.has_reopen_flag and HAS_CASE_KEY and CASE_KEY in email_cases.columns:
    _reopened_keys = set(email_sla_df.loc[email_sla_df["Reopened"], EMAIL_CASE_KEY])
    email_reopened = int(email_cases[CASE_KEY].astype(str).isin(_reopened_keys).sum())
else:
    email_reopened = 0
# Routing touches > 1 is a handling diagnostic (Change 4), distinct from reopening.
email_multi_touch = int((email_cases["Touches"] > 1).sum())
email_multi_agent = int((email_cases["Agents"] > 1).sum())
chat_open_n = int(chat_cases["Any_Open"].sum())
email_open_n = int(email_cases["Any_Open"].sum())

# Per-touch AHT, kept so the two definitions can be compared.
chat_aht_touch = (chat_cases["Total_Secs"].sum() / chat_cases["Touches"].sum()) if chat_total else None
email_aht_touch = (email_cases["Total_Secs"].sum() / email_cases["Touches"].sum()) if email_total else None

chat_sla_p = chat_sla_df[
    (chat_sla_df["Date/Time Opened"] >= ts_start) & (chat_sla_df["Date/Time Opened"] < ts_end)
    & (chat_sla_df["Date/Time Opened"] < pd.Timestamp(MESSAGING_SLA_START_DATE))
].copy()
# ---------------------------------------------------------------------------
# Overall Service Attainment inputs (Change 1; metrics/service_attainment.py).
# Email cohort = Date/Time Opened (queue entry); Messaging cohort = session Start Time.
# ---------------------------------------------------------------------------
email_att_ok = email_ingest.can_compute_attainment
email_att_all = classify_email(email_sla_df, DATA_ASOF) if email_att_ok else None
messaging_att_ok = MESSAGING_RESPONSE_COL in messaging_sessions.columns
messaging_att_all = (
    classify_messaging(messaging_sessions[
        messaging_sessions["Start Time"] >= pd.Timestamp(MESSAGING_SLA_START_DATE)])
    if messaging_att_ok else None
)


def attainment_for_window(w_start, w_end):
    """(email outcomes, messaging outcomes, OverallSummary) for contacts entering in [w_start, w_end)."""
    e = in_window(email_att_all, "Date/Time Opened", w_start, w_end) if email_att_ok else None
    m = in_window(messaging_att_all, "Start Time", w_start, w_end) if messaging_att_ok else None
    return e, m, build_overall(e, m, window_end=w_end, as_of=DATA_ASOF,
                               email_available=email_att_ok, messaging_available=messaging_att_ok)


email_att_p, messaging_att_p, overall_p = attainment_for_window(ts_start, ts_end)
_period_len = ts_end - ts_start
_, _, overall_prev = attainment_for_window(ts_start - _period_len, ts_start)

# Arrivals-only slice, for the contact-volume heatmap (that chart is about when
# contacts ARRIVE).
email_opened_p = email_sla_df[
    (email_sla_df["Date/Time Opened"] >= ts_start) & (email_sla_df["Date/Time Opened"] < ts_end)
].copy()

# Mean first-response time (business hours) of answered emails in the cohort.
_answered_bh = (email_att_p["Business Hours To Response"].dropna()
                if email_att_p is not None and "Business Hours To Response" in email_att_p.columns
                else pd.Series(dtype="float64"))
_answered_bh = _answered_bh[_answered_bh >= 0]
n_email_total = len(_answered_bh)
avg_resp_secs = float(_answered_bh.mean() * 3600) if n_email_total else None


# =============================================================================
# Availability & utilisation
# FIX #1 / #28 / #29: capacity-based utilisation. Chat uses all chat-eligible
# availability and two concurrent slots. Email uses full email-only capacity
# plus the residual capacity left by chats during shared 'Available_All' time.
# This avoids both numerator-dependent proportional allocation and double-
# counting shared capacity as fully available to both channels.
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

# Handling intervals, built from the OVERLAP slice (FIX #21). Conversation
# intervals (legacy Chat plus Enhanced Messaging) remain unmerged so concurrent
# conversations consume separate capacity slots; email intervals are merged
# because email capacity is measured as occupied wall-clock time.
chat_handles_map, email_handles_map = {}, {}
for devname, target in ((CHAT_DEVNAME, chat_handles_map), (EMAIL_DEVNAME, email_handles_map)):
    sub = df_overlap[df_overlap["Service Channel: Developer Name"] == devname]
    if devname == CHAT_DEVNAME:
        sub = sub[sub["Start DT"] < pd.Timestamp(MESSAGING_SLA_START_DATE)]
    for ag, grp in sub.groupby("User: Full Name"):
        ints = [clip_to_window(s, e, window_start, window_end)
                for s, e in zip(grp["Start DT"], grp["End DT Clamped"])]
        ints = [x for x in ints if x]
        if ints:
            target[ag] = ints if devname == CHAT_DEVNAME else merge_intervals(ints)

# Enhanced Messaging shares the same Available_Chat / Available_All pool and
# two-slot capacity as legacy Chat. Add accepted Messaging intervals to the
# conversation workload before utilization and residual email capacity are
# calculated.
_messaging_overlap = messaging_items_df[
    (messaging_items_df["Start DT"] < ts_end)
    & (messaging_items_df["End DT Clamped"] > ts_start)
]
for ag, grp in _messaging_overlap.groupby("User: Full Name"):
    ints = [clip_to_window(s, e, window_start, window_end)
            for s, e in zip(grp["Start DT"], grp["End DT Clamped"])]
    ints = [x for x in ints if x]
    if ints:
        chat_handles_map.setdefault(ag, []).extend(ints)

dept_chat_load = dept_email_load = dept_email_excess = 0.0
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

    chat_slot_secs = concurrent_slot_seconds(
        chat_handles_map.get(ag, []), chat_av_union, CHAT_CONCURRENCY_CAPACITY
    )
    # Convert slot-seconds to agent-equivalent load. At capacity two, one
    # active chat for an hour is 0.5 load-hours and two are 1.0 load-hours.
    chat_load = chat_slot_secs / CHAT_CONCURRENCY_CAPACITY
    email_intervals = email_handles_map.get(ag, [])

    # Chat uses the complete chat-eligible availability. Email uses full
    # email-only capacity plus shared status only while no chat is active.
    chat_av = sum_secs(chat_av_union)
    email_only_av = sum_secs(eo)
    email_only_load = intersect_sum(email_intervals, eo) if eo else 0.0
    email_shared_av, email_shared_load, email_excess = residual_capacity_usage(
        chat_handles_map.get(ag, []), email_intervals, sh,
        CHAT_CONCURRENCY_CAPACITY,
    )
    email_av = email_only_av + email_shared_av
    email_load = email_only_load + email_shared_load

    dept_chat_load += chat_load
    dept_email_load += email_load
    dept_email_excess += email_excess
    dept_chat_avail += chat_av
    dept_email_avail += email_av

    util_rows.append({
        "Agent": ag,
        "Conversation load (h)": chat_load / 3600,
        "Conversation slot-hours": chat_slot_secs / 3600,
        "Conversation available (h)": chat_av / 3600,
        "Conversation util": (chat_load / chat_av) if chat_av else None,
        "Email load (h)": email_load / 3600,
        "Email residual capacity (h)": email_av / 3600,
        "Email excess demand (h)": email_excess / 3600,
        "Email util": (email_load / email_av) if email_av else None,
        "_avail_secs": sum_secs(merge_intervals(co + eo + sh)),
        "_key": _norm_person_key(ag),
    })

chat_util = (dept_chat_load / dept_chat_avail) if dept_chat_avail else None
email_util = (dept_email_load / dept_email_avail) if dept_email_avail else None

df_util = pd.DataFrame(util_rows)

# ---------------------------------------------------------------------------
# Schedule adherence (Change 2) - metrics/adherence.py. Department level only.
# Verified active Email / Messaging handling is productive and overrides presence.
# ---------------------------------------------------------------------------
_handling_parts = [
    df_items.loc[df_items["Service Channel: Developer Name"] == EMAIL_DEVNAME,
                 ["User: Full Name", "Start DT", "End DT Clamped"]],
    messaging_items_df[["User: Full Name", "Start DT", "End DT Clamped"]],
]
_handling = pd.concat(_handling_parts, ignore_index=True).rename(columns={
    "User: Full Name": "Agent", "Start DT": "Start", "End DT Clamped": "End"})
adherence_sum, adherence_daily = compute_adherence(
    df_shifts, df_presence, _handling, ts_start, ts_end, DATA_ASOF)


# Period-level components
_cw_p = chat_sla_p[chat_sla_p["Wait Time"].notna()]
n_chat_total = len(chat_sla_p)
n_chat_answered = len(_cw_p)
n_chat_in_target = int((_cw_p["Wait Time"] <= CHAT_ANSWER_TARGET_SEC).sum())
n_chat_abandoned = int((chat_sla_p["Abandoned After"] > CHAT_ABANDON_AFTER_SEC).sum())

sla_comp_chat_in_target_pct = (n_chat_in_target / n_chat_answered * 100) if n_chat_answered else None
sla_comp_chat_avg_wait_secs = _cw_p["Wait Time"].mean() if n_chat_answered else None
sla_comp_chat_abandon_pct = (n_chat_abandoned / n_chat_total * 100) if n_chat_total else None

# =============================================================================
# Header & KPI tiles
# =============================================================================
st.markdown(
    '<div class="glass-hero">'
    f'<div class="gh-top"><div class="gt-icon">{_icon_svg("chart")}</div>'
    f'<div class="gh-arrow">{_icon_svg("arrow", 22)}</div></div>'
    '<div class="gh-kicker">Customer Service</div>'
    '<div class="gh-title">Department Performance Dashboard</div>'
    f'<div class="gh-bottom"><div class="gh-period">{start_date:%d %b %Y} — {end_date:%d %b %Y}</div>'
    '<div class="gh-track" aria-hidden="true"><i></i><i class="mid"></i><i></i><i></i>'
    '<i class="end"></i></div></div></div>',
    unsafe_allow_html=True,
)

st.subheader(":material/dashboard: Core Metrics")
if has_chat_data:
    c1, c2, c3, c4 = st.columns(4)
    render_custom_metric(c1, "💬 Total Chats", f"{chat_total:,}",
                         "Distinct chat cases first handled in the period. Counted once even if "
                         "the case was routed more than once.",
                         COLOR_NONE,
                         f"{chat_multi_touch:,} handled more than once" if chat_multi_touch else "")
    render_custom_metric(c2, "✉️ Total Emails", f"{email_total:,}",
                         "Distinct email cases first handled in the period. A case that is "
                         "re-opened and routed again counts ONCE, not once per touch.",
                         COLOR_NONE,
                         f"{email_reopened:,} re-opened" if email_reopened else "")
    render_custom_metric(c3, "⏳ Avg Chat Handle Time", fmt_mmss(chat_aht),
                         "Total handle time per chat case, summed across every touch",
                         COLOR_NONE,
                         f"{chat_touches:.2f} touches/case" if chat_touches else "")
    render_custom_metric(c4, "⏳ Avg Email Handle Time", fmt_mmss(email_aht),
                         "Total handle time per email case, summed across EVERY touch — including "
                         "re-opens handled later and by different agents",
                         COLOR_NONE,
                         f"{email_touches:.2f} touches/case" if email_touches else "")
else:
    _ce1, _ce2 = st.columns(2)
    render_custom_metric(_ce1, "✉️ Total Emails", f"{email_total:,}",
                         "Distinct email cases first handled in the period. A case that is "
                         "re-opened and routed again counts ONCE, not once per touch.",
                         COLOR_NONE,
                         f"{email_reopened:,} re-opened" if email_reopened else "")
    render_custom_metric(_ce2, "⏳ Avg Email Handle Time", fmt_mmss(email_aht),
                         "Total handle time per email case, summed across EVERY touch — including "
                         "re-opens handled later and by different agents",
                         COLOR_NONE,
                         f"{email_touches:.2f} touches/case" if email_touches else "")

st.markdown("#### :material/forum: Enhanced Messaging")
_m1, _m2, _m3, _m4 = st.columns(4)
render_custom_metric(
    _m1, "Messaging Sessions", f"{messaging_total:,}",
    "Unique Messaging sessions started in the selected period. Counted once even when "
    "routing, cancellation or transfer creates several Agent Work rows.",
    COLOR_NONE,
    f"{messaging_assignments:,} routing assignments",
)
render_custom_metric(
    _m2, "Sessions Handled", f"{messaging_handled:,}",
    "Sessions with at least one closed human Agent Work assignment.",
    COLOR_NONE,
    f"{messaging_abandoned:,} unhandled/abandoned"
    + (f" • {messaging_unclassified:,} unclassified" if messaging_unclassified else ""),
)
render_custom_metric(
    _m3, "Messaging AHT", fmt_mmss(messaging_aht),
    "Mean total Agent Work handle time per handled Messaging session. Multiple assignments "
    "for the same session are summed before averaging.",
    COLOR_NONE,
    "Agent Work Handle Time",
)
render_custom_metric(
    _m4, "Transferred Sessions", f"{messaging_transfers:,}",
    "Sessions handled by more than one human agent.",
    COLOR_NONE,
    f"{(messaging_transfers / messaging_handled):.1%} of handled"
    if messaging_handled else "",
)

_mr1, _mr2, _mr3, _mr4 = st.columns(4)
render_custom_metric(
    _mr1, "First Response ≤60s", fmt_pct(messaging_first_response_target_pct),
    "Share of sessions with a positive Service Rep First Response Time at or below 60 seconds. "
    "Zero means no measurable agent response and is excluded from this denominator.",
    COLOR_NONE,
    f"{messaging_first_response_in_target:,} of {len(_messaging_first_response):,} measurable",
)
render_custom_metric(
    _mr2, "Median First Response", fmt_mmss(messaging_first_response_median),
    "Median route-to-first-agent-response time. Median is used because asynchronous sessions "
    "create a substantial long tail.",
    COLOR_NONE,
    f"mean {fmt_mmss(messaging_first_response_mean)}",
)
render_custom_metric(
    _mr3, "Avg Speed to Answer", fmt_mmss(messaging_speed_to_answer),
    "Mean routing-request-to-acceptance time from Agent Work. This is different from the "
    "time until the first written response.",
    COLOR_NONE,
    f"median post-accept response {fmt_mmss(messaging_accept_to_response_median)}",
)
render_custom_metric(
    _mr4, "Messages Sent", f"{messaging_agent_messages:,}",
    "Total agent messages sent in sessions started in the selected period.",
    COLOR_NONE,
    f"{messaging_customer_messages:,} customer messages",
)

with st.expander("Daily Enhanced Messaging detail"):
    _messaging_daily_rows = []
    for _d, _g in messaging_period.groupby(messaging_period["Start Time"].dt.date):
        _fr = _positive_metric(_g, "Service Rep First Response Time")
        _handled = _g[_g["Any_Closed"]]
        _aht = pd.to_numeric(_handled["Total_Handle_Secs"], errors="coerce").dropna()
        _messaging_daily_rows.append({
            "Date": pd.Timestamp(_d),
            "Sessions": len(_g),
            "Handled": int(_g["Any_Closed"].sum()),
            "Abandoned": int(((~_g["Any_Closed"]) & _g["Any_Abandoned"]).sum()),
            "Transfers": int((_g["Human_Agents"] > 1).sum()),
            "AHT (s)": _aht.mean() if len(_aht) else None,
            "First response ≤60s %":
                float((_fr <= MESSAGING_FIRST_RESPONSE_TARGET_SEC).mean() * 100)
                if len(_fr) else None,
            "Median first response (s)": _fr.median() if len(_fr) else None,
            "Agent messages": int(pd.to_numeric(
                _g.get("Agent Message Count", pd.Series(0, index=_g.index)),
                errors="coerce").fillna(0).sum()),
            "Customer messages": int(pd.to_numeric(
                _g.get("End User Message Count", pd.Series(0, index=_g.index)),
                errors="coerce").fillna(0).sum()),
        })
    _messaging_daily = pd.DataFrame(_messaging_daily_rows)
    if _messaging_daily.empty:
        st.info("No completed Messaging sessions started in this period.")
    else:
        st.dataframe(
            _messaging_daily.sort_values("Date").style.format({
                "Date": lambda x: x.strftime("%d %b %Y"),
                "AHT (s)": "{:,.1f}",
                "First response ≤60s %": "{:.1f}",
                "Median first response (s)": "{:,.1f}",
            }, na_rep="—"),
            width="stretch",
        )

# Re-work is invisible when volume counts rows; surfaced explicitly instead.
if email_total and email_multi_touch:
    _single = email_cases.loc[email_cases["Touches"] == 1, "Total_Secs"].mean()
    _multi = email_cases.loc[email_cases["Touches"] > 1, "Total_Secs"].mean()
    st.caption(
        f":material/repeat: **{email_multi_touch:,} of {email_total:,} email cases ({email_multi_touch/email_total:.0%}) "
        f"were handled more than once**, {email_multi_agent:,} of them by more than one agent. "
        f"Those cases average **{fmt_mmss(_multi)}** of total handle time versus "
        f"**{fmt_mmss(_single)}** for cases resolved in a single touch. "
        f"Counting routing rows instead of cases would report "
        f"{int(email_cases['Touches'].sum()):,} emails instead of {email_total:,}."
    )

st.markdown("---")
st.subheader(":material/speed: Workload & Efficiency Diagnostics")
st.caption(
    ":material/info: **Diagnostics only.** These describe how work flows through the department; "
    "none has a target or a pass/fail status, and lower or higher is not treated as good or bad. "
    "Department totals only - there is no agent-level view.")

_chat_util_tooltip = (
    f"Capacity-adjusted legacy Chat plus Enhanced Messaging load divided by all "
    f"conversation-eligible availability. Each agent has {CHAT_CONCURRENCY_CAPACITY} slots: "
    f"one active conversation uses "
    f"{1 / CHAT_CONCURRENCY_CAPACITY:.0%} capacity and "
    f"{CHAT_CONCURRENCY_CAPACITY} simultaneous conversations use 100%. 'Available_Chat' "
    "and 'Available_All' are eligible; residual shared capacity is used for the email KPI."
)
_email_util_tooltip = (
    "Email load divided by assignable email capacity. Email-only presence supplies "
    "100% email capacity. During 'Available_All', email capacity is counted only while "
    "the agent has no active Messaging/chat session; any active session removes Email capacity."
)

_util_lines = [UtilisationLine("Conversation", dept_chat_load / 3600, dept_chat_avail / 3600),
               UtilisationLine("Email", dept_email_load / 3600, dept_email_avail / 3600)]
_op = st.columns(3)
render_custom_metric(_op[0], "Conversation utilisation", fmt_pct_floor(_util_lines[0].rate_pct),
                     _chat_util_tooltip, COLOR_NONE,
                     f"{_util_lines[0].load_hours:,.1f} load-h / {_util_lines[0].capacity_hours:,.1f} h available")
render_custom_metric(_op[1], "Email utilisation", fmt_pct_floor(_util_lines[1].rate_pct),
                     _email_util_tooltip, COLOR_NONE,
                     f"{_util_lines[1].load_hours:,.1f} load-h / {_util_lines[1].capacity_hours:,.1f} h residual")
render_custom_metric(_op[2], "Avg Email First Response", fmt_hms(avg_resp_secs),
                     "Mean first-response time in BUSINESS time (07:00-22:00 CAT, 7 days) of "
                     "answered emails that entered the queue in the period",
                     COLOR_NONE, f"across {n_email_total:,} answered emails" if n_email_total else "")
if dept_email_excess > 0:
    st.caption(
        f"⚠️ {dept_email_excess/3600:,.1f} email-hours were recorded while conversation load "
        "made shared status unavailable for email. These hours are shown as excess "
        "demand and are not allowed to push email utilisation above 100%.")

_eh = email_handling(email_cases, email_df["Row Secs"] if "Row Secs" in email_df.columns
                     else pd.Series(dtype="float64"))
_md = messaging_diagnostics(messaging_period)


def _secs(v):
    return fmt_mmss(v) if v is not None else "—"


_wl_a, _wl_b = st.columns(2)
with _wl_a:
    st.markdown("**Email handling**")
    st.dataframe(pd.DataFrame([
        {"Measure": "Handle time per case (all touches)", "Cases / touches": _eh.per_case.n,
         "Mean": _secs(_eh.per_case.mean), "Median": _secs(_eh.per_case.median),
         "p90": _secs(_eh.per_case.p90)},
        {"Measure": "Handle time per touch", "Cases / touches": _eh.per_touch.n,
         "Mean": _secs(_eh.per_touch.mean), "Median": _secs(_eh.per_touch.median),
         "p90": _secs(_eh.per_touch.p90)},
    ]).style.format({"Cases / touches": "{:,}"}), width="stretch", hide_index=True)
    st.caption(
        f"Touches per case: mean {_eh.touches_per_case.mean:.2f}, median {_eh.touches_per_case.median:.0f}, "
        f"p90 {_eh.touches_per_case.p90:.0f}. "
        f"{fmt_ratio(_eh.multi_touch_cases, _eh.cases)} of cases were handled more than once "
        "(a routing/handling measure; it is not reopening, which comes only from the `Re-opened` flag)."
        if _eh.cases else "No Email cases in this period.")
with _wl_b:
    st.markdown("**Enhanced Messaging**")
    st.dataframe(pd.DataFrame([
        {"Measure": "Handle time per handled session", "Sessions": _md.handling_per_session.n,
         "Mean": _secs(_md.handling_per_session.mean), "Median": _secs(_md.handling_per_session.median),
         "p90": _secs(_md.handling_per_session.p90)},
        {"Measure": "Routing: assign -> accept", "Sessions": _md.routing_assign_to_accept.n,
         "Mean": _secs(_md.routing_assign_to_accept.mean),
         "Median": _secs(_md.routing_assign_to_accept.median),
         "p90": _secs(_md.routing_assign_to_accept.p90)},
        {"Measure": "Speed to answer (Agent Work)", "Sessions": _md.speed_to_answer.n,
         "Mean": _secs(_md.speed_to_answer.mean), "Median": _secs(_md.speed_to_answer.median),
         "p90": _secs(_md.speed_to_answer.p90)},
    ]).style.format({"Sessions": "{:,}"}), width="stretch", hide_index=True)

_v1, _v2, _v3, _v4 = st.columns(4)
render_custom_metric(_v1, "Email cases", f"{email_total:,}",
                     "Distinct Email cases first handled in the period (counted once however many "
                     "times they were routed).", COLOR_NONE,
                     f"{_eh.multi_touch_cases:,} handled more than once" if _eh.cases else "")
render_custom_metric(_v2, "Messaging sessions", f"{_md.sessions:,}",
                     "Unique Messaging sessions started in the period.", COLOR_NONE,
                     f"{_md.handled:,} handled by a human")
render_custom_metric(_v3, "Messages", f"{int(_md.agent_messages.total):,} agent",
                     "Agent and customer messages in sessions started in the period.", COLOR_NONE,
                     f"{int(_md.customer_messages.total):,} customer • median "
                     f"{_md.agent_messages.median or 0:.0f} / {_md.customer_messages.median or 0:.0f} per session")
render_custom_metric(_v4, "Transferred sessions", f"{_md.transferred:,}",
                     "Handled Messaging sessions worked by more than one human agent.", COLOR_NONE,
                     fmt_ratio(_md.transferred, _md.handled) + " of handled" if _md.handled else "")
if has_chat_data:
    st.caption(f"Historical Live Chat (before 1 Nov 2026): {chat_total:,} chats, average handle time "
               f"{fmt_mmss(chat_aht)}, {chat_multi_touch:,} handled more than once.")

# ---- Schedule Adherence (Change 2) ---------------------------------------
# Department-level only (no per-agent view). See metrics/adherence.py for every rule.
st.markdown("---")
st.subheader(":material/event_available: Schedule Adherence")
if not shifts_path.exists():
    st.info("shifts.csv not found - Schedule Adherence is unavailable.")
elif df_shifts.empty:
    st.info("shifts.csv contained no usable shifts - Schedule Adherence is unavailable.")
else:
    _adh = adherence_sum
    _adh_color = {"green": COLOR_GOOD, "amber": COLOR_WARN, "red": COLOR_BAD,
                  "grey": COLOR_NONE}[_adh.status]
    _adh_text = {"green": "On target", "amber": "Near target", "red": "Below target",
                 "grey": "Not reliable"}[_adh.status]
    _ad1, _ad2, _ad3 = st.columns(3)
    render_custom_metric(
        _ad1, "Schedule Adherence", fmt_pct_floor(_adh.rate_pct),
        f"Productive scheduled minutes / expected productive minutes, time-weighted across "
        f"agents, only inside scheduled shifts, capped at 100%. Target {ADH_TARGET_PCT:.0f}%. "
        f"Green >= 95, Amber 90-94.9, Red < 90, Grey when coverage < {ADH_COVERAGE_MIN_PCT:.0f}%.",
        _adh_color,
        f"{fmt_ratio(int(round(_adh.productive_secs / 60)), int(round(_adh.expected_secs / 60)))} min"
        f" • {_adh_text}")
    render_custom_metric(
        _ad2, "Coverage", fmt_pct_floor(_adh.coverage_pct),
        "Scheduled shift time belonging to people who can be matched to presence records. "
        "Shifts with no presence data at all are a coverage gap, not scored as absence.",
        COLOR_GOOD if (_adh.coverage_pct or 0) >= ADH_COVERAGE_MIN_PCT else COLOR_NONE,
        f"{_adh.shifts:,} shifts scored • {_adh.unmatched_shifts:,} unmatched")
    render_custom_metric(
        _ad3, "Late / early incidents", f"{_adh.late_incidents + _adh.early_incidents:,}",
        "A late start or early finish counts only when the variance exceeds 5 minutes; the full "
        "variance is recorded.",
        COLOR_NONE,
        f"late {_adh.late_incidents:,} ({_adh.late_variance_secs / 60:,.0f} min) • "
        f"early {_adh.early_incidents:,} ({_adh.early_variance_secs / 60:,.0f} min)")
    if _adh.pending_shifts:
        st.info(f"**Provisional** - {_adh.pending_shifts:,} shift(s) had not finished by the data "
                f"horizon ({DATA_ASOF:%d %b %H:%M}) and are not yet scored.")
    for _m in _adh.messages:
        st.caption(f":material/warning: {_m}")
    with st.expander(":material/calculate: How adherence reconciles", expanded=True):
        st.dataframe(pd.DataFrame([
            {"Measure": "Expected productive time (h)", "Value": _adh.expected_secs / 3600},
            {"Measure": "Productive time within shifts (h)", "Value": _adh.productive_secs / 3600},
            {"Measure": "Approved exclusions removed from expected (meetings / other tasks) (h)",
             "Value": _adh.approved_exclusion_secs / 3600},
            {"Measure": "Lunch/break allowance removed from expected (<=60 min/shift) (h)",
             "Value": _adh.break_allowed_secs / 3600},
            {"Measure": "Lunch/break beyond the 60-min cap (non-adherent) (h)",
             "Value": _adh.break_over_cap_secs / 3600},
            {"Measure": "Shifts with zero productive time (absent)", "Value": _adh.absent_shifts},
            {"Measure": "Leave shifts (excluded)", "Value": _adh.leave_shifts},
            {"Measure": "Sick shifts (excluded)", "Value": _adh.sick_shifts},
        ]).style.format({"Value": "{:,.1f}"}), width="stretch", hide_index=True)
        st.caption("Productive = Available_All / Available_Email_and_Web / Available_Chat / "
                   "Available_Case, plus verified Email/Messaging handling. Lunch and break share "
                   "one 60-minute allowance per shift; an unused allowance is not deducted.")
    if len(adherence_daily):
        with st.expander("Daily adherence"):
            _ad = adherence_daily.copy()
            _ad["Date"] = _ad["Date"].dt.strftime("%a %d %b")
            st.dataframe(_ad.style.format({"Productive (h)": "{:,.1f}", "Expected (h)": "{:,.1f}",
                                           "Adherence %": "{:.1f}"}, na_rep="—"),
                         width="stretch", hide_index=True)


# ---- Overall Service Attainment (Change 1) ------------------------------
# Replaces the old "SLA Score Summary / Weighted SLA". Overall =
# (successful eligible Email + successful eligible Messaging) / combined mature eligible.
# Every % below is shown beside its numerator and denominator.
st.markdown("---")
st.subheader(":material/adjust: Overall Service Attainment")

_ATT_COLOR = {"green": COLOR_GOOD, "amber": COLOR_WARN, "red": COLOR_BAD, "grey": COLOR_NONE}
_ATT_TEXT = {"green": "On target", "amber": "Near target", "red": "Below target",
             "grey": "Not reliable"}

for _msg in email_ingest.messages:
    st.warning(_msg)
if not messaging_att_ok:
    st.warning(f"Messaging metric `{MESSAGING_RESPONSE_COL}` is missing from the Messaging "
               "metrics export - Messaging (and therefore Overall) attainment is unavailable.")


def _att_tile(container, title, summ, tooltip, provisional=False):
    rate = fmt_pct_floor(summ.rate_pct)
    sub = f"{fmt_ratio(summ.numerator, summ.denominator)} • {_ATT_TEXT[summ.status]}"
    if provisional:
        sub += " • PROVISIONAL"
    render_custom_metric(container, title, rate, tooltip, _ATT_COLOR[summ.status], sub)


_a1, _a2, _a3 = st.columns(3)
_att_tile(
    _a1, "Overall Service Attainment", overall_p,
    f"(Successful eligible Email + successful eligible Messaging) / combined mature eligible "
    f"contacts. Target {TARGET_PCT:.0f}%. Green >= 80, Amber 70-79.9, Red < 70, Grey when "
    f"coverage < {COVERAGE_MIN_PCT:.0f}% or nothing is mature. Overall alone decides pass/fail.",
    provisional=overall_p.provisional)
_att_tile(
    _a2, "Email first response", overall_p.email,
    f"Replied within {EMAIL_TARGET_BUSINESS_HOURS:.0f} business hours of queue entry "
    "(07:00-22:00 CAT, 7 days/week), recomputed from Date/Time Opened -> Completion Date. "
    "Overdue unanswered = fail; not yet due = pending and excluded. Informational: does not veto Overall. "
    "Counts cases that ENTERED THE QUEUE in the period (email.csv), which differs from Core Metrics "
    "'Total Emails' (cases first handled by an agent): see the reconciliation below.",
    provisional=overall_p.email.pending > 0)
_att_tile(
    _a3, "Messaging complete service", overall_p.messaging,
    f"Handled by a human AND first human response within {MESSAGING_TARGET_SEC:.0f}s of "
    "acceptance. Abandoned = fail. Canceled, unavailable, bot-only, test and unclassifiable "
    "sessions are excluded (counts below). Informational: does not veto Overall.")

_pp = overall_p.pp_vs(overall_prev)
_cap = [f"Target {TARGET_PCT:.0f}%."]
if _pp is not None:
    _cap.append(f"{_pp:+.1f} percentage points vs the previous equivalent period "
                f"({fmt_ratio(overall_prev.numerator, overall_prev.denominator)}).")
else:
    _cap.append("No comparable previous period.")
_cap.append(f"Data as of {DATA_ASOF:%d %b %Y %H:%M}.")
st.caption(" ".join(_cap))
if overall_p.provisional:
    st.info("**Provisional** - " + " ".join(overall_p.reasons))
elif overall_p.reasons:
    st.warning(" ".join(overall_p.reasons))
if start_date < MESSAGING_SLA_START_DATE:
    st.caption(
        f"Days before {MESSAGING_SLA_START_DATE:%d %b %Y} predate Enhanced Messaging: Live Chat "
        "is never part of current scorecards, so those days are Email-only (see the historical "
        "Live Chat section below).")


def _att_rows():
    rows = []
    for _s in (overall_p.email, overall_p.messaging):
        rows.append({
            "Channel": _s.name, "Successful": _s.numerator, "Mature eligible": _s.denominator,
            "Attainment": fmt_pct_floor(_s.rate_pct), "Pending (not yet due)": _s.pending,
            "Unmeasurable": _s.unmeasurable, "Excluded": _s.excluded,
            "Coverage": fmt_pct_floor(_s.coverage_pct), "Status": _ATT_TEXT[_s.status],
        })
    rows.append({
        "Channel": "Overall", "Successful": overall_p.numerator,
        "Mature eligible": overall_p.denominator,
        "Attainment": fmt_pct_floor(overall_p.rate_pct),
        "Pending (not yet due)": overall_p.pending, "Unmeasurable": overall_p.unmeasurable,
        "Excluded": overall_p.messaging.excluded + overall_p.email.excluded,
        "Coverage": fmt_pct_floor(overall_p.coverage_pct), "Status": _ATT_TEXT[overall_p.status],
    })
    return pd.DataFrame(rows)


with st.expander(":material/calculate: How the headline reconciles (numerators, denominators, exclusions)",
                 expanded=True):
    st.dataframe(_att_rows().style.format(
        {c: "{:,}" for c in ("Successful", "Mature eligible", "Pending (not yet due)",
                             "Unmeasurable", "Excluded")}),
        width="stretch", hide_index=True)
    st.caption(
        "Attainment = Successful / Mature eligible. Coverage = Mature eligible / (Mature eligible + "
        "Unmeasurable). Pending Email is not in either denominator. Percentages are truncated, "
        "not rounded, so a value never crosses a status band on display.")
    _b1, _b2 = st.columns(2)
    for _col, _s in ((_b1, overall_p.email), (_b2, overall_p.messaging)):
        with _col:
            st.markdown(f"**{_s.name} - every outcome**")
            _bd = _s.breakdown()
            if _bd.empty:
                st.caption("No contacts in this period.")
            else:
                st.dataframe(_bd.style.format({"Contacts": "{:,}"}), width="stretch",
                             hide_index=True)

# Why Core Metrics "Total Emails" and the Email attainment cohort differ (they count different things).
if email_att_ok and HAS_CASE_KEY and email_att_p is not None:
    _routed_all = df_cases.loc[df_cases["Service Channel: Developer Name"] == EMAIL_DEVNAME, CASE_KEY]
    _flag_keys = (email_sla_df.loc[email_sla_df["Reopened"], EMAIL_CASE_KEY]
                  if email_ingest.has_reopen_flag else [])
    _rc = email_cohort_reconciliation(
        email_att_p, email_sla_df[EMAIL_CASE_KEY], _routed_all, email_cases[CASE_KEY], _flag_keys,
        key_col=EMAIL_CASE_KEY)
    with st.expander(":material/compare_arrows: Why Total Emails (Core Metrics) differs from the "
                     "Email attainment cohort", expanded=False):
        st.markdown(
            f"**Core Metrics - Total Emails = {_rc['core_cases']:,}**: distinct cases an agent *first "
            f"handled* in the period (`report_items.csv`).  \n"
            f"**Attainment Email cohort = {_rc['cohort_cases']:,}**: distinct cases that *entered the "
            f"queue* in the period (`email.csv` Date/Time Opened).")
        st.dataframe(pd.DataFrame([
            {"Measure": "Attainment cohort: cases opened in the period (distinct)",
             "Cases": _rc["cohort_cases"]},
            {"Measure": "   rows in that cohort (equals cases: no double counting)",
             "Cases": _rc["cohort_rows"]},
            {"Measure": "   ...routed to an agent (present in report_items)", "Cases": _rc["cohort_routed"]},
            {"Measure": "   ...never routed to an agent (email export only)",
             "Cases": _rc["cohort_never_routed"]},
            {"Measure": "Core Metrics cases first handled in the period", "Cases": _rc["core_cases"]},
            {"Measure": "   ...also opened in the period", "Cases": _rc["core_in_cohort"]},
            {"Measure": "   ...opened BEFORE the period (carried in)",
             "Cases": _rc["core_opened_before_window"]},
            {"Measure": "   ...not in the email export at all", "Cases": _rc["core_missing_from_export"]},
            {"Measure": "Reopened: Core Metrics cases flagged `Re-opened`", "Cases": _rc["core_reopened"]},
            {"Measure": "Reopened: attainment cohort cases flagged `Re-opened`",
             "Cases": _rc["cohort_reopened"]},
        ]).style.format({"Cases": "{:,}"}), width="stretch", hide_index=True)
        _ra, _ru = _rc["routed_summary"], _rc["unrouted_summary"]
        st.markdown(
            f"**Attainment of routed cases:** {fmt_ratio(_ra.numerator, _ra.denominator)}  \n"
            f"**Attainment of never-routed cases:** {fmt_ratio(_ru.numerator, _ru.denominator)}  \n"
            "If never-routed cases score very differently, they (e.g. automated or system-handled "
            "mail) are moving the headline; check with the data owner whether they belong in scope.")
        st.caption(
            "Counts are distinct cases (the export is de-duplicated per Case Number before "
            "scoring). Email attainment is also less than the headline's eligible count because "
            "the headline adds Messaging sessions.")

# Reopened handling: authoritative `Re-opened` flag; timing gated on `Reopened Time`.
if email_att_ok and email_ingest.has_reopen_flag:
    _ro = reopened_summary(email_att_p)
    if _ro["reopened_den"]:
        st.caption(
            f":material/repeat: **{_ro['reopened_den']:,}** reopened Email(s) are inside the mature "
            f"eligible set ({fmt_ratio(_ro['reopened_num'], _ro['reopened_den'])} successful). "
            "For reopened cases `Completion Date` is the FINAL response, so their first-response "
            "result is approximate; they are flagged, not excluded.")
    if not reopen_timing_available(email_sla_df):
        st.caption(":material/schedule: Reopened-resolution clock / ageing: **awaiting data** "
                   "(reopen timestamps exist only for reopens from 19 Nov 2026).")

with st.expander(":material/info: Known limitations of this measure"):
    st.markdown(
        "- **Historical reopen timing is empty and unrecoverable.** The 12-hour reopened clock and "
        "reopened ageing fill in weekly from 19 Nov 2026.\n"
        "- **Reopened cases lack an initial first-response time** (`Completion Date` is the final "
        "response), so Email first-response is approximate for the reopened share.\n"
        "- Timestamps are treated as Central Africa Time (UTC+2); set `SOURCE_UTC_OFFSET_HOURS` "
        "if the export is in another zone.\n"
        "- Messaging sessions routed but never accepted with no abandon event are treated as "
        "**unmeasurable** (excluded, lowering coverage) pending a business decision.\n"
        "- Survey response rate and CSAT channel split are unavailable in the data.")

if len(chat_sla_p):
    with st.expander(":material/history: Historical Live Chat (before 1 Nov 2026 - not part of Overall)"):
        st.markdown(
            f"- Answered ≤{CHAT_ANSWER_TARGET_SEC:.0f}s: **{fmt_ratio(n_chat_in_target, n_chat_answered)}**\n"
            f"- Abandoned after >{CHAT_ABANDON_AFTER_SEC:.0f}s: "
            f"**{fmt_ratio(n_chat_abandoned, n_chat_total)}**\n"
            f"- Average wait: **{fmt_mmss(sla_comp_chat_avg_wait_secs)}**")

# ---- Trend ---------------------------------------------------------------
st.markdown("---")
st.subheader(":material/show_chart: Overall Service Attainment Trend")
_trend_src = combine_outcomes(email_att_p, messaging_att_p)
_trend = attainment_trend(_trend_src, ts_start, ts_end)
_tgrain = {"D": "day", "W": "week (Monday start)", "M": "month"}[adaptive_grain(ts_start, ts_end)]
_tplot = _trend.dropna(subset=["Attainment %"]) if not _trend.empty else _trend
if _tplot.empty:
    st.info("No mature eligible contacts in the selected period, so there is nothing to plot.")
else:
    _line = alt.Chart(_tplot).mark_line(point=True, color="#FFFFFF").encode(
        x=alt.X("Bucket:T", title=f"Arrival {_tgrain}", axis=date_axis(_tplot["Bucket"]),
                scale=alt.Scale(domain=[_tplot["Bucket"].min() - pd.Timedelta(hours=12),
                                        _tplot["Bucket"].max() + pd.Timedelta(hours=12)])),
        y=alt.Y("Attainment %:Q", title="Overall Service Attainment (%)",
                scale=alt.Scale(domain=[0, 105])),
        tooltip=[alt.Tooltip("Bucket:T", title="Period start", format="%d %b %Y"),
                 alt.Tooltip("Attainment %:Q", format=".1f"),
                 alt.Tooltip("Successful:Q"), alt.Tooltip("Eligible:Q", title="Mature eligible"),
                 alt.Tooltip("Pending:Q", title="Email pending")])
    _rule = alt.Chart(pd.DataFrame({"y": [TARGET_PCT]})).mark_rule(
        color="#FAE100", strokeDash=[5, 5]).encode(y="y:Q")
    altair_chart((_line + _rule).properties(height=350))
    st.caption(f"Grouped by arrival {_tgrain} (daily up to 31 days, weekly to 180, then monthly). "
               "Periods with no mature eligible contacts are gaps, not zeros.")
    with st.expander("Attainment by period"):
        _d = _trend.copy()
        _d["Bucket"] = _d["Bucket"].dt.strftime("%d %b %Y")
        st.dataframe(_d.rename(columns={"Bucket": "Period start", "Eligible": "Mature eligible"})
                     .style.format({"Attainment %": "{:.1f}"}, na_rep="—"),
                     width="stretch", hide_index=True)


# =============================================================================
# Email backlog & ageing (Change 5) - metrics/backlog.py
# =============================================================================
st.markdown("---")
st.subheader(":material/inbox: Email Backlog & Ageing")
if not email_att_ok:
    st.info("Email backlog is unavailable: email.csv is missing required columns (see Overall "
            "Service Attainment above).")
else:
    _bl_t = min(ts_end, pd.Timestamp(DATA_ASOF))
    _bl = build_populations(email_sla_df)
    _bl_new = _bl["new"]
    _bl_re = _bl["reopened"]
    _age_new = backlog_ageing(_bl_new.events, _bl_t, NEW_BANDS, NEW_TARGET_HOURS)
    _rec_new = backlog_reconcile(_bl_new.events, ts_start, _bl_t)
    _re_ready = email_ingest.has_reopen_flag and reopen_timing_available(email_sla_df)
    _age_re = backlog_ageing(_bl_re.events, _bl_t, REOPENED_BANDS, REOPENED_TARGET_HOURS)
    _rec_re = backlog_reconcile(_bl_re.events, ts_start, _bl_t)

    _BL_COLOR = {"green": COLOR_GOOD, "amber": COLOR_WARN, "red": COLOR_BAD}
    _BL_TEXT = {"green": "Within safeguards", "amber": "Near safeguards", "red": "Outside safeguards"}
    _bk1, _bk2 = st.columns(2)
    render_custom_metric(
        _bk1, "New-case backlog", f"{_age_new.open_cases:,} open",
        f"New Email cases with no first response at {_bl_t:%d %b %H:%M}, aged in business hours "
        f"(07:00-22:00 CAT, 7 days). Target {NEW_TARGET_HOURS:.0f}h. Safeguard: >= 80% within target "
        "AND none older than 48h.",
        _BL_COLOR[_age_new.status],
        f"{fmt_ratio(_age_new.within_target, _age_new.open_cases)} within target • "
        f"{_age_new.over_limit:,} over 48h • {_BL_TEXT[_age_new.status]}")
    if _re_ready:
        render_custom_metric(
            _bk2, "Reopened-case backlog", f"{_age_re.open_cases:,} open",
            f"Reopened Email cases with no response since the reopen at {_bl_t:%d %b %H:%M}, "
            f"aged from `Reopened Time`. Clock {REOPENED_TARGET_HOURS:.0f}h.",
            _BL_COLOR[_age_re.status],
            f"{fmt_ratio(_age_re.within_target, _age_re.open_cases)} within target • "
            f"{_age_re.over_limit:,} over 48h • {_BL_TEXT[_age_re.status]}")
    else:
        render_custom_metric(
            _bk2, "Reopened-case backlog", "—",
            "Needs reopen timestamps, which exist only for reopens from 19 Nov 2026.", COLOR_NONE,
            "awaiting data (reopens from 19 Nov 2026)")

    def _bands_chart(a):
        return alt.Chart(a.bands).mark_bar(color="#00FF66").encode(
            x=alt.X("Band:N", sort=list(a.bands["Band"]), title="Age (business hours)"),
            y=alt.Y("Cases:Q"), tooltip=["Band", "Cases", alt.Tooltip("Share %:Q", format=".1f")])

    _ba, _bb = st.columns(2)
    with _ba:
        st.markdown("**New cases by age**")
        altair_chart(_bands_chart(_age_new).properties(height=220))
    with _bb:
        st.markdown("**Reopened cases by age**")
        if _re_ready:
            altair_chart(_bands_chart(_age_re).properties(height=220))
        else:
            st.caption("Awaiting data: reopened ageing fills in weekly from 19 Nov 2026.")

    with st.expander(":material/calculate: Backlog reconciliation (opening + new - responded = closing)",
                     expanded=True):
        _recs = [("New cases", _rec_new, "New arrivals", "First response made")]
        if _re_ready:
            _recs.append(("Reopened cases", _rec_re, "Reopened", "Responded"))
        _rows = []
        for _nm, _r, _in, _out in _recs:
            _rows.append({"Population": _nm, "Opening": _r.opening, f"+ Inflow": _r.inflow,
                          "- Outflow": _r.outflow, "= Closing": _r.closing,
                          "Reconciles": "yes" if _r.reconciles else "NO"})
        st.dataframe(pd.DataFrame(_rows).style.format(
            {c: "{:,}" for c in ("Opening", "+ Inflow", "- Outflow", "= Closing")}),
            width="stretch", hide_index=True)
        st.caption(
            f"Window {ts_start:%d %b} to {_bl_t:%d %b %H:%M}. Inflow = cases whose clock started "
            "in the window (new arrivals; first reopen per case). Outflow = cases responded to in "
            "the window. Both populations use separate clocks (4h new, 12h reopened).")
        _notes = []
        if _bl_new.unmeasurable:
            _notes.append(f"{_bl_new.unmeasurable:,} new case(s) are closed with no response "
                          "timestamp or have a response before they opened; excluded.")
        if email_ingest.has_reopen_flag:
            if _bl_re.awaiting_data:
                _notes.append(f"{_bl_re.awaiting_data:,} case(s) are flagged reopened but have no "
                              "`Reopened Time` (all reopens before 19 Nov 2026): awaiting data, not aged. "
                              "They are also not aged on the new-case clock.")
            if _bl_re.unmeasurable:
                _notes.append(f"{_bl_re.unmeasurable:,} reopened case(s) were closed again at an "
                              "unknown time; excluded.")
        else:
            _notes.append("No `Re-opened` flag in this export: reopened cases cannot be separated, so "
                          "the new-case backlog may include reopened cases.")
        for _n in _notes:
            st.caption(f":material/info: {_n}")


# =============================================================================
# Contact volume heatmap
# =============================================================================
st.markdown("---")
st.subheader(":material/grid_view: Contact Volume Heatmap")

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_ORDER = [f"{h:02d}:00" for h in range(24)]

_hm_parts = []
for _src, _label in ((chat_sla_p, "Chat"), (email_opened_p, "Email")):
    if len(_src):
        _p = _src[["Date/Time Opened"]].copy()
        _p["Channel"] = _label
        _hm_parts.append(_p)
if len(messaging_period):
    _p = messaging_period[["Start Time"]].rename(
        columns={"Start Time": "Date/Time Opened"})
    _p["Channel"] = "Messaging"
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


if has_chat_data:
    _hm_tab_names = ["Combined", "Chat only", "Messaging only", "Email only"]
    _hm_tabs = st.tabs(_hm_tab_names)
    _hm_tab_data = [
        (_hm_tabs[0], _hm, "All Contacts (Chat + Messaging + Email)", "greens"),
        (_hm_tabs[1], _hm[_hm["Channel"] == "Chat"] if not _hm.empty else _hm,
         "Chat Contacts (incl. missed)", "yellowgreen"),
        (_hm_tabs[2], _hm[_hm["Channel"] == "Messaging"] if not _hm.empty else _hm,
         "Enhanced Messaging Sessions", "goldgreen"),
        (_hm_tabs[3], _hm[_hm["Channel"] == "Email"] if not _hm.empty else _hm,
         "Email Contacts", "oranges"),
    ]
else:
    _hm_tab_names = ["Combined", "Messaging only", "Email only"]
    _hm_tabs = st.tabs(_hm_tab_names)
    _hm_tab_data = [
        (_hm_tabs[0], _hm, "All Contacts (Messaging + Email)", "greens"),
        (_hm_tabs[1], _hm[_hm["Channel"] == "Messaging"] if not _hm.empty else _hm,
         "Enhanced Messaging Sessions", "goldgreen"),
        (_hm_tabs[2], _hm[_hm["Channel"] == "Email"] if not _hm.empty else _hm,
         "Email Contacts", "oranges"),
    ]
for _tab, _sub, _title, _scheme in _hm_tab_data:
    with _tab:
        if _sub.empty:
            st.info("No data in this period.")
        else:
            altair_chart(_build_heatmap(_sub, _title, _scheme))


# =============================================================================
# Customer experience (Change 3): CSAT / Survey FCR / Operational FCR / NPS
# Four SEPARATE measures (metrics/cx.py). Never blended into one score.
# =============================================================================
st.markdown("---")
st.subheader(":material/sentiment_satisfied: Customer Experience")

if survey_q is None:
    st.info("Survey data not found (survey.csv). Add it next to the app to see CSAT / FCR / NPS.")
else:
    _sv_all = prepare_survey(survey_q)
    _sv = cx_in_window(_sv_all, ts_start, ts_end)
    _unmapped_q = unmapped_questions(survey_q)
    if _unmapped_q:
        st.warning("These survey questions matched no CSAT / NPS / FCR rule and are excluded: "
                   + "; ".join(f"“{q}”" for q in _unmapped_q[:6])
                   + (f" (+{len(_unmapped_q) - 6} more)" if len(_unmapped_q) > 6 else ""))

    _csat = csat_summary(_sv)
    _sfcr = survey_fcr_summary(_sv)
    _sfcr_chat, _sfcr_email = survey_fcr_summary(_sv, "chat"), survey_fcr_summary(_sv, "email")
    _nps = nps_summary(_sv)
    _ofcr = operational_fcr(email_sla_df if email_ingest.has_reopen_flag else None,
                            DATA_ASOF, ts_start, ts_end)

    _CX_COLOR = {"green": COLOR_GOOD, "amber": COLOR_WARN, "red": COLOR_BAD, "grey": COLOR_NONE}

    def _cx_tile(container, title, summ, tooltip):
        if summ.unavailable_reason:
            render_custom_metric(container, title, "—", tooltip, COLOR_NONE,
                                 "awaiting data (reopens from 19 Nov 2026)")
            return
        sub = fmt_ratio(summ.numerator, summ.denominator)
        if not summ.sufficient:
            sub += f" • n < {summ.min_sample}: not reliable"
        elif summ.awaiting:
            sub += f" • {summ.awaiting:,} awaiting"
        render_custom_metric(container, title, fmt_pct_floor(summ.rate_pct), tooltip,
                             _CX_COLOR[summ.status], sub)

    _k1, _k2, _k3, _k4 = st.columns(4)
    _cx_tile(_k1, "😊 CSAT", _csat,
             "% of valid respondents rating 4 or 5 on the 1-5 scale. Target 80%, minimum 30 "
             "responses. Department total only: CSAT rows carry no channel.")
    _cx_tile(_k2, "🎯 Survey FCR", _sfcr,
             "% answering Yes to 'was the issue reported ... resolved by the agent?' (chat and "
             "email questions combined). Target 80%, minimum 30 responses.")
    _cx_tile(_k3, "🔁 Operational FCR", _ofcr,
             "Email cases resolved and NOT formally reopened within 7 days / resolved cases with a "
             "complete 7-day window. Needs reopen timestamps, which exist only for reopens from "
             "19 Nov 2026; earlier resolutions are never scored.")
    render_custom_metric(
        _k4, "⭐ NPS (supporting)",
        f"{_nps.score:.1f}" if _nps.score is not None else "—",
        "%Promoters (9-10) - %Detractors (0-6), Premier Bet and Mercury Bet combined. "
        "Supporting only: no pass/fail status. Shown from 30 responses.", COLOR_NONE,
        f"{_nps.promoters:,} promoters • {_nps.passives:,} passives • {_nps.detractors:,} detractors")

    with st.expander(":material/calculate: How these reconcile (numerators, denominators, safeguards)",
                     expanded=True):
        _rows = []
        for _s in (_csat, _sfcr, _sfcr_chat, _sfcr_email, _ofcr):
            _rows.append({
                "Measure": _s.name, "Numerator": _s.numerator, "Denominator": _s.denominator,
                "Rate": fmt_pct_floor(_s.rate_pct), "Excluded (unparseable)": _s.invalid,
                "Awaiting data": _s.awaiting,
                "Status": ("n/a" if _s.unavailable_reason else
                           {"green": "On target", "amber": "Near target", "red": "Below target",
                            "grey": f"n < {_s.min_sample}"}[_s.status]),
                "Shown as headline": ("channel only if n >= 30" if "(" in _s.name else "yes"),
            })
        st.dataframe(pd.DataFrame(_rows).style.format(
            {c: "{:,}" for c in ("Numerator", "Denominator", "Excluded (unparseable)",
                                 "Awaiting data")}), width="stretch", hide_index=True)
        st.caption(
            "Target 80% for CSAT and both FCRs: Green >= 80, Amber 70-79.9, Red < 70, Grey below 30 "
            "responses. Survey cohort = survey response date. Operational FCR cohort = resolution "
            "date (resolution time = the export's Completion Date; there is no separate close "
            "date). NPS is not pass/fail. **Survey response rate is unavailable** (issued volume "
            "is not collected) and there is **no CSAT channel split** (CSAT rows carry no channel).")
        _ch_lines = []
        for _s in (_sfcr_chat, _sfcr_email):
            if _s.sufficient:
                _ch_lines.append(f"**{_s.name}**: {fmt_ratio(_s.numerator, _s.denominator)}")
            else:
                _ch_lines.append(f"{_s.name}: only {_s.denominator:,} response(s) - below the "
                                 f"{_s.min_sample} needed to show this channel on its own")
        st.markdown("  \n".join(_ch_lines))
        _nps_note = (f"NPS: {_nps.n:,} valid response(s)" +
                     ("" if _nps.sufficient else f" - below {_nps.min_sample}, score suppressed"))
        st.caption(_nps_note)

    _ct = csat_trend(_sv_all, ts_start, ts_end)
    if not _ct.empty:
        st.markdown("#### CSAT trend")
        _ctp = _ct.dropna(subset=["CSAT %"])
        if _ctp.empty:
            st.caption("Every bucket in this period has fewer than 30 CSAT responses, so no "
                       "trend is drawn (shown as a table below).")
        else:
            _cl = alt.Chart(_ctp).mark_line(point=True, color="#00FF66").encode(
                x=alt.X("Bucket:T", title="Period start", axis=date_axis(_ctp["Bucket"])),
                y=alt.Y("CSAT %:Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("Bucket:T"), alt.Tooltip("CSAT %:Q", format=".1f"),
                         alt.Tooltip("Satisfied:Q"), alt.Tooltip("Valid:Q")])
            _cr = alt.Chart(pd.DataFrame({"y": [80.0]})).mark_rule(
                color="#FAE100", strokeDash=[5, 5]).encode(y="y:Q")
            altair_chart((_cl + _cr).properties(height=300))
        st.caption("Bucket size adapts to the period (daily up to 31 days, weekly to 180, then "
                   "monthly); buckets with fewer than 30 responses show no rate.")
        _ctd = _ct.copy()
        _ctd["Bucket"] = _ctd["Bucket"].dt.strftime("%d %b %Y")
        with st.expander("CSAT by period"):
            st.dataframe(_ctd.style.format({"CSAT %": "{:.1f}"}, na_rep="n < 30"),
                         width="stretch", hide_index=True)


# =============================================================================
# Conversation volume by country
# =============================================================================
st.markdown("---")
st.subheader(":material/public: Conversation Volume by Country")

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
    Map a legacy chat-button or Messaging channel name to a country.

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


_selected_has_live_chat = start_date < MESSAGING_SLA_START_DATE
_selected_has_messaging = end_date >= MESSAGING_SLA_START_DATE
_country_parts = []

# Retain the historical Live Chat view, but never count it on/after the
# migration date. chat_sla_p already applies that cut-off.
if _selected_has_live_chat:
    if DEVNAME_COL not in chat_sla_p.columns:
        st.info("Column 'Chat Button: Developer Name' not found in chat.csv.")
    elif not chat_sla_p.empty:
        _legacy_country = chat_sla_p[[DEVNAME_COL]].rename(
            columns={DEVNAME_COL: "Source Name"})
        _legacy_country["Channel"] = "Live Chat"
        _country_parts.append(_legacy_country)

# From 1 November, country comes from the Messaging Channel Name. The org's
# naming convention includes a country in every channel name, so the same
# mapping logic used for legacy chat buttons can classify each session.
if _selected_has_messaging:
    if MESSAGING_CHANNEL_FIELD not in messaging_period.columns:
        st.info(
            "To enable Messaging by country from 1 November, add "
            f"'{MESSAGING_CHANNEL_FIELD}' to report_messaging_metrics.csv and "
            "re-export the Salesforce report."
        )
    elif not messaging_period.empty:
        _messaging_country = messaging_period[[MESSAGING_CHANNEL_FIELD]].rename(
            columns={MESSAGING_CHANNEL_FIELD: "Source Name"})
        _messaging_country["Channel"] = "Messaging"
        _country_parts.append(_messaging_country)

if not _country_parts:
    st.info("No conversations with country information in the selected period.")
else:
    dfc = pd.concat(_country_parts, ignore_index=True)
    # No .str.title(): it corrupts canonical names such as "Congo (DRC)" and
    # "São Tomé and Príncipe".
    dfc["Country"] = dfc["Source Name"].apply(_country_from_devname)

    counts = (
        dfc.groupby("Country", dropna=False).size()
        .reset_index(name="Conversations")
        .sort_values("Conversations", ascending=False)
        .reset_index(drop=True)
    )

    if counts["Conversations"].sum() == 0:
        st.info("No conversations in the selected period.")
    else:
        top_n = st.sidebar.slider("Pie chart: top countries", 3, 12, 8, 1)
        min_share = st.sidebar.slider("Label slices ≥ this share", 0.0, 0.1, 0.02, 0.01)

        grand_total = int(counts["Conversations"].sum())
        plot_counts = counts.copy()
        if len(plot_counts) > top_n:
            top = plot_counts.head(top_n)
            others = int(plot_counts["Conversations"].iloc[top_n:].sum())
            plot_counts = pd.concat(
                [top, pd.DataFrame({"Country": ["Other"], "Conversations": [others]})],
                ignore_index=True,
            )
        plot_counts["Share"] = plot_counts["Conversations"] / grand_total

        if _selected_has_live_chat and _selected_has_messaging:
            _country_chart_title = "Conversation Volume by Country"
        elif _selected_has_messaging:
            _country_chart_title = "Messaging Volume by Country"
        else:
            _country_chart_title = "Live Chat Volume by Country"

        pie = (
            alt.Chart(plot_counts)
            .mark_arc(outerRadius=180, innerRadius=100, stroke="rgba(255,255,255,0.45)", strokeWidth=2)
            .encode(
                theta=alt.Theta("Conversations:Q", stack=True),
                color=alt.Color("Country:N", legend=alt.Legend(title="Country"),
                                scale=alt.Scale(range=["#00FF66", "#FAE100", "#FF9300", "#FFFFFF", "#29C2EE",
                                                      "#F2BB22", "#7DFFB0", "#FFC266", "#B3E6C7"])),
                order=alt.Order("Conversations:Q", sort="descending"),
                tooltip=[alt.Tooltip("Country:N", title="Country"),
                         alt.Tooltip("Conversations:Q", title="Conversations", format=","),
                         alt.Tooltip("Share:Q", title="Share", format=".1%")],
            )
            .properties(width=550, height=450,
                        title=alt.TitleParams(_country_chart_title,
                                              anchor="middle", fontSize=16))
        )

        label_df = plot_counts[plot_counts["Share"] >= min_share].copy()
        label_df["Label"] = label_df.apply(
            lambda r: f"{r['Country']} ({r['Share']:.0%})", axis=1)
        labels = (
            alt.Chart(label_df)
            .mark_text(radius=205, size=12)
            .encode(theta=alt.Theta("Conversations:Q", stack=True), text="Label:N",
                    order=alt.Order("Conversations:Q", sort="descending"),
                    color=alt.value("#FFFFFF"))
        )
        center_text = alt.Chart(
            pd.DataFrame({"text": [f"Total: {grand_total:,}"], "x": [0], "y": [0]})
        ).mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color="#FFFFFF").encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="text:N")

        altair_chart(alt.layer(pie, labels, center_text).configure_view(stroke=None))

        with st.expander("View country breakdown table"):
            _tbl = counts.copy()
            _tbl["Share"] = _tbl["Conversations"] / grand_total
            st.dataframe(
                _tbl.style.format({"Conversations": "{:,}", "Share": "{:.1%}"}),
                width="stretch",
            )

            # Keep naming gaps visible so a new Salesforce channel cannot be
            # silently buried in the Unknown slice.
            _unknown = dfc.loc[dfc["Country"] == "Unknown", ["Channel", "Source Name"]]
            if len(_unknown):
                _names = (
                    _unknown.value_counts(dropna=False).rename("Conversations").reset_index()
                )
                st.markdown(
                    f"**{len(_unknown):,} conversations did not map to a country.** "
                    "Unmapped chat buttons or Messaging channel names:"
                )
                st.dataframe(_names, width="stretch")


# =============================================================================
# Department breakdowns (Change 6) - metrics/breakdowns.py
# Overall Service Attainment split by department-level dimensions. No agent dimension.
# =============================================================================
st.markdown("---")
st.subheader(":material/account_tree: Department Breakdowns")
st.caption(
    "Overall Service Attainment (Email + Messaging) split by each dimension. A result is coloured "
    f"only with at least {BD_MIN_CONTACTS} eligible contacts and {COVERAGE_MIN_PCT:.0f}% coverage; "
    "otherwise the numbers are shown in grey. Each group's Successful / Eligible adds back to the "
    "headline above. Contacts are grouped by arrival time.")

_bd_queue = first_queue_by_case(df_items, channel_value=EMAIL_DEVNAME)
_bd_msg = messaging_att_p.copy() if messaging_att_p is not None else None
if _bd_msg is not None and not _bd_msg.empty and MESSAGING_CHANNEL_FIELD in _bd_msg.columns:
    _bd_msg["Messaging Country"] = _bd_msg[MESSAGING_CHANNEL_FIELD].map(_country_from_devname)
_bd = add_time_dims(combine_outcomes(
    email_att_p, _bd_msg, queue_map=_bd_queue, messaging_country_col="Messaging Country"))

_BD_COLOR = {"green": "#00FF66", "amber": "#FAE100", "red": "#FF9300", "grey": "#9ca3af"}
_BD_TEXT = {"green": "On target", "amber": "Near target", "red": "Below target", "grey": "Not reliable"}


def _dim_view(by, channel=None, title=None, key=""):
    src = _bd if channel is None else _bd[_bd["Channel"] == channel]
    t = dimension_table(src, by)
    if title:
        st.markdown(f"**{title}**")
    if t.empty:
        st.caption("No contacts in this period for this dimension.")
        return
    shown = t.copy()
    shown["Status"] = shown["Status"].map(_BD_TEXT)
    st.dataframe(shown.style.format({"Successful": "{:,}", "Eligible": "{:,}", "Pending": "{:,}",
                                     "Unmeasurable": "{:,}", "Excluded": "{:,}",
                                     "Attainment %": "{:.1f}", "Coverage %": "{:.1f}"}, na_rep="—"),
                 width="stretch", hide_index=True)
    plot = t.dropna(subset=["Attainment %"])
    if not plot.empty:
        order = list(t["Group"])
        altair_chart(alt.Chart(plot).mark_bar().encode(
            x=alt.X("Group:N", sort=order, title=None),
            y=alt.Y("Attainment %:Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Status:N", legend=None, scale=alt.Scale(
                domain=list(_BD_COLOR), range=list(_BD_COLOR.values()))),
            tooltip=["Group", "Successful", "Eligible", alt.Tooltip("Attainment %:Q", format=".1f"),
                     "Status"]).properties(height=240))


_bd_tabs = st.tabs(["Channel", "Email queue", "Email origin", "Country / market",
                    "Day & hour", "New vs reopened", "Messaging transfers"])
with _bd_tabs[0]:
    _dim_view("Channel")
with _bd_tabs[1]:
    _dim_view("Queue", "Email")
    st.caption("Queue = the first routing queue of the case in report_items.csv (joined on case "
               "number). 'Not routed / unknown' cases never reached an Omni-Channel queue.")
with _bd_tabs[2]:
    _dim_view("Origin", "Email")
    st.caption("Contact Us = Contact Form, Email = Direct Email. Priority is a legacy field and is "
               "not used.")
with _bd_tabs[3]:
    _dim_view("Country", "Email", "Email (country from email.csv `Country`)")
    _dim_view("Country", "Messaging", "Messaging (country from the Messaging channel name)")
    st.caption("Email and Messaging countries come from different sources and are shown separately. "
               "Raw email values are mapped to a controlled list; blank or unmapped = Unknown.")
with _bd_tabs[4]:
    _dim_view("Day of week", title="Day of week")
    _dim_view("Hour of day", title="Hour of day")
with _bd_tabs[5]:
    _dim_view("Case type", "Email")
    st.caption("From the authoritative `Re-opened` flag. Reopened cases are measured on their final "
               "response, so their first-response result is approximate.")
with _bd_tabs[6]:
    _dim_view("Transfer", "Messaging")


# =============================================================================
# Hourly Service Attainment
# =============================================================================
st.markdown("---")
st.subheader(":material/schedule: Hourly Service Attainment (selected day)")

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
                logged[h].add(agent)   # counted only; names are never displayed
                if status in AVAILABLE_STATUSES:
                    avail[h] += (o[1] - o[0]).total_seconds()
            h = nxt

    return pd.DataFrame({
        "Hour": list(hours),
        "Avail (min)": [avail[h] / 60.0 for h in hours],
        "Logged In Agents": [len(logged[h]) for h in hours],
    })


_h_start = pd.Timestamp(hourly_date)
_h_end = _h_start + pd.Timedelta(days=1)
_he, _hm, _ = attainment_for_window(_h_start, _h_end)
df_hourly = attainment_series(_he, _hm, _h_start, _h_end, "h").rename(columns={"Bucket": "Hour"})
df_hourly = df_hourly.merge(compute_hourly_availability(hourly_date), on="Hour", how="left")

if df_hourly["Overall Den"].sum() == 0 and df_hourly["Avail (min)"].sum() == 0:
    st.info("No activity for the selected day.")
else:
    show_breakdown = st.checkbox("Show Email & Messaging lines", value=False)
    show_avail = st.checkbox("Overlay available minutes (bars)", value=True)

    _hplot = df_hourly.dropna(subset=["Overall %"])
    layers = []
    if not _hplot.empty:
        layers.append(
            alt.Chart(_hplot).mark_line(point=True, color="#FFFFFF").encode(
                x=alt.X("Hour:T", title="Hour", axis=alt.Axis(format="%H:%M", labelAngle=-45)),
                y=alt.Y("Overall %:Q", title="Overall attainment (%)",
                        scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(orient="left")),
                tooltip=[alt.Tooltip("Hour:T", title="Hour", format="%H:%M"),
                         alt.Tooltip("Overall %:Q", format=".1f"),
                         alt.Tooltip("Overall Num:Q", title="Successful"),
                         alt.Tooltip("Overall Den:Q", title="Mature eligible"),
                         alt.Tooltip("Pending:Q", title="Email pending")]))
        if show_breakdown:
            for _col, _colour in (("Messaging %", "#FAE100"), ("Email %", "#FF9300")):
                _sub = df_hourly.dropna(subset=[_col])
                if _sub.empty:
                    continue
                layers.append(
                    alt.Chart(_sub).mark_line(point=True, color=_colour).encode(
                        x=alt.X("Hour:T", axis=None),
                        y=alt.Y(f"{_col}:Q", scale=alt.Scale(domain=[0, 105]), axis=None),
                        tooltip=[alt.Tooltip("Hour:T", format="%H:%M"),
                                 alt.Tooltip(f"{_col}:Q", format=".1f")]))

    if show_avail and df_hourly["Avail (min)"].sum() > 0:
        layers.append(
            alt.Chart(df_hourly).mark_bar(opacity=0.30, color="#00FF66").encode(
                x=alt.X("Hour:T", axis=None),
                y=alt.Y("Avail (min):Q", title="Available (min)", axis=alt.Axis(orient="right")),
                tooltip=[alt.Tooltip("Hour:T", format="%H:%M"),
                         alt.Tooltip("Avail (min):Q", title="Available (min)", format=",.0f"),
                         alt.Tooltip("Logged In Agents:Q", title="Logged In Agents", format=",.0f")]))

    if layers:
        altair_chart(alt.layer(*layers).resolve_scale(y="independent").properties(height=350))
        st.caption("Contacts are grouped by the hour they entered the queue; hours with no mature "
                   "eligible contact are gaps. Available minutes are summed across agents, so an "
                   "hour with 3 agents fully available reads 180.")
    else:
        st.info("No activity for the selected day.")

"""Customer experience measures (Change 3): CSAT, Survey FCR, Operational FCR, NPS.

The four measures are deliberately kept SEPARATE and never blended (docs/build-spec.md 2.7, 3).

* **CSAT** - % of valid respondents rating 4 or 5 (1-5 scale). Target 80 %, minimum 30 responses.
  Department total only: CSAT rows carry no channel.
* **Survey FCR** - % answering "Yes" to "Was the issue reported in the chat/email resolved by the
  agent?". The channel is part of the question wording. Target 80 %, minimum 30. A channel's own
  result is only shown where that channel independently reaches 30.
* **NPS** - %Promoters (9-10) - %Detractors (0-6), both brands. Supporting only: no pass/fail
  status. Minimum 30.
* **Operational FCR** - resolved Email cases NOT formally reopened within 7 days of resolution /
  resolved cases whose 7-day window is complete. Needs reopen TIMING, which exists only for
  reopens from 2026-11-19, so earlier resolutions are "awaiting data" and are never scored.
  Resolution time is the export's ``Completion Date`` (the export has no separate close date).

Survey response rate is unavailable (issued volume is not collected) and is never invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.ingest import REOPEN_TIMING_START
from metrics.service_attainment import EMAIL_CLOSED_STATUSES, pct

MIN_SAMPLE = 30
TARGET_PCT = 80.0
AMBER_FLOOR_PCT = 70.0
FCR_WINDOW_DAYS = 7

COL_ID = "Survey Taker: ID"
COL_DATE = "Survey Taker: Created Date"
COL_TITLE = "Survey Question: Question Title"
COL_RESPONSE = "Response"

_RE_NPS = re.compile(r"\brecommend\b", re.I)
_RE_CSAT = re.compile(r"\bsatisf(?:ied|action)\b", re.I)
_RE_FCR = re.compile(r"\bresolved by the agent\b", re.I)


def classify_question(title) -> Optional[str]:
    """-> 'NPS' | 'CSAT' | 'FCR_CHAT' | 'FCR_EMAIL' | None (unmapped)."""
    if title is None or (not isinstance(title, str) and pd.isna(title)):
        return None
    t = str(title)
    if _RE_NPS.search(t):
        return "NPS"
    if _RE_FCR.search(t):
        low = t.lower()
        if re.search(r"\bemail\b", low):
            return "FCR_EMAIL"
        if re.search(r"\bchat\b", low):
            return "FCR_CHAT"
        return None
    if _RE_CSAT.search(t):
        return "CSAT"
    return None


def _leading_int(x) -> Optional[int]:
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    m = re.match(r"\s*(\d+)", str(x))
    return int(m.group(1)) if m else None


def _yes_no(x) -> Optional[bool]:
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    m = re.match(r"^(yes|no|y|n|true|false)\b", s)
    if not m:
        return None
    return m.group(1) in ("yes", "y", "true")


def prepare_survey(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per respondent x measure with a parsed ``Value`` (None = unparseable/invalid).

    Columns: Respondent, Date, Measure, Value, Valid. A respondent counts once per measure
    (first answer wins), so a repeated row can never inflate a sample.
    """
    cols = ["Respondent", "Date", "Measure", "Value", "Valid"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({
        "Respondent": raw[COL_ID] if COL_ID in raw.columns else pd.Series(raw.index, index=raw.index),
        "Date": pd.to_datetime(raw[COL_DATE], errors="coerce") if COL_DATE in raw.columns
        else pd.NaT,
        "Measure": raw[COL_TITLE].map(classify_question),
        "Raw": raw[COL_RESPONSE],
    })
    df = df.dropna(subset=["Measure"])
    # rows with no respondent id are treated as distinct respondents
    df["Respondent"] = df["Respondent"].astype(object)
    missing = df["Respondent"].isna()
    if missing.any():
        df.loc[missing, "Respondent"] = ["__row_" + str(i) for i in df.index[missing]]
    df = df.drop_duplicates(subset=["Respondent", "Measure"], keep="first")

    value = pd.Series(np.nan, index=df.index, dtype="float64")
    is_nps, is_csat = df["Measure"] == "NPS", df["Measure"] == "CSAT"
    is_fcr = df["Measure"].isin(["FCR_CHAT", "FCR_EMAIL"])
    value[is_nps] = df.loc[is_nps, "Raw"].map(_leading_int).astype("float64")
    value[is_csat] = df.loc[is_csat, "Raw"].map(_leading_int).astype("float64")
    value[is_fcr] = df.loc[is_fcr, "Raw"].map(_yes_no).map(
        {True: 1.0, False: 0.0}).astype("float64")
    valid = value.notna()
    valid &= ~(is_nps & ~value.between(0, 10))
    valid &= ~(is_csat & ~value.between(1, 5))
    out = pd.DataFrame({"Respondent": df["Respondent"], "Date": df["Date"],
                        "Measure": df["Measure"], "Value": value.where(valid), "Valid": valid})
    return out.reset_index(drop=True)


def unmapped_questions(raw: pd.DataFrame) -> list:
    if raw is None or raw.empty or COL_TITLE not in raw.columns:
        return []
    mapped = raw[COL_TITLE].map(classify_question)
    return sorted(raw.loc[mapped.isna(), COL_TITLE].dropna().astype(str).unique())


# --------------------------------------------------------------------------- summaries
def cx_status(num: int, den: int, min_sample: int = MIN_SAMPLE, target_pct: float = TARGET_PCT,
              amber_floor_pct: float = AMBER_FLOOR_PCT) -> str:
    """green / amber / red, or grey when the sample is below the minimum."""
    if den < min_sample or den <= 0:
        return "grey"
    if num * 100 >= target_pct * den:
        return "green"
    if num * 100 >= amber_floor_pct * den:
        return "amber"
    return "red"


@dataclass
class RateSummary:
    """A proportion with its visible numerator / denominator and sample safeguard."""
    name: str
    numerator: int = 0
    denominator: int = 0
    invalid: int = 0              # answered but unparseable / out of scale (excluded)
    min_sample: int = MIN_SAMPLE
    unavailable_reason: Optional[str] = None
    awaiting: int = 0             # cases not yet measurable (e.g. 7-day window open)

    @property
    def rate_pct(self) -> Optional[float]:
        return pct(self.numerator, self.denominator)

    @property
    def sufficient(self) -> bool:
        return self.denominator >= self.min_sample

    @property
    def status(self) -> str:
        if self.unavailable_reason:
            return "grey"
        return cx_status(self.numerator, self.denominator, self.min_sample)


def csat_summary(prepared: pd.DataFrame) -> RateSummary:
    d = prepared[prepared["Measure"] == "CSAT"] if len(prepared) else prepared
    valid = d[d["Valid"]] if len(d) else d
    return RateSummary("CSAT", int((valid["Value"] >= 4).sum()) if len(valid) else 0,
                       int(len(valid)), invalid=int(len(d) - len(valid)))


def survey_fcr_summary(prepared: pd.DataFrame, channel: Optional[str] = None) -> RateSummary:
    """``channel`` in {None (combined), 'chat', 'email'}."""
    measures = {None: ["FCR_CHAT", "FCR_EMAIL"], "chat": ["FCR_CHAT"], "email": ["FCR_EMAIL"]}[channel]
    d = prepared[prepared["Measure"].isin(measures)] if len(prepared) else prepared
    valid = d[d["Valid"]] if len(d) else d
    label = "Survey FCR" + (f" ({channel})" if channel else "")
    return RateSummary(label, int((valid["Value"] == 1).sum()) if len(valid) else 0,
                       int(len(valid)), invalid=int(len(d) - len(valid)))


@dataclass
class NpsSummary:
    promoters: int = 0
    passives: int = 0
    detractors: int = 0
    invalid: int = 0
    min_sample: int = MIN_SAMPLE

    @property
    def n(self) -> int:
        return self.promoters + self.passives + self.detractors

    @property
    def sufficient(self) -> bool:
        return self.n >= self.min_sample

    @property
    def score(self) -> Optional[float]:
        """%Promoters - %Detractors; None below the minimum sample. Supporting only (no status)."""
        if not self.sufficient:
            return None
        return (self.promoters - self.detractors) / self.n * 100.0


def nps_summary(prepared: pd.DataFrame) -> NpsSummary:
    d = prepared[prepared["Measure"] == "NPS"] if len(prepared) else prepared
    valid = d[d["Valid"]] if len(d) else d
    v = valid["Value"] if len(valid) else pd.Series(dtype="float64")
    return NpsSummary(promoters=int((v >= 9).sum()), passives=int(((v >= 7) & (v <= 8)).sum()),
                      detractors=int((v <= 6).sum()), invalid=int(len(d) - len(valid)))


def in_window(prepared: pd.DataFrame, start, end) -> pd.DataFrame:
    """Respondents whose survey date (default cohort timestamp) is in [start, end)."""
    if prepared is None or prepared.empty:
        return prepared
    t = prepared["Date"]
    return prepared[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))]


# --------------------------------------------------------------------------- operational FCR
def operational_fcr(email: pd.DataFrame, as_of, start, end,
                    closed_statuses=EMAIL_CLOSED_STATUSES) -> RateSummary:
    """Operational FCR for cases RESOLVED in ``[start, end)``.

    * resolved = closed-type ``Status`` with a ``Completion Date`` (the resolution timestamp);
    * measurable only when resolved on/after 2026-11-19 (reopen timing exists from then) AND the
      7-day window has completed by ``as_of``;
    * success = not reopened; failure = reopened within 7 days of resolution (needs ``Reopened Time``);
    * a flagged reopen with no ``Reopened Time`` cannot be placed in time -> ``invalid`` (excluded).
    Resolved cases outside the measurable set are reported in ``awaiting`` - never scored.
    """
    s = RateSummary("Operational FCR")
    needed = {"Status", "Completion Date", "Reopened", "Reopened Time"}
    if email is None or email.empty or not needed.issubset(email.columns):
        s.unavailable_reason = "Reopen flag / timing not available in this export."
        return s
    status = email["Status"].astype("string").str.strip().str.casefold()
    resolved = status.isin({c.casefold() for c in closed_statuses}).fillna(False) \
        & email["Completion Date"].notna()
    res_ts = email["Completion Date"]
    in_win = resolved & (res_ts >= pd.Timestamp(start)) & (res_ts < pd.Timestamp(end))
    d = email[in_win]
    if d.empty:
        return s
    window_done = (d["Completion Date"] + pd.Timedelta(days=FCR_WINDOW_DAYS)) <= pd.Timestamp(as_of)
    timing_exists = d["Completion Date"] >= REOPEN_TIMING_START
    measurable = window_done & timing_exists
    s.awaiting = int((~measurable).sum())
    m = d[measurable]
    reopened = m["Reopened"].astype(bool)
    has_time = m["Reopened Time"].notna()
    within = reopened & has_time & (
        m["Reopened Time"] <= m["Completion Date"] + pd.Timedelta(days=FCR_WINDOW_DAYS))
    unknown = reopened & ~has_time
    s.invalid = int(unknown.sum())
    s.denominator = int((~unknown).sum())
    s.numerator = int((~reopened).sum() + (reopened & has_time & ~within).sum())
    return s


# --------------------------------------------------------------------------- trend
def adaptive_freq(start, end) -> str:
    """Spec 3 / Change 6: daily <= 31 days, weekly 32-180, monthly > 180 (pandas period codes)."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return "D" if days <= 31 else ("W-MON" if days <= 180 else "MS")


def csat_trend(prepared: pd.DataFrame, start, end, min_sample: int = MIN_SAMPLE) -> pd.DataFrame:
    """CSAT % per adaptive bucket; buckets below the minimum sample have a NaN rate."""
    cols = ["Bucket", "Satisfied", "Valid", "CSAT %"]
    d = in_window(prepared, start, end)
    d = d[(d["Measure"] == "CSAT") & d["Valid"]] if d is not None and len(d) else None
    if d is None or d.empty:
        return pd.DataFrame(columns=cols)
    freq = adaptive_freq(start, end)
    if freq == "D":
        bucket = d["Date"].dt.normalize()
    elif freq == "MS":
        bucket = d["Date"].dt.to_period("M").dt.to_timestamp()
    else:   # weeks start Monday
        bucket = (d["Date"].dt.normalize() - pd.to_timedelta(d["Date"].dt.weekday, unit="D"))
    g = pd.DataFrame({"Bucket": bucket, "Sat": (d["Value"] >= 4).astype(int)}).groupby("Bucket")
    out = g.agg(Satisfied=("Sat", "sum"), Valid=("Sat", "size")).reset_index()
    out["CSAT %"] = np.where(out["Valid"] >= min_sample, out["Satisfied"] / out["Valid"] * 100.0,
                             np.nan)
    return out[cols]

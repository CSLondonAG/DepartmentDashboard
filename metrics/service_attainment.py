"""Overall Service Attainment (Change 1) - contact-weighted Email + Messaging.

    Overall = (successful eligible Email + successful eligible Messaging)
              / (measurable mature eligible Email + eligible Messaging)

Pure pandas, no Streamlit. Every headline percentage is built from an integer numerator and
denominator that the UI shows next to it (spec section 6: "reconcile to visible num/den").

Definitions (docs/build-spec.md 2.1, 2.4, 2.5, 3):

* **Email** - first response within 4 *business* hours, recomputed by us from
  ``Date/Time Opened`` -> ``Completion Date`` on the 07:00-22:00, 7-days-a-week calendar
  (timestamps are treated as naive Central Africa Time; see ``config``).  Cohort = queue entry.
  answered <= target = success; answered later = fail; unanswered and overdue = fail;
  unanswered and not yet due = **pending** (excluded from the mature denominator).
* **Messaging** - complete-service rule: a session succeeds only if it was handled by a human
  AND the human's first response came within 20 s of acceptance.  Abandoned = fail.  Canceled,
  Unavailable, bot-only, test and unclassifiable sessions are excluded and shown with counts.
* **Status** - Green >= 80, Amber 70-79.9, Red < 70, Grey when coverage < 95 % or nothing mature.
* **Provisional** - set while any Email is pending or the window is not yet fully covered by data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- config
OPEN_HOUR = 7            # business day opens 07:00 CAT
CLOSE_HOUR = 22          # ... and closes 22:00 CAT; the clock pauses overnight
EMAIL_TARGET_BUSINESS_HOURS = 4.0
MESSAGING_TARGET_SEC = 20.0
TARGET_PCT = 80.0        # Overall (and channel) attainment target
AMBER_FLOOR_PCT = 70.0
COVERAGE_MIN_PCT = 95.0
#: Statuses that mean a case is finished. A finished case with no response timestamp cannot
#: be measured (it is neither a confirmed hit nor a datable miss).
EMAIL_CLOSED_STATUSES = frozenset({"resolved", "closed", "completed"})
MESSAGING_RESPONSE_COL = "Service Rep Accept To First Response Time"

BUSINESS_DAY_SECONDS = (CLOSE_HOUR - OPEN_HOUR) * 3600
_EPOCH = pd.Timestamp("1970-01-01")

# --------------------------------------------------------------------------- outcomes
#: outcome -> (group, human label).  Groups: success | fail | pending | unmeasurable | excluded
OUTCOMES: Dict[str, tuple] = {
    # Email
    "email_success": ("success", "Responded within target"),
    "email_fail_late": ("fail", "Responded after target"),
    "email_fail_unanswered": ("fail", "Unanswered and overdue"),
    "email_pending": ("pending", "Unanswered, not yet due (pending)"),
    "email_unmeasurable_negative": ("unmeasurable", "Completion before open (bad timestamps)"),
    "email_unmeasurable_closed_no_response": (
        "unmeasurable", "Closed with no response timestamp"),
    "email_unmeasurable_no_open_time": ("unmeasurable", "Missing open time"),
    # Messaging
    "msg_success": ("success", "Handled and first response within target"),
    "msg_fail_slow": ("fail", "Handled but first response too slow"),
    "msg_fail_no_response": ("fail", "Handled with no human response"),
    "msg_fail_abandoned": ("fail", "Abandoned by customer"),
    "msg_excluded_canceled": ("excluded", "Canceled (technical)"),
    "msg_excluded_unavailable": ("excluded", "Agent unavailable"),
    "msg_excluded_bot_only": ("excluded", "Bot-only (no human)"),
    "msg_excluded_test": ("excluded", "Test session"),
    "msg_unmeasurable_no_agent_work": ("unmeasurable", "No Agent Work record (unclassifiable)"),
    "msg_unmeasurable_not_accepted": (
        "unmeasurable", "Routed, never accepted, no abandon event"),
}
GROUPS = ("success", "fail", "pending", "unmeasurable", "excluded")


def outcome_group(outcome: str) -> str:
    return OUTCOMES[outcome][0]


def outcome_label(outcome: str) -> str:
    return OUTCOMES[outcome][1]


# --------------------------------------------------------------------------- calendar
def _cum_business_seconds(ts: pd.Series) -> pd.Series:
    """Business seconds elapsed since the epoch (open window 07:00-22:00, every day)."""
    ts = pd.to_datetime(ts)
    day = ts.dt.normalize()
    days = (day - _EPOCH).dt.days
    tod = (ts - day).dt.total_seconds()
    in_day = tod.clip(lower=OPEN_HOUR * 3600, upper=CLOSE_HOUR * 3600) - OPEN_HOUR * 3600
    return days * BUSINESS_DAY_SECONDS + in_day


def business_hours_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Business hours from ``start`` to ``end`` (negative if end < start, NaN if either is NaT).

    07:00-22:00 CAT, seven days a week; public holidays are normal operating days;
    22:00-07:00 pauses the clock.
    """
    return (_cum_business_seconds(end) - _cum_business_seconds(start)) / 3600.0


# --------------------------------------------------------------------------- email
def classify_email(
    df: pd.DataFrame,
    as_of,
    target_hours: float = EMAIL_TARGET_BUSINESS_HOURS,
    closed_statuses: Iterable[str] = EMAIL_CLOSED_STATUSES,
) -> pd.DataFrame:
    """Add ``Business Hours To Response``, ``Wait Business Hours`` and ``Outcome`` columns.

    ``as_of`` is the export horizon: an unanswered email is overdue only once the business
    time since it opened exceeds the target as of that moment.
    """
    out = df.copy()
    if out.empty:
        out["Business Hours To Response"] = pd.Series(dtype="float64")
        out["Wait Business Hours"] = pd.Series(dtype="float64")
        out["Outcome"] = pd.Series(dtype="object")
        return out

    opened = out["Date/Time Opened"]
    completed = out["Completion Date"]
    as_of_ts = pd.Series(pd.Timestamp(as_of), index=out.index)

    resp_bh = business_hours_between(opened, completed)
    wait_bh = business_hours_between(opened, as_of_ts)
    out["Business Hours To Response"] = resp_bh
    out["Wait Business Hours"] = wait_bh.where(completed.isna())

    closed = (out["Status"].astype("string").str.strip().str.casefold()
              .isin({s.casefold() for s in closed_statuses}).fillna(False).astype(bool))
    # Compare in whole seconds so 4h00m00s is "within" regardless of float noise.
    target_s = target_hours * 3600.0
    resp_s = (resp_bh * 3600.0).round(3)
    wait_s = (wait_bh * 3600.0).round(3)

    answered = completed.notna()
    outcome = np.select(
        [
            opened.isna(),
            answered & (resp_s < 0),
            answered & (resp_s <= target_s),
            answered,
            closed,
            wait_s > target_s,
        ],
        [
            "email_unmeasurable_no_open_time",
            "email_unmeasurable_negative",
            "email_success",
            "email_fail_late",
            "email_unmeasurable_closed_no_response",
            "email_fail_unanswered",
        ],
        default="email_pending",
    )
    out["Outcome"] = outcome
    return out


# --------------------------------------------------------------------------- messaging
def classify_messaging(
    sessions: pd.DataFrame,
    target_sec: float = MESSAGING_TARGET_SEC,
    response_col: str = MESSAGING_RESPONSE_COL,
) -> pd.DataFrame:
    """Classify session-level Messaging rows (one row per session).

    Expected columns (all produced by the dashboard's session roll-up):
    ``Assignments`` (Agent Work rows), ``Human_Agents`` (distinct non-bot agents),
    ``Any_Human_Closed``, ``Any_Abandoned``, ``Any_Canceled``, ``Any_Unavailable``, the
    accept->first-response metric ``response_col`` (seconds; 0/blank = no human response), and
    optionally ``Is Test``.

    Precedence: unclassifiable -> bot-only -> test -> handled -> abandoned -> canceled ->
    unavailable -> routed-but-never-accepted.
    """
    out = sessions.copy()
    if out.empty:
        out["First Response Sec"] = pd.Series(dtype="float64")
        out["Outcome"] = pd.Series(dtype="object")
        return out

    def _bool(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        return out[col].fillna(False).astype(bool)

    def _num(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(0.0, index=out.index)
        return pd.to_numeric(out[col], errors="coerce").fillna(0)

    assignments = _num("Assignments")
    humans = _num("Human_Agents")
    resp = pd.to_numeric(out.get(response_col), errors="coerce") if response_col in out.columns \
        else pd.Series(np.nan, index=out.index)
    out["First Response Sec"] = resp

    handled = _bool("Any_Human_Closed")
    has_response = resp > 0           # Salesforce emits 0 when no response exists
    outcome = np.select(
        [
            assignments <= 0,
            humans <= 0,
            _bool("Is Test"),
            handled & has_response & (resp <= target_sec),
            handled & has_response,
            handled,
            _bool("Any_Abandoned"),
            _bool("Any_Canceled"),
            _bool("Any_Unavailable"),
        ],
        [
            "msg_unmeasurable_no_agent_work",
            "msg_excluded_bot_only",
            "msg_excluded_test",
            "msg_success",
            "msg_fail_slow",
            "msg_fail_no_response",
            "msg_fail_abandoned",
            "msg_excluded_canceled",
            "msg_excluded_unavailable",
        ],
        default="msg_unmeasurable_not_accepted",
    )
    out["Outcome"] = outcome
    return out


# --------------------------------------------------------------------------- statuses
def status_for(numerator: int, denominator: int, coverage_num: int, coverage_den: int,
               target_pct: float = TARGET_PCT, amber_floor_pct: float = AMBER_FLOOR_PCT,
               coverage_min_pct: float = COVERAGE_MIN_PCT) -> str:
    """green / amber / red / grey. Exact cross-multiplication - no float rounding at 80/70/95."""
    if denominator <= 0 or coverage_den <= 0:
        return "grey"
    if coverage_num * 100 < coverage_min_pct * coverage_den:
        return "grey"
    if numerator * 100 >= target_pct * denominator:
        return "green"
    if numerator * 100 >= amber_floor_pct * denominator:
        return "amber"
    return "red"


def pct(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else numerator / denominator * 100.0


def fmt_pct_floor(value: Optional[float], dp: int = 1) -> str:
    """Truncate (not round) so 79.96 reads 79.9 - consistent with the Amber band 70-79.9."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    f = 10 ** dp
    return f"{math.floor(value * f + 1e-9) / f:.{dp}f}%"


def fmt_ratio(numerator: int, denominator: int) -> str:
    """``412 / 480 = 85.8%`` - the reconciliation string shown beside every headline %."""
    return f"{numerator:,} / {denominator:,} = {fmt_pct_floor(pct(numerator, denominator))}"


# --------------------------------------------------------------------------- summaries
@dataclass
class ChannelSummary:
    name: str
    counts: Dict[str, int] = field(default_factory=dict)   # by outcome
    available: bool = True

    def _group(self, group: str) -> int:
        return sum(n for o, n in self.counts.items() if outcome_group(o) == group)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def numerator(self) -> int:
        return self._group("success")

    @property
    def failures(self) -> int:
        return self._group("fail")

    @property
    def denominator(self) -> int:
        return self.numerator + self.failures

    @property
    def pending(self) -> int:
        return self._group("pending")

    @property
    def unmeasurable(self) -> int:
        return self._group("unmeasurable")

    @property
    def excluded(self) -> int:
        return self._group("excluded")

    @property
    def rate_pct(self) -> Optional[float]:
        return pct(self.numerator, self.denominator)

    @property
    def coverage_pct(self) -> Optional[float]:
        return pct(self.denominator, self.denominator + self.unmeasurable)

    @property
    def status(self) -> str:
        if not self.available:
            return "grey"
        return status_for(self.numerator, self.denominator,
                          self.denominator, self.denominator + self.unmeasurable)

    def breakdown(self) -> pd.DataFrame:
        """Every outcome with its count - nothing is silently dropped."""
        rows = [{"Outcome": outcome_label(o), "Group": outcome_group(o), "Contacts": int(n)}
                for o, n in self.counts.items() if n]
        return pd.DataFrame(rows, columns=["Outcome", "Group", "Contacts"])


def summarise(df: Optional[pd.DataFrame], name: str, available: bool = True) -> ChannelSummary:
    if df is None or df.empty or "Outcome" not in df.columns:
        return ChannelSummary(name=name, counts={}, available=available)
    counts = df["Outcome"].value_counts().to_dict()
    return ChannelSummary(name=name, counts={k: int(v) for k, v in counts.items()},
                          available=available)


@dataclass
class OverallSummary:
    email: ChannelSummary
    messaging: ChannelSummary
    provisional: bool = False
    reasons: List[str] = field(default_factory=list)
    target_pct: float = TARGET_PCT

    @property
    def numerator(self) -> int:
        return self.email.numerator + self.messaging.numerator

    @property
    def denominator(self) -> int:
        return self.email.denominator + self.messaging.denominator

    @property
    def unmeasurable(self) -> int:
        return self.email.unmeasurable + self.messaging.unmeasurable

    @property
    def pending(self) -> int:
        return self.email.pending + self.messaging.pending

    @property
    def rate_pct(self) -> Optional[float]:
        return pct(self.numerator, self.denominator)

    @property
    def coverage_pct(self) -> Optional[float]:
        return pct(self.denominator, self.denominator + self.unmeasurable)

    @property
    def status(self) -> str:
        if not (self.email.available and self.messaging.available):
            return "grey"
        return status_for(self.numerator, self.denominator,
                          self.denominator, self.denominator + self.unmeasurable)

    @property
    def target_met(self) -> Optional[bool]:
        """Overall alone decides pass/fail; Email and Messaging never veto."""
        if self.status == "grey":
            return None
        return self.numerator * 100 >= self.target_pct * self.denominator

    def pp_vs(self, other: "OverallSummary") -> Optional[float]:
        """Percentage-point change against a comparison period."""
        a, b = self.rate_pct, other.rate_pct
        return None if a is None or b is None else a - b


def build_overall(email_out: Optional[pd.DataFrame], messaging_out: Optional[pd.DataFrame],
                  window_end=None, as_of=None, email_available: bool = True,
                  messaging_available: bool = True,
                  complete_tolerance: pd.Timedelta = pd.Timedelta(minutes=1)) -> OverallSummary:
    """Combine channels. ``window_end`` is exclusive; provisional if data stops short of it."""
    email = summarise(email_out, "Email", available=email_available)
    msg = summarise(messaging_out, "Messaging", available=messaging_available)
    overall = OverallSummary(email=email, messaging=msg)
    if not email_available:
        overall.reasons.append("Email source unavailable or unusable - overall cannot be calculated.")
    if not messaging_available:
        overall.reasons.append("Messaging source unavailable - overall cannot be calculated.")
    if email.pending:
        overall.provisional = True
        overall.reasons.append(
            f"{email.pending:,} Email(s) not yet due - excluded from the mature denominator.")
    if window_end is not None and as_of is not None and \
            pd.Timestamp(as_of) < pd.Timestamp(window_end) - complete_tolerance:
        overall.provisional = True
        overall.reasons.append(
            f"Data runs to {pd.Timestamp(as_of):%d %b %H:%M}, before the end of the period.")
    cov = overall.coverage_pct
    if overall.denominator and cov is not None and cov * 1 < COVERAGE_MIN_PCT and \
            overall.status == "grey":
        overall.reasons.append(
            f"Low coverage: {fmt_pct_floor(cov)} of mature contacts are measurable "
            f"(needs {COVERAGE_MIN_PCT:.0f}%).")
    if overall.denominator == 0 and email_available and messaging_available:
        overall.reasons.append("No mature eligible contacts in this period.")
    return overall


def reopened_summary(email_out: pd.DataFrame) -> Dict[str, int]:
    """New vs reopened split of mature eligible Email, from the authoritative ``Re-opened`` flag.

    Reopened cases keep their best-available ``Completion Date`` (the FINAL response), so their
    first-response result is approximate (spec 4.2). They are flagged, never silently excluded.
    """
    if email_out is None or email_out.empty or "Reopened" not in email_out.columns:
        return {"reopened_den": 0, "reopened_num": 0, "new_den": 0, "new_num": 0}
    grp = email_out["Outcome"].map(outcome_group)
    elig = grp.isin(["success", "fail"])
    ok = grp == "success"
    r = email_out["Reopened"].astype(bool)
    return {
        "reopened_den": int((elig & r).sum()), "reopened_num": int((ok & r).sum()),
        "new_den": int((elig & ~r).sum()), "new_num": int((ok & ~r).sum()),
    }


# --------------------------------------------------------------------------- windows / series
def in_window(df: pd.DataFrame, time_col: str, start, end) -> pd.DataFrame:
    """Rows whose cohort timestamp is in ``[start, end)``."""
    if df is None or df.empty or time_col not in df.columns:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    t = df[time_col]
    return df.loc[(t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))]


def _bucket_counts(df: Optional[pd.DataFrame], time_col: str, freq: str, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix} {g}" for g in GROUPS]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols, dtype="int64")
    tmp = pd.DataFrame({
        "Bucket": df[time_col].dt.floor(freq),
        "Group": df["Outcome"].map(outcome_group),
    }).dropna(subset=["Bucket"])
    wide = tmp.groupby(["Bucket", "Group"]).size().unstack(fill_value=0)
    wide = wide.reindex(columns=list(GROUPS), fill_value=0)
    wide.columns = cols
    return wide.astype("int64")


def attainment_series(email_out: Optional[pd.DataFrame], messaging_out: Optional[pd.DataFrame],
                      start, end, freq: str = "D") -> pd.DataFrame:
    """Per-day (``freq='D'``) or per-hour (``'h'``) attainment over ``[start, end)``.

    Cohort timestamps: Email ``Date/Time Opened`` (queue entry), Messaging ``Start Time``.
    Buckets with no mature eligible contact have a NaN rate (a gap, not a zero).
    """
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq, inclusive="left")
    e = _bucket_counts(email_out, "Date/Time Opened", freq, "Email")
    m = _bucket_counts(messaging_out, "Start Time", freq, "Messaging")
    out = pd.DataFrame(index=idx)
    out = out.join(e, how="left").join(m, how="left").fillna(0).astype("int64")
    for ch in ("Email", "Messaging"):
        out[f"{ch} Num"] = out[f"{ch} success"]
        out[f"{ch} Den"] = out[f"{ch} success"] + out[f"{ch} fail"]
    out["Overall Num"] = out["Email Num"] + out["Messaging Num"]
    out["Overall Den"] = out["Email Den"] + out["Messaging Den"]
    out["Unmeasurable"] = out["Email unmeasurable"] + out["Messaging unmeasurable"]
    out["Pending"] = out["Email pending"] + out["Messaging pending"]
    for ch in ("Email", "Messaging", "Overall"):
        den = out[f"{ch} Den"].where(out[f"{ch} Den"] > 0)
        out[f"{ch} %"] = out[f"{ch} Num"] / den * 100.0
    out.index.name = "Bucket"
    return out.reset_index()

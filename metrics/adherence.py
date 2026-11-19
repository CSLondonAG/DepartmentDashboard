"""Schedule Adherence (Change 2) - department-level, time-weighted.

    Adherence = productive scheduled minutes / expected productive minutes      (target 95 %)

Rules (docs/build-spec.md 2.6 and 3):

* Only time **inside a scheduled shift** counts. Presence outside a shift is ignored.
* **Productive**: ``Available_All``, ``Available_Email_and_Web``, ``Available_Chat``,
  ``Available_Case``. Verified active Email/Messaging handling is also productive and overrides a
  conflicting presence status (the overridden time is logged as a data-quality exception).
  Overlapping productive intervals are merged, never double counted.
* **Approved exclusions** (``Busy_Meeting``, ``Busy_Other_Tasks``) are removed from the expected time.
* **Lunch / Break** (``Busy_Lunch``, ``Busy_Break``, ``On_Break``) share one 60-minute-per-shift
  allowance. Break time taken up to the cap is removed from the expected time; time beyond the cap
  stays in the expected time and is non-adherent. An unused allowance is NOT auto-deducted.
* Anything else (bare ``Busy``, offline, no presence) during a shift is non-adherent.
* Absent (a scheduled shift with no productive time) scores 0 for that shift. Leave / sick cells are
  excluded from the calculation and reported separately. Blank / day-off cells are ignored.
* A late start or early finish is an incident only when the variance exceeds 5 minutes; the FULL
  variance is recorded.
* Rate is capped at 100 %. Coverage (shift time belonging to agents who can be matched to presence
  records) must be >= 95 % or the status is grey. Statuses: Green >= 95, Amber 90-94.9, Red < 90.

No agent-level output: names are used only to join the sources and are never returned.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from metrics.service_attainment import status_for, pct

TARGET_PCT = 95.0
AMBER_FLOOR_PCT = 90.0
COVERAGE_MIN_PCT = 95.0
BREAK_ALLOWANCE_SEC = 60 * 60
INCIDENT_THRESHOLD_SEC = 5 * 60

PRODUCTIVE_STATUSES = frozenset(
    {"Available_All", "Available_Email_and_Web", "Available_Chat", "Available_Case"})
APPROVED_EXCLUSION_STATUSES = frozenset({"Busy_Meeting", "Busy_Other_Tasks"})
BREAK_STATUSES = frozenset({"Busy_Lunch", "Busy_Break", "On_Break"})

Interval = Tuple[pd.Timestamp, pd.Timestamp]


# --------------------------------------------------------------------------- interval maths
def merge(intervals: Iterable[Interval]) -> List[Interval]:
    out: List[Interval] = []
    for s, e in sorted((i for i in intervals if i[1] > i[0]), key=lambda x: x[0]):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def clip(intervals: Iterable[Interval], lo, hi) -> List[Interval]:
    return [(max(s, lo), min(e, hi)) for s, e in intervals if min(e, hi) > max(s, lo)]


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    a, b = merge(a), merge(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    b = merge(b)
    for s, e in merge(a):
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
        if cur < e:
            out.append((cur, e))
    return out


def total_seconds(intervals: Iterable[Interval]) -> float:
    return float(sum((e - s).total_seconds() for s, e in merge(intervals)))


# --------------------------------------------------------------------------- names (join key only)
def person_key(name) -> str:
    """Accent-, case- and word-order-insensitive key used ONLY to join shifts to presence."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(sorted(re.findall(r"[a-z0-9]+", s.lower())))


# --------------------------------------------------------------------------- shifts parsing
_TIME = r"(\d{1,2}(?::\d{2})?\s*(?:[AaPp][Mm])?)"
_RANGE = re.compile(rf"^\s*{_TIME}\s*[-—–]\s*{_TIME}\s*$")
_BLANK = re.compile(r"(?i)^\s*(off|off day|offday|day off|na|n/a|nan|none|-|—|–)?\s*$")


def _parse_clock(txt: str, day) -> Optional[pd.Timestamp]:
    txt = txt.strip().upper().replace(" ", "")
    for fmt in ("%I:%M%p", "%I%p", "%H:%M", "%H"):
        try:
            t = pd.to_datetime(txt, format=fmt)
        except (ValueError, TypeError):
            continue
        return pd.Timestamp(day) + pd.Timedelta(hours=t.hour, minutes=t.minute)
    return None


def parse_shift_cell(text, day) -> Tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """-> (kind, start, end). kind: shift | leave | sick | absent | blank | unrecognised."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return "blank", None, None
    t = str(text).strip()
    if _BLANK.match(t):
        return "blank", None, None
    low = t.lower()
    if "sick" in low:
        return "sick", None, None
    if "leave" in low:
        return "leave", None, None
    if "absent" in low:
        return "absent", None, None
    m = _RANGE.match(t)
    if m:
        s, e = _parse_clock(m.group(1), day), _parse_clock(m.group(2), day)
        if s is not None and e is not None:
            if e <= s:
                e = e + pd.Timedelta(days=1)
            return "shift", s, e
    return "unrecognised", None, None


def shifts_long_from_wide(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Wide roster (agent + one column per dd/mm/yyyy date) -> Agent | Date | Kind | Shift Start | Shift End.

    Blank / day-off cells are dropped. Leave, sick, absent and unrecognised cells are KEPT (with
    no times) so they can be reported separately instead of silently vanishing.
    """
    cols_out = ["Agent", "Date", "Kind", "Shift Start", "Shift End"]
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=cols_out)
    cols = [str(c).strip() for c in df_raw.columns]
    df = df_raw.copy()
    df.columns = cols
    date_cols = {}
    for c in cols:
        d = pd.to_datetime(c, format="%d/%m/%Y", errors="coerce")
        if pd.isna(d):
            d = pd.to_datetime(c, errors="coerce", dayfirst=True)
        if not pd.isna(d):
            date_cols[c] = d.normalize()
    if not date_cols:
        return pd.DataFrame(columns=cols_out)
    name_col = next((c for c in cols if c not in date_cols), cols[0])
    rows = []
    for _, r in df.iterrows():
        for c, d in date_cols.items():
            kind, s, e = parse_shift_cell(r[c], d)
            if kind == "blank":
                continue
            rows.append({"Agent": r[name_col], "Date": d, "Kind": kind,
                         "Shift Start": s, "Shift End": e})
    return pd.DataFrame(rows, columns=cols_out)


# --------------------------------------------------------------------------- result
@dataclass
class AdherenceSummary:
    productive_secs: float = 0.0
    expected_secs: float = 0.0
    shifts: int = 0                   # mature shifts included in the rate
    matched_shift_secs: float = 0.0   # scheduled time of agents matched to presence records
    scheduled_shift_secs: float = 0.0  # scheduled time of all mature shifts (coverage denominator)
    unmatched_shifts: int = 0
    absent_shifts: int = 0            # scheduled, matched, but zero productive time
    leave_shifts: int = 0
    sick_shifts: int = 0
    absent_marked_shifts: int = 0     # explicit "absent" cells (no times -> not quantifiable)
    unrecognised_cells: int = 0
    pending_shifts: int = 0           # shift not finished by the data horizon
    late_incidents: int = 0
    late_variance_secs: float = 0.0
    early_incidents: int = 0
    early_variance_secs: float = 0.0
    approved_exclusion_secs: float = 0.0
    break_allowed_secs: float = 0.0
    break_over_cap_secs: float = 0.0
    handling_override_secs: float = 0.0   # DQ exception: handling during non-productive presence
    messages: List[str] = field(default_factory=list)

    @property
    def rate_pct(self) -> Optional[float]:
        r = pct(int(round(self.productive_secs)), int(round(self.expected_secs)))
        return None if r is None else min(r, 100.0)

    @property
    def coverage_pct(self) -> Optional[float]:
        return pct(int(round(self.matched_shift_secs)), int(round(self.scheduled_shift_secs)))

    @property
    def status(self) -> str:
        return status_for(int(round(self.productive_secs)), int(round(self.expected_secs)),
                          int(round(self.matched_shift_secs)),
                          int(round(self.scheduled_shift_secs)),
                          target_pct=TARGET_PCT, amber_floor_pct=AMBER_FLOOR_PCT,
                          coverage_min_pct=COVERAGE_MIN_PCT)


def _intervals(df: Optional[pd.DataFrame], key_col: str, start_col: str, end_col: str,
               status_col: Optional[str] = None) -> Dict[str, List[Tuple[Interval, str]]]:
    out: Dict[str, List[Tuple[Interval, str]]] = {}
    if df is None or df.empty:
        return out
    d = df.dropna(subset=[start_col, end_col])
    keys = d[key_col].map(person_key)
    stat = d[status_col].astype(str).str.strip() if status_col else pd.Series("", index=d.index)
    for k, s, e, st in zip(keys, d[start_col], d[end_col], stat):
        if k:
            out.setdefault(k, []).append(((s, e), st))
    return out


def compute_adherence(
    shifts: pd.DataFrame,
    presence: pd.DataFrame,
    handling: Optional[pd.DataFrame],
    start, end, as_of,
    presence_agent_col: str = "Created By: Full Name",
    presence_status_col: str = "Service Presence Status: Developer Name",
    presence_start_col: str = "Start DT",
    presence_end_col: str = "End DT Clamped",
    handling_agent_col: str = "Agent",
    handling_start_col: str = "Start",
    handling_end_col: str = "End",
) -> Tuple[AdherenceSummary, pd.DataFrame]:
    """Department adherence over shifts that START in ``[start, end)`` and finish by ``as_of``.

    ``shifts``: output of :func:`shifts_long_from_wide`. ``presence``: status intervals.
    ``handling``: verified active Email / Messaging work intervals (optional).
    Returns (summary, per-day frame with Date / Productive / Expected / Shifts / Rate %).
    """
    summ = AdherenceSummary()
    day_rows: Dict[pd.Timestamp, Dict[str, float]] = {}
    start, end, as_of = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(as_of)
    if shifts is None or shifts.empty:
        return summ, _day_frame(day_rows)

    in_win = shifts[(shifts["Date"] >= start.normalize()) & (shifts["Date"] < end)]

    # Non-shift attendance cells: reported separately, excluded from the rate.
    summ.leave_shifts = int((in_win["Kind"] == "leave").sum())
    summ.sick_shifts = int((in_win["Kind"] == "sick").sum())
    summ.absent_marked_shifts = int((in_win["Kind"] == "absent").sum())
    summ.unrecognised_cells = int((in_win["Kind"] == "unrecognised").sum())

    real = in_win[in_win["Kind"] == "shift"]
    pres = _intervals(presence, presence_agent_col, presence_start_col, presence_end_col,
                      presence_status_col)
    hand = _intervals(handling, handling_agent_col, handling_start_col, handling_end_col)

    for _, r in real.iterrows():
        s, e = r["Shift Start"], r["Shift End"]
        if e > as_of:
            summ.pending_shifts += 1
            continue
        key = person_key(r["Agent"])
        length = (e - s).total_seconds()
        summ.scheduled_shift_secs += length
        if key not in pres:
            summ.unmatched_shifts += 1       # cannot tell absent from missing data -> coverage gap
            continue
        summ.matched_shift_secs += length

        p_pres, x_pres, b_pres = [], [], []
        for (ps, pe), st in pres[key]:
            seg = clip([(ps, pe)], s, e)
            if not seg:
                continue
            if st in PRODUCTIVE_STATUSES:
                p_pres += seg
            elif st in APPROVED_EXCLUSION_STATUSES:
                x_pres += seg
            elif st in BREAK_STATUSES:
                b_pres += seg
        h = merge(clip([iv for iv, _ in hand.get(key, [])], s, e))

        productive = merge(p_pres + h)
        summ.handling_override_secs += total_seconds(subtract(h, merge(p_pres)))
        excl = subtract(merge(x_pres), productive)
        brk = subtract(subtract(merge(b_pres), productive), excl)
        brk_taken = total_seconds(brk)
        allowed = min(brk_taken, BREAK_ALLOWANCE_SEC)

        expected = length - total_seconds(excl) - allowed
        if expected <= 0:            # whole shift was approved exclusion: nothing to score
            summ.approved_exclusion_secs += total_seconds(excl)
            continue
        prod = min(total_seconds(productive), expected)
        summ.approved_exclusion_secs += total_seconds(excl)
        summ.break_allowed_secs += allowed
        summ.break_over_cap_secs += brk_taken - allowed
        summ.expected_secs += expected
        summ.productive_secs += prod
        summ.shifts += 1

        day = pd.Timestamp(r["Date"]).normalize()
        d = day_rows.setdefault(day, {"Productive": 0.0, "Expected": 0.0, "Shifts": 0})
        d["Productive"] += prod
        d["Expected"] += expected
        d["Shifts"] += 1

        if prod <= 0:
            summ.absent_shifts += 1
        else:
            late = (productive[0][0] - s).total_seconds()
            early = (e - productive[-1][1]).total_seconds()
            if late > INCIDENT_THRESHOLD_SEC:
                summ.late_incidents += 1
                summ.late_variance_secs += late           # FULL variance, not minus the threshold
            if early > INCIDENT_THRESHOLD_SEC:
                summ.early_incidents += 1
                summ.early_variance_secs += early

    if summ.unmatched_shifts:
        summ.messages.append(
            f"{summ.unmatched_shifts:,} scheduled shift(s) belong to people with no presence "
            "records at all and are excluded (coverage gap, not scored as absence).")
    if summ.handling_override_secs > 0:
        summ.messages.append(
            f"{summ.handling_override_secs / 3600:,.1f}h of verified Email/Messaging handling "
            "happened while presence was not productive; counted as productive (data-quality exception).")
    if summ.unrecognised_cells:
        summ.messages.append(
            f"{summ.unrecognised_cells:,} roster cell(s) were not understood and were ignored.")
    if summ.absent_marked_shifts:
        summ.messages.append(
            f"{summ.absent_marked_shifts:,} roster cell(s) say 'absent' with no shift times; they "
            "cannot be quantified and are reported, not scored.")
    return summ, _day_frame(day_rows)


def _day_frame(day_rows: Dict[pd.Timestamp, Dict[str, float]]) -> pd.DataFrame:
    cols = ["Date", "Productive (h)", "Expected (h)", "Shifts", "Adherence %"]
    if not day_rows:
        return pd.DataFrame(columns=cols)
    rows = []
    for day in sorted(day_rows):
        d = day_rows[day]
        rate = None if d["Expected"] <= 0 else min(d["Productive"] / d["Expected"] * 100.0, 100.0)
        rows.append({"Date": day, "Productive (h)": d["Productive"] / 3600,
                     "Expected (h)": d["Expected"] / 3600, "Shifts": int(d["Shifts"]),
                     "Adherence %": rate})
    return pd.DataFrame(rows, columns=cols)

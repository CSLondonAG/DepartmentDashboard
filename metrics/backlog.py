"""Email backlog & ageing (Change 5).

Two populations with separate clocks (docs/build-spec.md 3 "Change 5"):

* **New cases** - 4 business-hour first-response clock from ``Date/Time Opened``. Bands
  <=4 / 4-8 / 8-24 / 24-48 / >48 business hours.
* **Reopened cases** - 12 business-hour clock from the reopen (``Reopened Time``). Bands
  <=12 / 12-24 / 24-48 / >48. **Gated**: a flagged reopen with no ``Reopened Time`` (all reopens
  before 2026-11-19) cannot be placed in time, is reported as "awaiting data" and is never aged.

A case is *in the backlog at time t* when its clock has started (< t) and no response has been made
by t. Flow reconciliation: ``opening + inflow - outflow = closing`` holds by construction for each
population and is asserted/visible in the UI.

Safeguard (default for brief 9.3, which is not in the spec): Green = >= 80 % of open cases within
target AND none older than 48 business hours; Amber = not green but >= 70 % within target; Red below
70 %. An empty backlog is Green.

Population rules (documented limitations):
* A closed-type case with no response timestamp, or a response before the clock start, has no usable
  timing: it is excluded and counted ("unmeasurable"), never aged.
* For a reopened case the export's ``Completion Date`` is the FINAL response; it only counts as a
  response to the reopen when it is on/after ``Reopened Time``.
* Reopened inflow counts distinct cases (one first ``Reopened Time`` per case) - spec 5.12 default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from metrics.service_attainment import EMAIL_CLOSED_STATUSES, business_hours_between

NEW_TARGET_HOURS = 4.0
REOPENED_TARGET_HOURS = 12.0
NEW_BANDS = [("<= 4h", 0.0, 4.0), ("4-8h", 4.0, 8.0), ("8-24h", 8.0, 24.0),
             ("24-48h", 24.0, 48.0), ("> 48h", 48.0, np.inf)]
REOPENED_BANDS = [("<= 12h", 0.0, 12.0), ("12-24h", 12.0, 24.0), ("24-48h", 24.0, 48.0),
                  ("> 48h", 48.0, np.inf)]
WITHIN_TARGET_MIN_PCT = 80.0
AMBER_FLOOR_PCT = 70.0
OVER_LIMIT_HOURS = 48.0


@dataclass
class Population:
    """Clock events: ``start`` (clock start) and ``end`` (response; NaT = still open)."""
    name: str
    events: pd.DataFrame                 # columns: start, end
    unmeasurable: int = 0                # excluded for unusable timing
    awaiting_data: int = 0               # reopened flagged but no Reopened Time
    notes: List[str] = field(default_factory=list)


def build_populations(email: pd.DataFrame, closed_statuses=EMAIL_CLOSED_STATUSES) -> Dict[str, Population]:
    """Split the cleaned email frame (data.ingest output) into new / reopened clock populations."""
    empty = pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                          "end": pd.Series(dtype="datetime64[ns]")})
    new = Population("New cases", empty.copy())
    reop = Population("Reopened cases", empty.copy())
    need = {"Date/Time Opened", "Completion Date", "Status"}
    if email is None or email.empty or not need.issubset(email.columns):
        return {"new": new, "reopened": reop}

    closed = (email["Status"].astype("string").str.strip().str.casefold()
              .isin({c.casefold() for c in closed_statuses}).fillna(False).astype(bool))
    flag = email["Reopened"].astype(bool) if "Reopened" in email.columns \
        else pd.Series(False, index=email.index)
    rt = email["Reopened Time"] if "Reopened Time" in email.columns \
        else pd.Series(pd.NaT, index=email.index)
    opened, comp = email["Date/Time Opened"], email["Completion Date"]

    # ---- new cases: reopened-flag cases are NOT aged on the new-case clock (their Completion Date
    #      is the final response, not the first).
    is_new = ~flag & opened.notna()
    bad = is_new & ((comp.isna() & closed) | (comp.notna() & (comp < opened)))
    ok = is_new & ~bad
    new.events = pd.DataFrame({"start": opened[ok], "end": comp[ok]}).reset_index(drop=True)
    new.unmeasurable = int(bad.sum())

    # ---- reopened cases
    is_re = flag
    reop.awaiting_data = int((is_re & rt.isna()).sum())
    timed = is_re & rt.notna()
    resp = comp.where(comp >= rt)           # a response only counts if it follows the reopen
    unknown = timed & resp.isna() & closed  # closed again at an unknown time
    keep = timed & ~unknown
    reop.events = pd.DataFrame({"start": rt[keep], "end": resp[keep]}).reset_index(drop=True)
    reop.unmeasurable = int(unknown.sum())
    return {"new": new, "reopened": reop}


# --------------------------------------------------------------------------- snapshot / flow
def open_at(events: pd.DataFrame, t) -> pd.DataFrame:
    """Events whose clock started before ``t`` and that had no response by ``t``."""
    if events.empty:
        return events
    t = pd.Timestamp(t)
    return events[(events["start"] < t) & (events["end"].isna() | (events["end"] >= t))]


@dataclass
class Reconciliation:
    opening: int
    inflow: int
    outflow: int
    closing: int

    @property
    def reconciles(self) -> bool:
        return self.opening + self.inflow - self.outflow == self.closing


def reconcile(events: pd.DataFrame, start, end) -> Reconciliation:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if events.empty:
        return Reconciliation(0, 0, 0, 0)
    inflow = int(((events["start"] >= start) & (events["start"] < end)).sum())
    outflow = int(((events["start"] < end) & events["end"].notna()
                   & (events["end"] >= start) & (events["end"] < end)).sum())
    return Reconciliation(len(open_at(events, start)), inflow, outflow, len(open_at(events, end)))


@dataclass
class Ageing:
    target_hours: float
    bands: pd.DataFrame            # Band | Cases | Share %
    open_cases: int
    within_target: int
    over_limit: int                # > 48 business hours

    @property
    def within_target_pct(self) -> Optional[float]:
        return None if self.open_cases == 0 else self.within_target / self.open_cases * 100.0

    @property
    def status(self) -> str:
        """green / amber / red (see module docstring). An empty backlog is green."""
        if self.open_cases == 0:
            return "green"
        if self.within_target * 100 >= WITHIN_TARGET_MIN_PCT * self.open_cases and self.over_limit == 0:
            return "green"
        if self.within_target * 100 >= AMBER_FLOOR_PCT * self.open_cases:
            return "amber"
        return "red"


def ageing(events: pd.DataFrame, t, bands, target_hours: float) -> Ageing:
    """Age (business hours from clock start to ``t``) of everything open at ``t``."""
    cols = ["Band", "Cases", "Share %"]
    o = open_at(events, t)
    if o.empty:
        return Ageing(target_hours, pd.DataFrame([[b[0], 0, np.nan] for b in bands], columns=cols),
                      0, 0, 0)
    age = business_hours_between(o["start"], pd.Series(pd.Timestamp(t), index=o.index))
    age_s = (age * 3600).round(3)
    counts = []
    for label, lo, hi in bands:
        if lo == 0.0:
            m = (age_s <= hi * 3600)
        elif np.isinf(hi):
            m = age_s > lo * 3600
        else:
            m = (age_s > lo * 3600) & (age_s <= hi * 3600)
        counts.append((label, int(m.sum())))
    n = len(o)
    table = pd.DataFrame([[l, c, c / n * 100.0] for l, c in counts], columns=cols)
    return Ageing(target_hours, table, n, int((age_s <= target_hours * 3600).sum()),
                  int((age_s > OVER_LIMIT_HOURS * 3600).sum()))

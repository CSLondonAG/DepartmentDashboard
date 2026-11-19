"""Efficiency / workload diagnostics (Change 4).

DIAGNOSTIC ONLY - none of these has a target or a pass/fail status, and nothing here labels lower
handling time, more messages or more transfers as good or bad. Department aggregates only; no
agent dimension exists in any output (docs/build-spec.md 3, "Change 4").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DistStats:
    n: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    p90: Optional[float] = None
    total: float = 0.0


def dist_stats(values, allow_zero: bool = True) -> DistStats:
    """mean / median / p90 of the finite, non-negative values (NaN and negatives ignored)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    v = v[np.isfinite(v)]
    v = v[v >= 0] if allow_zero else v[v > 0]
    if v.empty:
        return DistStats()
    return DistStats(n=int(len(v)), mean=float(v.mean()), median=float(v.median()),
                     p90=float(v.quantile(0.9)), total=float(v.sum()))


@dataclass
class EmailHandling:
    cases: int
    per_case: DistStats            # total handle seconds summed over every touch of a case
    per_touch: DistStats           # handle seconds of each routing row (one touch)
    touches_per_case: DistStats
    multi_touch_cases: int         # cases touched more than once (a handling measure, NOT reopening)

    @property
    def multi_touch_pct(self) -> Optional[float]:
        return None if self.cases == 0 else self.multi_touch_cases / self.cases * 100.0


def email_handling(cases: pd.DataFrame, touches_rows: pd.Series) -> EmailHandling:
    """``cases``: one row per case with ``Total_Secs`` and ``Touches``; ``touches_rows``: seconds of
    each routing row for the same cases (per-touch handle time)."""
    if cases is None or cases.empty:
        return EmailHandling(0, DistStats(), dist_stats(touches_rows if touches_rows is not None else []),
                             DistStats(), 0)
    touches = pd.to_numeric(cases["Touches"], errors="coerce")
    return EmailHandling(
        cases=int(len(cases)),
        per_case=dist_stats(cases["Total_Secs"]),
        per_touch=dist_stats(touches_rows),
        touches_per_case=dist_stats(touches),
        multi_touch_cases=int((touches > 1).sum()),
    )


@dataclass
class MessagingDiagnostics:
    sessions: int
    handled: int
    handling_per_session: DistStats     # summed Agent Work handle seconds per handled session
    routing_assign_to_accept: DistStats  # Assign Date -> Accept Date, accepted sessions
    speed_to_answer: DistStats
    agent_messages: DistStats           # per session
    customer_messages: DistStats        # per session
    transferred: int                    # sessions worked by more than one human

    @property
    def transfer_pct_of_handled(self) -> Optional[float]:
        return None if self.handled == 0 else self.transferred / self.handled * 100.0


def messaging_diagnostics(sessions: pd.DataFrame) -> MessagingDiagnostics:
    """Session-level (one row per session) diagnostics. Expects the dashboard's session roll-up."""
    if sessions is None or sessions.empty:
        return MessagingDiagnostics(0, 0, DistStats(), DistStats(), DistStats(), DistStats(),
                                    DistStats(), 0)

    def col(name, default=np.nan):
        return sessions[name] if name in sessions.columns else pd.Series(default, index=sessions.index)

    handled_mask = col("Any_Human_Closed", False).fillna(False).astype(bool)
    assign, accept = pd.to_datetime(col("First_Assign", pd.NaT)), pd.to_datetime(col("First_Accept", pd.NaT))
    route = (accept - assign).dt.total_seconds()
    humans = pd.to_numeric(col("Human_Agents", 0), errors="coerce").fillna(0)
    return MessagingDiagnostics(
        sessions=int(len(sessions)),
        handled=int(handled_mask.sum()),
        handling_per_session=dist_stats(col("Total_Handle_Secs")[handled_mask]),
        routing_assign_to_accept=dist_stats(route),
        speed_to_answer=dist_stats(col("Avg_Speed_To_Answer")),
        agent_messages=dist_stats(col("Agent Message Count")),
        customer_messages=dist_stats(col("End User Message Count")),
        transferred=int(((humans > 1) & handled_mask).sum()),
    )


@dataclass
class UtilisationLine:
    """Department capacity use with its visible numerator / denominator (hours)."""
    name: str
    load_hours: float
    capacity_hours: float

    @property
    def rate_pct(self) -> Optional[float]:
        return None if self.capacity_hours <= 0 else min(self.load_hours / self.capacity_hours, 1.0) * 100.0

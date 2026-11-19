"""Reconcile the two Email cohorts shown on the dashboard.

* **Core Metrics "Total Emails"**: distinct cases an agent FIRST HANDLED in the period
  (``report_items.csv``, email channel).
* **Overall Service Attainment (Email)**: distinct cases that ENTERED THE QUEUE in the period
  (``email.csv`` ``Date/Time Opened``).

They legitimately differ: email.csv also holds cases that never reached an Omni-Channel queue,
and the first-handled date can be later than the opened date. This module splits the attainment
cohort by "routed to an agent" so the gap is visible and any distortion can be seen.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from metrics.service_attainment import ChannelSummary, summarise


def _keys(values: Iterable) -> set:
    return {str(v).strip() for v in values if v is not None and str(v).strip() not in ("", "nan", "<NA>")}


def email_cohort_reconciliation(
    cohort: Optional[pd.DataFrame],
    export_keys: Iterable,
    routed_keys: Iterable,
    handled_window_keys: Iterable,
    reopened_keys: Iterable,
    key_col: str = "Case Number",
) -> Dict[str, object]:
    """Counts reconciling the two cohorts, and attainment for routed vs never-routed cases.

    ``cohort``: classified email rows (``Outcome`` column) opened in the window.
    ``export_keys``: every case key in the email export. ``routed_keys``: every email case key in
    report_items. ``handled_window_keys``: the Core Metrics cases. ``reopened_keys``: flagged cases.
    """
    cohort = cohort if cohort is not None else pd.DataFrame(columns=[key_col, "Outcome"])
    ck = _keys(cohort[key_col]) if key_col in cohort.columns else set()
    export, routed = _keys(export_keys), _keys(routed_keys)
    handled, reopened = _keys(handled_window_keys), _keys(reopened_keys)

    is_routed = cohort[key_col].astype(str).str.strip().isin(routed) if len(cohort) else pd.Series(dtype=bool)
    routed_sum: ChannelSummary = summarise(cohort[is_routed] if len(cohort) else cohort, "routed")
    unrouted_sum: ChannelSummary = summarise(cohort[~is_routed] if len(cohort) else cohort, "never routed")
    return {
        "cohort_cases": len(ck),
        "cohort_rows": int(len(cohort)),                       # equals cohort_cases when distinct
        "cohort_routed": len(ck & routed),
        "cohort_never_routed": len(ck - routed),
        "core_cases": len(handled),
        "core_in_cohort": len(handled & ck),
        "core_opened_before_window": len((handled & export) - ck),
        "core_missing_from_export": len(handled - export),
        "core_reopened": len(handled & reopened),
        "cohort_reopened": len(ck & reopened),
        "routed_summary": routed_sum,
        "unrouted_summary": unrouted_sum,
    }

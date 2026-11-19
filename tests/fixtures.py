"""Synthetic fixtures in the NEW email schema (no real / production data).

Every row is hand-worked. The calendar is 07:00-22:00, 7 days a week. 10 Nov 2026 is a Tuesday,
14 Nov a Saturday, 19 Nov a Thursday.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

FIXTURE_DIR = Path(__file__).parent / "fixtures"
NEW_COLUMNS = [
    "Status", "Priority", "Case Origin", "Date/Time Opened", "Completion Date", "Re-opened",
    "Reopened Time", "Elapsed Time (Hours)", "Your Email Address", "Case Number", "Case Owner",
    "Country",
]
AS_OF = pd.Timestamp("2026-11-19 12:00")


def ts(s: Optional[str]):
    return pd.Timestamp(s) if s else pd.NaT


def email_row(case: str, opened: str, completed: Optional[str] = None, status: str = "Closed",
              reopened: int = 0, reopened_time: Optional[str] = None, origin: str = "Email",
              country: str = "Ghana") -> Dict:
    return {
        "Status": status, "Priority": "Priority 1", "Case Origin": origin,
        "Date/Time Opened": ts(opened), "Completion Date": ts(completed),
        "Re-opened": reopened, "Reopened Time": ts(reopened_time),
        "Elapsed Time (Hours)": None, "Your Email Address": f"{case}@example.test",
        "Case Number": case, "Case Owner": "Synthetic Agent", "Country": country,
    }


#: (name, row, expected outcome) - each worked by hand against the business calendar.
EMAIL_CASES: List[tuple] = [
    ("2.5h", email_row("1001", "2026-11-10 10:00", "2026-11-10 12:30"), "email_success"),
    ("exactly 4h boundary", email_row("1002", "2026-11-10 10:00", "2026-11-10 14:00"),
     "email_success"),
    ("4h01m late", email_row("1003", "2026-11-10 10:00", "2026-11-10 14:01"), "email_fail_late"),
    # 21:00-22:00 (1h) + 07:00-08:00 (1h) = 2h across the overnight pause.
    ("overnight pause", email_row("1004", "2026-11-09 21:00", "2026-11-10 08:00"),
     "email_success"),
    # Saturday counts as a business day: 10:00 -> 15:00 = 5h.
    ("saturday is a business day", email_row("1005", "2026-11-14 10:00", "2026-11-14 15:00"),
     "email_fail_late"),
    # Opened after close, answered 07:30 next day: only 30 business minutes elapse.
    ("opened at night", email_row("1006", "2026-11-09 23:00", "2026-11-10 07:30"),
     "email_success"),
    # Unanswered, 2h old at AS_OF -> not yet due.
    ("pending", email_row("1007", "2026-11-19 10:00", None, status="Open"), "email_pending"),
    # Unanswered, 07:30 -> 12:00 = 4.5h -> overdue.
    ("unanswered overdue", email_row("1008", "2026-11-19 07:30", None, status="Open"),
     "email_fail_unanswered"),
    # Unanswered across overnight: 21:30-22:00 (0.5h) + 07:00-12:00 (5h) = 5.5h -> overdue.
    ("unanswered overnight overdue", email_row("1009", "2026-11-18 21:30", None, status="Open"),
     "email_fail_unanswered"),
    ("completion before open", email_row("1010", "2026-11-10 10:00", "2026-11-10 09:00"),
     "email_unmeasurable_negative"),
    ("closed, no response ts", email_row("1011", "2026-11-10 10:00", None, status="Closed"),
     "email_unmeasurable_closed_no_response"),
    ("reopened, late final response",
     email_row("1012", "2026-11-10 10:00", "2026-11-10 16:00", reopened=1,
               reopened_time="2026-11-19 09:00"), "email_fail_late"),
    ("reopened, quick", email_row("1013", "2026-11-10 10:00", "2026-11-10 10:30", reopened=1),
     "email_success"),
]


def email_frame(cases=EMAIL_CASES) -> pd.DataFrame:
    return pd.DataFrame([row for _, row, _ in cases], columns=NEW_COLUMNS)


MSG_COLUMNS = ["Messaging Session Name", "Start Time", "Assignments", "Human_Agents",
               "Any_Human_Closed", "Any_Abandoned", "Any_Canceled", "Any_Unavailable",
               "Service Rep Accept To First Response Time"]


def msg_row(name: str, start: str, assignments: int = 1, humans: int = 1, closed: bool = False,
            abandoned: bool = False, canceled: bool = False, unavailable: bool = False,
            response=None, is_test: Optional[bool] = None) -> Dict:
    row = {
        "Messaging Session Name": name, "Start Time": ts(start), "Assignments": assignments,
        "Human_Agents": humans, "Any_Human_Closed": closed, "Any_Abandoned": abandoned,
        "Any_Canceled": canceled, "Any_Unavailable": unavailable,
        "Service Rep Accept To First Response Time": response,
    }
    if is_test is not None:
        row["Is Test"] = is_test
    return row


MSG_CASES: List[tuple] = [
    ("fast", msg_row("M1", "2026-11-10 10:00", closed=True, response=10), "msg_success"),
    ("exactly 20s", msg_row("M2", "2026-11-10 10:05", closed=True, response=20), "msg_success"),
    ("20.1s", msg_row("M3", "2026-11-10 10:10", closed=True, response=20.1), "msg_fail_slow"),
    ("handled, zero metric = no response",
     msg_row("M4", "2026-11-10 10:15", closed=True, response=0), "msg_fail_no_response"),
    ("handled, missing metric", msg_row("M5", "2026-11-10 10:20", closed=True),
     "msg_fail_no_response"),
    ("abandoned", msg_row("M6", "2026-11-10 10:25", abandoned=True), "msg_fail_abandoned"),
    ("canceled", msg_row("M7", "2026-11-10 10:30", canceled=True), "msg_excluded_canceled"),
    ("unavailable", msg_row("M8", "2026-11-10 10:35", unavailable=True),
     "msg_excluded_unavailable"),
    ("bot only", msg_row("M9", "2026-11-10 10:40", humans=0, closed=True, response=5),
     "msg_excluded_bot_only"),
    ("no agent work", msg_row("M10", "2026-11-10 10:45", assignments=0, humans=0),
     "msg_unmeasurable_no_agent_work"),
    ("routed never accepted", msg_row("M11", "2026-11-10 10:50"),
     "msg_unmeasurable_not_accepted"),
    ("test session", msg_row("M12", "2026-11-10 10:55", closed=True, response=5, is_test=True),
     "msg_excluded_test"),
    # Abandoned on one agent, then handled after transfer -> handled wins.
    ("abandoned then handled", msg_row("M13", "2026-11-10 11:00", humans=2, closed=True,
                                       abandoned=True, response=15), "msg_success"),
]


def msg_frame(cases=MSG_CASES) -> pd.DataFrame:
    df = pd.DataFrame([row for _, row, _ in cases])
    if "Is Test" in df.columns:
        df["Is Test"] = df["Is Test"].fillna(False).astype(bool)
    return df

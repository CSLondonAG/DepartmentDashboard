"""Department breakdowns of Overall Service Attainment (Change 6).

Dimensions (docs/build-spec.md 3 "Change 6"): channel; Email queue (``report_items`` joined on case
number); Email case origin (Contact Us = Contact Form, Email = Direct Email); country / market;
day of week and hour of day; new vs reopened Email (``Re-opened`` flag); transferred vs
non-transferred Messaging. There is deliberately NO agent dimension and no Priority dimension.

Safeguards: a result is only *coloured* when it has >= 30 eligible contacts AND >= 95 % coverage;
otherwise the numbers are still shown, but grey and labelled. Time is grouped by cohort time
(arrival), with an adaptive grain: daily <= 31 days, weekly 32-180, monthly > 180.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from metrics.service_attainment import (
    COVERAGE_MIN_PCT, GROUPS, outcome_group, pct, status_for, TARGET_PCT, AMBER_FLOOR_PCT,
)

MIN_CONTACTS = 30
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ORIGIN_LABELS = {"contact us": "Contact Form", "email": "Direct Email"}
UNKNOWN = "Unknown"


# --------------------------------------------------------------------------- labels
def origin_label(raw) -> str:
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)) or not str(raw).strip():
        return UNKNOWN
    return ORIGIN_LABELS.get(str(raw).strip().casefold(), str(raw).strip())


def first_queue_by_case(items: pd.DataFrame, case_col: str = "Work Item: Name",
                        queue_col: str = "Queue: Name", start_col: str = "Start DT",
                        channel_col: Optional[str] = "Service Channel: Developer Name",
                        channel_value: Optional[str] = None) -> pd.Series:
    """Case key -> queue of its EARLIEST routing row (a case routed twice has one queue)."""
    if items is None or items.empty or case_col not in items.columns or queue_col not in items.columns:
        return pd.Series(dtype="object")
    d = items
    if channel_value is not None and channel_col in items.columns:
        d = d[d[channel_col] == channel_value]
    d = d.dropna(subset=[case_col, queue_col]).sort_values(start_col, na_position="last", kind="stable")
    d = d.drop_duplicates(subset=[case_col], keep="first")
    return pd.Series(d[queue_col].values, index=d[case_col].astype(str).values)


def combine_outcomes(email_out: Optional[pd.DataFrame], messaging_out: Optional[pd.DataFrame],
                     queue_map: Optional[pd.Series] = None,
                     messaging_country_col: Optional[str] = None) -> pd.DataFrame:
    """One long frame (Channel, Time, Outcome + dimensions) from the two classified channels."""
    cols = ["Channel", "Time", "Outcome", "Country", "Origin", "Queue", "Case type", "Transfer"]
    parts = []
    if email_out is not None and not email_out.empty:
        e = email_out
        queue = (e["Case Number"].astype(str).map(queue_map).fillna("Not routed / unknown")
                 if queue_map is not None and len(queue_map) and "Case Number" in e.columns
                 else pd.Series(UNKNOWN, index=e.index))
        reopened = e["Reopened"].astype(bool) if "Reopened" in e.columns else pd.Series(False, index=e.index)
        parts.append(pd.DataFrame({
            "Channel": "Email", "Time": e["Date/Time Opened"], "Outcome": e["Outcome"],
            "Country": e["Country"] if "Country" in e.columns else UNKNOWN,
            "Origin": e["Case Origin"].map(origin_label) if "Case Origin" in e.columns else UNKNOWN,
            "Queue": queue, "Case type": reopened.map({True: "Reopened", False: "New"}),
            "Transfer": "n/a"}))
    if messaging_out is not None and not messaging_out.empty:
        m = messaging_out
        humans = pd.to_numeric(m["Human_Agents"], errors="coerce").fillna(0) \
            if "Human_Agents" in m.columns else pd.Series(0, index=m.index)
        parts.append(pd.DataFrame({
            "Channel": "Messaging", "Time": m["Start Time"], "Outcome": m["Outcome"],
            "Country": m[messaging_country_col] if messaging_country_col and
            messaging_country_col in m.columns else UNKNOWN,
            "Origin": "n/a", "Queue": "n/a", "Case type": "n/a",
            "Transfer": (humans > 1).map({True: "Transferred", False: "Not transferred"})}))
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


def add_time_dims(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["Day of week"] = pd.Categorical([], categories=DOW_ORDER, ordered=True)
        out["Hour of day"] = pd.Series(dtype="object")
        return out
    out[time_col] = pd.to_datetime(out[time_col])
    out["Day of week"] = pd.Categorical(out[time_col].dt.day_name(), categories=DOW_ORDER, ordered=True)
    out["Hour of day"] = out[time_col].dt.strftime("%H:00")
    return out


# --------------------------------------------------------------------------- tables
def dimension_table(df: pd.DataFrame, by: str, min_contacts: int = MIN_CONTACTS,
                    keep_na_groups: bool = False) -> pd.DataFrame:
    """Attainment per value of ``by``. Every % has its Successful / Eligible shown.

    Status is grey when Eligible < ``min_contacts`` or coverage < 95 %.
    """
    cols = ["Group", "Successful", "Eligible", "Attainment %", "Pending", "Unmeasurable",
            "Excluded", "Coverage %", "Status", "Note"]
    if df is None or df.empty or by not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df.assign(_g=df["Outcome"].map(outcome_group))
    d = d[d[by].notna()] if not keep_na_groups else d
    counts = d.groupby([by, "_g"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(GROUPS), fill_value=0)
    rows = []
    for key, r in counts.iterrows():
        num, den = int(r["success"]), int(r["success"] + r["fail"])
        unm = int(r["unmeasurable"])
        cov_den = den + unm
        status = status_for(num, den, den, cov_den)
        note = ""
        if den < min_contacts:
            status, note = "grey", f"fewer than {min_contacts} eligible contacts"
        elif cov_den and den * 100 < COVERAGE_MIN_PCT * cov_den:
            note = f"coverage below {COVERAGE_MIN_PCT:.0f}%"
        rows.append({"Group": str(key), "Successful": num, "Eligible": den,
                     "Attainment %": pct(num, den), "Pending": int(r["pending"]),
                     "Unmeasurable": unm, "Excluded": int(r["excluded"]),
                     "Coverage %": pct(den, cov_den), "Status": status, "Note": note})
    out = pd.DataFrame(rows, columns=cols)
    if by == "Day of week":
        out["_o"] = out["Group"].map({d: i for i, d in enumerate(DOW_ORDER)})
        out = out.sort_values("_o").drop(columns="_o")
    elif by == "Hour of day":
        out = out.sort_values("Group")
    else:
        out = out.sort_values("Eligible", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- adaptive trend
def adaptive_grain(start, end) -> str:
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return "D" if days <= 31 else ("W" if days <= 180 else "M")


def bucket_start(ts: pd.Series, grain: str) -> pd.Series:
    day = ts.dt.normalize()
    if grain == "D":
        return day
    if grain == "W":
        return day - pd.to_timedelta(ts.dt.weekday, unit="D")      # Monday-start weeks
    return day - pd.to_timedelta(ts.dt.day - 1, unit="D")           # first of month


def attainment_trend(combined: pd.DataFrame, start, end) -> pd.DataFrame:
    """Overall attainment by arrival bucket (adaptive grain). Rate is NaN with no eligible contact."""
    cols = ["Bucket", "Successful", "Eligible", "Attainment %", "Unmeasurable", "Pending"]
    if combined is None or combined.empty:
        return pd.DataFrame(columns=cols)
    d = combined[(combined["Time"] >= pd.Timestamp(start)) & (combined["Time"] < pd.Timestamp(end))]
    if d.empty:
        return pd.DataFrame(columns=cols)
    g = d["Outcome"].map(outcome_group)
    frame = pd.DataFrame({"Bucket": bucket_start(d["Time"], adaptive_grain(start, end)), "g": g})
    wide = frame.groupby(["Bucket", "g"]).size().unstack(fill_value=0).reindex(
        columns=list(GROUPS), fill_value=0)
    out = pd.DataFrame({"Successful": wide["success"], "Eligible": wide["success"] + wide["fail"],
                        "Unmeasurable": wide["unmeasurable"], "Pending": wide["pending"]})
    out["Attainment %"] = out["Successful"] / out["Eligible"].where(out["Eligible"] > 0) * 100.0
    return out.reset_index()[cols]

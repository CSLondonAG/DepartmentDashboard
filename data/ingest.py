"""Ingestion of the weekly Email export (NEW schema, adopted 2026-11-19).

Pure pandas, no Streamlit, so every rule here is unit-testable.

New export columns::

    Status | Priority | Case Origin | Date/Time Opened | Completion Date | Re-opened |
    Reopened Time | Elapsed Time (Hours) | Your Email Address | Case Number | Case Owner | Country

Two known quirks of the export are handled here:

1. **Duplicated header row** - row 2 repeats the column names, so any row whose
   ``Status`` equals ``"Status"`` is dropped.
2. **Zero-padded ``Case Number``** (``04282933`` vs ``4282933`` in report_items), which is
   normalised so the join key matches ``report_items.Work Item: Name``.

The module also degrades gracefully: an OLD-schema file (no ``Country`` / ``Re-opened``)
loads without raising and reports exactly what is missing and what that disables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- schema
# Columns without which no Email attainment can be computed at all.
EMAIL_REQUIRED_FOR_ATTAINMENT = ["Status", "Date/Time Opened", "Completion Date"]
# Columns that make up the NEW schema and what losing each one costs.
EMAIL_NEW_SCHEMA_COLUMNS: Dict[str, str] = {
    "Country": "country / market breakdowns (Change 6)",
    "Re-opened": "reopened flag - new vs reopened Email split, reopen-based measures",
    "Reopened Time": "reopened-resolution clock and ageing (needs reopens from 2026-11-19)",
}
CASE_KEY = "Case Number"
DATE_COLUMNS = ["Date/Time Opened", "Completion Date", "Reopened Time"]
EMAIL_DATE_FORMAT = "%d/%m/%Y, %H:%M"
#: Salesforce Flow starts stamping ``Reopened Time`` on this date (spec 2.2).
REOPEN_TIMING_START = pd.Timestamp("2026-11-19")

# --------------------------------------------------------------------------- country
#: Controlled mapping from the raw ``Country`` value. Everything else -> "Unknown".
COUNTRY_MAP: Dict[str, str] = {
    "DR Congo": "Democratic Republic of the Congo",
    "R Congo": "Republic of the Congo",
    "Nigeria - NairaBet": "Nigeria",
}
_CLEAN_COUNTRIES = [
    "Zimbabwe", "Malawi", "Liberia", "Namibia", "Sierra Leone", "Cameroon", "Uganda",
    "Guinea", "Togo", "Tanzania", "Angola", "Gabon", "Mali", "Senegal", "Ghana",
    "Central African Republic", "Chad",
]
COUNTRY_MAP.update({c: c for c in _CLEAN_COUNTRIES})
_COUNTRY_LOOKUP = {k.strip().casefold(): v for k, v in COUNTRY_MAP.items()}
UNKNOWN_COUNTRY = "Unknown"


def map_country(raw) -> str:
    """Map a raw export value to the controlled country name ("Unknown" if blank/unmapped)."""
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return UNKNOWN_COUNTRY
    return _COUNTRY_LOOKUP.get(str(raw).strip().casefold(), UNKNOWN_COUNTRY)


# --------------------------------------------------------------------------- helpers
def normalise_case_number(s: pd.Series) -> pd.Series:
    """Canonical string key: trimmed, no ``.0`` suffix, no leading zeros.

    ``04282933`` -> ``4282933``. An all-zero or blank value becomes ``""`` so it can never
    collide with a real case. Safe to apply to ``report_items.Work Item: Name`` as well.
    """
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0+$", "", regex=True)
    out = out.str.lstrip("0")
    return out.fillna("").astype(object)


def _parse_dt(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    s = s.astype("string").str.strip()
    parsed = pd.to_datetime(s, format=EMAIL_DATE_FORMAT, errors="coerce")
    rest = parsed.isna() & s.notna() & (s != "")
    if rest.any():  # tolerate other day-first / ISO precisions rather than dropping rows
        parsed = parsed.where(
            ~rest, pd.to_datetime(s.where(rest), format="mixed", dayfirst=True, errors="coerce"))
    return parsed


def _parse_flag(s: pd.Series) -> pd.Series:
    """0/1 (or true/false/yes/no) -> bool. Blank / unrecognised -> False."""
    txt = s.astype("string").str.strip().str.casefold()
    return txt.isin(["1", "1.0", "true", "yes", "y"]).fillna(False).astype(bool)


# --------------------------------------------------------------------------- result
@dataclass
class EmailIngest:
    """Cleaned Email frame plus a full account of what ingestion did and could not do."""
    frame: pd.DataFrame
    schema: str  # "new" | "legacy" | "unusable"
    missing_columns: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

    @property
    def can_compute_attainment(self) -> bool:
        return self.schema != "unusable"

    @property
    def has_reopen_flag(self) -> bool:
        return "Re-opened" not in self.missing_columns and self.schema != "unusable"

    @property
    def has_reopen_timing(self) -> bool:
        return reopen_timing_available(self.frame)


def reopen_timing_available(frame: pd.DataFrame) -> bool:
    """True only when at least one reopened case actually carries a ``Reopened Time``.

    Reopen *timing* (12h clock, ageing, 7-day FCR window) must be gated on this - never
    fabricated from the flag. Historical reopens (before 2026-11-19) are unrecoverable.
    """
    if frame is None or frame.empty or "Reopened Time" not in frame.columns:
        return False
    if "Reopened" not in frame.columns:
        return False
    return bool((frame["Reopened"] & frame["Reopened Time"].notna()).any())


def reopened_with_timing_mask(frame: pd.DataFrame) -> pd.Series:
    """Reopened cases whose reopen timestamp is present (the only ones timing can use)."""
    if "Reopened Time" not in frame.columns or "Reopened" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["Reopened"] & frame["Reopened Time"].notna()


# --------------------------------------------------------------------------- main entry
def clean_email_frame(raw: pd.DataFrame) -> EmailIngest:
    """Normalise a raw email export (new OR legacy schema). Never raises on schema gaps."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    stats: Dict[str, int] = {"rows_in": int(len(df))}
    messages: List[str] = []

    missing_required = [c for c in EMAIL_REQUIRED_FOR_ATTAINMENT if c not in df.columns]
    if missing_required:
        messages.append(
            "email.csv is missing required column(s) "
            + ", ".join(f"`{c}`" for c in missing_required)
            + " - Email attainment cannot be calculated."
        )
        return EmailIngest(frame=df.iloc[0:0].copy(), schema="unusable",
                           missing_columns=missing_required, stats=stats, messages=messages)

    # Quirk 1: duplicated header row.
    is_header = df["Status"].astype("string").str.strip().eq("Status").fillna(False)
    stats["header_rows_dropped"] = int(is_header.sum())
    df = df.loc[~is_header].copy()

    # Dates (dayfirst, "dd/mm/yyyy, HH:MM").
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = _parse_dt(df[col])

    # Quirk 2: zero-padded case number -> canonical join key.
    if CASE_KEY in df.columns:
        df[CASE_KEY] = normalise_case_number(df[CASE_KEY])
        blank_key = df[CASE_KEY] == ""
        stats["blank_case_numbers"] = int(blank_key.sum())
        # One row per case. Keep the answered row if the export repeats a case.
        before = len(df)
        keyed = df.loc[~blank_key]
        keyed = keyed.assign(_answered=keyed["Completion Date"].notna()).sort_values(
            "_answered", ascending=False, kind="stable"
        ).drop_duplicates(subset=[CASE_KEY], keep="first").drop(columns="_answered")
        df = pd.concat([keyed, df.loc[blank_key]]).sort_index()
        stats["duplicate_cases_dropped"] = int(before - len(df))
    else:
        stats["blank_case_numbers"] = 0
        stats["duplicate_cases_dropped"] = 0

    stats["rows_without_open_time"] = int(df["Date/Time Opened"].isna().sum())

    missing_new = [c for c in EMAIL_NEW_SCHEMA_COLUMNS if c not in df.columns]
    schema = "new" if not missing_new else "legacy"

    # Reopen flag: authoritative `Re-opened`. NEVER inferred from routing touches.
    if "Re-opened" in df.columns:
        df["Reopened"] = _parse_flag(df["Re-opened"])
    else:
        df["Reopened"] = False
        messages.append(
            "email.csv has no `Re-opened` column (legacy export) - reopened cases cannot be "
            "identified, so they are NOT flagged and reopen-based measures are unavailable. "
            "Re-export using the new schema."
        )

    if "Reopened Time" not in df.columns:
        df["Reopened Time"] = pd.NaT
        if "Re-opened" in df.columns:
            messages.append("email.csv has no `Reopened Time` column - reopen timing unavailable.")

    # Country: controlled mapping, raw preserved for audit.
    if "Country" in df.columns:
        df["Country Raw"] = df["Country"]
        df["Country"] = df["Country Raw"].map(map_country)
    else:
        df["Country Raw"] = pd.NA
        df["Country"] = UNKNOWN_COUNTRY
        messages.append("email.csv has no `Country` column - country breakdowns unavailable.")

    if schema == "legacy":
        messages.insert(
            0,
            "email.csv uses the LEGACY schema; missing new-schema column(s): "
            + ", ".join(f"`{c}`" for c in missing_new)
            + ". Email attainment still runs on Date/Time Opened -> Completion Date, but "
              "functions listed above are degraded.",
        )

    df = df.reset_index(drop=True)
    stats["rows_out"] = int(len(df))
    return EmailIngest(frame=df, schema=schema, missing_columns=missing_new,
                       stats=stats, messages=messages)


def load_email_csv(path: str | Path) -> EmailIngest:
    """Read email.csv (utf-8-sig, falling back to cp1252) and clean it."""
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            raw = pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False,
                              na_values=[""])
            return clean_email_frame(raw)
        except UnicodeDecodeError as exc:  # pragma: no cover - depends on file bytes
            last_err = exc
    raise last_err  # pragma: no cover

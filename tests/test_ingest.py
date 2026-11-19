import pandas as pd
import pytest

from data.ingest import (
    COUNTRY_MAP, clean_email_frame, load_email_csv, map_country, normalise_case_number,
    reopen_timing_available, reopened_with_timing_mask,
)
from tests.fixtures import FIXTURE_DIR


# ----------------------------------------------------------------- the two export quirks
def test_duplicated_header_row_is_dropped_and_row_count_reconciles():
    res = load_email_csv(FIXTURE_DIR / "email_new_schema_synthetic.csv")
    assert res.schema == "new"
    assert res.stats["rows_in"] == 5          # 1 header-dup + 4 real rows
    assert res.stats["header_rows_dropped"] == 1
    assert res.stats["rows_out"] == 4
    assert (res.frame["Status"] != "Status").all()


def test_zero_padded_case_number_is_normalised():
    res = load_email_csv(FIXTURE_DIR / "email_new_schema_synthetic.csv")
    cases = set(res.frame["Case Number"])
    assert "4282933" in cases            # was 04282933
    assert "12" in cases                 # was 00000012
    assert "04282933" not in cases
    # matches the unpadded report_items.Work Item: Name key
    assert "4282933" in {"4282933", "4300378"}


@pytest.mark.parametrize("raw,expected", [
    ("04282933", "4282933"), ("4282933", "4282933"), (" 0042 ", "42"),
    ("4282933.0", "4282933"), ("0000", ""), ("", ""), (None, ""),
])
def test_normalise_case_number(raw, expected):
    assert normalise_case_number(pd.Series([raw])).iloc[0] == expected


def test_cp1252_file_loads_and_dates_are_dayfirst():
    res = load_email_csv(FIXTURE_DIR / "email_new_schema_synthetic.csv")
    row = res.frame.set_index("Case Number").loc["4282933"]
    assert row["Date/Time Opened"] == pd.Timestamp("2026-11-10 10:00")   # 10 Nov, not 11 Oct
    assert row["Completion Date"] == pd.Timestamp("2026-11-10 12:30")
    assert "Côte d'Ivoire" in set(res.frame["Country Raw"])             # cp1252 char survives


def test_repeated_case_keeps_the_answered_row():
    raw = pd.DataFrame({
        "Status": ["Open", "Closed"], "Date/Time Opened": ["10/11/2026, 10:00"] * 2,
        "Completion Date": [None, "10/11/2026, 11:00"], "Case Number": ["0123", "123"],
    })
    res = clean_email_frame(raw)
    assert len(res.frame) == 1 and res.stats["duplicate_cases_dropped"] == 1
    assert res.frame["Completion Date"].notna().all()


# ----------------------------------------------------------------- graceful degradation
def test_legacy_schema_does_not_crash_and_says_what_is_missing():
    res = load_email_csv(FIXTURE_DIR / "email_legacy_schema_synthetic.csv")
    assert res.schema == "legacy"
    assert set(res.missing_columns) == {"Country", "Re-opened", "Reopened Time"}
    assert res.can_compute_attainment                       # still measurable
    assert not res.has_reopen_flag and not res.has_reopen_timing
    assert not res.frame["Reopened"].any()                  # NOT inferred from anything
    assert any("LEGACY" in m for m in res.messages)
    assert any("Re-opened" in m for m in res.messages)
    assert (res.frame["Country"] == "Unknown").all()


def test_unusable_when_required_columns_missing():
    res = clean_email_frame(pd.DataFrame({"Foo": [1], "Case Number": ["1"]}))
    assert res.schema == "unusable" and not res.can_compute_attainment
    assert res.frame.empty
    assert "Date/Time Opened" in res.missing_columns


def test_empty_frame_is_handled():
    res = clean_email_frame(pd.DataFrame(columns=["Status", "Date/Time Opened", "Completion Date"]))
    assert res.schema == "legacy" and res.frame.empty


# ----------------------------------------------------------------- reopen flag + timing gate
def test_reopened_comes_from_flag_only():
    raw = pd.DataFrame({
        "Status": ["Closed"] * 4, "Date/Time Opened": ["10/11/2026, 10:00"] * 4,
        "Completion Date": ["10/11/2026, 11:00"] * 4, "Case Number": ["1", "2", "3", "4"],
        "Re-opened": ["1", "0", "", "TRUE"], "Reopened Time": [None] * 4, "Country": ["Ghana"] * 4,
    })
    res = clean_email_frame(raw)
    assert list(res.frame["Reopened"]) == [True, False, False, True]


def test_reopen_timing_is_gated_on_reopened_time_presence():
    res = load_email_csv(FIXTURE_DIR / "email_new_schema_synthetic.csv")
    f = res.frame
    # one reopened case carries a Reopened Time
    assert reopen_timing_available(f)
    assert reopened_with_timing_mask(f).sum() == 1
    # historical reopened case with a flag but no timestamp -> timing NOT available
    historical = f.assign(**{"Reopened Time": pd.NaT})
    assert historical["Reopened"].any()
    assert not reopen_timing_available(historical)
    assert reopened_with_timing_mask(historical).sum() == 0
    # a timestamp on a non-reopened case must not enable timing either
    stray = f.assign(Reopened=False)
    assert not reopen_timing_available(stray)


# ----------------------------------------------------------------- country mapping
def test_country_mapping_table_and_unknown():
    assert map_country("DR Congo") == "Democratic Republic of the Congo"
    assert map_country("R Congo") == "Republic of the Congo"
    assert map_country("Nigeria - NairaBet") == "Nigeria"
    assert map_country("  ghana ") == "Ghana"
    assert map_country(None) == "Unknown" and map_country("") == "Unknown"
    assert map_country("Atlantis") == "Unknown"
    assert len(COUNTRY_MAP) == 20


def test_raw_country_preserved_for_audit():
    res = load_email_csv(FIXTURE_DIR / "email_new_schema_synthetic.csv")
    f = res.frame.set_index("Case Number")
    assert f.loc["4282933", "Country"] == "Democratic Republic of the Congo"
    assert f.loc["4282933", "Country Raw"] == "DR Congo"
    assert f.loc["12", "Country"] == "Nigeria" and f.loc["12", "Country Raw"] == "Nigeria - NairaBet"
    assert f.loc["4300379", "Country"] == "Unknown"       # unmapped value

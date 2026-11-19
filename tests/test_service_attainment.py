import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metrics.service_attainment import (
    EMAIL_TARGET_BUSINESS_HOURS, attainment_series, build_overall, business_hours_between,
    classify_email, classify_messaging, fmt_pct_floor, fmt_ratio, in_window, outcome_group,
    reopened_summary, status_for, summarise,
)
from tests.fixtures import AS_OF, EMAIL_CASES, MSG_CASES, email_frame, email_row, msg_frame, msg_row


def bh(a, b):
    return float(business_hours_between(pd.Series([pd.Timestamp(a)]), pd.Series([pd.Timestamp(b)])).iloc[0])


# ------------------------------------------------------------------ business calendar
@pytest.mark.parametrize("a,b,expected", [
    ("2026-11-10 10:00", "2026-11-10 12:30", 2.5),
    ("2026-11-10 21:00", "2026-11-11 08:00", 2.0),     # overnight pause
    ("2026-11-10 23:00", "2026-11-11 07:30", 0.5),     # opened after close
    ("2026-11-10 03:00", "2026-11-10 08:00", 1.0),     # opened before open
    ("2026-11-10 10:00", "2026-11-11 10:00", 15.0),    # one full 07-22 day
    ("2026-11-14 10:00", "2026-11-14 15:00", 5.0),     # Saturday counts
    ("2026-11-14 20:00", "2026-11-16 08:00", 2 + 15 + 1),  # Sat night -> Mon morning, Sunday full
    ("2026-11-10 23:00", "2026-11-11 06:00", 0.0),     # entirely outside hours
    ("2026-11-10 12:00", "2026-11-10 11:00", -1.0),    # negative when reversed
])
def test_business_hours(a, b, expected):
    assert bh(a, b) == pytest.approx(expected)


def test_business_hours_nat_is_nan():
    r = business_hours_between(pd.Series([pd.NaT]), pd.Series([pd.Timestamp("2026-11-10")]))
    assert r.isna().all()


# ------------------------------------------------------------------ email classification
def test_email_every_hand_worked_case():
    out = classify_email(email_frame(), AS_OF)
    for (name, row, expected), actual in zip(EMAIL_CASES, out["Outcome"]):
        assert actual == expected, name


def test_email_pending_becomes_fail_once_overdue():
    f = email_frame([c for c in EMAIL_CASES if c[2] == "email_pending"])
    assert classify_email(f, pd.Timestamp("2026-11-19 12:00"))["Outcome"].iloc[0] == "email_pending"
    # 10:00 + 4h business = 14:00; at exactly 14:00 still not overdue, a minute later it is.
    assert classify_email(f, pd.Timestamp("2026-11-19 14:00"))["Outcome"].iloc[0] == "email_pending"
    assert classify_email(f, pd.Timestamp("2026-11-19 14:01"))["Outcome"].iloc[0] == "email_fail_unanswered"


def test_email_overnight_pending_is_not_overdue():
    f = pd.DataFrame([email_row("1", "2026-11-18 21:30", None, status="Open")])
    # 0.5h + 07:00-09:00 (2h) = 2.5h at 09:00 -> pending
    assert classify_email(f, pd.Timestamp("2026-11-19 09:00"))["Outcome"].iloc[0] == "email_pending"


def test_email_missing_open_time_is_unmeasurable():
    f = pd.DataFrame([email_row("1", None, "2026-11-10 10:00")])
    assert classify_email(f, AS_OF)["Outcome"].iloc[0] == "email_unmeasurable_no_open_time"


def test_email_classify_empty_frame():
    out = classify_email(email_frame([]), AS_OF)
    assert out.empty and "Outcome" in out.columns


def test_email_uses_our_calendar_not_the_export_elapsed_column():
    f = email_frame([c for c in EMAIL_CASES if c[0] == "saturday is a business day"])
    f["Elapsed Time (Hours)"] = 0.1          # a lying export must not matter
    assert classify_email(f, AS_OF)["Outcome"].iloc[0] == "email_fail_late"


# ------------------------------------------------------------------ messaging classification
def test_messaging_every_hand_worked_case():
    out = classify_messaging(msg_frame())
    for (name, row, expected), actual in zip(MSG_CASES, out["Outcome"]):
        assert actual == expected, name


def test_messaging_missing_response_can_never_help_the_score():
    ok = classify_messaging(msg_frame([MSG_CASES[0]]))
    none = classify_messaging(msg_frame([("x", msg_row("X", "2026-11-10 10:00", closed=True), "")]))
    assert summarise(ok, "m").numerator == 1
    s = summarise(none, "m")
    assert s.numerator == 0 and s.denominator == 1


def test_messaging_handles_missing_optional_columns():
    out = classify_messaging(pd.DataFrame({"Assignments": [1], "Human_Agents": [1],
                                           "Any_Human_Closed": [True]}))
    assert out["Outcome"].iloc[0] == "msg_fail_no_response"
    assert classify_messaging(pd.DataFrame()).empty


# ------------------------------------------------------------------ channel summaries
def test_email_summary_counts_reconcile():
    s = summarise(classify_email(email_frame(), AS_OF), "Email")
    # success: 1001 1002 1004 1006 1013 -> 5 ; fail: 1003 1005 1008 1009 1012 -> 5
    assert (s.numerator, s.failures, s.denominator) == (5, 5, 10)
    assert s.pending == 1 and s.unmeasurable == 2 and s.excluded == 0
    assert s.total == 13 == len(EMAIL_CASES)
    assert s.rate_pct == 50.0
    assert s.coverage_pct == pytest.approx(10 / 12 * 100)       # pending not in coverage
    assert s.status == "grey"                                   # 83% coverage < 95%
    assert s.breakdown()["Contacts"].sum() == s.total


def test_messaging_summary_counts_reconcile():
    s = summarise(classify_messaging(msg_frame()), "Messaging")
    # success M1 M2 M13 ; fail M3 M4 M5 M6 ; excluded M7 M8 M9 M12 ; unmeasurable M10 M11
    assert (s.numerator, s.failures, s.denominator) == (3, 4, 7)
    assert s.excluded == 4 and s.unmeasurable == 2 and s.total == len(MSG_CASES)
    assert s.coverage_pct == pytest.approx(7 / 9 * 100)


# ------------------------------------------------------------------ statuses and thresholds
@pytest.mark.parametrize("num,den,expected", [
    (80, 100, "green"), (800, 1000, "green"), (79, 100, "amber"), (7999, 10000, "amber"),
    (70, 100, "amber"), (69, 100, "red"), (6999, 10000, "red"), (0, 10, "red"), (10, 10, "green"),
])
def test_status_thresholds_exact(num, den, expected):
    assert status_for(num, den, den, den) == expected


def test_coverage_gate_exact_95():
    assert status_for(90, 100, 95, 100) == "green"      # exactly 95% coverage -> reliable
    assert status_for(90, 100, 94, 100) == "grey"       # 94% -> grey regardless of rate
    assert status_for(10, 10, 10, 10) == "green"


def test_grey_when_nothing_mature():
    assert status_for(0, 0, 0, 0) == "grey"


def test_floor_formatting_never_rounds_up_across_a_band():
    assert fmt_pct_floor(79.96) == "79.9%"
    assert fmt_pct_floor(80.0) == "80.0%"
    assert fmt_pct_floor(None) == "-" and fmt_pct_floor(float("nan")) == "-"
    assert fmt_ratio(412, 480) == "412 / 480 = 85.8%"
    assert fmt_ratio(0, 0) == "0 / 0 = -"


# ------------------------------------------------------------------ overall
def _email(n_ok, n_late=0, n_unmeasurable=0, n_pending=0):
    rows = []
    for i in range(n_ok):
        rows.append(email_row(f"a{i}", "2026-11-10 10:00", "2026-11-10 11:00"))
    for i in range(n_late):
        rows.append(email_row(f"b{i}", "2026-11-10 10:00", "2026-11-10 18:00"))
    for i in range(n_unmeasurable):
        rows.append(email_row(f"c{i}", "2026-11-10 10:00", None))
    for i in range(n_pending):
        rows.append(email_row(f"d{i}", "2026-11-19 11:00", None, status="Open"))
    return classify_email(pd.DataFrame(rows), AS_OF)


def _msg(n_ok, n_slow=0, n_abandoned=0):
    rows = [msg_row(f"o{i}", "2026-11-10 10:00", closed=True, response=5) for i in range(n_ok)]
    rows += [msg_row(f"s{i}", "2026-11-10 10:00", closed=True, response=45) for i in range(n_slow)]
    rows += [msg_row(f"a{i}", "2026-11-10 10:00", abandoned=True) for i in range(n_abandoned)]
    return classify_messaging(pd.DataFrame(rows))


def test_overall_is_contact_weighted_not_a_mean_of_channel_rates():
    # Email 90/100 = 90 %, Messaging 5/10 = 50 %.  Mean of rates = 70 %, contact-weighted = 95/110.
    o = build_overall(_email(90, 10), _msg(5, 5))
    assert (o.numerator, o.denominator) == (95, 110)
    assert o.rate_pct == pytest.approx(95 / 110 * 100)
    assert o.rate_pct != pytest.approx(70.0)
    assert o.email.numerator + o.messaging.numerator == o.numerator


def test_overall_alone_decides_pass_fail_channels_do_not_veto():
    o = build_overall(_email(98, 2), _msg(1, 1))     # Messaging 50 % (red) but overall ~ 97 %
    assert o.messaging.status == "red"
    assert o.status == "green" and o.target_met is True


def test_overall_red_when_below_70():
    o = build_overall(_email(50, 50), _msg(10, 10))
    assert o.status == "red" and o.target_met is False


def test_overall_grey_on_low_coverage_and_says_so():
    o = build_overall(_email(90, 0, n_unmeasurable=10), _msg(10))   # 10 of 100 email unmeasurable
    assert o.status == "grey" and o.target_met is None
    assert any("Low coverage" in r for r in o.reasons)


def test_overall_coverage_uses_combined_contacts():
    o = build_overall(_email(95, 0, n_unmeasurable=0), _msg(0, 0, 0))
    assert o.coverage_pct == 100.0
    o2 = build_overall(_email(94, 0, n_unmeasurable=6), _msg(0))
    assert o2.coverage_pct == pytest.approx(94.0) and o2.status == "grey"
    o3 = build_overall(_email(95, 0, n_unmeasurable=5), _msg(0))
    assert o3.coverage_pct == pytest.approx(95.0) and o3.status == "green"


def test_pending_email_is_excluded_from_denominator_and_marks_provisional():
    o = build_overall(_email(8, 2, n_pending=5), _msg(0))
    assert o.email.denominator == 10 and o.email.pending == 5
    assert o.provisional is True
    assert any("not yet due" in r for r in o.reasons)


def test_not_provisional_when_nothing_pending_and_window_covered():
    o = build_overall(_email(8, 2), _msg(5), window_end="2026-11-11", as_of="2026-11-19 12:00")
    assert o.provisional is False


def test_provisional_when_data_stops_before_window_end():
    o = build_overall(_email(8, 2), _msg(5), window_end="2026-11-20", as_of="2026-11-19 12:00")
    assert o.provisional is True
    # data complete to 23:59 of the last day counts as covered
    o = build_overall(_email(8, 2), _msg(5), window_end="2026-11-20", as_of="2026-11-19 23:59")
    assert o.provisional is False


def test_overall_grey_when_email_source_unavailable():
    o = build_overall(None, _msg(10), email_available=False)
    assert o.status == "grey" and o.target_met is None
    assert any("Email source" in r for r in o.reasons)
    assert o.messaging.rate_pct == 100.0          # Messaging still reportable by itself


def test_overall_with_no_messaging_in_period_is_email_only_not_grey():
    o = build_overall(_email(9, 1), None)
    assert (o.numerator, o.denominator) == (9, 10) and o.status == "green"


def test_overall_empty_period():
    o = build_overall(None, None)
    assert o.rate_pct is None and o.status == "grey"
    assert any("No mature" in r for r in o.reasons)


def test_percentage_point_change():
    cur = build_overall(_email(85, 15), None)
    prev = build_overall(_email(80, 20), None)
    assert cur.pp_vs(prev) == pytest.approx(5.0)
    assert cur.pp_vs(build_overall(None, None)) is None


# ------------------------------------------------------------------ reopened
def test_reopened_summary_uses_flag_and_flags_not_excludes():
    from data.ingest import clean_email_frame
    out = classify_email(clean_email_frame(email_frame()).frame, AS_OF)   # adds `Reopened` from `Re-opened`
    r = reopened_summary(out)
    # reopened: 1012 (late -> fail) and 1013 (quick -> success)
    assert (r["reopened_den"], r["reopened_num"]) == (2, 1)
    assert r["new_den"] + r["reopened_den"] == summarise(out, "e").denominator
    assert r["new_num"] + r["reopened_num"] == summarise(out, "e").numerator


def test_reopened_summary_without_flag_column():
    assert reopened_summary(pd.DataFrame({"Outcome": ["email_success"]}))["reopened_den"] == 0


# ------------------------------------------------------------------ windows and series
def test_in_window_is_half_open():
    f = classify_email(email_frame(), AS_OF)
    w = in_window(f, "Date/Time Opened", "2026-11-10", "2026-11-11")
    assert set(w["Case Number"]) >= {"1001", "1002", "1003"}
    assert in_window(f, "Date/Time Opened", "2026-11-10 10:00", "2026-11-10 10:00").empty


def test_daily_series_reconciles_to_period_totals():
    e = classify_email(email_frame(), AS_OF)
    m = classify_messaging(msg_frame())
    s = attainment_series(e, m, "2026-11-09", "2026-11-20", "D")
    assert len(s) == 11
    ov = build_overall(e, m)
    assert s["Overall Num"].sum() == ov.numerator
    assert s["Overall Den"].sum() == ov.denominator
    day = s[s["Bucket"] == pd.Timestamp("2026-11-10")].iloc[0]
    assert day["Email Den"] == 5 and day["Messaging Den"] == 7        # cases opened / sessions started on the 10th
    assert np.isnan(s[s["Bucket"] == pd.Timestamp("2026-11-12")]["Overall %"].iloc[0])   # gap, not 0


def test_hourly_series_places_contacts_in_their_cohort_hour():
    e = classify_email(email_frame(), AS_OF)
    s = attainment_series(e, None, "2026-11-10", "2026-11-11", "h")
    assert len(s) == 24
    ten = s[s["Bucket"] == pd.Timestamp("2026-11-10 10:00")].iloc[0]
    assert ten["Email Den"] == 5      # 1001, 1002, 1003, 1012, 1013 (1010/1011 unmeasurable)
    assert s["Pending"].sum() == 0


def test_series_with_no_data_returns_gaps_not_zeros():
    s = attainment_series(None, None, "2026-11-01", "2026-11-04", "D")
    assert len(s) == 3 and s["Overall %"].isna().all() and (s["Overall Den"] == 0).all()


# ------------------------------------------------------------------ guard rails
def test_no_agent_level_columns_in_any_output():
    e = classify_email(email_frame(), AS_OF)
    m = classify_messaging(msg_frame())
    ov = build_overall(e, m)
    for frame in (attainment_series(e, m, "2026-11-09", "2026-11-20"), ov.email.breakdown(),
                  ov.messaging.breakdown()):
        joined = " ".join(map(str, frame.columns)).lower()
        assert "owner" not in joined and "agent" not in joined and "user" not in joined


def test_banned_reopen_proxy_is_gone_from_the_dashboard():
    src = (Path(__file__).resolve().parents[1] / "department_dashboard.py").read_text(encoding="utf-8")
    assert not re.search(r'\["Reopened"\]\s*=\s*email_sla_df\["Touches"\]\s*>\s*1', src)
    assert not re.search(r'Reopened.*=.*Touches.*>\s*1', src)


def test_outcome_vocabulary_is_closed():
    for o in set(classify_email(email_frame(), AS_OF)["Outcome"]) | set(classify_messaging(msg_frame())["Outcome"]):
        assert outcome_group(o) in {"success", "fail", "pending", "unmeasurable", "excluded"}
    assert EMAIL_TARGET_BUSINESS_HOURS == 4.0

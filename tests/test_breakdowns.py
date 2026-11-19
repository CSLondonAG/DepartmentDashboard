import pandas as pd
import pytest

from metrics.breakdowns import (
    MIN_CONTACTS, add_time_dims, adaptive_grain, attainment_trend, bucket_start, combine_outcomes,
    dimension_table, first_queue_by_case, origin_label,
)
from metrics.service_attainment import build_overall, classify_email, classify_messaging
from tests.fixtures import AS_OF, MSG_CASES, EMAIL_CASES, email_frame, email_row, msg_frame, msg_row


def test_origin_labels():
    assert origin_label("Contact Us") == "Contact Form"
    assert origin_label(" email ") == "Direct Email"
    assert origin_label("Web") == "Web" and origin_label(None) == "Unknown" and origin_label("") == "Unknown"


def test_first_queue_uses_earliest_routing_row_per_case_and_channel():
    items = pd.DataFrame({
        "Work Item: Name": ["1", "1", "2", "3"],
        "Queue: Name": ["Second", "First", "Q2", "ChatQ"],
        "Start DT": pd.to_datetime(["2026-11-10 12:00", "2026-11-10 10:00", "2026-11-10 10:00",
                                    "2026-11-10 10:00"]),
        "Service Channel: Developer Name": ["casesChannel", "casesChannel", "casesChannel", "sfdc_liveagent"],
    })
    q = first_queue_by_case(items, channel_value="casesChannel")
    assert q["1"] == "First" and q["2"] == "Q2" and "3" not in q.index
    assert first_queue_by_case(pd.DataFrame()).empty


def build(n_email_ok=40, n_email_late=10):
    rows = [email_row(f"a{i}", "2026-11-10 10:00", "2026-11-10 11:00", origin="Contact Us",
                      country="Ghana") for i in range(n_email_ok)]
    rows += [email_row(f"b{i}", "2026-11-11 10:00", "2026-11-11 20:00", origin="Email",
                       country="Chad") for i in range(n_email_late)]
    e = classify_email(pd.DataFrame(rows), AS_OF)
    e["Case Number"] = [r["Case Number"] for r in rows]
    return e


def test_combine_outcomes_has_dimensions_and_no_agent_columns():
    e = build(3, 1)
    m = classify_messaging(msg_frame())
    m["Messaging Country"] = "Ghana"
    c = combine_outcomes(e, m, queue_map=pd.Series({"a0": "VIP"}), messaging_country_col="Messaging Country")
    assert set(c["Channel"]) == {"Email", "Messaging"}
    assert set(c.columns) == {"Channel", "Time", "Outcome", "Country", "Origin", "Queue", "Case type", "Transfer"}
    assert c.loc[c["Channel"] == "Email", "Queue"].tolist().count("VIP") == 1
    assert (c.loc[c["Channel"] == "Email", "Queue"] == "Not routed / unknown").sum() == 3
    assert set(c.loc[c["Channel"] == "Messaging", "Transfer"]) == {"Transferred", "Not transferred"}
    assert not any("owner" in x.lower() or "agent" in x.lower() for x in c.columns)


def test_dimension_table_numerators_denominators_and_safeguard():
    c = combine_outcomes(build(40, 10), None)
    t = dimension_table(c, "Origin").set_index("Group")
    assert (t.loc["Contact Form", "Successful"], t.loc["Contact Form", "Eligible"]) == (40, 40)
    assert t.loc["Contact Form", "Status"] == "green"
    assert t.loc["Direct Email", "Eligible"] == 10 and t.loc["Direct Email", "Attainment %"] == 0.0
    assert t.loc["Direct Email", "Status"] == "grey" and "fewer than 30" in t.loc["Direct Email", "Note"]
    assert t["Successful"].sum() == 40 and t["Eligible"].sum() == 50          # reconciles to total


def test_exactly_30_contacts_is_coloured_29_is_not():
    for n, status in ((30, "green"), (29, "grey")):
        c = combine_outcomes(build(n, 0), None)
        assert dimension_table(c, "Country").iloc[0]["Status"] == status


def test_low_coverage_group_is_grey_even_with_enough_contacts():
    rows = [email_row(f"a{i}", "2026-11-10 10:00", "2026-11-10 11:00") for i in range(30)]
    rows += [email_row(f"u{i}", "2026-11-10 10:00", None, status="Closed") for i in range(5)]  # unmeasurable
    c = combine_outcomes(classify_email(pd.DataFrame(rows), AS_OF), None)
    t = dimension_table(c, "Country").iloc[0]
    assert t["Eligible"] == 30 and t["Unmeasurable"] == 5
    assert t["Status"] == "grey" and "coverage" in t["Note"]       # 30/35 = 85.7 %


def test_group_totals_match_overall_summary():
    e = build(40, 10)
    m = classify_messaging(msg_frame())
    c = combine_outcomes(e, m)
    ov = build_overall(e, m)
    t = dimension_table(c, "Channel")
    assert t["Successful"].sum() == ov.numerator and t["Eligible"].sum() == ov.denominator


def test_day_of_week_and_hour_dimensions_in_order():
    c = add_time_dims(combine_outcomes(build(40, 10), None))
    dow = dimension_table(c, "Day of week")
    assert list(dow["Group"]) == ["Tuesday", "Wednesday"]               # 10 Nov Tue, 11 Nov Wed
    hr = dimension_table(c, "Hour of day")
    assert list(hr["Group"]) == ["10:00"] and hr.iloc[0]["Eligible"] == 50


def test_new_vs_reopened_dimension_uses_the_flag():
    rows = [email_row(f"n{i}", "2026-11-10 10:00", "2026-11-10 11:00") for i in range(3)]
    rows += [email_row("r1", "2026-11-10 10:00", "2026-11-10 20:00", reopened=1)]
    from data.ingest import clean_email_frame
    e = classify_email(clean_email_frame(pd.DataFrame(rows)).frame, AS_OF)
    t = dimension_table(combine_outcomes(e, None), "Case type").set_index("Group")
    assert (t.loc["New", "Eligible"], t.loc["Reopened", "Eligible"]) == (3, 1)


def test_empty_inputs():
    assert dimension_table(pd.DataFrame(), "Origin").empty
    assert combine_outcomes(None, None).empty
    assert attainment_trend(pd.DataFrame(), "2026-11-01", "2026-11-05").empty


# ------------------------------------------------------------------ adaptive time grain
@pytest.mark.parametrize("s,e,grain", [("2026-11-01", "2026-12-02", "D"), ("2026-11-01", "2026-12-03", "W"),
                                       ("2026-01-01", "2026-06-30", "W"), ("2026-01-01", "2026-12-31", "M")])
def test_adaptive_grain(s, e, grain):
    assert adaptive_grain(s, e) == grain


def test_bucket_start():
    ts = pd.Series(pd.to_datetime(["2026-11-12 15:30", "2026-11-16 00:00", "2026-11-29 23:59"]))
    assert list(bucket_start(ts, "D")) == list(pd.to_datetime(["2026-11-12", "2026-11-16", "2026-11-29"]))
    assert list(bucket_start(ts, "W")) == list(pd.to_datetime(["2026-11-09", "2026-11-16", "2026-11-23"]))
    assert list(bucket_start(ts, "M")) == list(pd.to_datetime(["2026-11-01"] * 3))


def test_trend_groups_by_arrival_with_adaptive_grain_and_reconciles():
    c = combine_outcomes(build(40, 10), None)
    daily = attainment_trend(c, "2026-11-09", "2026-11-14")
    assert list(daily["Eligible"]) == [40, 10] and list(daily["Successful"]) == [40, 0]
    weekly = attainment_trend(c, "2026-10-01", "2026-12-01")               # 61 days -> weekly
    assert len(weekly) == 1 and weekly.iloc[0]["Bucket"] == pd.Timestamp("2026-11-09")
    assert weekly.iloc[0]["Eligible"] == 50 and weekly.iloc[0]["Attainment %"] == 80.0

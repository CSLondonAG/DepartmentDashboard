import pandas as pd
import pytest

from metrics.adherence import (
    compute_adherence, merge, parse_shift_cell, person_key, shifts_long_from_wide, subtract,
    total_seconds,
)

D = "2026-11-10"
AS_OF = pd.Timestamp("2026-11-19 12:00")
H = 3600.0


def t(hhmm, day=D):
    return pd.Timestamp(f"{day} {hhmm}")


def shift(agent="Ann Lee", s="09:00", e="18:00", day=D, kind="shift"):
    return {"Agent": agent, "Date": pd.Timestamp(day), "Kind": kind,
            "Shift Start": t(s, day) if kind == "shift" else pd.NaT,
            "Shift End": t(e, day) if kind == "shift" else pd.NaT}


def shifts(*rows):
    return pd.DataFrame(list(rows))


def presence(*rows):
    """rows: (agent, status, start, end)"""
    return pd.DataFrame([{
        "Created By: Full Name": a, "Service Presence Status: Developer Name": st,
        "Start DT": t(s), "End DT Clamped": t(e)} for a, st, s, e in rows])


def handling(*rows):
    return pd.DataFrame([{"Agent": a, "Start": t(s), "End": t(e)} for a, s, e in rows])


def run(sh, pr, hd=None, as_of=AS_OF):
    return compute_adherence(sh, pr, hd, "2026-11-09", "2026-11-16", as_of)


AV = "Available_Chat"


# ------------------------------------------------------------------ core formula
def test_fully_productive_shift_is_100():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "09:00", "18:00")))
    assert (s.productive_secs, s.expected_secs) == (9 * H, 9 * H)
    assert s.rate_pct == 100.0 and s.status == "green"


@pytest.mark.parametrize("status", ["Available_All", "Available_Email_and_Web", "Available_Chat",
                                    "Available_Case"])
def test_every_productive_status_counts(status):
    s, _ = run(shifts(shift()), presence(("Ann Lee", status, "09:00", "18:00")))
    assert s.rate_pct == 100.0


def test_presence_outside_the_shift_is_ignored():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "08:00", "19:00")))
    assert (s.productive_secs, s.expected_secs) == (9 * H, 9 * H)      # capped to the shift


def test_bare_busy_is_non_adherent():
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", AV, "09:00", "14:00"), ("Ann Lee", "Busy", "14:00", "15:00"),
        ("Ann Lee", AV, "15:00", "18:00")))
    assert (s.productive_secs, s.expected_secs) == (8 * H, 9 * H)
    assert s.rate_pct == pytest.approx(88.8888, abs=1e-3) and s.status == "red"


def test_approved_exclusion_leaves_the_expected_time():
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", AV, "09:00", "10:00"), ("Ann Lee", "Busy_Meeting", "10:00", "11:00"),
        ("Ann Lee", AV, "11:00", "18:00")))
    assert (s.productive_secs, s.expected_secs) == (8 * H, 8 * H)
    assert s.approved_exclusion_secs == 1 * H and s.rate_pct == 100.0


def test_busy_other_tasks_is_also_approved():
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", "Busy_Other_Tasks", "09:00", "10:00"), ("Ann Lee", AV, "10:00", "18:00")))
    assert s.expected_secs == 8 * H and s.rate_pct == 100.0


def test_break_within_allowance_is_removed_from_expected():
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", AV, "09:00", "12:00"), ("Ann Lee", "Busy_Lunch", "12:00", "12:45"),
        ("Ann Lee", AV, "12:45", "18:00")))
    assert s.expected_secs == 8.25 * H and s.productive_secs == 8.25 * H
    assert s.break_allowed_secs == 0.75 * H and s.break_over_cap_secs == 0


def test_lunch_and_break_share_one_60_minute_cap():
    # 45 min lunch + 30 min break = 75 min taken: 60 allowed, 15 over the cap.
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", AV, "09:00", "12:00"), ("Ann Lee", "Busy_Lunch", "12:00", "12:45"),
        ("Ann Lee", AV, "12:45", "15:00"), ("Ann Lee", "On_Break", "15:00", "15:30"),
        ("Ann Lee", AV, "15:30", "18:00")))
    assert s.break_allowed_secs == 1 * H and s.break_over_cap_secs == 0.25 * H
    assert s.expected_secs == 8 * H
    assert s.productive_secs == 7.75 * H                       # 3 + 2.25 + 2.5
    assert s.rate_pct == pytest.approx(96.875)


def test_unused_break_allowance_is_not_deducted():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "09:00", "18:00")))
    assert s.expected_secs == 9 * H and s.break_allowed_secs == 0


def test_break_over_cap_is_non_adherent():
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", AV, "09:00", "12:00"), ("Ann Lee", "Busy_Break", "12:00", "13:30"),
        ("Ann Lee", AV, "13:30", "18:00")))
    assert (s.productive_secs, s.expected_secs) == (7.5 * H, 8 * H)
    assert s.rate_pct == pytest.approx(93.75) and s.status == "amber"


def test_overlapping_productive_intervals_are_merged_not_double_counted():
    s, _ = run(shifts(shift()), presence(
        ("Ann Lee", AV, "09:00", "13:00"), ("Ann Lee", "Available_All", "12:00", "18:00")))
    assert s.productive_secs == 9 * H and s.rate_pct == 100.0


def test_handling_overrides_conflicting_presence_and_is_logged():
    s, _ = run(shifts(shift()),
               presence(("Ann Lee", "Busy", "09:00", "10:00"), ("Ann Lee", AV, "10:00", "18:00")),
               handling(("Ann Lee", "09:00", "10:00")))
    assert s.productive_secs == 9 * H and s.rate_pct == 100.0
    assert s.handling_override_secs == 1 * H
    assert any("data-quality" in m for m in s.messages)


def test_handling_during_productive_presence_is_not_an_exception():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "09:00", "18:00")),
               handling(("Ann Lee", "10:00", "11:00")))
    assert s.handling_override_secs == 0


def test_rate_is_capped_at_100():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "09:00", "18:00"),
                                         ("Ann Lee", "Available_All", "09:00", "18:00")),
               handling(("Ann Lee", "09:00", "18:00")))
    assert s.rate_pct <= 100.0 and s.productive_secs <= s.expected_secs


def test_department_rate_is_time_weighted_across_agents():
    sh = shifts(shift("Ann Lee"), shift("Bob Ray", "09:00", "13:00"))
    pr = presence(("Ann Lee", AV, "09:00", "18:00"), ("Bob Ray", AV, "09:00", "10:00"))
    s, _ = run(sh, pr)
    assert (s.productive_secs, s.expected_secs) == (10 * H, 13 * H)   # not a mean of 100% and 25%
    assert s.rate_pct == pytest.approx(10 / 13 * 100)


# ------------------------------------------------------------------ attendance
def test_absent_shift_scores_zero_when_agent_has_other_presence_data():
    sh = shifts(shift("Ann Lee"), shift("Ann Lee", day="2026-11-11"))
    pr = presence(("Ann Lee", AV, "09:00", "18:00"))        # nothing on the 11th
    s, daily = run(sh, pr)
    assert s.absent_shifts == 1 and s.shifts == 2
    assert s.rate_pct == 50.0
    assert daily.set_index("Date").loc[pd.Timestamp("2026-11-11"), "Adherence %"] == 0.0


def test_agent_with_no_presence_at_all_is_a_coverage_gap_not_absence():
    sh = shifts(shift("Ann Lee"), shift("Ghost Name"))
    s, _ = run(sh, presence(("Ann Lee", AV, "09:00", "18:00")))
    assert s.unmatched_shifts == 1 and s.shifts == 1
    assert s.rate_pct == 100.0                                # ghost not scored
    assert s.coverage_pct == 50.0 and s.status == "grey"


def test_leave_and_sick_are_excluded_and_reported_separately():
    sh = shifts(shift(), shift(kind="leave", day="2026-11-11"), shift(kind="sick", day="2026-11-12"))
    s, _ = run(sh, presence(("Ann Lee", AV, "09:00", "18:00")))
    assert (s.leave_shifts, s.sick_shifts) == (1, 1)
    assert s.shifts == 1 and s.expected_secs == 9 * H and s.rate_pct == 100.0


def test_unfinished_shift_is_pending_not_scored():
    sh = shifts(shift(day="2026-11-19"))
    s, _ = compute_adherence(sh, presence(("Ann Lee", AV, "09:00", "18:00")), None,
                             "2026-11-16", "2026-11-23", AS_OF)
    assert s.pending_shifts == 1 and s.shifts == 0 and s.rate_pct is None


def test_window_filters_by_shift_date():
    sh = shifts(shift(), shift(day="2026-11-17"))
    s, _ = run(sh, presence(("Ann Lee", AV, "09:00", "18:00")))
    assert s.shifts == 1


# ------------------------------------------------------------------ late / early incidents
def test_late_start_over_5_minutes_records_full_variance():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "09:20", "18:00")))
    assert s.late_incidents == 1 and s.late_variance_secs == 20 * 60
    assert s.early_incidents == 0


@pytest.mark.parametrize("start,incidents", [("09:05", 0), ("09:06", 1), ("09:00", 0)])
def test_incident_threshold_is_strictly_over_5_minutes(start, incidents):
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, start, "18:00")))
    assert s.late_incidents == incidents


def test_early_finish_records_full_variance():
    s, _ = run(shifts(shift()), presence(("Ann Lee", AV, "09:00", "17:30")))
    assert s.early_incidents == 1 and s.early_variance_secs == 30 * 60


def test_absent_shift_is_not_also_a_late_incident():
    sh = shifts(shift("Ann Lee"), shift("Ann Lee", day="2026-11-11"))
    s, _ = run(sh, presence(("Ann Lee", AV, "09:00", "18:00")))
    assert s.late_incidents == 0 and s.early_incidents == 0


# ------------------------------------------------------------------ statuses
@pytest.mark.parametrize("prod_h,status", [(8.55, "green"), (8.5, "amber"), (8.1, "amber"),
                                           (8.0, "red")])
def test_status_bands_on_a_9h_shift(prod_h, status):
    end = (pd.Timestamp("2026-11-10 09:00") + pd.Timedelta(hours=prod_h)).strftime("%H:%M")
    pr = presence(("Ann Lee", AV, "09:00", end))
    s, _ = run(shifts(shift()), pr)
    # 8.55h/9h = 95.0 (green), 8.5/9 = 94.4 (amber), 8.1/9 = 90 (amber), 8/9 = 88.9 (red)
    assert s.status == status


def test_empty_inputs():
    s, d = run(pd.DataFrame(), pd.DataFrame())
    assert s.rate_pct is None and s.status == "grey" and d.empty
    s, _ = run(shifts(shift()), pd.DataFrame())
    assert s.status == "grey" and s.unmatched_shifts == 1


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text,kind,s,e", [
    ("7:00 AM - 4:00 PM", "shift", "07:00", "16:00"),
    ("1:00 PM - 10:00 PM", "shift", "13:00", "22:00"),
    ("9:00 AM - 6:00 PM", "shift", "09:00", "18:00"),
    ("07:00-16:00", "shift", "07:00", "16:00"),
    ("10:00 PM - 6:00 AM", "shift", "22:00", "06:00+1"),
    ("Annual Leave", "leave", None, None), ("SICK", "sick", None, None),
    ("Absent", "absent", None, None), ("OFF", "blank", None, None),
    ("", "blank", None, None), (None, "blank", None, None), ("lunch?", "unrecognised", None, None),
])
def test_parse_shift_cell(text, kind, s, e):
    k, a, b = parse_shift_cell(text, pd.Timestamp(D))
    assert k == kind
    if s:
        assert a == t(s)
        assert b == (t(e[:5], "2026-11-11") if e.endswith("+1") else t(e))


def test_wide_to_long_keeps_leave_and_drops_blank():
    wide = pd.DataFrame({"Agent Name": ["Ann Lee", "Bob Ray"],
                         "10/11/2026": ["9:00 AM - 6:00 PM", ""],
                         "11/11/2026": ["Sick", "1:00 PM - 10:00 PM"]})
    long = shifts_long_from_wide(wide)
    assert len(long) == 3
    assert sorted(long["Kind"]) == ["shift", "shift", "sick"]
    assert long.loc[(long["Agent"] == "Bob Ray"), "Shift Start"].iloc[0] == t("13:00", "2026-11-11")


# ------------------------------------------------------------------ helpers / guard rails
def test_interval_helpers():
    a = [(t("09:00"), t("12:00")), (t("11:00"), t("13:00"))]
    assert total_seconds(a) == 4 * H and merge(a) == [(t("09:00"), t("13:00"))]
    assert subtract(a, [(t("10:00"), t("11:00"))]) == [(t("09:00"), t("10:00")),
                                                        (t("11:00"), t("13:00"))]


def test_person_key_is_order_and_accent_insensitive():
    assert person_key("Lee, Ann") == person_key("ann LEE") == person_key("Ánn Lee")
    assert person_key(None) == ""


def test_name_join_is_word_order_insensitive():
    s, _ = run(shifts(shift("Ann Lee")), presence(("Lee Ann", AV, "09:00", "18:00")))
    assert s.unmatched_shifts == 0 and s.rate_pct == 100.0


def test_no_agent_names_in_outputs():
    s, daily = run(shifts(shift("Ann Lee")), presence(("Ann Lee", AV, "09:20", "18:00")))
    assert "agent" not in " ".join(daily.columns).lower()
    assert "Ann" not in " ".join(s.messages) and "Ann" not in repr(daily.to_dict())

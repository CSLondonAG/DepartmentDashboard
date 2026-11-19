import pandas as pd
import pytest

from metrics.backlog import (
    NEW_BANDS, NEW_TARGET_HOURS, REOPENED_BANDS, REOPENED_TARGET_HOURS, Ageing, ageing,
    build_populations, open_at, reconcile,
)

AS_OF = pd.Timestamp("2026-11-19 12:00")


def row(opened, comp=None, status="Open", reopened=False, rtime=None):
    ts = lambda x: pd.Timestamp(x) if x else pd.NaT
    return {"Date/Time Opened": ts(opened), "Completion Date": ts(comp), "Status": status,
            "Reopened": reopened, "Reopened Time": ts(rtime)}


# Hand-worked ages at AS_OF (07:00-22:00, 7 days):
#  N1 2h | N2 5h | N3 10h + 5h = 15h | N4 10h + 15h + 5h = 30h | N5 10h + 4*15h + 5h = 75h
EMAIL = pd.DataFrame([
    row("2026-11-19 10:00"),                                              # N1
    row("2026-11-19 07:00"),                                              # N2
    row("2026-11-18 12:00"),                                              # N3
    row("2026-11-17 12:00"),                                              # N4
    row("2026-11-14 12:00"),                                              # N5
    row("2026-11-19 07:00", "2026-11-19 08:00", "Closed"),                # N6 answered
    row("2026-11-10 10:00", None, "Closed"),                              # N7 closed, no response
    row("2026-11-10 10:00", "2026-11-10 09:00", "Closed"),                # N8 response before open
    row("2026-11-10 10:00", "2026-11-19 10:00", "Closed", True, "2026-11-19 09:00"),  # reopened, answered
    row("2026-11-01 10:00", None, "Open", True, "2026-11-19 09:00"),      # R1 open, age 3h
    row("2026-11-01 10:00", None, "Open", True, "2026-11-18 12:00"),      # R2 open, age 15h
    row("2026-11-01 10:00", None, "Open", True, None),                    # R3 awaiting data
    row("2026-11-01 10:00", "2026-11-18 10:00", "Closed", True, "2026-11-18 09:00"),  # R4 answered
    row("2026-11-01 10:00", "2026-11-16 10:00", "Closed", True, "2026-11-17 09:00"),  # R5 unknown
    row("2026-11-01 10:00", "2026-11-16 10:00", "Open", True, "2026-11-17 09:00"),    # R6 open 33h
])
POP = build_populations(EMAIL)


def test_population_split_and_exclusions():
    new, re = POP["new"], POP["reopened"]
    assert len(new.events) == 6 and new.unmeasurable == 2          # N1-N6 usable; N7, N8 not
    assert re.awaiting_data == 1                                   # R3: flagged, no Reopened Time
    assert re.unmeasurable == 1                                    # R5: closed again, time unknown
    assert len(re.events) == 5                                     # answered-one, R1, R2, R4, R6


def test_reopened_flag_cases_are_never_aged_on_the_new_case_clock():
    # every flagged row has opened date 2026-11-01 (would be >48h) but must not appear in `new`
    assert not (POP["new"].events["start"] == pd.Timestamp("2026-11-01 10:00")).any()


def test_response_before_the_reopen_is_not_a_response_to_it():
    ev = POP["reopened"].events
    r6 = ev[ev["start"] == pd.Timestamp("2026-11-17 09:00")]
    assert len(r6) == 1 and r6["end"].isna().all()


def test_new_case_ageing_bands_and_status():
    a = ageing(POP["new"].events, AS_OF, NEW_BANDS, NEW_TARGET_HOURS)
    assert a.open_cases == 5
    assert list(a.bands["Cases"]) == [1, 1, 1, 1, 1]
    assert list(a.bands["Band"]) == ["<= 4h", "4-8h", "8-24h", "24-48h", "> 48h"]
    assert (a.within_target, a.over_limit) == (1, 1)
    assert a.within_target_pct == 20.0 and a.status == "red"
    assert a.bands["Share %"].sum() == pytest.approx(100.0)


def test_reopened_ageing_uses_the_12h_clock_from_the_reopen():
    a = ageing(POP["reopened"].events, AS_OF, REOPENED_BANDS, REOPENED_TARGET_HOURS)
    # R1 3h, R2 15h, R6 33h; the other reopened rows were answered
    assert a.open_cases == 3 and list(a.bands["Cases"]) == [1, 1, 1, 0]
    assert a.within_target == 1 and a.over_limit == 0 and a.status == "red"


@pytest.mark.parametrize("opened,band", [
    ("2026-11-19 08:00", 0),      # exactly 4h -> within target
    ("2026-11-19 07:59", 1),      # 4h01m -> 4-8h
])
def test_band_boundaries_at_4_hours(opened, band):
    ev = pd.DataFrame({"start": [pd.Timestamp(opened)], "end": [pd.NaT]})
    a = ageing(ev, AS_OF, NEW_BANDS, NEW_TARGET_HOURS)
    assert list(a.bands["Cases"]).index(1) == band


def test_exactly_48h_is_not_over_the_limit():
    # 12:00 on the 17th -> 30h; need 48h: open 2026-11-16 21:00 -> 1 + 15 + 15 + 5 = 36 ... use 15h days
    start = pd.Timestamp("2026-11-16 07:00")         # 3 full days (16,17,18) = 45h + 07-10 (3h) = 48h
    ev = pd.DataFrame({"start": [start], "end": [pd.NaT]})
    a = ageing(ev, pd.Timestamp("2026-11-19 10:00"), NEW_BANDS, NEW_TARGET_HOURS)
    assert a.over_limit == 0 and list(a.bands["Cases"]) == [0, 0, 0, 1, 0]


def test_status_safeguards():
    def mk(within, old):
        return Ageing(4.0, pd.DataFrame(), within + old + 0, within, old)
    assert Ageing(4.0, pd.DataFrame(), 0, 0, 0).status == "green"               # empty backlog
    assert mk(5, 0).status == "green"
    assert Ageing(4.0, pd.DataFrame(), 5, 4, 0).status == "green"               # exactly 80 %
    assert Ageing(4.0, pd.DataFrame(), 5, 4, 1).status == "amber"               # 80 % but one > 48h
    assert Ageing(4.0, pd.DataFrame(), 10, 7, 0).status == "amber"              # 70 %
    assert Ageing(4.0, pd.DataFrame(), 10, 6, 0).status == "red"                # 60 %


def test_empty_backlog_ageing():
    a = ageing(pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"),
                             "end": pd.Series(dtype="datetime64[ns]")}), AS_OF, NEW_BANDS, 4.0)
    assert a.open_cases == 0 and a.within_target_pct is None and a.status == "green"
    assert list(a.bands["Cases"]) == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("start,end,expected", [
    ("2026-11-18", "2026-11-20", (2, 4, 1, 5)),
    ("2026-11-19", "2026-11-20", (3, 3, 1, 5)),
])
def test_new_case_flow_reconciles(start, end, expected):
    r = reconcile(POP["new"].events, start, end)
    assert (r.opening, r.inflow, r.outflow, r.closing) == expected
    assert r.reconciles


def test_reopened_flow_reconciles():
    r = reconcile(POP["reopened"].events, "2026-11-18", "2026-11-20")
    assert (r.opening, r.inflow, r.outflow, r.closing) == (1, 4, 2, 3)  # R6 | Ra R1 R2 R4 | Ra R4 | R1 R2 R6
    assert r.reconciles


def test_flow_identity_holds_for_every_window():
    ev = POP["new"].events
    days = pd.date_range("2026-11-10", "2026-11-21")
    for s in days:
        for e in days[days > s]:
            assert reconcile(ev, s, e).reconciles


def test_open_at_is_time_travel_safe():
    ev = POP["new"].events
    # at 07:30 on the 19th: N2 and N6 (opened 07:00; N6 answered 08:00) plus N3, N4, N5 are open
    assert len(open_at(ev, "2026-11-19 07:30")) == 5
    assert len(open_at(ev, AS_OF)) == 5          # N1..N5 (N6 answered, N2 still open)


def test_missing_columns_or_empty_input_degrade():
    pop = build_populations(pd.DataFrame({"Foo": [1]}))
    assert pop["new"].events.empty and pop["reopened"].awaiting_data == 0
    assert build_populations(None)["new"].events.empty

import dataclasses

import pandas as pd
import pytest

from metrics.workload import (
    UtilisationLine, dist_stats, email_handling, messaging_diagnostics,
)


def test_dist_stats_hand_worked():
    s = dist_stats([10, 20, 30, 40, 100])
    assert (s.n, s.mean, s.median, s.total) == (5, 40.0, 30.0, 200.0)
    assert s.p90 == pytest.approx(76.0)            # linear interpolation: 40 + 0.6*(100-40)


def test_dist_stats_ignores_nan_negative_and_inf():
    s = dist_stats([1, None, float("nan"), -5, float("inf"), 3])
    assert (s.n, s.mean) == (2, 2.0)
    assert dist_stats([]).n == 0 and dist_stats([]).mean is None
    assert dist_stats([0, 0], allow_zero=False).n == 0


def test_email_handling_per_case_per_touch_and_multi_touch():
    cases = pd.DataFrame({"Total_Secs": [100, 300, 600], "Touches": [1, 1, 3]})
    rows = pd.Series([100, 300, 200, 200, 200])     # 5 routing rows = 1 + 1 + 3 touches
    h = email_handling(cases, rows)
    assert h.cases == 3 and h.per_case.mean == pytest.approx(333.3333, abs=1e-3)
    assert h.per_case.median == 300 and h.per_touch.n == 5 and h.per_touch.mean == 200
    assert h.touches_per_case.mean == pytest.approx(5 / 3) and h.multi_touch_cases == 1
    assert h.multi_touch_pct == pytest.approx(100 / 3)


def test_email_handling_empty():
    h = email_handling(pd.DataFrame(), pd.Series(dtype="float64"))
    assert h.cases == 0 and h.multi_touch_pct is None


def sessions():
    t = pd.Timestamp("2026-11-10 10:00")
    return pd.DataFrame({
        "Any_Human_Closed": [True, True, False, True],
        "First_Assign": [t, t, t, t],
        "First_Accept": [t + pd.Timedelta(seconds=10), t + pd.Timedelta(seconds=30), pd.NaT,
                         t + pd.Timedelta(seconds=20)],
        "Human_Agents": [1, 2, 1, 3],
        "Total_Handle_Secs": [100, 200, 999, 300],
        "Avg_Speed_To_Answer": [5, 15, None, 10],
        "Agent Message Count": [4, 6, 0, 8], "End User Message Count": [3, 5, 1, 7],
    })


def test_messaging_diagnostics():
    d = messaging_diagnostics(sessions())
    assert (d.sessions, d.handled) == (4, 3)
    assert d.handling_per_session.n == 3 and d.handling_per_session.mean == 200   # 999 unhandled ignored
    assert d.routing_assign_to_accept.n == 3 and d.routing_assign_to_accept.median == 20
    assert d.speed_to_answer.mean == 10
    assert d.agent_messages.total == 18 and d.customer_messages.median == 4
    assert d.transferred == 2 and d.transfer_pct_of_handled == pytest.approx(200 / 3)


def test_messaging_diagnostics_empty_and_missing_columns():
    assert messaging_diagnostics(pd.DataFrame()).sessions == 0
    d = messaging_diagnostics(pd.DataFrame({"Any_Human_Closed": [True]}))
    assert d.handled == 1 and d.routing_assign_to_accept.n == 0 and d.transferred == 0


def test_utilisation_line_is_capped_and_has_visible_parts():
    assert UtilisationLine("c", 5, 10).rate_pct == 50.0
    assert UtilisationLine("c", 12, 10).rate_pct == 100.0
    assert UtilisationLine("c", 1, 0).rate_pct is None


def test_diagnostics_carry_no_status_target_or_agent_fields():
    names = set()
    for cls in (type(email_handling(pd.DataFrame(), [])), type(messaging_diagnostics(None))):
        names |= {f.name.lower() for f in dataclasses.fields(cls)}
    assert not any(any(w in n for w in ("status", "target", "agent_name", "owner", "pass"))
                   for n in names)

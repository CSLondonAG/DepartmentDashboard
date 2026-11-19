import pandas as pd

from metrics.reconcile import email_cohort_reconciliation


def cohort():
    # a1..a4 routed, u1..u2 never routed; a4 and u2 fail, the rest succeed
    return pd.DataFrame({
        "Case Number": ["a1", "a2", "a3", "a4", "u1", "u2"],
        "Outcome": ["email_success", "email_success", "email_success", "email_fail_late",
                    "email_success", "email_fail_late"],
    })


def test_cohorts_reconcile_and_routed_split_sums_to_the_total():
    r = email_cohort_reconciliation(
        cohort(), export_keys=["a1", "a2", "a3", "a4", "u1", "u2", "old1", "old2"],
        routed_keys=["a1", "a2", "a3", "a4", "old1", "gone"],
        handled_window_keys=["a1", "a2", "old1", "gone"], reopened_keys=["a2", "u1", "old1"])
    assert (r["cohort_cases"], r["cohort_rows"]) == (6, 6)
    assert (r["cohort_routed"], r["cohort_never_routed"]) == (4, 2)
    assert r["core_cases"] == 4
    assert r["core_in_cohort"] == 2                  # a1, a2
    assert r["core_opened_before_window"] == 1       # old1 is in the export but opened earlier
    assert r["core_missing_from_export"] == 1        # gone
    assert r["core_in_cohort"] + r["core_opened_before_window"] + r["core_missing_from_export"] == r["core_cases"]
    assert (r["core_reopened"], r["cohort_reopened"]) == (2, 2)
    a, u = r["routed_summary"], r["unrouted_summary"]
    assert (a.numerator, a.denominator) == (3, 4) and (u.numerator, u.denominator) == (1, 2)
    assert a.denominator + u.denominator == 6


def test_duplicate_rows_would_show_as_rows_greater_than_cases():
    dup = pd.concat([cohort(), cohort().iloc[:1]], ignore_index=True)
    r = email_cohort_reconciliation(dup, [], [], [], [])
    assert r["cohort_rows"] == 7 and r["cohort_cases"] == 6


def test_empty_inputs():
    r = email_cohort_reconciliation(None, [], [], [], [])
    assert r["cohort_cases"] == 0 and r["routed_summary"].denominator == 0

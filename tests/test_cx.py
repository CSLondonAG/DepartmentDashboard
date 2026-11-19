import pandas as pd
import pytest

from metrics.cx import (
    adaptive_freq, classify_question, csat_summary, csat_trend, cx_status, in_window,
    nps_summary, operational_fcr, prepare_survey, survey_fcr_summary, unmapped_questions,
)

Q_CSAT = "How satisfied were you with the Customer Service you received today?"
Q_NPS_P = "How likely are you to recommend Premier Bet to a friend or colleague?"
Q_NPS_M = "How likely would you to to recommend Mercury Bet to a friend or colleague?"
Q_FCR_CHAT = "Was the issue reported in the chat resolved by the agent?"
Q_FCR_EMAIL = "Was the issue reported in the email resolved by the agent?"


def survey(rows):
    return pd.DataFrame([{"Survey Taker: ID": i, "Survey Taker: Created Date": pd.Timestamp(d),
                          "Survey Question: Question Title": q, "Response": r}
                         for i, d, q, r in rows])


# ------------------------------------------------------------------ question mapping
@pytest.mark.parametrize("title,expected", [
    (Q_CSAT, "CSAT"), (Q_NPS_P, "NPS"), (Q_NPS_M, "NPS"),
    (Q_FCR_CHAT, "FCR_CHAT"), (Q_FCR_EMAIL, "FCR_EMAIL"),
    ("Something else entirely", None), (None, None),
])
def test_classify_question(title, expected):
    assert classify_question(title) == expected


def test_unmapped_questions_are_listed():
    df = survey([(1, "2026-11-10", "Rate our biscuits", "5"), (2, "2026-11-10", Q_CSAT, "5")])
    assert unmapped_questions(df) == ["Rate our biscuits"]


# ------------------------------------------------------------------ CSAT
def test_csat_is_percent_rating_4_or_5_of_valid():
    rows = [(i, "2026-11-10", Q_CSAT, r) for i, r in enumerate(
        ["5 - Very Satisfied >", "4", "5", "3", "2", "1 - <Not Satisfied", "1"], start=1)]
    s = csat_summary(prepare_survey(survey(rows)))
    assert (s.numerator, s.denominator) == (3, 7) and s.rate_pct == pytest.approx(3 / 7 * 100)


def test_csat_invalid_responses_are_excluded_and_counted():
    rows = [(1, "2026-11-10", Q_CSAT, "5"), (2, "2026-11-10", Q_CSAT, "9"),
            (3, "2026-11-10", Q_CSAT, "no idea"), (4, "2026-11-10", Q_CSAT, None)]
    s = csat_summary(prepare_survey(survey(rows)))
    assert (s.numerator, s.denominator, s.invalid) == (1, 1, 3)


def test_repeated_respondent_counts_once_per_measure():
    rows = [(1, "2026-11-10", Q_CSAT, "5"), (1, "2026-11-10", Q_CSAT, "1")]
    assert csat_summary(prepare_survey(survey(rows))).denominator == 1


def test_one_respondent_can_answer_different_measures():
    rows = [(1, "2026-11-10", Q_CSAT, "5"), (1, "2026-11-10", Q_FCR_CHAT, "Yes"),
            (1, "2026-11-10", Q_NPS_P, "10 - Very Likely >")]
    p = prepare_survey(survey(rows))
    assert csat_summary(p).denominator == 1 and survey_fcr_summary(p).denominator == 1
    assert nps_summary(p).n == 1


# ------------------------------------------------------------------ min sample and status
def test_status_requires_30_responses():
    assert cx_status(29, 29) == "grey"
    assert cx_status(24, 30) == "green" and cx_status(23, 30) == "amber"
    assert cx_status(21, 30) == "amber" and cx_status(20, 30) == "red"
    assert cx_status(0, 0) == "grey"


def test_csat_status_and_sufficiency():
    rows = [(i, "2026-11-10", Q_CSAT, "5") for i in range(29)]
    s = csat_summary(prepare_survey(survey(rows)))
    assert not s.sufficient and s.status == "grey" and s.rate_pct == 100.0   # shown, not coloured
    rows.append((99, "2026-11-10", Q_CSAT, "5"))
    s = csat_summary(prepare_survey(survey(rows)))
    assert s.sufficient and s.status == "green"


# ------------------------------------------------------------------ survey FCR
def test_survey_fcr_channel_split_comes_from_the_question():
    rows = ([(i, "2026-11-10", Q_FCR_CHAT, "Yes") for i in range(40)]
            + [(100 + i, "2026-11-10", Q_FCR_CHAT, "No") for i in range(10)]
            + [(200 + i, "2026-11-10", Q_FCR_EMAIL, "Yes") for i in range(5)]
            + [(300, "2026-11-10", Q_FCR_EMAIL, "No, I had to call back")])
    p = prepare_survey(survey(rows))
    chat, email, both = (survey_fcr_summary(p, "chat"), survey_fcr_summary(p, "email"),
                         survey_fcr_summary(p))
    assert (chat.numerator, chat.denominator) == (40, 50) and chat.sufficient
    assert (email.numerator, email.denominator) == (5, 6) and not email.sufficient
    assert (both.numerator, both.denominator) == (45, 56)


def test_fcr_ignores_unparseable_answers():
    rows = [(1, "2026-11-10", Q_FCR_CHAT, "Yes"), (2, "2026-11-10", Q_FCR_CHAT, "not resolved"),
            (3, "2026-11-10", Q_FCR_CHAT, "Maybe")]
    s = survey_fcr_summary(prepare_survey(survey(rows)), "chat")
    assert (s.numerator, s.denominator, s.invalid) == (1, 1, 2)


# ------------------------------------------------------------------ NPS
def test_nps_counts_both_brands_and_has_no_status():
    rows = ([(i, "2026-11-10", Q_NPS_P, "10 - Very Likely >") for i in range(12)]       # promoters
            + [(100 + i, "2026-11-10", Q_NPS_M, "9") for i in range(8)]                 # promoters
            + [(200 + i, "2026-11-10", Q_NPS_P, "8") for i in range(5)]                 # passives
            + [(300 + i, "2026-11-10", Q_NPS_P, "0 - < Very Unlikely") for i in range(5)])
    n = nps_summary(prepare_survey(survey(rows)))
    assert (n.promoters, n.passives, n.detractors, n.n) == (20, 5, 5, 30)
    assert n.score == pytest.approx((20 - 5) / 30 * 100)
    assert not hasattr(n, "status")


def test_nps_boundaries_and_suppression():
    rows = [(i, "2026-11-10", Q_NPS_P, str(v)) for i, v in enumerate([6, 7, 8, 9, 10, 0, 11], 1)]
    n = nps_summary(prepare_survey(survey(rows)))
    assert (n.promoters, n.passives, n.detractors, n.invalid) == (2, 2, 2, 1)
    assert n.score is None          # 6 responses < 30


# ------------------------------------------------------------------ windows
def test_in_window_uses_survey_created_date():
    rows = [(1, "2026-11-10 10:00", Q_CSAT, "5"), (2, "2026-11-11", Q_CSAT, "5")]
    p = prepare_survey(survey(rows))
    assert len(in_window(p, "2026-11-10", "2026-11-11")) == 1


# ------------------------------------------------------------------ operational FCR
def case(resolved, reopened=False, reopened_time=None, status="Closed"):
    return {"Status": status, "Completion Date": pd.Timestamp(resolved) if resolved else pd.NaT,
            "Reopened": reopened,
            "Reopened Time": pd.Timestamp(reopened_time) if reopened_time else pd.NaT}


AS_OF = pd.Timestamp("2026-12-10 12:00")
WIN = ("2026-11-19", "2026-12-01")


def test_operational_fcr_basic_ratio_with_visible_counts():
    df = pd.DataFrame(
        [case("2026-11-20 10:00") for _ in range(3)]
        + [case("2026-11-20 10:00", True, "2026-11-22 10:00")]          # reopened in 2 days -> fail
        + [case("2026-11-20 10:00", True, "2026-12-01 10:00")])         # reopened day 11 -> success
    s = operational_fcr(df, AS_OF, *WIN)
    assert (s.numerator, s.denominator) == (4, 5) and s.rate_pct == 80.0
    assert s.awaiting == 0 and s.invalid == 0


def test_reopen_exactly_at_7_days_counts_as_within():
    df = pd.DataFrame([case("2026-11-20 10:00", True, "2026-11-27 10:00"),
                       case("2026-11-20 10:00", True, "2026-11-27 10:01")])
    s = operational_fcr(df, AS_OF, *WIN)
    assert (s.numerator, s.denominator) == (1, 2)


def test_incomplete_7_day_window_is_awaiting_not_scored():
    df = pd.DataFrame([case("2026-11-20 10:00"), case("2026-12-08 10:00")])
    s = operational_fcr(df, AS_OF, "2026-11-19", "2026-12-20")
    assert (s.numerator, s.denominator, s.awaiting) == (1, 1, 1)     # 8 Dec + 7d > as_of


def test_historical_resolutions_before_reopen_timing_exists_are_awaiting_data():
    df = pd.DataFrame([case("2026-11-10 10:00"), case("2026-11-10 10:00", True)])
    s = operational_fcr(df, AS_OF, "2026-11-01", "2026-11-19")
    assert (s.numerator, s.denominator, s.awaiting) == (0, 0, 2)
    assert s.rate_pct is None and s.status == "grey"


def test_flagged_reopen_without_time_is_excluded_not_counted_as_success():
    df = pd.DataFrame([case("2026-11-20 10:00"), case("2026-11-20 10:00", True, None)])
    s = operational_fcr(df, AS_OF, *WIN)
    assert (s.numerator, s.denominator, s.invalid) == (1, 1, 1)


def test_unresolved_and_out_of_window_cases_are_ignored():
    df = pd.DataFrame([case(None, status="Open"), case("2026-12-05 10:00"),
                       case("2026-11-20 10:00", status="Open")])
    s = operational_fcr(df, AS_OF, *WIN)
    assert s.denominator == 0 and s.awaiting == 0


def test_operational_fcr_unavailable_without_reopen_columns():
    s = operational_fcr(pd.DataFrame({"Status": ["Closed"]}), AS_OF, *WIN)
    assert s.unavailable_reason and s.status == "grey"


# ------------------------------------------------------------------ trend
def test_adaptive_frequency():
    assert adaptive_freq("2026-11-01", "2026-12-02") == "D"          # 31 days
    assert adaptive_freq("2026-11-01", "2026-12-03") == "W-MON"      # 32 days
    assert adaptive_freq("2026-01-01", "2026-06-30") == "W-MON"
    assert adaptive_freq("2026-01-01", "2026-12-31") == "MS"


def test_csat_trend_suppresses_buckets_below_minimum():
    rows = ([(i, "2026-11-10", Q_CSAT, "5") for i in range(30)]
            + [(100 + i, "2026-11-11", Q_CSAT, "5") for i in range(5)])
    t = csat_trend(prepare_survey(survey(rows)), "2026-11-09", "2026-11-12")
    t = t.set_index("Bucket")
    assert t.loc[pd.Timestamp("2026-11-10"), "CSAT %"] == 100.0
    assert pd.isna(t.loc[pd.Timestamp("2026-11-11"), "CSAT %"])
    assert t.loc[pd.Timestamp("2026-11-11"), "Valid"] == 5            # still visible


def test_empty_survey():
    p = prepare_survey(pd.DataFrame())
    assert csat_summary(p).denominator == 0 and nps_summary(p).n == 0
    assert csat_trend(p, "2026-11-01", "2026-11-05").empty

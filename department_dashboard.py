# ---- CSAT (line) & NPS (bars) Trend ----
daily_survey = (survey_period
    .assign(Date=survey_period["Survey Date"].dt.normalize())
    .groupby("Date", as_index=False)
    .agg(
        CSAT_pct=("CSAT%","mean"),
        NPS=("NPS_raw", _nps_from_0_10),
        Surveys=("CSAT%","size")
    )
)
if not daily_survey.empty:
    st.subheader("CSAT & NPS Trend")

    tick_dates2 = daily_survey["Date"].tolist()
    x_min2 = datetime.combine(start_date, datetime.min.time()) - timedelta(days=0.5)
    x_max2 = datetime.combine(end_date,   datetime.max.time()) + timedelta(days=0.5)

    base = alt.Chart(daily_survey).encode(
        x=alt.X("Date:T",
                axis=alt.Axis(format="%d %b", labelAngle=-45, values=tick_dates2),
                scale=alt.Scale(domain=[x_min2, x_max2]),
                title="Date")
    )

    # CSAT: line + labels (left axis)
    csat_line = base.mark_line(point=True, color="#2F80ED").encode(
        y=alt.Y("CSAT_pct:Q", title="CSAT (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("Date:T", format="%d %b"),
            alt.Tooltip("CSAT_pct:Q", format=".1f", title="CSAT (%)"),
            alt.Tooltip("NPS:Q", format=".1f", title="NPS"),
            alt.Tooltip("Surveys:Q", format="d", title="# Surveys")
        ]
    )
    csat_labels = base.mark_text(dy=-10, color="#2F80ED").encode(
        y="CSAT_pct:Q",
        text=alt.Text("CSAT_pct:Q", format=".1f")
    )

    # NPS: bars + value labels (right axis)
    nps_bars = base.mark_bar(color="#27AE60", opacity=0.45).encode(
        y=alt.Y("NPS:Q",
                title="NPS",
                axis=alt.Axis(orient="right"),
                scale=alt.Scale(domain=[-100, 100])),
        tooltip=[
            alt.Tooltip("Date:T", format="%d %b"),
            alt.Tooltip("NPS:Q", format=".1f", title="NPS"),
            alt.Tooltip("CSAT_pct:Q", format=".1f", title="CSAT (%)"),
            alt.Tooltip("Surveys:Q", format="d", title="# Surveys")
        ]
    )
    nps_labels = base.mark_text(
        dy=-6, color="#1B5E20"
    ).encode(
        y="NPS:Q",
        text=alt.Text("NPS:Q", format=".0f")
    )

    trend = (nps_bars + nps_labels + csat_line + csat_labels).resolve_scale(y='independent')
    st.altair_chart(trend.properties(width=700, height=350), use_container_width=True)

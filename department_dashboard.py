# department_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path
import re

# ----------------------------
# Page config & theme accents
# ----------------------------
st.set_page_config(
    page_title="Department Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small CSS touchups (keeps the app tidy, non-intrusive)
st.markdown("""
<style>
/* cards */
.block-container {padding-top: 1.2rem;}
.kpi-card {background:#f8fbff;border:1px solid #e5eefb;border-radius:10px;padding:16px;}
.kpi-title {color:#64748b;font-size:0.9rem;margin:0 0 6px 0;}
.kpi-value {color:#1d4ed8;font-weight:700;font-size:1.6rem;margin:0;}
/* info panel */
.info {background:#eef6ff;border:1px solid #dbeafe;border-radius:8px;padding:14px;}
/* light table tweak */
thead tr th { white-space:nowrap; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Paths & load helpers
# ----------------------------
BASE_DIR = Path(__file__).parent

@st.cache_data(ttl=600)
def _read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)

@st.cache_data(ttl=600)
def load_data():
    # Core sources
    chat_df  = _read_csv_safe(BASE_DIR / "chat.csv",
                              dayfirst=True,
                              parse_dates=["Date/Time Opened"])
    email_df = _read_csv_safe(BASE_DIR / "email.csv",
                              dayfirst=True,
                              parse_dates=["Date/Time Opened"])
    items_df = _read_csv_safe(BASE_DIR / "report_items.csv",
                              dayfirst=True,
                              parse_dates=["Start DT","End DT"])
    pres_df  = _read_csv_safe(BASE_DIR / "report_presence.csv",
                              dayfirst=True,
                              parse_dates=["Start DT","End DT"])
    shifts   = _read_csv_safe(BASE_DIR / "shifts.csv")
    survey   = _read_csv_safe(BASE_DIR / "survey.csv", dayfirst=True, parse_dates=["Survey Date"])  # optional

    for df in (chat_df, email_df, items_df, pres_df, shifts, survey):
        if not df.empty:
            df.columns = df.columns.str.strip()
    return chat_df, email_df, items_df, pres_df, shifts, survey

chat_sla_df, email_sla_df, df_items, df_presence, df_shifts, df_survey = load_data()

if chat_sla_df.empty or email_sla_df.empty:
    st.error("Please ensure chat.csv and email.csv exist alongside the app.")
    st.stop()

# ----------------------------
# Sidebar filter: date range
# ----------------------------
st.sidebar.header("Filters")

min_date = min(
    chat_sla_df["Date/Time Opened"].dt.date.min(),
    email_sla_df["Date/Time Opened"].dt.date.min(),
)
max_date = max(
    chat_sla_df["Date/Time Opened"].dt.date.max(),
    email_sla_df["Date/Time Opened"].dt.date.max(),
)

start_date = st.sidebar.date_input("Start", value=max_date - timedelta(days=6),
                                   min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("End", value=max_date,
                                   min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.sidebar.error("Start date must be on or before End date.")
    st.stop()

# ----------------------------
# Utility formatters / helpers
# ----------------------------
def fmt_mmss(sec: float | None) -> str:
    if not sec or np.isnan(sec):
        return "—"
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02}:{s:02}"

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

# ----------------------------
# KPI block (basic volumes & AHT)
# ----------------------------
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.divider()

# Compute AHT from report_items
mask = ((df_items["Start DT"].dt.date >= start_date) &
        (df_items["Start DT"].dt.date <= end_date))
df_period_items = df_items.loc[mask].copy()
if not df_period_items.empty:
    df_period_items["Duration_sec"] = (df_period_items["End DT"] - df_period_items["Start DT"]).dt.total_seconds()
    chat_items = df_period_items[df_period_items["Service Channel: Developer Name"]=="sfdc_liveagent"]
    email_items = df_period_items[df_period_items["Service Channel: Developer Name"]=="casesChannel"]
    chat_total = int(len(chat_items))
    email_total = int(len(email_items))
    chat_aht = chat_items["Duration_sec"].mean() if chat_total else np.nan
    email_aht = email_items["Duration_sec"].mean() if email_total else np.nan
else:
    chat_total = email_total = 0
    chat_aht = email_aht = np.nan

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Total Chats</div>'
                f'<div class="kpi-value">{chat_total:,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Total Emails</div>'
                f'<div class="kpi-value">{email_total:,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Avg Chat AHT</div>'
                f'<div class="kpi-value">{fmt_mmss(chat_aht)}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Avg Email AHT</div>'
                f'<div class="kpi-value">{fmt_mmss(email_aht)}</div></div>', unsafe_allow_html=True)

# ----------------------------
# SLA (daily) — Chat & Email + Weighted
# ----------------------------
chat_slice = chat_sla_df[
    (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
].copy()

email_slice = email_sla_df[
    (email_sla_df["Date/Time Opened"].dt.date >= start_date) &
    (email_sla_df["Date/Time Opened"].dt.date <= end_date)
].copy()

days = pd.date_range(start_date, end_date, freq="D")
daily_rows = []
for d in days.date:
    # Chat
    cd = chat_slice[chat_slice["Date/Time Opened"].dt.date == d]
    # Only include rows with wait time for %<=60 calculation
    cw = cd[cd["Wait Time"].notna()]
    chats_vol = len(cw)
    pct_60s = (cw["Wait Time"] <= 60).mean()*100 if chats_vol else 0.0
    avg_wait_m = (cw["Wait Time"].mean()/60.0) if chats_vol else 0.0
    abandon_rate = (cd["Abandoned After"] > 20).mean()*100 if len(cd) else 0.0
    chat_raw = (0.5*pct_60s) - (0.3*avg_wait_m) - (0.2*abandon_rate)
    chat_sla = np.clip(((chat_raw / 0.45) * 80), 0, 100) if chats_vol else 0.0

    # Email
    ed = email_slice[email_slice["Date/Time Opened"].dt.date == d]
    emails_vol = len(ed)
    pct_1h = (ed["Elapsed Time (Hours)"] <= 1).mean()*100 if emails_vol else 0.0
    avg_resp_h = ed["Elapsed Time (Hours)"].mean() if emails_vol else 0.0
    email_raw = (0.6*pct_1h) - (0.4*avg_resp_h)
    email_sla = np.clip(((email_raw / 0.5625) * 80), 0, 100) if emails_vol else 0.0

    daily_rows.append({
        "Date": pd.to_datetime(d),
        "Chat SLA": chat_sla, "Chat Vol": chats_vol,
        "Email SLA": email_sla, "Email Vol": emails_vol
    })

df_daily = pd.DataFrame(daily_rows)
if df_daily.empty:
    st.info("No activity in the selected period.")
else:
    total_vol = (df_daily["Chat Vol"] + df_daily["Email Vol"]).replace(0, np.nan)
    df_daily["Weighted SLA"] = ((df_daily["Chat SLA"]*df_daily["Chat Vol"] +
                                 df_daily["Email SLA"]*df_daily["Email Vol"]) / total_vol).fillna(0)

st.divider()
st.header("🎯 SLA Score Summary")
if not df_daily.empty:
    chat_weighted = (df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum() / (df_daily["Chat Vol"].sum() or 1)
    email_weighted = (df_daily["Email SLA"]*df_daily["Email Vol"]).sum() / (df_daily["Email Vol"].sum() or 1)
    total_vol_period = (df_daily["Chat Vol"] + df_daily["Email Vol"]).sum()
    weighted_sla = ((df_daily["Chat SLA"]*df_daily["Chat Vol"] +
                     df_daily["Email SLA"]*df_daily["Email Vol"]).sum() / (total_vol_period or 1))
else:
    chat_weighted = email_weighted = weighted_sla = 0.0

s1, s2, s3 = st.columns(3)
s1.metric("Chat SLA Score", f"{chat_weighted:.1f}")
s2.metric("Email SLA Score", f"{email_weighted:.1f}")
s3.metric("Weighted SLA Score", f"{weighted_sla:.1f}")

# ----------------------------
# Weighted SLA trend
# ----------------------------
if not df_daily.empty:
    trend = (
        alt.Chart(df_daily.sort_values("Date"))
        .mark_line(point=True, color="#1d4ed8")
        .encode(
            x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%d %b", labelAngle=-45)),
            y=alt.Y("Weighted SLA:Q", title="Weighted SLA", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("Weighted SLA:Q", format=".1f"),
                alt.Tooltip("Chat Vol:Q", title="Chats"),
                alt.Tooltip("Email Vol:Q", title="Emails"),
            ]
        )
        .properties(height=340)
    )
    target_rule = alt.Chart(pd.DataFrame({"y":[80]})).mark_rule(color="red", strokeDash=[5,5]).encode(y="y:Q")
    st.altair_chart(trend + target_rule, use_container_width=True)

# ==========================================================
# 🌍 Chats by Country (volume) — uses 'Chat Button: Developer Name'
#     (Improved normalizer to combine "centre sierra leone",
#      "website sierra leone", "Sierra-Leone", etc.)
# ==========================================================
st.markdown("---")
st.subheader("🌍 Chats by Country (volume)")

# Country dictionary for ISO-2
_ISO2_TO_NAME = {
    "AO":"Angola","BJ":"Benin","BW":"Botswana","BF":"Burkina Faso","BI":"Burundi",
    "CM":"Cameroon","CV":"Cabo Verde","CF":"Central African Republic","TD":"Chad","KM":"Comoros",
    "CG":"Congo","CD":"Congo (DRC)","CI":"Côte d’Ivoire","DJ":"Djibouti","EG":"Egypt",
    "GQ":"Equatorial Guinea","ER":"Eritrea","SZ":"Eswatini","ET":"Ethiopia","GA":"Gabon",
    "GM":"Gambia","GH":"Ghana","GN":"Guinea","GW":"Guinea-Bissau","KE":"Kenya",
    "LS":"Lesotho","LR":"Liberia","MG":"Madagascar","MW":"Malawi","ML":"Mali",
    "MR":"Mauritania","MU":"Mauritius","MA":"Morocco","MZ":"Mozambique","NA":"Namibia",
    "NE":"Niger","NG":"Nigeria","RW":"Rwanda","ST":"São Tomé & Príncipe","SN":"Senegal",
    "SC":"Seychelles","SL":"Sierra Leone","SO":"Somalia","ZA":"South Africa","SS":"South Sudan",
    "SD":"Sudan","TZ":"Tanzania","TG":"Togo","TN":"Tunisia","UG":"Uganda","ZM":"Zambia","ZW":"Zimbabwe",
}
_KNOWN_COUNTRIES = {v.lower(): v for v in _ISO2_TO_NAME.values()}
_LANG_CODES = {"EN","FR","PT","ES","AR"}  # ignore when they appear as suffixes
_STOPWORDS = {
    "premier","premierbet","pb","mercury","bet","button","developer","name",
    "chat","support","customer","service","cs","care","help","web","live","agent"
}

def _matches_country_name(text_low: str, country_name: str) -> bool:
    """
    True if text_low contains country_name allowing any non-letters between words.
    E.g. 'sierra-leone', 'website sierra   leone', 'centre_sierra_leone' all match 'Sierra Leone'.
    """
    tokens = re.findall(r"[a-z]+", country_name.lower())
    if not tokens:
        return False
    pattern = r"\b" + r"\W+".join(map(re.escape, tokens)) + r"\b"
    return re.search(pattern, text_low) is not None

def _country_from_button(val: object) -> str | None:
    """Normalize 'Chat Button: Developer Name' to a clean country label."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None

    # If "City, Country" -> take last token
    if "," in s and len(s) < 100:
        s = s.split(",")[-1].strip()

    low = s.lower()

    # 1) Strong match by country name with flexible separators
    for cname_low, cname in _KNOWN_COUNTRIES.items():
        if _matches_country_name(low, cname):
            return cname

    # 2) Locale/suffix like "en-TZ", "..._TZ" -> trailing 2 letters
    m = re.search(r"[_\-\s]([A-Za-z]{2})$", s)
    if m:
        code = m.group(1).upper()
        if code not in _LANG_CODES:
            return _ISO2_TO_NAME.get(code, code)

    # 3) Pure 2-letter code?
    if re.fullmatch(r"[A-Za-z]{2}", s):
        code = s.upper()
        if code not in _LANG_CODES:
            return _ISO2_TO_NAME.get(code, code)

    # 4) Token fallback: prefer tokens that are country names
    toks = re.findall(r"[A-Za-z]{2,}", s)
    toks = [t for t in toks if t.lower() not in _STOPWORDS]
    for t in toks:
        if t.lower() in _KNOWN_COUNTRIES:
            return _KNOWN_COUNTRIES[t.lower()]

    # 5) Last resort: cleaned label
    if toks:
        return " ".join(toks).title()

    return None

# Slice chats by date range
btn_col = "Chat Button: Developer Name"
if btn_col not in chat_sla_df.columns:
    st.info(f"Column '{btn_col}' not found in chat.csv.")
else:
    chat_country_df = chat_sla_df[
        (chat_sla_df["Date/Time Opened"].dt.date >= start_date) &
        (chat_sla_df["Date/Time Opened"].dt.date <= end_date)
    ].copy()

    countries = chat_country_df[btn_col].map(_country_from_button).fillna("Unknown")
    counts = (countries.value_counts(dropna=False)
              .rename_axis("Country").reset_index(name="Chats")
              .sort_values("Chats", ascending=False).reset_index(drop=True))

    total_chats = int(counts["Chats"].sum()) if not counts.empty else 0
    counts["Share"] = counts["Chats"] / total_chats if total_chats else 0.0

    top_n = st.sidebar.slider("Pie chart: top countries", min_value=3, max_value=12, value=8, step=1)
    if len(counts) > top_n:
        top = counts.head(top_n)
        others_total = counts["Chats"].iloc[top_n:].sum()
        counts = pd.concat(
            [top, pd.DataFrame({"Country":["Other"], "Chats":[others_total], "Share":[(others_total/(total_chats or 1))]})],
            ignore_index=True
        )

    if counts.empty:
        st.info("No chats in the selected period.")
    else:
        pie = (
            alt.Chart(counts)
            .mark_arc(outerRadius=140, innerRadius=60)
            .encode(
                theta=alt.Theta("Chats:Q", stack=True),
                color=alt.Color("Country:N", legend=alt.Legend(title="Country")),
                tooltip=[
                    alt.Tooltip("Country:N"),
                    alt.Tooltip("Chats:Q", format=","),
                    alt.Tooltip("Share:Q", format=".1%")
                ]
            )
            .properties(width=520, height=360, title="Chat Volume by Country")
        )
        labels = (
            alt.Chart(counts)
            .mark_text(radius=105, size=11)
            .encode(
                theta=alt.Theta("Chats:Q", stack=True),
                text=alt.Text("Chats:Q", format=",.0f")
            )
        )
        st.altair_chart(pie + labels, use_container_width=True)

        with st.expander("View country breakdown table"):
            st.dataframe(
                counts[["Country","Chats","Share"]].style.format({"Chats": "{:,}", "Share": "{:.1%}"}),
                use_container_width=True
            )

# ------------- end of file -------------


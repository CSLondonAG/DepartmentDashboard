import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path
import re

# =========================================================
# PAGE CONFIG & STYLING
# =========================================================
st.set_page_config(page_title="Department Performance Dashboard", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem;}
.kpi-card {background:#f8fbff;border:1px solid #e5eefb;border-radius:10px;padding:16px;}
.kpi-title {color:#64748b;font-size:0.9rem;margin:0 0 6px 0;}
.kpi-value {color:#1d4ed8;font-weight:700;font-size:1.6rem;margin:0;}
thead tr th {white-space:nowrap;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent

# =========================================================
# DATA LOADING HELPERS
# =========================================================
def _read_csv_safe(path: Path, **kwargs):
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path, **kwargs)

def _first_existing_column(df, candidates):
    if df.empty: return None
    norm = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.strip().lower() in norm:
            return norm[name.strip().lower()]
    return None

def _prep_survey(df):
    if df.empty: return df
    df.columns = df.columns.str.strip()
    date_col = _first_existing_column(df, [
        "Survey Date","Survey Taker: Created Date","Created Date","Date","Response Date"
    ])
    if date_col:
        df["Survey Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce").dt.normalize()
    else:
        df["Survey Date"] = pd.NaT
    return df

def _ensure_dt(df, col):
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df

@st.cache_data(ttl=600)
def load_data():
    chat  = _read_csv_safe(BASE_DIR / "chat.csv")
    email = _read_csv_safe(BASE_DIR / "email.csv")
    items = _read_csv_safe(BASE_DIR / "report_items.csv")
    pres  = _read_csv_safe(BASE_DIR / "report_presence.csv")
    shifts= _read_csv_safe(BASE_DIR / "shifts.csv")
    survey= _read_csv_safe(BASE_DIR / "survey.csv")

    for df in (chat,email,items,pres,shifts,survey):
        if not df.empty: df.columns = df.columns.str.strip()

    for df,col in [(chat,"Date/Time Opened"),(email,"Date/Time Opened"),
                   (items,"Start DT"),(items,"End DT"),
                   (pres,"Start DT"),(pres,"End DT")]:
        _ensure_dt(df,col)

    survey=_prep_survey(survey)
    return chat,email,items,pres,shifts,survey

chat_df,email_df,items_df,pres_df,shifts_df,survey_df=load_data()
if chat_df.empty or email_df.empty:
    st.error("chat.csv and email.csv are required.")
    st.stop()

# =========================================================
# SIDEBAR DATE RANGE
# =========================================================
min_date=min(chat_df["Date/Time Opened"].dropna().dt.date.min(),
             email_df["Date/Time Opened"].dropna().dt.date.min())
max_date=max(chat_df["Date/Time Opened"].dropna().dt.date.max(),
             email_df["Date/Time Opened"].dropna().dt.date.max())

st.sidebar.header("Filters")
start_date=st.sidebar.date_input("Start",max_date-timedelta(days=6),min_date,max_date)
end_date=st.sidebar.date_input("End",max_date,min_date,max_date)
if start_date>end_date: st.stop()

def fmt_mmss(sec):
    if pd.isna(sec): return "—"
    m,s=divmod(int(sec),60);return f"{m:02}:{s:02}"

# =========================================================
# KPI SECTION (AHT)
# =========================================================
st.title("📊 Department Performance Dashboard")
st.markdown(f"### Period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")
st.divider()

mask=(items_df["Start DT"].dt.date>=start_date)&(items_df["Start DT"].dt.date<=end_date)
period=items_df.loc[mask].copy()
if not period.empty:
    period["Duration_sec"]=(period["End DT"]-period["Start DT"]).dt.total_seconds()
    chat_items=period[period["Service Channel: Developer Name"]=="sfdc_liveagent"]
    email_items=period[period["Service Channel: Developer Name"]=="casesChannel"]
    chat_total,email_total=len(chat_items),len(email_items)
    chat_aht,email_aht=chat_items["Duration_sec"].mean(),email_items["Duration_sec"].mean()
else:
    chat_total=email_total=0;chat_aht=email_aht=np.nan

c1,c2,c3,c4=st.columns(4)
for c,title,val in zip([c1,c2,c3,c4],
    ["Total Chats","Total Emails","Avg Chat AHT","Avg Email AHT"],
    [chat_total,email_total,fmt_mmss(chat_aht),fmt_mmss(email_aht)]):
    c.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div>"
               f"<div class='kpi-value'>{val}</div></div>",unsafe_allow_html=True)

# =========================================================
# SLA CALCULATIONS (Chat, Email, Weighted)
# =========================================================
chat_s=chat_df[(chat_df["Date/Time Opened"].dt.date>=start_date)&
               (chat_df["Date/Time Opened"].dt.date<=end_date)].copy()
email_s=email_df[(email_df["Date/Time Opened"].dt.date>=start_date)&
                 (email_df["Date/Time Opened"].dt.date<=end_date)].copy()

for col in ["Wait Time","Abandoned After"]:
    if col in chat_s.columns: chat_s[col]=pd.to_numeric(chat_s[col],errors="coerce")
if "Elapsed Time (Hours)" in email_s.columns:
    email_s["Elapsed Time (Hours)"]=pd.to_numeric(email_s["Elapsed Time (Hours)"],errors="coerce")

rows=[]
for d in pd.date_range(start_date,end_date):
    day=d.date()
    cd=chat_s[chat_s["Date/Time Opened"].dt.date==day]
    ed=email_s[email_s["Date/Time Opened"].dt.date==day]

    vchat=len(cd[cd["Wait Time"].notna()])
    vemail=len(ed)
    pct60=(cd["Wait Time"]<=60).mean()*100 if vchat else 0
    avgw=(cd["Wait Time"].mean()/60) if vchat else 0
    aban=(cd["Abandoned After"]>20).mean()*100 if len(cd) else 0
    chat_score=(0.5*pct60)-(0.3*avgw)-(0.2*aban)
    chat_sla=np.clip((chat_score/0.4167)*80,0,100)

    pct1h=(ed["Elapsed Time (Hours)"]<=1).mean()*100 if vemail else 0
    avgr=ed["Elapsed Time (Hours)"].mean() if vemail else 0
    email_score=(0.6*pct1h)-(0.4*avgr)
    email_sla=np.clip((email_score/0.5625)*80,0,100)
    rows.append({"Date":pd.to_datetime(day),
                 "Chat SLA":chat_sla,"Chat Vol":vchat,
                 "Email SLA":email_sla,"Email Vol":vemail})

df_daily=pd.DataFrame(rows)
df_daily["Weighted SLA"]=((df_daily["Chat SLA"]*df_daily["Chat Vol"]+
                           df_daily["Email SLA"]*df_daily["Email Vol"])/
                          (df_daily["Chat Vol"]+df_daily["Email Vol"]).replace(0,np.nan))

st.divider();st.header("🎯 SLA Summary")
chat_w=(df_daily["Chat SLA"]*df_daily["Chat Vol"]).sum()/df_daily["Chat Vol"].sum()
email_w=(df_daily["Email SLA"]*df_daily["Email Vol"]).sum()/df_daily["Email Vol"].sum()
totalv=(df_daily["Chat Vol"]+df_daily["Email Vol"]).sum()
weighted=(df_daily["Weighted SLA"]*(df_daily["Chat Vol"]+df_daily["Email Vol"])).sum()/totalv
s1,s2,s3=st.columns(3)
for c,t,v in zip([s1,s2,s3],
                 ["Chat SLA","Email SLA","Weighted SLA"],
                 [chat_w,email_w,weighted]): c.metric(t,f"{v:.1f}")

# =========================================================
# UTILIZATION (Proportional split)
# =========================================================
window_start=datetime.combine(start_date,datetime.min.time())
window_end=datetime.combine(end_date+timedelta(days=1),datetime.min.time())

chat_pres=pres_df[pres_df["Service Presence Status: Developer Name"].isin(["Available_Chat","Available_All"])]
email_pres=pres_df[pres_df["Service Presence Status: Developer Name"].isin(["Available_Email_and_Web","Available_All"])]

def total_overlap(a,b):
    total=0
    for s1,e1 in a:
        for s2,e2 in b:
            o_s=max(s1,s2);o_e=min(e1,e2)
            if o_e>o_s: total+=(o_e-o_s).total_seconds()
    return total

def merge_intervals(intervals):
    if not intervals:return []
    intervals=sorted(intervals,key=lambda x:x[0])
    merged=[intervals[0]]
    for s,e in intervals[1:]:
        if s<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],e)
        else: merged.append([s,e])
    return merged

# group by agent
chat_handle=email_handle=0
chat_avail=email_avail=0

# availability dicts
avail_by_agent={}
for agent,g in pres_df.groupby("Created By: Full Name"):
    avail_by_agent[agent]=list(zip(g["Start DT"],g["End DT"]))

for agent,g in period.groupby("User: Full Name"):
    chan=g["Service Channel: Developer Name"].iloc[0] if not g.empty else ""
    intervals=list(zip(g["Start DT"],g["End DT"]))
    intervals=[tuple(x) for x in merge_intervals(intervals)]
    if agent not in avail_by_agent: continue
    avail=avail_by_agent[agent]
    handle_secs=total_overlap(intervals,avail)
    if chan=="sfdc_liveagent": chat_handle+=handle_secs
    elif chan=="casesChannel": email_handle+=handle_secs

# proportional split for Available_All
chat_avail_secs=email_avail_secs=0
for _,p in pres_df.iterrows():
    if p["End DT"]<=window_start or p["Start DT"]>=window_end: continue
    s=max(p["Start DT"],window_start);e=min(p["End DT"],window_end)
    dur=(e-s).total_seconds()
    if p["Service Presence Status: Developer Name"]=="Available_All":
        total=chat_handle+email_handle
        chat_share=chat_handle/total if total>0 else 0.5
        email_share=email_handle/total if total>0 else 0.5
        chat_avail_secs+=dur*chat_share;email_avail_secs+=dur*email_share
    elif p["Service Presence Status: Developer Name"]=="Available_Chat": chat_avail_secs+=dur
    elif p["Service Presence Status: Developer Name"]=="Available_Email_and_Web": email_avail_secs+=dur

chat_util=chat_handle/chat_avail_secs if chat_avail_secs else 0
email_util=email_handle/email_avail_secs if email_avail_secs else 0

st.divider();m1,m2=st.columns(2)
m1.metric("Chat Utilization",f"{chat_util:.1%}")
m2.metric("Email Utilization",f"{email_util:.1%}")

# =========================================================
# HOURLY WEIGHTED SLA + AVAILABLE + LOGGED-IN
# =========================================================
st.divider();st.header("⏰ Hourly Weighted SLA & Availability")
hourly=[]
for hr in range(24):
    mask=(chat_df["Date/Time Opened"].dt.date==end_date)&(chat_df["Date/Time Opened"].dt.hour==hr)
    ch=chat_df.loc[mask];emails=email_df[(email_df["Date/Time Opened"].dt.date==end_date)&
                                         (email_df["Date/Time Opened"].dt.hour==hr)]
    vchat,vemail=len(ch),len(emails)
    if vchat+vemail==0: continue
    cwait=(ch["Wait Time"]<=60).mean()*100 if vchat else 0
    aw=(ch["Wait Time"].mean()/60) if vchat else 0
    aban=(ch["Abandoned After"]>20).mean()*100 if vchat else 0
    csla=np.clip(((0.5*cwait-0.3*aw-0.2*aban)/0.4167)*80,0,100)
    pct1=(emails["Elapsed Time (Hours)"]<=1).mean()*100 if vemail else 0
    avgh=emails["Elapsed Time (Hours)"].mean() if vemail else 0
    esla=np.clip(((0.6*pct1-0.4*avgh)/0.5625)*80,0,100)
    w=((csla*vchat+esla*vemail)/(vchat+vemail))
    hourly.append({"Hour":hr,"Weighted SLA":w})

hourly_df=pd.DataFrame(hourly)
chart=alt.Chart(hourly_df).mark_line(point=True,color="#2563eb").encode(
    x=alt.X("Hour:O",title="Hour of Day"),
    y=alt.Y("Weighted SLA:Q",scale=alt.Scale(domain=[0,100]),title="Weighted SLA"),
    tooltip=["Hour:O","Weighted SLA:Q"]
)
st.altair_chart(chart,width="stretch")
# =========================================================
# SURVEY METRICS (CSAT / NPS / FCR)
# =========================================================
st.divider()
st.header("💬 Customer Feedback Metrics")

if survey_df.empty or "Survey Date" not in survey_df.columns:
    st.info("No survey data available.")
else:
    survey_period = survey_df[
        (survey_df["Survey Date"].dt.date >= start_date) &
        (survey_df["Survey Date"].dt.date <= end_date)
    ].copy()

    if not survey_period.empty:
        # --- CSAT ---
        csat_mask = survey_period["Survey Question: Question Title"].str.contains("satisfied", case=False, na=False)
        csat_scores = pd.to_numeric(survey_period.loc[csat_mask, "Response"], errors="coerce")
        csat_pct = (csat_scores >= 4).mean() * 100 if not csat_scores.empty else 0

        # --- NPS ---
        nps_mask = survey_period["Survey Question: Question Title"].str.contains("recommend", case=False, na=False)
        nps_vals = survey_period.loc[nps_mask, "Response"].astype(str).str.extract(r"(\d+)").astype(float)
        nps_vals = nps_vals[0].dropna()
        promoters = (nps_vals >= 9).sum()
        detractors = (nps_vals <= 6).sum()
        total_nps = len(nps_vals)
        nps_score = ((promoters - detractors) / total_nps * 100) if total_nps else 0

        # --- FCR ---
        fcr_mask = survey_period["Survey Question: Question Title"].str.contains("resolved", case=False, na=False)
        fcr_vals = survey_period.loc[fcr_mask, "Response"].astype(str).str.lower()
        fcr_pct = (fcr_vals == "yes").mean() * 100 if not fcr_vals.empty else 0

        c1, c2, c3 = st.columns(3)
        color_csat = "red" if csat_pct < 70 else "#1d4ed8"
        color_nps  = "red" if nps_score < 0 else "#1d4ed8"
        color_fcr  = "red" if fcr_pct < 50 else "#1d4ed8"

        c1.markdown(f"<div class='kpi-card' style='border-color:{color_csat};'>"
                    f"<div class='kpi-title'>CSAT (%)</div>"
                    f"<div class='kpi-value' style='color:{color_csat};'>{csat_pct:.1f}</div></div>",
                    unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card' style='border-color:{color_nps};'>"
                    f"<div class='kpi-title'>NPS</div>"
                    f"<div class='kpi-value' style='color:{color_nps};'>{nps_score:.1f}</div></div>",
                    unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-card' style='border-color:{color_fcr};'>"
                    f"<div class='kpi-title'>FCR (%)</div>"
                    f"<div class='kpi-value' style='color:{color_fcr};'>{fcr_pct:.1f}</div></div>",
                    unsafe_allow_html=True)

        # --- Daily Trend Chart ---
        daily_survey = (
            survey_period
            .groupby(survey_period["Survey Date"].dt.normalize(), as_index=False)
            .agg(
                CSAT_pct=("Response", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                NPS=("Survey Question: Question Title", lambda _: nps_score),
                FCR=("Response", lambda _: fcr_pct)
            )
        )
        daily_survey["Period"] = daily_survey["Survey Date"].dt.strftime("%d %b")

        # Build chart: CSAT line, NPS bar
        csat_line = alt.Chart(daily_survey).mark_line(point=True, color="#2563eb", strokeWidth=3).encode(
            x=alt.X("Period:N", title="Period", sort=daily_survey["Period"].tolist(), axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("CSAT_pct:Q", title="CSAT (%)", scale=alt.Scale(domain=[0,100])),
            tooltip=["Period:N", alt.Tooltip("CSAT_pct:Q", format=".1f", title="CSAT (%)")]
        )
        nps_bar = alt.Chart(daily_survey).mark_bar(color="#dc2626", opacity=0.6).encode(
            x=alt.X("Period:N", sort=daily_survey["Period"].tolist()),
            y=alt.Y("NPS:Q", title="NPS Score", scale=alt.Scale(domain=[-100,100])),
            tooltip=["Period:N", alt.Tooltip("NPS:Q", format=".1f", title="NPS Score")]
        )
        zero_rule = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="#9ca3af", strokeDash=[6,4]).encode(y="y:Q")
        st.altair_chart((csat_line + nps_bar + zero_rule)
                        .resolve_scale(y="independent")
                        .properties(width=800, height=400,
                                    title="CSAT (%) and NPS Trend"), width="stretch")
    else:
        st.info("No surveys in the selected range.")

# =========================================================
# DAILY SCHEDULE SUMMARY
# =========================================================
st.divider()
st.header(f"👥 Daily Schedule Summary ({end_date:%d %b %Y})")

def build_daily_schedule(shifts, presence, selected_date):
    sched = shifts.copy()
    sched.columns = sched.columns.str.strip()
    presence = presence.copy()
    presence["Date"] = presence["Start DT"].dt.date
    presence_day = presence[presence["Date"] == selected_date]

    results=[]
    for _,row in sched.iterrows():
        agent=row.get("Agent") or row.get("Name") or row.get("User")
        if not agent: continue
        pres=presence_day[presence_day["Created By: Full Name"].str.lower()==agent.lower()]
        if pres.empty: continue
        avail=pres[pres["Service Presence Status: Developer Name"].str.contains("Available",case=False)]
        login=avail["Start DT"].min();logout=avail["End DT"].max()
        lunch=pres[pres["Service Presence Status: Developer Name"].str.lower().eq("busy_lunch")]
        lunch_time=lunch["Start DT"].min()
        shift_start=pd.to_datetime(row.get("Shift Start",""), errors="coerce").time() if "Shift Start" in row else None
        shift_end=pd.to_datetime(row.get("Shift End",""), errors="coerce").time() if "Shift End" in row else None
        late=early="—"
        if login and shift_start:
            diff=(login.time().hour+login.time().minute/60)-(shift_start.hour+shift_start.minute/60)
            if diff>0.1: late=f"{diff:.2f} h"
        if logout and shift_end:
            diff=(shift_end.hour+shift_end.minute/60)-(logout.time().hour+logout.time().minute/60)
            if diff>0.1: early=f"{diff:.2f} h"
        results.append({
            "Agent":agent,
            "Shift Start":shift_start.strftime("%H:%M") if shift_start else "—",
            "Login":login.strftime("%H:%M") if pd.notna(login) else "—",
            "Lunch":lunch_time.strftime("%H:%M") if pd.notna(lunch_time) else "—",
            "Logout":logout.strftime("%H:%M") if pd.notna(logout) else "—",
            "Shift End":shift_end.strftime("%H:%M") if shift_end else "—",
            "Late Start":late,"Early Finish":early
        })
    return pd.DataFrame(results)

schedule_df = build_daily_schedule(shifts_df, pres_df, end_date)
if schedule_df.empty:
    st.info(f"No scheduled agents found for {end_date:%d %b %Y}.")
else:
    def color_lunch(val, shift_start):
        if val=="—": return ""
        try:
            lunch_dt=datetime.strptime(val,"%H:%M")
            shift_dt=datetime.strptime(shift_start,"%H:%M")
            diff=(lunch_dt-shift_dt).seconds/3600
            if diff<3 or diff>5:
                return "background-color:#fee2e2;color:#b91c1c;"
        except: return ""
        return ""
    styled=schedule_df.style.apply(lambda r:[
        color_lunch(r["Lunch"],r["Shift Start"]),
        "", "", "", "", "", "", ""
    ], axis=1)
    st.dataframe(styled,width="stretch")

# =========================================================
# CHATS BY COUNTRY
# =========================================================
st.divider()
st.header("🌍 Chats by Country")

_ISO2_TO_NAME={"AO":"Angola","BJ":"Benin","SL":"Sierra Leone","NG":"Nigeria","GH":"Ghana","MW":"Malawi","ZM":"Zambia"}
_KNOWN={v.lower():v for v in _ISO2_TO_NAME.values()}
def _matches_country_name(txt,country):
    tokens=re.findall(r"[a-z]+",country.lower())
    if not tokens:return False
    pattern=r"\b"+r"\W+".join(tokens)+r"\b"
    return re.search(pattern,txt) is not None
def _country_from_button(val):
    if pd.isna(val):return None
    s=str(val).strip().lower()
    for cname in _KNOWN:
        if _matches_country_name(s,cname): return _KNOWN[cname]
    if "sierra" in s and "leone" in s: return "Sierra Leone"
    return s.title()
if "Chat Button: Developer Name" in chat_df.columns:
    col="Chat Button: Developer Name"
else:
    col="Chat Button: Developer Name"
chats=chat_df[(chat_df["Date/Time Opened"].dt.date>=start_date)&
              (chat_df["Date/Time Opened"].dt.date<=end_date)].copy()
chats["Country"]=chats[col].map(_country_from_button).fillna("Unknown")
counts=(chats["Country"].value_counts().rename_axis("Country")
        .reset_index(name="Chats").sort_values("Chats",ascending=False))
if not counts.empty:
    total=int(counts["Chats"].sum())
    counts["Share"]=counts["Chats"]/total
    pie=(alt.Chart(counts).mark_arc(outerRadius=140,innerRadius=60)
         .encode(theta="Chats:Q",color="Country:N",
                 tooltip=["Country:N","Chats:Q","Share:Q"])
         .properties(width=520,height=360,title="Chat Volume by Country"))
    st.altair_chart(pie,width="stretch")
else:
    st.info("No chat country data for this range.")

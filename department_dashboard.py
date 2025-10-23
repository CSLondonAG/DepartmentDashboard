# =========================================================
# DEPARTMENT PERFORMANCE DASHBOARD (Final Version)
# =========================================================
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
# DATA LOADING
# =========================================================
def _read_csv_safe(path: Path, **kwargs):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)

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
        if not df.empty:
            df.columns = df.columns.str.strip()

    for df,col in [(chat,"Date/Time Opened"),(email,"Date/Time Opened"),
                   (items,"Start DT"),(items,"End DT"),
                   (pres,"Start DT"),(pres,"End DT")]:
        _ensure_dt(df,col)

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
if start_date>end_date:
    st.stop()

def fmt_mmss(sec):
    if pd.isna(sec):
        return "—"
    m,s=divmod(int(sec),60)
    return f"{m:02}:{s:02}"

# =========================================================
# KPI SECTION
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
    chat_total=email_total=0
    chat_aht=email_aht=np.nan

c1,c2,c3,c4=st.columns(4)
for c,title,val in zip([c1,c2,c3,c4],
    ["Total Chats","Total Emails","Avg Chat AHT","Avg Email AHT"],
    [chat_total,email_total,fmt_mmss(chat_aht),fmt_mmss(email_aht)]):
    c.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div>"
               f"<div class='kpi-value'>{val}</div></div>",unsafe_allow_html=True)

# =========================================================
# SLA CALCULATIONS
# =========================================================
chat_s=chat_df[(chat_df["Date/Time Opened"].dt.date>=start_date)&
               (chat_df["Date/Time Opened"].dt.date<=end_date)].copy()
email_s=email_df[(email_df["Date/Time Opened"].dt.date>=start_date)&
                 (email_df["Date/Time Opened"].dt.date<=end_date)].copy()

for col in ["Wait Time","Abandoned After"]:
    if col in chat_s.columns:
        chat_s[col]=pd.to_numeric(chat_s[col],errors="coerce")
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
# UTILIZATION (with merge_intervals fix)
# =========================================================
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = [list(i) for i in intervals]
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s,e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s,e])
    return [tuple(m) for m in merged]

def total_overlap(a,b):
    total=0
    for s1,e1 in a:
        for s2,e2 in b:
            o_s=max(s1,s2)
            o_e=min(e1,e2)
            if o_e>o_s: total+=(o_e-o_s).total_seconds()
    return total

avail_by_agent={agent:list(zip(g["Start DT"],g["End DT"])) for agent,g in pres_df.groupby("Created By: Full Name")}
chat_handle=email_handle=0
for agent,g in period.groupby("User: Full Name"):
    chan=g["Service Channel: Developer Name"].iloc[0] if not g.empty else ""
    intervals=list(zip(g["Start DT"],g["End DT"]))
    intervals=merge_intervals(intervals)
    if agent not in avail_by_agent: continue
    avail=avail_by_agent[agent]
    handle_secs=total_overlap(intervals,avail)
    if chan=="sfdc_liveagent": chat_handle+=handle_secs
    elif chan=="casesChannel": email_handle+=handle_secs

chat_avail_secs=email_avail_secs=0
for _,p in pres_df.iterrows():
    s=max(p["Start DT"], datetime.combine(start_date, datetime.min.time()))
    e=min(p["End DT"], datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
    dur=(e-s).total_seconds()
    if p["Service Presence Status: Developer Name"]=="Available_All":
        total=chat_handle+email_handle
        chat_share=chat_handle/total if total>0 else 0.5
        email_share=email_handle/total if total>0 else 0.5
        chat_avail_secs+=dur*chat_share
        email_avail_secs+=dur*email_share
    elif p["Service Presence Status: Developer Name"]=="Available_Chat": chat_avail_secs+=dur
    elif p["Service Presence Status: Developer Name"]=="Available_Email_and_Web": email_avail_secs+=dur

chat_util=chat_handle/chat_avail_secs if chat_avail_secs else 0
email_util=email_handle/email_avail_secs if email_avail_secs else 0
st.divider();m1,m2=st.columns(2)
m1.metric("Chat Utilization",f"{chat_util:.1%}")
m2.metric("Email Utilization",f"{email_util:.1%}")

# =========================================================
# CHATS BY COUNTRY (safe rendering)
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

col_name="Chat Button: Developer Name"
chats=chat_df[(chat_df["Date/Time Opened"].dt.date>=start_date)&
              (chat_df["Date/Time Opened"].dt.date<=end_date)].copy()
chats["Country"]=chats[col_name].map(_country_from_button).fillna("Unknown")
counts=(chats["Country"].value_counts().rename_axis("Country")
        .reset_index(name="Chats").sort_values("Chats",ascending=False))

if not counts.empty:
    total=int(counts["Chats"].sum())
    counts["Share"]=counts["Chats"]/total
    combined_top=(
        alt.Chart(counts)
        .mark_arc(outerRadius=140,innerRadius=60)
        .encode(
            theta="Chats:Q",
            color="Country:N",
            tooltip=["Country:N","Chats:Q","Share:Q"]
        )
        .properties(width=520,height=360,title="Chat Volume by Country")
    )
    st.altair_chart(combined_top, use_container_width=True)
else:
    st.info("No chat country data for this range.")

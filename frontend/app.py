"""light_gbm_endpoint — stand-up app. Pick an app (Multifamily / Brokerage / Marketing)
and run live predictions through the API, with charts. Internal users can also call the
API directly; this UI is for human exploration.
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


def _api_key() -> str:
    if os.environ.get("GBME_API_KEY"):
        return os.environ["GBME_API_KEY"]
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip().startswith("GBME_API_KEY"):
                return line.split("=", 1)[1].strip().strip("'").strip('"')
    return ""


API = os.environ.get("GBME_API_URL", "http://127.0.0.1:8910")
KEY = _api_key()
HDR = {"X-API-Key": KEY, "Content-Type": "application/json"}

st.set_page_config(page_title="RTS Prediction Hub", layout="wide", page_icon=":bar_chart:")


def api_get(path, **params):
    r = requests.get(f"{API}{path}", headers=HDR, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path, body):
    r = requests.post(f"{API}{path}", headers=HDR, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def get_buildings():
    return api_get("/multifamily/buildings")


@st.cache_data(ttl=300)
def get_companies(limit=60):
    return api_get("/brokerage/companies", limit=limit)["companies"]


@st.cache_data(ttl=300)
def get_marketing_options():
    return api_get("/marketing/options")


# ---------------- sidebar ----------------
st.sidebar.title("RTS Prediction Hub")
st.sidebar.caption("Shared LightGBM endpoint - DGX Spark")
try:
    h = api_get("/health")
    db_ok = h.get("db", {}).get("db") == "ok"
    st.sidebar.success(f"API up - DB {'connected' if db_ok else 'DOWN'}")
except Exception as e:
    st.sidebar.error(f"API unreachable at {API}\n{str(e)[:120]}")
app = st.sidebar.radio("Choose an application",
                       ["Multifamily - tenant churn", "Brokerage - prospect conversion",
                        "Marketing - cohort analytics"])
st.sidebar.caption(f"Endpoint: {API}")


def _risk_color(df, col="risk_tier"):
    cmap = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
    return df.style.apply(lambda s: [f"color:{cmap.get(v,'')}" for v in s] if s.name == col else ["" for _ in s])


# ---------------- Multifamily ----------------
def render_multifamily():
    st.header("Multifamily - Tenant Move-Out Risk")
    st.caption("Pick buildings (or enter tenant SCODEs); the API pulls live features from "
               "Yardi via SSMS and scores move-out probability. Model: interim stand-in (v0).")
    buildings = get_buildings()
    bmap = {b["name"]: b["property_id"] for b in buildings}
    c1, c2, c3 = st.columns([3, 2, 1])
    picks = c1.multiselect("Buildings", list(bmap), default=list(bmap)[:1])
    scodes_txt = c2.text_input("...or tenant SCODEs (comma-separated)", "")
    as_of = c3.date_input("As-of date")
    limit = st.slider("Max units per building", 10, 400, 80, 10)
    if st.button("Run prediction", type="primary"):
        body = {"as_of": str(as_of), "limit": limit}
        if scodes_txt.strip():
            body["scodes"] = [s.strip() for s in scodes_txt.split(",") if s.strip()]
        else:
            body["property_ids"] = [bmap[p] for p in picks]
        with st.spinner("Scoring against live data..."):
            res = api_post("/predict/multifamily-churn", body)
        df = pd.DataFrame(res.get("results", []))
        if df.empty:
            st.warning("No tenants returned for that selection.")
            return
        df["churn_probability"] = df["churn_probability"].astype(float)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Units scored", len(df))
        k2.metric("Mean move-out prob", f"{df['churn_probability'].mean():.1%}")
        k3.metric("High risk", int((df["risk_tier"] == "high").sum()))
        k4.metric("With notice filed", int(pd.to_numeric(df.get("has_notice"), errors="coerce").fillna(0).sum()))
        st.caption(f"model_version={res.get('model_version')} ({res.get('model_source')}) - as_of {res.get('as_of')}")
        g1, g2 = st.columns(2)
        g1.plotly_chart(px.histogram(df, x="churn_probability", nbins=20, color="risk_tier",
                                     color_discrete_map={"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"},
                                     title="Move-out probability distribution"), width="stretch")
        tier = df["risk_tier"].value_counts().reindex(["low", "medium", "high"]).fillna(0).reset_index()
        tier.columns = ["risk_tier", "count"]
        g2.plotly_chart(px.bar(tier, x="risk_tier", y="count", color="risk_tier",
                               color_discrete_map={"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"},
                               title="Risk tier breakdown"), width="stretch")
        st.subheader("Top at-risk tenants")
        top = df.sort_values("churn_probability", ascending=False).head(15)
        st.plotly_chart(px.bar(top, x="churn_probability", y="scode", orientation="h",
                               color="churn_probability", color_continuous_scale="Reds",
                               hover_data=["property", "unit"]), width="stretch")
        st.dataframe(df[["scode", "property", "unit", "churn_probability", "risk_tier",
                         "has_notice", "actual_churn"]].sort_values("churn_probability", ascending=False),
                     width="stretch", height=320)


# ---------------- Brokerage ----------------
def render_brokerage():
    st.header("Brokerage - Prospect Conversion (Signal Desk)")
    st.caption("Enter a company (hMy or name). The API pulls the company's full snapshot "
               "history from SSMS and runs the model across every point in time.")
    samples = get_companies(60)
    smap = {f"{c['display_name']} (hMy {int(c['hMy'])}, {c.get('snapshots')} snaps)": int(c["hMy"]) for c in samples}
    c1, c2 = st.columns([2, 3])
    typed = c1.text_input("Company hMy or name", "")
    picked = c2.selectbox("...or pick a company with history", ["-"] + list(smap))
    if st.button("Run prediction", type="primary"):
        company = typed.strip() or (str(smap[picked]) if picked != "-" else "")
        if not company:
            st.warning("Enter or pick a company.")
            return
        with st.spinner("Scoring history against live data..."):
            res = api_post("/predict/signal-desk-brokerage", {"company": company})
        if res.get("matches") == [] and not res.get("history"):
            st.warning(res.get("error", "No company found.")); return
        co = res["company"]
        st.subheader(f"{co.get('display_name')} - hMy {co.get('hMy')}")
        hist = pd.DataFrame(res.get("history", []))
        rules = res.get("current_rules_score") or {}
        k1, k2, k3, k4 = st.columns(4)
        if not hist.empty:
            hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"])
            hist["conversion_probability"] = hist["conversion_probability"].astype(float)
            k1.metric("Latest model prob", f"{hist['conversion_probability'].iloc[-1]:.1%}")
            k2.metric("History points", len(hist))
        k3.metric("Current rules prob", f"{float(rules.get('Probability',0)):.0%}" if rules else "n/a")
        k4.metric("Rules decile", rules.get("Decile", "n/a"))
        st.caption(f"market={co.get('market')} - industry={co.get('industry')} - "
                   f"model_version={res.get('model_version')} ({res.get('model_source')})")
        if not hist.empty:
            fig = px.line(hist, x="snapshot_date", y="conversion_probability", markers=True,
                          title="Conversion probability over time (model, per SSMS snapshot)")
            if rules:
                fig.add_hline(y=float(rules.get("Probability", 0)), line_dash="dash",
                              annotation_text="current rules score")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(hist[["snapshot_date", "conversion_probability", "risk_tier"]],
                         width="stretch", height=300)
        if res.get("candidates"):
            st.info("Multiple matches: " + ", ".join(c.get("display_name", "") for c in res["candidates"][:8]))


# ---------------- Marketing ----------------
def render_marketing():
    st.header("Marketing - Cohort & Industry Analytics")
    st.caption("Aggregate model outputs over a selectable cohort (concept) or industry, "
               "live from SSMS.")
    opts = get_marketing_options()
    variant = st.radio("Segment by", ["cohort", "industry"], horizontal=True)
    if variant == "cohort":
        cmap = {f"{c['ConceptLabel']} ({c['ConceptKey']})": c["ConceptKey"]
                for c in opts.get("cohorts", []) if c.get("ConceptKey")}
        label = st.selectbox("Cohort", list(cmap)) if cmap else None
        key = cmap.get(label)
    else:
        inds = [i["industry"] for i in opts.get("industries", [])]
        key = st.selectbox("Industry", inds) if inds else None
    if st.button("Aggregate", type="primary") and key:
        with st.spinner("Aggregating..."):
            res = api_post("/predict/marketing", {"variant": variant, "key": key})
        bm = pd.DataFrame(res.get("by_market", []))
        dec = pd.DataFrame(res.get("decile_distribution", []))
        summ = pd.DataFrame(res.get("summary", []))
        st.subheader(f"Segment: {key}")
        if not summ.empty:
            s = summ.iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Companies", int(s.get("n", 0)))
            k2.metric("Avg prob", f"{float(s.get('avg_prob') or 0):.1%}")
            k3.metric("Markets", int(s.get("markets", 0)))
        g1, g2 = st.columns(2)
        if not bm.empty:
            ycol = "avg_win_prob" if "avg_win_prob" in bm else "avg_score"
            g1.plotly_chart(px.bar(bm, x="Market", y=ycol, color="Market", text="n",
                                   title=f"{ycol} by market"), width="stretch")
        if not dec.empty:
            dec["Decile"] = pd.to_numeric(dec["Decile"], errors="coerce")
            g2.plotly_chart(px.bar(dec.sort_values("Decile"), x="Decile", y="n",
                                   title="Decile distribution", color="n",
                                   color_continuous_scale="Blues"), width="stretch")
        wr = pd.DataFrame(res.get("cohort_win_rates", []))
        if not wr.empty:
            st.subheader("Deal win-rates by cohort (mkt.CohortCatalog)")
            wr2 = wr.dropna(subset=["Vertical", "SizeBand"])
            if not wr2.empty:
                pv = wr2.pivot_table(index="Vertical", columns="SizeBand", values="WinRate", aggfunc="mean")
                st.plotly_chart(px.imshow(pv, text_auto=".2f", color_continuous_scale="Greens",
                                          title="Win-rate heatmap (vertical x size band)"),
                                width="stretch")
            st.dataframe(wr.head(40), width="stretch", height=280)


if app.startswith("Multifamily"):
    render_multifamily()
elif app.startswith("Brokerage"):
    render_brokerage()
else:
    render_marketing()

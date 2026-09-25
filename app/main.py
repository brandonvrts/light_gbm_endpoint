"""light_gbm_endpoint FastAPI service (v1). One service, 3 app contracts, live SSMS data.

Endpoints:
  GET  /health                              (open)
  GET  /contracts                           list contracts + manifests
  GET  /contracts/{contract}/schema         introspect input/output
  POST /predict/multifamily-churn           {scodes[] | property_ids[], as_of?}
  POST /predict/signal-desk-brokerage       {company, include_history}
  POST /predict/marketing                   {variant: cohort|industry, key}
  GET  /multifamily/churn-report            monthly cohort + outcomes + scores + revenue-at-risk
  GET  /multifamily/churn-trend             month-over-month churn summary series
  POST/GET /multifamily/outreach            log/list retention outreach (uplift tracking)
  GET  /multifamily/calibration-drift       logged predicted-vs-realized per settled month
  GET  /multifamily/cohorts                 resident-cohort catalog with live churn performance
  GET  /multifamily/cohort-report           one cohort: stats + top at-risk members + summary
  GET  /multifamily/persuadables            edge residents: on-the-fence list + why + economics
  GET  /multifamily/unit-economics          per-bedroom turnover cost/rate ledger (Phase 1)
  GET  /multifamily/concession-cohort       intern surface: band residents + full data + offer breakevens
  GET  /multifamily/edge-list               STABLE fight-for roster (hysteresis) + watch band
  GET  /multifamily/incentives              concession history (TRANS) + logged offers
  GET  /marketing/space-demand              per-cohort "who needs space" rollup (spacedemand bundle)
  GET  /marketing/space-demand/leads        ranked space-need leads inside one cohort
  GET  /multifamily/buildings | /brokerage/companies | /marketing/options   (UI helpers)
"""
from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException

from app.auth import require_key
from app.schemas import MultifamilyRequest, BrokerageRequest, MarketingRequest
from core import registry, db
from core.jsonify import clean
from contracts.multifamily_churn import (contract as mf, data as mf_data,
                                         schema as mf_schema, outreach as mf_outreach)
from pydantic import BaseModel
from contracts.signal_desk_brokerage import contract as bk, schema as bk_schema, scorecard as bk_scorecard
from contracts.signal_desk_spacedemand import contract as spd
from contracts.marketing import contract as mkt, schema as mkt_schema

app = FastAPI(title="light_gbm_endpoint", version="1.0.0",
              description="RTS shared LightGBM prediction API (multifamily / brokerage / marketing)")

_SCHEMAS = {"multifamily-churn": mf_schema, "signal-desk-brokerage": bk_schema, "marketing": mkt_schema}

# --- Phase 3: PII-free request/prediction logging (feeds future drift monitoring) ---
import time as _time
import json as _json
import datetime as _dt
from pathlib import Path as _Path

_LOGDIR = _Path(__file__).resolve().parent.parent / "monitoring" / "logs"
_LOGDIR.mkdir(parents=True, exist_ok=True)


@app.middleware("http")
async def _request_log(request, call_next):
    t0 = _time.time()
    resp = await call_next(request)
    try:
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"),
               "method": request.method, "path": request.url.path,
               "status": resp.status_code, "ms": int((_time.time() - t0) * 1000)}
        with (_LOGDIR / "requests.log").open("a") as fh:
            fh.write(_json.dumps(rec) + "\n")
    except Exception:  # logging must never break a response
        pass
    return resp


@app.get("/health")
def health():
    out = {"status": "ok", "db": db.healthcheck(), "contracts": registry.list_contracts()}
    try:  # Phase 0.2: surface the brokerage feature-pipeline freshness (froze ~2026-04)
        out["brokerage_data"] = mf_data_brokerage_freshness()
    except Exception:
        pass
    return clean(out)


def mf_data_brokerage_freshness():
    from contracts.signal_desk_brokerage import data as _bkd
    f = _bkd.feature_freshness()
    return {k: f.get(k) for k in ("feature_snapshot_latest", "rules_score_latest",
                                  "feature_snapshot_age_days", "stale", "note")}


@app.get("/contracts")
def contracts(_=Depends(require_key)):
    return clean({c: {"versions": v, "manifest": registry.manifest(c)}
                  for c, v in registry.list_contracts().items()})


@app.get("/contracts/{contract}/schema")
def contract_schema(contract: str, _=Depends(require_key)):
    s = _SCHEMAS.get(contract)
    if not s:
        raise HTTPException(404, f"unknown contract '{contract}'")
    return clean(s.describe())


@app.post("/predict/multifamily-churn")
def predict_multifamily(body: MultifamilyRequest, _=Depends(require_key)):
    if body.scode:
        return clean(mf.score_one(body.scode, body.as_of))
    if body.expires_from or body.expires_to:
        if not (body.expires_from and body.expires_to):
            raise HTTPException(400, "provide BOTH expires_from and expires_to")
        return clean(mf.score_lease_window(body.expires_from, body.expires_to,
                                           body.property_ids, body.as_of, body.limit))
    if body.property_ids:
        return clean(mf.score_buildings(body.property_ids, body.as_of, body.limit))
    if body.scodes:
        return clean(mf.score_scodes(body.scodes, body.as_of))
    raise HTTPException(400, "provide scode, scodes[], expires_from+expires_to, or property_ids[]")


@app.post("/predict/signal-desk-brokerage")
def predict_brokerage(body: BrokerageRequest, _=Depends(require_key)):
    return clean(bk.score_company(body.company, body.include_history))


@app.post("/predict/marketing")
def predict_marketing(body: MarketingRequest, _=Depends(require_key)):
    return clean(mkt.aggregate(body.variant, body.key))


# ---- UI convenience (pickers) ----
@app.get("/multifamily/buildings")
def mf_buildings(_=Depends(require_key)):
    return clean(mf_data.list_properties())


@app.get("/multifamily/cohorts")
def mf_cohorts(properties: str | None = None, as_of: str | None = None,
               _=Depends(require_key)):
    """Resident-cohort catalog (chatbot surface): every named cohort with live size and
    predicted performance at the next lease decision (avg churn, lift vs portfolio,
    expected move-outs). `properties` = optional comma-separated HPROPERTY ids."""
    from contracts.multifamily_churn import cohorts as mf_cohorts_mod
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    return clean(mf_cohorts_mod.catalog(pids, as_of))


@app.get("/multifamily/unit-economics")
def mf_unit_economics(properties: str | None = None, as_of: str | None = None,
                      _=Depends(require_key)):
    """The per-bedroom turnover-economics LEDGER (Phase 1 decision ledger): what one
    turn costs (vacancy months x in-place rent + make-ready, gross and net of the
    re-lease premium), how often units turn (occupied-unit-year denominators), and the
    expected loss per unit-year — per bedroom, per property. Make-ready is GL-measured
    where the property's ledger carries the U= unit-note convention (62W today) and a
    labeled flat fallback elsewhere. Assumption-free by construction: no model score,
    no uplift, no flip rate. Derivation: docs/PHASE0_DISCOVERY_62W_2026-08-18.md."""
    from contracts.multifamily_churn import unit_economics as ue
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    try:
        return clean(ue.build(pids, as_of))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/multifamily/persuadables")
def mf_persuadables(properties: str | None = None, as_of: str | None = None,
                    band_low: float = 0.20, band_high: float = 0.60,
                    incentive_cost: float | None = None, flip_rate: float | None = None,
                    top: int = 50, month: str | None = None, _=Depends(require_key)):
    """The EDGE-RESIDENT / incentive-target list: residents in the persuadable churn
    band (default 0.2-0.6 at the 180-day renewal decision, notice-filed excluded),
    ranked by monthly rent at risk. Scored on the h180 renewal-forecast bundle (the
    h60 triage default compresses the band — Phase 0 memo 2026-08-18). Each member
    carries WHY they're on the fence (conflicting risk/protective cohort memberships
    + the model's own drivers), their incentive history (past TRANS concessions +
    logged offers), and a value-if-saved (avg turn days x daily rent + $600
    make-ready). `properties` scopes to buildings ("edge residents at Winwood" ->
    properties=26); `month` (YYYY-MM) scopes to lease decisions landing that month.
    Scenario economics take incentive_cost (default: one month of band avg rent);
    `flip_rate` has NO default — omitted, the scenario reports the assumption-free
    breakeven flip rate instead of projected saves (no measured rate exists; log
    offers via POST /multifamily/outreach to measure one)."""
    from contracts.multifamily_churn import persuadables as mf_pers
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    try:
        return clean(mf_pers.build(pids, as_of, band_low, band_high, incentive_cost,
                                   flip_rate, top, month))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/multifamily/concession-cohort")
def mf_concession_cohort(properties: str | None = None, as_of: str | None = None,
                         band_low: float = 0.20, band_high: float = 0.60,
                         decision_window_days: int | None = 180,
                         include_notice: bool = False,
                         assumed_uplift: float | None = None,
                         garage_months_to_rerent: float = 2.0,
                         top: int | None = None, format: str = "json",
                         _=Depends(require_key)):
    """The CONCESSION COHORT (economics-intern surface, 2026-09-25): residents whose
    calibrated churn probability at the next lease decision sits in [band_low,
    band_high) — default 0.20-0.60, the agreed persuadable band shared with
    /persuadables and the edge roster (a 0.60-0.70 "30-40% chance of staying" band is
    empty on this model: without a notice h180 never exceeds ~0.45; band_low=0.30 keeps
    the top ~35) — scored on the h180 renewal bundle, with a decision inside `decision_window_days` (default 180;
    omit for none) and no notice filed (include_notice=true to add them). Rows are
    identifier-only (no names) and carry lease timing, rent (in-place/prior/market),
    tenure, the h180 drivers, cohort memberships, garage status (RENTABLEITEMS), consumed
    concessions + logged offers, INTERNAL duration estimates, the full model feature
    vector, the per-bedroom turnover ledger (TC_net, rate, expected loss) and, per offer
    (free month / 5% off / garage new / garage waiver), cost-if-renewed + breakeven
    uplift + verdict. No uplift is assumed unless `assumed_uplift` (0-1) is passed, which
    adds labeled scenario fields. `top` trims the displayed rows only (roll-ups are
    always over the whole cohort). format=csv streams one flattened row per resident."""
    from fastapi.responses import Response
    from contracts.multifamily_churn import concession_cohort as cc
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    try:
        out = cc.build(pids, as_of, band_low, band_high, decision_window_days,
                       include_notice, assumed_uplift, garage_months_to_rerent, top)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if (format or "json").strip().lower() == "csv":
        return Response(content=cc.to_csv(out), media_type="text/csv",
                        headers={"Content-Disposition":
                                 f'attachment; filename="concession-cohort-{out["as_of"]}.csv"'})
    return clean(out)


@app.get("/multifamily/edge-list")
def mf_edge_list(refresh: bool = True, history: bool = False, _=Depends(require_key)):
    """The STABLE edge roster (hysteresis, battery fix 2026-07-22): residents enter at
    churn p>=0.20 (no notice) and stay listed until something actually resolves —
    renewal, notice, move-out — or risk falls below 0.15. `fight_for` (stayer-type vs
    roster median, duration model) is decided at entry and never flips. Includes the
    0.15-0.20 watch band and this refresh's enter/exit events. history=true -> the
    audit log of all roster changes. refresh=false -> read-only (no state update)."""
    from contracts.multifamily_churn import edge
    if history:
        return clean(edge.roster_history())
    return clean(edge.edge_list(persist=refresh))


@app.get("/multifamily/incentives")
def mf_incentives(scode: str | None = None, property: int | None = None,
                  since: str | None = None, limit: int = 100, _=Depends(require_key)):
    """Incentive/concession history: consumed concessions from the Yardi TRANS ledger
    (the system of record — 8 landlord concession accounts) plus logged incentive
    OFFERS from the outreach log. Filter by resident `scode` and/or `property`
    (HPROPERTY id) and/or `since` (YYYY-MM-DD)."""
    from contracts.multifamily_churn import incentives
    return clean(incentives.history(scode, property, since, limit))


@app.get("/multifamily/cohort-report")
def mf_cohort_report(cohort: str, properties: str | None = None, as_of: str | None = None,
                     top: int = 10, _=Depends(require_key)):
    """One cohort's predicted performance: size, avg churn, lift, expected move-outs,
    tier mix, top-N at-risk members, and a chatbot-ready natural-language `summary`.
    `cohort` = a key from GET /multifamily/cohorts (e.g. holdover-mtm, multi-pet)."""
    from contracts.multifamily_churn import cohorts as mf_cohorts_mod
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    try:
        return clean(mf_cohorts_mod.report(cohort, pids, as_of, top))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/multifamily/lease-window")
def mf_lease_window(start: str, end: str, properties: str | None = None,
                    as_of: str | None = None, limit: int | None = None, _=Depends(require_key)):
    """Snapshot of residents whose lease EXPIRES within [start, end], scored by the churn
    model. `properties` = optional comma-separated HPROPERTY ids (default all 4). `as_of`
    defaults to min(today, start). Returns the same shape as /predict + window bounds."""
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    return clean(mf.score_lease_window(start, end, pids, as_of, limit))


@app.get("/multifamily/churn-report")
def mf_churn_report(month: str, properties: str | None = None,
                    include_scores: bool = True, top: int | None = None,
                    outcome: str | None = None, _=Depends(require_key)):
    """Monthly churn report: every resident whose lease expired/expires in `month`
    (YYYY-MM), their renewal OUTCOME as known today (renewed | churned | pending), and
    the model's churn score as of the month start. Cohort is reconstructed point-in-time
    from Yardi Lease_History, so past months keep their full size even after renewals
    overwrite TENANT.DTLEASETO. Rows are ranked highest churn risk first; `top=10`
    returns just the top 10, `outcome=pending` filters to a decision status first, and
    every response carries a chatbot-ready natural-language `summary` (always about the
    FULL month, not the trimmed rows). `include_scores=false` skips model scoring
    (fast). `properties` = optional comma-separated HPROPERTY ids (default all 5)."""
    if outcome and outcome.strip().lower() not in ("renewed", "churned", "pending"):
        raise HTTPException(400, "outcome must be renewed|churned|pending")
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    return clean(mf.churn_report_view(month, pids, include_scores, top, outcome))


@app.get("/multifamily/churn-trend")
def mf_churn_trend(start: str, end: str, properties: str | None = None,
                   include_predicted: bool = False, _=Depends(require_key)):
    """Month-over-month churn summary series for months [start, end] (YYYY-MM each):
    cohort size, renewed/churned/pending counts, churn + renewal rates. Default is
    outcome-only (fast, max 36 months). include_predicted=true adds the model's
    predicted + projected churn rate per month (scores every cohort: ~1 min per
    uncached month, cached 1h, max 12 months)."""
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    return clean(mf.churn_trend(start, end, pids, include_predicted))


class OutreachRecord(BaseModel):
    scode: str
    action: str = "note"
    staff: str | None = None
    note: str | None = None
    month: str | None = None
    churn_probability: float | None = None
    risk_tier: str | None = None
    incentive_cost: float | None = None   # expected $ if consumed (incentive_* actions)
    # typed-offer fields (Phase 0 measurement discipline) — offer_type is REQUIRED for
    # incentive_* actions; see contracts/multifamily_churn/outreach.py OFFER_TYPES
    offer_type: str | None = None         # free_months_1/2/3 | free_garage_new |
                                          # free_garage_waiver | rent_discount_pct |
                                          # post_notice_rescue_1mo | other
    offer_structure: str | None = None    # one_time_credit | recurring_discount_term |
                                          # recurring_discount_base | waiver
    offer_value: float | None = None      # months / pct / $ per the offer_type
    property_id: int | None = None
    bedroom: int | None = None
    arm: str | None = None                # treated | control (randomized trials)
    posting_account: int | None = None    # TRANS HRETENTIONACCT the credit posts to


@app.post("/multifamily/outreach")
def mf_outreach_record(body: OutreachRecord, _=Depends(require_key)):
    """Log a retention-outreach event for an at-risk resident (Phase 3.3). Pass the
    churn_probability/risk_tier shown to staff at contact time so a later uplift analysis
    can compare contacted vs comparable non-contacted residents. action in:
    called|emailed|visited|offer_sent|renewed_offer|no_answer|note."""
    try:
        return clean(mf_outreach.record(**body.model_dump()))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/multifamily/outreach")
def mf_outreach_history(scode: str | None = None, month: str | None = None,
                        summary: bool = False, _=Depends(require_key)):
    """Outreach log. `summary=true` -> coverage by month + action mix; else the event
    list, optionally filtered by `scode` and/or `month` (YYYY-MM)."""
    return clean(mf_outreach.summary() if summary
                 else mf_outreach.history(scode, month))


@app.get("/multifamily/calibration-drift")
def mf_calibration_drift(_=Depends(require_key)):
    """Phase 1.5: logged predicted-vs-realized churn rate for settled past months
    (monitoring/calibration.log). `alert=true` rows exceed the ±5pt band."""
    from pathlib import Path as _P
    import json as _j
    p = _P(__file__).resolve().parent.parent / "monitoring" / "calibration.log"
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(_j.loads(line))
            except Exception:
                continue
    return clean({"contract": "multifamily-churn", "view": "calibration-drift",
                  "count": len(rows), "alerts": sum(1 for r in rows if r.get("alert")),
                  "months": rows})


@app.get("/multifamily/tenant/search")
def mf_tenant_search(q: str, as_of: str | None = None, _=Depends(require_key)):
    """Resolve a (partial) resident NAME -> current SCODE(s). Exactly one match embeds the
    scored result + chatbot-ready `summary` under `tenant`; else returns `matches`."""
    return clean(mf.lookup_tenant(q, as_of))


@app.get("/multifamily/tenant/{scode}")
def mf_tenant(scode: str, as_of: str | None = None, _=Depends(require_key)):
    """Score ONE resident by SCODE -> churn_probability + stay_probability + risk_tier +
    lease/notice + a natural-language `summary` a chatbot can read back to the user."""
    return clean(mf.score_one(scode, as_of))


@app.get("/brokerage/companies")
def bk_companies(limit: int = 50, _=Depends(require_key)):
    return clean(bk.list_companies(limit))


@app.get("/brokerage/scorecard")
def bk_scorecard_ep(sort: str = "score", limit: int = 50, market: str | None = None, _=Depends(require_key)):
    """Company-centric, broker-free ranking: REA score + WinProb + EV + scoop momentum +
    lease proximity per company. sort=score|winprob|value|scoops|recency|lease."""
    return clean(bk_scorecard.company_scorecard(sort, limit, market))


class LeadEvent(BaseModel):
    company_id: int | None = None
    hmy: int | None = None
    action: str = "note"
    staff: str | None = None
    note: str | None = None
    market: str | None = None
    rea_score: float | None = None
    win_prob: float | None = None
    conversion_probability: float | None = None
    opportunity_tier: str | None = None
    source: str | None = None
    propensity: float | None = None      # space-demand score shown at contact time
    cohort_key: str | None = None        # marketing cohort the lead was worked from


@app.post("/brokerage/leads")
def bk_lead_record(body: LeadEvent, _=Depends(require_key)):
    """Log a lead-outcome event (Phase 3.2): which scorecard/space-demand lead a broker
    pursued and how it resolved. Snapshot the shown signals (rea_score/win_prob/
    opportunity_tier) for later uplift analysis. action in: pursued|called|emailed|
    meeting|proposal_sent|won|lost|disqualified|skip|note."""
    from contracts.signal_desk_brokerage import leadlog
    try:
        return clean(leadlog.record(**body.model_dump()))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/brokerage/leads")
def bk_lead_history(company_id: int | None = None, hmy: int | None = None,
                    summary: bool = False, _=Depends(require_key)):
    """Lead-outcome log. summary=true -> pursuit coverage + per-tier win rates (uplift)."""
    from contracts.signal_desk_brokerage import leadlog
    return clean(leadlog.summary() if summary else leadlog.history(company_id, hmy))


@app.get("/brokerage/space-demand-leads")
def bk_space_demand(market: str | None = None, limit: int = 25, _=Depends(require_key)):
    """PROTOTYPE space-demand lead-gen (self-contained ZoomInfo scoop panel): ranks in-market
    companies by propensity to post a Facilities Relocation/Expansion signal (a space NEED) in
    ~180d. Experimental; a prioritization list, NOT a conversion predictor. market=IA,NE."""
    return clean(spd.space_demand_leads(market, limit))


@app.get("/marketing/options")
def mkt_options(_=Depends(require_key)):
    return clean(mkt.options())


@app.get("/marketing/targeting")
def mkt_targeting(_=Depends(require_key)):
    return clean(mkt.targeting())


@app.get("/marketing/space-demand")
def mkt_space_demand(market: str | None = None, min_scored: int = 3, _=Depends(require_key)):
    """Per-cohort 'who needs space' rollup over the PROTOTYPE space-demand model (the
    same frozen bundle brokerage serves): expected space-need events in ~180d, hot-lead
    counts (top decile), and top previews per canonical marketing cohort
    (size band x vertical). market=IA,NE. Missing cohorts = no scoop data, not no demand."""
    return clean(mkt.space_demand(market, min_scored))


@app.get("/marketing/space-demand/leads")
def mkt_space_demand_leads(cohort: str, market: str | None = None, limit: int = 25,
                           _=Depends(require_key)):
    """Ranked space-need leads inside one marketing cohort. `cohort` accepts the canonical
    key ('size=mid|vertical=healthcare'), a single dim ('vertical=healthcare'), or a bare
    token ('healthcare'). Same propensities as /brokerage/space-demand-leads."""
    try:
        return clean(mkt.space_demand_leads(cohort, market, limit))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/models")
def models(_=Depends(require_key)):
    """Model-validation cards (training setup, ROC-AUC/Brier/log-loss, calibration,
    feature importance) for the technical sector."""
    from core import modelcard
    return clean(modelcard.build_all())


# ---- stand-up UI (served same-origin so the browser calls /predict/* with the key) ----
from pathlib import Path as _Path                 # noqa: E402
from fastapi.responses import HTMLResponse        # noqa: E402

_WEB = _Path(__file__).resolve().parent.parent / "web"


@app.get("/", response_class=HTMLResponse)
def ui_root():
    html = (_WEB / "index.html").read_text()
    return html.replace("__GBME_API_KEY__", db.ENV.get("GBME_API_KEY", ""))


import threading  # noqa: E402


@app.on_event("startup")
def _warm_cache():
    """Keep the slow views permanently warm so interactive callers (UI, chatbot) hit the
    cache: the default all-buildings score once at boot, then the monthly churn reports
    for a rolling window (previous month .. +3) recomputed every ~50 min -- inside the
    1h cache TTL, so a user/chatbot call for those months is always sub-second. The old
    copy keeps serving while the recompute runs (the cache entry is only replaced at the
    end), and each month is ~30-60s of feature-SP work in this background thread."""
    def _warm_months() -> list[str]:
        from datetime import date as _d
        t = _d.today()
        cur = (_d(t.year - 1, 12, 1) if t.month == 1 else _d(t.year, t.month - 1, 1))
        out = []
        for _ in range(5):
            out.append(cur.isoformat()[:7])
            cur = (_d(cur.year + 1, 1, 1) if cur.month == 12
                   else _d(cur.year, cur.month + 1, 1))
        return out

    def _go():
        import time as _t
        _t.sleep(20)                 # let boot settle so SP chunk calls don't race startup
        try:
            mf.score_buildings(list(mf_data.DEFAULT_PROPERTY_IDS))
        except Exception:
            pass
        while True:
            for m in _warm_months():
                try:
                    mf.monthly_churn_report(m, use_cache=False)   # force refresh + re-cache
                except Exception:
                    pass
            _t.sleep(50 * 60)
    threading.Thread(target=_go, daemon=True).start()

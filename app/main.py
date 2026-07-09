"""light_gbm_endpoint FastAPI service (v1). One service, 3 app contracts, live SSMS data.

Endpoints:
  GET  /health                              (open)
  GET  /contracts                           list contracts + manifests
  GET  /contracts/{contract}/schema         introspect input/output
  POST /predict/multifamily-churn           {scodes[] | property_ids[], as_of?}
  POST /predict/signal-desk-brokerage       {company, include_history}
  POST /predict/marketing                   {variant: cohort|industry, key}
  GET  /multifamily/buildings | /brokerage/companies | /marketing/options   (UI helpers)
"""
from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException

from app.auth import require_key
from app.schemas import MultifamilyRequest, BrokerageRequest, MarketingRequest
from core import registry, db
from core.jsonify import clean
from contracts.multifamily_churn import contract as mf, data as mf_data, schema as mf_schema
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
    return clean({"status": "ok", "db": db.healthcheck(), "contracts": registry.list_contracts()})


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


@app.get("/multifamily/lease-window")
def mf_lease_window(start: str, end: str, properties: str | None = None,
                    as_of: str | None = None, limit: int | None = None, _=Depends(require_key)):
    """Snapshot of residents whose lease EXPIRES within [start, end], scored by the churn
    model. `properties` = optional comma-separated HPROPERTY ids (default all 4). `as_of`
    defaults to min(today, start). Returns the same shape as /predict + window bounds."""
    pids = [int(x) for x in properties.split(",") if x.strip()] if properties else None
    return clean(mf.score_lease_window(start, end, pids, as_of, limit))


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
    """Pre-score the default all-buildings view so the first network user hits a warm
    cache (the full-portfolio SP recompute is slow)."""
    def _go():
        import time as _t
        _t.sleep(20)                 # let boot settle so SP chunk calls don't race startup
        try:
            mf.score_buildings([62, 72, 82, 26])
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()

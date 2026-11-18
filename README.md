# light_gbm_endpoint

A shared internal **prediction-serving API** for Realty Trust Services (RTS). It wraps trained **LightGBM** models as versioned, per-project **contracts** and exposes them over HTTP so any internal app can score by calling one service — instead of each app re-implementing feature engineering, calibration, and post-processing.

> **Status: v1 RUNNING (2026-06-23).** Live on the DGX Spark, CPU-only, reading the live SQL Server (SSMS) read-only. All 3 app contracts are callable; a Streamlit stand-up app fronts them.
>
> - **API:** `http://10.1.6.17:8910` (LAN) / `http://127.0.0.1:8910` — auth header `X-API-Key` (see `.env`)
> - **Stand-up app (UI):** `http://10.1.6.17:8911`
> - **Run:** `./scripts/serve.sh` (API) and `./scripts/serve_ui.sh` (UI), or install the systemd unit — see [deploy/README.md](deploy/README.md).
>
> Models are **interim stand-ins** (real-data-trained, leakage-prone, anchored): drop the production `bundle.joblib` into `bundles/{contract}/<version>/` and point `default.txt` to promote it. Design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick call
```bash
KEY=rts-local-dev-key
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"property_ids":[62],"limit":50}' http://10.1.6.17:8910/predict/multifamily-churn
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"company":"840"}' http://10.1.6.17:8910/predict/signal-desk-brokerage
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"variant":"cohort","key":"healthcare-vertical"}' http://10.1.6.17:8910/predict/marketing
```

This is the actionable, consumer-grounded successor to the design docs in `../multifamily-prediction/gbm_endpoint/` (which remain the design lineage). The difference here: it is **scoped to one real first model** (multifamily churn) and mapped against the **three real consumers** below.

> **Note (2026-08-18):** `../multifamily-prediction/v2/` was the original model source but has been stale since 2026-07-07. Active iteration on the multifamily churn and duration models — feature experiments, retraining, calibration, backtests, new bundle versions — now happens **in this repo** (`scripts/train_multifamily_*.py`, `scripts/experiment_*.py`, `bundles/multifamily_churn/`, `bundles/multifamily_duration/`). Treat `light_gbm_endpoint` as the live working copy for these models; `multifamily-prediction` is legacy/reference only.

## The one model we build first (scope)

**multifamily-churn** — a production-validated LightGBM tenant renewal/churn model. Its feature pipeline, model, calibration, and post-processing **originated** in `../multifamily-prediction/v2/`, but that repo has gone stale. Ongoing retraining and feature work (v5 → v7 and beyond, plus the newer duration model) now happens directly here — see `scripts/` and `bundles/multifamily_churn/`, `bundles/multifamily_duration/`. See [docs/CONTEXT_MULTIFAMILY.md](docs/CONTEXT_MULTIFAMILY.md) (also due for a refresh to match).

## The three consumers

| Consumer | Folder | Role | Feature mode | Status as a caller |
|----------|--------|------|--------------|--------------------|
| **Multifamily** | `../multifamily-prediction/` (legacy source) / **this repo** (active) | First caller (property teams score renewal cohorts); model training/iteration now lives here, not in `multifamily-prediction` | identifier-in (Yardi SP) | Ready — actively iterated (churn + duration bundles) |
| **Brokerage** | `../signal-desk/` | CRE prospect scoring (`rea.ProspectScore`, today rules-based `v0.1-rules`) | feature-in (its own `rea.ProspectFeatureSnapshot`) or identifier-in | Caller-ready; its **own** GBM contract is data-blocked (~78 positives, needs ≥500) |
| **Marketing** | `../marketing/` | Space↔cohort fit scoring (today an LM Studio LLM) + prospect ranking | feature-in (computes vectors locally) | Caller-ready; wants a fast deterministic replacement for LLM scoring |

Full per-consumer maps: [CONTEXT_MULTIFAMILY](docs/CONTEXT_MULTIFAMILY.md) · [CONTEXT_BROKERAGE](docs/CONTEXT_BROKERAGE.md) · [CONTEXT_MARKETING](docs/CONTEXT_MARKETING.md).

## Planned API surface

```
POST /predict/{contract}          # body: version?, variant, inputs   -> calibrated prediction(s), model-version stamped
GET  /contracts                   # list contracts + published versions
GET  /contracts/{contract}/schema # introspect input/output schema
GET  /health                      # liveness (unauthenticated)
```

## Layout

```
light_gbm_endpoint/
├── app/                        # FastAPI: main.py (routes+logging), auth.py, schemas.py
├── core/                       # db.py (live SSMS), registry.py (bundles), jsonify.py
├── contracts/
│   ├── multifamily_churn/      # data.py, prepare_features.py, contract.py, schema.py
│   ├── signal_desk_brokerage/  # data.py, prepare_features.py, contract.py, schema.py
│   └── marketing/              # data.py, contract.py, schema.py (aggregate mode)
├── bundles/{contract}/{version}/   # manifest.json (tracked) + bundle.joblib (gitignored, out-of-band)
│   └── multifamily_churn/, multifamily_duration/   # actively iterated here — see scripts/train_multifamily_*.py
├── frontend/app.py             # Streamlit stand-up app (3-app switcher + charts)
├── scripts/                    # serve.sh, serve_ui.sh, train_*, experiment_*, backtest_*, introspect/eval_*
├── deploy/                     # systemd unit + deploy README
├── monitoring/logs/            # PII-free request log
├── docs/                       # ARCHITECTURE + 3 context maps + CONTRACTS
├── .env                        # live DB creds + API key (gitignored)
└── requirements.txt
```

## Related (knowledge base)
- [[gbm-endpoint-model-server]] — the design topic in the vault
- [[multifamily-churn-lightgbm-model]] — the model being served
- Companion report: `../outputs/multifamily-prediction/reports/2026-06-23-gbm-endpoint-on-dgx-spark.md`

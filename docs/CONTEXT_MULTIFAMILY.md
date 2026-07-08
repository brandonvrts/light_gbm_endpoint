# Context Map — Multifamily (the model source + first caller)

**Project:** `../multifamily-prediction/` (production code in `v2/`) · **Feature mode:** identifier-in · **Role:** supplies the v1 contract; also the first consumer.

This is the project we **pull from**. The endpoint's job for this consumer is to wrap the *existing, validated* v2 assets behind HTTP — not to rebuild any model logic.

## 1. Data sources
- **DB:** SQL Server `rrSQL2024`, database `InsightMart_Test` (test) / `InsightMart` (prod), sourced from `Yardi7_Reporting.dbo`.
- **Stored procedures** (`v2/sql/`):
  - `sp_GetTenantFeaturesByScode(@ScodeList NVARCHAR(MAX), @AsOfDate DATE=NULL, @WindowMonths INT=18)` → point-in-time inference rows for a SCODE list. **This is the call the endpoint makes for identifier-in.**
  - `sp_GetAnchorSnapshot(@AnchorDate, @HorizonDays=60, @WindowMonths=18)` → training rows (adds the `MoveOutInHorizon` label). Training-time only.
  - `sp_GetVacancyByPropertyBedroom` → vacancy roll-ups (economics layer).
- **Entity / identifier:** caller passes **SCODE** (`'t0001847'`); SP resolves to `HMYPERSON` internally.
- **Scope filter:** `HPROPERTY IN (62, 72, 82, 26)` = Country Club Village / Sierra Pointe / Residences at 62W / Winwood = **976 rentable units** (`UNIT.EXCLUDE = 0`).

## 2. The artifact to serve (model bundle)
Self-contained `joblib` dict saved by `v2/train_model_v4.py`, filename `churn_model_v4_{h60|h180}_{timestamp}.joblib`. Keys:
```
model, scaler, property_encoder, pet_encoder, feature_columns (45),
target_column ('MoveOutInHorizon'), feature_medians (training-time, for imputation),
version ('v4-anchor'), trained_at, horizon_days, algorithm, data_source,
train_window{...}, thresholds{churn_decision, risk_low_max, risk_medium_max},
calibration{method, prob_floor, prob_ceil, ...}, metrics{walkforward,...}, feature_importance[]
```
- **Feature count: 45 (confirmed)** — tenure/lease, rent + RentIncreasePct, notice/NSF, work-order aggregates (counts, recency 90/180/365d, emergency, open-at-anchor), memo aggregates (eviction/payment/complaint/maintenance), pets, `PropertyName_encoded`, `PrimaryPetType_encoded`.
- **Two variants:** `h60` (60-day triage **ranker** — over-confident, ROC-AUC 0.749) and `h180` (180-day renewal forecast — the business model: walk-forward ROC-AUC **0.891**, PR-AUC **0.945**, Brier **0.083**).

## 3. How scoring is invoked (today, and via the endpoint)
**Today:** CLI `python churn.py score --scodes t0001,t0002 --as-of 2026-06-01`. Path: load latest bundle → `EXEC sp_GetTenantFeaturesByScode` → `prepare_features(df, property_encoder, pet_encoder, feature_columns, feature_medians)` → `predict()` (scale → `predict_proba` → clip `[floor,ceil]`) → risk tier → CSV.
- `prepare_features` lives in `v2/validate_april_renewals_v3.py:267` — encodes property/pet (with unseen-class fallback), imputes from `feature_medians`, selects the 45 columns. **This function ports verbatim into the contract's feature stage.**
- ⚠ Known bug to fix on port: `churn.py` loads bundles with a stale glob `churn_model_v1.1_v3_*.joblib`; the v4 save name is `churn_model_v4_*`. Use the v4 glob.

**Via the endpoint (identifier-in):** caller POSTs `{version, variant, as_of_date, scodes[]}`; the service runs the same SP + `prepare_features` + bundle server-side and returns `{model_version, results:[{scode, churn_probability, risk_tier, ...}]}` — **SCODE + probability + tiers only, no tenant names**.

## 4. Config / secrets (the pattern to adapt)
`v2/config.py`: env prefix **`CHURN_`** — `CHURN_SERVER`, `CHURN_DATABASE`, `CHURN_USERNAME`, `CHURN_PASSWORD` (+ optional `CHURN_MODEL_ALGO`, `CHURN_CALIBRATION`, `CHURN_PROB_CEIL`). No source defaults; loads `.env` then OS env. ODBC `{ODBC Driver 18 for SQL Server}`, `TrustServerCertificate=yes`. The endpoint re-uses this exact connection approach inside `core/feature_io.py` (renamed to a `GBME_`/per-contract prefix).

## 5. Benefits to multifamily
- Property teams get scoring as a **service** (callable from a dashboard / Excel / nightly job) instead of a manual CLI run.
- The feature pipeline is **frozen with the bundle** → no train/inference skew across callers.
- Versioned bundles + default-pointer + rollback → safe model promotion.

## 6. Reusable assets checklist (what the endpoint lifts)
- [ ] `v2/train_model_v4.py` bundle format → frozen under `bundles/multifamily_churn/{version}/`
- [ ] `prepare_features()` (`validate_april_renewals_v3.py:267`) → `contracts/multifamily_churn/prepare_features.py`
- [ ] `sp_GetTenantFeaturesByScode` call → `core/feature_io.py` (identifier-in)
- [ ] `predict()` + clip + risk-tier logic → `contracts/multifamily_churn/contract.py` post-processing
- [ ] `recalibrate_h180.py` isotonic + 32.3% anchor → applied when freezing the h180 bundle (see ARCHITECTURE §calibration)
- [ ] `config.py` `CHURN_` connection pattern → endpoint config + feature_io

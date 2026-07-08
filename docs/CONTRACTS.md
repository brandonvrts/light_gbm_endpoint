# Contracts

A **contract** = input schema + feature pipeline + model bundle + post-processing + output schema, **versioned together**. One contract is in scope now (`multifamily_churn`); the others are documented so the abstraction is built to fit them.

## In scope — `multifamily_churn`
- **Feature mode:** identifier-in (Yardi). Caller sends IDs + as-of date; service computes features.
- **Variants:** `h180` (renewal forecast — the business model, isotonic + 32.3% anchor) and `h60` (triage ranker).
- **Input:** `{ "version": "v4", "variant": "h180", "as_of_date": "2026-06-23", "scodes": ["t0001847", ...] }`
- **Output (no PII):** `{ "model_version": "v4.0.0", "as_of": "2026-06-23", "results": [ { "scode": "t0001847", "churn_probability": 0.118, "risk_tier": "low" }, ... ] }`
- **Bundle:** frozen recalibrated LightGBM bundle (see ARCHITECTURE §4). Backing DB `Yardi7_Reporting.dbo` via `sp_GetTenantFeaturesByScode`; scope `HPROPERTY IN (62,72,82,26)`.

## Future — `signal_desk_brokerage` (brokerage)
- **Feature mode:** feature-in initially (Signal Desk POSTs `rea.ProspectFeatureSnapshot` vectors) → identifier-in later if given commercial-table access.
- **Entity:** commercial prospect `hMy`. **Blocked on labels** (~78 positives, need ≥500). Replaces `rea.ProspectScore` `v0.1-rules` → `v1.0-xgb`.

## Future — marketing contracts (`marketing_space_fit`, `marketing_propensity`)
- **Feature mode:** feature-in (marketing computes vectors locally). **New models to train** (not the multifamily bundle). Replace today's LM Studio LLM fit-scoring with fast deterministic GBM + SHAP.

## Adding a contract — checklist
- [ ] Input + output schema (Pydantic) → surfaced via `GET /contracts/{c}/schema`
- [ ] Feature mode (identifier-in vs feature-in)
- [ ] Feature pipeline, versioned with the bundle (identifier-in: wire to its SP through `feature_io`)
- [ ] Freeze the `joblib` bundle under `bundles/{contract}/{version}/` + `manifest.json`
- [ ] Post-processing (calibration, thresholds, tiers)
- [ ] Register in the manifest; set default-version pointer; rollback target
- [ ] AAD app registration(s) + optional per-contract scope
- [ ] Smoke test: bundle loads + reproduces a known prediction

Place code under `contracts/{contract_name}/`, bundle under `bundles/{contract_name}/{version}/`.

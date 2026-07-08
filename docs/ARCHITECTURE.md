# light_gbm_endpoint — Architecture (scoped to multifamily-churn)

**Status:** scoping/scaffold, 2026-06-23. **Scope:** build exactly **one** contract — `multifamily-churn` — by wrapping the existing, validated `../multifamily-prediction/v2/` assets. Brokerage and marketing are documented future callers (see their context maps), explicitly **out of scope for the first build**, but the abstractions below are designed so they drop in later.

## 1. Thesis
The model logic is **frozen and trusted**. The new work is the API, the registry, auth, and the data seam. **If the wrapper changes the numbers, the wrapper is wrong.** Every design choice below optimizes for: (a) zero train/inference skew, (b) reusing v2 verbatim, (c) a second contract being a drop-in.

## 2. The contract (5 parts, versioned together)
A contract is the whole path from caller input to caller output, versioned as one atomic unit. For `multifamily-churn`:

| # | Part | multifamily-churn realization | Source in v2 |
|---|------|-------------------------------|--------------|
| 1 | Input schema | `{version?, variant: h60\|h180, as_of_date, scodes[]}` | new (Pydantic) |
| 2 | Feature pipeline | `sp_GetTenantFeaturesByScode` → `prepare_features()` (45 cols, encoders, medians) | `validate_april_renewals_v3.py:267` |
| 3 | Model bundle | `joblib` dict (model, scaler, encoders, feature_columns, calibration, thresholds, …) | `train_model_v4.py` save |
| 4 | Post-processing | scale → `predict_proba` → clip `[floor,ceil]` → risk tier (+ optional capturability/economics) | `predict()` + thresholds |
| 5 | Output schema | `{model_version, as_of, results:[{scode, churn_probability, risk_tier, …}]}` — **no PII** | new |

Bumping the version mints a coherent snapshot of all five. A caller asking `version:"v4"` gets exactly the schema + pipeline + model + post-proc that were validated together.

## 3. Request flow (identifier-in)
```
POST /predict/multifamily-churn   (AAD JWT bearer)
  → app/auth: validate token + (optional) per-contract scope
  → app/schemas: validate body (variant in {h60,h180}, as_of date, scodes well-formed)
  → core/registry: resolve {contract, version|default-pointer, variant} -> Contract + cached bundle (LRU)
  → core/feature_io (identifier-in): EXEC sp_GetTenantFeaturesByScode @ScodeList,@AsOfDate  (Yardi read-only acct)
        -> contracts/multifamily_churn/prepare_features: encode + impute(medians) + select 45 cols
  → contracts/multifamily_churn/contract.predict: scaler.transform -> model.predict_proba -> clip -> risk tier
  → response stamped with resolved model_version ; monitoring/: log request+prediction (no names)
```
feature-in callers (future: marketing, optionally signal-desk) skip the SP and pass the vector; `feature_io` validates it against the schema and hands it straight to part 4.

## 4. Freezing the bundle (Phase 0 — the one subtlety)
`v2/train_model_v4.py` defaults to **sigmoid** calibration and an **xgb**-overridable algorithm. The **published h180 business model is different**: it is rebuilt by `recalibrate_h180.py` to use **isotonic regression + a 32.3% validated-rate cohort anchor**, clipped `[0.01, 0.99]`, on **LightGBM**. These are two stages, not a contradiction: train → recalibrate → deploy.

**Decision for the contract:** freeze the **recalibrated h180 bundle** (LightGBM, isotonic + 32.3% anchor) as `multifamily_churn/v4.0.0` for the `h180` variant, and the `h60` bundle for the ranker variant. Record the exact `algorithm` + `calibration.method` + `feature_columns` (45) read from the frozen bundle in its manifest, and assert them in the smoke test. This closes the "isotonic vs sigmoid / 45 vs 58 / lgbm vs xgb" ambiguity by pinning to the actual artifact rather than to either doc or trainer-default.

## 5. Module design (maps stub files → responsibility)
- **`app/main.py`** — FastAPI app; 4 routes (`/predict/{contract}`, `/contracts`, `/contracts/{c}/schema`, `/health`). Routing + DI only.
- **`app/auth.py`** — Azure AD / JWT bearer validation (issuer, audience, signature, expiry); optional per-contract scope check. Platform-agnostic. Runs **before** any bundle load or SP call.
- **`app/schemas.py`** — Pydantic request/response models for multifamily (identifier-in body; SCODE-only output).
- **`core/contract.py`** — the `Contract` abstraction: `validate / compute_features / predict / post_process / describe`. Generic across contracts.
- **`core/registry.py`** — filesystem + manifest registry: resolve `{name,version}`, **default-version pointer**, **promote/rollback** (pointer move over immutable version dirs), **LRU bundle cache** (immutable → never invalidate, new version = new key). Stamps resolved version onto responses.
- **`core/feature_io.py`** — the data seam. identifier-in: the Yardi read-only connection (`{ODBC Driver 18 for SQL Server}`, the `v2/config.py` `CHURN_` pattern) + SP call. feature-in: schema-validate a supplied vector. **The only place DB credentials live.**
- **`contracts/multifamily_churn/`** — `prepare_features.py` (port of v2), `schema.py` (in/out), `contract.py` (wires the 5 parts, post-processing = clip + risk tiers).
- **`bundles/multifamily_churn/{version}/`** — frozen `bundle.joblib` + `manifest.json` (contract, version, variants, pipeline version, algorithm, calibration, feature count, created, owner). **`.joblib` gitignored.**
- **`monitoring/`** — append request+prediction (model-version stamped, no names) for later actual-join + calibration re-validation.
- **`tests/test_smoke.py`** — offline: config loads, schema contract, **bundle loads + reproduces a known prediction** (catches version/arch skew), CLI/route parsing. No DB needed.

## 6. Compute & hosting
- **CPU-only.** LightGBM scoring is sub-ms tree traversal; 976 units score in <1s on a few cores. No GPU.
- **Two viable hosts:** (a) FastAPI as a **Windows service behind IIS** (matches internal `fastapi-yardi-backend`; near Yardi); (b) **this DGX Spark (ARM64 Linux)** as the CPU complement to its GPU LLM role — swap IIS→nginx/Caddy, Windows-service→systemd; needs `msodbcsql18` (aarch64, verified available) for the Yardi path. Full analysis: `../outputs/multifamily-prediction/reports/2026-06-23-gbm-endpoint-on-dgx-spark.md`. **Decide per ops preference; record a decision note.**
- **Auth/PII:** AAD app registration per caller; optional per-contract scopes; responses are **SCODE + probability + tiers only**.

## 7. Build order (scoped to multifamily)
| Phase | Deliverable | Mostly-existing? |
|-------|-------------|------------------|
| 0 Extract/freeze | Copy + freeze recalibrated h180 + h60 bundles into `bundles/multifamily_churn/v4.0.0/` + manifest | ✅ assets exist |
| 1 Contract+registry | `core/contract.py` + `core/registry.py` (FS+manifest, default pointer, rollback, LRU) | new |
| 2 Service | `app/` 4 routes + AAD/JWT + `feature_io` (SP + `prepare_features` port) + version stamp | new wrap of v2 |
| 3 Deploy | host (IIS *or* systemd+nginx) + request/prediction logging | new |
| 4 Govern | named owner, promote/rollback runbook, drift/calibration monitoring (log→join actuals), async/batch, rate limits | new |
| 5 Second contract | onboard `signal_desk_brokerage` (feature-in or identifier-in via commercial tables) **when labels ≥500** | future |

## 8. Risks / decisions to confirm before Phase 2
- **Pin library versions** (lightgbm/scikit-learn/numpy/scipy) to the bundle's training env — joblib pickles are version-sensitive; enforce via the smoke test.
- **Fix the bundle-load glob** when porting (`churn_model_v4_*`, not the stale `v1.1_v3` glob).
- **Confirm Yardi network reachability** from the chosen host (esp. the Spark) to `rrSQL2024:1433`.
- **Output schema (part 5) is "locked last":** ship minimal `{probability + model_version}` first; add `risk_tier`/capturability/economics under a new version once the business layer is final.
- **Secret hygiene:** never carry the legacy `../multifamily-prediction/config.py` hardcoded-password pattern; the endpoint uses env/`.env` only (rotate that leaked credential separately).

# Context Map — Brokerage / Signal Desk (consumer)

**Project:** `../signal-desk/` (app in `signal-desk/signal-desk/`) · **Feature mode:** feature-in (or identifier-in) · **Role:** future caller; its own `signal_desk_brokerage` contract is the Phase-5 second contract, currently **data-blocked**.

CRE sales-intelligence platform for REA brokers (Des Moines / Omaha). React SPA → Node 20 + Express API → SQL Server, with a nightly PowerShell pipeline. It already has a scoring surface; the endpoint replaces its rules engine when CRE labels are sufficient.

## 1. Data sources
- SQL Server `InsightMart` on `rrSQL2024`, three schemas:
  - `zi.*` — ZoomInfo firmographics/scoops (Company ~1.2k, Contact ~2.2k, Scoop ~1.7k).
  - `dbo.*Snapshot` — nightly Yardi **commercial** snapshots: `CommercialProspectSnapshot` (~1,881, keyed `hMy`), `CommAmendmentSnapshot` (~41k), `CommDealStageHistorySnapshot` (~96k), `CrmActivitySnapshot` (~8.7k).
  - `rea.*` — app layer: `ProspectCompanyLink`, **`ProspectFeatureSnapshot`** (40+ feature cols, point-in-time), `ProspectLabelSnapshot` (0/1 conversion in 180d; ~48k snapshots), **`ProspectScore`** + `ScoreRun`.
- **Entity:** commercial **prospect** (Yardi `hMy`); the Ranking surface also scores `CompanyID` directly for cold ZI-only companies.

## 2. The model integration point (the call site)
- **Today = rules-based v0.** `rea.vw_CompanyScore` (`sql/47-rea-vw-company-score.sql`) sums 11 weighted sub-scores (scoop recency/volume, inference strength, mentions, lease proximity, size/industry fit, pipeline engagement, broker activity, bookmark velocity, deal momentum) → 0–100 → per-market decile. Materialized nightly into `rea.ProspectScore` with `ModelVersion = 'v0.1-rules'`.
- **Planned = `v1.0-xgb`.** `rea.ProspectScore.ModelVersion` already reserves the GBM slot; docs (`data-model.md`, `docs/scoring-and-win-model.md`) say "upgrade to gradient-boosted trees after the training clock banks thousands of labels."
- **Blocker:** only ~78 distinct positive labels (signed amendments within 180d) in all history; ≥500 needed for a stable classifier. This is a **data** problem on this tenant, not architecture.

## 3. How it would call the endpoint
- **Nightly batch (primary):** `api/refresh-company-score.mjs` (run from `scripts/nightly-refresh.ps1`) currently materializes the rules view. Swap to: POST the batch of `rea.ProspectFeatureSnapshot` rows → endpoint returns probabilities + deciles → `MERGE rea.ProspectScore`.
- **Live API (secondary):** `api/routes/prospect.mjs` (GET `/api/prospect/:hMy`) for on-demand fresh scoring on a prospect detail page.
- **Mode:** most naturally **feature-in** — Signal Desk already assembles `ProspectFeatureSnapshot`, so it POSTs the vector and the endpoint only does inference + calibration. (identifier-in is possible later if the endpoint is given Yardi commercial-table access.)
- **Identifier passed:** `hMy` (+ `SnapshotDate`) alongside the feature vector.

## 4. CRE-specific features (differ from residential)
Entity is a multi-location company, not one lease. Adds: amendment **type** (renewal/expansion/contraction/proposal), deal-stage history (Proposal→Won state machine), lease economics (TI, CAM, options), co-tenancy/co-mention signals, broker-conviction layering (Yardi-tracked vs cold ZI-only). ~80% of the signal family overlaps residential (firmographics, engagement recency, size); the rest is CRE structure.

## 5. Benefits to brokerage
- **Leverages the multifamily model IP** and the shared contract/calibration/governance instead of standing up bespoke ML ops on ~78 labels.
- A consistent, calibrated probability + decile feeding the Ranking view, morning Brief, and Prospect 360 — replacing the rules artifact (current decile distribution has gaps: no D7/D9, 27% piled at p=0 in D10).
- When CRE labels reach ≥500, its `signal_desk_brokerage` contract slots into the same registry with no new infra.

## 6. Implication for our scope
Build the endpoint and the contract abstraction now (multifamily proves it). Signal Desk consumes the **infrastructure** immediately if useful, but its **own** contract waits on label volume — so design the registry/auth/feature-in path so a second contract is a drop-in, and don't block multifamily on CRE readiness.

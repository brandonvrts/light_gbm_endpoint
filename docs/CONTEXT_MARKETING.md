# Context Map — Marketing (consumer)

**Project:** `../marketing/` · **Feature mode:** feature-in · **Role:** future caller; wants a fast, deterministic GBM to replace LLM-based scoring.

A mature, actively developed internal tool: **ASP.NET Core 9** server + **Vite/React** SPA ("Marketing Hub" for social-post generation, approval workflows, and marketing-cohort discovery) plus a LED-wall video upscaler. Already calls external HTTP services (LM Studio, MailChimp), so adding an HTTP model call fits its patterns.

> Note: some table names below were read from the marketing codebase by survey; verify exact `mkt.*` / `rea.*` names against the project before wiring.

## 1. Data sources
- **Reads** from `InsightMart` `rea.*` (shared with signal-desk): e.g. `rea.vw_BootstrapProspects`, `rea.EntityConcept` (ontology tags: tech-vertical, growth-mode…), `rea.DealPattern` (size × vertical deal-history cohorts), `rea.CompanyInference`, `rea.EntityLink`.
- **Writes** `mkt.*`: e.g. `SpaceInferenceFit` (space×cohort fit cache), `CohortCatalog`, `Campaign`, `ContentDraft`, `AudienceSegment`, `PostEngagement`, `SignalTrigger`.
- **External:** MailChimp / LinkedIn / Meta (channel pushes), **LM Studio** (on-prem LLM for fit narratives).
- **Stack:** C# (Dapper + SQL Server) backend; React/TypeScript SPA; Python Real-ESRGAN subprocess for video (unrelated to scoring).

## 2. The model integration point (the call site)
- **Space→cohort fit scoring** is today computed **nightly by an LLM** (LM Studio) producing a 0–1 fit score + narrative per (space, cohort), cached in `mkt.SpaceInferenceFit`. **This is the prime GBM replacement** — a tree model scores in milliseconds, deterministically, with SHAP reasons.
- **Prospect match/rank** (e.g. `SignalsStore.MatchProspectsAsync`) ranks companies by a weighted composite (fit, decile band, signal recency, ontology confidence) — a candidate for a propensity GBM.

## 3. How it would call the endpoint
- **Mode: feature-in.** Marketing computes feature vectors locally from `mkt`/`rea` (space attributes: sqft, floor, rent, building/neighborhood; company signals: size, scoop recency, ontology tags) and **POSTs the vector**; the endpoint returns scores. Identifier-in is undesirable here (would couple the endpoint to marketing's data joins).
- **Natural endpoints to add on the C# side:** `POST /api/marketing/inferences/score-space` (space features → cohort fit probabilities) and/or `POST /api/marketing/prospects/rank` (candidates → propensities), each calling `light_gbm_endpoint` `POST /predict/{contract}` and persisting to `mkt.SpaceInferenceFit` / ranking tables.
- **Auth:** its own Azure AD app registration + bearer token (it already has a gateway-signed identity model).

## 4. What it would want predicted
Space-to-cohort fit probability (per cohort), company propensity to lease in a cohort, campaign conversion/engagement likelihood. These are **separate contracts** from multifamily churn (different features/labels) — marketing is a *feature-in* caller, so onboarding it later is "validate a vector against a schema + run a bundle," no Yardi access needed.

## 5. Benefits to marketing
- **Speed + determinism:** replace nightly LLM batches with sub-ms synchronous scoring; real-time re-ranking as listings change.
- **Repeatability:** deterministic scores make A/B tests and engagement metrics meaningful (LLM scoring varies run-to-run).
- **Explainability:** SHAP "top-3 reasons this space fits retail" surfaced in the UI.
- **Cost / footprint:** no per-call LLM compute; predictable CPU footprint; frees LM Studio/GPU for generative work.

## 6. Implication for our scope
Marketing validates the **feature-in** half of the contract abstraction (the non-Yardi path) and motivates **multiple contracts** beyond churn. Out of scope for the first build (which is multifamily only), but the `core/feature_io.py` feature-in branch and `GET /contracts/{c}/schema` introspection should be designed now so marketing can self-serve a vector later. Marketing's GBMs (space-fit, propensity) are **new models to train**, not the multifamily bundle — track separately.

# Brokerage (Signal Desk) Models & API — Full Audit and Improvement Roadmap

**Date:** 2026-07-16 · **Scope:** everything served under `/predict/signal-desk-brokerage` and `/brokerage/*`, plus the four brokerage bundle families (`signal_desk_brokerage`, `signal_desk_dealwin`, `signal_desk_recency`, `signal_desk_spacedemand`), their training pipelines, and the `rea.*` data they depend on.

> **STATUS UPDATE (2026-07-16):** Parts 1–3 below are the original audit. The roadmap was then IMPLEMENTED the same day — see **"Implementation status"** at the bottom. The negatives in Part 2 describe the pre-implementation state; most are now fixed (default model is `v0.5-split-honest`). Two items remain owner-blocked (restart the frozen nightly job; backfill 2020–2023 features).

Method mirrors the multifamily audit (2026-07-15): every claim below was verified against the live DB, the live service, or a leakage-corrected re-run of the deployed model's own evaluation. Companion context: `docs/CONTEXT_BROKERAGE.md`, `docs/SIGNAL_DESK_DATA_VALIDATION.md`.

The brokerage surface differs from multifamily in one important way: the primary product signal (`/brokerage/scorecard`) is a **rules-based live read** of `rea.CompanyScoreSnapshot`, not ML. The ML conversion model (`v0.3-corrected-logistic`) is served on a secondary endpoint and is self-labeled INTERIM. Two further model families (dealwin, recency) were built, honestly evaluated, **failed**, and sit unserved; one prototype (space-demand) shows real lift.

---

## Part 1 — What's working (positives)

| Area | Evidence |
|---|---|
| **Statistical honesty culture** | Manifests self-label INTERIM / PROTOTYPE / shadow. The v0.3 stack is genuinely small-n-defensible: Riley event-budget feature cap (≤8 features for ~54 events), Firth+FLIC logistic, paired-bootstrap CIs, no resampling. |
| **The process caught its own leaks** | dealwin v0.1's +0.07 AUC edge was traced to `leadBroker` (an era proxy: prevalence 7%→40% train→test) by its own graduation gate; the leak-free v0.2 showed no signal (AUC ≈ 0.5) and was correctly **not** shipped. The broken proposal-keyed `ProspectLabelSnapshot` was caught (stand-in AUC 0.993 = red flag) and replaced with the corrected `vw_Deal → CommAmendmentSnapshot` bridge label. The production Signal Desk app never consumed the broken label. |
| **Evidence-driven scoop decision** | After the zi.Scoop backfill (~112k rows, 2022→2026), scoops were re-tested for conversion lift with a purpose-built paired ablation; delta CI spanned 0 twice → v0.4-corrected-scoops correctly left unpromoted despite a higher point estimate (0.115 vs 0.108). |
| **The deployed model has real skill** | Identity-honest check (5-fold GroupKFold by prospect, this audit): PR-AUC **0.087** vs 0.008 no-skill ≈ **11× lift on companies the model never saw**. Top decile of served scores captures 193/259 positive snapshots. Level calibration is decent: mean served p 0.0092 vs realized base 0.0078 (×1.18). Scores are smooth (1,196 distinct values — no snapping). |
| **Space-demand prototype works** | PR-AUC 0.335 vs 0.037 base (~9×), honestly labeled "prioritization list, NOT a conversion predictor", fast (0.5s cold / 2ms cached). |
| **Serving is fast** | score_company 0.09s (19-snapshot history), scorecard SQL sub-second, space-demand 0.5s cold. No latency problem anywhere. |
| **Ops scaffolding exists** | Versioned bundles + manifests, deterministic trainer, retrain pipeline wired with a PR-AUC gate, read-only DB seam, injection-safe parameterization throughout. |

## Part 2 — What's broken or weak (negatives, ranked)

1. **LIVE BUG: the scorecard's default call returns HTTP 500.** `company_scorecard(sort="score")` builds `ORDER BY c.Score DESC, c.Score DESC` (the tiebreak duplicates the sort column) — SQL Server error 169. Every default-sort call to the flagship brokerage endpoint fails; only non-default sorts (`winprob`, `value`, `lease`) work. One-line fix (`scorecard.py:80`, tiebreak on `c.CompanyID`).
2. **The ML-side data pipeline has been dead since April.** `rea.ProspectFeatureSnapshot` last row **2026-04-30**; `rea.ProspectScore` (rules) last scored **2026-04-21**. (`rea.CompanyScoreSnapshot` IS fresh — updated this morning — so the upstream job is only *partially* dead.) The conversion endpoint silently serves 2.5-month-old predictions: every response today says `data_as_of: 2026-04-30`, `rules_score_stale: true`. Nothing alerts; `/health` doesn't check it.
3. **The deployed headline metric is inflated by evaluation leakage.** Verified by re-running the trainer's own protocol (reproduces PR-AUC 0.107 ≈ manifest 0.1082), then correcting it: (a) elastic-net feature selection runs on the **full dataset before** the walk-forward; (b) 180-day label windows straddle quarter boundaries and prospects persist across quarters — **69% of test-quarter positives (154/222) belong to prospects already labeled positive in training**; 48/54 converting prospects span ≥2 quarters; (c) an identity-honest *temporal* eval is **infeasible** on this panel (excluding test-quarter prospects leaves zero valid folds at any gate). The honest generalization number is the grouped-CV **PR-AUC 0.087** (~20% below the reported 0.108). Still real skill — but the reported eval cannot detect prospect memorization by construction.
4. **The model is largely a one-feature echo.** The served score's Spearman correlation with `days_in_current_stage` is **−0.93** — the ranking is nearly a monotone transform of one CRM field ("recently moved stage → hot"). 4 of its 8 selected features are pipeline-position; the team's own ablation shows PR-AUC collapses 0.108 → 0.032 without them. As shipped, it mostly re-detects deals already in motion — low value for *cold* prospecting, and it conflates "already closing" with "worth waking up" (the docs' own unexecuted v0.5 recommendation). |
5. **Risk tiers are decorative.** Thresholds are hardcoded 0.33/0.66 but the maximum served probability is **0.176** → **100.0% of all rows tier as "low"**, at every company, always. The tier has never distinguished anything. (Also `risk_tier` naming is inverted for brokerage — here high probability is a *good* lead.)
6. **Promotion/audit-trail inconsistencies.** `monitoring/retrain.log` records a real (non-dry-run) PROMOTE of `v0.3-cand-20260707_140956` at 2026-07-07 14:10, but `default.txt` was hand-reverted to `v0.3-corrected-logistic` two minutes later with **no log entry** — the deployed pointer contradicts the last logged decision (harmless only because the deterministic retrain made the candidate byte-identical). The gate itself is weak: absolute tol 0.01 ≈ 9% of the incumbent metric, no calibration gate (multifamily has one), and a deterministic trainer + unchanged data makes every scheduled run log a vacuous "PROMOTE". Log format is mixed JSON/free-text, so a parser silently misses half the entries.
7. **Shadow models one step from serving.** `signal_desk_dealwin` has **no `default.txt`**, so the registry silently resolves it to `v0.2-dealwin-leakfree` — a model whose own manifest says `graduates: false` with *negative* AUC lift — if anything ever loads it. `signal_desk_recency` v0.1 IS pinned as default despite test AUC **0.505, CI [0.464, 0.545]** (statistically random), served uncalibrated. Neither is wired to an endpoint today, but nothing prevents it, and `/models` already exposes dealwin.
8. **Panel/label hygiene gaps.** 3,026 post-conversion snapshots (175 prospects) remain in the panel as negatives — converted customers keep being ranked as prospects with no explicit repeat-deal policy. The amendment bridge passes unfiltered garbage dates into labeling (26 conversions dated pre-2000, 7 dated post-2026). `data.training_data()` still joins the **known-broken** `ProspectLabelSnapshot` with a nondeterministic `TOP 8000` (no ORDER BY) — only legacy stand-in code uses it, but it's a loaded trap.
9. **Scorecard data gaps.** `months_until_expiration` is NULL for **5,018 of 5,019** companies — the "lease proximity" lens and `sort=lease` sort a dead column (the multifamily side's point-in-time lease machinery could fill this). `ExpectedValue` NULL for 37%. `WinProb` is a formula (cohort base × concept lift), not the (failed) ML model — fine, but the naming invites confusion. |
10. **Stale documentation.** `CONTEXT_BROKERAGE.md`/`CONTRACTS.md` still describe the conversion model as "planned, data-blocked, ~78 positives, need ≥500" — in reality v0.3 is served and the ≥500 bar was formally withdrawn (Riley basis). An auditor or new teammate reading the wrong doc gets the wrong reality. Zero brokerage tests exist in the golden suite.
11. **Structural data limits (the real ceiling).** Only **54 converting prospects** in the feature window; feature history starts 2024-11 while most real conversions cluster 2020–2023 (~100–380 stranded positives with no contemporaneous features); ZI↔Yardi bridge links only ~46% of prospects. No amount of modeling fixes these — they are data-engineering work.

## Part 3 — Improvement roadmap

(S) < half a day · (M) 1–3 days · (L) 1–2 weeks+ / owner-dependent.

### Phase 0 — Stop the bleeding (do first)

| # | Item | Effort |
|---|---|---|
| 0.1 | **Fix the scorecard 500**: tiebreak `ORDER BY {col} {dir}, c.CompanyID` (skip duplicate when col is `c.Score`). Add a golden test that the *default* call returns 200. | S |
| 0.2 | **Staleness surfacing + escalation**: add ProspectFeatureSnapshot/ProspectScore freshness to `/health` and a `data_stale` flag + banner text in score_company responses; escalate the dead nightly job (feature snapshots + rules score frozen since ~2026-04-21/30) to the Signal Desk pipeline owner — CompanyScoreSnapshot proves the job is only partially running. | S + owner |
| 0.3 | **Make tiers real**: derive cutoffs from the served-score distribution (e.g., top decile = "hot", next two = "warm"), rename to `opportunity_tier` for brokerage semantics. | S |
| 0.4 | **Registry/pointer hygiene**: pin explicit `default.txt` for dealwin (or make the registry refuse to implicitly serve `status: shadow` bundles); append the missing 7/07 reversal entry to `retrain.log`; normalize the log to JSON-lines for both contracts. | S |
| 0.5 | **Brokerage golden tests**: default-sort scorecard 200, tier sanity vs score distribution, staleness flag fires on old `data_as_of`, all four `/brokerage/*` endpoints return 200. | S |

### Phase 1 — Honest evaluation and reporting

| # | Item | Effort |
|---|---|---|
| 1.1 | **Leak-corrected eval protocol** for the conversion model: feature selection inside each fold; event-level dedup (or prospect blackout) so one conversion can't label both train and test; report grouped-CV PR-AUC (~0.087-style) as the headline; regenerate manifest metrics. Note in the manifest that identity-honest *temporal* eval is infeasible until the panel deepens. | M |
| 1.2 | **Split the score in two** (the docs' own v0.5 plan): a *pipeline-momentum* score (stage features — trivially strong, honest about being an echo) and a *cold-fit* score (firmographic/engagement only — honest PR-AUC ~0.03, but the only part useful for cold prospecting). Kills the days-in-stage masquerade (ρ = −0.93). | M |
| 1.3 | **Label hygiene**: filter amendment dates to a sane range; define an explicit post-conversion policy (censor after first conversion, or model repeat/renewal deals separately); delete or fix `training_data()` (broken-label join + nondeterministic TOP). | S–M |
| 1.4 | **Gate upgrade**: relative (not absolute) regression tolerance; add a calibration gate like multifamily's; hash the training data and log NOOP instead of a vacuous PROMOTE when nothing changed. | S |

### Phase 2 — Data (the real unlock; mostly pipeline-owner work)

| # | Item | Effort |
|---|---|---|
| 2.1 | Restart + harden the nightly `rea.*` snapshot job; add a freshness monitor with alerting (mirror multifamily's calibration-drift log pattern). | owner + S |
| 2.2 | **Backfill `ProspectFeatureSnapshot` to 2020–2023** — un-strands the ~100–380 historical conversions and is the single gate between "interim" and a production-grade conversion model. Forward-accumulation is the fallback if backfill is impossible. | L / owner |
| 2.3 | ZI↔Yardi bridge coverage 46% → review the ~1,010 unmatched rows (highest-leverage data lever per the validation doc). | M |
| 2.4 | Fill the scorecard's dead columns: `months_until_expiration` from Yardi lease-ends (the multifamily PIT lease machinery already computes exactly this), `ExpectedValue` basis for the 37% NULLs. | M |

### Phase 3 — Product (after Phases 0–2)

| # | Item | Effort |
|---|---|---|
| 3.1 | **Graduate space-demand** from prototype: verify the self-referential scoop label is strictly point-in-time, add size segmentation (it tilts large), dedupe leads already in pipeline, fuse lease-expiration timing. It's the one brokerage model with headroom. | M |
| 3.2 | **Lead-outcome feedback loop**: log which scorecard/space-demand leads brokers actually pursue (reuse the multifamily outreach-tracking pattern) → enables uplift measurement instead of pure ranking. | M |
| 3.3 | **Retire the failed families explicitly**: tombstone dealwin (no viable signal on safe features at current deal volume — twice measured) and recency (target endogenous to RTS's own outreach; AUC ≈ 0.5) so they can't be accidentally served. Revisit dealwin only when resolved-deal volume grows materially. | S |

### Explicitly not recommended

- **More model tuning on the conversion label** — 54 positives is a data bound, not a modeling one; every additional point of reported PR-AUC from tuning is more likely eval noise than signal (per-fold PR-AUC already ranges 0.07–0.25).
- **Re-adding scoops to the conversion model** — measured twice (deal-grain and company-grain), CI spans zero both times. Scoops earn their keep in space-demand and marketing, not conversion.
- **Building an ML WinProb now** — dealwin failed honestly; the served formula (cohort × concept lift) is what the data supports until deal volume grows.
- **Deep learning / embeddings** — same answer as multifamily: the constraints are labels and data depth, not model capacity.

---

*Audit only — no code, models, bundles, or configuration were changed. Evidence: leakage-corrected re-runs of the deployed trainer's own evaluation (reproduced 0.107 vs manifest 0.1082 before correction), live-DB checks (2026-07-16), and live-endpoint probes. The scorecard 500, the April data freeze, and the 100%-"low" tier mix are all reproducible right now against the running service.*

---

## Implementation status — 2026-07-16 (same day)

All in-repo roadmap items implemented and deployed; owner/DB-write items handed off. The
brokerage default is now **`v0.5-split-honest`**. 12/12 golden tests pass (5 new brokerage).

**Phase 0 (done):** scorecard default-sort 500 fixed (tiebreak on CompanyID); `/health` +
`score_company` now surface `brokerage_data`/`data_stale` (feature pipeline frozen 77 days —
escalate to the pipeline owner); opportunity tiers are distribution-derived (`opportunity_tier`
hot/warm/cold, high = pursue); registry now refuses to implicitly serve shadow/tombstoned
bundles; the missing 7/07 default.txt reversal was backfilled and the log normalized to
JSON-lines; brokerage golden tests added.

**Phase 1 (done):** new `v0.5-split-honest` two-headed model — `pipeline_momentum`
(honest PR-AUC 0.119, the deal-in-motion echo) and `cold_fit` (0.027, the genuine cold
signal, lift CI excludes 0) and `combined` (0.083, ~20% below the old inflated 0.108).
Evaluation is nested (selection in-fold) and group-disjoint by prospect. Label hygiene:
date-filtered corrected bridge + post-conversion negatives censored (259 pos / 54 prospects
preserved, ~3k polluting negatives dropped). `training_data()` no longer joins the broken
label. Promotion gate is now relative-tolerance with a NOOP-identical guard.

**Phase 2 (partly done; rest owner-blocked):** `months_until_expiration` is filled at serve
time from Yardi amendments (40+/200 in a scorecard page, was ~1/5019). Bridge-coverage
analysis script emits an exact-name review queue (`scripts/analyze_brokerage_bridge.py`).
**OWNER-BLOCKED (cannot do from a read-only seam):** restarting the frozen nightly `rea.*`
job (2.1) and backfilling `ProspectFeatureSnapshot` to 2020–2023 (2.2) — the latter is the
single gate to a production-grade conversion model; the freshness monitor now makes the
freeze visible so it can't rot silently.

**Phase 3 (done):** space-demand leads now carry `emp_band` + `size_band_mix` (confirms the
large-employer tilt — top leads are all 1000+ employees) and `in_pipeline` dedup; its label
was verified point-in-time clean. Lead-outcome feedback loop shipped
(`POST/GET /brokerage/leads`, JSONL store, snapshots rank signals for uplift). dealwin +
recency tombstoned (manifests + registry guard) so they can never be accidentally served,
while `/models` still shows their cards.

Rollback: `bundles/signal_desk_brokerage/default.txt` -> `v0.3-corrected-logistic`.

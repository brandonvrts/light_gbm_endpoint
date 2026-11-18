# Multifamily Churn Model & API — Full Audit and Improvement Roadmap

**Date:** 2026-07-15 · **Model:** `v5-person-tenure` (deployed 2026-07-14) · **Scope:** model quality, data integrity, serving/API, operations. Audit only — nothing in this document has been implemented.

Numbers below come from a fresh evaluation of the deployed bundle on a 9,211-row person-grouped holdout (20% of the corrected point-in-time training anchors), plus live-system checks. Caveat: the sigmoid calibrator was fit on an overlapping holdout, so single-split metrics read slightly optimistic — the honest headline is the walk-forward number.

---

## Part 1 — What's working (positives)

| Area | Evidence |
|---|---|
| **Discrimination** | Walk-forward AUC **0.843 ± 0.024** across 12 quarterly folds (stable, no regime luck). Holdout AUC 0.898, PR-AUC 0.706 at a 6.5% base rate. |
| **The high tier means something** | "High" flags 6% of resident-months and **67% of them actually churn** — a 10× lift over base. It captures 61.6% of all churners. |
| **Cohort-level honesty** | On the expiring-soon population (what the monthly report scores): predicted 21.1% vs realized 21.9%; recent anchors 23.0% vs 23.0%. The v4 ~10-point overshoot is gone at the source, not patched. |
| **No property blind spots** | Per-property AUC 0.86–0.91 (Winwood 0.911, CCV 0.863, Sierra Pointe 0.914, R62W 0.911). Townhomes (473) now trained in-distribution. |
| **Stable across time** | Predicted vs realized by anchor year tracks within ~1pt every year 2022–2026 — the fake base-rate "decline" from the corrupted extraction is gone. |
| **Tenure behaves correctly** | Monotone-decreasing by construction *and* calibrated by bucket: realized churn falls 7.7% (<1y) → 1.3% (12y+), predictions fall with it. The business prior is now enforced and empirically verified. |
| **Smooth, explainable scores** | No isotonic plateaus or display caps; every score ships with SHAP-derived `why`/`drivers`, so consumers (chatbot, staff) see real reasons. |
| **Cohort truth** | Lease-expiration cohorts rebuilt from immutable `Lease_History`: 78/79 of the April ground-truth list recovered, plus genuine expirations the manual Yardi export itself missed. |
| **Ops hygiene** | Versioned bundles with one-line rollback (`default.txt`), promotion audit log, warm cache (~2ms chatbot answers for current months), request logging. |

## Part 2 — What's weak or wrong (negatives)

Ranked roughly by how much they matter.

1. **Mid-range under-prediction.** Deciles 5–8 predict 0.9–2.5% vs realized 1.5–3.4% (~1.5–2× low). The expiring cohort is calibrated (the intercept targeted it), but portfolio-wide expected-churn totals will *undercount* from the big low-risk mass. Single global sigmoid is too stiff for both ends.
2. **Bottom clumping at the floor.** 72% of holdout scores sit at the 0.02 floor — we removed the 47%/75% snaps but created a 2% snap. Fine for ranking a cohort's top; useless for distinguishing "very safe" from "safe."
3. **~6.3% of churn labels are unit transfers.** A resident moving from unit A to unit B at the same property counts as a "move-out" in training labels *and* as "churned" in report outcomes. The model is partly learning to predict internal moves; retention stats are slightly overstated as losses.
4. **Silent date-parsing corruption (latent bug).** The extraction CSVs contain mixed date formats; `pd.to_datetime(errors="coerce")` infers a format from early rows and silently NaT's the rest (this corrupted the first pass of this very audit — 2025+ rows lost `DaysToLeaseEnd`). Training used in-memory frames and was unaffected, but the `--reuse-csv` retrain path and any future CSV consumer will hit it. One-line fix (`format="mixed"`), high silent-damage potential.
5. **Serving features are anchored to the wrong date.** `sp_GetTenantFeaturesByScode` anchors work-order/memo windows at `COALESCE(DTMOVEOUT, DTLEASETO, asof)` — i.e., lease end, not the as-of date. Two consequences: (a) mild feature shift for residents far from lease end; (b) **scoring past months leaks outcomes** for decided residents (their features are computed at move-out, notice on file), which forbids honest backtests through the serve path and once produced a degenerate calibration fit (caught, removed).
6. **Medium tier is mushy.** Realized churn: low 1.9% → medium 9.7% → high 67%. Medium is only ~1.5× base — it reads more like "not low" than an action tier. Cutoffs (0.05/0.27) were derived from cumulative-rate targets, not from outreach capacity.
7. **Training coverage gap.** The corrected extraction requires a `Lease_History` row (INNER JOIN) — ~25% of active residents at any anchor have none and are excluded from training (they still get scored at serve). Sample skews toward lease-history-covered residents.
8. **Calibration intercept rests on n=378.** The recent expiring-soon subpopulation used to set the level is small; a few noisy months could move it ±3–4 pts. No drift monitor exists to catch that.
9. **Name-based person chaining.** Transfer chaining joins on (name, property, ≤31-day gap). Name changes (marriage), shared names, or >31-day gaps break lineage silently. It fixed ~127 residents but is a heuristic, not identity.
10. **Point-in-time gaps in small features.** `NSFCount`/`BNOPAYMENTS` are current values applied to historical anchors; rescinded notices are unobservable (`DTNOTICE` cleared in place); memo categories are crude `LIKE` matches (e.g., `%notice%` counts benign memos as "eviction").
11. **Ops/process gaps.** The retrain pipeline's gated-promotion flow (`scripts/retrain_pipeline.py`) still has no multifamily entry — the v5 trainer exists but retraining is manual. Tests are stubs. The report cache is in-process (lost on every restart; ~5 cold minutes). No predicted-vs-realized monitoring loop.
12. **Two "churn rate" definitions coexist.** Model rates mean "moves out within ~60 days of month start"; the report's `churn_rate` means "churned share of decided outcomes" (reads higher because holdover pendings are excluded). Documented, but easy to misquote — the chatbot could state both without knowing they differ.

## Part 3 — Improvement roadmap

Ordered for implementation; each item independent unless noted. (S) < half a day, (M) 1–3 days, (L) 1–2 weeks.

### Phase 0 — Safety and hygiene (do first)

| # | Item | Effort | Pays for |
|---|---|---|---|
| 0.1 | Robust date parsing (`format="mixed"`) in `apply_semantic_fixes` + trainer CSV reload; add an assertion that no date column is all-NaT per year | S | Kills the silent-corruption class of bugs (negative #4) |
| 0.2 | Wire `multifamily_churn` into `retrain_pipeline.py` `CONTRACTS` — trainer exists; gate on walk-forward AUC (tol 0.01) **and** expiring-cohort calibration error (< 3 pts); schedule via the existing systemd timer | S | Hands-off monthly retrains with automatic rollback (negative #11) |
| 0.3 | Persist the monthly-report cache to disk (or warm immediately at boot, no 20s + recompute) | S | No cold chatbot answers after restarts |
| 0.4 | Golden tests: Kathleen-style pinned-score snapshot, April-2026 cohort reconstruction vs the 79-code list, tier-threshold sanity | M | Regression safety for every future change |

### Phase 1 — Label and calibration quality (next retrain cycle)

| # | Item | Effort | Pays for |
|---|---|---|---|
| 1.1 | **Transfer-aware labels**: a move-out followed by a chained record ≠ churn (person-lineage already computed). Add a `transferred` outcome to `renewal_outcomes` so reports stop counting internal moves as losses | M | Removes ~6% label noise; truthful retention stats (negative #3) |
| 1.2 | **Two-segment / spline monotone calibration** instead of one global sigmoid; lower floor to ~0.005 | M | Fixes mid-range under-prediction + bottom clumping (negatives #1, #2) |
| 1.3 | **Capacity-based tiers**: set "high" so it yields roughly the monthly outreach list size the property team can work; publish the recall that buys; consider renaming medium to "watch" | S | Tiers become operational, not statistical (negative #6) |
| 1.4 | **True out-of-time holdout**: reserve the last 6 months of anchors entirely (no group shuffle) for reported metrics | S | Removes audit optimism; honest headline numbers |
| 1.5 | Calibration drift monitor: when a month closes, auto-compare its predicted vs realized rate (the report already computes both) and log it; alert past ±5 pts; feed the intercept refit | M | Catches drift instead of discovering it a year later (negative #8) |

### Phase 2 — Structural data fixes (bigger, most leverage)

| # | Item | Effort | Pays for |
|---|---|---|---|
| 2.1 | **Set-based, as-of-anchored serving SQL**: replace per-resident `sp_GetTenantFeaturesByScode` calls for cohort scoring with the corrected point-in-time query already written in `scripts/train_multifamily_v5.py`, parameterized at as-of. One query per cohort | L | Kills lease-end window skew *and* past-month leakage → honest backtests; cohort scoring drops from minutes to seconds, warm loop becomes optional (negative #5) |
| 2.2 | Coverage: LEFT JOIN residents without lease history into training with explicit "no lease record" indicators | M | Un-skews the training sample (negative #7) |
| 2.3 | Investigate a real Yardi person key (roommate/occupant tables) to replace name-based chaining; keep the heuristic as fallback | M | Robust identity (negative #9) |
| 2.4 | Memo classification upgrade: replace `LIKE` buckets with a curated keyword set or a small LLM-assisted offline labeling pass over distinct memo texts | M | Cleaner behavioral features (negative #10) |

### Phase 3 — Product value (after the above)

| # | Item | Effort | Pays for |
|---|---|---|---|
| 3.1 | **Revenue-at-risk**: expected churns × current rent per month in the report and trend ("~$31k of September rent at risk") | S | Leadership-legible output |
| 3.2 | Unify the two churn-rate definitions in API responses (label them explicitly; give the chatbot one canonical field to quote) | S | Prevents misquoted stats (negative #12) |
| 3.3 | Outreach tracking: record which high-risk residents were contacted and outcomes → enables uplift ("does outreach change renewals?") rather than pure risk ranking | L | Turns predictions into measured savings |
| 3.4 | Longer-horizon variant (h180) retrained on the corrected pipeline for budgeting/occupancy forecasts | M | Planning use-case beyond monthly ops |

### Explicitly not recommended

- **Embeddings/vector retrieval** — the data is structured and exactly keyed; latency was feature computation, addressed by 2.1.
- **Deep-learning replacement** — at ~46k samples with strong tabular features, gradient boosting is the right tool; effort is better spent on labels and calibration (Phase 1) where the remaining error actually lives.
- **More features before Phase 1/2** — label noise and the serving-SQL skew dominate the current error budget; new features on top of them would inherit the same problems.

---

*Prepared as an audit only — no code, model, or configuration was changed. The evaluation script used is reproducible at the scratchpad path in the session, and all claims about live behavior were verified against the running service on 2026-07-15.*

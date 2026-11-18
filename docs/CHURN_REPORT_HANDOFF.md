# Monthly Churn Report — Team Handoff

**Service:** `light_gbm_endpoint` — shared RTS prediction API on the DGX Spark.
**What's new:** the lease-expiration cohort is now historically accurate, and there are two new endpoints that turn it into a month-over-month churn report.
**Verified live:** 2026-07-13, against the April/May 2026 manual renewal lists.
**Updated 2026-07-14:** model **v5-person-tenure** deployed — see "Model v5" at the bottom: honest smooth probabilities (no more scores snapped to 47%/75%), per-resident `why` explanations, corrected tenure.

## TL;DR — what was wrong and what changed

Asking the API for "leases expiring in April 2026" used to return **5 residents**. The
manual Yardi export for the same month had **79**. The API wasn't undercounting risk —
it was reading the wrong table.

**Root cause:** the cohort was built from `TENANT.DTLEASETO`. Yardi **overwrites that
column in place** the moment a renewal is signed (and sets `DTMOVEOUT` when someone
leaves). So for any month in the past, everyone who renewed or moved out had already
disappeared from the window — the query could only ever see the not-yet-decided stragglers.

**Fix:** the cohort is now reconstructed from `Yardi7_Reporting.dbo.Lease_History`, which
keeps every lease term as an immutable row. The resident's *active lease as of the month
start* (latest real `Past`/`Current` row — `Denied`/`Approved` rows are unaccepted renewal
proposals) determines which month they belong to. Renewing no longer erases you from history.

Also fixed along the way:

- **Property scope** now includes HPROPERTY **473 = Country Club Village Townhomes**
  (~40 active residents). It was in the manual Yardi exports but not in the API's
  hardcoded 4-property list. (Model v5 trains on all 5 properties, so Townhomes residents
  are fully in-distribution.)
- **Early terminations count as churn.** Someone who moved out in March against an April
  lease-end is April churn. Move-outs up to 90 days before the month are kept; anything
  older is a mid-lease break and excluded from that month's cohort.

**Validation:** for April 2026 the rebuilt cohort captures **78 of the 79** residents on
the manual list (the one miss has no `Lease_History` row for its expiring term — not
reconstructable from data). It also finds ~15 *additional* genuine April expirations the
manual export missed — mostly residents who renewed before the export was pulled, plus the
Townhomes. Same in-place-update blind spot, just in the Yardi report instead of our API.

## New endpoints

Auth and base URL are the same as everything else (see `API_HANDOFF.md`):
`http://10.1.6.17:8910` with header `X-API-Key: rts-local-dev-key`.

### 1) `GET /multifamily/churn-report?month=YYYY-MM`

Full report for one calendar month: every resident whose lease expired/expires that
month, what actually happened (as known today), and the model's churn score *as of the
month start* so you can compare prediction vs. outcome.

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/churn-report?month=2026-04"
```

Response (top level):

| Field | Meaning |
|---|---|
| `cohort_size` | residents whose lease expired/expires in the month |
| `outcomes` | `{renewed, churned, pending}` counts |
| `churn_rate` / `renewal_rate` | share of *decided* outcomes (pending excluded) |
| `model_flagged_churners` | of the residents who churned, the fraction the model had rated medium/high risk |
| `results[]` | per-resident rows, sorted highest churn probability first |

Each row in `results[]`: `scode`, `full_name`, `unit`, `property`, `lease_to`,
`outcome` (`renewed` / `churned` / `pending`), `churn_probability`, `risk_tier`,
`has_notice`, `days_to_lease_end`.

Outcome rules (same logic as the manual April/May validation workflow):
- **churned** — `TENANT.DTMOVEOUT` is set
- **renewed** — `TENANT.DTLEASETO` now extends past the month, *or* a signed
  `Scheduled`/`Current` lease row exists in `Lease_History`
- **pending** — none of the above (offer outstanding, or month-to-month holdover)

Options: `&properties=62,72` limits to specific HPROPERTY ids (default all 5:
62 CCV, 72 Sierra Pointe, 82 Residences 62W, 26 Winwood, 473 CCV Townhomes).
`&include_scores=false` skips model scoring — returns instantly, outcomes only.

**Predicted churn rate** — scored reports also carry a `predicted` block driven by the
model's calibrated probabilities (each resident's `model_probability`, which is NOT the
display-capped `churn_probability` — the capped one would understate the tail):

| Field | Meaning |
|---|---|
| `predicted.expected_churns` / `predicted_churn_rate` | pure model view: sum/mean of calibrated probabilities over the whole cohort, features as of month start |
| `predicted.projected_churns` / `projected_churn_rate` | best estimate: decided outcomes taken as fact + model probabilities for the still-pending residents |

**Which one to quote:** `projected_churn_rate` (decided outcomes are facts; the model
fills in only the undecided). Historical note: the v4 model over-predicted the cohort
rate by ~10 points; **model v5 (see the section at the bottom) fixed this at the source**
— on held-out data it predicts 23.2% vs 23.0% realized, so the two rates now broadly
agree, with `projected_*` still preferred once outcomes start landing.

**Top-N / chatbot mode** — `results[]` is ranked highest churn risk first, so:

```bash
# "Who is going to churn in September 2026?" -> top 10, ready to read back
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/churn-report?month=2026-09&top=10"
```

- `top=10` trims `results` to the 10 highest-risk residents (`results_total` /
  `results_shown` tell you what was trimmed; all month-level stats still cover the full
  cohort).
- `outcome=pending` first filters to a decision status — `pending` gives the actionable
  retention call list (undecided residents only), `churned` gives the known departures.
  Combine: `?month=2026-09&outcome=pending&top=10`.
- Every response carries a **`summary`** field — one natural-language paragraph a chatbot
  can quote verbatim, e.g.: *"Of the 56 residents whose leases expire in September 2026,
  2 have renewed, 5 have a move-out or notice on file, and 49 are still undecided.
  Projected churn rate: 34% (~19 of 56 expected to leave). Highest-risk undecided
  residents: …"* This pairs with the existing single-tenant chatbot endpoints
  (`/multifamily/tenant/{scode}` and `/multifamily/tenant/search?q=name`), which return a
  per-resident `summary` the same way.

Chatbot wiring: give the bot one HTTP tool — `GET /multifamily/churn-report` with
`month` (map "September 2026" → `2026-09`), optional `top` and `outcome` — plus the
`X-API-Key` header. It can answer with `summary` directly, or format `results[]`
(`full_name`, `unit`, `property`, `lease_to`, `outcome`, `churn_probability`,
`risk_tier`) into a table. First uncached call for a month takes ~30–60 s (then 1 h
cache), so set the tool's timeout accordingly.

### 2) `GET /multifamily/churn-trend?start=YYYY-MM&end=YYYY-MM`

The month-over-month series — one summary per month, no per-resident scoring, fast
enough to chart directly. Max 36 months per call.

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/churn-trend?start=2026-01&end=2026-08"
```

Returns `months[]` of `{month, cohort_size, outcomes, churn_rate, renewal_rate}`.
Add `&include_predicted=true` to also get `predicted_churn_rate` and
`projected_churn_rate` per month — that scores every cohort, so the first call takes
~1 min per uncached month (cached 1 h afterwards) and the span is capped at 12 months.

Live sample (pulled 2026-07-13):

| Month | Cohort | Renewed | Churned | Pending | Churn rate |
|---|---|---|---|---|---|
| 2026-03 | 93 | 61 | 28 | 4 | 31.5% |
| 2026-04 | 93 | 64 | 27 | 2 | 29.7% |
| 2026-05 | 84 | 53 | 25 | 6 | 32.1% |
| 2026-06 | 124 | 73 | 37 | 14 | 33.6% |
| 2026-07 | 109 | 60 | 29 | 20 | 32.6% |
| 2026-08 | 75 | 24 | 14 | 37 | 36.8%* |

\* churn/renewal rates only count decided outcomes, so recent/future months with many
`pending` will move as decisions land.

### Also fixed: the existing lease-window endpoints

`GET /multifamily/lease-window?start=&end=` and the `expires_from`/`expires_to` mode of
`POST /predict/multifamily-churn` use the same rebuilt cohort, so their counts now match
reality for past *and* future windows (April 2026: 5 → 93).

## Things to know

- **The rolling window (previous month → +3) is kept permanently warm.** A background
  loop in the service recomputes those 5 monthly reports every ~50 minutes, so chatbot/UI
  calls for them return in well under a second. Only months *outside* that window pay the
  compute cost on first call (~30–60 s — the feature stored procedure runs per resident;
  cached for 1 h afterwards). `include_scores=false` and `churn-trend` are always fast.
- **Why it's slow uncached (for the curious):** it is not the model — LightGBM scores a
  cohort in milliseconds — it's `sp_GetTenantFeaturesByScode` computing an 18-month
  feature lookback per resident in SQL Server (~0.3 s each). The durable fix, if we ever
  need arbitrary months fast, is a set-based version of that SP (score the whole
  portfolio in one pass, like `sp_GetRenewalWindows` does for training) or a nightly
  scored table; the warm loop makes that unnecessary for the practical use cases.
- **Scores for a past month are point-in-time**: computed with features as of the month
  start, so "would the model have caught it" comparisons are honest.
- **`pending` for an old month usually means month-to-month holdover** (no new lease, no
  move-out on file).
- **CCV Townhomes (473)** residents score via an encoding fallback until the model is
  retrained with the property included — treat their probabilities as approximate.
- Interactive docs with all parameters: `http://10.1.6.17:8910/docs`.

---

## Model v5 (`v5-person-tenure`) — deployed 2026-07-14

The v4 model had three train/serve defects that made individual scores untrustworthy
(the visible symptom: a 14-year resident with no notice reading 75%, and scores snapping
to values like 47%/75%). All three are fixed at the root in v5:

1. **Tenure was broken twice.** Training defined "tenure" as days into the *current
   lease* (0–18 months); serving fed days since the *record's* move-in (years, far out of
   training range); and Yardi creates a new tenant record on every unit transfer, which
   truncated real residency (~127 current residents affected). v5 trains and serves on
   **person-level residency** (records chained across transfers). On honest data, tenure
   shows the relationship the historical data always implied — churn falls from ~7.9%
   (<1y) to ~2.0% (12y+) — and the model carries a **monotone constraint: longer tenure
   can never raise predicted churn**. Tenure importance moved from rank 30 to rank 10.
2. **The training snapshots themselves were corrupted** by Yardi's in-place lease-date
   updates: historical anchors silently *excluded* residents who later renewed and
   mislabeled lease timing for the rest (this also faked a base-rate "decline" from 8% to
   2%). v5 trains on corrected point-in-time snapshots rebuilt from `Lease_History`
   (read-only SQL — no DB changes).
3. **Snapped scores.** v4's isotonic calibration is piecewise-constant (scores pile on
   plateau values) and the serve layer capped display at 75%. v5 uses smooth sigmoid
   calibration plus a cohort-level intercept and serves probabilities **as-is** in
   [0.02, 0.95]. A September cohort now shows ~31 distinct values instead of 2–5.

**Quality:** walk-forward AUC 0.843 ± 0.024 (12 quarterly folds); on the held-out
recent expiring-soon population the model predicts 23.2% churn vs 23.0% realized —
the old ~10-point aggregate overshoot is gone at the source.

**New response fields** (churn-report rows, `/multifamily/tenant/{scode}`, tenant search):

| Field | Meaning |
|---|---|
| `why` | one human-readable sentence: top score drivers and offsets (e.g. *"Driven by a move-out notice is on file, notice given 67 days ago; offset by long tenure (7 years)"*). Chatbots should quote this instead of guessing. |
| `drivers` | the structured version: `[{feature, direction, weight, value}]` ranked by influence |
| `tenure_months` | corrected person-level residency (transfer-chained) |

**Reading the scores now:** probabilities are honest — a resident with a notice on file
reads 65–75%, a long-tenured no-notice resident can read 2%. Risk tiers changed with the
calibration: `low` < 0.06, `medium` < 0.27, `high` ≥ 0.27. Expect *fewer* residents in
"high" than before — that's the calibration being truthful, not the model going soft.
Predicted/projected rates are calibrated to "moves out within ~60 days of the month
start"; the decided-only churn rate for a month will read higher because holdover
pendings are excluded from its denominator.

**Rollback:** point `bundles/multifamily_churn/default.txt` back to `v4-recal` and
restart the service. Retrain: `PYTHONPATH=. .venv/bin/python scripts/train_multifamily_v5.py --promote`.

**Still open (SQL work item):** the serving feature SP anchors work-order/memo windows to
lease-end rather than the as-of date, and `sp_GetAnchorSnapshot` should be replaced with
the corrected point-in-time SQL now embedded in `scripts/train_multifamily_v5.py`.

---

## Model v7 (`v7-occupancy-h60` serving, `v7-occupancy-h180` renewal pin) — deployed 2026-09-25

**What changed: the population, not the features.** A 2026-09-25 audit found that `TENANT`
rows with `ISTATUS 9` (Application Denied) and `7` (Application Canceled) carry a planned
`DTMOVEIN` and never a `DTMOVEOUT`, so every `DTMOVEIN`/`DTMOVEOUT` eligibility filter admitted
them as residents. They were ~50% of every v6 training anchor (959 of 1,901 rows at
2024-06-30), 100% of the `HasLeaseRecord = 0` rows v6 chose to keep, and 35 of the 950
scodes the service scored (8 of them displacing the unit's real resident). v7 trains and
serves on occupancy statuses only (`data.OCC_SQL = "ISTATUS IN (0,1,2,3,4)"`); features,
labels, calibration method and tier policy are unchanged. Same trainer
(`scripts/train_multifamily_v6.py`, `GBME_BUNDLE_VERSION=v7-occupancy-h{60,180}`).

**Headline numbers moved because the phantoms were trivially separable never-churners.**
v6's walk-forward AUC was inflated by ranking denied applicants below real residents; the
honest v7 number is lower on paper and better in fact. The fair test is both bundles on the
same real-resident rows in months neither base model trained on:

| | v6 | v7 |
|---|---|---|
| **h60**, shared unseen 2026-01..04, expiring-soon (n=690, realized 17.4%) | AUC 0.874 · PR-AUC 0.727 · Brier 0.0754 | **AUC 0.895 · PR-AUC 0.754 · Brier 0.0733** |
| h60, same window, all real residents (n=3,655, realized 5.4%) | AUC 0.914 · Brier 0.0287 | **AUC 0.922 · Brier 0.0281** |
| **h180**, shared unseen 2025-09..12, expiring-soon (n=235, realized 26.4%) | AUC 0.828 · PR-AUC 0.763 · Brier 0.1140 | **AUC 0.830 · PR-AUC 0.765 · Brier 0.1117** |
| h180, same window, all real residents (n=3,711, realized 13.2%) | AUC 0.793 · Brier 0.0920 | **AUC 0.802 · Brier 0.0913** |
| h60 walk-forward AUC (13 folds) | 0.905 ± 0.026 *(phantom-inflated)* | 0.854 ± 0.025 |
| h180 walk-forward AUC | 0.822 ± 0.061 *(phantom-inflated)* | 0.718 ± 0.040 |
| h60 OOT expiring cohort: predicted vs realized | 20.5% vs 18.3% (AUC 0.863) | 22.5% vs 19.6% (AUC 0.891) |
| h180 OOT expiring cohort: predicted vs realized | 26.9% vs 27.5% (AUC 0.859) | 25.9% vs 25.7% (AUC 0.808) |
| h60 high tier (top ~22% of an expiring cohort) | 67% recall @ 56% realized churn | **70.5% recall @ 63% realized churn** |
| Training base rate / samples | 3.1% / 98,008 (h60) · 9.2% / 90,079 (h180) | 6.2% / 50,698 · 18.2% / 47,036 |
| `HasLeaseRecord = 1` share | ~75% | 99.7% |

**Serving population:** 923 scodes (`ISTATUS` 0: 901, 2: 1, 3: 1, 4: 20) — was 950 with 35
application records.

**Things to know**
- Anyone quoting v6's 0.905 / 0.822 walk-forward AUCs should stop; the honest comparables are the shared-window rows above.
- The h180 calibrator has visible plateaus (many residents share p ≈ 0.085–0.087 and ≈ 0.30) — isotonic flattening on a smaller training set. Ranking inside a plateau is a tie; economics that use `p` as a level are unaffected. Follow-up: knot count / bin size in `calibration.SplineCalibrator`.
- The turnover ledger (`/multifamily/unit-economics`) got the same filter plus `as_of`; portfolio annual turn cost rose from $908k to $949k (CCV +$17k) because application records had been truncating vacancy gaps.
- `monitoring/calibration.log` continues on the serving default, now v7 h60; the first v7 row lands 2026-10-01.

## Model v6 (`v6-roadmap-h60`) — deployed 2026-07-15

v6 implements the full audit roadmap (Phases 0–3). It is a strict upgrade over
v5-person-tenure; the API shapes are backward-compatible (new fields added, none removed).

**Model quality (out-of-time, honest):** the discrimination number to cite is the
**out-of-time expiring-cohort AUC 0.863** — measured on the reserved last-6-months
holdout, over the lease-holders we actually serve. On that same holdout, predicted churn
20.5% vs realized 18.3% (calibration checked out-of-sample, not on the fitting data).
Scores are smooth (≈600 distinct values in a 56-resident cohort; no snapping). Kathleen
Gray still reads ~1%.

  *(The full-population walk-forward AUC of 0.905 is reported in the manifest but should
  NOT be quoted as decision-level accuracy: ~half the active population has no
  lease-history row and essentially never churns, so that number partly measures a
  trivial "lease vs no-lease" split. The retrain pipeline gates on the honest 0.863-style
  expiring-cohort AUC for this reason.)*

**What changed under the hood:**
- **Transfer-aware outcomes** — an internal unit move is no longer counted as churn.
  Reports now show a fourth outcome, `transferred` (retained), and the churn rate math
  excludes it from the loss column.
- **Set-based point-in-time serving** — cohort scoring runs ONE as-of-anchored query
  (`data.pit_features_sql`, shared with training) instead of per-resident SP calls: same
  features at train and serve, no lease-end feature skew, and past months can be scored
  without leaking the outcome.
- **Coverage** — residents with no lease-history row (~half the active population) are now
  trained and scored with an explicit `HasLeaseRecord` flag instead of being dropped or
  silently median-filled (feature count 45 → 46).
- **Spline calibration** — replaces v5's single sigmoid; fixes the mid-range
  under-prediction and the floor clumping.
- **Capacity-based tiers** — "high" is sized to the monthly outreach list the team can
  actually work (~top fifth of a cohort; recall recorded in the bundle). Tier display
  labels: high = "Priority outreach", medium = "Watch", low = "Stable".
- **Curated memo categories** — dropped the bare `%notice%`/`%emergency%` matches that
  mislabeled benign memos.

**New / changed API fields (churn-report):**

| Field | Meaning |
|---|---|
| `rates` | one labeled block: `realized_churn_rate`, `predicted_churn_rate`, `projected_churn_rate`, a single canonical **`headline_churn_rate`**, and `definitions`. Quote `headline_churn_rate`. |
| `revenue_at_risk` | `expected_` / `projected_monthly_revenue_at_risk`, `cohort_monthly_rent` (USD). Also per-month in `churn-trend?include_predicted=true`. |
| `outcomes.transferred` | internal unit moves (retained, not churn) |
| `tier_labels` | display names for the tiers |
| row `monthly_rent`, `tenure_months`, `why`, `drivers` | per-resident |

**New endpoints:**
- `POST /multifamily/outreach` / `GET /multifamily/outreach[?summary=true|scode=|month=]` —
  log and review retention outreach (snapshots the score at contact time for later uplift).
- `GET /multifamily/calibration-drift` — logged predicted-vs-realized per settled month
  (drift monitor; alerts past ±5 pts).

**Operations:**
- Retrains are wired into `scripts/retrain_pipeline.py` (gated on walk-forward AUC **and**
  a ±6pt cohort-calibration band; auto-rollback on regression). Manual: `PYTHONPATH=.
  .venv/bin/python scripts/train_multifamily_v6.py --horizon 60 [--promote]`.
- The monthly-report cache now persists to `monitoring/report_cache/` (keyed by model
  version) so a restart doesn't cost a cold recompute.
- **h180 variant** (`v6-roadmap-h180`) is trained and available for 180-day
  budgeting/occupancy forecasts (not the default; predicted 26.9% vs realized 27.5% OOT).
- Golden regression tests: `PYTHONPATH=. .venv/bin/python tests/test_smoke.py`.
- Rollback: point `bundles/multifamily_churn/default.txt` back to `v5-person-tenure`.

**Known limits (next backlog):** ~half the active population has no lease-history row and
is scored via the `HasLeaseRecord` fallback path; a real Yardi person-key for transfers
does not exist (no stable key in TENANT — verified), so lineage is name+email+31-day-gap
heuristic; outreach uplift needs outcomes joined downstream once enough events accrue.

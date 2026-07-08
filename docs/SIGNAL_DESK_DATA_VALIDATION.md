# Signal Desk (brokerage) — data validation & why it can't be walk-forward validated yet

**Date:** 2026-06-23. All numbers are live from `InsightMart` (SSMS = source of truth). **No model was fabricated** — the data does not currently support a trustworthy conversion model.

## Verdict
Signal Desk **cannot be walk-forward validated on the current data.** Two independent blockers, both data (not modeling):

> **Update 2026-06-23 (details at end):** the prospect→signed-amendment linkage has since been **found** — `rea.vw_Deal.tenantId ↔ dbo.CommAmendmentSnapshot.hTenant` (96% of deal-tenants). So blocker #1 (mislinked label) is now *fixable*. The surviving blocker is **feature-history depth**: real signed conversions cluster in 2020-2023, but `ProspectFeatureSnapshot` only starts 2024-11, so in-window positives are still ~5 (by sign date) to ~45 (by lease-start) — short of ~500.

### 1. The conversion label is mislinked (broken)
`rea.ProspectLabelSnapshot` is not a usable label:
- Prospect `hMy=5` is `Label=1` at snapshots in 2023-2024, but its `AmendmentID` points to an amendment **signed 2020-01-14** — years before the 2023-26 snapshot window — with a fabricated `DaysToConvert=33` (the real gap is ~‑1,400 days).
- Every converting prospect has **all 30 of its monthly snapshots = `Label=1`** (`total_snaps=30, pos_snaps=30`), yet each has exactly **one** amendment. That can't be reconciled with "converts within 180 days of this snapshot" — so the label encodes *"is this one of the converters"* (company identity), not timing. This is why the interim stand-in scored a meaningless **AUC 0.993**.

### 2. Even with a CORRECT label, there are far too few positives
Rebuilt a correct, time-localized label read-only from `rea.vw_Deal` (the only source with a working prospect link, `commProspectId`): `Label=1` iff the prospect has a deal that reached **`Executed`** within `(snapshot, snapshot+180d]`:

| | count |
|---|---|
| Distinct converting prospects (correct label) | **13** |
| Positive snapshots | 41 |
| Needed for stable walk-forward | ~500 |
| `vw_Deal` executed events (all-time) | 224 across 35 prospects |
| `vw_Deal.stageBeganAt` date coverage | **only 2025-08 → 2026-06** (444 of 2,767 rows) |

13 distinct positives, all since mid-2025 → no quarter has enough positives to form a fold. Walk-forward is not possible.

## Why the linkage is hard (paths tried)
- `ProspectLabelSnapshot.AmendmentID` → wrong (2020 amendments).
- `CommProspect.hDeal` → `CommAmendmentSnapshot.hDeal` → **0 matches** (deal-id namespaces don't align).
- `rea.vw_Deal.commProspectId` → **works** (1,308/1,350 prospects match `ProspectFeatureSnapshot`), has `dealType` (New/Renew/Expansion/Relocation) + `stageName`, but `stageBeganAt` only exists for ~16% of rows and only since Aug-2025 → no historical dated events.
- `dbo.CommAmendmentSnapshot` has real `dtSign` (back to 2020) + `iType` (renewal/new/expansion…), but **no working prospect FK** was found.

## How to fix it (owner of the `rea.*` pipeline)
1. **Fix the label generation** so `Label=1` only for snapshots within 180 days *before* an actual signed conversion, with the correct date — and **broaden** the positive to New + Expansion + Renewal + Relocation (per your call). Establish the correct prospect→signed-amendment linkage first (the current one is wrong).
2. **Backfill dated conversion events** across the full snapshot window — `vw_Deal.stageBeganAt` only goes back to Aug-2025, so historical conversions aren't dated. Source executed/signed dates from `CommAmendmentSnapshot.dtSign` once the prospect FK is established.
3. **Accumulate** until there are ≥~500 distinct positive prospects.
4. **Then validate**: group-disjoint (a prospect never spans train/test) quarterly walk-forward, report **PR-AUC + Brier + log loss** (pooled across folds for small N) — the same methodology that validated multifamily.
5. **Until then:** keep brokerage on the rules score (`v0.1-rules`), clearly labeled INTERIM in the UI and Model tab. Do not deploy a model on the current label.

## The corrected-label prototype (read-only; hand this to the table owner)
```sql
-- Correct, time-localized conversion label (Executed deal within 180d AFTER each snapshot).
-- Replaces the mislinked rea.ProspectLabelSnapshot. Broaden by adding amendment types
-- (New/Renew/Expansion/Relocation) once a dated, prospect-linked amendment source exists.
SELECT f.hMy, f.SnapshotDate,
  CASE WHEN EXISTS (
    SELECT 1 FROM rea.vw_Deal d
    WHERE d.commProspectId = f.hMy
      AND d.stageName = 'Executed'
      AND d.stageBeganAt >  f.SnapshotDate
      AND d.stageBeganAt <= DATEADD(DAY, 180, f.SnapshotDate)
  ) THEN 1 ELSE 0 END AS Label
FROM rea.ProspectFeatureSnapshot f;
```
Prototype: `light_gbm_endpoint/scripts/signaldesk_label_prototype.py` (measures the positive count + attempts CV).

---

## Update — 2026-06-23 (bridge hunt): linkage IS solved; the surviving blocker is feature-history depth
A read-only follow-up (SELECT-only; nothing written to Yardi/InsightMart) resolved the linkage this doc earlier called unfound, and sharpened the root cause.

### Root cause, exact: the label points at PROPOSALS, not signed leases
`rea.ProspectLabelSnapshot` resolves (via the prospect's `hAmendment`) to **iType=13 = "Proposal"** amendments — confirmed against `Yardi7_Reporting.dbo.CommAmendmentType` (`0=Original Lease, 1=Renewal, 2=Expansion, 3=Contraction, 4=Termination, 13=Proposal, 15=Relocation`). iType=13 is the single largest bucket (15,866 rows). So the label is keyed on **proposals**, which is why `dtSign` is the proposal date (often years before the snapshot) and `DaysToConvert` is garbage. It is systematically the wrong amendment *type*, not "a random 2020 amendment."

### The working bridge (revises "no working prospect FK was found")
`rea.vw_Deal` carries BOTH the prospect and the tenant on every deal row: `commProspectId` (= `ProspectFeatureSnapshot.hMy`) and `tenantId` (Yardi tenant). Coverage: **1,350 prospects ↔ 1,385 tenants, all populated** (2,767 deal rows). And `vw_Deal.tenantId = dbo.CommAmendmentSnapshot.hTenant` matches **1,334 / 1,385 (96%)**. The earlier 0-match joined on `hDeal` (which is null in `CommAmendmentSnapshot`); the real keys are `hMy` (amendment PK) and **`hTenant`** (tenant). Canonical chain:

```
prospect hMy = rea.vw_Deal.commProspectId
             → rea.vw_Deal.tenantId
             → dbo.CommAmendmentSnapshot.hTenant
             → signed amendments  (iType IN (0,1,2,15);  date = dtSign, fallback dtStart)
```

Secondary cross-check bridge (name-based, for prospects lacking a deal row): `rea.TenantCompanyAlias` (462 approved tenant-name→ZI-company rows) + `rea.vw_BootstrapProspects` (`companyId` ↔ prospect `hMy`).

### But linkage was not the only blocker — feature depth is
Via the bridge, **184 prospects** link to conversion-type amendments; **522** of those have a `dtSign`, spanning 2009→2026. Those signings cluster in **2020-2023** (peak 2021: 49 prospects / 129 amendments). `ProspectFeatureSnapshot` only spans **2024-11-30 → 2026-04-30** (33,181 rows, 1,881 prospects), so the historical conversions have no contemporaneous features. Positives inside the feature window:

| conversion event date | distinct prospects (with features) converting 2024-11-01 .. 2026-04-30 |
|---|---|
| `dtSign` (signed) | **5** |
| `dtStart` (lease commencement) | **45** |

Up from 13, but still far short of ~500.

### Revised fix path
1. **Linkage: solved.** Regenerate the label off the `vw_Deal.tenantId ↔ CommAmendmentSnapshot.hTenant` chain (conversion = `iType IN (0,1,2,15)`; date = `dtSign`, fallback `dtStart`), replacing the proposal-keyed `ProspectLabelSnapshot`.
2. **Close the positive gap (the real gate):** either (a) **backfill `rea.ProspectFeatureSnapshot` to 2020-2023** so the ~380 historical signings become trainable, or (b) accumulate forward. Walk-forward stays blocked until one of these reaches ~500 distinct positives — but the label can now be built correctly.

### App impact (asked separately)
The production Signal Desk app (`signal-desk/`) does **not** consume the broken label — a full grep found zero references to `ProspectLabelSnapshot`. Its user-facing score is the rules-based `rea.CompanyScoreSnapshot` / `rea.vw_CompanyScore`; deal/lease history reads real Yardi via `rea.vw_Deal`. The broken label is read ONLY in offline training (`contracts/signal_desk_brokerage/data.py::training_data()`); only the light_gbm "Brokerage" tab surfaces the leaky model output, already labeled INTERIM. Caveat: the app's rules score is stale (`ScoredAt ~2026-04-21`).

### Historical feature sources for backfill (archived/history tables) — PARTIAL, one hard gap
Hunted InsightMart for tables that could supply point-in-time features for the 2020-2023 conversions (the missing piece for trainable historical positives). Most feature families have deep history and map to prospects/tenants via the bridges — EXCEPT the ZoomInfo scoop/intent family, which is current-only.

**Have deep history (usable for backfill):**
- `dbo.CommDealStageHistorySnapshot` — deal-stage transitions, 96,260 rows back to 2019-12 (70,583 pre-2024-11), keyed by `hAmendmentID` (→ `CommAmendmentSnapshot` → `hTenant` → prospect). Stage / days-in-stage features.
- `dbo.CommAmendmentSnapshot`, `dbo.LeaseSnapshot` (`LeaseFrom` 1993+), `dbo.CommercialRentStepHistory` — lease term/expiration/rent features.
- `dbo.CrmActivitySnapshot` — 8,744 activities back to 2020-12 (5,351 pre-floor); prospect link is via `hRecord` (its `hMy` is the row PK — confirm `hRecord` target). Activity-count features.
- `rea.LegacyTouch` — 68,523 broker touches back to 1990, covering **223 of the feature-prospects** (`hMy` = prospect). Touch/engagement features.
- `zi.Contact` (`ValidDate` 2002+); `dbo.CommercialProspectSnapshot.dtCreated` shows prospects exist back to 2019-12.

**Current-only — the GAP:**
- `zi.Scoop` — 8,847 scoops, but BOTH `PublishedDate` and `OriginalPublishedDate` are entirely **2026-03-22 → 2026-06-23 (zero before 2026-03)**. `loan.ZoomInfoSignals` likewise 2026-only. So the scoop/intent signal features (funding/hiring/layoff/leadership/relocation) cannot be reconstructed for 2020-2023 — and are near-empty for most of even the current 2024-11→2026-02 window.
- `dbo.prospect_history` — recent (2025-10→2026-03) and a disjoint id space (0 overlap with prospect `hMy`); not a backfill source.

**Implication:** a PARTIAL historical feature vector (lease / stage / activity / touch / contact) can be rebuilt for the ~100+ bridged converters of 2020-2023, but the scoop/intent family would be missing → feature-distribution drift vs the 2024-2026 rows. Options: (a) train a reduced scoop-free historical model, (b) model the missingness explicitly, or (c) source dated historical scoops externally — the one place ZoomInfo credits could help, though `zi.Scoop` here holds only ~3 months, so a deep dated backfill may not be available even from the vendor.

---

## Update — 2026-06-24: acquisition reality, the right sample-size bar, and the v0.3 model

### Historical signal acquisition (external research, verifier-checked)
- **ZoomInfo historical scoops are unrecoverable on any tier.** Search-Scoops has no date-range filter (sort by `originalPublishedDate` newest-first only); Intent is forward-only/daily/fast-decaying (G2 path backfills only 90d at auth, no restore after disconnect; the "2-year range" is a query filter on already-ingested data, not a backfill); Scaling API + Bobsled DaaS are throughput/delivery tiers, not history; credits are enrichment, not signals. `zi.Scoop` empty before 2026-03 is expected behavior.
- **Intent co-ops can't reach back either:** Bombora ~12 mo, 6sense 180 d–24 mo, Demandbase no multi-year archive. (Bombora's "Historical Buyer Journey Analysis" is an 18-mo per-customer *service*, not a dated feed; its one-sheeter is a stale 2017-18 artifact.)
- **Reconstruct the underlying events instead:** **Revelio Labs** for hiring/expansion + layoffs — headcount + flows (Count, Inflow/Outflow) monthly **back to 2008**, layoffs (WARN-style) from 2008, via API + flat file (genuine 2020-2023 backfill). ⚠️ `Expected_hires` is NOT workforce-flow / hiring-intent — it's a COSMOS-tier job-posting count (history from 2021); the load-bearing part is **headcount/flows**. **Crunchbase / SEC Form D** for funding; **press archives** for leadership change. Otherwise forward-accumulate scoops from 2026-03.

### Sample size: you do NOT need ~500 (retract the earlier figure)
"~500" / the 10-EPV rule is not literature-derived. The standard is **Riley et al. 2019** (compute via `pmsampsize`: p, base rate, anticipated Cox-Snell R²). The binding unit here is **prospect-level events** (monthly snapshots of one prospect are correlated), not rows. Computed (Riley criterion-1) for our case:
- **Row-level** (33,181 snapshots, 259 events, prev 0.78%): minimum ~4k-12k rows — we *exceed* it; but this double-counts correlated snapshots.
- **Prospect-level** (1,881 prospects, **54 events**, prev 2.9%): need ~**88** converting prospects for a 12-feature model (EPP≈7), but only ~**44** at **6 features** (EPP≈9) — so **cutting effective parameters makes today's 54 defensible**. Confirm with `pmsampsize`.

### Method stack for a defensible small-n model (adopted in v0.3)
Firth's bias-reduced logistic (FLIC/FLAC so low-prevalence probabilities aren't biased upward) and/or penalized logistic (ridge for prediction, lasso/elastic-net for selection; report penalty sensitivity). **Do NOT SMOTE/resample** (leaves AUC flat, wrecks calibration) — shift the decision threshold instead. **Calibrate with Platt/beta, not isotonic** (isotonic overfits below ~1000 points). Validate with **expanding-window rolling-origin walk-forward + paired-bootstrap CIs** vs a per-fold base-rate baseline. (Multifamily-churn's isotonic is on 1,534 points >~1000, so defensible there — but worth a Platt cross-check.)

### v0.3 model (served default)
`bundles/signal_desk_brokerage/v0.3-corrected-logistic` (`scripts/train_brokerage_v03.py`): corrected bridge label, scoop-free; **elastic-net selected 8 of 29 features**; fit a penalized-ridge logistic AND a **Firth+FLIC** logistic (no resampling); **expanding-window rolling-origin** (6 quarterly test folds) pooled OOF + **paired-bootstrap CI**. Winner **Firth+FLIC**: PR-AUC **0.108** (no-skill 0.007, **lift CI95 [0.079, 0.134] — excludes 0**), ROC-AUC 0.924 (imbalance-inflated), Brier 0.007, calibration slope 0.75. Replaces the v0.2 LightGBM (29-feat tree was over-parameterized at 54 events; `is_unbalance` reweighting dropped). Still INTERIM/data-limited; the lever is reconstructing the 2020-2023 signal families (above) to un-strand the ~100+ historical conversions.

---

## Update — 2026-06-25: ZoomInfo scoops WERE backfilled; deal-win model (Model A) built

### Correction: historical scoops are recoverable (reverses the 2026-06-24 finding)
`zi.Scoop` was backfilled to **109,654 rows spanning 2022-01-01 → 2026-06-25 (52,341 pre-2024-11)**, credit-free via scoop search (`signal-desk/.../zi_backfill_scoops_historical.py`). The 2026-06-24 "historical ZoomInfo scoops unrecoverable on every tier" conclusion is **superseded** — point-in-time scoop features are now reconstructable 2022+. Revelio/Crunchbase/Form D become optional enrichment, not the critical path.

### Data now available (verified live)
- Bridge `rea.ProspectCompanyLink` (hMy↔CompanyID), ~46% matched; `rea.YardiDeal.ZiCompanyId` on the deal side.
- **Deal grain** `rea.YardiDeal`: 906 resolved (Executed=won 224 / Cancelled+Dead 682), **24.7% win** — clean label, ~4× the prospect-grain positives.
- Cohorts: `rea.DealPattern` (size×vertical win-rate + median rent/term), `mkt.CohortCatalog`, `rea.Concept`.

### Model A — deal-win re-ranker (SHADOW): `scripts/train_brokerage_dealwin.py`
Port of the Signal Desk shadow win-model (L2 logistic) + point-in-time scoop add-on, corrected resolved-date backtest (ModifiedAt split, train-only cohort prior, paired-bootstrap AUC-lift CI).
- **Base model reproduces the app edge:** mean AUC **0.781** vs cohort baseline **0.711**, lift **+0.069**, CI excludes 0 on the 2 larger splits. Bundle `bundles/signal_desk_dealwin/v0.1-dealwin-shadow`, surfaced as a shadow card in `/models`.
- **Scoops do NOT help at the deal grain** (AUC 0.764 < 0.781, worse ECE): only **122/906** resolved deals have a scoop *before* deal-open (41% ZI coverage × deals predating the company's first scoop). The scoop richness is for the **prospect + marketing** grains (current, high-coverage), not the historical deal-win model.
- ECE ~0.09-0.14 → STAYS SHADOW. Graduation gate: beat live P(win), ablate leadBroker (era-proxy), confirm ModifiedAt split integrity.

### Roadmap
- **Brokerage:** (A) deal-win re-ranker [graduate via the gate; no scoops]; (B) cold-prospect v0.4 [re-add the now-available scoop family to v0.3 + ProspectCompanyLink enrichment].
- **Marketing:** enrich cohort targeting with `rea.DealPattern` win-rates + median-rent $ EV + **scoop/intent momentum per cohort** (where the backfill pays off) + company-grain signal density.
- **Top data lever:** bridge coverage ~46% → review the ~1,010 unmatched `ProspectCompanyLink` rows.

---

## Update — 2026-06-25 (cont.): company-centric brokerage (scorecard shipped; recency model = negative)
Reframe: rank the COMPANY, not the broker. The deal-win model is dropped (its only edge was `leadBroker`, a broker-selection era-proxy: ablation collapses it to the cohort baseline; `leadBroker` prevalence shifts ~7%→40% train→test).
- **Company scorecard (SHIPPED, served):** `GET /brokerage/scorecard` — broker-free per-company ranking read live from `rea.CompanyScoreSnapshot`: REA Score+Decile, **WinProb** (cohort base × concept lift) + base/cohort/lift/tags, **ExpectedValue** (populated here, unlike DealPattern), scoop momentum (90d count + recency), months-to-lease-end, LLM lifecycle. Sortable by score/winprob/value/scoops/recency/lease. This is the company-ranking brokerage product (multiple grounded probabilities), real-time + API-callable like multifamily.
- **Learned recency model — NEGATIVE:** `scripts/train_brokerage_recency.py` trained a deal-activity propensity (label = company opens a deal ≤90d; 677 companies, 31k panel rows, 6% base) on point-in-time scoop-recency + firmographics. Logistic AUC **0.505** (CI straddles 0.5), LightGBM 0.483 → **no signal**. Deal-*open* is endogenous to RTS outreach, not company news. Not served; recency stays an *observed* scorecard signal, not a learned predictor.
- **Served models (all real-time, API-callable):** multifamily `POST /predict/multifamily-churn`; brokerage `POST /predict/signal-desk-brokerage` (v0.3) + `GET /brokerage/scorecard`; marketing `GET /marketing/targeting`; `GET /models` (4 cards). The signal lever remains point-in-time concept tags (B2) + bridge coverage — not more scoops, not recency-as-predictor.

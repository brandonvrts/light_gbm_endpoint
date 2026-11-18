# Phase 0 Discovery — Residences at 62W (property 82)

**Date:** 2026-08-18 · **Scope:** property 82 only · **Method:** read-only live Yardi queries
**Answers:** §5.3, §7.3, §8.2 of [DESIGN_UNIT_ECONOMICS_CONCESSIONS_62W.md](DESIGN_UNIT_ECONOMICS_CONCESSIONS_62W.md)
**Revised 2026-08-18 (same day) after adversarial review:** corrected the occupied-unit-years denominator (§5 — earlier sum-of-spans double-counted 47 overlapping co-tenancies), the h180 metric citation (§4 — 0.891 was the v4 figure), the post-notice flip count (§3 — 15 attempts / 3 renewed, not "never run"), and marked the §3 garage ATT as degenerate. The $248k headline is unchanged by all of these (it reduces to turns÷4 × cost-per-turn).

---

## Verdict

| Workstream | Question | Answer |
|---|---|---|
| **§5.3 Make-ready cost** | Can we get real per-bedroom turn costs? | **YES** — unit-attributed in GL. $600 flat replaced with $791 / $872 / $1,094 all-in. |
| **§7.3 Garage economics** | Are 62W garages scarce or stagnant? | **SEASONALLY SCARCE, never stagnant.** The `C_garage ≈ $0` premise is refuted. Rent was also wrong ($87, not $74). |
| **§8.2 Observational uplift** | Do concessions measurably reduce churn? | **Historically unmeasurable — pilot required.** The estimand the program needs (offer before the decision persuades *that* decision) was almost never run; the measurable N→N+1 proxy is single digits inside a ±5–13pp confounding floor. This is *not* evidence that concessions don't work. |
| *(new)* **Model horizon** | Can the optimizer run on the serving bundle? | **NO** — the default h60 bundle yields 3 persuadables at 62W; h180 yields 18. |

**Net effect on the design:** the *unit-economics* half (Phase 1) is validated and should proceed as specified. The *optimizer* half (Phase 2) needs restructuring before implementation — two of the three offer families are dead on arrival at 62W economics, the garage offer is capacity-capped and splits into two products, and no assumed-uplift default can be justified.

The headline business number: **62W loses ≈$248,000/year in net turnover cost, 68% of it in 2-bedrooms.**

---

## Executive layer — the three decisions this document asks for

**Decision 1 — Ship the unit-economics surface (Phase 1).** Engineering only, no concession spend. It prices every turn, per bedroom, per property, and it is the measurement backbone for the operational levers below. *Recommended: approve.*

**Decision 2 — Point the first retention dollars at the two operational levers, not concessions.** Turn-cycle tail management (~$31k/yr at 62W; the 2BR vacancy tail, not the median, is the target) and Jan–Feb lease-end migration (up to ~$24k/yr; winter move-outs take 78 median days to re-fill vs 32.5 in spring). Both are deterministic, need no uplift assumption, and carry no fair-housing surface. *Recommended: approve — this is ~10× the concession program's optimistic case.*

**Decision 3 — Concession measurement: pick a rung on the ladder.**
- **(a) Post-notice rescue trial — recommended now.** Offer a uniform 1-free-month renewal (13-mo term) to *every* notice-giver, randomized. Nearly free under the null (~$9k/yr portfolio spend if nothing works, because the concession is only consumed on the ~3% who renew), strongly profitable if it works (each flip is worth ~$4,700), detects a 2.7%→10% effect in ~1 year portfolio-wide and 2.7%→20% in ~3 months. Cleanest compliance posture: uniform offer to a self-identified group, no algorithmic targeting. History's 3-of-15 says the dose plausibly works.
- **(b) Portfolio-wide all-decisions randomization** — the only design that answers the *pre-notice* question (~10 months at +10pp). But it is a **knowledge purchase, not a profit play**: worst case ≈ −$165k/yr in concession spend under the null (3%-discount arm), roughly breakeven if +10pp is real, against a program EV ceiling of $18–40k/yr. Fund only if the strategic answer is worth that, and only after counsel review (§12.3).
- **(c) Wait for the duration-model horizon (v2).** At 2 retained lease terms per save, the 3% discount's breakeven falls from 10.8pp to **5.0pp** and one free month's from 40.7pp to 15.9pp — the multi-term view may change the answer with zero concession spend. The duration model is already built; validating it is engineering, not spend.

*Recommended: (a) now, (c) in parallel, (b) only on an explicit strategic call.*

---

---

## 1. §5.3 — Make-ready cost per bedroom: ANSWERED (yes)

62W's GL puts the unit number in `GLDETAIL.SNOTES` as a `U=<unit> :<description>` prefix on every make-ready invoice line. 100% of parsed unit codes resolve to a `UNIT.SCODE` at property 82, and the zero-attributed-cost share of turns is **0% (1BR) / 1.4% (2BR) / 0% (3BR)** — essentially every turn has real dollars attached.

### In-scope accounts (property 82, `IBOOK=1` accrual, 4 years)

| Acct | Name | Rows | $ (4y) | Verdict |
|---|---|---|---|---|
| 52501002 | Painting — Make Readies | 409 | $116,542 | IN — routine |
| 52201002 | Janitor Service — Make Readies | 387 | $43,826 | IN — routine |
| 52251003 | Carpet Cleaning — Turnover | 304 | $29,662 | IN — routine |
| 52501004 | Painting Repairs — Make Readies | 27 | $4,996 | IN — routine |
| 87000003 | Carpet Replacement — Turnover | 57 | $47,338 | IN — capital-coded but explicitly turnover |
| 87000006 | Vinyl Replacement — Turnover | 3 | $4,714 | IN — same |
| 87000005 | Carpet Replacement — Reimburse | 4 | $2,101 | IN — resident chargeback contra |

Excluded: occupied-unit equivalents (52251001, 52501000/3, 52141000), Corporate Unit Services (52360000, $63.6k — corporate-apartment program), and the wider 87xxx/88xxx capital-replacement block (appliances, ceiling fans, window coverings, A/C & furnace, shower walls, subfloor — ~$130k). Only the three accounts whose names literally say *Turnover* were pulled in; including life-cycle capex would double-count capital into a retention decision. `52110000` Appliances-Parts ($24k, ~19% tagged) is genuinely ambiguous — occupied and turn work mixed with no split — and was excluded.

### Direct cost per turn

| BR | n turns | Zero-cost | p25 | **Median** | Mean | p75 |
|---|---|---|---|---|---|---|
| 1 | 84 | 0 (0%) | $406 | **$463** | $621 | $605 |
| 2 | 145 | 2 (1.4%) | $519 | **$544** | $727 | $738 |
| 3 | 7 | 0 (0%) | $612 | *$2,126* | $1,948 | $2,518 |

Medians are stable year over year (1BR $455→$455→$473→$482 for 2022–25; 2BR $519→$519→$544→$555). The ~15–30% jump in 2026 move-outs is **at least partly mix shift, not vendor inflation** *(corrected post-review)*: painting cost per event is flat 2025→2026 ($276 → $269), but carpet-replacement events per painting-turn rose (14% → 18%) — 2026 turns are drawing the expensive card more often. Watch another two quarters before repricing the make-ready constants. A censor-safe re-run (move-outs ≤ 2026-04-20) changes nothing.

**3BR is not usable at n=7** — the individual turns are $262 / $594 / $630 / $2,126 / $2,413 / $2,624 / $4,984, so the median is a coin flip on whether carpet got replaced. Use the 2BR value scaled by sqft (1511/1073) → **$766**.

### Property-level pool (not unit-attributable)

Make-ready stock, painting supplies, re-key locks, and vacant-unit electricity carry no unit tag but are cleanly per-property-per-period:

| | 2023 | 2024 | 2025 |
|---|---|---|---|
| Total | $19,201 | $20,587 | $23,149 |
| Turns | 61 | 72 | 59 |
| **$/turn** | $315 | $286 | $392 |

3-year pooled: **$62,937 / 192 turns = $328 per turn**, applied as a flat adder.

### Recommendation

**All-in make-ready: 1BR $791 · 2BR $872 · 3BR $1,094.**

The $600 flat placeholder was the right order of magnitude but the wrong shape — it overcharges 1BR direct cost by ~30% and undercharges the all-in figure by 25–45%. The design's sqft-scaled fallback ($450/$630/$890) over-scales: the true 1BR:2BR ratio is **0.85**, not the 0.74 sqft ratio, because cleaning (~$112–123) and carpet shampoo (~$123–140) are near-flat per unit regardless of size.

**These figures are contracted-vendor + materials only.** In-house maintenance labor (`51050000` payroll $242k/4y + `51060000` benefits $56k/4y) is property-level with no make-ready split; if 100% were make-ready it would add $1,270/turn. The true share is not derivable from this data and must be stated on any surface that shows these numbers.

**Work orders are a dead end for dollars.** Legacy `MMWO`/`MMWODET` have 0 rows at property 82. `mm2wo` has 10,440 work orders but the `Make Ready` category shows only 7 of 300 with any dollar value ($7,273 total, all materials); `mm2wodet.dunitprice` is $0.00 on essentially every make-ready line. Hours in `dquan` are wildly inconsistent (866 hrs 2023 → 96 hrs 2024 → 26 hrs YTD 2026) and must not be used to impute labor cost.

---

## 2. §7.3 — Garage economics: ANSWERED (premise refuted)

`RENTABLEITEMS` **is populated** — the empty-purpose-built-table landmine that killed the RMConcession tables does not apply here.

### Inventory: 137 stalls (high confidence, direct)

Property 82 has 149 rentable items: **137 `gar` (Garage Stall)** + 12 `store` (Storage Unit). All 137 first appear in `RentableItems_Status` in 2017 with zero conversions since — inventory has been fixed since the property opened. It equals the 137 non-excluded units exactly (47 × 1BR + 84 × 2BR + 6 × 3BR): purpose-built one stall per unit.

### Occupancy: 84.7% now, but that is the seasonal trough

Current state: 116 of 137 occupied (112 Occupied-No-Notice + 3 Notice-Unrented + 1 Notice-Rented), 21 Vacant-Unrented. Revenue run-rate $10,100/mo; forgone at list on the 21 vacant stalls is $1,850/mo ($22,200/yr).

Two structural facts the design missed:

**(a) Strong winter seasonality.** Iowa garages are a winter good. Mean occupancy by calendar month 2023–2026: Dec 91.7%, Jan 88.2%, Feb 87.6%, Mar 88.2% … Jul 84.1%, **Aug 83.0% (trough)**. In Jan–Mar 2026 occupancy hit **96.4%**. A static 90% threshold read in August says STAGNANT and in January says SCARCE for the same physical asset — **the proposed regime switch is a seasonality detector, not a scarcity detector.**

**(b) Demand is rising.** Annual mean: 2021 73.0% → 2022 75.1% → 2023 84.5% → 2024 88.3% → 2025 84.1% → **2026 93.0%**. The genuinely stagnant era was the 2020–2022 COVID trough (68.6% low). It is over.

### There are no stagnant garages

Months-vacant for the 21 currently-vacant stalls: 7 under 1 month, 7 at 1–3, 3 at 3–6, 1 at 6–12, and **zero over 12 months**. Median 1.2 months. Across 259 completed vacancy spells 2022–2026, **median time to re-rent is 2 months** (mean 4.4, p75 5); 130 of 259 re-rented within 2 months.

This kills the `C_garage ≈ $0` branch. A vacant 62W garage is a stall between tenants, not dead inventory.

**Replace the binary switch with a continuous opportunity cost:**

```
C_garage ≈ (12 − E[months_to_re_rent | stall, tier, season]) × tier_rent
```

At the mean (4.4 mo): ≈ 7.6 × $87 ≈ **$661**. At the median (2 mo): ≈ 10 × $87 ≈ **$870**. Either way, nowhere near zero — so "free garage strictly dominates every cash concession" does not hold.

### Rent: two tiers, $75 and $100 — blended $87, not $74

`RENTABLEITEMSTYPE` defines one garage type (list rent $100), but per-stall `RENTABLEITEMS.CRENT` carries a real two-tier price: **70 stalls at $75, 67 at $100**. Billed amounts confirm it ($75.00 × 1,908 rows, $100.00 × 1,758 rows = 83.7% of all rows since 2024-01).

The remaining 16% of rows are move-in/move-out prorations (19.35 = 75×8/31, 48.39 = 75×20/31, …) and reversals. **The $73.87 figure reported earlier was proration-contaminated and understates true rent by 15%.** Correct blended full-month rent is **$86.99**; scarce-regime annual cost is **$1,044**, not $888. Interestingly, $100-tier stalls sit vacant longer (median 1.75 mo) than $75-tier (0.65 mo) — the premium tier is the softer market.

### Attach rate — and a hard capacity ceiling

| Bedrooms | Residents | With ≥1 garage | Attach % | Stalls/resident |
|---|---|---|---|---|
| 1BR | 37 | 27 | 73.0% | 0.76 |
| 2BR | 69 | 53 | 76.8% | 1.00 |
| 3BR | 6 | 6 | 100.0% | 1.50 |
| **All** | **112** | **86** | **76.8%** | 0.95 |

*(Count reconciliation, added post-review: the "92 garage holders" in the payer-trap callout below vs the 86 residents-with-garage here is not an inconsistency — 4+ occupied stalls are held by `ISTATUS=4` tenants who are no longer current residents, so stall-holder counts and current-resident-holder counts legitimately differ.)*

**The most consequential finding for concession design, and it was not in the design doc:** only **26 of 112 residents lack a garage**, and only **21 stalls are vacant**. Therefore:

- **"Free garage" is two different products.** For the ~23% without one, it is a new assignment from the vacant pool at the discounted cost above. For the **77% who already rent one**, it is *waiving an existing charge* — a pure $87/mo revenue give-up at the full scarce price, with no occupancy offset and no effect on garage occupancy.
- **The addressable pool is ~21 residents** (min of 21 vacant stalls and 26 garage-less residents), and it is **seasonal — only 4–6 stalls were free in Jan–Mar 2026**. Any optimizer must carry this as a capacity constraint.

### The design's proposed fallback would have failed badly

Counting "active charge-type-17 payers / inventory" undercounts because **20 of 92 garage holders rent multiple stalls** (17 hold 2, two hold 3, one holds 4 → 116 stalls across 92 holders). Max monthly distinct payers over 4 years is 104 vs true inventory 137 — a 24% understatement that would have pinned occupancy near 70% and hard-locked the model into the STAGNANT regime year-round. **Count stalls, never payers.**

---

## 3. §8.2 — Observational uplift: ANSWERED (negative)

### The headline number is a tautology, not an effect

`HRETENTIONACCT=53` (Renewal Concession) credits are posted **at the moment the renewal is executed** — median **18 days** before the decision, with 79% inside 60 days and 18% posted *after* the new term starts. In-term acct-53 events run 137 renew / 9 move-out.

> Naïve comparison: **93.3% renewal (n=135) vs 39.3% (n=876) = "+54.1pp."**
> Stratified: +53.7pp. Regression-adjusted: +54.2pp.
> **This is a definition. The credit exists *because* the renewal was signed. Discard it.**

The same mechanism inflates the garage concession from ~+1pp to +15.5pp when the lag filter is removed. Any model consuming TRANS concessions without a lag filter will leak the label.

### The one clean, dose-relevant estimate

Treatment must be defined as a concession posted inside the expiring term and well before the decision. Doing that reframes the only leakage-free question available: *does a ~1-month cash credit at renewal N buy renewal N+1?*

| Treatment (dose) | n treated | n control | Renew: T | Renew: C | Stratified ATT | 95% CI |
|---|---|---|---|---|---|---|
| **Acct 53 @ renewal (1.00 mo rent)** | **71** | 330 | 57.7% | 54.2% | **+9.6pp** | **[−4.9, +23.4]** |
| same, transfer-corrected | 71 | 330 | — | — | **+11.8pp** | — |
| Acct 54 @ renewal (1.03 mo) | 16 | 385 | 56.2% | 54.8% | +4.3pp | [−15.5, +24.1] |
| Acct 30 @ renewal (0.34 mo) | 25 | 376 | 48.0% | 55.3% | ~~+5.0pp~~ **degenerate — do not quote** | [−14.8, +24.8] |

Base renewal rate on renewal terms: 54.9% (n=401).

**Why the ATT column differs from the raw T/C columns (reconciliation, added post-review):** the raw columns are unmatched means; the ATT weights only strata containing both arms. For acct 53 the steps are raw +3.5 → matched +5.6 (64 of 330 controls have no treated counterpart and drop) → ATT **+9.6 against an implied counterfactual control mean of 48.9%** — same-signed throughout, ordinary composition (treated skew to longer tenure and earlier, lower-base periods). Acct 54 likewise (+1.4 → +3.9 → +4.3, counterfactual 51.9%). **Acct 30 is different: its raw −7.3 → matched −0.5 step is legitimate composition (treated are 72% in 2020–21), but the matched −0.5 → +5.1 step is manufactured by five single-observation strata** — the two largest strata (13 of 24 treated, 54% of weight) both show a *negative* diff totalling −9.9pp, while five n=1 strata where the lone treated resident happened to renew contribute +12.6pp. Report the renewal-terms garage effect as "no estimable effect, n too small." The substantive verdict (garage ≈ null) is unchanged — the CI already spanned zero — but the +5.0 point estimate must not be quoted. (The *full-panel* garage estimate in the table above, +3.6pp raw-flipped-by-period, IS a clean Simpson's correction and stands: base renewal rose 39.4% → 51.7% across periods while 76.6% of treated decisions sit in 2017–21.)

### The negative controls are the load-bearing result

Accounts **50** (Senior Citizen Discount) and **51** (Employee Discount) are *entitlements* — nobody ever granted one to retain anybody, so their true retention effect is **exactly zero**. Under the stratified design they measure **+13.2pp** and **+8.4pp**; under full regression adjustment, **+6.0pp** and **+3.4pp**.

> **The stratified estimator therefore carries ≈+10pp of pure confounding bias in this dataset, and the regression-adjusted estimator ≈+5pp.** That is the same order of magnitude as every candidate signal, and larger than the garage breakeven.

Every estimate here must be read as a point estimate ± **5–13pp of irreducible confounding**, on top of sampling error. Negative controls cost nothing and should be standing practice in any future uplift measurement — this is the single most useful number in the memo.

### Bias direction: both present, on different accounts

**Acct 53 — net bias strongly UPWARD.** Recipients are simultaneously squeaky wheels (3.7× late charges, 4× NSF, 1.6× work orders, 5.6× MTM events → biases down) *and* near-certain renewers (2.1× tenure, 96% already repeat renewers vs 38% → biases up). The upward imbalances are 2–3× larger in standardized mean difference (1.06–1.58) than the downward ones (0.36–0.58), and tenure-band strata cannot absorb an SMD of 1.1. Common support is poor — there is essentially no untreated 3.5-year repeat-renewer population to match against. **+21pp is an upper bound, not an estimate.**

**Acct 30 — was never a retention tool.** Recipients are brand-new move-ins (83 days into tenancy at term start), in cheaper units, with *fewer* work orders and late charges, concentrated in 2020 (SMD −1.01 on year). The garage concession at 62W was a **lease-up sweetener during the 2020 soft market**. There is no organizational precedent for what the design proposes.

### Notice is (almost) terminal — and the play was barely run *(corrected post-review)*

| | n |
|---|---|
| Decisions with a Notice Given event during the term | **480** |
| …that renewed anyway (base rate) | **13 (2.7%)** |
| Post-notice **discretionary** concession attempts (accts 53/30/54/29/51, union) | **15 decisions** |
| …that renewed | **3 (20.0%)** |
| Post-notice **entitlement** credits still accruing (accts 33/50) | 90 decisions |
| …that renewed | 3 (3.3%) |

An earlier draft said "the play was never run — 8 attempts." Wrong on both counts: **15 discretionary attempts in nine years, and 3 of them renewed (20% vs the 2.7% post-notice base).** All 3 credits posted on/within 19 days of the decision, so posting dates can't establish causation — but one (t0008203, 2BR, 2022) carries an explicit note, *"Renewal concession 1 mo free w/13 mo approved by JL"*, posted 46 days after the notice: a genuine documented post-notice flip at a 1-free-month dose. Read as: **n=15, 3 apparent saves, 1 unambiguously documented — directionally encouraging but far too thin to be a flip rate.** The entitlement contrast sharpens the point: 90 post-notice decisions with a standing recurring discount renewed at 3.3%, indistinguishable from base — recurring discounts do not flip notices; a discretionary lump credit occasionally may.

The operational conclusion survives softened: 97% of notices end in a move-out, the optimizer should still fire **pre-notice** as the primary play — but post-notice rescue at a ~1-month dose is not disproven, and at 20%-of-15 it is arguably the cheapest experiment available (the notice population self-identifies, no targeting model needed).

### Verdict against the breakevens

| Offer family | Breakeven | Best observational estimate | Verdict |
|---|---|---|---|
| 1 free month, 2BR | **−33pp** | +10 to +12pp on the *next* term, CI [−5, +23], upward-biased; immediate effect **unmeasurable** | **Cannot be certified.** Even the CI upper bound falls short, and the point estimate sits inside the confounding floor. |
| Free garage | **−8 to −17pp** | +1 to +5pp, CI [−12, +12] | **Neither supported nor refuted.** Recipients were 2020 move-ins, never at-risk renewers. |
| Rent discount d% | **−9 to −13pp** | acct 33 (5% Discount): +4 to +5pp, CI [−5, +14] | Null; dose (0.05 mo/mo) is far below the modelled grid. |

**Magnitude answer: "a few points," not "twenty."** The honest range for a ~1-month concession is 0 to +15pp, centred in single digits, and that is for the *next* term rather than the term being decided.

---

## 4. New finding — the optimizer cannot run on the serving bundle

Scoring today's 123-resident 62W population through both v6 bundles:

| Bundle | Horizon | Mean p_churn | Expected move-outs | **Persuadables (0.20–0.60, no notice)** |
|---|---|---|---|---|
| `v6-roadmap-h60` *(current default)* | 60 days | 0.073 | 9.0 | **3** (1BR 1 · 2BR 2 · 3BR 0) |
| `v6-roadmap-h180` | 180 days | 0.129 | 15.9 | **18** (1BR 7 · 2BR 11 · 3BR 0) |

The h60 bundle is a 60-day triage ranker; asking "will they leave in the next 60 days" is low-probability for almost everyone at any moment, so it collapses the persuadable band. The h180 model is the renewal forecast the business case needs. Its **own** manifest metrics are walk-forward ROC-AUC **0.822 ± 0.061**, OOT expiring-holdout AUC **0.859** / PR-AUC 0.797 / Brier 0.106. *(An earlier draft quoted 0.891/0.945/0.083 — those are the **v4** figures from CONTEXT_MULTIFAMILY, mis-cited; corrected post-review.)* **The optimizer must score on h180 — this is a design change, not a config detail.** It also pairs correctly with the pre-notice requirement from §3: a 180-day horizon gives real lead time before notice.

Two honesty caveats for any surface built on these scores *(added post-review)*:

- **Reconciliation of expected move-outs vs observed turns:** 15.9 expected move-outs per 180 days annualizes to ~32/yr, against ~59 observed turns/yr (235/4y). The gap is population and definition, not model failure: the scored frame covers the 123 *currently active, rentable-scoped* residents, while turns count every vacancy event including lease-breaks, mid-term departures, and units cycling more than once a year — and h180's calibration is anchored on the lease-*ending* cohort, so probabilities over the full active book are an off-population read. Any leadership surface must state which denominator it is using or the model will appear to "miss half the churn."
- **The band is eligibility, not persuadability.** The churn model ranks risk; "persuadable" (calibrated p ∈ [0.20, 0.60), no notice) is a targeting band we chose, not measured responsiveness. Treatment-effect heterogeneity is only learnable after randomized offers.

Even at h180, 18 persuadables at one property will not produce a measurable flip rate in one season. Rollout is a **measurement requirement**, not just coverage.

---

## 5. Revised unit economics — 62W

Corrections applied: real make-ready (§1), `UNIT.EXCLUDE = 0` filter (pseudo-units — see §7), occupied-unit-years denominator computed as the **per-unit union of tenancy intervals** *(fixed post-review — an earlier draft summed tenancy spans, which double-counted 47 overlapping co-tenancies for ~104 phantom unit-years and made the table physically impossible: 598.8 unit-years against a 548 maximum)*, 3BR vacancy shrunk toward the property mean (k=20, property turn-weighted vacancy 68.0 days over 235 turns).

| | 1BR | 2BR | 3BR |
|---|---|---|---|
| Rentable units | 47 | 84 | 6 |
| Turns (4y) | 84 | 144 | 7 ⚠ |
| Vacancy, raw | 58.2 d | 75.0 d | 41.9 d |
| Vacancy, shrunk | 60.1 d | 74.2 d | 61.2 d |
| Vacancy months | 1.97 | 2.44 | 2.01 |
| Avg in-place rent | $1,455 | $1,668 | $2,332 |
| Lost rent | $2,866 | $4,070 | $4,687 |
| Make-ready (all-in) | $791 | $872 | $1,094 |
| **Turnover cost, gross** | **$3,662** | **$4,936** | **$5,785** |
| − Relet netting (premium net of renewal increase, 12 mo) | $315 | $255 | $426 |
| **Turnover cost, net** | **$3,347** | **$4,681** | **$5,359** |
| Occupied unit-years (4y, interval union) | 175.3 | 306.0 | 23.7 |
| **Turnover rate** | **47.9%/yr** | **47.1%/yr** | **29.5%/yr** |
| **Expected loss / unit-yr** | **$1,604** | **$2,203** | **$1,582** |
| **Annual property cost** | **$70,294** | **$168,515** | **$9,378** |

**Total: ≈$248,187/year in net turnover cost — 68% of it in 2-bedrooms** (36 turns a year at $4,681 each). The total is invariant to the denominator fix — it reduces algebraically to turns÷4 × net cost per turn. On a routine-only make-ready basis it is $229k; if all in-house maintenance payroll were make-ready it would be $323k.

**The denominator fix changes the story in one place:** with correct occupied-unit-years, **1BR and 2BR turn over at nearly the same rate (47.9% vs 47.1%)** — the earlier draft's "2BRs churn less per unit but cost more" framing is dead. The 68% cost concentration in 2BR is **unit mix (84 of 137 units) plus cost-per-turn ($4,681 vs $3,347)**, not worse retention behavior. Per unit-year the gap is $2,203 vs $1,604 (+37%) — real, but no reason to over-rotate targeting toward 2BR specifically. Also worth noting: 3BR units were occupied 23.7 of a possible 24 unit-years — effectively never vacant.

Sensitivity: gross costs are +$328/turn if the property pool is excluded from make-ready, and vacant-unit electricity ($119/turn) could arguably sit in the vacancy-carry term instead.

---

## 6. Revised offer economics

Breakeven uplift `Δp* = ((1 − p) × C) / (TC − C)` at 2BR, TC_net $4,681:

| Offer | Cost if renewed | p=0.30 | p=0.40 | p=0.50 |
|---|---|---|---|---|
| 1 free month | $1,668 | 38.7pp | 33.2pp | 27.6pp |
| 2 free months | $3,336 | 173pp | 148pp | 124pp |
| 3 free months | $5,004 | impossible | impossible | impossible |
| 3% rent discount | $600 | 10.3pp | **8.8pp** | **7.3pp** |
| 5% rent discount | $1,001 | 19.0pp | 16.3pp | 13.6pp |
| 10% rent discount | $2,002 | 52.2pp | 44.7pp | 37.3pp |
| Garage **waiver** (77% of residents) | $1,044 | 20.1pp | 17.2pp | 14.3pp |
| Garage **new**, E[T]=2mo | $870 | 16.0pp | 13.7pp | 11.4pp |
| Garage **new**, $75 tier, E[T]=4.4mo | $570 | **9.7pp** | **8.3pp** | **6.9pp** |
| ~~Stagnant garage $0~~ | ~~$0~~ | *premise refuted (§2)* | | |

Against a measured effect of "single digits ± 5–13pp confounding," only the **3% rent discount** and a **new $75-tier garage assignment in trough season** land in a range the evidence can plausibly reach — and even they are inside the noise floor. Multi-month free rent is arithmetically dead: three months costs more than the turn it prevents.

---

## 7. Corrections the design doc needs

1. **Unit mix is wrong.** `UNIT.EXCLUDE = -1` marks 9 pseudo-units at 62W (the `GARAGE` placeholder plus seven `WAIT*` waitlist units) holding 20 real TENANT records. True residential mix is **47 / 84 / 6 = 137 units**, not 50 / 88 / 7. Filter `EXCLUDE = 0` everywhere. (This does not change the turn or occupancy figures — the pseudo-units had no turns in the window.)
2. **Make-ready:** $600 flat → $791 / $872 / $1,094 all-in; retire the sqft-scaled fallback ratio (real ratio is 0.85, not 0.74).
3. **Garage rent:** $74 → **$87 blended**, ideally tier-specific $75/$100. Scarce cost $888 → $1,044.
4. **Retire the binary regime switch** for the continuous `(12 − E[T_re-rent]) × tier_rent`. If a threshold is kept, use a trailing-12-month mean (currently 91.0%), never a spot reading.
5. **Set garage inventory = 137** and read it live from `RENTABLEITEMS`; never derive it from payer counts.
6. **Split the garage offer** into `free_garage_new` (from the vacant pool, capacity-constrained, seasonal) and `free_garage_waiver` (comp an existing charge, full price, no occupancy offset).
7. **Retire the "stagnant garage" premise** entirely — it is empirically false at 62W in 2026.
8. **Score on h180, not the h60 default** (§4).
9. **§8.2 closes as answered-negative.** Keep breakeven-uplift reporting as the primary surface; do **not** wire an assumed-uplift default from these numbers.
10. **Posting discipline needs more than account choice:** post the credit at the *new term's* start with an explicit offer date, and log the offer in the outreach log at offer time. As booked today, TRANS timestamps cannot distinguish cause from consequence — that is what destroyed the uplift analysis.
11. **Internal transfers are miscoded as churn** — 69 of 541 move-outs (12.8%) all-time, 23 in the trailing 4 years. Correcting them moves base retention 46.5% → 53.3%. Note the asymmetry: a transfer still incurs make-ready and vacancy cost on the vacated unit, but retains the resident. The cost model should count the turn; the retention model should not count the churn.
12. **Retire or relabel the live 25% flip-rate default.** `persuadables.py` currently ships scenario economics with `flip_rate=0.25` as the default assumption. This memo demolishes that number — the honest historical range is single digits inside a ±5–13pp confounding floor. Anyone who has seen value-if-saved output built on 25% has seen a number this analysis cannot support; the contradiction should be surfaced by us, not discovered. Either drop the default (require the caller to pass one) or label it "illustrative, unsupported by measurement" in the API response.

---

## 8. Reusable assets discovered

### `Lease_History` — the point-in-time lease spine (previously undocumented)

Not an event log. It is Yardi's lease-term / renewal-proposal table: one row per lease term *or per renewal offer variant*.

- **Key:** `hTent` → `TENANT.HMYPERSON`, verified 0 orphans across 34,041 rows DB-wide. (`TENANT` has no `HMY` column in this reporting DB.) `hTenant_History` is 100% NULL at property 82 — reach `tenant_history` via the person key instead.
- **Executed terms = `iStatus IN (4,5)`** (4 = Current = final executed term, exactly one per tenancy; 5 = Past = superseded, i.e. the tenant renewed out of it). Statuses 0/1/2/7 are *proposal options* and duplicate a single decision across term lengths — exclude them from any panel.
- **`cRent` is the rent in force for that term** — the point-in-time rent `TENANT.SRENT` cannot give you. This is the correct basis for historical rent, and the direct fix for the TENANT-mutates-in-place problem.
- **Deriving renewal vs move-out:** order executed terms by `(dtLeaseFrom, hMy)` within `hTent`; a term with a successor → renewed; last term with `DTMOVEOUT` → moved out; last term without → censored, exclude. Use `min(dtLeaseTo, DTMOVEOUT)` as the decision date — early termination is a material minority.
- **Validated at 97.9% / 95.6% concordance** against `tenant_history` Lease Renewal events. At 62W: 1,129 executed terms / 657 tenancies → 472 renewals, 545 move-outs, 112 active; base renewal rate 46.5% (53.3% transfer-corrected).
- ⚠ `sStatus='Current'` does **not** mean active resident — it means final executed term, including tenancies that ended in 2018. 661 tenancies have a Current row; only 112 are occupied.

### `tenant_history` — the genuinely append-only event log

`sEvent` ∈ Lease Renewal / Notice Given / Cancel Notice / Move Out / Month to Month / Rent Change, with `sReason` and `sResult`. This is the right source for notice behavior and the validation counterpart to `Lease_History`.

### The `U=` GL convention

`GLDETAIL.SNOTES` unit tagging is a **free-text local practice at 62W, not a schema feature** — 85–97% coverage on core make-ready accounts, 0–5% on supply/utility accounts. It will very likely not exist, or will take a different form, at properties 62 / 72 / 26 / 473. Re-run the coverage check per property before Phase 4 and fall back gracefully to a property-level average.

### `RENTABLEITEMS` + `RentableItems_Status`

Populated and reliable for garage inventory, per-stall status, and vacancy-spell history. Status decode via the `dbo.RentableItemStatus` view: 10 Vacant-Unrented, 20 Notice-Unrented, 30 Vacant-Rented, 40 Notice-Rented, 50 Occupied-No-Notice.

---

## 9. Landmines for the implementer

**GL / make-ready**
1. **`IBOOK` is a duplicate-expense trap.** Cash (`IBOOK=0`) and Accrual (`IBOOK=1`) both carry the same expense ($210,658 vs $204,557). Filter `IBOOK=1`; summing both nearly doubles every number.
2. **`ACCT.SCODE` is space-padded `char`.** `WHERE a.SCODE IN ('52501002')` silently returns nothing depending on driver/collation. Always `LTRIM(RTRIM(...))`.
3. **`GLDETAIL` has no `HUNIT` and no populated `SSEGMENT*`** — the `SNOTES` parse is the only unit link. Balance-sheet contra rows (Accounts Payable, Interco Due To/From) also carry `U=` notes (5,034 rows), so you must filter to in-scope expense accounts, not just `SNOTES LIKE 'U=%'`.
4. **Invoice-date lag is material.** A strict `[move-out, move-in]` window captures only 62% of dollars and fabricates a 1–4% zero-cost share. Padding to `[move-out − 7d, move-in + 45d]` reaches 91% and 0%.
5. **Some in-scope lines are credits/reversals**, so a per-turn total can legitimately be ≤ 0. Do not filter `DAMOUNT > 0`.
6. One duplicate `(HUNIT, DTMOVEIN)` pair makes the 2BR turn count 145 in SQL vs 144 de-duplicated — immaterial to medians, but implementations will differ by 1.

**Garages**
7. **`RENTABLEITEMS.hTenant` joins to `TENANT.HMYPERSON`**, not `HMY`.
8. **Use live `RENTABLEITEMS.iStatus` for current state.** 3 of 137 stalls have dangling open status intervals from stall transfers, so the history reconstruction is biased ~+2pp high (agrees with live on 134/137).
9. **`RentableItems_Status` intervals overlap** — Yardi writes concurrent and zero-length correction rows. Resolve point-in-time with `TOP 1 … ORDER BY hMy DESC`. Do not use `dtDateOccurred`.
10. **Never average raw charge-17 amounts** — 16% are prorations/reversals. Filter to full-month amounts or take rent from `RENTABLEITEMS.CRENT`.
11. **Don't derive tier from the camrule** — 5 vacant $100 stalls point at $75 camrules. Ignore `hTenantReserved`/`hProspectReserved` (stale on 133 of 137 stalls).
12. `RentableItemsHistory`, `RentableItemsChargeXref`, `RentableItemsPropObj`, `RentableItemsTypeCategoryXref`, and `GlTenantLedgerHistory` are all **empty**. `DETAIL.hChargeCode` is NULL portfolio-wide — charge code lives in `TRANS.HRETENTIONACCT`.

**Uplift / labels**
13. **Acct-53 posting timing is a label leak** (median 18 days before the decision, 18% after the new term starts). Any "received concession → renewed" feature leaks.
14. **`TENANT` mutation, concretely:** `SRENT`, `DTLEASEFROM`, `DTLEASETO`, `ISTATUS`, `iLateCount`, `iNSFCount` are all current-state. Rebuild historical risk counters from TRANS charge types 8/9/57/61 with a date cutoff; take historical rent from `Lease_History.cRent`.
15. **Right-censoring inflates recent periods** — raw renewal rate rises 33% (2017) → 51% (2023) → 61.6% (2026) purely because recent move-outs haven't happened yet. Never report an uncontrolled trend.
16. **Off-lease concession postings are common** — 57% of acct-54 and 45% of acct-30 events fall outside any executed term (mostly pre-first-lease move-in credits). A "concession during the term" feature silently drops half the volume.
17. **`incentives.py` does not filter `VOID`/`VOIDED`.** Zero voided rows at 62W, but add the filter before rollout.
18. **62W's de-facto free month is acct 54, not 53.** Acct 54 is the largest bucket ($973k, 355 residents, median 1.03 months of rent) but 275 of 485 events post *before* the first lease starts — it is predominantly a move-in/lease-up concession. Both 53 and 54 carry ~1-month doses with mixed use, which the posting convention must reckon with.

---

## 10. Recommendation for Phase 1

**Proceed with `unit_economics.py` as designed.** All inputs are now measured, the numbers are defensible, and the $248k/year figure with a 68% 2BR concentration is a real business finding that stands on its own regardless of what happens to the concession work.

**Restructure Phase 2 before implementing.** The optimizer as specified would recommend offers whose effects the data cannot support. Minimum changes: score on h180; breakeven-only output with no assumed-uplift default; garage split into two products with a seasonal capacity constraint; drop 2- and 3-month free rent from the default grid (keep them visible with an explicit "cannot pay within one lease term" verdict); fire pre-notice only.

**Promote prospective measurement (§8.3) to the critical path.** It was scoped as Phase 3. With 480 notices at 62W and only 8 concession attempts against them in nine years, the flip rate is unmeasured because the play was never run — that is cheaply fixable going forward, and it is the only route to a defensible ROI number. Extend the outreach log with `offer_type` and `bedroom`, randomize offer choice within the band where volume allows, and carry accounts 50/51 as standing negative controls.

**Open questions from the design doc that Phase 0 resolved:** §10.1 rent basis (use `Lease_History.cRent` for historical, in-place for forward-looking), §10.2 relet netting (yes, optimizer uses net), §10.5 garage threshold (retired in favor of continuous cost), §10.6 3BR floor (fold into 2BR+ — n=7 supports neither a cost nor a rate estimate). Still open: §10.3 discount grid, §10.4 posting convention (now with the added timing requirement above).

---

## 11. Decision package (Tier B, added 2026-08-18 post-review)

Everything above says what the data can and cannot support. This section prices the three decisions actually on the table: fund a pilot (of what design), expect what return, and where concessions rank among retention levers. All flows measured from `Lease_History` executed terms, trailing 4 years; band stocks from a live h180 scoring pass across the portfolio.

### 11.1 Decision flow and band flow (measured, not estimated)

| Property | Decisions/yr | Renewal rate | Actives scored | Band stock (h180) | **Band flow/yr** |
|---|---|---|---|---|---|
| 26 Winwood | 387.5 | 61.3% | 404 | 41 | 39.3 |
| 62 CCV | 282.2 | 60.2% | 292 | 40 | 38.7 |
| 82 **62W** | 128.0 | 50.8% | 123 | 18 | 18.7 |
| 72 Sierra Pointe | 112.2 | 58.1% | 120 | 11 | 10.3 |
| 473 CCV Townhomes | 16.5 | 72.7% | 16 | 0 | 0.0 |
| **Portfolio** | **926** | 59.3% | 955 | 110 | **~107** |

Two load-bearing facts: the portfolio generates **~926 renewal decisions a year** (more than earlier rough estimates), and the persuadable band's flow is only **~107/yr portfolio-wide, ~19/yr at 62W**. Band mean churn probability is **0.26** at every property — not the 0.40 the worked examples used, which makes real-world breakevens *higher* than the illustrative tables (§6's p=0.30 column is the operative one).

### 11.2 Pilot power — the design decision

Two-proportion test, α = 0.05 two-sided, 80% power, ~55% baseline renewal. n per arm: **+5pp needs 1,534 · +10pp 376 · +15pp 163 · +20pp 89.** Years to reach total n at measured flows:

| Randomization pool | Flow/yr | Detect +5pp | +10pp | +15pp | +20pp |
|---|---|---|---|---|---|
| 62W, persuadables only | 19 | 164 y | 40 y | 17 y | 10 y |
| Portfolio, persuadables only | 107 | 29 y | 7.0 y | 3.0 y | 1.7 y |
| 62W, all decisions | 128 | 24 y | 5.9 y | 2.5 y | 1.4 y |
| **Portfolio, all decisions** | **926** | 3.3 y | **0.8 y** | **0.4 y** | 0.2 y |

**The conclusion is not close: a persuadables-only pilot cannot measure anything in a relevant timeframe, at any property count.** The only design that produces an answer within a year is randomizing across all portfolio renewal decisions (offer vs no-offer, or offer-A vs offer-B, assigned at decision entry). If offering outside the band is unacceptable (sleeping-dog cost on sure renewers), the honest fallback is portfolio-band randomization accepting ~3 years to certify a +15pp effect — and the interim readout is directional only. The post-notice rescue experiment (§3) runs in parallel for free: ~40 notices/yr portfolio-wide self-identify, need no targeting model, and the historical 3-of-15 hints the dose works there.

### 11.3 Program expected value — the ceiling, so $248k stops being the anchor

$248k/yr is the disease burden at 62W; the concession program addresses only the slice that is (a) in the band, (b) truly persuadable, (c) net of concession cost. EV per treated decision at the band's real mean p = 0.26, 2BR terms (TC_net $4,681):

| Offer | True uplift +5pp | +10pp | +15pp | +20pp |
|---|---|---|---|---|
| 3% discount ($600) | −$237 | −$33 | +$171 | +$375 |
| Garage new, $75 tier ($570) | −$214 | −$8 | +$198 | +$403 |
| Garage new, E[T]=2mo ($870) | −$449 | −$259 | −$68 | +$123 |
| 1 free month ($1,668) | −$1,076 | −$925 | −$774 | −$624 |

**Program EV at 62W, optimistic case (+15pp true uplift, cheapest offer, ~19 treated/yr): ≈ $3,200/yr.** Even at +20pp it is ≈$7k/yr. Portfolio-wide, scaling by band flow (~107/yr, with the caveat that TC and make-ready are 62W-measured and must be re-derived per property): **≈$18–40k/yr in the optimistic scenarios — an order of magnitude below the $248k anchor.** And the measured evidence (§3) puts the *likely* uplift in single digits, where every offer is EV-negative. One free month is EV-negative at every plausible uplift; it exists only as a v2 duration-horizon play or a post-notice rescue.

### 11.4 Where concessions rank among retention levers (62W per-year figures)

| Lever | Value/yr at 62W | Certainty | Notes |
|---|---|---|---|
| **Shave 10 days off average turn vacancy** | **≈$31,100** | Deterministic — no uplift assumption | 21 × 1BR + 36 × 2BR + 1.75 × 3BR turns × 10 days × daily rent. The target is the tail, not the median: 2BR median gap is 47 days vs mean 75 (p75 = 106, p90 = 152) — the money is in killing outlier holds. Audit what the p90 turns actually were (reno holds and strategic vacancy shouldn't be billed to churn). |
| **Expiration management — move lease-ends out of winter** | **up to ≈$24,000** | High (physics measured, capture partial) | Winter (Nov–Feb) move-outs take a median **78 days** to fill vs **32.5** for Mar–Jun; 62W averages 13.5 winter move-outs/yr × ~33 extra mean days × $55/day. 62W already writes no Nov/Dec lease-ends, but 25 current terms still end Jan–Feb. Same physics the garage seasonality proved. |
| **Concession program (this design), success case** | **≈$3,200–7,000** | Speculative — requires +15–20pp true uplift, which history cannot support | The number this document exists to size. |
| Renewal-increase optimization | unpriced | Needs elasticity measurement | The same randomization infrastructure the pilot builds can measure it. |
| Service quality / work-order response | unpriced | Correlational | Work-order recency features are top churn drivers in the model; causality unknown. |

**The honest ranking: the two operational levers are worth ~10× the concession program's optimistic case, are deterministic, and carry no fair-housing surface.** The strongest version of this program is therefore: ship the unit-economics table (it prices the ops levers too), run the cheap post-notice rescue experiment, randomize portfolio-wide if leadership wants the concession answer — and point the first real dollars at turn-cycle tail management and Jan–Feb lease-end migration.

---

## 12. Offer structure, market context, and governance (Tier C, added 2026-08-18)

### 12.1 A discount and a free month are not interchangeable dollars

The §6/§11 tables price offers on a one-year "cost if renewed" basis, which systematically flatters the rent discount. Multifamily offer structure has three mechanics the one-year view hides:

| Structure (2BR, $1,668 rent, 3% annual increases) | Year-1 cost | 3-year cost | Face rent | Base rent |
|---|---|---|---|---|
| **1 free month** (or garage waiver) | $1,668 | $1,668 | preserved | preserved |
| **3% discount, removed at next renewal** | $600 | $600 | degraded | restored — but creates a **renewal cliff** at month 13 (a reinstatement shock that defers churn risk rather than removing it) |
| **3% discount baked into base** | $600 | **$1,857 and compounding** | degraded | permanently lowered — every future increase computes off the discounted base |

**The "cheapest" offer inverts after ~2.8 years if the discount persists.** Discounted face rent also leaks into loss-to-lease reporting, revenue-management comps, and appraisal/lender NOI reads in a way a one-time credit does not. Consequence for the design: **pilot arms should prefer face-rent-preserving instruments** (free month, garage waiver) or an explicitly time-boxed discount with the cliff handled at renewal — and the optimizer's cost model must carry a `structure` field, not just a dollar amount.

**The multi-term counterweight (cuts the other way):** if a save retains k lease terms instead of one, every breakeven divides by roughly k. At the band's real p = 0.265 (2BR):

| Offer | k=1 | k=1.5 | k=2 | k=3 |
|---|---|---|---|---|
| 3% discount | 10.8pp | 6.9pp | **5.0pp** | 3.3pp |
| 1 free month | 40.7pp | 22.9pp | **15.9pp** | 9.9pp |

At k=2, the cheap offers' breakevens land *inside* the historically plausible single-digit range. **This is the strongest argument for validating the duration model (v2) before spending on a large pilot** — the multi-term view may change the verdict for free.

### 12.2 Market context — the regime this data was measured in

*(External figures verified 2026-08-18; session note filed in the vault.)*

- **Nationally, this is the most concessionary rental market on record:** 39.8% of Zillow listings offered a concession in April 2026 (record; more than double pre-pandemic), 39.7% in June; national rental vacancy 7.3% (Census, Q1 and Q2 2026) on the construction wave. The bloodbath is Sun Belt-concentrated (Denver 68%, Charlotte 67%, Dallas 64%).
- **Des Moines is tighter but supply is landing:** metro vacancy ~4.8–5.8% (vs 7.3% national), yet 6,000+ new units are being added and **new lease-ups are offering 1–2 free months**. That is the concession-arbitrage threat in concrete local terms: a leaver can capture at a competitor what we won't give at renewal — it raises baseline churn and competes directly with our retention offers.
- **Regime-dependence caveat:** the 9-year history behind every number here spans lease-up, COVID, the 2021–22 surge, and today's renter-lean market (period strata absorb this in the uplift analysis, but the *levels* — 47% turnover, 74-day 2BR vacancy, fill-time seasonality — are partly regime, and any measured uplift will be too). Re-baseline after the supply wave absorbs.
- **External benchmarks for credibility:** our net turnover cost ($3,347–$5,359) brackets the commonly cited $3,872 national all-in average (Zego 2026, via industry summaries), and our expected loss ($1,604–$2,203/unit-yr) sits above NAA's 2024 vacancy-and-rent-loss benchmark of $1,323/unit-yr — consistent with our ~47%/yr turnover running hot. Cited as secondary-source context, not audited.

### 12.3 Governance and fair housing — required before any pilot

This program hands **individually differentiated renewal terms to algorithm-selected residents**, using a model whose features include NSF counts, late fees, tenure, and pets — attributes that can correlate with protected classes. Before any offer is made:

1. **Counsel review** (Fair Housing Act + Iowa Civil Rights Act) of the targeting criteria and offer set; written, consistent-criteria documentation of who is eligible for what and why.
2. **Feature audit** of the h180 model for disparate-impact surface if it gates offers; if counsel prefers, randomize *eligibility windows or properties* rather than individuals.
3. **Post-RealPage-litigation posture:** concessions on our own portfolio are legally distinct from pooled price-setting, but anything describable as "an algorithm decides who gets a deal" invites scrutiny — one page of governance converts the landmine into a strength.
4. **The post-notice rescue trial is the cleanest instrument on every axis:** a uniform offer to a self-identified group (notice-givers), no model in the loop, randomized only for measurement. Start there.
5. **Behavioral contagion is a design constraint, not an afterthought:** residents talk. Targeted deals train renewal-brinkmanship and create fairness grievances (the 77% who watch a neighbor get a free garage). The pilot needs offer-handling scripts and an explicit leak-tolerance stance before launch.

### 12.4 What generalizes vs what is 62W-local

| Generalizes to 62/72/26/473 (method) | 62W-local (re-measure per property before use) |
|---|---|
| Posting-time tautology guard; negative-control calibration; treatment lag filters | Make-ready $791/$872/$1,094 — the `U=` GL convention is a 62W free-text habit; coverage must be re-checked per property |
| Stall-vs-payer inventory counting; `RentableItems_Status` point-in-time resolution | Garage inventory 137, tiers $75/$100, occupancy seasonality, attach rates |
| Interval-union occupied-unit-years; turn detection; relet netting | Turn days (60/74/61), relet %, unit mix, rents |
| `Lease_History` renewal derivation; transfer correction; power math | Band stock/flow, renewal rates (50.8% at 62W vs 72.7% at 473 — property character differs hugely) |
| Breakeven-uplift framing; offer-structure economics (§12.1) | Every dollar figure in this memo |

Anyone applying $4,681 to Winwood is misusing this document.

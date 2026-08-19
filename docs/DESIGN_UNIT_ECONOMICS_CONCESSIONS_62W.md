# Design: Per-Unit-Type Turnover Economics & Concession Optimization — Residences at 62W

**Status:** DESIGN ONLY (no implementation) · **Date:** 2026-08-18 · **Scope:** property 82 (Residences at 62W) exclusively — rollout to other properties only after this design is validated end-to-end at 62W.

> ### ⚠ Phase 0 discovery has superseded parts of this document (2026-08-18)
> Read **[PHASE0_DISCOVERY_62W_2026-08-18.md](PHASE0_DISCOVERY_62W_2026-08-18.md)** alongside this doc. It answers §5.3, §7.3 and §8.2 with live data and overturns three premises:
> - **Make-ready is measurable** — $600 flat → **$791 / $872 / $1,094** all-in per bedroom. §5.3, §5.4 numbers below are superseded.
> - **The "stagnant garage" premise is empirically FALSE at 62W** — 137 stalls, 93.0% mean 2026 occupancy, median 2 months to re-rent, zero stalls vacant over 12 months. The `C_garage ≈ $0` branch and the "free garage strictly dominates" conclusion in §6.4 / §7.2 are retracted. Garage rent is **$87 blended, not $74** ($74 was proration-contaminated).
> - **No usable observational uplift exists** — negative controls put the confounding floor at ±5–13pp, larger than several breakevens. §8.2 closes as answered-negative; do not wire an assumed-uplift default.
>
> Two further findings not anticipated here: the optimizer must score on **h180**, not the default h60 bundle (which yields only 3 persuadables at 62W vs 18); and the unit mix below counts 9 `EXCLUDE = -1` pseudo-units — the true residential mix is **47 / 84 / 6 = 137 units**.

## 1. Goal

Move turnover economics from **per-property** (what `persuadables.py` does today) to **per-unit-type (bedroom count)**, and use it as a market model for concession decisions. Three deliverables:

1. **Turnover cost per bedroom type** — what one turn actually costs us on a 1BR vs 2BR vs 3BR.
2. **Turnover rate per bedroom type, monetized from avg rent** — vacancy duration expressed in months of that bedroom's average rent (e.g., a 2-month turn on a $1,000 unit = $2,000 + one-time make-ready = $2,600).
3. **Expected loss** — churn probability × turnover cost, per resident and rolled up per bedroom class.

On top of that, a **concession optimizer** answering: *"What concession should we give this tenant to maximize NOI?"* over three offer families:

- **N free months of rent** (N ∈ {1, 2, 3} — parameterized, 3 is the max, per 2026-08-18 decision)
- **Free garage** (also yields the garage's own turnover/stagnancy economics)
- **Discounted rent** (d% off for the lease term)

## 2. Decisions already made (2026-08-18, Brandon)

| Decision | Choice |
|----------|--------|
| Free-months concession | Parameterize 1–3 months; optimizer evaluates each as a separate offer |
| NOI horizon | One 12-month lease term now; duration-model expected-tenure horizon designed in as v2 |
| Make-ready cost per bedroom | Mine actuals from Yardi GL / work-order costs per turn, grouped by bedroom; fall back to scaled flat fee if data is thin |
| "Turnover rate (from avg rent)" | Vacancy duration in months × avg rent for that bedroom + one-time make-ready fee = turnover cost |
| First property | 62W only (property 82) |

## 3. What already exists (build on, don't rebuild)

| Asset | Where | What it gives us |
|-------|-------|------------------|
| Per-property turn economics | `contracts/multifamily_churn/persuadables.py` | `_turn_days()` (TENANT unit-vacancy gaps, 0–365d, last 4y, 24h cache), `_relet_premium()` (consecutive-tenancy rent delta, net of renewal increase), value-if-saved formula, scenario math with assumed flip rates |
| Calibrated churn probability | churn contract (`v6-roadmap-h60` serving; v7 age experiment pending) | per-resident `p_churn` at the next lease decision |
| Duration model | `bundles/multifamily_duration/` (INTERNAL, report-only; backtest MAE 2.1 mo, Spearman 0.90) | expected remaining tenure — the v2 NOI horizon |
| Concession ledger | `contracts/multifamily_churn/incentives.py` | TRANS credits on 8 concession accounts (consumed only); acct 53 = Renewal Concession, acct 30 = Garage Concession |
| Offer log | `contracts/multifamily_churn/outreach.py` | `incentive_offered/accepted/declined` actions — the future flip-rate denominator |
| Bedroom-grain vacancy SP | `sp_GetVacancyByPropertyBedroom` (referenced in CONTEXT_MULTIFAMILY §1) | existing bedroom roll-up precedent in the DB layer |

## 4. Grounded data facts — 62W (live queries, 2026-08-18)

### Unit mix and rents (`UNIT.IBEDROOMS`, active leases `TENANT.ISTATUS=0`)

| Bedroom | Units *(as first queried)* | Rentable (`EXCLUDE = 0`) | Active residents | Avg in-place rent | Avg sqft |
|---------|-------|-----|------------------|-------------------|----------|
| 1BR | 50 | **47** | 37 | $1,495 | 799 |
| 2BR | 88 | **84** | 69 | $1,813 | 1,073 |
| 3BR | 7 | **6** | 6 | $2,332 | 1,511 |

⚠ **Use the rentable column.** Phase 0 found `UNIT.EXCLUDE = -1` marks 9 pseudo-units at 62W — the `GARAGE 10` placeholder (0BR, $236) plus seven `WAIT*` waitlist units — and they hold 20 real TENANT records. True residential mix is **47 / 84 / 6 = 137 units**; filter `EXCLUDE = 0` everywhere. The `GARAGE` row is a placeholder, **not** the garage inventory (which is 137 stalls in `RENTABLEITEMS` — see §7).

### Turn history, last 4 years (TENANT consecutive tenancies per unit, gaps 0–365d)

| Bedroom | Turns | Avg vacant days | Avg relet delta |
|---------|-------|-----------------|-----------------|
| 1BR | 84 | 58.2 | +4.7% |
| 2BR | 144 | 75.0 | +4.3% |
| 3BR | **7** | 41.9 | +3.2% |

⚠ 3BR is statistically thin (7 turns / 7 units). All per-bedroom estimates get empirical-Bayes shrinkage toward the property mean (§5.1).

### Garages

- Garage rent = **charge type 17 ("Garage")**: 4,367 charge rows since 2024-01, 211 distinct residents billed, ~**$74/month**.
- **Garage Concession (TRANS acct 30)**: $265k over 297 events / 179 residents historically (2017–2023 heavy use), nearly dormant since 2024 (3 events). We have organizational precedent for this concession — and its historical recipients are an observational dataset (§8).
- Related charge types: 70 "Garage Deposit", 112 "Garage Income 2".

## 5. Metric definitions (the unit-economics table)

All metrics computed per bedroom class `b ∈ {1, 2, 3}` at 62W.

### 5.1 Vacancy duration, shrunk

```
raw_turn_days_b   = avg move-out → next move-in gap on same unit, bedroom b, last 4y
turn_days_b       = (n_b × raw_turn_days_b + k × property_turn_days) / (n_b + k),  k = 20
vacancy_months_b  = turn_days_b / 30.44
```

Shrinkage matters only for 3BR (n=7): raw 41.9d shrinks to ≈61d against the property-weighted mean of ≈68d. 1BR/2BR are barely moved. Report both raw and shrunk, with n.

### 5.2 Average rent

Default basis: **in-place rent** (avg `TENANT.SRENT`, active leases) — that's the revenue we actually lose during vacancy on a like-for-like unit. Market rent (`UNIT.SRENT`) reported alongside for reference. (Open question §10.1.)

### 5.3 Make-ready cost per bedroom (Phase 0 discovery)

Mine actuals: for each turn event (move-out → next move-in on a unit), sum turnover-tagged costs incurred in the vacancy window:

- **Primary source:** Yardi GL — turnover/make-ready expense accounts posted against the unit/property in the vacancy window. Requires a GL account census (which accounts are "turn costs": paint, carpet/flooring, cleaning, maintenance make-ready) — same census exercise as the 2026-07-21 concession-account census.
- **Secondary source:** work orders closed on the unit during the vacancy window with cost fields populated.
- **Aggregation:** median per turn, grouped by bedroom; median (not mean) because turn costs are heavy-tailed (occasional full renovations).
- **Fallback when thin:** `$600 × (sqft_b / property_avg_sqft)` — i.e., ≈$450 (1BR) / $630 (2BR) / $890 (3BR) at 62W — clearly labeled `method: "sqft-scaled fallback"`.

### 5.4 Turnover cost (the headline number)

```
turnover_cost_b (gross) = vacancy_months_b × avg_rent_b + make_ready_b
```

This is exactly the agreed convention ($1,000 rent, 2-month turn → $2,000 + $600 = $2,600).

**Relet-premium netting:** `persuadables.py` deducts the net re-lease premium (relet delta minus the renewal increase we'd have taken anyway, over 12 months) from the value of a save. At 62W the relet delta is +4.3–4.7%, so netting is material (≈$500–900/turn against a 12-mo term). To stay consistent across surfaces, the table reports **both**: `turnover_cost_gross` (the agreed convention) and `turnover_cost_net_relet`. The **optimizer uses net** — otherwise it will systematically over-recommend concessions. (Flagged §10.2 for sign-off.)

**SUPERSEDED — Phase 0 measured make-ready and the proper occupied-unit-years denominator. Current numbers:**

| Bedroom | Vacancy months | × avg rent | + make-ready (measured) | Gross | Net of relet | Turnover rate | Annual property cost |
|---------|---------------|------------|--------------|-------|--------------|---------------|------|
| 1BR | 1.97 | $2,866 | $791 | $3,662 | **$3,347** | 47.9%/yr | $70,294 |
| 2BR | 2.44 | $4,070 | $872 | $4,936 | **$4,681** | 47.1%/yr | $168,515 |
| 3BR | 2.01 (shrunk) | $4,687 | $1,094 | $5,785 | **$5,359** | 29.5%/yr | $9,378 |

**Total ≈$248,187/year in net turnover cost at 62W, 68% of it in 2-bedrooms.** Full derivation and sensitivity in the Phase 0 memo §5.

### 5.5 Turnover rate and expected loss

```
turnover_rate_b       = (turns_b / 4) / occupied_unit_years_b     # annualized; denominator from
                                                                   # occupancy history, not raw unit count
expected_loss_resident = p_churn × turnover_cost_b                # per resident, this decision window
expected_loss_b        = Σ residents in b                          # per bedroom class
expected_loss_unit_yr  = turnover_rate_b × turnover_cost_b        # steady-state per unit-year
```

Measured figures (Phase 0, per-unit interval-union denominators): 1BR 47.9%/yr → $1,604/unit-yr; 2BR 47.1%/yr → $2,203/unit-yr; 3BR 29.5%/yr → $1,582/unit-yr. **1BR and 2BR turn over at nearly the same rate** — the 2BR cost concentration (68% of the property's ≈$248k/yr) is unit mix plus cost-per-turn, not worse retention behavior.

## 6. The concession optimizer

### 6.1 Decision frame

For each **persuadable** resident (calibrated `p_churn ∈ [0.20, 0.60)`, no notice — same band as `persuadables.py`; below the band = sleeping dogs, above = mostly lost), evaluate the offer set:

```
OFFERS = { none,
           free_months(n)   for n in {1, 2, 3},
           free_garage,                       # 12-month term
           rent_discount(d) for d in {3%, 5%, 10%} }   # grid — open question §10.3
```

### 6.2 Offer cost model (12-month term, v1 horizon)

A retention concession is only paid **if the resident renews**, so expected cost = `(1 − p′) × C_offer` where `p′` is post-offer churn probability.

| Offer | Cost if renewed `C_offer` | Notes |
|-------|--------------------------|-------|
| `free_months(n)` | `n × rent` | Post to acct 53 (Renewal Concession) so history stays measurable — §10.4 |
| `rent_discount(d)` | `d × rent × 12` | Same posting convention |
| `free_garage` | `12 × garage_rent (≈$888)` **if garages are scarce**; `≈$0 marginal` if a garage would otherwise sit stagnant | The two-regime model — §7 |

### 6.3 NOI comparison

Let `TC = turnover_cost_net_relet_b`, `p = p_churn`, `p′ = p − Δp(offer)` where `Δp` is the offer's churn reduction (uplift). Relative to doing nothing:

```
E[net gain | offer] = Δp × TC − (1 − p′) × C_offer
```

**The uplift Δp is the one thing we cannot compute from data today** (TRANS records consumed concessions, not offers; no control group). The design handles this honestly with three output layers:

1. **Breakeven uplift** (assumption-free, ship first):
   `Δp* = ((1 − p) × C) / (TC − C)` — "this offer pays iff it cuts this resident's churn probability by ≥ Δp* points." Rank offers by breakeven; low breakeven = safe bet.
2. **Scenario mode** (assumed Δp as labeled input — same convention as `persuadables.py` `flip_rate`).
3. **Measured mode** (v2, once the outreach log accumulates offers — §8).

### 6.4 Worked example — 2BR persuadable, p = 0.40, TC_net $4,681 (Phase 0 numbers)

| Offer | Cost if renewed | Breakeven Δp | Verdict |
|-------|-----------------|--------------|---------|
| 1 free month | $1,668 | **33 pts** | Cannot be certified — measured effect is single digits |
| 2 free months | $3,336 | ~148 pts | Never pays |
| 3 free months | $5,004 | impossible (cost > TC) | Never pays on a 12-mo horizon |
| 3% rent discount | $600 | **8.8 pts** | Cheapest cash lever |
| 5% rent discount | $1,001 | 16 pts | Plausible only for high-p residents |
| Garage **waiver** (77% of residents already rent one) | $1,044 | 17 pts | Expensive — full price, no occupancy offset |
| Garage **new**, $75 tier, trough season | $570 | **8.3 pts** | Best garage variant, capacity-capped at ~21 residents |
| ~~Free garage (stagnant)~~ | ~~≈$0~~ | ~~≈0 pts~~ | **RETRACTED — premise false (§7)** |

Two design consequences fall straight out:

- **Multi-month free rent cannot pay within one lease term at 62W economics.** It only makes sense under the v2 duration-model horizon (a save worth 2–3 expected terms). The optimizer must say this explicitly rather than quietly never recommending it. *(Phase 0 confirms this.)*
- ~~**The stagnant garage is the standout cheap lever.**~~ **RETRACTED.** Phase 0 found no stagnant garages at 62W: 93.0% mean 2026 occupancy, median 2 months to re-rent, zero stalls vacant over 12 months. The cheapest levers are instead a **3% rent discount (8.8 pts)** and a **new $75-tier garage assignment in trough season (8.3 pts)** — and both sit inside the ±5–13pp confounding floor, so neither can be certified from history.

### 6.5 v2 horizon: expected tenure

Replace the single-term save value with `E[extra_months | saved]` from the duration model (trait variant for ranking, full variant for timing): `value_of_save = Σ over expected extra terms of (turnover cost avoided, discounted)`. Design hook only — the duration model is INTERNAL/report-only and stays that way until validated in production.

## 7. Garage economics (the stagnancy question)

> **ANSWERED by Phase 0 — and the answer is no.** 62W has **137 stalls** (`RENTABLEITEMS`, fixed since 2017, one per rentable unit). Occupancy is **84.7% today but 96.4% in Jan–Mar 2026** and 93.0% for the year — garages are a winter good in Iowa, so a static threshold detects seasonality, not scarcity. **No stall has sat vacant over 12 months**; median vacancy is 1.2 months and median time to re-rent is 2 months across 259 spells. Rent is two tiers, **$75 × 70 and $100 × 67, blended $87** (the $74 below was proration-contaminated). And **77% of residents already rent a garage**, so the offer splits into a cheap new-assignment (capacity-capped at ~21 residents, 4–6 in winter) and an expensive waiver. Replace the binary switch with `C_garage ≈ (12 − E[months_to_re_rent]) × tier_rent` ≈ $661–$870. The subsections below are retained for design lineage; their conclusions are superseded.

**Question being answered:** if garages sit stagnant, is it beneficial long-run to use them as retention currency instead of leaving them empty?

### 7.1 Garage "turnover cost"

Effectively **≈$0**: no make-ready, no unit-vacancy revenue loss beyond the forgone $74/mo. A garage is not a unit — its cost model is pure opportunity cost.

### 7.2 Two-regime marginal cost

```
garage_occupancy = active charge-type-17 payers / garage inventory
scarce   (occupancy ≥ threshold, e.g. 90%):  C_garage = 12 × $74 ≈ $888   (real forgone revenue)
stagnant (occupancy < threshold):            C_garage ≈ $0 marginal        (it earns nothing anyway)
```

In the stagnant regime the free garage strictly dominates every cash concession: any Δp > 0 is NOI-positive. The long-run answer to "leave it stagnant?" is then: **a stagnant garage's highest-value use is as a retention concession**, with an option cost of re-renting it later (report months-stagnant per garage so we don't give away one that's about to rent).

### 7.3 Phase 0 discovery required

- **Garage inventory count at 62W is unknown.** The UNIT table has one placeholder row. Likely lives in Yardi rentable items; alternatively derive an inventory floor from max concurrent charge-type-17 payers. Occupancy denominator blocks the regime switch.
- Confirm $74/mo (blended avg — check for size tiers).
- Historical garage-concession recipients (acct 30, 179 residents at 62W) — did their renewal rates beat matched non-recipients? First observational uplift estimate for exactly this offer (§8).

## 8. Measuring uplift (from assumption to measurement)

> **Phase 0 result: step 2 is closed as answered-NEGATIVE.** The observational history cannot adjudicate any breakeven in §6. The acct-53 headline (+54pp) is a posting-timing tautology — the credit is booked ~18 days before the decision, i.e. when the renewal is signed. The one leakage-free estimate is +10 to +12pp on the *next* term, CI [−4.9, +23.4], and entitlement accounts 50/51 (true effect exactly zero) measure +6 to +13pp under the same estimator — so the confounding floor is **±5–13pp**, larger than the garage breakeven. Also: **notice is almost terminal at 62W** (480 notices, 13 renewed = 2.7% base; 15 discretionary concession attempts against notice in nine years, of which 3 renewed — 20%, one explicitly documented) — too thin for a flip rate, so the optimizer fires pre-notice as the primary play, though post-notice rescue at a ~1-month dose is undisproven and cheap to test. Step 3 is now the critical path.

1. **Now — breakeven-uplift reporting.** No assumptions; turns "should we?" into "do we believe ≥Δp*?"
2. **Now — observational bounds from history.** 62W has 179 garage-concession recipients and years of acct-53 renewal concessions. Compare subsequent renewal vs matched non-recipients (same bedroom, tenure band, rent band, ± notice status). Selection-biased (concessions went to squeaky wheels or likely-renewers) — report as bounds with the bias direction stated, never as causal truth.
3. **Prospective — the outreach log.** Extend `incentive_offered/accepted/declined` logging with `offer_type` (`free_months_1/2/3`, `free_garage`, `rent_discount_pct`) and `bedroom`. Accumulates real flip rates per offer type per bedroom. If volume allows, randomize offer choice within the persuadable band (even partial randomization de-biases).
4. **Posting discipline.** Free months / discounts → acct 53; garage → acct 30. Consistent posting keeps TRANS usable as the consumed-side record. (§10.4)

## 9. API & module design (no implementation yet)

```
contracts/multifamily_churn/
  unit_economics.py     # per-bedroom turnover table: turn days (raw+shrunk), vacancy months,
                        # avg rent (in-place + market), make-ready (mined | fallback),
                        # turnover_cost gross+net, turnover_rate, expected_loss, n's, methods
  concessions.py        # optimizer: per-resident offer table w/ breakeven uplift, scenario
                        # mode, garage regime switch, portfolio roll-up
```

- `GET /predict/multifamily-churn` views: `unit-economics` `{property_id: 82}` and `concession-optimizer` `{property_id | scode, offers?, assumed_uplift?, garage_occupancy_threshold?}`.
- Reuse `persuadables.py` caching patterns (24h TTL, degrade-to-fallback, never 500 on economics).
- Every number carries `method` + `n` — same honesty convention as the persuadables economics block.
- Frontend (Streamlit) panel after the API view stabilizes.

## 10. Open questions

**Resolved by Phase 0:**
1. ~~**Rent basis**~~ → use `Lease_History.cRent` for historical rent (the point-in-time value `TENANT.SRENT` cannot give); in-place avg for forward-looking.
2. ~~**Relet netting**~~ → yes, the optimizer uses net-of-relet; the headline still reports both.
5. ~~**Garage scarcity threshold**~~ → retired in favor of the continuous `(12 − E[T_re-rent]) × tier_rent` cost.
6. ~~**3BR floor**~~ → fold into "2BR+". n=7 supports neither a cost nor a rate estimate (individual turns range $262–$4,984).

**Still need Brandon's call:**
3. **Rent-discount grid:** 3/5/10% — right options? Phase 0 makes 3% the cheapest cash lever at 8.8pp breakeven; is a 3% discount even meaningful to a resident, i.e. would anyone stay for $50/month?
4. **Posting convention:** Phase 0 adds a timing requirement beyond account choice — post the credit at the *new term's* start with an explicit offer date, and log the offer in the outreach log at offer time. Note that 62W's de-facto free month is currently acct **54**, not 53 ($973k / 355 residents / median 1.03 months of rent), and 275 of 485 acct-54 events post before the first lease even starts, so it is mostly a move-in concession. Which account should the retention offer use going forward?
7. *(new)* **Garage offer scope:** do we offer the waiver to the 77% who already rent a garage (17pp breakeven, full price, no occupancy offset), or restrict the offer to new assignments from the vacant pool (8.3pp, but capped at ~21 residents and only 4–6 in winter)?
8. *(new)* **Pilot without a certified uplift:** given that no offer can be certified from history, do we run a deliberately randomized pilot at 62W to *create* the measurement, accepting that ~18 persuadables per season means a multi-property, multi-season horizon before the flip rate is credible?

## 11. Phasing

| Phase | Work | Output | Status |
|-------|------|--------|--------|
| **0 — Discovery** | GL/WO make-ready census per turn per bedroom; garage inventory + occupancy at 62W; garage rent tier check; observational uplift pull (acct 30/53 recipients vs matched) | Data memo answering §5.3, §7.3, §8.2 | ✅ **DONE** — [PHASE0_DISCOVERY_62W_2026-08-18.md](PHASE0_DISCOVERY_62W_2026-08-18.md) |
| **1 — Unit economics** | `unit_economics.py` + API view for 62W | The per-bedroom table (§5) | Ready — all inputs measured, proceed as designed |
| **2 — Optimizer** | `concessions.py` + API view; breakeven-only output; garage split into new-vs-waiver with seasonal capacity constraint; h180 scoring; pre-notice firing | Per-resident offer recommendations (§6) | **Needs restructuring first** — see Phase 0 §10 |
| **3 — Measurement** | Outreach-log `offer_type` + `bedroom` extension; **portfolio-wide all-decisions randomization** (Phase 0 §11.2: band-only pilots are unpowered — 62W band needs 40 years to detect +10pp; portfolio all-decisions needs ~10 months); post-notice rescue experiment in parallel; accts 50/51 as standing negative controls | Assumption → measurement (§8) | **Promoted to critical path** — the only route to a defensible ROI number |
| **4 — v2 horizon** | Duration-model expected-tenure NOI | Multi-term NOI (§6.5) | Unblocks multi-month free rent, which cannot pay on a 12-mo horizon |
| **5 — Rollout** | Generalize property_id; re-run Phase 0 discovery per property (the `U=` GL convention is a 62W local practice and may not exist elsewhere) | 62, 72, 26, 473 | Also a measurement requirement — 18 persuadables at one property is too few |

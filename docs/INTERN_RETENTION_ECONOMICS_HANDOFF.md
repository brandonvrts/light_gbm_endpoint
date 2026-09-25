# Intern Handoff — Retention Economics & the Concession Decision

**Date:** 2026-09-23, revised 2026-09-25 (endpoint + snapshots) · **Audience:** economics intern (NOI, cost-risk, trial design) · **Model/repo owner:** Brandon Valadez
**Companion docs (read in this order):** `PHASE0_DISCOVERY_62W_2026-08-18.md` (what history can and cannot tell us), `DESIGN_UNIT_ECONOMICS_CONCESSIONS_62W.md` (the original design; parts superseded), `MULTIFAMILY_COHORTS_HANDOFF.md` (the API you will call), and the research baseline in the vault (`outputs/multifamily-prediction/reports/2026-08-20-retention-decision-science-baseline.md`).

> **Split of work.** You own the economics: turning churn probability, turnover rate and turnover cost into an NOI answer, a cost-risk analysis, and trial designs that can actually be measured. Brandon owns the model and this repo: scores, ledgers, logging, endpoints. §1 says exactly what crosses the boundary in each direction.

---

## Start here — access, quickstart, attachments

**Access**

| | |
|---|---|
| API (LAN) | `http://10.1.6.17:8910` — header `X-API-Key: rts-local-dev-key`; interactive docs at `/docs` |
| Your endpoint | `GET /multifamily/concession-cohort` (§1b) — JSON or `format=csv` |
| Repo | `~/Documents/spark/light_gbm_endpoint` on the DGX Spark (10.1.6.17); the economics modules are `contracts/multifamily_churn/{concession_cohort,unit_economics,persuadables,incentives,outreach}.py` |
| Yardi | read-only SQL Server (`Yardi7_Reporting`); ask Brandon for credentials when you reach the GL / notice-flow tasks in §9 — everything in weeks 1–2 except those runs off the API |
| Tools | Python (pandas) or Excel both work; the CSV is flat, one row per resident, 133 columns |

**First hour**

1. Open `concession-cohort-2026-09-25.csv` (attached). Sort by `economics.expected_loss`. That is the cohort.
2. Pick one row and reproduce its four `breakeven_uplift_pp` values by hand from `model.churn_probability`, `rent.monthly_rent`, `economics.turnover_cost_net` and §3.2 — §5.5 walks through resident `t0018814`. If your numbers match, you own the math.
3. Reproduce the §2 ledger table from `unit-economics-2026-09-25.json` (attached), then call the live endpoint and confirm nothing has moved.
4. Read Phase 0 §3 and §11 (why history cannot give you an uplift, and why a band-only pilot cannot either). Do not re-derive an observational uplift — that question is closed.
5. Call the endpoint yourself with `band_low=0.3`, then `assumed_uplift=0.10`, and watch what changes.

**Attachments (frozen 2026-09-25, `as_of=2026-09-25`)**

| File | What it is |
|---|---|
| `concession-cohort-2026-09-25.csv` | the cohort, flattened — 150 residents, 133 columns |
| `concession-cohort-2026-09-25.json` | the same call, nested, with `cohort` / `offers_summary` / `context` roll-ups |
| `concession-cohort-top-p30-2026-09-25.csv` | `band_low=0.3` — the 34 residents the model is most sure about |
| `unit-economics-2026-09-25.json` | the Phase 1 turnover ledger for all five properties (§2) |

Keep these with your workbook; the live cohort moves as leases roll and scores refresh.

**What the cohort looks like today (0.20–0.60, decision ≤ 180 days, no notice)**

| | n | Expected move-outs | Expected loss | Rent at risk / mo |
|---|---|---|---|---|
| **Portfolio** | **150** | **41.6** | **$107,566** | $54,732 |
| Winwood | 63 | — | $31,327 | |
| CCV | 41 | — | $23,184 | |
| Sierra Pointe | 18 | — | $18,739 | |
| 62W | 26 | 7.7 | $33,342 | |
| Townhomes | 2 | — | $974 | |
| 1BR / 2BR / 3BR | 72 / 77 / 1 | | $45,043 / $61,680 / $843 | |

| Offer | Applicable | Cost if all renew | Median breakeven | Best case | Verdicts |
|---|---|---|---|---|---|
| One month free | 150 | $197,065 | 90.8pp | 30.2pp | 150 red |
| 5% off (term-limited) | 150 | $118,239 | 36.8pp | 14.9pp | 1 amber, 149 red |
| Garage — new assignment | 88 | $65,910 | 41.5pp | 12.7pp | 6 amber, 82 red |
| Garage — waiver | 60 | $62,580 | 52.1pp | 12.5pp | 1 amber, 59 red |

Nothing is green. That is the starting point, not a failure: it is what §0 predicts, and it is why the work in §6–§9 (multi-term horizon, post-notice, lease-end migration, trials) is where the answer lives.


## 0. The one-page answer

**The question.** A resident is coming up for a renewal decision. We know their churn probability `p`, and we know what losing them costs (`TC_net`, per bedroom, per property). Should we offer a concession — a free garage, one month free, or 5% off — and which one?

**The rule.** An offer with cost `C` (paid only if the resident renews) is justified when

```
C  ≤  C_max  =  Δ · TC_net / (1 − p + Δ)
```

where `Δ` is the uplift — the reduction in churn probability — we are willing to believe the offer produces. Equivalently, the offer's **breakeven uplift** `Δ* = (1 − p) · C / (TC_net − C)` must be at or below `Δ`. Every number except `Δ` is measured. **`Δ` is the one thing we do not know, and history cannot tell us** (Phase 0 §3: the observational estimate is single digits inside a ±5–13pp confounding floor).

**What the rule says today, with `Δ` held to what history supports (≤ 5–10pp):**

| Situation | Verdict |
|---|---|
| **Pre-notice, one lease term, any property** | **No offer clears.** At `Δ = 10pp` a band resident (`p ≈ 0.26`) justifies spending ~12% of `TC_net`: **$607** on a 62W 2BR, **$223** on a Winwood 2BR. One free month ($1,065–2,332) and 5% off ($580–1,399) never fit. Only a new garage assignment at 62W/Sierra 2BR (~$750) is close, and only in the amber zone. |
| **Post-notice** (resident has filed to leave, `p ≈ 0.97`) | **One free month is justified without new evidence.** Breakeven **1.5pp** at 62W 2BR, because almost no one who gets the offer would have renewed anyway — the "sleeping dog" cost that kills pre-notice offers vanishes. Worst case (offer does nothing): ~$49 per offer. This is trial #1. |
| **Winwood / CCV / Townhomes** | Turns cost $1,800–2,100 (fast fills, low rent). **No concession pays on one term at any plausible uplift.** Their retention money is operational: vacancy tail and winter fill (§6). |
| **62W / Sierra 2BR** | Turns cost $4,900–5,100. This is where the concession conversation is real — and still only for cheap offers, multi-term saves (§5.3), or offers that also buy something deterministic (lease-end migration, §6C). |

**So the "jump" from our numbers to a justification is not "the offer pays." It is one of three honest statements:** (1) *"this offer pays if it moves churn by ≥ Δ\* points — do we believe that?"*, (2) *"this offer pays because the population's p makes the wasted-spend term tiny"* (post-notice), or (3) *"this offer pays because it also buys a deterministic gain"* (lease-end migration). Anything else needs a trial first, and your job is to make those trials answerable.

---

## 1. Roles and interfaces

| Model side provides (Brandon, this repo) | You produce (economics) | Flows back to the model side |
|---|---|---|
| **`GET /multifamily/concession-cohort`** — your primary surface (§1b): every band resident with lease/rent/tenure, the h180 `p` + drivers, garage status, incentive history, the full model input, `TC_net` and per-offer breakevens, JSON or CSV. Underlying pieces: `/multifamily/persuadables`, `/multifamily/edge-list`, `POST /predict/multifamily-churn` | **NOI bridge** per property: baseline turnover loss → with-program, as a function of `Δ` (§7) | Your `Δ_believed` policy recommendation (what leadership is willing to act on) |
| **Unit-economics ledger**: `TC_gross`, `TC_net`, turnover rate, expected loss per bedroom × property (`/multifamily/unit-economics`; §2) | **Cost-risk grid**: `Δ ∈ {0,5,10,15,20pp}` × offer × population × property; the `Δ = 0` row is the budget at risk (§7.3) | Measured make-ready per property for 26/62/72 (replaces the $600 fallback) |
| Monthly expiring cohorts with outcomes + month-start scores (`/multifamily/churn-report`, `/churn-trend`) | **Verdict tables** refreshed from the live ledger; the `C_max` policy memo (§5) | Garage inventory, tiers, occupancy per property (`RENTABLEITEMS`) |
| Consumed concessions from Yardi TRANS (`/multifamily/incentives`) and the typed **offer log** (`/multifamily/outreach`) | **Pre-registered trial designs**: post-notice rescue; Jan–Feb lease-end migration; (optional) all-decisions (§6) | Measured uplift by segment, once a trial reports — the input to any future optimizer |
| Calibration drift log (`/multifamily/calibration-drift`) | **Monthly realized turn-cost series**, economic calibration, cohort scorecard (§8) | The "addressable share" mapping of drivers → lever type (§3.4) |

Base URL `http://10.1.6.17:8910`, header `X-API-Key: rts-local-dev-key`, interactive docs at `/docs`. All Yardi access is **read-only**; anything we write (offers, arms, outcomes) lives in `monitoring/*.jsonl` in this repo.

---

## 1b. Your primary surface — `GET /multifamily/concession-cohort` (added 2026-09-25)

One call returns the modeling set: residents the model puts at real risk at their next lease decision, with every input the offer math needs on the same row. No names — `scode` / `hmyperson` / `unit` / `property` only. `hmyperson` is the Yardi person key; it joins `Lease_History.hTent` and `RENTABLEITEMS.hTenant` directly.

```bash
# JSON, default cohort (churn p 0.20-0.60 on h180, decision within 180 days, no notice)
curl -s -H "X-API-Key: rts-local-dev-key" "http://10.1.6.17:8910/multifamily/concession-cohort"

# one property, as CSV for Excel; band_low=0.3 keeps only the highest-risk ~35 portfolio-wide
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/concession-cohort?properties=82&format=csv" -o cohort-62w.csv

# scenario mode: adds expected_net / pays per offer at an ASSUMED 10-point uplift (labeled)
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/concession-cohort?assumed_uplift=0.10"
```

| Parameter | Default | Meaning |
|---|---|---|
| `band_low`, `band_high` | `0.20`, `0.60` | churn-probability band `[low, high)` at the next lease decision, h180 bundle, unclipped — the persuadable band agreed 2026-07-21, shared with `/persuadables` and the edge roster |
| `decision_window_days` | `180` | keep residents whose lease ends within N days (holdovers, negative, stay in); omit for none |
| `include_notice` | `false` | notice-filed residents are a different population (p ≈ 0.97); include to see them |
| `properties` | all 5 | comma-separated HPROPERTY ids (62 CCV, 72 Sierra, 82 62W, 26 Winwood, 473 Townhomes) |
| `as_of` | today | point-in-time rescoring for reproducible snapshots (`YYYY-MM-DD`) |
| `assumed_uplift` | none | `0–1`; adds `expected_net_at_assumed_uplift` / `pays_at_assumed_uplift` per offer and a `recommended_offer_at_assumed_uplift` — scenario math, labeled |
| `garage_months_to_rerent` | `2.0` | horizon in the garage-new cost `(12 − E[T]) × stall rent` (Phase 0 median at 62W) |
| `top` | all | trims the displayed rows only — `cohort` and `offers_summary` roll-ups always cover the whole cohort |
| `format` | `json` | `csv` flattens nested blocks to dotted columns (`economics.offers.free_months_1.breakeven_uplift_pp`, `features.TenureMonths`, …) |

**Why the default band is 0.20–0.60 and not 0.60–0.70.** "A 30–40% chance of staying" is a churn band of 0.60–0.70, and on this model it is **empty**: without a notice on file the h180 bundle never exceeds ~0.45 (no-notice 99th percentile 0.39 on 2026-09-25; one resident at 0.75). Notice is the dominant feature; once it is filed the probability jumps to ~0.95 and the resident belongs to the post-notice population (§6B), not this one. So the default is the persuadable band the rest of the API already uses — 0.20–0.60, ~150 residents with a decision inside 180 days, band mean p ≈ 0.26. `band_low=0.3` keeps the ~35 the model is most sure about. The response's `population` block always reports how many sat below/above the band and how many the window or notice filter removed, so you can see what a sweep is doing.

**What each row carries**

| Block | Fields |
|---|---|
| identity | `scode`, `hmyperson`, `unit`, `property_id`, `property`, `bedrooms`, `sqft` |
| `lease` | `lease_from`, `lease_to`, `days_to_lease_end`, `holdover`, `lease_term_months`, `decision_number`, `tenure_months`, `move_in`, `has_notice` |
| `rent` | `monthly_rent` (in-place), `prior_rent`, `rent_increase_pct`, `market_rent` (unit), `rent_vs_market_pct` |
| `model` | `churn_probability` (h180, unclipped), `stay_probability`, `risk_tier`, `why`, `drivers[]` (feature, direction, weight, value — from the h180 booster itself) |
| `cohorts` | `risk_cohorts[]` / `protective_cohorts[]` memberships with measured lift (holdover-mtm, complaints, new-and-frustrated, …) |
| `garage` | `has_garage`, `stalls_n`, `garage_rent_monthly` — from `RENTABLEITEMS` (stalls, not payers) |
| `duration_internal` | `expected_remaining_months_full`, `stayer_trait_months` — duration bundle v0.1, INTERNAL/unvalidated; inputs for the multi-term horizon, **not** `E[terms | saved]` |
| `incentives` | `past_concessions_n/total`, `recent_concessions[]` (TRANS, consumed), `offers_logged_n`, `last_offer` (outreach log) |
| `economics` | `turnover_cost_net/gross` + source + `make_ready_method`, `turnover_rate_per_unit_yr`, `expected_loss = p × TC_net`, `monthly_rent_at_risk`, `c_max{at_5pp,at_10pp,at_15pp}`, `cheapest_breakeven_offer`, and `offers{free_months_1, rent_discount_5pct, free_garage_new, free_garage_waiver}` each with `cost_if_renewed`, `breakeven_uplift_pp`, `verdict` (green ≤ 5 / amber ≤ 15 / red / never), `applicable` + reason, structure, and the outreach `offer_type` to log it under |
| `features` | the exact 46-column model input (bundle order, imputed from training medians, property/pet label-encoded) |

Top-level: `population` (scored / in-band / excluded-by-notice / excluded-by-window / below / above), `cohort` (expected move-outs, expected loss, rent at risk, by property, by bedroom, which offer is cheapest-breakeven for how many), `offers_summary` (per offer: applicable n, total cost if all renew, median/min breakeven, verdict counts), `context` (ledger provenance per property — make-ready measured or fallback — garage capacity per property, portfolio turn cost), `assumptions`, `formulas`, and a chatbot-safe `summary`.

**Reading the offer block honestly.** `free_garage_new` is applicable only when the resident has no garage *and* a stall is vacant at that property right now (62W: 20 vacant of 137 in September; 4–6 in winter). `free_garage_waiver` prices what they already pay. `rent_discount_5pct` is costed term-limited (0.6 months of rent) with `cost_if_baked_into_base_3yr` alongside — the compounding case. `TC_net` comes from the Phase 1 ledger; where make-ready is the $600 fallback (Winwood, CCV, Townhomes) the row's `make_ready_method` says so. Nothing in the default response assumes an uplift.

**Snapshot for reproducibility.** Pull the CSV once with an explicit `as_of` and keep it with your workbook; the live cohort moves as leases roll and scores refresh (the scored frame is cached one hour).


## 2. The numbers you start with — live ledger, 2026-09-23

Trailing 4-year window. `TC_gross = vacancy_months × avg in-place rent + make-ready`. `TC_net = TC_gross − (relet premium − renewal increase we'd have taken) × rent × 12` — what a lost resident *actually* costs once we account for re-letting the unit at a different rent. **Use `TC_net` in every decision.** Turnover rate = turns ÷ occupied unit-years (per-unit interval union). Loss/unit-yr = rate × `TC_net`.

| Property | BR | Units | Avg rent | Turns (4y) | Vac. days | Make-ready | TC gross | **TC net** | Turnover /unit-yr | Loss /unit-yr | Annual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Winwood (26) | 1 | 216 | $1,178 | 312 | 41.2 | $600 † | $2,191 | **$1,795** | 0.373 | $670 | $140,010 |
| Winwood (26) | 2 | 202 | $1,368 | 241 | 37.6 | $600 † | $2,283 | **$1,875** | 0.305 | $572 | $112,969 |
| CCV (62) | 1 | 102 | $967 | 130 | 38.5 | $600 † | $1,818 | **$1,883** ‡ | 0.321 | $604 | $61,198 |
| CCV (62) | 2 | 186 | $1,065 | 270 | 42.0 | $600 † | $2,069 | **$2,065** | 0.366 | $756 | $139,388 |
| CCV (62) | 3 | 6 | $1,248 | 15 | 39.3 | $600 † | $2,210 | **$2,665** ‡ | 0.625 | $1,666 | $9,994 |
| Sierra Pointe (72) | 1 | 46 | $1,465 | 68 | 38.3 | $600 † | $2,446 | **$2,347** | 0.380 | $892 | $39,899 |
| Sierra Pointe (72) | 2 | 81 | $1,768 | 114 | **74.7** | $600 † | $4,932 | **$4,873** | 0.382 | $1,861 | $138,880 |
| 62W (82) | 1 | 47 | $1,495 | 83 | 58.4 | $820 | $3,690 | **$3,538** | 0.474 | $1,677 | $73,414 |
| 62W (82) | 2 | 84 | $1,817 | 137 | **72.8** | $899 | $5,241 | **$5,097** | 0.448 | $2,283 | $174,572 |
| 62W (82) | 3 | 6 | $2,332 | 7 | 60.0 | $1,119 | $5,712 | **$5,865** | 0.295 | $1,730 | $10,264 |
| CCV Townhomes (473) | 2 | 16 | $1,379 | 15 | 41.8 | $600 † | $2,489 | **$1,957** | 0.238 | $466 | $7,339 |
| **Portfolio** | | **992** | | **1,392** | | | | | | | **$907,927** |

† **Fallback, unmeasured.** GL `U=<unit>` make-ready tagging exists only at 62W (99% coverage; Sierra 45%, others 0%). Three of four properties carry a flat $600 placeholder. Getting real per-turn make-ready for 26/62/72 is one of your first tasks (§9) — it changes `TC_net` there by hundreds of dollars.
‡ `TC_net > TC_gross` at CCV 1BR/3BR: the re-let premium (4.1% / 1.7%) is *below* the renewal increase (4.7%), so losing the resident forgoes the larger increase — netting adds cost rather than subtracting it.

**Read this table before anything else:** a 2BR turn costs **2.7× more at 62W than at Winwood**. Winwood fills in 38 days at $1,368; 62W takes 73 days at $1,817. Concession breakevens scale with `TC_net`, so the same offer is a completely different decision across the portfolio. Also: 1BR and 2BR at 62W turn at nearly the same rate (0.47 vs 0.45) — the 2BR cost concentration is mix and cost-per-turn, not worse behavior.

---

## 3. The decision math

### 3.1 Definitions

| Symbol | Meaning | Source |
|---|---|---|
| `p` | churn probability at the **next lease decision**, scored **pre-notice** on the **h180** bundle | `/multifamily/persuadables` (`churn_probability`), `/edge-list` |
| `TC_net` | net cost of one turn, for this resident's bedroom and property | §2 ledger |
| `C` | cost of the offer **if the resident renews** (a concession is only consumed on renewal) | §4 |
| `Δ` | uplift: reduction in churn probability caused by the offer; `p′ = p − Δ` | **unknown — policy input or trial output** |
| `r` | turnover rate per occupied unit-year | §2 ledger |

### 3.2 Expected gain, breakeven, and max spend

```
E[net gain | offer]  =  Δ · TC_net  −  (1 − p′) · C            (relative to doing nothing)

Breakeven uplift     Δ*    =  (1 − p) · C / (TC_net − C)         (offer pays iff Δ ≥ Δ*)
Max justified spend  C_max =  Δ · TC_net / (1 − p + Δ)           (offer pays iff C ≤ C_max)
```

The first term is what we save; the second is what we pay to residents who would have renewed anyway — the **sleeping-dog cost**. It is proportional to `(1 − p)`, which is why the same offer is hopeless at `p = 0.26` and nearly free at `p = 0.97` (§5.4). If `C ≥ TC_net` the offer can never pay on this horizon (shown as "never" in §5).

**Rule of thumb:** at `Δ = 10pp` and band-typical `p = 0.26`, `C_max ≈ 0.12 × TC_net`. At `Δ = 5pp`, `≈ 0.06 × TC_net`. Post-notice (`p = 0.97`, `Δ = 10pp`), `≈ 0.77 × TC_net`.

### 3.3 Where `p` must come from (and three ways to get it wrong)

1. **Use h180, not the serving default h60.** The band is defined on the renewal decision, a 180-day question. The h60 triage ranker compresses almost everyone below 0.20 (3 vs 18 band members at 62W — Phase 0 §4). `persuadables.py` pins `v6-roadmap-h180` for this reason.
2. **Use the calibrated model probability, never the display-capped `churn_probability` shown in the churn report** for any arithmetic (it is clipped to 0.10–0.75 for display). The persuadables/edge-list endpoints return usable probabilities.
3. **Pre-notice only for the band; post-notice is a different population** (§6B) with its own base rate (2.7% renew after notice), not a model score.

### 3.4 Gate 0 — is a concession even the right lever?

Before the arithmetic, ask *why* the resident is at risk. The model returns per-resident `drivers` (SHAP-style contributions, `contracts/multifamily_churn/explain.py`). Route them:

| Driver family | Example features | Right lever |
|---|---|---|
| **Price-addressable** | `RentIncreasePct`, rent level vs market | Concession / renewal-increase moderation |
| **Service-addressable** | `WorkOrders_Last90Days`, `EmergencyWorkOrders`, complaint & maintenance memos | Fix the service issue; a concession here buys silence, not retention |
| **Timing / structural** | `DaysToLeaseEnd` (holdover), short tenure, first decision | Earlier outreach, term structure (§6C) |
| **Payment friction** | `NSFCount`, payment memos | Careful — a discount to a credit-risk signal is not a retention play |

This is a correlational routing heuristic, not causal truth — but it shrinks the treated population to residents whose churn a price lever can plausibly move, which both lowers the budget at risk and raises the plausible `Δ` inside the treated group. **Your task:** with Brandon, define the mapping as written, consistent criteria (this is a fair-housing requirement, §10) and estimate the **addressable share** of the band per property.

### 3.5 Turnover rate's two jobs

`r` is not a decision input — `p` is, per resident. `r` does two other things:

- **Sizes the prize.** Steady-state loss per unit-year `L = r × TC_net` (62W 2BR: 0.448 × $5,097 = $2,283; Winwood 2BR: $572). Summed over units it is the "annual" column — the disease burden a program can address a slice of.
- **Is the baseline any program will be judged against** — and pre/post comparisons of `r` are confounded by season and market regime (Phase 0 §12.2). Only randomization moves `r` attributably. Track `r` monthly (§8) so you know the baseline is drifting *before* a program starts.

---

## 4. The three offers — cost models and structure

| Offer | `C` if renewed (12-mo term) | Structure | What to know |
|---|---|---|---|
| **Free garage — new assignment** (resident does not have one) | `(12 − E[months to re-rent]) × tier rent` → **$750** ($75 tier) / **$1,000** ($100 tier) at 62W with `E[T] = 2 mo` | waiver; face rent preserved; non-compounding | Cheapest lever. **Capacity-capped**: 62W has 137 stalls, 93% mean occupancy, 4–6 free in winter, ~21 in the trough; 77% of residents already rent one, so this applies to a minority. Inventory, tiers and occupancy at the other four properties are **unknown** → your task (`RENTABLEITEMS` + `RentableItems_Status`). |
| **Free garage — waiver** (resident already rents one) | `12 × tier rent` ≈ **$1,044** (blended $87) | waiver | Full price, no occupancy offset — as expensive as a 5% discount with none of its downside protection. |
| **One month free** | `1 × rent` — **$967–2,332** across the portfolio (§2) | one-time credit; face rent preserved | Cannot pay pre-notice on one term anywhere (§5.1). Justified **post-notice** (§5.4), plausibly at `k ≥ 2` terms (§5.3), or as the vehicle for lease-end migration (§6C). Post to acct **53** at the *new* term's start with the offer date recorded in the outreach log — that is what keeps TRANS joinable and defeats the posting-time tautology. |
| **5% off rent** | if term-limited: `0.05 × 12 × rent = 0.6 months` — **$580–1,399**; if baked into base: compounds (≈ **$3,100 over 3 years** at 62W 2BR) | recurring discount, **term** or **base** | Same cash as 0.6 free months, worse structure: degrades face rent (loss-to-lease reporting, comps, appraisal/lender NOI reads) and, if removed at month 13, creates a **renewal cliff** that defers churn rather than removing it. **Default recommendation: offer the same dollars as a one-time credit.** Keep the rate form only as a trial arm testing whether *framing* ("lower rent") moves residents more than an equivalent credit — a behavioral question a trial can answer. |

Every offer record must carry `offer_type`, `offer_structure`, `offer_value`, `bedroom`, `property_id`, `arm`, `posting_account` (the outreach log enforces the first). A dollar amount alone cannot be costed.

---

## 5. Verdict tables — computed from the live ledger (2026-09-23)

Read: **green ≤ 5pp** (inside what history plausibly supports) · **amber 5–15pp** (trial territory) · **red > 15pp** (not on this horizon) · **never** = `C ≥ TC_net`. `p = 0.26` is the measured band mean at every property (Phase 0 §11.1); `p = 0.40` is the upper-band case.

### 5.1 Breakeven uplift `Δ*` (pp) — pre-notice, one lease term

| Property | BR | TC net | 1 free month `C` | `Δ*` @ p=.26 | @ p=.40 | 5% (0.6 mo) `C` | `Δ*` @ p=.26 | @ p=.40 |
|---|---|---|---|---|---|---|---|---|
| Winwood | 1 | $1,795 | $1,178 | 141 | 115 | $707 | 48 | 39 |
| Winwood | 2 | $1,875 | $1,368 | 200 | 162 | $821 | 58 | 47 |
| CCV | 1 | $1,883 | $967 | 78 | 63 | $580 | 33 | 27 |
| CCV | 2 | $2,065 | $1,065 | 79 | 64 | $639 | 33 | 27 |
| CCV | 3 | $2,665 | $1,248 | 65 | 53 | $749 | 29 | 23 |
| Sierra Pointe | 1 | $2,347 | $1,465 | 123 | 100 | $879 | 44 | 36 |
| Sierra Pointe | 2 | $4,873 | $1,768 | **42** | **34** | $1,061 | **21** | **17** |
| 62W | 1 | $3,538 | $1,495 | 54 | 44 | $897 | 25 | 20 |
| 62W | 2 | $5,097 | $1,817 | **41** | **33** | $1,090 | **20** | **16** |
| 62W | 3 | $5,865 | $2,332 | 49 | 40 | $1,399 | 23 | 19 |
| CCV Townhomes | 2 | $1,957 | $1,379 | 177 | 143 | $827 | 54 | 44 |

Everything is red. The best case in the whole portfolio — 5% off at 62W/Sierra 2BR for a `p = 0.40` resident — needs 16–17 points of uplift; history's leakage-free estimate is +10–12 with a CI that includes zero.

### 5.2 Max justified spend `C_max` per resident

| Property | BR | TC net | Δ=5pp, p=.26 | Δ=10pp, p=.26 | Δ=10pp, p=.40 | **Post-notice** Δ=10pp |
|---|---|---|---|---|---|---|
| Winwood | 1 | $1,795 | $114 | $214 | $256 | $1,381 |
| Winwood | 2 | $1,875 | $119 | $223 | $268 | $1,442 |
| CCV | 1 | $1,883 | $119 | $224 | $269 | $1,448 |
| CCV | 2 | $2,065 | $131 | $246 | $295 | $1,588 |
| CCV | 3 | $2,665 | $169 | $317 | $381 | $2,050 |
| Sierra Pointe | 1 | $2,347 | $149 | $279 | $335 | $1,805 |
| Sierra Pointe | 2 | $4,873 | $308 | $580 | $696 | $3,748 |
| 62W | 1 | $3,538 | $224 | $421 | $505 | $2,722 |
| 62W | 2 | $5,097 | $323 | **$607** | **$728** | **$3,921** |
| 62W | 3 | $5,865 | $371 | $698 | $838 | $4,512 |
| CCV Townhomes | 2 | $1,957 | $124 | $233 | $280 | $1,505 |

This is the table to hand a PM: *"for this resident, at the uplift we're willing to assume, you may spend up to $X."* Pre-notice at 10pp, the only offer under any `C_max` is a new $75-tier garage at 62W/Sierra 2BR. Post-notice, a free month fits everywhere except Winwood 1BR / CCV 1BR (and fits there too at Δ ≥ 12pp).

### 5.3 62W garage variants, and the multi-term horizon

| 62W 2BR, p=.26 | `C` | `Δ*` |
|---|---|---|
| Garage new, $75 tier, E[T]=2 mo | $750 | **12.8pp** (amber) |
| Garage new, $100 tier | $1,000 | 18.1pp |
| Garage waiver | $1,044 | 19.1pp |

If a save retains `k` lease terms instead of one, breakevens divide by roughly `k` (62W 2BR, p=.26):

| `k` | 1 free month | 5% / 0.6 mo | Garage new $750 |
|---|---|---|---|
| 1 | 41.0 | 20.1 | 12.8 |
| 1.5 | 23.1 | 12.3 | 8.0 |
| **2** | **16.1** | **8.9** | **5.9** |
| 3 | 10.0 | 5.7 | 3.8 |

At `k = 2` the cheap offers land inside the historically plausible range. The duration model (`bundles/multifamily_duration/v0.1`, backtest MAE 2.1 months, Spearman 0.90, **internal/report-only**) is what would supply `k` per resident. **Validating it costs nothing in concessions and may change every verdict** — see §9, week 5+.

### 5.4 Post-notice: the same offer, a different decision

62W 2BR, one free month (`C = $1,817`), base renewal after notice **2.7%** (`p = 0.973`, from 480 notices at 62W):

- `Δ* = (1 − 0.973) × 1,817 / (5,097 − 1,817) = 0.027 × 1,817 / 3,280 = 1.5pp`
- `E[net]` at `Δ = 10pp`: `0.10 × 5,097 − 0.127 × 1,817 = +$279` per offer; at `Δ = 17pp` (the historical 3-of-15 hint): `+$509`
- Worst case (`Δ = 0`): `0.027 × $1,817 ≈ $49` per offer — the credit is consumed only by the 2.7% who renew anyway

The economics flip because `(1 − p)` collapses. No model is needed to find these residents — notice self-identifies them — which also makes this the cleanest governance shape available (uniform offer, self-identified group, randomized only for measurement).

### 5.5 One row, end to end — `t0018814` (62W, unit 236)

From the attached CSV (2026-09-25): 2BR, in-place rent **$1,900** (= market), moved in 2026-08-15 on a 7-month term ending 2027-03-15 (**171 days** out), tenure 1.3 months, no notice. Rents one garage stall at **$100/mo**. Cohort memberships: `first-year` (risk, lift 1.29). Model: `p = 0.3819` — *"driven by the lease ends in 171 days, a 212-day planned lease, rent level ($1,900), short tenure; offset by no move-out notice on file."* No prior concessions, no logged offers. Duration model (internal): 6.2 expected remaining months, stayer-trait 6.5.

Ledger for 62W 2BR: `TC_net = $5,120` (measured make-ready). So:

```
expected_loss = 0.3819 × 5,120                       = $1,955
C_max @ 5pp   = 0.05 × 5,120 / (1 − 0.3819 + 0.05)   = $383
C_max @ 10pp  = 0.10 × 5,120 / 0.7181                = $713
C_max @ 15pp  = 0.15 × 5,120 / 0.7681                = $1,000

one month free       C = 1,900   Δ* = 0.6181 × 1,900 / (5,120 − 1,900) = 36.5pp   red
5% off, term-limited C = 1,140   Δ* = 0.6181 × 1,140 / 3,980            = 17.7pp   red
                                 (baked into base, 3 yrs at 3.72%/yr:   $3,549)
garage — new         n/a: already rents a stall
garage — waiver      C = 1,200   Δ* = 0.6181 × 1,200 / 3,920            = 18.9pp   red
```

Reading: at any uplift history can support (≤ 10pp) you may spend **$713** on this resident; the cheapest instrument costs $1,140. No offer clears. The three things that would change that, in order of cheapness: a second retained term (`k = 2` halves every breakeven — 5% off drops to ~9pp), a measured uplift above ~18pp from a trial, or a cheaper instrument (a $500 one-time credit has `Δ* = 0.6181 × 500 / 4,620 = 6.7pp` — amber). Note also *why* the model has him here: lease timing and tenure, not price or service. §3.4 says a concession is not obviously the right lever for a resident whose risk is "new and the lease is short" — a longer renewal term may do more than a credit.


---

## 6. Four populations — where each offer can pay, and how to measure it

| | **A. Pre-notice persuadable band** | **B. Post-notice** | **C. Jan–Feb lease-ends** | **D. All renewal decisions** |
|---|---|---|---|---|
| Who | h180 `p ∈ [0.20, 0.60)`, no notice | Filed notice to vacate | Active executed terms ending Nov–Feb (eligibility = lease-end month, not a score) | Every renewal decision |
| Flow / yr | ~107 portfolio; 39 Winwood, 39 CCV, 19 62W, 10 Sierra (2026-08-18 pass) | Reconcile first (§9) — memo cites ~40/yr portfolio but its own power math implies ~400 | **181 active terms** today: Winwood 78, CCV 49, 62W 28, Sierra 23, 473 3 (19–25% of book) | **926** |
| Typical `p` | 0.26 | 0.97 | population mix | 0.41 churn |
| Offers that clear | none at Δ ≤ 10pp; garage-new at 62W/Sierra 2BR is amber | one free month (Δ\* ≈ 1.5pp) | 15-month term ± 1 free month — see below | measurement design, not a program |
| Governance | model-gated → counsel review, feature audit, written criteria | uniform, self-identified, no model | uniform criterion (calendar), no model | uniform |
| Power (two-proportion, 80%, α=.05) | **hopeless alone**: 62W needs 40 years to see +10pp; portfolio band 7 years | 52/arm detects 2.7%→20% in months (once flow is reconciled) | pooled across properties, ~180/yr → +15pp in ~2 years; migration effect is deterministic and needs no power | +10pp in **~10 months** (376/arm); worst case ≈ −$165k/yr |
| Role of the model | selection (with governance) | none | **stratify and adjust** — block on `p`-decile × bedroom, adjust on `p`; read heterogeneous effects by `p`-band afterward | same |

**C in detail — the concession as the vehicle for lease-end migration.** Winter move-outs fill slower everywhere (mean days, Nov–Feb vs Mar–Jun): CCV 53 vs 40 (**+13**), Winwood 47 vs 34 (**+14**), 62W 82 vs 51 (**+32**), Sierra 84 vs 48 (**+36**). Moving a resident's lease-end into spring saves that penalty at their *eventual* turn, whenever it comes: ≈ **$1,910** at 62W 2BR and **$2,090** at Sierra 2BR, but only ≈ $630 at Winwood and $455 at CCV. Compare with the free month that would buy the odd term (`0.74 × rent` consumed by renewers vs `0.74 × penalty` recovered):

| 2BR, p=.26 | Free-month cost (renewers) | Migration value (renewers × penalty) | Ratio |
|---|---|---|---|
| 62W | $1,345 | $1,413 | **1.05** — self-funding before any retention uplift |
| Sierra Pointe | $1,308 | $1,547 | **1.18** |
| Winwood | $1,012 | $466 | 0.46 — offer the 15-month term *without* the free month |
| CCV | $788 | $337 | 0.43 — same |

Arms: **12-mo standard / 15-mo term, no concession / 15-mo term + 1 free month**. The middle arm isolates whether the free month is needed at all to get the odd term accepted. Caveats you must carry into the design: the migration value is realized once at the eventual turn (discount it), assumes later renewals keep the spring date, uses *mean* penalties (medians differ), and 62W's 25 Jan–Feb terms are a small base — pool across properties.

**The two deterministic levers, for context (`scripts/audit_ops_levers.py`, 2026-09-23).** Excess vacancy days above each property's median: Winwood $77k/yr, 62W $61k/yr, Sierra $53k/yr, CCV $45k/yr (≈ $237k/yr, an upper bound that includes reno holds). At 62W, where GL tagging allows classification: 11 tail turns were **rent-ready then sat empty** (≈ $114k over 4 years — a leasing problem), 11 had **late make-ready** (≈ $113k), 2 were heavy renovations (exclude from churn billing). These dwarf the concession program's optimistic ceiling ($18–40k/yr portfolio) and carry no fair-housing surface. **Take this money first**; it also funds the trials.

---

## 7. NOI bridge and cost-risk analysis — your core deliverable

### 7.1 Baseline (per property, per bedroom)

```
turns/yr        = turns_window / 4
L_baseline      = turns/yr × TC_net            (= "Annual" in §2; portfolio $907,927)
L per unit-year = r × TC_net
```

Present the baseline as an NOI line item: **turnover loss = vacancy revenue loss + make-ready − net relet recapture**, with the make-ready line flagged measured/fallback per property.

### 7.2 Program layer

For a treated population with `N_T` decisions/yr, treated-group churn rate `c_T`, offer cost `C`, uplift `Δ`:

```
turns avoided/yr     = N_T × Δ
value/yr             = N_T × Δ × TC_net
concession cost/yr   = N_T × (1 − c_T + Δ) × C          ← renewers consume the credit
net NOI impact/yr    = value − concession cost
Δ* (program)         = (1 − c_T) × C / (TC_net − C)      ← same formula, aggregated
```

Second-order lines to carry separately: **face-rent effects** for rate discounts (loss-to-lease, comps, appraisal NOI); **garage revenue forgone** (already inside `C`); **capacity** (garage-new is capped by free stalls, seasonally); **behavioral spillover** (residents talk — a leak-tolerance assumption, stated).

### 7.3 The cost-risk grid

Build one grid per property: rows = offers (§4), columns = `Δ ∈ {0, 5, 10, 15, 20pp}`, one sheet per population (§6). **The `Δ = 0` column is the budget at risk** — every dollar spent on residents who would have renewed anyway. Phase 0 §11.3 did this for 62W 2BR at p=.26 (3% discount: −$237 → +$375 across 5–20pp; one free month: −$1,076 → −$624 — negative everywhere). Reproduce it from the live ledger, extend to all properties and populations, and add the post-notice and migration sheets. Report the program **ceiling** (optimistic `Δ`, cheapest offer, full flow) next to the $908k disease burden so the burden stops being the anchor.

### 7.4 Sensitivities to run

- `TC_net` with make-ready at fallback vs measured (26/62/72 once you have GL data).
- Vacancy days: full mean vs **churn-attributable** mean (exclude heavy-reno holds and strategic vacancy identified in the tail audit). The mean includes 300-day holds that were never a retention problem.
- Rent basis: in-place (default) vs market (`avg_market_rent` in the ledger).
- `k` terms: 1 / 1.5 / 2 from the duration model.
- Season: `TC_net` for a Jan lease-end vs a June one (Brandon is adding a season-conditional cost to the ledger; until then apply the §6 penalties by hand).

---

## 8. Month over month — how we will know the numbers are holding

**Model calibration (exists).** `monitoring/calibration.log` records predicted vs realized churn on each month's expiring cohort (h60 bundle). 2026 months run −1 to −5pp (slight, consistent under-prediction; healthy). Retro-scored pre-2026 months alert badly — trust contemporaneous months only. Brandon is adding h180 and the band's realized rate. **Your check each month:** does the persuadable band actually churn at ~26%? Does the edge roster's exit mix (renewed / notice / moved out / subsided) look like the band's `p` implies?

**Economic calibration (build this).** From `/multifamily/churn-report?month=YYYY-MM` (per-resident outcome + month-start score) joined to the §2 ledger by property × bedroom:

```
realized turn cost (month)   = Σ_churned  TC_net(bed, property)
predicted $ at risk (month)  = Σ_cohort   p × TC_net(bed, property)
```

Calibration in dollars weights errors by what they cost — a 2BR miss at 62W is 2.7× a 2BR miss at Winwood. Plot both series monthly; the gap is the number leadership will ask about.

**Cohort scorecard.** The cohort catalog (`/multifamily/cohorts`: holdover-mtm, complaints, new-and-frustrated, multi-pet, payment-friction, first-year, pending-decision, loyal-2y/5y, renewal-veterans) reports live lift but no outcomes over time. Apply the predicates to each 2026 expiring cohort (endpoints take `as_of`), join outcomes, and produce realized churn, `$` lost and predicted-vs-realized per cohort per month. Add **bedroom** and **persuadable band** as cohorts. The targeting question becomes concrete: which cohort × bedroom × property cells combine high churn, high `TC_net`, and high addressable share (§3.4)?

**Turnover rate series.** Monthly turns and vacancy days per bedroom per property (Brandon can expose the turn frame; `unit_economics._turns_frame` is the source). This is the baseline any program is judged against — watch its seasonality before attributing anything to a program.

---

## 9. Deliverables and sequence (six weeks)

**Days 1–3 checklist**

- [ ] Reproduce §5.5 by hand from the CSV; then reproduce §2 from the ledger JSON
- [ ] Call the live endpoint (JSON + CSV), once with `band_low=0.3`, once with `assumed_uplift=0.10`
- [ ] Build the cost-risk grid skeleton (§7.3) for 62W 2BR only — five uplift columns, four offers, the `Δ = 0` column labeled "budget at risk"
- [ ] Read Phase 0 §3, §11, §12 and the research baseline §7; write down the three questions you would ask before trusting any number in this document
- [ ] Agree the check-in cadence with Brandon and what "done" looks like for weeks 1–2


| Weeks | You | Brandon (repo) | Depends on |
|---|---|---|---|
| **1–2** | **Reconcile notice flow** per property from `tenant_history` / `Lease_History` (the post-notice trial's denominator). **GL make-ready census** for Winwood/CCV/Sierra: which accounts, whether any unit attribution exists, per-turn medians. **Garage inventory** per property (`RENTABLEITEMS`, tiers, occupancy, attach rate). Reproduce §2 and §5 from the live ledger in a workbook. | Restart service so `/multifamily/unit-economics` is live; schedule the edge roster nightly; add h180 + band rows to the calibration log; month-start score snapshots. | DB read access for you (Brandon) |
| **3–4** | **NOI bridge + cost-risk grid** (§7) for all properties and populations. **Monthly realized turn-cost series** and economic calibration for 2026 (§8). Draft the **`C_max` policy memo**: what `Δ_believed` should be, by population, until trials report. | Season-conditional `TC_net` in the ledger; make `persuadables.py` consume the ledger (it still uses a flat $600 make-ready); expose the turn frame; `concession-decision` view per resident × offer (`C`, `Δ*`, `C_max`, class, driver routing, garage capacity check). | Weeks 1–2 make-ready + garage data |
| **5–6** | **Pre-registered trial designs** for B (post-notice rescue) and C (Jan–Feb migration, three arms): eligibility rule, randomization by auditable rule, primary outcome (renewal), secondary (net NOI = saved turns × `TC_net` − consumed credits from TRANS), covariate-adjusted analysis on `p`, power at measured flows, stopping rules. **Governance packet** with Brandon: consistent-criteria doc, feature audit request, offer-handling scripts. | Decision register (monthly expiring cohort + pre-assigned arm + PM-filled offer/outcome) wired to the outreach log → TRANS `posting_account` → `Lease_History` outcome. | Notice flow; counsel review scheduled |
| **ongoing** | **Duration-model validation for `k`** (§5.3) — the single cheapest thing that could change the verdicts. Cohort scorecard monthly. | Randomization + heterogeneous-effect readout by `p`-band once arms exist. | |

---

## 10. Landmines (each one has already cost us a rewrite)

- **`UNIT.EXCLUDE = 0` everywhere.** `EXCLUDE = −1` marks pseudo-units (`GARAGE 10`, `WAIT*`) that hold real TENANT rows. True 62W mix is 47/84/6 = 137 units.
- **TENANT mutates in place.** Renewals overwrite `DTLEASETO` and `SRENT`; historical cohorts and rents come from **`Lease_History`** (`iStatus` 4/5 = executed terms, `cRent` = the rent at the time). Never rebuild a past month from TENANT.
- **Acct 53 "Renewal Concession" timing is a tautology.** The credit posts ~18 days *before* the decision — i.e., when the renewal is signed — so recipients renew 100% by construction. Any uplift you compute from TRANS is leakage unless you filter on treatment lag. The only leakage-free estimate is the *next*-term effect (+10–12pp, CI [−4.9, +23.4]), and entitlement accounts 50/51 (true effect zero) show +6–13pp under the same estimator. **Do not re-derive an observational uplift; Phase 0 §3 closed it.**
- **62W's de-facto free month is acct 54, not 53** ($973k / 355 residents, mostly move-in concessions). Posting convention for retention offers is an open decision (Phase 0 §10.4).
- **The `U=<unit>` GL convention is a 62W bookkeeping habit.** It does not exist at Winwood/CCV; Sierra is partial. Do not apply 62W make-ready to any other property.
- **3BR is statistically thin** (7 turns at 62W, 15 at CCV). Fold into "2BR+" for any estimate.
- **h60 vs h180; display-capped probabilities** (§3.3).
- **Garages are a winter good in Iowa.** 84.7% occupancy in August, 96.4% in Jan–Mar. A static "stagnant" threshold detects season, not scarcity. The design doc's `C_garage ≈ $0` branch is retracted.
- **Regime.** The 4–9-year history spans lease-up, COVID, the 2021–22 surge, and today's most-concessionary market on record (39.8% of national listings with a concession, Apr 2026; Des Moines adding 6,000+ units with 1–2 free months at lease-up). Levels — 45% turnover, 73-day 2BR fills — are partly regime; so will any measured uplift be.
- **Fair housing.** Model-gated offers use features (NSF, late fees, tenure, pets) that can correlate with protected classes. Counsel review (FHA + Iowa Civil Rights Act) and written consistent criteria before any model-selected offer. Populations B and C avoid the problem by construction — start there.

---

## 11. Formulas card

```
TC_gross              = vacancy_months × avg_inplace_rent + make_ready
TC_net                = TC_gross − (relet_pct − renewal_increase_pct) × rent × 12
turnover_rate r       = turns / occupied_unit_years               (interval union)
loss per unit-year    = r × TC_net
expected loss (res.)  = p × TC_net

E[net gain | offer]   = Δ × TC_net − (1 − p + Δ) × C
breakeven Δ*          = (1 − p) × C / (TC_net − C)                ("never" if C ≥ TC_net)
max spend C_max       = Δ × TC_net / (1 − p + Δ)
multi-term            ≈ replace TC_net with k × TC_net             (breakevens ÷ k)
migration value       ≈ (1 − p) × winter_penalty_days × daily_rent (realized at eventual turn)

offer costs (if renewed, 12-mo): free month = rent · 5% term-limited = 0.6 × rent ·
                                 garage new = (12 − E[T_re-rent]) × tier · waiver = 12 × tier
```

## 12. Reproducibility

- Ledger: `GET /multifamily/unit-economics` (live; cached 24 h in-process) or `.venv/bin/python -c "from contracts.multifamily_churn import unit_economics as ue; import json; print(json.dumps(ue.build(), indent=1))"`. §2 table built 2026-09-23; the attached JSON is the 2026-09-25 pull (identical).
- Ops levers: `.venv/bin/python scripts/audit_ops_levers.py` (read-only; prints tail, seasonality, lease-end exposure per property).
- Phase 0 numbers: `scripts/eval_phase0.py`; design + memo in `docs/`.
- Band flow and power math: Phase 0 memo §11.1–11.2.
- Offer logging contract: `contracts/multifamily_churn/outreach.py` (`OFFER_TYPES`, `OFFER_STRUCTURES`); `POST /multifamily/outreach`.
- Frozen snapshots (this handoff's attachments): vault `outputs/multifamily-prediction/datasets/concession-cohort-2026-09-25.{csv,json}`, `concession-cohort-top-p30-2026-09-25.csv`, `unit-economics-2026-09-25.json` — regenerate with the same `as_of=2026-09-25` to diff against.
- Concession cohort module + tests: `contracts/multifamily_churn/concession_cohort.py`; `tests/test_smoke.py::test_concession_offer_economics_offline` pins the offer math with no DB.

**Questions:** Brandon Valadez (valadez.brandon@realtyts.com)

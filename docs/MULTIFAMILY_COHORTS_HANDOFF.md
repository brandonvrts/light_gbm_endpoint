# Multifamily Resident Cohorts — Chatbot Handoff

**Feature:** named resident cohorts over the churn model, plus the incentive-targeting
surfaces built on them. Everything here is **live and deployed** (2026-07-22) and
returns chatbot-safe `summary` sentences.
**Model:** `v6-roadmap-h60` (calibrated LightGBM). "Perform" always means: **expected
move-out risk at each resident's NEXT lease decision** (60-day horizon).
Companion docs: `API_HANDOFF.md` (general), `CHURN_REPORT_HANDOFF.md` (monthly report
the chatbot already uses).

**The four surfaces (PM question → endpoint):**

| PM asks | Endpoint |
|---|---|
| "How is every cohort doing?" / "which cohorts exist?" | `GET /multifamily/cohorts` (§1) |
| "How will the multi-pet cohort perform?" | `GET /multifamily/cohort-report?cohort=multi-pet` (§2) |
| "Who are the edge residents at Winwood / for August?" | `GET /multifamily/persuadables?properties=26` / `?month=2026-08` (§3) |
| "What incentives has this resident/property received?" | `GET /multifamily/incentives?scode=…` / `?property=…` (§3) |

## Connection

Same service and auth as everything else:

| | |
|---|---|
| **Base URL (LAN)** | `http://10.1.6.17:8910` |
| **Auth** | Header `X-API-Key: rts-local-dev-key` |
| **Interactive docs** | `http://10.1.6.17:8910/docs` |

---

## 1. Cohort catalog — "which cohorts can I ask about?" / "how is every cohort doing?"

```bash
curl -s -H "X-API-Key: rts-local-dev-key" "http://10.1.6.17:8910/multifamily/cohorts"
# optional: ?properties=62,26   (HPROPERTY ids; default all 5)
```

```json
{
  "contract": "multifamily-churn",
  "model_version": "v6-roadmap-h60",
  "as_of": "2026-07-21",
  "view": "cohort-catalog",
  "population": {"n": 955, "avg_churn": 0.081, "expected_moveouts": 77.1},
  "cohorts": [
    {"cohort": "holdover-mtm", "label": "Holdover / month-to-month",
     "definition": "days_to_lease_end < 0",
     "n": 74, "share_pct": 7.7, "avg_churn": 0.292, "lift_vs_portfolio": 3.62,
     "expected_moveouts": 21.6, "high_risk_n": 32, "notice_given_n": 12},
    ...
  ]
}
```

Cohorts come back **sorted by lift** (riskiest first). Use `cohort` as the key for the
report endpoint, `label`/`description` for display.

## 2. Cohort report — "how will the holdover cohort perform?"

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/cohort-report?cohort=holdover-mtm&top=5"
# optional: &properties=26  &as_of=YYYY-MM-DD  &top=N (at-risk members, default 10)
```

```json
{
  "view": "cohort-report",
  "cohort": {"key": "holdover-mtm", "label": "Holdover / month-to-month",
             "definition": "days_to_lease_end < 0",
             "description": "Lease already expired but still in the unit — no renewal signed..."},
  "population": {"n": 955, "avg_churn": 0.081, "expected_moveouts": 77.1},
  "n": 74, "share_pct": 7.7, "avg_churn": 0.292, "lift_vs_portfolio": 3.62,
  "expected_moveouts": 21.6, "high_risk_n": 32, "notice_given_n": 12,
  "tier_mix": {"low": 22, "medium": 20, "high": 32},
  "members_top": [
    {"scode": "t0014040", "full_name": "Mike Jennings", "unit": "318",
     "property": "Sierra Pointe", "lease_to": "2026-05-25", "days_to_lease_end": -57,
     "has_notice": 1, "churn_probability": 0.95, "risk_tier": "high"}
  ],
  "summary": "The Holdover / month-to-month cohort (days_to_lease_end < 0) has 74 residents (7.7% of the 955 scored). Expected performance at their next lease decision: 29.2% average move-out risk (3.62x the portfolio average of 8.1%), about 22 expected move-out(s). 32 are high risk; 12 already gave notice."
}
```

**`summary` is chatbot-safe to read back verbatim** — same convention as the churn
report. For a per-resident "why", call `POST /predict/multifamily-churn`
`{"scode": "..."}` with the member's scode (returns `why` + `drivers`).

## The cohort catalog (v1) — with today's real numbers

| key | definition | n | avg churn | lift |
|---|---|---|---|---|
| `holdover-mtm` ⭐ | lease already expired, still in unit | 74 | 29.2% | 3.62× |
| `complaints` | complaint memos on file | 33 | 10.7% | 1.32× |
| `new-and-frustrated` | first year + 6+ work orders/yr | 73 | 9.9% | 1.23× |
| `multi-pet` | 2+ pets | 147 | 9.5% | 1.18× |
| `payment-friction` | NSF or payment memos | 120 | 9.1% | 1.13× |
| `first-year` | tenure < 12 months | 302 | 8.2% | 1.02× |
| `pending-decision` | expires ≤ 90d, **no notice yet** | 195 | 6.4% | 0.80× |
| `loyal-2y` | tenure 24+ months | 484 | 6.9% | 0.85× |
| `loyal-5y` | tenure 60+ months | 225 | 5.4% | 0.67× |
| `renewal-veterans` | renewed 2+ times | 380 | 6.1% | 0.75× |

⭐ **holdover-mtm is the standout**: 7.7% of residents carrying ~28% of all expected
portfolio churn. If a user asks "who should we worry about," this is the answer.

Numbers above are the 2026-07-21 snapshot — they move with the population; always show
what the API returns, not this table.

## How to interpret (bake this into the bot's phrasing)

- **avg_churn** — average calibrated probability of moving out at the *next lease
  decision* (not "this month"). The model is calibration-audited, so these read as
  honest percentages.
- **expected_moveouts** — sum of member probabilities = expected count at the next
  decision cycle. Safe to say "about 22 of these 74 residents are expected to leave."
- **lift_vs_portfolio** — vs. the same-scope population average (recomputed when you
  filter by `properties`, so lifts stay honest per building).
- **Low-lift cohorts are good news** — frame `loyal-*` / `renewal-veterans` as "your
  stable base," not as a risk list. `pending-decision` is below-average risk but is the
  *retention-outreach* list: those renewal decisions arrive within 90 days.
- **notice_given_n** — members who already filed notice (their probabilities are ~0.95
  and they dominate a cohort's high tier; `pending-decision` excludes them by design).

## Deliberately excluded (don't ask us to add these back without new evidence)

- **Rent-increase cohorts** — survivor-biased: a 5%+ increase implies an already-signed
  renewal, so they read misleadingly *low* risk (0.72×).
- **Bare "recent work order"** — non-differentiating (65% of residents had one in 90
  days; lift 0.98×). The maintenance signal that works is `new-and-frustrated`.
- **Eviction-memo cohort** — real signal (2.1×) but n=8 today; numbers too unstable to
  quote. Revisit if the population grows.

## 3. Persuadables / "edge residents" — who's on the fence, and what's an incentive worth?

The incentive-targeting surface (added 2026-07-21). Band = churn probability **0.20–0.60,
notice-filed excluded** — below it residents renew anyway (do-not-disturb), above it the
notice is already filed.

**PM question mapping** (the phrasings this endpoint exists for):
- *"Who are the edge residents at Winwood?"* → `?properties=26`
- *"Who are the edge residents for August?"* → `?month=2026-08` (lease decision lands
  in that month; the chatbot should resolve bare month names to YYYY-MM)
- Both compose: `?properties=26&month=2026-08`

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/persuadables?top=25"
# optional: &properties=26 &month=2026-08 &incentive_cost=250 &flip_rate=0.25
#           &band_low=0.2 &band_high=0.6
```

Response essentials:

- `population` — full-portfolio context incl. `sure_things_n` (the do-not-disturb count).
- `members` — ranked by **monthly rent at risk** (probability × rent). Each carries:
  - `on_the_fence_because` — the Venn story: `risk_cohorts` vs `protective_cohorts`
    memberships (classified by measured live lift, e.g. holdover-mtm RISK vs loyal-5y
    PROTECTIVE), for the PM conversation.
  - `why` + `drivers` — the model's own per-resident explanation.
  - `value_if_saved` — that resident's turn cost: avg turn days (measured per property
    from unit-vacancy gaps; portfolio ≈ 47 days) × their daily rent + $600 make-ready.
- `economics.scenario` — spend vs expected value at the given `incentive_cost` (default:
  one month of band avg rent) and `flip_rate`. **The flip rate is ASSUMED and labeled as
  such** — no measured one exists yet (the TRANS ledger records consumed concessions,
  not offers). The chatbot must never present `net` as a prediction; it's scenario math.
- `summary` — chatbot-safe verbatim sentence, includes the assumption flags (and the
  month scope when `month` is passed).
- `incentives` (per member) — what this resident has already been given/offered:
  `past_concessions_n` / `past_concessions_total` / `recent_concessions` (from the
  Yardi TRANS ledger) + `offers_logged_n` / `last_offer` (from the outreach log). A PM
  seeing "already received $1,800 in concessions" makes a different offer than one
  seeing a blank history.

**Standalone incentive history** — `GET /multifamily/incentives?scode=t0014040` (or
`?property=26`, `&since=2024-01-01`, `&limit=`): consumed concessions from TRANS (8
landlord concession accounts; per-account totals cover the **full** filtered set —
`events` is just the display slice) plus logged offers. Answers "what incentives has
this resident/property received?" directly.

**The measurement loop (tell the PMs):** every offer should be logged via
`POST /multifamily/outreach` with `action` = `incentive_offered` →
`incentive_accepted` / `incentive_declined`, plus `incentive_cost` and the
`churn_probability` shown at offer time. Declined offers are the data TRANS can never
capture — after 2–3 renewal cycles this log replaces the assumed flip rate with a
measured one.

## Good to know

- **Latency:** first call after a service restart scores the whole population (~30–60 s);
  after that everything is cached ~1 h and returns instantly. Both endpoints share the
  cache. Set a generous first-call timeout (120 s+), like the churn report.
- **Composability:** `properties` (comma-separated HPROPERTY ids: 62 CCV, 72 Sierra
  Pointe, 82 Residences 62W, 26 Winwood, 473 CCV Townhomes) and `as_of` work on both
  endpoints. Population, lifts, and summaries re-baseline to the filtered scope.
- **Errors:** unknown cohort key → `400` whose message lists all valid keys (usable as
  a fallback prompt); `401` missing/bad API key.
- **PII:** `members_top` includes resident names/units — same internal-use convention as
  the rest of the multifamily API.
- **Adding a cohort** is a small PR: one entry in
  `contracts/multifamily_churn/cohorts.py` (label + description + predicate over the
  feature frame) plus the golden test. Ask Brandon — new cohorts should be validated
  against the population first (that's how the rent-hike traps were caught).

**Questions:** Brandon Valadez (valadez.brandon@realtyts.com)

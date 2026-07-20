# Marketing "Who Needs Space" — Team Handoff

**Feature:** cohort views over the brokerage **space-demand model** — "which marketing cohorts
actually need space right now", plus ranked in-cohort leads and outreach logging for uplift.
**Model:** `signal_desk_spacedemand v0.1-prototype` — the *same frozen bundle* the brokerage side
serves at `/brokerage/space-demand-leads` (one scoring pass, identical propensities).
**Landed:** 2026-07-20 (both repos). Companion doc: `docs/API_HANDOFF.md` (general API handoff).

> ⚠️ **What the score means (put this in any UI):** propensity that a company posts a
> **Facilities Relocation/Expansion signal within ~180 days**. It is a *prioritization list*,
> **NOT a conversion predictor** and not verified space need. Only scoop-covered companies are
> scorable (~2,165 companies, ~1,914 in IA/NE) — **a missing cohort means NO DATA, not no demand.**

---

## Connection (raw prediction API)

Same as the multifamily handoff:

| | |
|---|---|
| **Base URL (LAN)** | `http://10.1.6.17:8910` |
| **Auth** | Header `X-API-Key: rts-local-dev-key` on every call |
| **Interactive docs** | `http://10.1.6.17:8910/docs` |

The marketing app **does not call these directly from the browser** — its C# server proxies them
(section 4). The raw endpoints are documented here for testing and for any batch/ETL use.

---

## 1. Per-cohort demand rollup

`GET /marketing/space-demand` — every marketing cohort (canonical **size band × vertical**),
ranked by expected space-need events.

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/marketing/space-demand?market=IA,NE"
# optional: &min_scored=3 (hide tiny cohorts; default 3)
```

```json
{
  "contract": "marketing",
  "view": "space-demand-cohorts",
  "model_source": {"contract": "signal-desk-spacedemand", "version": "v0.1-prototype",
                   "status": "prototype", "label": "Facilities Relocation/Expansion scoop (TypeID 8) within 180d"},
  "as_of": "2026-07-20",
  "hot_definition": "propensity >= p90 of in-market scored set",
  "hot_threshold": 0.0212,
  "totals": {"scored_in_market": 1914, "hot_total": 192,
             "expected_events_180d": 42.5, "already_in_pipeline": 8},
  "cohorts": [
    {"cohort_key": "size=enterprise|vertical=healthcare", "size_band": "enterprise",
     "vertical": "healthcare", "n_scored": 26, "expected_events_180d": 2.5,
     "hot_n": 9, "hot_new_n": 9, "avg_propensity": 0.0969,
     "top_leads": [{"company": "Home Instead", "company_id": 53185174, "propensity": 0.9793}]}
  ],
  "verticals": [ ... same shape, one row per vertical across all sizes ... ]
}
```

Field semantics:
- `expected_events_180d` — **sum of calibrated propensities** in the cohort = the expected
  number of space-need signals in ~180 days. Defensible to quote as "≈2.5 expected moves".
- `hot_n` / `hot_new_n` — companies in the top decile of in-market propensity / same but
  **excluding companies already in the RTS deal pipeline** (the ones marketing should touch).
- Propensity comparisons **across size bands** are confounded by the model's large-employer
  tilt — rank *within* a cohort, or compare cohorts by `expected_events_180d`, not by
  `avg_propensity`.

## 2. Ranked leads inside one cohort

`GET /marketing/space-demand/leads?cohort=<selector>&limit=25`

The `cohort` selector accepts (URL-encode it):
- the canonical key: `size=mid|vertical=healthcare`
- a single dim: `size=mid` or `vertical=healthcare`
- a bare token: `healthcare` or `mid`

Junk selectors get a `400` whose message lists the allowed vocabulary.

```json
{
  "view": "space-demand-leads",
  "cohort": {"selector": "healthcare", "size_band": null, "vertical": "healthcare",
             "cohort_key": "vertical=healthcare"},
  "scored_in_cohort": 250, "count": 25,
  "new_vs_pipeline": {"new": 24, "already_in_pipeline": 1},
  "leads": [
    {"rank": 1, "company_id": 53185174, "company": "Home Instead", "state": "Nebraska",
     "industry": "Healthcare Services", "size_band": "enterprise", "vertical": "healthcare",
     "cohort_key": "size=enterprise|vertical=healthcare", "propensity": 0.9793,
     "hot": true, "in_pipeline": false,
     "signals": {"hiring_365d": 14, "fundmna_365d": 0, "execmove_365d": 1,
                 "project_365d": 0, "had_prior_reloc": true, "days_since_scoop": 13},
     "why": "relocated/expanded before, very recent activity, recent hiring"}
  ]
}
```

`why` is display-ready; `signals` are the raw features behind it. `in_pipeline: true` means the
company already has an open RTS deal — don't campaign at it.

## 3. Fused cohort targeting (which cohorts to campaign at)

`GET /marketing/targeting` now ranks by **`expected_opportunity` = hot NEW space-need leads ×
historical win-rate** (was: addressable × win-rate, which survives as `expected_winnable` and
is the fallback rank for cohorts with no scoop coverage). Each cohort row also carries
`cohort_key` and `needs_space_now` (the rollup stats above, or `null` = no data).

Today's fused top-5 (2026-07-20): `large·other` (5.75), `large·retail-hospitality` (2.67),
`enterprise·retail-hospitality` (2.5), `mid·other` (1.8), `enterprise·finance-insurance` (1.67)
— the two enterprise cohorts were near the *bottom* of the historical ranking and surfaced only
because they have 10 hot new leads each right now.

## 4. Marketing app integration (what's already wired)

The C# server (`marketing` repo) proxies everything — the SPA never needs the API key:

| Route (under `/api/marketing`) | Proxies | Notes |
|---|---|---|
| `GET /space-demand` | rollup (§1) | `min_scored=1` upstream; UI filters |
| `GET /space-demand/leads?cohort=&limit=` | leads (§2) | upstream 400s pass through |
| `POST /space-demand/outreach` | `POST /brokerage/leads` | see below |

- **Config (env / `.env` next to the server):** `MKT_GBM_ENDPOINT_BASE_URL`
  (e.g. `http://10.1.6.17:8910`), `MKT_GBM_ENDPOINT_API_KEY` (= the endpoint's key),
  `MKT_GBM_ENDPOINT_TIMEOUT_S` (default 20). **Key unset ⇒ routes return 503 and the UI
  simply hides the feature** — it never blocks boot or the cohort catalog.
- **Client:** `Services/GbmEndpointClient.cs` (typed, mirrors `LmStudioClient`); DTOs in
  `Models/SpaceDemand.cs` are pinned to the wire with fixture tests (`SpaceDemandModelsTests`).
- **UI already built:** Cohorts catalog rows show a green "**N need space now**" pill +
  `Hot leads (ML)` / `Exp. moves 180d` stats (overlay is best-effort — catalog renders without
  it when the endpoint is down). Cohort detail gets a "**Who needs space now**" panel: top-10
  ranked leads with propensity, why-string, in-pipeline flag, and a **Log pursued** button.

### Outreach logging (uplift loop)

`POST /api/marketing/space-demand/outreach` body:

```json
{"company_id": 53185174, "action": "pursued", "propensity": 0.9793,
 "cohort_key": "size=enterprise|vertical=healthcare", "note": "optional"}
```

Actions: `pursued | called | emailed | meeting | proposal_sent | won | lost | disqualified | skip | note`.
The server stamps `staff` from the authenticated identity and pins `source: "marketing"` —
the SPA cannot spoof attribution. Events append to the endpoint's lead-outcome log
(`monitoring/brokerage_leads.jsonl`; summary at `GET /brokerage/leads?summary=true`), with the
propensity + cohort snapshotted at contact time so per-cohort **uplift** (did pursuing hot
leads convert more?) is measurable once won/lost events accrue. **Please log `won`/`lost`
resolutions too** — that's what turns the ranking into a measurable claim.

---

## The cohort-key contract (read this before touching cohort code)

`CohortKey` (`size=<band>|vertical=<vertical>`, e.g. `size=mid|vertical=healthcare`) is the
**literal join key** across marketing's `CohortVocab.cs`, signal-desk's `ontology-lib.mjs`,
`rea.DealPattern`, `mkt.CohortCatalog`, and now the prediction API's
`contracts/marketing/cohort.py`. All three implementations must stay **byte-identical**:

- **Size bands** (from exact employee count): `micro <25 ≤ small <100 ≤ mid <500 ≤ large <2000 ≤ enterprise`.
  These are **NOT** the display bands on `/brokerage/space-demand-leads` (`emp_band`,
  50/250/1000) — never join on those.
- **Verticals** (ordered regex over the industry string, first match wins — e.g. "Accounting"
  → `finance-insurance`, not `professional-services`). Nine values; unmatched → `other`.
- Parity is pinned by live tests against `rea.DealPattern.CohortKey` in **both** repos:
  `tests/CohortVocabTests.cs` (marketing) and `test_cohort_key_parity_with_dealpattern`
  (endpoint). If you change the vocabulary anywhere, those tests are the tripwire.
- **`sizesqft=<band>` rows** in DealPattern/CohortCatalog are a *separate deal-sqft-grain
  family* (BackoffLevel 4) that overloads the SizeBand column. They are **not** company
  cohorts; both parity tests exclude them. (Heads-up: the marketing repo's parity test used
  to read them and would have failed on first real DB run — fixed 2026-07-20.)

## Known limits (honest edges)

- **Prototype model.** PR-AUC 0.335 vs 0.037 base (~9×), but the label is "posts a scoop
  signal", not "signed a lease". Keep the framing.
- **Coverage:** only scooped companies score. The three `size=micro` cohorts
  (healthcare / logistics / retail-hospitality) currently have **no coverage** — companies
  under 25 employees rarely get ZoomInfo scoops. They show `needs_space_now: null`.
- **`vertical=other` is big** (~613 companies): school districts, utilities, government-adjacent
  orgs match no vertical regex. That's faithful to `CohortVocab.VerticalOf` — same bucketing
  on your side.
- **Hot = top decile** of the in-market propensity distribution (threshold returned in every
  response, so it's auditable). It moves with the distribution, by design.
- **Freshness:** features come from `zi.Scoop`; the scored frame is TTL-cached 1 hour
  (first call after restart ~1–2 s cold, then instant).

## Ops / verify

- Endpoint service (on the Spark): `systemctl --user restart light_gbm_endpoint` — required
  once to pick up these endpoints. Goldens: `PYTHONPATH=. .venv/bin/python tests/test_smoke.py`
  (21 tests, marketing-cohort ones included).
- Marketing server: `dotnet test` (wire-shape + vocab parity tests included). The dev smoke:
  boot with `MKT_DEV_BYPASS_AUTH=true` + the two `MKT_GBM_ENDPOINT_*` vars and hit
  `GET /api/marketing/space-demand`.

**Questions:** Brandon Valadez (valadez.brandon@realtyts.com)

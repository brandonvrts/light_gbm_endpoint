# RTS Prediction API — Team Handoff

**Service:** `light_gbm_endpoint` — shared LightGBM prediction API for RTS internal apps.
**Host:** DGX Spark (CPU-only), runs as a systemd service, auto-restarts, survives reboot/logout.
**Verified live:** 2026-07-09.

## Connection

| | |
|---|---|
| **Base URL (LAN)** | `http://10.1.6.17:8910` |
| **Base URL (on the Spark itself)** | `http://127.0.0.1:8910` |
| **Auth** | Header `X-API-Key: rts-local-dev-key` on every call (except `/health`) |
| **Content type** | `Content-Type: application/json` for POST bodies |
| **Interactive docs** | `http://10.1.6.17:8910/docs` (Swagger — try calls from the browser) |
| **Browser UI** | `http://10.1.6.17:8910/` (stand-up UI, no key needed in browser) |

Quick connectivity check (no key required):

```bash
curl http://10.1.6.17:8910/health
# {"status":"ok","db":{"db":"ok",...},"contracts":{...}}
```

If `db` is not `"ok"`, the SQL Server connection is down — predictions that need live data will fail.

---

## 1. Multifamily churn (tenant renewal risk)

Model: `v4-anchor-recal` — production LightGBM, isotonic-calibrated. Returns a calibrated
move-out probability per resident plus a risk tier (`low` ≤ 0.25 < `medium` ≤ 0.45 < `high`).

Properties covered (HPROPERTY ids): **62** = CCV, **72** = Sierra Pointe, **82** = Residences 62W, **26** = Winwood.

All modes go through one endpoint: `POST /predict/multifamily-churn`. Provide **one** of the
input shapes below.

### a) Single tenant (by SCODE)

```bash
curl -s -H "X-API-Key: rts-local-dev-key" -H "Content-Type: application/json" \
  -d '{"scode":"t0014834"}' \
  http://10.1.6.17:8910/predict/multifamily-churn
```

```json
{
  "contract": "multifamily-churn",
  "model_version": "v4-anchor-recal",
  "as_of": "2026-07-09",
  "found": true,
  "scode": "t0014834",
  "full_name": "Kent Nebel",
  "unit": "13-205",
  "property": "Winwood",
  "risk_tier": "low",
  "churn_probability": 0.2379,
  "stay_probability": 0.7621,
  "has_notice": false,
  "lease_to": "2026-08-17",
  "days_to_lease_end": 38,
  "summary": "Kent Nebel (unit 13-205, Winwood) has an estimated 24% move-out risk (low) as of 2026-07-10 -- about a 76% chance of staying. Their lease ends in 38 days (2026-08-17); no move-out notice is on file."
}
```

`summary` is a ready-made natural-language sentence (chatbot-safe to read back verbatim).
If the SCODE isn't a current resident you get `"found": false`.

Don't know the SCODE? Resolve a name first:

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/multifamily/tenant/search?q=Ingle"
# -> {"count":2,"matches":[{"scode":"t0011940","full_name":"Grishma Ingle",...}, ...]}
# exactly 1 match -> the scored result is embedded under "tenant"
```

### b) Lease-expiration window (expires-from → expires-to)

Scores every resident whose lease **expires within the date range**:

```bash
curl -s -H "X-API-Key: rts-local-dev-key" -H "Content-Type: application/json" \
  -d '{"expires_from":"2026-08-01","expires_to":"2026-09-30"}' \
  http://10.1.6.17:8910/predict/multifamily-churn
```

```json
{
  "contract": "multifamily-churn",
  "model_version": "v4-anchor-recal",
  "as_of": "2026-07-09",
  "view": "lease-window",
  "window_start": "2026-08-01",
  "window_end": "2026-09-30",
  "eligible_units": 136,
  "scored_units": 136,
  "tier_thresholds": {"low_max": 0.25, "medium_max": 0.45},
  "results": [
    {"scode": "t0014834", "full_name": "Kent Nebel", "unit": "13-205",
     "property": "Winwood", "churn_probability": 0.2379, "risk_tier": "low",
     "has_notice": 0, "lease_to": "2026-08-17", "days_to_lease_end": 38}
  ]
}
```

Optional body fields: `"property_ids":[62,72]` to restrict buildings (default: all 4),
`"limit":50` to cap rows, `"as_of":"YYYY-MM-DD"` to score as of a specific date
(defaults to today for future windows, window start for past windows — past windows also
return realized `actual_churn` so you can self-verify).

### c) Other multifamily inputs (same endpoint)

```json
{"scodes": ["t0014834", "t0011940"]}      // a specific list of tenants
{"property_ids": [62], "limit": 50}       // whole building(s), ranked by risk
```

Helper: `GET /multifamily/buildings` lists the properties with ids.

---

## 2. Brokerage — space-demand leads (⚠️ prototype)

Ranks in-market (IA/NE) companies by propensity to post a **Facilities Relocation/Expansion
signal within ~180 days** — i.e., "who is likely to need space soon." This is a
**prioritization list for research, not a conversion predictor**, and not broker-facing truth.
Model `v0.1-prototype` (ROC-AUC 0.86, ~9× lift over base rate).

```bash
curl -s -H "X-API-Key: rts-local-dev-key" \
  "http://10.1.6.17:8910/brokerage/space-demand-leads?limit=25"
# optional: &market=IA  or  &market=NE
```

```json
{
  "contract": "signal-desk-spacedemand",
  "model_version": "v0.1-prototype",
  "status": "prototype",
  "as_of": "2026-07-09",
  "scored_in_market": 1906,
  "leads": [
    {
      "rank": 1,
      "company_id": 53185174,
      "company": "Home Instead",
      "state": "Nebraska",
      "propensity": 0.9826,
      "signals": {"hiring_365d": 13, "fundmna_365d": 0, "execmove_365d": 1,
                  "project_365d": 0, "had_prior_reloc": true, "days_since_scoop": 22},
      "why": "relocated/expanded before, very recent activity, recent hiring"
    }
  ]
}
```

Each lead includes a `why` explanation string and the raw `signals` behind the score.

Related brokerage endpoints (same auth):
- `POST /predict/signal-desk-brokerage` body `{"company":"840"}` (company hMy or name substring) — REA prospect score + history
- `GET /brokerage/scorecard?sort=score&limit=50` — company-centric ranking (REA score, win prob, EV, scoop momentum)
- `GET /brokerage/companies?limit=50` — company picker

---

## 3. Marketing — cohort data

Aggregate fit/targeting stats over a **cohort** (attribute / geography / lifecycle segment)
or an **industry**.

**Discover the available cohort keys first:**

```bash
curl -s -H "X-API-Key: rts-local-dev-key" http://10.1.6.17:8910/marketing/options
```

Returns all cohort keys, e.g. `high-vacancy`, `owner-occupied`, `public-company`,
`dsm-market`, `oma-market`, `distress`, `downsizing`, `expanding-footprint`,
`growth-mode`, `healthcare-vertical`, … plus available industries.

**Then score a cohort:**

```bash
curl -s -H "X-API-Key: rts-local-dev-key" -H "Content-Type: application/json" \
  -d '{"variant":"cohort","key":"healthcare-vertical"}' \
  http://10.1.6.17:8910/predict/marketing
```

```json
{
  "segment": "healthcare-vertical",
  "variant": "cohort",
  "by_market": [
    {"Market": "DSM", "n": 55, "avg_score": 11.13, "avg_win_prob": 0.21},
    {"Market": "OMA", "n": 34, "avg_score": 11.17, "avg_win_prob": 0.161}
  ],
  "decile_distribution": [{"Decile": 1, "n": 75}, ...],
  "cohort_win_rates": [{"Vertical": "healthcare", "DealCount": 169, "WinRate": 0.167, ...}]
}
```

For an industry instead: `{"variant":"industry","key":"<industry name>"}`.
Also available: `GET /marketing/targeting` — targeting summary table.

---

## Python example

```python
import requests

BASE = "http://10.1.6.17:8910"
HDRS = {"X-API-Key": "rts-local-dev-key"}

# single tenant
r = requests.post(f"{BASE}/predict/multifamily-churn",
                  json={"scode": "t0014834"}, headers=HDRS, timeout=60)
print(r.json()["summary"])

# expiring leases Aug-Sep
r = requests.post(f"{BASE}/predict/multifamily-churn",
                  json={"expires_from": "2026-08-01", "expires_to": "2026-09-30"},
                  headers=HDRS, timeout=300)
high = [t for t in r.json()["results"] if t["risk_tier"] == "high"]

# space-demand leads
leads = requests.get(f"{BASE}/brokerage/space-demand-leads",
                     params={"limit": 25}, headers=HDRS, timeout=120).json()["leads"]

# cohort data
cohort = requests.post(f"{BASE}/predict/marketing",
                       json={"variant": "cohort", "key": "dsm-market"},
                       headers=HDRS, timeout=60).json()
```

---

## Good to know

- **Latency:** single-tenant and cached calls return in well under a second. A **cold**
  lease-window or whole-building call runs the Yardi feature stored procedure and can take
  tens of seconds; results are cached (~TTL) so repeats are fast. The service pre-warms the
  all-buildings view on startup. Set generous client timeouts (60–300 s) for first calls.
- **Lease dates:** `lease_to` is the lease-expiration date; `days_to_lease_end` is counted
  from the `as_of` date (negative = already expired). If `lease_to` is null the date wasn't
  available from the feature source — the churn probability is still valid.
- **PII:** responses include resident name/unit for internal convenience — treat downstream
  storage accordingly.
- **Discovery endpoints:** `GET /contracts` (versions + manifests),
  `GET /contracts/{contract}/schema` (input/output shapes), `GET /models` (model cards:
  training setup, ROC-AUC/Brier, calibration, feature importance).
- **Errors:** `401/403` bad or missing `X-API-Key` · `400` invalid body (message says what to
  provide) · `404` unknown contract.
- **Ops (on the Spark):** `systemctl --user status light_gbm_endpoint` ·
  logs `journalctl --user -u light_gbm_endpoint -f` · restart
  `systemctl --user restart light_gbm_endpoint`. Unit is enabled with linger, so it
  survives logout and reboot.

**Questions:** Brandon Valadez (valadez.brandon@realtyts.com)

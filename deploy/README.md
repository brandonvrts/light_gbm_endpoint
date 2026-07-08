# Deploy — light_gbm_endpoint on the DGX Spark

CPU-only FastAPI service. GPU stays free for LLM/embedding workloads.

## Run (foreground / dev)
```bash
cd ~/Documents/spark/light_gbm_endpoint
./scripts/serve.sh                 # API on 0.0.0.0:8910
./scripts/serve_ui.sh              # Streamlit UI on 0.0.0.0:8911
```

## Run as a managed service (systemd user unit — survives logout)
```bash
mkdir -p ~/.config/systemd/user
cp deploy/light_gbm_endpoint.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now light_gbm_endpoint
loginctl enable-linger "$USER"     # keep running after logout
systemctl --user status light_gbm_endpoint
journalctl --user -u light_gbm_endpoint -f
```

## Endpoints (this machine)
- Local:  `http://127.0.0.1:8910`
- LAN:    `http://10.1.6.17:8910`   (Spark LAN IP — other RTS apps call this)
- Auth:   header `X-API-Key: <GBME_API_KEY from .env>`

### How each project calls it
```bash
# Multifamily — tenants or whole buildings
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"property_ids":[62],"limit":50}' http://10.1.6.17:8910/predict/multifamily-churn

# Brokerage (Signal Desk) — a company, full historical prediction series from SSMS
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"company":"840"}' http://10.1.6.17:8910/predict/signal-desk-brokerage

# Marketing — aggregate over a cohort or industry
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"variant":"cohort","key":"healthcare-vertical"}' http://10.1.6.17:8910/predict/marketing
```
Node example (Signal Desk's Express API / nightly job):
```js
const r = await fetch("http://10.1.6.17:8910/predict/signal-desk-brokerage", {
  method: "POST", headers: {"Content-Type": "application/json", "X-API-Key": process.env.GBME_API_KEY},
  body: JSON.stringify({ company: String(hMy) }) });
const { history, current_rules_score } = await r.json();
```

## Reverse proxy (optional, recommended for a stable name + TLS)
Put Caddy or nginx in front (the design's IIS analog). Example Caddyfile:
```
models.rts.local {
    reverse_proxy 127.0.0.1:8910
}
```
Then callers use `https://models.rts.local`. Bind uvicorn to 127.0.0.1 (not 0.0.0.0) once the proxy fronts it.

## Notes
- Read-only DB access (FreeTDS -> SQL Server). The service never writes to SSMS.
- Request logging: `monitoring/logs/requests.log` (PII-free: method/path/status/ms).
- Swap the interim stand-in model: drop `bundle.joblib` into `bundles/multifamily_churn/<new-version>/`
  and set `bundles/multifamily_churn/default.txt` to that version (instant promote/rollback).
- Security TODO: rotate the shared `innovation_usr` credential (it is committed in legacy repos);
  move to Azure AD JWT (the `require_key` dependency is the swap point).

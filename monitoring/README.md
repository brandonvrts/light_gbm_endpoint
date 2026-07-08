# monitoring/

Prediction logging + drift/calibration checks (Phase 3–4).

- **Log every request + prediction** with the **stamped model version** — and **no tenant names**
  (PII posture). Write to `monitoring/logs/` (gitignored).
- **Drift/calibration (Phase 4):** periodically join logged predictions to realized outcomes
  (actuals) and re-validate calibration (Brier, reliability) over time. The multifamily h180 model
  was calibrated to a validated 32.3% non-renewal rate; monitoring confirms it holds as data shifts.
- Retrain cadence: quarterly (aligned to the walk-forward folds) + drift-triggered once monitoring exists.

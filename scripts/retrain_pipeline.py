"""Retrain -> validate -> promote (with rollback) pipeline.

For each configured contract:
  1. read the INCUMBENT (current default.txt) + its validation metric,
  2. retrain a CANDIDATE bundle into its own version dir WITHOUT touching default.txt
     (trainer honors GBME_BUNDLE_VERSION + GBME_NO_PROMOTE),
  3. GATE: promote only if the candidate does not regress the primary walk-forward metric
     beyond tolerance (and is itself valid); else keep the incumbent (rollback = no-op),
  4. append an auditable record to monitoring/retrain.log.

The served systemd process picks up a promoted bundle on its next registry.load (new
version path -> fresh load). Never deletes a bundle, so any promotion is reversible by
pointing default.txt back. Safe to run on a schedule (systemd timer).

Run:  PYTHONPATH=. .venv/bin/python scripts/retrain_pipeline.py [contract]
Dry:  add --dry-run to skip the default.txt flip (still trains + logs the decision).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import registry                                   # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
BUNDLES = ROOT / "bundles"
LOG = ROOT / "monitoring" / "retrain.log"

# Each retrainable contract: how to train a candidate, and how to judge it.
# metric_path = keys into manifest.json; higher_better + regression_tol define the gate.
CONTRACTS = {
    "signal_desk_brokerage": {
        "trainer": [PY, str(ROOT / "scripts" / "train_brokerage_v03.py")],
        "candidate_prefix": "v0.3-cand",
        "metric_path": ["metrics", "pr_auc"],
        "metric_name": "rolling-origin PR-AUC",
        "higher_better": True,
        "regression_tol": 0.01,   # don't promote a candidate more than 0.01 below incumbent
        "timeout": 900,
    },
    # multifamily_churn: real v4 bundle is trained by the multifamily-prediction repo's
    # anchor-date pipeline (not local). Wire here once that trainer is vendored:
    #   "trainer": [PY, "scripts/train_multifamily_v5.py"], "metric_path": ["metrics","walkforward","mean_auc"]
    # marketing: no trained model (live cohort aggregation) -> nothing to retrain.
}


def _dig(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def run_contract(name: str, dry: bool) -> dict:
    cfg = CONTRACTS[name]
    incumbent = registry.default_version(name)
    inc_metric = _dig(registry.manifest(name, incumbent) or {}, cfg["metric_path"]) if incumbent else None

    cand = f"{cfg['candidate_prefix']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = {**os.environ, "GBME_BUNDLE_VERSION": cand, "GBME_NO_PROMOTE": "1", "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(cfg["trainer"], env=env, cwd=str(ROOT),
                          capture_output=True, text=True, timeout=cfg["timeout"])
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "contract": name,
           "incumbent": incumbent, "incumbent_metric": inc_metric, "candidate": cand,
           "metric_name": cfg["metric_name"]}
    if proc.returncode != 0:
        rec.update({"decision": "TRAIN_FAILED", "reason": (proc.stderr or "")[-400:]})
        return rec

    cand_metric = _dig(registry.manifest(name, cand) or {}, cfg["metric_path"])
    rec["candidate_metric"] = cand_metric
    tol = cfg["regression_tol"]
    if cand_metric is None:
        promote, reason = False, "candidate metric missing"
    elif inc_metric is None:
        promote, reason = True, "no incumbent"
    elif cfg["higher_better"]:
        promote = cand_metric >= inc_metric - tol
        reason = (f"{cand_metric:.4f} >= {inc_metric:.4f}-{tol} (no regression)" if promote
                  else f"REGRESSION {cand_metric:.4f} < {inc_metric:.4f}-{tol}")
    else:
        promote = cand_metric <= inc_metric + tol
        reason = f"lower-better gate: {promote}"

    rec.update({"decision": "PROMOTE" if promote else "KEEP_INCUMBENT", "reason": reason, "dry_run": dry})
    if promote and not dry:
        (BUNDLES / name / "default.txt").write_text(cand)
        rec["default_now"] = cand
    else:
        rec["default_now"] = incumbent
    return rec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    targets = args or list(CONTRACTS)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for name in targets:
        if name not in CONTRACTS:
            print(f"skip {name}: not a retrainable contract"); continue
        print(f"\n=== retrain {name} (dry_run={dry}) ===")
        try:
            rec = run_contract(name, dry)
        except Exception as e:  # noqa: BLE001
            rec = {"ts": datetime.now().isoformat(timespec="seconds"), "contract": name,
                   "decision": "ERROR", "reason": str(e)[:300]}
        with LOG.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print("  " + json.dumps({k: rec.get(k) for k in
              ("incumbent", "incumbent_metric", "candidate", "candidate_metric", "decision", "reason")}, default=str))
    print("\nDONE (audit: monitoring/retrain.log)")


if __name__ == "__main__":
    main()

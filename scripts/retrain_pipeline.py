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
        "trainer": [PY, str(ROOT / "scripts" / "train_brokerage_v05.py")],
        "candidate_prefix": "v0.5-cand",
        "metric_path": ["metrics", "pr_auc"],   # honest nested-grouped-CV combined PR-AUC
        "metric_name": "nested-grouped-CV PR-AUC (combined head)",
        "higher_better": True,
        # RELATIVE tolerance (Phase 1.4): 0.01 absolute was ~10% of the metric. Allow a
        # candidate within 15% of the incumbent (small-n PR-AUC is noisy fold-to-fold).
        "regression_tol_rel": 0.15,
        "timeout": 1200,
    },
    "multifamily_churn": {
        "trainer": [PY, str(ROOT / "scripts" / "train_multifamily_v6.py"), "--horizon", "60"],
        "candidate_prefix": "v6-cand",
        # Gate on the OUT-OF-TIME EXPIRING-COHORT AUC, not walk-forward mean_auc: the full
        # population is ~50% no-lease residents (a trivially-separable near-zero-churn
        # stratum) that inflates population AUC toward "lease vs no-lease" separation. The
        # expiring cohort is 100% lease-holders and is the population we actually serve, so
        # its AUC is the honest decision-level discrimination number.
        "metric_path": ["metrics", "oot_expiring_holdout", "auc"],
        "metric_name": "OOT expiring-cohort AUC",
        "higher_better": True,
        "regression_tol": 0.03,
        # secondary: the OOT cohort's predicted vs realized churn must stay within band,
        # else the candidate is miscalibrated even at good AUC.
        "calibration_gate": {"path": ["metrics", "oot_expiring_holdout"],
                             "max_abs_rate_error": 0.06},
        "timeout": 1800,
    },
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

    cand_manifest = registry.manifest(name, cand) or {}
    cand_metric = _dig(cand_manifest, cfg["metric_path"])
    rec["candidate_metric"] = cand_metric
    # absolute tolerance, or a relative fraction of the incumbent metric (Phase 1.4)
    if "regression_tol_rel" in cfg and inc_metric:
        tol = abs(inc_metric) * cfg["regression_tol_rel"]
    else:
        tol = cfg.get("regression_tol", 0.0)
    # Phase 1.4: a deterministic retrain on UNCHANGED data reproduces the incumbent exactly
    # -> log NOOP instead of a vacuous "PROMOTE" that churns default.txt for no reason.
    inc_shape = _dig(registry.manifest(name, incumbent) or {}, ["metrics", "n_pos_rows"])
    cand_shape = _dig(cand_manifest, ["metrics", "n_pos_rows"])
    if (inc_metric is not None and cand_metric == inc_metric and inc_shape == cand_shape):
        rec.update({"decision": "NOOP_IDENTICAL", "candidate_metric": cand_metric,
                    "reason": "deterministic retrain reproduced incumbent (unchanged data); "
                              "keeping current default", "dry_run": dry,
                    "default_now": incumbent})
        return rec
    if cand_metric is None:
        promote, reason = False, "candidate metric missing"
    elif inc_metric is None:
        promote, reason = True, "no incumbent"
    elif cfg["higher_better"]:
        promote = cand_metric >= inc_metric - tol
        reason = (f"{cand_metric:.4f} >= {inc_metric:.4f}-{tol:.4f} (no regression)" if promote
                  else f"REGRESSION {cand_metric:.4f} < {inc_metric:.4f}-{tol:.4f}")
    else:
        promote = cand_metric <= inc_metric + tol
        reason = f"lower-better gate: {promote}"

    # secondary gate: cohort calibration error must be within band (a model can hold AUC
    # while its published rate drifts -- that would mislead the property team's planning)
    cg = cfg.get("calibration_gate")
    if promote and cg:
        h = _dig(cand_manifest, cg["path"]) or {}
        pred, real = h.get("predicted_rate"), h.get("realized_rate")
        if pred is None or real is None:
            promote, reason = False, "calibration-gate metrics missing"
        elif abs(pred - real) > cg["max_abs_rate_error"]:
            promote = False
            reason = (f"CALIBRATION off: |{pred:.3f}-{real:.3f}|="
                      f"{abs(pred - real):.3f} > {cg['max_abs_rate_error']}")
        else:
            reason += f"; calibration ok (|{pred:.3f}-{real:.3f}|)"

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

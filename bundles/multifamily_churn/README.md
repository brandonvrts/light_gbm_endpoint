# bundles/multifamily_churn/

Frozen, **immutable** model artifacts. `.joblib` files are **gitignored** (distributed out-of-band, not via git).

Expected layout once Phase 0 runs:
```
multifamily_churn/
├── default.txt            # the default-version pointer, e.g. "v4.0.0"
└── v4.0.0/
    ├── bundle_h180.joblib  # recalibrated LightGBM (isotonic + 32.3% anchor), clip [0.01,0.99]
    ├── bundle_h60.joblib   # ranker variant
    └── manifest.json       # contract, version, variants, algorithm, calibration.method,
                            # feature count (45), feature_pipeline_version, created, owner
```

**Phase 0 freeze:** copy the validated bundles from `../../../multifamily-prediction/v2/models/`
(`churn_model_v4_h180_*` after `recalibrate_h180.py`, and `churn_model_v4_h60_*`), record their
real `algorithm` / `calibration.method` / `len(feature_columns)` into `manifest.json`, and pin the
training-env library versions in the service venv. Versions are immutable — a fix is a new version.

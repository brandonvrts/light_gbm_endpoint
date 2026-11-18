"""v8 renewal-decision churn model (point-in-time safe).

Ported from RealtyRTS/multifamily-prediction@feature/mu-fa25-mom-churn-reports (commit
195cceb, `v2/`): pit_features.py, pit_training.py, extra_features.py and
train_renewal_model.py are copied verbatim except for the `config` import (see _cfg.py).

Target: ChurnedAtLeaseEnd -- does the resident move out within [lease_end - 30d,
lease_end + 90d] -- for leases ending 45-210 days after the scoring date.
"""

"""The 5-part Contract abstraction. STATUS: stub (interface only).

A contract = input schema + feature pipeline + model bundle + post-processing +
output schema, versioned together (docs/CONTRACTS.md, docs/ARCHITECTURE.md §2).
Generic across contracts; concrete wiring lives in contracts/{name}/contract.py.
"""
from __future__ import annotations
from typing import Any, Protocol


class Contract(Protocol):
    name: str               # "multifamily-churn"
    version: str            # resolved version, e.g. "v4.0.0" — pins all five parts
    feature_mode: str       # "identifier-in" | "feature-in"
    variants: list[str]     # ["h60", "h180"]

    def validate(self, inputs: dict) -> dict:
        """Validate request against the input schema (part 1)."""

    def compute_features(self, inputs: dict) -> Any:
        """Part 2. identifier-in: call feature_io -> Yardi SP -> prepare_features.
        feature-in: validate/order the supplied vector."""

    def predict(self, features: Any) -> Any:
        """Part 3. Run the bundle's model on the prepared features (raw output)."""

    def post_process(self, raw: Any, model_version: str) -> dict:
        """Part 4+5. Calibrate -> clip -> tiers -> output schema (PII-free), version-stamped."""

    def describe(self) -> dict:
        """Return input/output schema for GET /contracts/{c}/schema."""

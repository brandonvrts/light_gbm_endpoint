"""signal-desk-brokerage contract metadata. feature-in conceptually, but v1 is
identifier-in by hMy: the service pulls the company's historical ProspectFeatureSnapshot
rows from SSMS and scores each (the 'run through all historical predictions' path).
"""
from __future__ import annotations

NAME = "signal-desk-brokerage"
FEATURE_MODE = "identifier-in"   # caller passes company hMy; service pulls history from rea.*
VARIANTS = ["conversion-180d"]

# 37 numeric/bit features from rea.ProspectFeatureSnapshot (string zi_* range cols excluded).
FEATURES = [
    "prospect_age_days", "source_code", "brand", "sales_category", "is_specialty",
    "is_cml_lead", "why_moving_type", "contact_method", "pipeline_status", "pipeline_stage",
    "pipeline_probability", "stage_transitions_count", "days_in_current_stage",
    "had_stage_regression", "activity_count_30d", "activity_count_90d", "activity_count_365d",
    "days_since_last_activity", "first_response_hours", "unique_agents", "zi_revenue_k_usd",
    "zi_employee_count", "zi_company_age_years", "scoop_count_90d", "scoop_count_365d",
    "days_since_last_scoop", "had_funding_scoop_180d", "had_leadership_move_180d",
    "had_office_relo_365d", "had_hiring_scoop_365d", "had_layoff_scoop_180d", "tech_count",
    "tech_has_crm", "linked_contact_count", "has_c_level_contact", "max_contact_accuracy",
    "is_linked_to_zi",
]
LABEL = "Label"  # rea.ProspectLabelSnapshot.Label (converted within window)


def describe() -> dict:
    return {
        "name": NAME, "feature_mode": FEATURE_MODE, "variants": VARIANTS,
        "input": {"company": "hMy (numeric) or name string", "history": "bool (all snapshots)"},
        "output": {"model_version": "str", "company": "hMy + display_name",
                   "history": [{"snapshot_date": "date", "conversion_probability": "float",
                                "decile": "int", "rules_score": "float (current rea.ProspectScore)"}]},
    }

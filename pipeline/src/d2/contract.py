"""Frozen D2-S V1 score contract.

This module is the human- and machine-readable SCORE_CONTRACT.  Changing
any constant here is a contract change and requires a new ``SCORE_CONTRACT_ID``.
"""

from __future__ import annotations

from typing import Literal

SCORE_CONTRACT_ID = "d2s-v1.0.0-pairwise8-20260816"

SCORE_CONTRACT_STATEMENT = (
    "A non-LLM statistical application-consistency reviewer that learns "
    "stable legitimate application relationships from Months 0–5 and "
    "assigns an inconsistency score to a submitted application."
)

D2_IS_NOT = (
    "a second fraud classifier",
    "an XGBoost replacement",
    "an LLM semantic reviewer",
    "a REJECT decision system",
)

DECISION_LABELS: tuple[str, ...] = ("CLEAR", "REVIEW")
# REVIEW means escalation to human/additional verification, not rejection.
# V1 of the scorer does not apply a CLEAR/REVIEW threshold.

REFERENCE_MONTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
CALIBRATION_MONTHS: tuple[int, ...] = (6,)
SEALED_MONTHS: tuple[int, ...] = (7,)

REFERENCE_FRAUD_BOOL = 0

RELATIONSHIP_IDS: tuple[str, ...] = (
    "C01",
    "C14",
    "C13",
    "C09",
    "C03",
    "C10",
    "C11",
    "C15",
)

Editability = Literal["E", "I"]


class RelationshipSpec:
    """One qualified pairwise relationship. Editability is metadata only."""

    __slots__ = (
        "relationship_id",
        "x_field",
        "y_field",
        "x_editability",
        "y_editability",
        "description",
    )

    def __init__(
        self,
        relationship_id: str,
        x_field: str,
        y_field: str,
        x_editability: Editability,
        y_editability: Editability,
        description: str,
    ) -> None:
        self.relationship_id = relationship_id
        self.x_field = x_field
        self.y_field = y_field
        self.x_editability = x_editability
        self.y_editability = y_editability
        self.description = description

    @property
    def editability_pair(self) -> str:
        return f"{self.x_editability}\u2194{self.y_editability}"


RELATIONSHIP_SPECS: tuple[RelationshipSpec, ...] = (
    RelationshipSpec(
        "C01",
        "payment_type",
        "bank_months_count_presence",
        "E",
        "I",
        "Rarity of bank_months_count presence given payment_type.",
    ),
    RelationshipSpec(
        "C14",
        "payment_type",
        "intended_balcon_amount_presence",
        "E",
        "E",
        "Rarity of intended_balcon_amount presence given payment_type.",
    ),
    RelationshipSpec(
        "C13",
        "current_address_tenure_bin",
        "prev_address_months_count_missingness",
        "E",
        "E",
        "Rarity of previous-address missingness given current-address tenure.",
    ),
    RelationshipSpec(
        "C09",
        "housing_status",
        "current_address_tenure_bin",
        "E",
        "E",
        "Rarity of current-address tenure given housing_status.",
    ),
    RelationshipSpec(
        "C03",
        "customer_age",
        "date_of_birth_distinct_emails_4w_bin",
        "E",
        "I",
        "Rarity of DOB-email-history bin given customer_age.",
    ),
    RelationshipSpec(
        "C10",
        "housing_status",
        "customer_age",
        "E",
        "E",
        "Rarity of customer_age given housing_status.",
    ),
    RelationshipSpec(
        "C11",
        "employment_status",
        "customer_age",
        "E",
        "E",
        "Rarity of customer_age given employment_status.",
    ),
    RelationshipSpec(
        "C15",
        "phone_home_valid",
        "phone_mobile_valid",
        "E",
        "E",
        "Rarity of mobile-phone validity given home-phone validity.",
    ),
)

REQUIRED_APPLICATION_FIELDS: tuple[str, ...] = (
    "payment_type",
    "bank_months_count",
    "intended_balcon_amount",
    "prev_address_months_count",
    "current_address_months_count",
    "housing_status",
    "customer_age",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "phone_home_valid",
    "phone_mobile_valid",
)

# Keys that must never influence D2-S scoring.  If present they are ignored.
FORBIDDEN_INFERENCE_KEYS: frozenset[str] = frozenset(
    {
        "fraud_bool",
        "d1_score",
        "d1_probability",
        "d1_proba",
        "d1_risk_score",
        "risk_score",
        "y_score",
        "shap",
        "shap_values",
        "attacker_id",
        "attacker_identity",
        "attacker_kind",
        "attack_history",
        "changed_field_mask",
        "changed_fields",
        "edited_fields",
        "provenance",
        "reference_pool",
        "reference_pool_membership",
        "reference_id",
        "attack_success",
        "success",
        "month",
    }
)

# Additive smoothing for empirical conditionals.  Not a decision threshold.
LAPLACE_ALPHA = 1.0

CURRENT_ADDRESS_N_BINS = 5
DOB_EMAIL_N_BINS = 4

# V1 aggregation: equal-weight mean of the eight standardized scores.
# Confirmed only after the Months 0–5 score-score redundancy matrix is
# inspected.  Editability (E↔I vs E↔E) is never a numeric weight.
AGGREGATION_METHOD_ID = "equal_mean_v1"
AGGREGATION_FORMULA = (
    "d2_score = (s_C01 + s_C14 + s_C13 + s_C09 + s_C03 + s_C10 + s_C11 + s_C15) / 8"
)
REDUNDANCY_REVIEW_SPEARMAN_FLAG = 0.70

STANDARDISATION_METHOD_ID = "ref_cdf_strict_less"
STANDARDISATION_FORMULA = (
    "raw_rarity = 1 - P_hat_Laplace(y | x); "
    "s = P_ref(R < raw_rarity); "
    "unseen conditioner or rarity above the reference maximum maps to 1"
)


def relationship_spec(relationship_id: str) -> RelationshipSpec:
    for spec in RELATIONSHIP_SPECS:
        if spec.relationship_id == relationship_id:
            return spec
    raise KeyError(f"Unknown relationship id: {relationship_id!r}")


def score_contract_payload() -> dict[str, object]:
    """JSON-serialisable snapshot of the frozen contract."""
    return {
        "score_contract_id": SCORE_CONTRACT_ID,
        "statement": SCORE_CONTRACT_STATEMENT,
        "is_not": list(D2_IS_NOT),
        "decision_labels": list(DECISION_LABELS),
        "threshold_in_scorer": False,
        "reference_months": list(REFERENCE_MONTHS),
        "calibration_months": list(CALIBRATION_MONTHS),
        "sealed_months": list(SEALED_MONTHS),
        "reference_fraud_bool": REFERENCE_FRAUD_BOOL,
        "relationship_ids": list(RELATIONSHIP_IDS),
        "higher_order_relationships": [],
        "aggregation_method_id": AGGREGATION_METHOD_ID,
        "aggregation_formula": AGGREGATION_FORMULA,
        "standardisation_method_id": STANDARDISATION_METHOD_ID,
        "standardisation_formula": STANDARDISATION_FORMULA,
        "laplace_alpha": LAPLACE_ALPHA,
        "editability_is_weight": False,
        "forbidden_inference_keys": sorted(FORBIDDEN_INFERENCE_KEYS),
        "required_application_fields": list(REQUIRED_APPLICATION_FIELDS),
        "relationships": [
            {
                "id": spec.relationship_id,
                "x": spec.x_field,
                "y": spec.y_field,
                "editability": spec.editability_pair,
                "description": spec.description,
            }
            for spec in RELATIONSHIP_SPECS
        ],
    }

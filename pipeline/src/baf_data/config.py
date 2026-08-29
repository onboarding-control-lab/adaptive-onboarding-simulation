"""Single source of truth for every frozen data-layer decision.

All other modules read these constants; no column name, sentinel rule or
split boundary may be duplicated elsewhere. Frozen decisions recorded on
28 July 2026:

- ``fraud_bool`` is the target only and never enters the feature matrix.
- ``month`` is split-only and never enters the feature matrix.
- Temporal split: months 0-5 train, month 6 development, month 7 test.
- Primary exclusions: ``device_fraud_count`` (constant in Base.csv),
  ``days_since_request`` (excluded from all current training),
  ``credit_risk_score`` (reserved for a possible later sensitivity
  experiment).
- Sentinel rules are limited to the six verified columns below; all other
  negative values (for example valid negative ``velocity_6h``) are
  preserved.
- Six feature columns are declared strictly binary (0/1) and must be
  kept unscaled by model preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ColumnKind = Literal["integer", "float", "string"]
SentinelStrategy = Literal["equals", "below"]


@dataclass(frozen=True)
class ColumnSpec:
    """Name and broad dtype kind of one raw column."""

    name: str
    kind: ColumnKind


@dataclass(frozen=True)
class SentinelRule:
    """One verified sentinel-to-missing mapping.

    ``equals``: values exactly equal to ``value`` become NaN.
    ``below``: values strictly less than ``value`` become NaN.
    """

    column: str
    strategy: SentinelStrategy
    value: float


#: Raw schema of Base.csv in file order (32 columns, 1,000,000 rows).
RAW_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("fraud_bool", "integer"),
    ColumnSpec("income", "float"),
    ColumnSpec("name_email_similarity", "float"),
    ColumnSpec("prev_address_months_count", "integer"),
    ColumnSpec("current_address_months_count", "integer"),
    ColumnSpec("customer_age", "integer"),
    ColumnSpec("days_since_request", "float"),
    ColumnSpec("intended_balcon_amount", "float"),
    ColumnSpec("payment_type", "string"),
    ColumnSpec("zip_count_4w", "integer"),
    ColumnSpec("velocity_6h", "float"),
    ColumnSpec("velocity_24h", "float"),
    ColumnSpec("velocity_4w", "float"),
    ColumnSpec("bank_branch_count_8w", "integer"),
    ColumnSpec("date_of_birth_distinct_emails_4w", "integer"),
    ColumnSpec("employment_status", "string"),
    ColumnSpec("credit_risk_score", "integer"),
    ColumnSpec("email_is_free", "integer"),
    ColumnSpec("housing_status", "string"),
    ColumnSpec("phone_home_valid", "integer"),
    ColumnSpec("phone_mobile_valid", "integer"),
    ColumnSpec("bank_months_count", "integer"),
    ColumnSpec("has_other_cards", "integer"),
    ColumnSpec("proposed_credit_limit", "float"),
    ColumnSpec("foreign_request", "integer"),
    ColumnSpec("source", "string"),
    ColumnSpec("session_length_in_minutes", "float"),
    ColumnSpec("device_os", "string"),
    ColumnSpec("keep_alive_session", "integer"),
    ColumnSpec("device_distinct_emails_8w", "integer"),
    ColumnSpec("device_fraud_count", "integer"),
    ColumnSpec("month", "integer"),
)


@dataclass(frozen=True)
class DataLayerConfig:
    """Immutable bundle of every frozen data-layer decision."""

    expected_sha256: str
    expected_rows: int
    expected_column_count: int
    raw_columns: tuple[ColumnSpec, ...]
    target_column: str
    split_column: str
    split_months: dict[str, tuple[int, ...]] = field(hash=False)
    excluded_features: dict[str, str] = field(hash=False)
    sentinel_rules: tuple[SentinelRule, ...] = ()
    #: Feature columns whose values are strictly 0/1 and must be kept
    #: unscaled by model preprocessing. Mirrored (non-executably) by the
    #: "Retain as binary" rows of ``config/feature_handling.csv``.
    binary_features: tuple[str, ...] = ()

    @property
    def raw_column_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.raw_columns)

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self.split_months)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Primary feature matrix columns, in raw file order.

        Excludes the target, the split-only column and every frozen
        primary exclusion.
        """
        dropped = {self.target_column, self.split_column, *self.excluded_features}
        return tuple(name for name in self.raw_column_names if name not in dropped)

    def validate(self) -> None:
        """Fail fast if the frozen configuration is internally inconsistent."""
        names = self.raw_column_names
        if len(set(names)) != len(names):
            raise ValueError("Duplicate column names in the frozen raw schema.")
        if len(names) != self.expected_column_count:
            raise ValueError(
                f"Frozen schema lists {len(names)} columns; "
                f"expected {self.expected_column_count}."
            )
        for required in (self.target_column, self.split_column):
            if required not in names:
                raise ValueError(f"Column '{required}' missing from the frozen schema.")
        for excluded in self.excluded_features:
            if excluded not in names:
                raise ValueError(f"Excluded column '{excluded}' is not a raw column.")
        for rule in self.sentinel_rules:
            if rule.column not in names:
                raise ValueError(f"Sentinel rule targets unknown column '{rule.column}'.")
            if rule.column in (self.target_column, self.split_column):
                raise ValueError(
                    f"Sentinel rule must not target '{rule.column}' "
                    "(target/split column)."
                )
        kinds = {spec.name: spec.kind for spec in self.raw_columns}
        features = set(self.feature_columns)
        if len(set(self.binary_features)) != len(self.binary_features):
            raise ValueError("Duplicate entries in binary_features.")
        for binary in self.binary_features:
            if binary not in names:
                raise ValueError(f"Binary feature '{binary}' is not a raw column.")
            if kinds[binary] != "integer":
                raise ValueError(
                    f"Binary feature '{binary}' must be integer-kind, "
                    f"got '{kinds[binary]}'."
                )
            if binary not in features:
                raise ValueError(
                    f"Binary feature '{binary}' is not a feature column "
                    "(target, split-only or excluded)."
                )
        seen_months: set[int] = set()
        for split_name, months in self.split_months.items():
            if not months:
                raise ValueError(f"Split '{split_name}' has no months assigned.")
            overlap = seen_months.intersection(months)
            if overlap:
                raise ValueError(f"Months {sorted(overlap)} assigned to multiple splits.")
            seen_months.update(months)


#: The frozen configuration used by the pipeline and the model layer.
FROZEN_CONFIG = DataLayerConfig(
    expected_sha256="7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809",
    expected_rows=1_000_000,
    expected_column_count=32,
    raw_columns=RAW_COLUMNS,
    target_column="fraud_bool",
    split_column="month",
    split_months={
        "train": (0, 1, 2, 3, 4, 5),
        "dev": (6,),
        "test": (7,),
    },
    excluded_features={
        "device_fraud_count": "Constant (always 0) in Base.csv; carries no information.",
        "days_since_request": "Frozen exclusion from all current training.",
        "credit_risk_score": (
            "Reserved for a possible later sensitivity experiment; "
            "values are left untouched before exclusion."
        ),
    },
    sentinel_rules=(
        SentinelRule("prev_address_months_count", "equals", -1),
        SentinelRule("current_address_months_count", "equals", -1),
        SentinelRule("intended_balcon_amount", "below", 0),
        SentinelRule("bank_months_count", "equals", -1),
        SentinelRule("session_length_in_minutes", "equals", -1),
        SentinelRule("device_distinct_emails_8w", "equals", -1),
    ),
    binary_features=(
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "keep_alive_session",
    ),
)

FROZEN_CONFIG.validate()

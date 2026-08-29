"""Safe rehearsal fixture: synthetic schema, no Month-7 rows.

Categorical values are remapped onto the frozen D1 vocabulary so the real
pipeline can score. Month 7 is dropped before any scoring or pool construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attack_lab.cases import StartingCase
from attack_lab.defender import FrozenXGBoostDefender
from baf_data.config import FROZEN_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[3]
ROWS_PER_MONTH = 6


class RehearsalFixtureError(RuntimeError):
    """Raised when a safe rehearsal fixture cannot be built."""


def make_synthetic_frame() -> pd.DataFrame:
    """Deterministic small frame matching the frozen raw schema."""
    rng = np.random.default_rng(20260728)
    n = ROWS_PER_MONTH * 8
    month = np.repeat(np.arange(8), ROWS_PER_MONTH)
    fraud = np.zeros(n, dtype="int64")
    fraud[::ROWS_PER_MONTH] = 1  # one fraud case per month

    df = pd.DataFrame(
        {
            "fraud_bool": fraud,
            "income": rng.uniform(0.1, 0.9, n),
            "name_email_similarity": rng.uniform(0, 1, n),
            "prev_address_months_count": np.where(
                np.arange(n) % 3 == 0, -1, rng.integers(0, 380, n)
            ).astype("int64"),
            "current_address_months_count": np.where(
                np.arange(n) % 8 == 0, -1, rng.integers(0, 420, n)
            ).astype("int64"),
            "customer_age": rng.integers(10, 90, n).astype("int64"),
            "days_since_request": rng.uniform(0, 78, n),
            "intended_balcon_amount": np.where(
                np.arange(n) % 2 == 0, rng.uniform(-16, -0.01, n), rng.uniform(0, 112, n)
            ),
            "payment_type": np.array(["AA", "AB", "AC", "AD", "AE"])[np.arange(n) % 5],
            "zip_count_4w": rng.integers(1, 6700, n).astype("int64"),
            "velocity_6h": np.concatenate(
                [np.array([-170.6, -5.5]), rng.uniform(0, 16000, n - 2)]
            ),
            "velocity_24h": rng.uniform(1300, 9500, n),
            "velocity_4w": rng.uniform(2825, 6994, n),
            "bank_branch_count_8w": rng.integers(0, 2385, n).astype("int64"),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 39, n).astype("int64"),
            "employment_status": np.array(["CA", "CB", "CC", "CD", "CE", "CF", "CG"])[
                np.arange(n) % 7
            ],
            "credit_risk_score": np.concatenate(
                [np.array([-1, -170, -42]), rng.integers(0, 389, n - 3)]
            ).astype("int64"),
            "email_is_free": rng.integers(0, 2, n).astype("int64"),
            "housing_status": np.array(["BA", "BB", "BC", "BD", "BE", "BF", "BG"])[
                np.arange(n) % 7
            ],
            "phone_home_valid": rng.integers(0, 2, n).astype("int64"),
            "phone_mobile_valid": rng.integers(0, 2, n).astype("int64"),
            "bank_months_count": np.where(
                np.arange(n) % 4 == 0, -1, rng.integers(0, 32, n)
            ).astype("int64"),
            "has_other_cards": rng.integers(0, 2, n).astype("int64"),
            "proposed_credit_limit": rng.choice([190.0, 200.0, 500.0, 2100.0], n),
            "foreign_request": rng.integers(0, 2, n).astype("int64"),
            "source": np.array(["INTERNET", "TELEAPP"])[np.arange(n) % 2],
            "session_length_in_minutes": np.where(
                np.arange(n) % 6 == 0, -1.0, rng.uniform(0, 85, n)
            ),
            "device_os": np.array(["windows", "linux", "macintosh", "x11", "other"])[
                np.arange(n) % 5
            ],
            "keep_alive_session": rng.integers(0, 2, n).astype("int64"),
            "device_distinct_emails_8w": np.where(
                np.arange(n) % 12 == 0, -1, rng.integers(0, 3, n)
            ).astype("int64"),
            "device_fraud_count": np.zeros(n, dtype="int64"),
            "month": month.astype("int64"),
        }
    )
    assert tuple(df.columns) == FROZEN_CONFIG.raw_column_names
    return df


def _synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


def drop_month7(frame: pd.DataFrame) -> pd.DataFrame:
    if "month" not in frame.columns:
        raise RehearsalFixtureError("Fixture frame missing month.")
    kept = frame.loc[~frame["month"].astype(int).eq(7)].copy()
    if 7 in set(int(m) for m in kept["month"].unique()):
        raise RehearsalFixtureError("Month 7 rows remained after the drop.")
    return kept


def remap_categoricals_to_d1(
    frame: pd.DataFrame,
    defender: FrozenXGBoostDefender,
) -> pd.DataFrame:
    work = frame.copy()
    vocab = defender.categorical_vocabularies()
    for name, categories in vocab.items():
        if name not in work.columns or not categories:
            continue
        allowed = set(categories)
        fallback = categories[0]
        work[name] = work[name].map(lambda value, _allowed=allowed, _fb=fallback: value if value in _allowed else _fb)
    return work


@dataclass(frozen=True)
class RehearsalFixture:
    anchors: list[StartingCase]
    training_frame: pd.DataFrame
    raw_frame: pd.DataFrame
    month7_rows_retained: bool


def build_rehearsal_fixture(
    defender: FrozenXGBoostDefender,
    *,
    n_anchors: int = 2,
    synthetic_override: pd.DataFrame | None = None,
) -> RehearsalFixture:
    raw = (
        synthetic_override.copy()
        if synthetic_override is not None
        else _synthetic_frame()
    )
    safe = drop_month7(raw)
    prepared = remap_categoricals_to_d1(safe, defender)

    month6 = prepared.loc[prepared["month"].eq(6)].copy()
    if len(month6) < n_anchors:
        raise RehearsalFixtureError("Not enough Month-6 rows in fixture.")
    feature_cols = list(FROZEN_CONFIG.feature_columns)
    missing = [col for col in feature_cols if col not in month6.columns]
    if missing:
        raise RehearsalFixtureError(f"Fixture missing features: {missing}")

    anchors: list[StartingCase] = []
    for i in range(n_anchors):
        row = month6.iloc[i]
        feats = {col: row[col] for col in feature_cols}
        feats["income"] = 0.95
        anchors.append(
            StartingCase(
                case_id=f"rehearsal_{i+1}",
                source_row_id=900000 + i,
                label=1,
                features=feats,
                initial_score=0.99,
                initial_decision="BLOCK",
                data_split="rehearsal_synthetic",
            )
        )
    return RehearsalFixture(
        anchors=anchors,
        training_frame=prepared,
        raw_frame=raw,
        month7_rows_retained=False,
    )

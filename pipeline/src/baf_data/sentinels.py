"""In-memory normalisation of verified sentinel values to NaN.

Only the frozen, per-column rules in the configuration are applied.
Negative values are *not* treated as missing in general: valid negative
``velocity_6h`` values and all ``credit_risk_score`` values are
deliberately left untouched.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from baf_data.config import SentinelRule

logger = logging.getLogger(__name__)


def normalise_sentinels(
    df: pd.DataFrame, rules: tuple[SentinelRule, ...]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a new dataframe with sentinel values replaced by NaN.

    The input dataframe is never mutated and no rows are dropped.
    Affected integer columns are promoted to float64 so they can hold
    NaN. Returns the normalised copy and a per-column count of converted
    values (zero counts included, so the mapping is deterministic).
    """
    result = df.copy()
    conversion_counts: dict[str, int] = {}

    for rule in rules:
        series = result[rule.column]
        if rule.strategy == "equals":
            mask = series == rule.value
        else:  # "below"
            mask = series < rule.value
        count = int(mask.sum())
        conversion_counts[rule.column] = count
        if count:
            converted = series.astype("float64")
            converted[mask] = np.nan
            result[rule.column] = converted
        logger.info(
            "Sentinel rule %s(%s %s): %d value(s) set to NaN.",
            rule.column,
            "==" if rule.strategy == "equals" else "<",
            rule.value,
            count,
        )

    if len(result) != len(df):
        raise AssertionError("Sentinel normalisation must never change the row count.")
    return result, conversion_counts

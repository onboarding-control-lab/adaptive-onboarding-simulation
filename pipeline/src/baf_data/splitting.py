"""Deterministic temporal train/development/test split by month."""

from __future__ import annotations

import logging

import pandas as pd

from baf_data.config import DataLayerConfig
from baf_data.errors import SplitValidationError

logger = logging.getLogger(__name__)


def build_temporal_indices(
    df: pd.DataFrame, config: DataLayerConfig
) -> dict[str, pd.Index]:
    """Assign every row to exactly one split based on the month column.

    Returns a mapping of split name to row index (positions in the
    dataframe's index). The result is fully determined by the month
    column and the frozen split definition; no randomness is involved.
    """
    months = df[config.split_column]

    assigned_months = {m for values in config.split_months.values() for m in values}
    observed_months = set(months.unique())
    unassigned = observed_months - assigned_months
    if unassigned:
        raise SplitValidationError(
            f"Month value(s) {sorted(unassigned)} are not assigned to any split. "
            "The frozen split definition does not cover this data."
        )

    indices: dict[str, pd.Index] = {}
    for split_name, split_months in config.split_months.items():
        indices[split_name] = df.index[months.isin(split_months)]
        logger.info("Split '%s' (months %s): %d rows.",
                    split_name, list(split_months), len(indices[split_name]))

    validate_split_indices(indices, df, config)
    return indices


def validate_split_indices(
    indices: dict[str, pd.Index], df: pd.DataFrame, config: DataLayerConfig
) -> None:
    """Fail fast unless the splits are disjoint, complete and month-pure."""
    names = list(indices)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = indices[first].intersection(indices[second])
            if len(overlap):
                raise SplitValidationError(
                    f"Splits '{first}' and '{second}' overlap on {len(overlap)} rows."
                )

    total = sum(len(idx) for idx in indices.values())
    if total != len(df):
        raise SplitValidationError(
            f"Splits cover {total:,} rows but the dataset has {len(df):,}; "
            "every row must belong to exactly one split."
        )

    months = df[config.split_column]
    for split_name, idx in indices.items():
        observed = set(months.loc[idx].unique())
        allowed = set(config.split_months[split_name])
        if not observed <= allowed:
            raise SplitValidationError(
                f"Split '{split_name}' contains months {sorted(observed - allowed)} "
                f"outside its frozen definition {sorted(allowed)}."
            )

"""Feature/target views per split, with frozen-schema enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from baf_data.config import DataLayerConfig
from baf_data.errors import SchemaValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitView:
    """Feature matrix and unchanged target labels for one split."""

    name: str
    X: pd.DataFrame
    y: pd.Series


def validate_feature_schema(X: pd.DataFrame, config: DataLayerConfig) -> None:
    """Fail fast unless X contains exactly the frozen feature columns.

    Guarantees the target, the split-only column and every frozen
    exclusion are absent, and that no unexpected column sneaked in.
    """
    observed = tuple(X.columns)
    expected = config.feature_columns
    if observed != expected:
        forbidden = [
            name
            for name in observed
            if name in (config.target_column, config.split_column)
            or name in config.excluded_features
        ]
        if forbidden:
            raise SchemaValidationError(
                f"Forbidden column(s) present in the feature matrix: {forbidden}."
            )
        raise SchemaValidationError(
            f"Feature matrix columns {list(observed)} do not match the frozen "
            f"feature schema {list(expected)}."
        )


def create_feature_target_views(
    df: pd.DataFrame,
    indices: dict[str, pd.Index],
    config: DataLayerConfig,
) -> dict[str, SplitView]:
    """Build X/y views for every split from the normalised dataframe.

    X contains only the frozen feature columns; y is the target column
    with values unchanged. No rows are dropped and no transformation
    (imputation, encoding, scaling) is fitted or applied here.
    """
    feature_columns = list(config.feature_columns)
    views: dict[str, SplitView] = {}
    for split_name, idx in indices.items():
        X = df.loc[idx, feature_columns].copy()
        y = df.loc[idx, config.target_column].copy()
        validate_feature_schema(X, config)
        if len(X) != len(idx) or len(y) != len(idx):
            raise SchemaValidationError(
                f"Split '{split_name}' views lost rows: expected {len(idx)}, "
                f"got X={len(X)}, y={len(y)}."
            )
        views[split_name] = SplitView(name=split_name, X=X, y=y)
        logger.info(
            "View '%s': X shape %s, positives %d.",
            split_name, X.shape, int(y.sum()),
        )
    return views

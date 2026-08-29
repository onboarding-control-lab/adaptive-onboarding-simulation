"""Read-only loading and fail-fast schema validation of the raw CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from baf_data.config import ColumnKind, DataLayerConfig
from baf_data.errors import SchemaValidationError

logger = logging.getLogger(__name__)

_KIND_CHECKS: dict[ColumnKind, object] = {
    "integer": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "string": lambda dtype: pd.api.types.is_string_dtype(dtype)
    or pd.api.types.is_object_dtype(dtype),
}


def load_raw_data(raw_path: Path, config: DataLayerConfig) -> pd.DataFrame:
    """Load Base.csv read-only and validate it against the frozen schema.

    The file is only ever opened for reading; no value is altered here.
    Returns the validated dataframe with a default RangeIndex.
    """
    logger.info("Loading raw CSV from %s ...", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows x %d columns.", df.shape[0], df.shape[1])
    validate_raw_schema(df, config)
    return df


def validate_raw_schema(
    df: pd.DataFrame,
    config: DataLayerConfig,
    *,
    require_expected_rows: bool = True,
) -> None:
    """Fail fast unless the dataframe matches the frozen raw schema exactly.

    Checks column names and order, broad dtype kinds, optional full-file row
    count, and the binary domain of the target column.
    """
    observed = tuple(df.columns)
    expected = config.raw_column_names
    if observed != expected:
        missing = set(expected) - set(observed)
        unexpected = set(observed) - set(expected)
        raise SchemaValidationError(
            "Column mismatch against the frozen schema. "
            f"Missing: {sorted(missing) or 'none'}; "
            f"unexpected: {sorted(unexpected) or 'none'}; "
            "order must also match the raw file."
        )

    if require_expected_rows and len(df) != config.expected_rows:
        raise SchemaValidationError(
            f"Expected {config.expected_rows:,} rows, found {len(df):,}."
        )

    for spec in config.raw_columns:
        check = _KIND_CHECKS[spec.kind]
        if not check(df[spec.name].dtype):  # type: ignore[operator]
            raise SchemaValidationError(
                f"Column '{spec.name}' has dtype {df[spec.name].dtype}, "
                f"which is not of kind '{spec.kind}'."
            )

    target_values = set(df[config.target_column].unique())
    if not target_values <= {0, 1}:
        raise SchemaValidationError(
            f"Target '{config.target_column}' contains non-binary values: "
            f"{sorted(target_values - {0, 1})}."
        )
    logger.info("Raw schema validated: %d columns, %d rows.", len(expected), len(df))

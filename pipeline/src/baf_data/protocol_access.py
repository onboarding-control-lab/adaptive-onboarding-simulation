"""Fail-closed experimental data-access contract.

Physical I/O of the monolithic ``Base.csv`` cannot isolate Month 7 at the
byte level. This module makes month filtering the first semantic operation
after the file is opened, and refuses illegal phase/month combinations.

Phases:

- ``development``: months must be a non-empty subset of {0,1,2,3,4,5,6}.
  Month 7 cannot be requested.
- ``final``: months must be exactly {7}. Development months cannot be
  requested and there is no silent Month-6 fallback.

Historical :func:`baf_data.pipeline.load_prepared_splits` remains the
data-layer inventory path used by already-frozen D1 training. Experimental
runners must use :func:`load_dataset_for_protocol`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

import pandas as pd

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.errors import ProtocolAccessError, SchemaValidationError
from baf_data.integrity import verify_raw_source
from baf_data.loading import validate_raw_schema
from baf_data.sentinels import normalise_sentinels
from baf_data.views import SplitView, create_feature_target_views

logger = logging.getLogger(__name__)

ProtocolPhase = Literal["development", "final"]

DEVELOPMENT_MONTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
FINAL_MONTHS: tuple[int, ...] = (7,)
_CHUNK_SIZE = 100_000

_SPLIT_FOR_MONTH: dict[int, str] = {
    0: "train",
    1: "train",
    2: "train",
    3: "train",
    4: "train",
    5: "train",
    6: "dev",
    7: "test",
}


@dataclass(frozen=True)
class ProtocolDataset:
    """Month-filtered, sentinel-normalised experimental frame."""

    phase: ProtocolPhase
    allowed_months: tuple[int, ...]
    frame: pd.DataFrame
    views: dict[str, SplitView]
    indices: dict[str, pd.Index]
    conversion_counts: dict[str, int]
    raw_sha256: str
    raw_path: Path
    month_filter_was_first_semantic_operation: bool
    month7_rows_retained: bool


def validate_phase_months(
    phase: str,
    allowed_months: Iterable[int],
) -> tuple[ProtocolPhase, tuple[int, ...]]:
    """Fail closed unless phase and months are an explicit legal pair."""
    if phase not in {"development", "final"}:
        raise ProtocolAccessError(
            f"Unknown experimental phase {phase!r}; use 'development' or 'final'."
        )
    requested = tuple(int(m) for m in allowed_months)
    if not requested:
        raise ProtocolAccessError("allowed_months must be a non-empty sequence.")
    if len(set(requested)) != len(requested):
        raise ProtocolAccessError("allowed_months must not contain duplicates.")
    illegal_values = sorted(set(requested) - set(range(8)))
    if illegal_values:
        raise ProtocolAccessError(
            f"Month value(s) {illegal_values} are outside the frozen 0–7 range."
        )

    if phase == "development":
        if 7 in requested:
            raise ProtocolAccessError(
                "Development paths cannot request Month 7."
            )
        extra = sorted(set(requested) - set(DEVELOPMENT_MONTHS))
        if extra:
            raise ProtocolAccessError(
                f"Development paths cannot request months {extra}."
            )
        return "development", requested

    if set(requested) != {7}:
        raise ProtocolAccessError(
            "Final paths must request allowed_months=[7] exactly; "
            "silent Month-6 fallback is forbidden."
        )
    return "final", requested


def load_dataset_for_protocol(
    raw_path: Path,
    *,
    phase: str,
    allowed_months: Iterable[int],
    config: DataLayerConfig = FROZEN_CONFIG,
    verify_hash: bool = True,
) -> ProtocolDataset:
    """Load only the requested experimental months.

    The monolithic CSV may be opened (physical I/O). Month filtering is the
    first semantic operation: no sentinel conversion, unique(), aggregation,
    logging of dropped-month statistics, fitting, or view construction occurs
    before non-allowed rows are discarded.
    """
    resolved_phase, months = validate_phase_months(phase, allowed_months)
    raw_path = Path(raw_path)
    raw_sha256 = (
        verify_raw_source(raw_path, config.expected_sha256) if verify_hash else ""
    )

    wanted = set(months)
    pieces: list[pd.DataFrame] = []
    header_columns: tuple[str, ...] | None = None
    n_read = 0

    for chunk in pd.read_csv(raw_path, chunksize=_CHUNK_SIZE):
        if header_columns is None:
            header_columns = tuple(chunk.columns)
            if header_columns != config.raw_column_names:
                missing = set(config.raw_column_names) - set(header_columns)
                unexpected = set(header_columns) - set(config.raw_column_names)
                raise SchemaValidationError(
                    "Column mismatch against the frozen schema. "
                    f"Missing: {sorted(missing) or 'none'}; "
                    f"unexpected: {sorted(unexpected) or 'none'}; "
                    "order must also match the raw file."
                )
        n_chunk = len(chunk)
        chunk = chunk.copy()
        chunk.index = pd.RangeIndex(n_read, n_read + n_chunk)
        n_read += n_chunk
        keep = chunk[config.split_column].astype("int64").isin(wanted)
        kept = chunk.loc[keep]
        if not kept.empty:
            pieces.append(kept)

    if pieces:
        frame = pd.concat(pieces, axis=0)
    else:
        frame = pd.DataFrame(columns=list(config.raw_column_names))

    if header_columns is None:
        raise ProtocolAccessError(f"Raw CSV was empty: {raw_path}")

    validate_raw_schema(frame, config, require_expected_rows=False)
    observed_months = (
        set(int(m) for m in frame[config.split_column].unique()) if len(frame) else set()
    )
    if observed_months - wanted:
        raise ProtocolAccessError(
            "Retained months exceeded the requested experimental set."
        )
    if resolved_phase == "development" and 7 in observed_months:
        raise ProtocolAccessError("Development load retained Month 7 rows.")
    if resolved_phase == "final" and observed_months - {7}:
        raise ProtocolAccessError("Final load retained non-Month-7 rows.")

    normalised, conversion_counts = normalise_sentinels(frame, config.sentinel_rules)
    indices = _indices_for_allowed_months(normalised, months, config)
    views = create_feature_target_views(normalised, indices, config)

    logger.info(
        "Protocol load phase=%s retained_months=%s n_retained=%d",
        resolved_phase,
        list(months),
        len(normalised),
    )
    return ProtocolDataset(
        phase=resolved_phase,
        allowed_months=months,
        frame=normalised,
        views=views,
        indices=indices,
        conversion_counts=conversion_counts,
        raw_sha256=raw_sha256,
        raw_path=raw_path,
        month_filter_was_first_semantic_operation=True,
        month7_rows_retained=(resolved_phase == "final" and 7 in months),
    )


def _indices_for_allowed_months(
    frame: pd.DataFrame,
    allowed_months: tuple[int, ...],
    config: DataLayerConfig,
) -> dict[str, pd.Index]:
    """Build split indices only for splits that intersect the request."""
    wanted_splits: set[str] = set()
    for month in allowed_months:
        wanted_splits.add(_SPLIT_FOR_MONTH[month])
    months_series = frame[config.split_column]
    indices: dict[str, pd.Index] = {}
    for split_name in ("train", "dev", "test"):
        if split_name not in wanted_splits:
            continue
        split_months = config.split_months[split_name]
        indices[split_name] = frame.index[months_series.isin(split_months)]
    covered = sum(len(idx) for idx in indices.values())
    if covered != len(frame):
        raise ProtocolAccessError(
            f"Protocol split indices cover {covered} rows but {len(frame)} were retained."
        )
    return indices


def assert_no_module_level_month7_frame(module_globals: Mapping[str, object]) -> None:
    """Refuse imported module state that already holds a Month-7 dataframe."""
    for name, value in module_globals.items():
        if not isinstance(value, pd.DataFrame):
            continue
        if "month" not in value.columns:
            continue
        months = set(int(m) for m in value["month"].dropna().unique())
        if 7 in months:
            raise ProtocolAccessError(
                f"Module global {name!r} contains Month 7 rows; "
                "import-time Month-7 loads are forbidden."
            )


__all__ = [
    "DEVELOPMENT_MONTHS",
    "FINAL_MONTHS",
    "ProtocolDataset",
    "ProtocolPhase",
    "assert_no_module_level_month7_frame",
    "load_dataset_for_protocol",
    "validate_phase_months",
]

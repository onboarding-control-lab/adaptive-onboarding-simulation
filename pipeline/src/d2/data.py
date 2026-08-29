"""Month-filtered BAF loading for D2-S.

Months 0–5 may be loaded for reference fitting.
Month 6 may be loaded only after the score contract is frozen, for
calibration/evaluation.
Month 7 is sealed in development: requesting it raises, and sealed-month
rows are never retained, summarised, or scored.
Final phase may request Month 7 explicitly and only Month 7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from baf_data.config import FROZEN_CONFIG, DataLayerConfig, SentinelRule
from baf_data.integrity import verify_raw_source
from baf_data.sentinels import normalise_sentinels
from d2.contract import (
    CALIBRATION_MONTHS,
    REFERENCE_FRAUD_BOOL,
    REFERENCE_MONTHS,
    REQUIRED_APPLICATION_FIELDS,
    SEALED_MONTHS,
)
from d2.errors import D2DataError

DEFAULT_RAW_PATH = Path(os.getenv("BAF_BASE_CSV", "Base.csv"))
_CHUNK_SIZE = 100_000

_LOAD_COLUMNS: tuple[str, ...] = (
    "fraud_bool",
    "month",
    *REQUIRED_APPLICATION_FIELDS,
)


@dataclass(frozen=True)
class LoadedD2Frame:
    """A month-filtered, sentinel-normalised frame plus provenance."""

    frame: pd.DataFrame
    months: tuple[int, ...]
    raw_sha256: str
    raw_path: Path
    n_rows_read: int
    n_rows_retained: int
    n_sealed_rows_skipped: int
    fraud_bool_filter: int | None
    month7_opened: bool


def present_sentinel_rules(
    columns: Iterable[str],
    config: DataLayerConfig = FROZEN_CONFIG,
) -> tuple[SentinelRule, ...]:
    """Official sentinel rules restricted to columns actually present.

    Semantics are unchanged for every overlapping field.  Rules for columns
    that D2-S does not load are skipped rather than inventing new sentinels.
    """
    present = set(columns)
    return tuple(rule for rule in config.sentinel_rules if rule.column in present)


def apply_official_sentinels(
    frame: pd.DataFrame,
    config: DataLayerConfig = FROZEN_CONFIG,
) -> pd.DataFrame:
    rules = present_sentinel_rules(frame.columns, config)
    if not rules:
        return frame.copy()
    normalised, _counts = normalise_sentinels(frame, rules)
    return normalised


def assert_months_allowed(
    months: Iterable[int],
    *,
    allow_calibration: bool,
    phase: str = "development",
) -> tuple[int, ...]:
    """Fail closed if a sealed month is requested or Month 6 is used too early."""
    requested = tuple(int(m) for m in months)
    if not requested:
        raise D2DataError("At least one month must be requested.")
    if phase not in {"development", "final"}:
        raise D2DataError(f"Unknown D2 load phase {phase!r}.")
    if phase == "final":
        extra = sorted(set(requested) - {7})
        if extra:
            raise D2DataError(
                "Final D2 loads may request Month 7 only; "
                f"refusing months {extra}."
            )
        if 7 not in requested:
            raise D2DataError("Final D2 loads must request Month 7.")
        return requested
    sealed = sorted(set(requested).intersection(SEALED_MONTHS))
    if sealed:
        raise D2DataError(
            f"Month(s) {sealed} are sealed and cannot be opened, loaded, "
            "summarised, or scored by D2-S."
        )
    allowed = set(REFERENCE_MONTHS)
    if allow_calibration:
        allowed.update(CALIBRATION_MONTHS)
    illegal = sorted(set(requested) - allowed)
    if illegal:
        raise D2DataError(
            f"Month(s) {illegal} are not permitted for this D2-S load "
            f"(allow_calibration={allow_calibration})."
        )
    return requested


def load_d2_frame(
    raw_path: Path,
    months: Iterable[int],
    *,
    allow_calibration: bool,
    fraud_bool: int | None,
    config: DataLayerConfig = FROZEN_CONFIG,
    verify_hash: bool = True,
    phase: str = "development",
) -> LoadedD2Frame:
    """Load selected months by chunk and immediately drop every other row.

    Sealed-month rows are counted only as a skip total in development.
    They are never appended to the returned frame and are never summarised.
    Final phase may retain Month 7 only when explicitly requested.
    """
    requested = assert_months_allowed(
        months, allow_calibration=allow_calibration, phase=phase
    )
    raw_path = Path(raw_path)
    raw_sha256 = (
        verify_raw_source(raw_path, config.expected_sha256)
        if verify_hash
        else ""
    )
    if tuple(config.raw_column_names[:2]) != ("fraud_bool", "income"):
        raise D2DataError("Unexpected frozen schema; refusing to load.")

    wanted = set(requested)
    sealed = set() if phase == "final" else set(SEALED_MONTHS)
    pieces: list[pd.DataFrame] = []
    n_read = 0
    n_sealed_skipped = 0
    usecols = [c for c in _LOAD_COLUMNS if c in config.raw_column_names]

    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=_CHUNK_SIZE):
        n_chunk = len(chunk)
        source_row_id = pd.RangeIndex(n_read, n_read + n_chunk)
        n_read += n_chunk
        month_values = chunk["month"].astype("int64")
        sealed_mask = month_values.isin(sealed)
        n_sealed_skipped += int(sealed_mask.sum())
        keep = month_values.isin(wanted)
        if fraud_bool is not None:
            keep = keep & (chunk["fraud_bool"].astype("int64") == int(fraud_bool))
        kept = chunk.loc[keep].copy()
        if not kept.empty:
            kept.insert(0, "source_row_id", source_row_id[keep.to_numpy()].astype("int64"))
            pieces.append(kept)

    if pieces:
        frame = pd.concat(pieces, ignore_index=True)
    else:
        frame = pd.DataFrame(columns=list(usecols))

    if phase != "final" and (
        frame["month"].isin(list(SEALED_MONTHS)).any() if len(frame) else False
    ):
        raise D2DataError("Sealed-month rows were retained; aborting.")
    if phase == "final" and len(frame) and set(int(m) for m in frame["month"].unique()) - {7}:
        raise D2DataError("Final D2 load retained non-Month-7 rows.")

    normalised = apply_official_sentinels(frame, config)
    observed_months = tuple(sorted(int(m) for m in normalised["month"].unique())) if len(
        normalised
    ) else tuple()
    if set(observed_months) - set(requested):
        raise D2DataError(
            f"Loaded months {observed_months} exceed the requested set {requested}."
        )
    return LoadedD2Frame(
        frame=normalised,
        months=observed_months,
        raw_sha256=raw_sha256,
        raw_path=raw_path,
        n_rows_read=n_read,
        n_rows_retained=len(normalised),
        n_sealed_rows_skipped=n_sealed_skipped,
        fraud_bool_filter=fraud_bool,
        month7_opened=(phase == "final" and 7 in requested),
    )


def load_reference_legitimate(
    raw_path: Path,
    *,
    config: DataLayerConfig = FROZEN_CONFIG,
    verify_hash: bool = True,
) -> LoadedD2Frame:
    """Months 0–5 and fraud_bool == 0 only."""
    return load_d2_frame(
        raw_path,
        REFERENCE_MONTHS,
        allow_calibration=False,
        fraud_bool=REFERENCE_FRAUD_BOOL,
        config=config,
        verify_hash=verify_hash,
        phase="development",
    )


def load_month6_applications(
    raw_path: Path,
    *,
    fraud_bool: int | None,
    config: DataLayerConfig = FROZEN_CONFIG,
    verify_hash: bool = True,
) -> LoadedD2Frame:
    """Month 6 only.  Never used for reference fitting."""
    return load_d2_frame(
        raw_path,
        CALIBRATION_MONTHS,
        allow_calibration=True,
        fraud_bool=fraud_bool,
        config=config,
        verify_hash=verify_hash,
        phase="development",
    )


def load_month7_applications(
    raw_path: Path,
    *,
    fraud_bool: int | None,
    config: DataLayerConfig = FROZEN_CONFIG,
    verify_hash: bool = True,
) -> LoadedD2Frame:
    """Month 7 only. Explicit final-phase evaluation load."""
    return load_d2_frame(
        raw_path,
        (7,),
        allow_calibration=False,
        fraud_bool=fraud_bool,
        config=config,
        verify_hash=verify_hash,
        phase="final",
    )

"""Orchestration of the deterministic BAF data layer.

Two entry points:

- :func:`load_prepared_splits` — the importable interface for the later
  model pipeline. Verifies the raw hash, loads, normalises sentinels and
  returns in-memory X/y views per split. Writes nothing.
- :func:`run_pipeline` — the CLI-facing run. Does everything above, then
  writes the split/feature manifests and a structured run log, and
  re-verifies the raw hash after completion.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.integrity import (
    ensure_output_path_allowed,
    raw_file_is_writable,
    sha256_of_file,
    verify_raw_source,
)
from baf_data.loading import load_raw_data
from baf_data.manifests import build_feature_manifest, build_split_manifest, write_manifests
from baf_data.sentinels import normalise_sentinels
from baf_data.splitting import build_temporal_indices
from baf_data.views import SplitView, create_feature_target_views

logger = logging.getLogger(__name__)

RUN_LOG_NAME = "run_log.jsonl"


@dataclass(frozen=True)
class PreparedData:
    """In-memory result of the data layer, ready for the model pipeline."""

    views: dict[str, SplitView]
    indices: dict[str, pd.Index]
    conversion_counts: dict[str, int]
    raw_sha256: str
    normalised_frame: pd.DataFrame


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a full manifest-writing pipeline run."""

    prepared: PreparedData
    split_manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    written_paths: dict[str, Path]


def load_prepared_splits(
    raw_path: Path, config: DataLayerConfig = FROZEN_CONFIG
) -> PreparedData:
    """Load, validate and split the raw data entirely in memory.

    Refuses to run when the raw file's SHA-256 differs from the frozen
    expectation. The raw file is opened read-only; nothing is written.
    """
    raw_sha256 = verify_raw_source(raw_path, config.expected_sha256)
    if raw_file_is_writable(raw_path):
        logger.warning(
            "Raw file %s is writable by this process; POSIX read-only "
            "protection is not in effect. Hash verification remains the guard.",
            raw_path,
        )
    df = load_raw_data(raw_path, config)
    normalised, conversion_counts = normalise_sentinels(df, config.sentinel_rules)
    indices = build_temporal_indices(normalised, config)
    views = create_feature_target_views(normalised, indices, config)
    return PreparedData(
        views=views,
        indices=indices,
        conversion_counts=conversion_counts,
        raw_sha256=raw_sha256,
        normalised_frame=normalised,
    )


def run_pipeline(
    raw_path: Path,
    output_dir: Path,
    config: DataLayerConfig = FROZEN_CONFIG,
) -> PipelineResult:
    """Full deterministic run: verify, prepare, write manifests, re-verify.

    Raises if the output directory is inside (or above) the raw data
    directory, if the source hash mismatches before the run, or if the
    source hash changed during the run.
    """
    raw_path = raw_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    ensure_output_path_allowed(output_dir, raw_path)

    started_utc = datetime.now(timezone.utc).isoformat()
    prepared = load_prepared_splits(raw_path, config)

    split_manifest = build_split_manifest(
        prepared.normalised_frame, prepared.indices, raw_path, prepared.raw_sha256, config
    )
    feature_manifest = build_feature_manifest(prepared.conversion_counts, config)
    written_paths = write_manifests(split_manifest, feature_manifest, output_dir)

    final_sha256 = sha256_of_file(raw_path)
    if final_sha256 != config.expected_sha256:
        raise RuntimeError(
            f"Raw source hash changed during the run: {final_sha256}. "
            "The raw file must be treated as compromised."
        )
    logger.info("Raw source hash re-verified after completion.")

    _append_run_log(
        output_dir / RUN_LOG_NAME,
        {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(raw_path),
            "raw_sha256_before": prepared.raw_sha256,
            "raw_sha256_after": final_sha256,
            "raw_file_writable": raw_file_is_writable(raw_path),
            "rows_total": split_manifest["total_rows"],
            "splits": {
                name: {
                    "row_count": entry["row_count"],
                    "fraud_count": entry["fraud_count"],
                }
                for name, entry in split_manifest["splits"].items()
            },
            "sentinel_conversions": prepared.conversion_counts,
            "feature_count": feature_manifest["feature_count"],
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "outcome": "success",
        },
    )
    return PipelineResult(
        prepared=prepared,
        split_manifest=split_manifest,
        feature_manifest=feature_manifest,
        written_paths=written_paths,
    )


def _append_run_log(log_path: Path, record: dict[str, Any]) -> None:
    """Append one structured JSON line describing a completed run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info("Run log appended to %s", log_path)

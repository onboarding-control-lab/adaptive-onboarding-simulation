"""Deterministic split and feature manifests.

Manifest content is fully determined by the frozen configuration and the
raw file, so repeated runs produce byte-identical files. Timestamps
belong in the run log, never in a manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from baf_data.config import DataLayerConfig

logger = logging.getLogger(__name__)

SPLIT_MANIFEST_NAME = "split_manifest.json"
FEATURE_MANIFEST_NAME = "feature_manifest.json"


def _index_sha256(index: pd.Index) -> str:
    """Digest of the row positions in a split, proving determinism."""
    return hashlib.sha256(index.to_numpy(dtype="int64").tobytes()).hexdigest()


def build_split_manifest(
    df: pd.DataFrame,
    indices: dict[str, pd.Index],
    raw_path: Path,
    raw_sha256: str,
    config: DataLayerConfig,
) -> dict[str, Any]:
    """Assemble the machine-readable split manifest."""
    target = df[config.target_column]
    splits: dict[str, Any] = {}
    for split_name, idx in indices.items():
        fraud_count = int(target.loc[idx].sum())
        splits[split_name] = {
            "months": list(config.split_months[split_name]),
            "row_count": int(len(idx)),
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_count / len(idx), 10),
            "row_index_sha256": _index_sha256(idx),
        }
    return {
        "source": {
            "filename": raw_path.name,
            "sha256": raw_sha256,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
        },
        "split_column": config.split_column,
        "splits": splits,
        "total_rows": int(sum(len(idx) for idx in indices.values())),
        "total_fraud": int(target.sum()),
    }


def build_feature_manifest(
    conversion_counts: dict[str, int],
    config: DataLayerConfig,
) -> dict[str, Any]:
    """Assemble the machine-readable feature/schema manifest."""
    return {
        "target_column": config.target_column,
        "split_only_column": config.split_column,
        "excluded_features": dict(config.excluded_features),
        "feature_columns": [
            {"name": spec.name, "kind": spec.kind}
            for spec in config.raw_columns
            if spec.name in set(config.feature_columns)
        ],
        "feature_count": len(config.feature_columns),
        "sentinel_rules": [
            {
                "column": rule.column,
                "strategy": rule.strategy,
                "value": rule.value,
                "values_converted_to_nan": conversion_counts[rule.column],
            }
            for rule in config.sentinel_rules
        ],
        "row_deletion": "never",
        "note": (
            "Sentinel normalisation is applied in memory only; no cleaned "
            "CSV is persisted. No imputer, encoder, scaler or model is "
            "fitted by the data layer."
        ),
    }


def write_manifests(
    split_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write both manifests as stable, sorted JSON. Returns written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, payload in (
        (SPLIT_MANIFEST_NAME, split_manifest),
        (FEATURE_MANIFEST_NAME, feature_manifest),
    ):
        path = output_dir / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info("Wrote %s", path)
        paths[filename] = path
    return paths

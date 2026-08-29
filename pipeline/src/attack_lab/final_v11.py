"""Optional D2-S v1.1 artefact verification. Never blocks primary readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attack_lab.experiment_config import sha256_file
from d2.iforest_v11 import (
    FIXED_IFOREST_PARAMS,
    FROZEN_D2S_V10_FINGERPRINT,
    IFOREST_FEATURE_IDS,
    SCORE_CONTRACT_ID_V11,
    D2SV11IForestAggregator,
)

EXPECTED_V11_PARAMS = {
    "n_estimators": 500,
    "max_samples": 256,
    "max_features": 1.0,
    "bootstrap": False,
    "contamination": "auto",
    "random_state": 20260816,
}


def verify_v11_artefact(path: Path | None, expected_sha256: str | None = None) -> dict[str, Any]:
    """Return a verification record. Missing/unverified artefacts disable v1.1."""
    record: dict[str, Any] = {
        "status": "V11_SECONDARY_DISABLED",
        "enabled": False,
        "path": str(path) if path else None,
        "sha256": None,
        "reason": None,
        "blocks_primary": False,
    }
    if path is None or not Path(path).is_file():
        record["reason"] = "artefact_missing"
        return record
    digest = sha256_file(Path(path))
    record["sha256"] = digest
    if expected_sha256 and digest != expected_sha256:
        record["reason"] = "sha256_mismatch"
        return record
    try:
        model = D2SV11IForestAggregator.load(Path(path))
    except Exception as exc:  # noqa: BLE001
        record["reason"] = f"load_failed:{type(exc).__name__}"
        return record
    params = {key: model.params.get(key) for key in EXPECTED_V11_PARAMS}
    if params != EXPECTED_V11_PARAMS:
        record["reason"] = "params_mismatch"
        record["observed_params"] = dict(model.params)
        return record
    if model.v10_fingerprint != FROZEN_D2S_V10_FINGERPRINT:
        record["reason"] = "parent_v10_fingerprint_mismatch"
        return record
    if model.month7_opened:
        record["reason"] = "month7_opened"
        return record
    if list(IFOREST_FEATURE_IDS) != [
        "payment_channel",
        "C13",
        "C09",
        "C03",
        "C10",
        "C11",
        "C15",
    ]:
        record["reason"] = "feature_contract_drift"
        return record
    if getattr(model, "n_train", None) is None:
        record["reason"] = "n_train_missing"
        return record
    record.update(
        {
            "status": "V11_SECONDARY_READY",
            "enabled": True,
            "reason": None,
            "score_contract_id": SCORE_CONTRACT_ID_V11,
            "v10_fingerprint": model.v10_fingerprint,
            "n_train": int(model.n_train),
            "fitted_utc": model.fitted_utc,
            "params": dict(model.params),
            "fixed_iforest_params": dict(FIXED_IFOREST_PARAMS),
        }
    )
    return record


__all__ = ["EXPECTED_V11_PARAMS", "verify_v11_artefact"]

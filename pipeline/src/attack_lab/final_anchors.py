"""Paired-anchor manifest contract for the final runner.

This module validates manifests only. It does not generate Month-7 production
anchors and must not open Month 7.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from attack_lab.experiment_config import canonical_json_hash

FINAL_N = 100
SELECTION_DOMAIN = "final_month7_paired_anchor_selection"
SELECTION_SEED = 1
PROPOSED_SELECTION_SEED = SELECTION_SEED  # historical alias


PRODUCTION_SCHEMA = "final-month7-paired-anchors-v1"
REHEARSAL_SCHEMA = "final-month7-rehearsal-anchors-v1"


class AnchorManifestError(RuntimeError):
    """Fail-closed paired-anchor contract violation."""


def fingerprint_anchor_ids(anchor_ids: Sequence[str]) -> str:
    payload = json.dumps([str(x) for x in anchor_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_anchor_manifest(path: Path) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        raise AnchorManifestError(f"Anchor manifest missing: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnchorManifestError("Anchor manifest must be a JSON object.")
    return payload


def validate_anchor_manifest(
    payload: Mapping[str, Any],
    *,
    require_production: bool,
    expected_n: int | None = None,
) -> list[str]:
    errors: list[str] = []
    ids = [str(x) for x in (payload.get("anchor_ids") or [])]
    if not ids:
        errors.append("anchor_ids missing")
    if len(ids) != len(set(ids)):
        errors.append("anchor_ids contain duplicates")
    sample_size = payload.get("sample_size")
    if sample_size is not None and int(sample_size) != len(ids):
        errors.append("sample_size disagrees with anchor_ids")
    if expected_n is not None and len(ids) != int(expected_n):
        errors.append(f"expected {expected_n} anchors, got {len(ids)}")
    claimed = str(payload.get("fingerprint") or "")
    computed = fingerprint_anchor_ids(ids) if ids else ""
    if claimed and claimed != computed:
        errors.append("anchor fingerprint mismatch")
    if require_production:
        if str(payload.get("schema_version")) != PRODUCTION_SCHEMA:
            errors.append(f"production schema must be {PRODUCTION_SCHEMA}")
        if payload.get("month7_source") is not True:
            errors.append("production manifest must set month7_source=true")
        try:
            eval_month = int(payload.get("evaluation_month"))
        except (TypeError, ValueError):
            eval_month = -1
        if eval_month != 7:
            errors.append("production evaluation_month must be 7")
        if str(payload.get("phase") or "") != "final":
            errors.append("production phase must be final")
    else:
        if payload.get("month7_source") is True:
            errors.append("rehearsal/test manifest must not claim Month-7 source")
        if payload.get("evaluation_month") in {7, "7"}:
            errors.append("rehearsal/test manifest must not use evaluation_month=7")
    return errors


def require_valid_anchor_manifest(
    payload: Mapping[str, Any],
    *,
    require_production: bool,
    expected_n: int | None = None,
) -> list[str]:
    errors = validate_anchor_manifest(
        payload,
        require_production=require_production,
        expected_n=expected_n,
    )
    if errors:
        raise AnchorManifestError("Invalid anchor manifest: " + "; ".join(errors))
    return [str(x) for x in payload["anchor_ids"]]


def select_paired_anchor_ids(
    eligible_ids: Sequence[str | int],
    *,
    n: int = FINAL_N,
    seed: int = SELECTION_SEED,
    domain: str = SELECTION_DOMAIN,
) -> list[str]:
    """Deterministic, outcome-independent downsample of an eligible ID list.

    This function never opens Month 7. Callers must already possess a complete
    eligible-ID universe. Issuing a production manifest from Month-7 rows is
    sealed until PI approves ``selection_rule``.
    """
    unique = sorted({str(x) for x in eligible_ids}, key=lambda x: int(x))
    if len(unique) != len(list(eligible_ids)):
        raise AnchorManifestError("Eligible IDs contain duplicates.")
    if len(unique) < int(n):
        raise AnchorManifestError(
            f"Eligible universe has {len(unique)} IDs; need at least {n}."
        )
    digest = hashlib.sha256(f"{int(seed)}:{domain}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    ordered = unique.copy()
    rng.shuffle(ordered)
    return [str(x) for x in ordered[: int(n)]]


def production_manifest_payload(
    anchor_ids: Sequence[str],
    *,
    selection_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a production manifest object from already-selected IDs.

    Does not read Month 7. Does not write a file.
    """
    ids = [str(x) for x in anchor_ids]
    if len(ids) != FINAL_N:
        raise AnchorManifestError(f"Production manifest must contain {FINAL_N} IDs.")
    payload = {
        "schema_version": PRODUCTION_SCHEMA,
        "phase": "final",
        "evaluation_month": 7,
        "month7_source": True,
        "anchor_ids": ids,
        "sample_size": len(ids),
        "fingerprint": fingerprint_anchor_ids(ids),
        "selection": dict(selection_meta or {}),
        "manifest_hash": canonical_json_hash(
            {
                "anchor_ids": ids,
                "month7_source": True,
                "evaluation_month": 7,
            }
        ),
    }
    return payload


def refuse_month7_manifest_reissue(path: Path | None = None) -> None:
    """Fail closed if an immutable production manifest already exists."""
    if path is not None and Path(path).is_file():
        raise AnchorManifestError(
            f"Production manifest already issued and is immutable: {path}"
        )


def refuse_month7_manifest_issuance() -> None:
    """Deprecated alias retained so older tests fail closed on re-issue."""
    raise AnchorManifestError(
        "Month-7 production-manifest re-issue is sealed. "
        "The approved issuer writes exactly one immutable file."
    )


def rehearsal_manifest(anchor_ids: Sequence[str]) -> dict[str, Any]:
    ids = [str(x) for x in anchor_ids]
    return {
        "schema_version": REHEARSAL_SCHEMA,
        "phase": "rehearsal",
        "evaluation_month": "not_accessed",
        "month7_source": False,
        "anchor_ids": ids,
        "sample_size": len(ids),
        "fingerprint": fingerprint_anchor_ids(ids),
        "note": "Synthetic/development rehearsal only. Not Month-7 applications.",
        "manifest_hash": canonical_json_hash(
            {
                "anchor_ids": ids,
                "month7_source": False,
            }
        ),
    }


__all__ = [
    "AnchorManifestError",
    "FINAL_N",
    "PRODUCTION_SCHEMA",
    "PROPOSED_SELECTION_SEED",
    "REHEARSAL_SCHEMA",
    "SELECTION_DOMAIN",
    "SELECTION_SEED",
    "fingerprint_anchor_ids",
    "load_anchor_manifest",
    "production_manifest_payload",
    "refuse_month7_manifest_issuance",
    "refuse_month7_manifest_reissue",
    "rehearsal_manifest",
    "require_valid_anchor_manifest",
    "select_paired_anchor_ids",
    "validate_anchor_manifest",
]

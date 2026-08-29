"""One-shot Month-7 paired-anchor issuance.

Authorised only for frozen eligibility and manifest writing.
Does not run attackers, does not persist D1 scores/margins, and does not
inspect post-attack outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from attack_lab.defender import FrozenXGBoostDefender
from attack_lab.experiment_config import canonical_json_hash, sha256_file
from attack_lab.final_anchors import (
    FINAL_N,
    SELECTION_DOMAIN,
    SELECTION_SEED,
    AnchorManifestError,
    fingerprint_anchor_ids,
    production_manifest_payload,
    refuse_month7_manifest_reissue,
    require_valid_anchor_manifest,
    select_paired_anchor_ids,
)
from attack_lab.final_protocol import FinalProtocolConfig, resolve_repo_path
from attack_lab.reference_pool import (
    ReferencePoolConfig,
    ReferencePoolError,
    ReferencePoolProvider,
)
from baf_data.config import FROZEN_CONFIG
from baf_data.protocol_access import load_dataset_for_protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = (
    REPO_ROOT
    / "runs"
    / "anchors"
    / "final_month7_paired_anchors_n100.json"
)


def _sha256_text(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def issue_production_anchor_manifest(
    *,
    protocol: FinalProtocolConfig,
    dest: Path | None = None,
    overwrite: bool = False,
    raw_path_override: Path | None = None,
) -> dict[str, Any]:
    """Enumerate eligible Month-7 anchors and write one immutable manifest."""
    dest = Path(dest or DEFAULT_MANIFEST_PATH)
    if dest.is_file() and not overwrite:
        refuse_month7_manifest_reissue(dest)
    rule = protocol.payload["paired_anchors"]["selection_rule"]
    if str(rule.get("status") or "") != "APPROVED":
        raise AnchorManifestError("selection_rule is not PI-approved.")
    if int(rule.get("selection_seed") or 0) != SELECTION_SEED:
        raise AnchorManifestError("selection_seed mismatch.")
    if str(rule.get("shuffle_domain") or "") != SELECTION_DOMAIN:
        raise AnchorManifestError("shuffle_domain mismatch.")

    if raw_path_override is not None:
        raw_path = Path(raw_path_override)
    else:
        env_csv = os.getenv("BAF_BASE_CSV")
        if env_csv:
            raw_path = Path(env_csv)
        else:
            raw_path = resolve_repo_path(protocol.payload["raw_dataset"]["path"])

    if not raw_path or not raw_path.is_file():
        raise AnchorManifestError(
            f"Raw BAF dataset not found at {raw_path}. Provide via --data /path/to/Base.csv "
            "or set BAF_BASE_CSV environment variable."
        )
    month7 = load_dataset_for_protocol(
        raw_path,
        phase="final",
        allowed_months=[7],
    )
    train = load_dataset_for_protocol(
        raw_path,
        phase="development",
        allowed_months=[0, 1, 2, 3, 4, 5],
    )
    if "test" not in month7.views:
        raise AnchorManifestError("Final load did not produce the test view.")
    features = month7.views["test"].X
    labels = month7.views["test"].y.astype(int)
    if len(features) != len(labels):
        raise AnchorManifestError("Month-7 feature/label alignment failed.")

    fraud_ids = [int(idx) for idx in labels.index[labels == 1]]
    feature_cols = list(FROZEN_CONFIG.feature_columns)
    missing_cols = [name for name in feature_cols if name not in features.columns]
    if missing_cols:
        raise AnchorManifestError(
            "Frozen preprocessing dropped required features: " + ", ".join(missing_cols)
        )

    defender = FrozenXGBoostDefender.from_artefact_dir(
        Path(protocol.payload["d1"]["artifact_dir"])
    )
    fraud_frame = features.loc[fraud_ids, feature_cols]
    scores = defender._pipeline.predict_proba(fraud_frame)[:, 1]  # noqa: SLF001
    blocked_mask = scores >= float(defender.threshold)
    blocked_ids = [int(idx) for idx in fraud_frame.index[blocked_mask]]
    # Scores are eligibility-only and are discarded; they are not persisted.
    del scores

    provider = ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(),
        training_frame=train.frame,
    )
    pool_seed = int(protocol.payload["seeds"]["reference_pool_seed"])
    eligible: list[str] = []
    n_pool_fail = 0
    for row_id in blocked_ids:
        try:
            pool = provider.get_pool(str(row_id), seed=pool_seed)
            if int(pool.K) != int(protocol.k):
                raise ReferencePoolError(f"K={pool.K}")
        except ReferencePoolError:
            n_pool_fail += 1
            continue
        eligible.append(str(row_id))

    selected = select_paired_anchor_ids(
        eligible,
        n=FINAL_N,
        seed=SELECTION_SEED,
        domain=SELECTION_DOMAIN,
    )
    selection_meta = {
        "status": "APPROVED",
        "selection_seed": SELECTION_SEED,
        "shuffle_domain": SELECTION_DOMAIN,
        "eligible_universe_size": len(eligible),
        "n_month7_rows": int(len(features)),
        "n_fraud": len(fraud_ids),
        "n_original_d1_block": len(blocked_ids),
        "n_reference_pool_failures": n_pool_fail,
        "outcome_independent": True,
    }
    payload = production_manifest_payload(selected, selection_meta=selection_meta)
    payload["raw_dataset_sha256"] = protocol.payload["raw_dataset"]["sha256"]
    payload["observed_raw_sha256"] = month7.raw_sha256
    payload["d1_artefact_id"] = protocol.payload["d1"]["artefact_id"]
    payload["d1_threshold"] = protocol.payload["d1"]["threshold"]
    payload["governance_fingerprint"] = protocol.payload["governance"]["fingerprint"]
    payload["reference_pool_seed"] = pool_seed
    payload["protocol_hash_at_eligibility"] = protocol.config_hash
    payload["issued_utc"] = datetime.now(timezone.utc).isoformat()
    payload["purpose"] = "eligibility_and_paired_anchor_issuance_only"
    payload["attack_outcomes_inspected"] = False
    payload["d1_scores_persisted"] = False
    require_valid_anchor_manifest(payload, require_production=True, expected_n=FINAL_N)

    dest.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    dest.write_text(text, encoding="utf-8")
    dest.chmod(0o444)

    log = {
        "issued_utc": payload["issued_utc"],
        "manifest_path": str(dest),
        "manifest_sha256": sha256_file(dest),
        "manifest_hash": payload["manifest_hash"],
        "anchor_fingerprint": payload["fingerprint"],
        "eligible_universe_size": len(eligible),
        "selected_n": len(selected),
        "selection_seed": SELECTION_SEED,
        "shuffle_domain": SELECTION_DOMAIN,
        "n_month7_rows": int(len(features)),
        "n_fraud": len(fraud_ids),
        "n_original_d1_block": len(blocked_ids),
        "n_reference_pool_failures": n_pool_fail,
        "raw_dataset_sha256": protocol.payload["raw_dataset"]["sha256"],
        "observed_raw_sha256": month7.raw_sha256,
        "protocol_hash_at_eligibility": protocol.config_hash,
        "d1_artefact_id": protocol.payload["d1"]["artefact_id"],
        "governance_fingerprint": protocol.payload["governance"]["fingerprint"],
        "month7_accessed": True,
        "purpose": "eligibility_and_paired_anchor_issuance_only",
        "attack_outcomes_inspected": False,
        "d1_scores_persisted": False,
        "live_api_calls": 0,
        "issuance_payload_sha256": _sha256_text(payload),
    }
    log_path = dest.with_name("ISSUANCE_LOG.json")
    if log_path.exists():
        raise AnchorManifestError(f"Issuance log already exists: {log_path}")
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest_path": str(dest), "log_path": str(log_path), **log}


__all__ = ["DEFAULT_MANIFEST_PATH", "issue_production_anchor_manifest"]

"""Frozen final Month-7 protocol: load, hash, and fail-closed validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from attack_lab.benchmark_pins import (
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_GOVERNANCE_FINGERPRINT,
    PINNED_REQUIRE_REFERENCE_PROVENANCE,
)
from attack_lab.experiment_config import canonical_json_hash, sha256_file
from attack_lab.query_semantics import RetryPolicy
from d2.contract import SCORE_CONTRACT_ID
from d2.iforest_v11 import FROZEN_D2S_V10_FINGERPRINT, SCORE_CONTRACT_ID_V11

FINAL_PROTOCOL_SCHEMA_VERSION = "final-month7-protocol-v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "config" / "final_month7_protocol.json"
D2S_V10_ID = "d2s-v1.0.0-pairwise8-20260816"
D2S_V11_ID = "d2s-v1.1.0-isolation-forest-20260816"


def resolve_repo_path(path: Path | str | None) -> Path | None:
    """Resolve a path against the repository root if it is relative."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


class FinalProtocolError(RuntimeError):
    """Raised when the frozen final protocol is missing or inconsistent."""


@dataclass(frozen=True)
class FinalProtocolConfig:
    """Validated immutable final Month-7 protocol."""

    payload: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "FinalProtocolConfig":
        protocol_path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
        if not protocol_path.is_file():
            raise FinalProtocolError(f"Final protocol config missing: {protocol_path}")
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FinalProtocolError("Final protocol must be a JSON object.")
        config = cls(payload=payload, path=protocol_path)
        config.validate()
        expected = str(payload.get("config_hash") or "")
        if expected and expected != config.config_hash:
            raise FinalProtocolError(
                f"Final protocol hash mismatch: expected={expected}, "
                f"computed={config.config_hash}."
            )
        return config

    @property
    def config_hash(self) -> str:
        body = {k: v for k, v in self.payload.items() if k != "config_hash"}
        return canonical_json_hash(body)

    @property
    def phase(self) -> str:
        return str(self.payload["phase"])

    @property
    def month(self) -> int:
        return int(self.payload["evaluation_month"])

    @property
    def q_max(self) -> int:
        return int(self.payload["budget"]["Q"])

    @property
    def m_max(self) -> int:
        return int(self.payload["budget"]["m"])

    @property
    def k(self) -> int:
        return int(self.payload["reference_pool"]["K"])

    @property
    def retry_policy(self) -> RetryPolicy:
        api = self.payload["api"]
        return RetryPolicy(
            max_transport_retries_per_call=int(api["max_transport_retries_per_call"]),
            timeout_seconds=float(api["timeout_seconds"]),
        )

    def validate(self) -> None:
        p = self.payload
        errors: list[str] = []
        if p.get("schema_version") != FINAL_PROTOCOL_SCHEMA_VERSION:
            errors.append("unsupported schema_version")
        if p.get("phase") != "final":
            errors.append("phase must be 'final'")
        if int(p.get("evaluation_month", -1)) != 7:
            errors.append("evaluation_month must be 7")
        if list(p.get("data_months", {}).get("training_reference", [])) != [0, 1, 2, 3, 4, 5]:
            errors.append("training/reference months must be 0-5")
        if list(p.get("data_months", {}).get("development_calibration", [])) != [6]:
            errors.append("development/calibration month must be 6")
        if list(p.get("data_months", {}).get("final_evaluation", [])) != [7]:
            errors.append("final evaluation month must be 7")
        budget = p.get("budget") or {}
        if int(budget.get("K", -1)) != 10 or int(self.k) != 10:
            errors.append("K must be 10")
        if int(budget.get("m", -1)) != 2:
            errors.append("m must be 2")
        if int(budget.get("Q", -1)) != 5:
            errors.append("Q must be 5")
        if budget.get("invalid_charges_q") is not False:
            errors.append("final Q contract A requires invalid_charges_q=false")
        if set(p.get("feedback_labels") or []) != {"PASS", "BLOCK", "INVALID"}:
            errors.append("feedback labels must be PASS/BLOCK/INVALID")
        if p.get("require_reference_provenance") is not True:
            errors.append("require_reference_provenance must be true")
        if PINNED_REQUIRE_REFERENCE_PROVENANCE is not True:
            errors.append("pinned provenance requirement drifted")
        attackers = p.get("attackers") or {}
        a0 = attackers.get("A0") or {}
        a1 = attackers.get("A1") or {}
        a2 = attackers.get("A2") or {}
        a3 = attackers.get("A3") or {}
        if str(a0.get("kind")) != "random_nonadaptive_baseline":
            errors.append("A0 kind mismatch")
        if str(a1.get("prompt_version")) != PINNED_A1_PROMPT_VERSION:
            errors.append(f"A1 pin must be {PINNED_A1_PROMPT_VERSION}")
        if str(a1.get("model")) != MODEL_PRO:
            errors.append("A1 model must be deepseek-v4-pro")
        if a1.get("thinking") != "OFF" or a1.get("thinking_disabled") is not True:
            errors.append("A1 Thinking must be OFF")
        if str(a2.get("gower_policy")) != PINNED_A2_GOWER_POLICY:
            errors.append(f"A2 pin must be {PINNED_A2_GOWER_POLICY}")
        if str(a3.get("prompt_version")) != PINNED_A3_PROMPT_VERSION:
            errors.append(f"A3 pin must be {PINNED_A3_PROMPT_VERSION}")
        if str(a3.get("model")) != MODEL_PRO:
            errors.append("A3 model must be deepseek-v4-pro")
        if a3.get("thinking") != "OFF" or a3.get("thinking_disabled") is not True:
            errors.append("A3 Thinking must be OFF")
        d1 = p.get("d1") or {}
        if str(d1.get("artefact_id")) != PINNED_D1_ARTEFACT_ID:
            errors.append("D1 artefact id mismatch")
        d2 = p.get("d2") or {}
        primary = d2.get("primary") or {}
        secondary = d2.get("secondary_prespecified") or {}
        if str(primary.get("id")) != D2S_V10_ID or str(primary.get("id")) != SCORE_CONTRACT_ID:
            errors.append("D2-S v1.0 primary id mismatch")
        if str(primary.get("fingerprint")) != FROZEN_D2S_V10_FINGERPRINT:
            errors.append("D2-S v1.0 fingerprint mismatch")
        if str(primary.get("role")) != "PRIMARY":
            errors.append("D2-S v1.0 role must be PRIMARY")
        if str(secondary.get("id")) != D2S_V11_ID or str(secondary.get("id")) != SCORE_CONTRACT_ID_V11:
            errors.append("D2-S v1.1 secondary id mismatch")
        if str(secondary.get("role")) != "OPTIONAL_SECONDARY_EXPLORATORY":
            errors.append("D2-S v1.1 role must be OPTIONAL_SECONDARY_EXPLORATORY")
        if secondary.get("blocks_primary_readiness") is True:
            errors.append("D2-S v1.1 must not block primary readiness")
        if "D2-L" not in (d2.get("excluded") or []):
            errors.append("D2-L must be excluded")
        if d2.get("selection_after_month7_forbidden") is not True:
            errors.append("post-Month-7 defence selection must be forbidden")
        if str((p.get("governance") or {}).get("fingerprint")) != PINNED_GOVERNANCE_FINGERPRINT:
            errors.append("governance fingerprint mismatch")
        budgets = p.get("review_budgets") or {}
        if abs(float(budgets.get("primary", -1)) - 0.10) > 1e-12:
            errors.append("primary review budget must be 10%")
        if [float(x) for x in budgets.get("sensitivity", [])] != [0.05, 0.15]:
            errors.append("sensitivity review budgets must be 5% and 15%")
        api = p.get("api") or {}
        if str(api.get("model")) != MODEL_PRO:
            errors.append("API model must be deepseek-v4-pro")
        if api.get("thinking") != "OFF":
            errors.append("API Thinking must be OFF")
        if api.get("transport_retry_does_not_charge_Q") is not True:
            errors.append("transport retries must not charge Q")
        if api.get("invalid_does_not_charge_Q") is not True:
            errors.append("INVALID must not charge Q under contract A")
        if api.get("parse_local_generation_does_not_charge_Q") is not True:
            errors.append("local parse/generation must not charge Q")
        if not p.get("seeds"):
            errors.append("seeds must be explicit")
        if not p.get("paired_anchors"):
            errors.append("paired-anchor manifest must be explicit")
        anchors = p.get("paired_anchors") or {}
        if int(anchors.get("n") or 0) != 100:
            errors.append("final paired-anchor N must be 100")
        if anchors.get("production_path") not in {None, ""}:
            # Path may be set only after PI-approved issuance. Presence is
            # allowed; execute-final still validates the manifest.
            pass
        rule = anchors.get("selection_rule") or {}
        if int(rule.get("selection_seed") or 0) != 1:
            errors.append("anchor selection_seed must reuse experiment_seed=1")
        if str(rule.get("shuffle_domain") or "") != "final_month7_paired_anchor_selection":
            errors.append("anchor shuffle_domain mismatch")
        if str(rule.get("status") or "") != "APPROVED":
            errors.append("anchor selection_rule must be PI-approved")
        if errors:
            raise FinalProtocolError("Final protocol invalid: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload, "config_hash": self.config_hash}


def verify_frozen_artefacts(protocol: FinalProtocolConfig) -> list[str]:
    """Return fail-closed artefact errors (empty means artefacts match)."""
    errors: list[str] = []
    p = protocol.payload
    d1 = p["d1"]
    d1_dir = resolve_repo_path(d1["artifact_dir"])
    for name, expected in (d1.get("artifact_sha256") or {}).items():
        path = d1_dir / name if d1_dir else Path(name)
        if not path.is_file():
            errors.append(f"missing D1 artefact: {name}")
            continue
        if sha256_file(path) != expected:
            errors.append(f"D1 artefact hash mismatch: {name}")
    d2_primary = resolve_repo_path(p["d2"]["primary"]["artefact_path"])
    if d2_primary and d2_primary.is_file():
        if sha256_file(d2_primary) != p["d2"]["primary"]["artefact_sha256"]:
            errors.append("D2-S v1.0 artefact hash mismatch")
    else:
        errors.append(f"missing D2-S v1.0 artefact: {d2_primary}")
    gov = resolve_repo_path(p["governance"]["path"])
    if gov and gov.is_file():
        if sha256_file(gov) != p["governance"]["file_sha256"]:
            errors.append("governance file hash mismatch")
    else:
        errors.append(f"missing governance file: {gov}")
    d2_sec = p["d2"].get("secondary_prespecified") or {}
    sec_model = resolve_repo_path(d2_sec.get("model_artefact"))
    if sec_model:
        if not sec_model.is_file():
            errors.append(f"missing D2-S v1.1 artefact: {sec_model}")
        elif d2_sec.get("model_sha256") and sha256_file(sec_model) != d2_sec["model_sha256"]:
            errors.append("D2-S v1.1 artefact hash mismatch")
    return errors


def protocol_role_statement() -> dict[str, Any]:
    return {
        "primary": {
            "id": D2S_V10_ID,
            "fingerprint": FROZEN_D2S_V10_FINGERPRINT,
            "role": "PRIMARY",
            "description": "D2-S v1.0 equal-mean consistency layer",
        },
        "secondary_prespecified": {
            "id": D2S_V11_ID,
            "role": "OPTIONAL_SECONDARY_EXPLORATORY",
            "description": "D2-S v1.1 Isolation Forest optional offline comparison",
            "blocks_primary_readiness": False,
        },
        "exploratory": ["D1-R", "D2-HS"],
        "excluded": ["D2-L", "D2-L V1", "D2-L V2"],
        "selection_after_month7_forbidden": True,
        "statement": (
            "NO MODEL OR DEFENCE SELECTION MAY OCCUR AFTER MONTH 7 IS OPENED."
        ),
    }


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "D2S_V10_ID",
    "D2S_V11_ID",
    "FINAL_PROTOCOL_SCHEMA_VERSION",
    "FinalProtocolConfig",
    "FinalProtocolError",
    "protocol_role_statement",
    "resolve_repo_path",
    "verify_frozen_artefacts",
]

"""Authoritative configuration contract for the formal A0--A3 comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from attack_lab.candidate_identity import CANDIDATE_IDENTITY_VERSION
from attack_lab.types import to_jsonable

FORMAL_CONFIG_SCHEMA_VERSION = "formal-a0-a3-config-v1"
FORMAL_MANIFEST_SCHEMA_VERSION = "formal-a0-a3-manifest-v1"
REQUIRED_ATTACKERS = ("a0", "a1", "a2", "a3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FormalExperimentConfig:
    """Validated immutable formal comparison configuration."""

    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "FormalExperimentConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Formal config must be a JSON object.")
        config = cls(payload=payload)
        config.validate()
        expected = str(payload.get("config_hash", ""))
        if expected and expected != config.config_hash:
            raise ValueError(
                f"Formal config hash mismatch: expected={expected}, "
                f"computed={config.config_hash}."
            )
        return config

    @property
    def config_hash(self) -> str:
        body = {k: v for k, v in self.payload.items() if k != "config_hash"}
        return canonical_json_hash(body)

    @property
    def attacker_ids(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.payload["attackers"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.payload["experiment_seeds"])

    @property
    def q_max(self) -> int:
        return int(self.payload["budget"]["Q"])

    @property
    def m_max(self) -> int:
        return int(self.payload["budget"]["m"])

    @property
    def k(self) -> int:
        return int(self.payload["reference_pool"]["K"])

    def validate(self) -> None:
        p = self.payload
        if p.get("schema_version") != FORMAL_CONFIG_SCHEMA_VERSION:
            raise ValueError("Unsupported formal config schema_version.")
        if p.get("status") != "FROZEN_READY_TO_RUN":
            raise ValueError("Formal config status must be FROZEN_READY_TO_RUN.")
        if p.get("data_split") != "dev_month6_reserved_formal_comparison":
            raise ValueError("Formal comparison must use the reserved month-6 split.")
        if p.get("month7_opened") is not False:
            raise ValueError("month7_opened must be false.")
        if self.attacker_ids != REQUIRED_ATTACKERS:
            raise ValueError(f"attackers must be exactly {REQUIRED_ATTACKERS}.")
        if self.q_max < 1 or self.m_max < 1 or self.k < 1:
            raise ValueError("Q, m and K must be positive.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("experiment_seeds must be a non-empty unique list.")
        anchors = p.get("anchors") or {}
        if int(anchors.get("sample_size", 0)) < 1:
            raise ValueError("anchors.sample_size must be positive.")
        if not anchors.get("path") or not anchors.get("sha256"):
            raise ValueError("anchors path and sha256 are required.")
        if p.get("candidate_identity_version") != CANDIDATE_IDENTITY_VERSION:
            raise ValueError("Candidate identity version is not the frozen canonical one.")
        for key in ("governance", "d1", "constraint_profile", "aggregation"):
            if key not in p:
                raise ValueError(f"Missing formal config section: {key}.")
        a3 = p["attackers"]["a3"]
        for key in ("version", "prompt_version", "model", "temperature", "config_hash"):
            if key not in a3:
                raise ValueError(f"A3 formal config missing {key}.")

    def to_dict(self) -> dict[str, Any]:
        return {**to_jsonable(self.payload), "config_hash": self.config_hash}

    def episode_manifest(
        self,
        *,
        attacker_id: str,
        anchor_id: str,
        seed: int,
        reference_pool_fingerprint: str,
        code_revision: Mapping[str, Any],
    ) -> dict[str, Any]:
        attacker = dict(self.payload["attackers"][attacker_id])
        return {
            "manifest_schema_version": FORMAL_MANIFEST_SCHEMA_VERSION,
            "formal_config_hash": self.config_hash,
            "experiment_id": self.payload["experiment_id"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attacker": attacker_id,
            "attacker_version": attacker["version"],
            "prompt_version": attacker.get("prompt_version"),
            "model_identifier": attacker.get("model"),
            "model_provider_version": attacker.get("model_provider_version"),
            "temperature": attacker.get("temperature"),
            "seed": int(seed),
            "anchor_id": str(anchor_id),
            "anchor_set_identifier": self.payload["anchors"]["identifier"],
            "anchor_set_sha256": self.payload["anchors"]["sha256"],
            "Q": self.q_max,
            "m": self.m_max,
            "K": self.k,
            "reference_pool_fingerprint": reference_pool_fingerprint,
            "governance_version": self.payload["governance"]["version"],
            "governance_fingerprint": self.payload["governance"]["fingerprint"],
            "constraint_profile": self.payload["constraint_profile"],
            "candidate_identity_version": self.payload["candidate_identity_version"],
            "dataset_split": self.payload["data_split"],
            "d1_version": self.payload["d1"]["version"],
            "d1_hashes": self.payload["d1"]["artifact_sha256"],
            "code_revision": to_jsonable(dict(code_revision)),
            "output_schema_version": self.payload["output_schema_version"],
        }


__all__ = [
    "FORMAL_CONFIG_SCHEMA_VERSION",
    "FORMAL_MANIFEST_SCHEMA_VERSION",
    "FormalExperimentConfig",
    "canonical_json_hash",
    "sha256_file",
]

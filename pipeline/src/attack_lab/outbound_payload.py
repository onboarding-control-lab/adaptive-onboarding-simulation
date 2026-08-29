"""Fail-closed sanitisation and audit for external LLM payloads."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from attack_lab.types import to_jsonable

OUTBOUND_POLICY_VERSION = "deepseek-month6-development-allowlist-v1"

_DENIED_KEYS = frozenset(
    {
        "fraud_bool",
        "risk_score",
        "y_score",
        "threshold",
        "d1_risk_score",
        "d1_threshold",
        "model_object",
        "model_parameters",
        "feature_importance",
        "shap",
        "gradient",
        "gradients",
        "internal_defence",
        "researcher_only",
        "researcher_diagnostics",
        "source_row_id",
        "source_row_ids",
        "raw_path",
        "local_path",
        "api_key",
        "environment_variables",
    }
)
_LOCAL_PATH_MARKERS = ("/Users/", "/Volumes/", "/private/", "file://")
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|authorization|bearer\s+[a-z0-9._-]+|secret[_ -]?key)"
)
_ALLOWED_LABELS = frozenset({"PASS", "BLOCK", "INVALID"})


class OutboundPayloadError(RuntimeError):
    """Raised before any API call when a payload violates the allowlist."""


def temporary_episode_id(seed_material: str | int) -> str:
    digest = hashlib.sha256(
        f"{OUTBOUND_POLICY_VERSION}:{seed_material}".encode("utf-8")
    ).hexdigest()
    return f"dev-anchor-{digest[:16]}"


def sanitise_reference_pool(
    public_pool: Mapping[str, Any],
    *,
    temporary_anchor_id: str,
    allowed_fields: Iterable[str],
) -> dict[str, Any]:
    """Strip source-linkable IDs/provenance and retain allowlisted fields only."""

    allowed = {str(name) for name in allowed_fields}
    profiles = []
    for index, profile in enumerate(public_pool.get("profiles") or (), start=1):
        if not isinstance(profile, Mapping):
            raise OutboundPayloadError("Reference profile must be a mapping.")
        fields = dict(profile.get("fields") or {})
        unexpected = sorted(set(fields) - allowed)
        if unexpected:
            raise OutboundPayloadError(
                f"Reference profile contains non-allowlisted fields: {unexpected}."
            )
        profiles.append(
            {
                "profile_id": f"ref-{index:02d}",
                "fields": to_jsonable(fields),
            }
        )
    return {
        "anchor_id": temporary_anchor_id,
        "K": int(public_pool.get("K", len(profiles))),
        "context_fields": [
            str(x) for x in (public_pool.get("context_fields") or ()) if str(x) in allowed
        ],
        "action_fields": [
            str(x) for x in (public_pool.get("action_fields") or ()) if str(x) in allowed
        ],
        "read_only_context_fields": [
            str(x)
            for x in (public_pool.get("read_only_context_fields") or ())
            if str(x) in allowed
        ],
        "profiles": profiles,
    }


def audit_outbound_payload(
    payload: Mapping[str, Any],
    *,
    allowed_top_level_keys: Sequence[str],
    allowed_feature_fields: Iterable[str],
) -> dict[str, Any]:
    """Validate the exact structured payload and return a local audit manifest."""

    allowed_top = {str(key) for key in allowed_top_level_keys}
    unexpected_top = sorted(set(payload) - allowed_top)
    if unexpected_top:
        raise OutboundPayloadError(
            f"Outbound payload has non-allowlisted top-level keys: {unexpected_top}."
        )
    allowed_fields = {str(name) for name in allowed_feature_fields}
    observed_feature_fields: set[str] = set()
    observed_labels: set[str] = set()

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_s = str(key)
                if key_s.lower() in _DENIED_KEYS:
                    raise OutboundPayloadError(
                        f"Denied outbound key {key_s!r} at {path}."
                    )
                is_feature_map = key_s in {"visible_fields", "fields"}
                is_action_map = key_s in {
                    "locked_episode_static_choices",
                    "changes",
                } and "output_schema" not in path.lower()
                if (is_feature_map or is_action_map) and isinstance(value, Mapping):
                    names = {str(name) for name in value}
                    unexpected = sorted(names - allowed_fields)
                    if unexpected:
                        raise OutboundPayloadError(
                            f"Non-allowlisted feature fields at {path}.{key_s}: "
                            f"{unexpected}."
                        )
                    observed_feature_fields.update(names)
                if key_s in {"case_id", "anchor_id"} and isinstance(value, str):
                    if not value.startswith("dev-anchor-"):
                        raise OutboundPayloadError(
                            f"Non-temporary identifier at {path}.{key_s}."
                        )
                if key_s in {"public_label", "outcome", "label"} and isinstance(
                    value, str
                ):
                    if value not in _ALLOWED_LABELS:
                        raise OutboundPayloadError(
                            f"Non-public feedback label {value!r} at {path}.{key_s}."
                        )
                    observed_labels.add(value)
                walk(value, f"{path}.{key_s}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str):
            if any(marker in node for marker in _LOCAL_PATH_MARKERS):
                raise OutboundPayloadError(f"Local absolute path found at {path}.")
            if _CREDENTIAL_PATTERN.search(node):
                raise OutboundPayloadError(f"Credential-like text found at {path}.")

    walk(payload, "payload")
    encoded = json.dumps(
        to_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "outbound_policy_version": OUTBOUND_POLICY_VERSION,
        "preflight": "PASS",
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "top_level_keys": sorted(payload),
        "external_feature_fields": sorted(observed_feature_fields),
        "public_feedback_labels_present": sorted(observed_labels),
        "contains_local_absolute_path": False,
        "contains_credentials": False,
        "contains_researcher_only_diagnostics": False,
        "contains_month7": False,
    }


__all__ = [
    "OUTBOUND_POLICY_VERSION",
    "OutboundPayloadError",
    "audit_outbound_payload",
    "sanitise_reference_pool",
    "temporary_episode_id",
]

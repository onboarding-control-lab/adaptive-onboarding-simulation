"""Canonical attacker-public reference-pool view (shared A0–A3 evidence).

Derives a single safe representation from ``ReferencePool.attacker_view()``
by excluding trusted proxy-raw targets and research-only provenance.

Profiles remain observable exemplars only — never PASS / legitimate /
low-risk labels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable

# Trusted proxy raw targets: resolver-only; never in public decision evidence.
TRUSTED_PROXY_RAW_TARGETS: frozenset[str] = frozenset(PROXY_RAW_FEATURE_NAMES)

_HARD_PUBLIC_EXCLUSIONS: frozenset[str] = frozenset(
    {
        *TRUSTED_PROXY_RAW_TARGETS,
        "fraud_bool",
        "month",
        "y_score",
        "risk_score",
        "threshold",
        "score",
        "d1_risk_score",
        "d1_threshold",
        "feature_importance",
        "shap",
        "gradient",
        "gradients",
        "source_row_id",
        "source_row_ids",
        "true_rejection_reason",
    }
)


def public_safe_reference_field_names(pool: ReferencePool) -> tuple[str, ...]:
    """Exact public-safe field names derived from pool context metadata.

    = ``pool.context_fields`` minus trusted proxy raws and hard exclusions.
    Includes legitimate read-only context fields already present on the pool.
    """
    out: list[str] = []
    for name in pool.context_fields:
        key = str(name)
        if key in _HARD_PUBLIC_EXCLUSIONS:
            continue
        if key in TRUSTED_PROXY_RAW_TARGETS:
            continue
        out.append(key)
    return tuple(out)


def public_safe_gower_field_names(pool: ReferencePool) -> tuple[str, ...]:
    """A2 Gower field set: public-safe fields ∩ action_fields."""
    public = set(public_safe_reference_field_names(pool))
    return tuple(name for name in pool.action_fields if name in public)


def build_canonical_public_reference_view(pool: ReferencePool) -> dict[str, Any]:
    """Canonical PUBLIC reference view for attacker decision policies.

    Uses ``attacker_view()`` as the base, then strips proxy-raw / forbidden
    fields from every profile. Never includes source_row_ids.
    """
    base = pool.attacker_view()
    allowed = set(public_safe_reference_field_names(pool))
    profiles: list[dict[str, Any]] = []
    for profile in base.get("profiles") or ():
        if not isinstance(profile, Mapping):
            continue
        raw_fields = dict(profile.get("fields") or {})
        safe_fields = {
            str(k): to_jsonable(v) for k, v in raw_fields.items() if str(k) in allowed
        }
        profiles.append(
            {
                "profile_id": str(profile.get("profile_id")),
                "fields": safe_fields,
            }
        )
    return {
        "anchor_id": base.get("anchor_id"),
        "K": int(base.get("K") or len(profiles)),
        "generation_seed": base.get("generation_seed"),
        "pool_fingerprint": base.get("pool_fingerprint"),
        "context_fields": list(public_safe_reference_field_names(pool)),
        "action_fields": [
            name for name in (base.get("action_fields") or ()) if str(name) in allowed
        ],
        "read_only_context_fields": [
            name
            for name in (base.get("read_only_context_fields") or ())
            if str(name) in allowed
        ],
        "profiles": profiles,
        "note": (
            "Observable reference exemplars only. Not known PASS cases; "
            "no defender outcomes, scores, or trusted proxy raw targets."
        ),
    }


def assert_public_reference_view_safe(view: Mapping[str, Any]) -> None:
    """Fail closed if a public view leaks forbidden material."""
    if "source_row_ids" in view or "profiles_with_source" in view:
        raise ValueError("Public reference view must not expose source_row_ids.")
    for key in ("context_fields", "action_fields", "read_only_context_fields"):
        names = view.get(key) or ()
        if not isinstance(names, (list, tuple)):
            continue
        overlap = sorted(set(str(n) for n in names) & _HARD_PUBLIC_EXCLUSIONS)
        if overlap:
            raise ValueError(f"Public view {key} includes forbidden names: {overlap}.")
    for profile in view.get("profiles") or ():
        if not isinstance(profile, Mapping):
            raise ValueError("Public profile must be a mapping.")
        if "source_row_id" in profile:
            raise ValueError("Public profile must not expose source_row_id.")
        fields = profile.get("fields") or {}
        if not isinstance(fields, Mapping):
            raise ValueError("Public profile fields must be a mapping.")
        overlap = sorted(set(str(k) for k in fields) & _HARD_PUBLIC_EXCLUSIONS)
        if overlap:
            raise ValueError(f"Public profile fields include forbidden names: {overlap}.")


def choice_public_value_lookup(
    *,
    catalog_choices: Sequence[Mapping[str, Any]],
    public_view: Mapping[str, Any],
    choice_id: str,
) -> Any | None:
    """Recover the public reference field value for a choice_id from LLM-visible data.

    Uses only ``choice_id -> action_key -> reference_id`` plus public profiles.
    Returns None if the choice/action/profile is unavailable or the field is
    not public-safe.
    """
    target = str(choice_id)
    meta = None
    for item in catalog_choices:
        if isinstance(item, Mapping) and str(item.get("choice_id")) == target:
            meta = item
            break
    if meta is None:
        return None
    action_key = str(meta.get("action_key") or "")
    reference_id = str(meta.get("reference_id") or "")
    if not action_key or not reference_id:
        return None
    for profile in public_view.get("profiles") or ():
        if not isinstance(profile, Mapping):
            continue
        if str(profile.get("profile_id")) != reference_id:
            continue
        fields = profile.get("fields") or {}
        if isinstance(fields, Mapping) and action_key in fields:
            return fields[action_key]
        return None
    return None


__all__ = [
    "TRUSTED_PROXY_RAW_TARGETS",
    "assert_public_reference_view_safe",
    "build_canonical_public_reference_view",
    "choice_public_value_lookup",
    "public_safe_gower_field_names",
    "public_safe_reference_field_names",
]

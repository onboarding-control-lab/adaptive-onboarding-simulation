"""Shared reference-backed action selection, resolution and provenance audit.

Attackers propose :class:`ReferenceSelection` (a pool profile id) for an action.
Trusted code resolves that selection to the underlying raw feature value from
the current anchor-specific K-reference pool.  Governance may accept or reject
the resolved value; it is never used as a value *source*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from attack_lab.governance import CompiledFieldPolicy, is_sentinel
from attack_lab.reference_pool import ReferencePool, ReferenceProfile


class ReferenceActionError(RuntimeError):
    """Raised when a reference-backed action cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class ReferenceSelection:
    """Immutable attacker-facing selection of one K-pool profile fragment.

    Carries only ``reference_id``.  Attackers must not copy or invent the
    underlying raw feature value; trusted resolution supplies it.
    """

    reference_id: str

    def __post_init__(self) -> None:
        if not str(self.reference_id).strip():
            raise ReferenceActionError("reference_id must be a non-empty string.")


def is_reference_selection(value: Any) -> bool:
    """Return True for :class:`ReferenceSelection` instances."""
    return isinstance(value, ReferenceSelection)


def profile_by_id(pool: ReferencePool, reference_id: str) -> ReferenceProfile:
    """Return the pool profile for ``reference_id`` or fail closed."""
    wanted = str(reference_id)
    for profile in pool.profiles:
        if profile.profile_id == wanted:
            return profile
    raise ReferenceActionError(
        f"reference_id {wanted!r} is not in the current reference pool."
    )


def resolve_reference_selection(
    action_key: str,
    selection: ReferenceSelection,
    pool: ReferencePool,
    rule: CompiledFieldPolicy,
) -> Any:
    """Resolve a selection to the raw feature value from the current K-pool.

    For proxy actions the returned value is the underlying raw feature from the
    selected profile (never a nearest catalogue constant).
    """
    _ = action_key
    if not isinstance(selection, ReferenceSelection):
        raise ReferenceActionError("Expected ReferenceSelection.")
    if (
        rule.feature not in pool.action_fields
        and rule.feature not in pool.context_fields
    ):
        raise ReferenceActionError(
            f"Feature {rule.feature!r} is not present in the reference pool schema."
        )
    profile = profile_by_id(pool, selection.reference_id)
    if rule.feature not in profile.fields:
        raise ReferenceActionError(
            f"Reference {selection.reference_id!r} is missing field {rule.feature!r}."
        )
    raw = profile.fields[rule.feature]
    return coerce_pool_value(rule, raw)


def coerce_pool_value(rule: CompiledFieldPolicy, raw: Any) -> Any:
    """Canonicalise a pool fragment to the field's compiled data type."""
    if rule.data_type == "categorical":
        return str(raw)
    if raw is None or (isinstance(raw, float) and raw != raw):  # NaN
        raise ReferenceActionError("Reference fragment is missing/NaN.")
    if rule.data_type in {"binary", "integer"}:
        number = float(raw)
        if not float(number).is_integer():
            raise ReferenceActionError("Reference fragment is not an integer.")
        return int(number)
    if rule.data_type == "float":
        return float(raw)
    item = getattr(raw, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    return raw


def canonical_compare_value(value: Any) -> Any:
    """Type-canonical form for exact provenance equality (no nearest/similarity)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if float(value).is_integer():
            return int(value)
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def values_exactly_equal(left: Any, right: Any) -> bool:
    """Exact equality after unified numeric/string canonicalisation."""
    return canonical_compare_value(left) == canonical_compare_value(right)


def governance_domain_accepts(
    rule: CompiledFieldPolicy,
    value: Any,
    *,
    anchor_value: Any | None = None,
) -> bool:
    """Return True if governance domain constraints accept ``value``.

    This is a legality filter only.  It must never invent values.
    """
    if (
        rule.sentinel_policy == "retain_anchor_only"
        and is_sentinel(value, rule.sentinel_spec)
        and (
            anchor_value is None
            or not values_exactly_equal(value, anchor_value)
        )
    ):
        return False
    if rule.data_type == "binary" and value not in (0, 1):
        return False
    if rule.allowed_values and not any(
        values_exactly_equal(value, allowed) for allowed in rule.allowed_values
    ):
        return False
    if rule.lower_bound is not None and float(value) < float(rule.lower_bound):
        return False
    if rule.upper_bound is not None and float(value) > float(rule.upper_bound):
        return False
    return True


def reference_backed_selections_for_action(
    *,
    action_key: str,
    pool: ReferencePool,
    rule: CompiledFieldPolicy,
    anchor_value: Any | None = None,
    require_change_from_anchor: bool = False,
) -> tuple[ReferenceSelection, ...]:
    """Finite, deduplicated K-pool selections for one action dimension.

    Values are taken only from the current pool.  Governance domain filters
    legality; catalogue / observed_support / interpolation are never sources.
    """
    selections: list[ReferenceSelection] = []
    seen: list[Any] = []
    for profile in sorted(pool.profiles, key=lambda item: item.profile_id):
        selection = ReferenceSelection(reference_id=profile.profile_id)
        try:
            resolved = resolve_reference_selection(
                action_key, selection, pool, rule
            )
        except ReferenceActionError:
            continue
        if not governance_domain_accepts(
            rule, resolved, anchor_value=anchor_value
        ):
            continue
        if require_change_from_anchor and values_exactly_equal(
            resolved, anchor_value
        ):
            continue
        if any(values_exactly_equal(resolved, prev) for prev in seen):
            continue
        seen.append(resolved)
        selections.append(selection)
    return tuple(selections)


# Attacker-facing abstract proxy keys → underlying raw features.  Used only by
# researcher-side provenance audit/reporting; never exposed as writable raw
# names in attacker catalogues.
ABSTRACT_PROXY_ACTION_TO_RAW_FEATURE: dict[str, str] = {
    "name_email_alignment": "name_email_similarity",
    "home_phone_configuration": "phone_home_valid",
    "mobile_phone_configuration": "phone_mobile_valid",
}


def raw_feature_for_provenance_field(field_name: str) -> str:
    """Map an abstract proxy action key to the raw feature audited against K10."""
    name = str(field_name)
    return str(ABSTRACT_PROXY_ACTION_TO_RAW_FEATURE.get(name, name))


def raw_changed_fields_for_provenance(changed_fields: Sequence[str]) -> tuple[str, ...]:
    """Deduplicated raw feature names for researcher-side provenance audit."""
    mapped: list[str] = []
    seen: set[str] = set()
    for name in changed_fields:
        raw = raw_feature_for_provenance_field(name)
        if raw in seen:
            continue
        seen.add(raw)
        mapped.append(raw)
    return tuple(mapped)


def provenance_audit_counts(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Per-field PASS/FAIL counts from a provenance audit payload."""
    fields = audit.get("fields") or {}
    backed = 0
    failed = 0
    for detail in fields.values():
        if not isinstance(detail, Mapping):
            continue
        status = str(detail.get("status") or "")
        if status == "PASS":
            backed += 1
        elif status == "FAIL":
            failed += 1
    return {
        "reference_backed": backed,
        "non_reference_backed": failed,
        "audited_fields": backed + failed,
        "status": audit.get("status"),
    }


def audit_reference_provenance(
    *,
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    pool: ReferencePool,
    changed_fields: Sequence[str],
) -> dict[str, Any]:
    """Researcher-only exact K-pool provenance audit for changed raw fields.

    Abstract proxy action keys are mapped to the underlying raw feature before
    comparison.  Every changed raw value must exactly match at least one pool
    profile on that same field.  Failures are reported with status FAIL; no
    nearest match.  This mapping does not change candidate generation or
    trusted resolution.
    """
    fields: dict[str, Any] = {}
    all_ok = True
    for name in raw_changed_fields_for_provenance(changed_fields):
        if name not in candidate:
            all_ok = False
            fields[name] = {
                "value": None,
                "matching_reference_ids": [],
                "status": "FAIL",
                "reason": "missing_from_candidate",
            }
            continue
        value = candidate[name]
        if values_exactly_equal(value, anchor.get(name)):
            continue
        matching = [
            profile.profile_id
            for profile in pool.profiles
            if name in profile.fields
            and values_exactly_equal(profile.fields[name], value)
        ]
        status = "PASS" if matching else "FAIL"
        if status == "FAIL":
            all_ok = False
        fields[name] = {
            "value": canonical_compare_value(value),
            "matching_reference_ids": matching,
            "status": status,
        }
    return {
        "status": "PASS" if all_ok else "FAIL",
        "reference_pool_fingerprint": pool.pool_fingerprint,
        "fields": fields,
    }


def reference_ids_from_changes(changes: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect unique reference_ids from proposal changes."""
    ids: list[str] = []
    for value in changes.values():
        if isinstance(value, ReferenceSelection):
            ids.append(value.reference_id)
    return tuple(dict.fromkeys(ids))


__all__ = [
    "ABSTRACT_PROXY_ACTION_TO_RAW_FEATURE",
    "ReferenceActionError",
    "ReferenceSelection",
    "audit_reference_provenance",
    "canonical_compare_value",
    "coerce_pool_value",
    "governance_domain_accepts",
    "is_reference_selection",
    "profile_by_id",
    "provenance_audit_counts",
    "raw_changed_fields_for_provenance",
    "raw_feature_for_provenance_field",
    "reference_backed_selections_for_action",
    "reference_ids_from_changes",
    "resolve_reference_selection",
    "values_exactly_equal",
]

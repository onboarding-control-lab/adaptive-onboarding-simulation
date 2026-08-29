"""A1 V4.3 public-reference-view successor (does not modify V4.2).

Preserves all V4.2 one-shot / bounded-slot / freeze semantics. The only
substantive information change is that the LLM receives the canonical safe
K10 public reference view alongside the existing choice_id mapping.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_1_contract import ActionSlotCatalog
from attack_lab.archive.contracts.a1_v4_2_contract import (
    DIVERSIFICATION_PRINCIPLE_V4_2,
    PROMPT_VERSION_V4_2,
    assert_v4_2_prompt_hard_contract,
    build_v4_2_plan_conditioned_output_schema,
    build_v4_2_prompt_payload,
    build_v4_2_repair_output_schema,
    parse_a1_v4_2_plan,
    parse_a1_v4_2_slot_replacements,
)
from attack_lab.archive.contracts.a1_v4_contract import StaticPlanOption, V4ChoiceCatalog
from attack_lab.budget import AttackBudget
from attack_lab.public_reference_view import (
    TRUSTED_PROXY_RAW_TARGETS,
    assert_public_reference_view_safe,
    build_canonical_public_reference_view,
    public_safe_reference_field_names,
)
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_V4_3 = "a1_oneshot_v4_3_public_reference_view"

PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3 = (
    "The K reference profiles are observable exemplars of the reference "
    "population. You may examine their attacker-visible field combinations "
    "when deciding which reference-backed choices to use. They are not known "
    "PASS cases and reveal no defender outcome. Decide for yourself which "
    "reference patterns and choices are strategically relevant."
)


def build_v4_3_prompt_payload(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    budget: AttackBudget,
    q_max: int,
    visible_anchor: Mapping[str, Any],
    case_id: str,
    catalog: V4ChoiceCatalog,
    static_plans: Sequence[StaticPlanOption],
    action_slots: ActionSlotCatalog,
) -> dict[str, Any]:
    """V4.2 payload plus canonical safe public K10 reference view."""
    base = build_v4_2_prompt_payload(
        validator=validator,
        pool=pool,
        budget=budget,
        q_max=q_max,
        visible_anchor=visible_anchor,
        case_id=case_id,
        catalog=catalog,
        static_plans=static_plans,
        action_slots=action_slots,
    )
    # Rebuild as V4.3: replace version string and add public reference evidence.
    public_view = build_canonical_public_reference_view(pool)
    assert_public_reference_view_safe(public_view)
    payload = dict(base)
    payload["prompt_version"] = PROMPT_VERSION_V4_3
    payload["public_reference_profiles"] = {
        "K": public_view["K"],
        "pool_fingerprint": public_view.get("pool_fingerprint"),
        "public_safe_fields": list(public_safe_reference_field_names(pool)),
        "profiles": list(public_view["profiles"]),
        "note": public_view.get("note"),
    }
    payload["public_reference_reasoning_principle"] = (
        PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3
    )
    # Keep V4.2 diversification principle; add the neutral reference principle.
    payload["planning_principle"] = (
        f"{DIVERSIFICATION_PRINCIPLE_V4_2} {PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3}"
    )
    # Choice catalogue already exposes choice_id -> action_key -> reference_id.
    assert_v4_3_prompt_hard_contract(payload, pool=pool)
    return payload


def assert_v4_3_prompt_hard_contract(
    payload: Mapping[str, Any],
    *,
    pool: ReferencePool,
) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_V4_3:
        raise ValueError("V4.3 payload prompt_version mismatch.")
    if str(payload.get("prompt_version")) == PROMPT_VERSION_V4_2:
        raise ValueError("V4.3 builder must not emit the V4.2 version string.")
    public = payload.get("public_reference_profiles")
    if not isinstance(public, Mapping):
        raise ValueError("V4.3 must include public_reference_profiles.")
    profiles = public.get("profiles") or ()
    if int(public.get("K") or 0) != int(pool.K) or len(profiles) != int(pool.K):
        raise ValueError("V4.3 public_reference_profiles K mismatch.")
    expected_fields = set(public_safe_reference_field_names(pool))
    listed = set(str(x) for x in (public.get("public_safe_fields") or ()))
    if listed != expected_fields:
        raise ValueError("V4.3 public_safe_fields disagree with helper.")
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("V4.3 public profile must be a mapping.")
        if "source_row_id" in profile:
            raise ValueError("V4.3 public profile leaks source_row_id.")
        fields = profile.get("fields") or {}
        if not isinstance(fields, Mapping):
            raise ValueError("V4.3 public profile fields must be a mapping.")
        overlap = sorted(set(str(k) for k in fields) & set(TRUSTED_PROXY_RAW_TARGETS))
        if overlap:
            raise ValueError(f"V4.3 public profiles expose proxy raws: {overlap}.")
        unexpected = sorted(set(str(k) for k in fields) - expected_fields)
        if unexpected:
            raise ValueError(f"V4.3 public profile has non-public fields: {unexpected}.")
    if "choice_catalogue" not in payload:
        raise ValueError("V4.3 must preserve choice_catalogue mapping.")
    for item in payload.get("choice_catalogue") or []:
        if not isinstance(item, Mapping):
            continue
        for key in ("choice_id", "action_key", "reference_id"):
            if key not in item:
                raise ValueError(f"V4.3 choice_catalogue missing {key}.")
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4.3 choice_catalogue exposes proxy raw action_key.")
    if PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3 not in str(
        payload.get("public_reference_reasoning_principle") or ""
    ):
        raise ValueError("V4.3 missing public reference reasoning principle.")
    # Reuse V4.2 structural checks by temporarily rewriting version.
    probe = dict(payload)
    probe["prompt_version"] = PROMPT_VERSION_V4_2
    # Strip V4.3-only keys that would fail V4.2 named-term scan only if they
    # contain forbidden tokens; public profiles must already be safe.
    assert_v4_2_prompt_hard_contract(probe)
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(PROXY_RAW_FEATURE_NAMES):
        if f'"{term}"' in text:
            raise ValueError(f"V4.3 prompt embeds proxy raw field name {term!r}.")


__all__ = [
    "PROMPT_VERSION_V4_3",
    "PUBLIC_REFERENCE_REASONING_PRINCIPLE_V4_3",
    "assert_v4_3_prompt_hard_contract",
    "build_v4_2_plan_conditioned_output_schema",
    "build_v4_2_repair_output_schema",
    "build_v4_3_prompt_payload",
    "parse_a1_v4_2_plan",
    "parse_a1_v4_2_slot_replacements",
]

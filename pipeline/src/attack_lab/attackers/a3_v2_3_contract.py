"""A3 V2.3 public-reference-view successor (does not modify V2.2).

Preserves all V2.2 sequential / reflection / cardinality-repair semantics.
Adds only the canonical safe K10 public reference view and an explicit
choice_id -> action_key -> reference_id catalogue for strategic reasoning.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_1_contract import ActionSlotCatalog
from attack_lab.archive.contracts.a1_v4_contract import V4ChoiceCatalog
from attack_lab.archive.contracts.a3_v2_2_contract import (
    ATTACK_OBJECTIVE_V2_2,
    CARDINALITY_REPAIR_INSTRUCTION,
    MAX_HYPOTHESIS_CHARS_V2_2,
    PROMPT_VERSION_A3_V2_2,
    REFLECTION_MODE_DEFINITIONS_V2_2,
    REFLECTION_PURPOSE_V2_2,
    SELECTIONS_VS_HYPOTHESIS_NOTE,
    STATIC_RULE_DISCLOSURE_V2_2,
    assert_a3_v2_2_prompt_hard_contract,
    build_a3_v2_2_cardinality_repair_schema,
    build_a3_v2_2_episode_action_slots,
    build_a3_v2_2_output_schema,
    build_a3_v2_2_prompt_payload,
    build_a3_v2_2_repair_schema,
    filter_mechanically_valid_proposed_pairs,
    parse_a3_v2_2_repair_selections,
    parse_a3_v2_2_strategic_response,
)
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

PROMPT_VERSION_A3_V2_3 = "a3_episodic_reflective_v2_3_public_reference_view"
MAX_HYPOTHESIS_CHARS_V2_3 = MAX_HYPOTHESIS_CHARS_V2_2
ATTACK_OBJECTIVE_V2_3 = ATTACK_OBJECTIVE_V2_2
REFLECTION_PURPOSE_V2_3 = REFLECTION_PURPOSE_V2_2
STATIC_RULE_DISCLOSURE_V2_3 = STATIC_RULE_DISCLOSURE_V2_2
REFLECTION_MODE_DEFINITIONS_V2_3 = dict(REFLECTION_MODE_DEFINITIONS_V2_2)

PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3 = (
    "Use the K reference profiles as observable exemplars of the reference "
    "population. Before selecting a candidate, examine their attacker-visible "
    "field combinations and consider whether your proposed changes preserve or "
    "improve profile-level consistency with patterns present in the reference "
    "pool. The profiles are not known PASS cases and reveal no defender "
    "outcome. You must decide yourself which patterns, fields and "
    "reference-backed choices are strategically relevant."
)


def build_a3_v2_3_episode_action_slots(
    catalog: V4ChoiceCatalog,
    *,
    validator: ConstraintValidator,
) -> ActionSlotCatalog:
    return build_a3_v2_2_episode_action_slots(catalog, validator=validator)


def build_a3_v2_3_prompt_payload(
    *,
    case_id: str,
    visible_anchor: Mapping[str, Any],
    current_application: Mapping[str, Any],
    budget: AttackBudget,
    q_remaining: int,
    query_index: int,
    static_edit_cost: int,
    residual_m: int,
    locked_static_values: Mapping[str, Any],
    slots: ActionSlotCatalog,
    slot_entries: Sequence[Mapping[str, Any]],
    episodic_memory: Sequence[Mapping[str, Any]],
    pool: ReferencePool,
    catalog: V4ChoiceCatalog,
    episode_slot_map: Sequence[Mapping[str, Any]] | None = None,
    local_repair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = build_a3_v2_2_prompt_payload(
        case_id=case_id,
        visible_anchor=visible_anchor,
        current_application=current_application,
        budget=budget,
        q_remaining=q_remaining,
        query_index=query_index,
        static_edit_cost=static_edit_cost,
        residual_m=residual_m,
        locked_static_values=locked_static_values,
        slots=slots,
        slot_entries=slot_entries,
        episodic_memory=episodic_memory,
        episode_slot_map=episode_slot_map,
        local_repair=local_repair,
    )
    public_view = build_canonical_public_reference_view(pool)
    assert_public_reference_view_safe(public_view)
    choice_ids = list(catalog.static_choice_ids) + list(catalog.per_attempt_choice_ids)
    choice_catalogue = catalog.public_choices(choice_ids)
    payload = dict(base)
    payload["prompt_version"] = PROMPT_VERSION_A3_V2_3
    payload["public_reference_profiles"] = {
        "K": public_view["K"],
        "pool_fingerprint": public_view.get("pool_fingerprint"),
        "public_safe_fields": list(public_safe_reference_field_names(pool)),
        "profiles": list(public_view["profiles"]),
        "note": public_view.get("note"),
    }
    payload["choice_catalogue"] = choice_catalogue
    payload["choice_to_reference_mapping"] = [
        {
            "choice_id": item["choice_id"],
            "action_key": item["action_key"],
            "reference_id": item["reference_id"],
        }
        for item in choice_catalogue
        if isinstance(item, Mapping)
    ]
    payload["public_reference_reasoning_principle"] = (
        PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3
    )
    payload["task"] = (
        f"{str(base.get('task') or '')} {PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3}"
    )
    assert_a3_v2_3_prompt_hard_contract(payload, pool=pool)
    return payload


def assert_a3_v2_3_prompt_hard_contract(
    payload: Mapping[str, Any],
    *,
    pool: ReferencePool,
) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_A3_V2_3:
        raise ValueError("A3 V2.3 prompt_version mismatch.")
    if str(payload.get("prompt_version")) == PROMPT_VERSION_A3_V2_2:
        raise ValueError("A3 V2.3 builder must not emit the V2.2 version string.")
    public = payload.get("public_reference_profiles")
    if not isinstance(public, Mapping):
        raise ValueError("A3 V2.3 must include public_reference_profiles.")
    profiles = public.get("profiles") or ()
    if int(public.get("K") or 0) != int(pool.K) or len(profiles) != int(pool.K):
        raise ValueError("A3 V2.3 public_reference_profiles K mismatch.")
    expected_fields = set(public_safe_reference_field_names(pool))
    listed = set(str(x) for x in (public.get("public_safe_fields") or ()))
    if listed != expected_fields:
        raise ValueError("A3 V2.3 public_safe_fields disagree with helper.")
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("A3 V2.3 public profile must be a mapping.")
        fields = profile.get("fields") or {}
        if not isinstance(fields, Mapping):
            raise ValueError("A3 V2.3 public profile fields must be a mapping.")
        overlap = sorted(set(str(k) for k in fields) & set(TRUSTED_PROXY_RAW_TARGETS))
        if overlap:
            raise ValueError(f"A3 V2.3 public profiles expose proxy raws: {overlap}.")
    mapping = payload.get("choice_to_reference_mapping") or []
    if not mapping:
        raise ValueError("A3 V2.3 must expose choice_id -> action_key -> reference_id.")
    for item in mapping:
        if not isinstance(item, Mapping):
            raise ValueError("A3 V2.3 mapping entry must be a mapping.")
        for key in ("choice_id", "action_key", "reference_id"):
            if key not in item:
                raise ValueError(f"A3 V2.3 mapping missing {key}.")
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("A3 V2.3 mapping exposes proxy raw action_key.")
    if PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3 not in str(
        payload.get("public_reference_reasoning_principle") or ""
    ):
        raise ValueError("A3 V2.3 missing public reference reasoning principle.")
    # Structural inheritance: reuse V2.2 checks under temporary version rewrite.
    probe = dict(payload)
    probe["prompt_version"] = PROMPT_VERSION_A3_V2_2
    assert_a3_v2_2_prompt_hard_contract(probe)
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(PROXY_RAW_FEATURE_NAMES):
        if f'"{term}"' in text:
            raise ValueError(f"A3 V2.3 prompt embeds proxy raw field name {term!r}.")


def render_a3_v2_3_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    from attack_lab.archive.contracts.a1_v4_1_contract import GENERIC_UNAVAILABLE_NOTICE

    system = (
        "You are a fraud-evasion attacker under sparse label-only defender "
        "feedback and a fixed provenance-constrained K-pool action space. "
        f"{ATTACK_OBJECTIVE_V2_3} {REFLECTION_PURPOSE_V2_3} "
        f"{SELECTIONS_VS_HYPOTHESIS_NOTE} "
        f"{PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3} "
        "Emit ONLY the JSON object required by the output schema. "
        "Never emit raw feature values, action_key strings as outputs, "
        "reference_id, or a changes object. "
        "At query 1, reflection_update.mode must be INITIALIZE. "
        "After a submitted BLOCK or INVALID, first produce RETAIN, REVISE, or "
        "ABANDON based only on public outcomes, then choose the next selections "
        "in the same response. "
        "You do not know which selected action, field, value or interaction "
        "caused BLOCK. "
        f"{GENERIC_UNAVAILABLE_NOTICE}"
    )
    repair = payload.get("local_selection_repair")
    if repair is not None:
        residual = (payload.get("budget") or {}).get("residual_m")
        system = (
            "Local compliance repair only. "
            f"{CARDINALITY_REPAIR_INSTRUCTION} "
            f"Current residual_m / maximum selections this query: {residual}. "
            "Return ONLY {\"selections\": {...}}. "
            f"{GENERIC_UNAVAILABLE_NOTICE}"
        )
    user = json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Re-export V2.2 mechanical helpers used by the agent wiring.
parse_a3_v2_3_strategic_response = parse_a3_v2_2_strategic_response
parse_a3_v2_3_repair_selections = parse_a3_v2_2_repair_selections
build_a3_v2_3_output_schema = build_a3_v2_2_output_schema
build_a3_v2_3_repair_schema = build_a3_v2_2_repair_schema
build_a3_v2_3_cardinality_repair_schema = build_a3_v2_2_cardinality_repair_schema


__all__ = [
    "ATTACK_OBJECTIVE_V2_3",
    "CARDINALITY_REPAIR_INSTRUCTION",
    "MAX_HYPOTHESIS_CHARS_V2_3",
    "PROMPT_VERSION_A3_V2_3",
    "PUBLIC_REFERENCE_REASONING_PRINCIPLE_V2_3",
    "REFLECTION_MODE_DEFINITIONS_V2_3",
    "REFLECTION_PURPOSE_V2_3",
    "SELECTIONS_VS_HYPOTHESIS_NOTE",
    "STATIC_RULE_DISCLOSURE_V2_3",
    "assert_a3_v2_3_prompt_hard_contract",
    "build_a3_v2_3_cardinality_repair_schema",
    "build_a3_v2_3_episode_action_slots",
    "build_a3_v2_3_output_schema",
    "build_a3_v2_3_prompt_payload",
    "build_a3_v2_3_repair_schema",
    "filter_mechanically_valid_proposed_pairs",
    "parse_a3_v2_3_repair_selections",
    "parse_a3_v2_3_strategic_response",
    "render_a3_v2_3_messages",
]

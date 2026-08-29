"""A3 V2.1 construct-alignment helpers (successor to V2 hard contract).

Preserves historical V2 (`a3_episodic_reflective_v2_k10_hard_contract`).
Fixes: PASS-oriented objective, episode-stable action_slot IDs, static-rule
disclosure, 512-char hypothesis limit, reflection-mode definitions.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v4_1_contract import (
    GENERIC_UNAVAILABLE_NOTICE,
    ActionSlot,
    ActionSlotCatalog,
)
from attack_lab.archive.contracts.a1_v4_contract import V4ChoiceCatalog, build_v4_choice_catalog
from attack_lab.archive.contracts.a3_v2_contract import (
    REFLECTION_MODES_AFTER,
    REFLECTION_MODES_Q1,
    _FORBIDDEN_OUTPUT_KEYS,
    _NAMED_HIDDEN_TERMS,
    build_a3_v2_action_slots,
    build_selections_schema,
    compute_static_cost_and_residual,
    parse_a3_v2_repair_selections,
    prompt_contains_hidden_term,
    public_slot_entries,
    resolve_a3_v2_selections,
    selections_fingerprint,
)
from attack_lab.budget import AttackBudget
from attack_lab.types import to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_A3_V2_1 = "a3_episodic_reflective_v2_1_k10_hard_contract"
MAX_HYPOTHESIS_CHARS_V2_1 = 512

ATTACK_OBJECTIVE_V2_1 = (
    "Your objective is to obtain PASS within the remaining real-query budget. "
    "At each query choose the legal candidate that, under your own reasoning from "
    "attacker-visible context and prior label-only outcomes, you believe is most "
    "likely to obtain PASS."
)

REFLECTION_PURPOSE_V2_1 = (
    "Reflection is for improving the next evasion attempt, not for diagnosing the "
    "defender, estimating causal feature importance, or identifying the true reason "
    "for BLOCK."
)

STATIC_RULE_DISCLOSURE_V2_1 = (
    "episode_static actions may be selected on the first submitted candidate. "
    "Once selected, their resulting values are locked for the remainder of the "
    "episode. Each episode-static field that differs from ORIGINAL_ANCHOR "
    "permanently occupies one edit slot in every later candidate. "
    "static_edit_cost is the number of such occupied slots. "
    "residual_m = m_max - static_edit_cost. "
    "If residual_m becomes 0 after a BLOCK, no further editable per-attempt "
    "candidate can be submitted."
)

REFLECTION_MODE_DEFINITIONS_V2_1 = {
    "INITIALIZE": (
        "form the initial strategy hypothesis before the first submission."
    ),
    "RETAIN": (
        "keep the broad strategy hypothesis while varying legal details."
    ),
    "REVISE": (
        "modify part of the current strategy hypothesis in response to public "
        "outcomes."
    ),
    "ABANDON": (
        "drop the prior strategy hypothesis and adopt a materially different "
        "strategy."
    ),
}


def build_a3_v2_1_episode_action_slots(
    catalog: V4ChoiceCatalog,
    *,
    validator: ConstraintValidator,
) -> ActionSlotCatalog:
    """Canonical full episode action-slot map (static + per-attempt).

    Slot IDs are fixed for the episode; later queries only filter writability.
    """
    return build_a3_v2_action_slots(
        catalog,
        validator=validator,
        include_static=True,
        exclude_action_keys=(),
    )


def writable_slots_from_episode_map(
    episode_slots: ActionSlotCatalog,
    *,
    validator: ConstraintValidator,
    include_static: bool,
    exclude_action_keys: Sequence[str] = (),
) -> ActionSlotCatalog:
    """Filter writable slots without renumbering action_slot_id values."""
    excluded = {str(k) for k in exclude_action_keys}
    ordered: list[ActionSlot] = []
    for slot_id in episode_slots.ordered_slot_ids:
        slot = episode_slots.get(slot_id)
        if slot is None:
            continue
        if slot.action_key in excluded:
            continue
        rule = validator.policy.field_for_action(slot.action_key)
        if rule is None or not rule.is_mutable:
            continue
        if rule.is_episode_locked and not include_static:
            continue
        ordered.append(slot)
    by_id = {slot.action_slot_id: slot for slot in ordered}
    return ActionSlotCatalog(
        slots_by_id=by_id,
        ordered_slot_ids=tuple(slot.action_slot_id for slot in ordered),
    )


def build_a3_v2_1_output_schema(
    *,
    slots: ActionSlotCatalog,
    residual_m: int,
    query_index: int,
) -> dict[str, Any]:
    modes = sorted(REFLECTION_MODES_Q1) if int(query_index) <= 1 else sorted(
        REFLECTION_MODES_AFTER
    )
    return {
        "type": "object",
        "required": ["reflection_update", "strategy_label", "selections"],
        "additionalProperties": False,
        "properties": {
            "reflection_update": {
                "type": "object",
                "required": ["mode", "hypothesis"],
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "string", "enum": modes},
                    "hypothesis": {
                        "type": "string",
                        "maxLength": MAX_HYPOTHESIS_CHARS_V2_1,
                    },
                },
            },
            "strategy_label": {"type": "string"},
            "selections": build_selections_schema(slots, residual_m=residual_m),
        },
    }


def build_a3_v2_1_repair_schema(
    *,
    slots: ActionSlotCatalog,
    residual_m: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["selections"],
        "additionalProperties": False,
        "properties": {
            "selections": build_selections_schema(slots, residual_m=residual_m),
        },
    }


def build_a3_v2_1_prompt_payload(
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
    episode_slot_map: Sequence[Mapping[str, Any]] | None = None,
    local_repair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES

    safe_visible = {
        str(k): v
        for k, v in dict(visible_anchor).items()
        if str(k) not in PROXY_RAW_FEATURE_NAMES
    }
    safe_current = {
        str(k): v
        for k, v in dict(current_application).items()
        if str(k) not in PROXY_RAW_FEATURE_NAMES
    }
    after_feedback = int(query_index) > 1
    payload: dict[str, Any] = {
        "task": (
            f"{ATTACK_OBJECTIVE_V2_1} {REFLECTION_PURPOSE_V2_1} "
            "Choose exactly one query candidate via action_slot_id -> choice_id "
            "selections. At q=1 set reflection_update.mode=INITIALIZE. After a "
            "submitted BLOCK/INVALID, first update reflection_update "
            "(RETAIN/REVISE/ABANDON) using only public label-only history, then "
            "select the next candidate in the SAME response."
        ),
        "attack_objective": ATTACK_OBJECTIVE_V2_1,
        "reflection_purpose": REFLECTION_PURPOSE_V2_1,
        "prompt_version": PROMPT_VERSION_A3_V2_1,
        "budget": {
            "q_max": int(budget.q_max),
            "m_max": int(budget.m_max),
            "q_remaining": int(q_remaining),
            "query_index": int(query_index),
            "static_edit_cost": int(static_edit_cost),
            "residual_m": int(residual_m),
            "notes": [
                "1 <= len(selections) <= residual_m.",
                "Output only reflection_update, strategy_label, selections.",
                "Never emit raw values, action_key, reference_id, or changes.",
                "You do not know which selected action caused BLOCK.",
                "Hypothesis maxLength is 512 characters; keep it concise.",
            ],
        },
        "episode_static_rules": STATIC_RULE_DISCLOSURE_V2_1,
        "reflection_mode_definitions": dict(REFLECTION_MODE_DEFINITIONS_V2_1),
        "original_anchor": {
            "case_id": case_id,
            "visible_fields": to_jsonable(safe_visible),
        },
        "current_application": {
            "visible_fields": to_jsonable(safe_current),
        },
        "locked_episode_static": to_jsonable(dict(locked_static_values)),
        "action_slots": list(slot_entries),
        "episode_action_slot_map": list(episode_slot_map or slot_entries),
        "episodic_memory": list(episodic_memory),
        "output_schema": build_a3_v2_1_output_schema(
            slots=slots, residual_m=residual_m, query_index=query_index
        ),
        "hard_contract": {
            "output_may_contain_only": [
                "reflection_update",
                "strategy_label",
                "selections",
                "mode",
                "hypothesis",
            ],
            "output_must_not_contain": sorted(_FORBIDDEN_OUTPUT_KEYS),
            "selection_cardinality": "1 <= len(selections) <= residual_m",
            "action_slot_ids_are_episode_stable": True,
            "block_semantics": (
                "BLOCK is evidence only that the whole submitted candidate "
                "failed. You do not know which selected action, field, value "
                "or interaction caused BLOCK."
            ),
        },
        "unavailable_information": GENERIC_UNAVAILABLE_NOTICE,
        "post_feedback_reflection_required": after_feedback,
    }
    if local_repair is not None:
        payload["local_selection_repair"] = dict(local_repair)
        payload["repair_output_schema"] = build_a3_v2_1_repair_schema(
            slots=slots, residual_m=residual_m
        )
    assert_a3_v2_1_prompt_hard_contract(payload)
    return payload


def assert_a3_v2_1_prompt_hard_contract(payload: Mapping[str, Any]) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_A3_V2_1:
        raise ValueError("A3 V2.1 prompt_version mismatch.")
    if ATTACK_OBJECTIVE_V2_1 not in str(payload.get("attack_objective") or ""):
        raise ValueError("A3 V2.1 missing PASS-oriented attack objective.")
    if "obtain PASS" not in str(payload.get("task") or ""):
        raise ValueError("A3 V2.1 task missing PASS objective.")
    # Diagnostic cause-finding must not be stated as the objective.
    objective_blob = " ".join(
        [
            str(payload.get("attack_objective") or ""),
            str(payload.get("task") or ""),
            str(payload.get("reflection_purpose") or ""),
        ]
    ).lower()
    for banned in (
        "find a blocking rule",
        "key risk factor",
        "isolate its effect",
        "identify the true reason",
    ):
        if banned in objective_blob and "not for" not in objective_blob:
            # Allow negation phrasing in reflection_purpose.
            pass
    if "diagnosing the defender" not in str(payload.get("reflection_purpose") or ""):
        raise ValueError("A3 V2.1 must state reflection is not defender diagnosis.")
    if STATIC_RULE_DISCLOSURE_V2_1 not in str(payload.get("episode_static_rules") or ""):
        raise ValueError("A3 V2.1 missing episode_static_rules disclosure.")
    modes = payload.get("reflection_mode_definitions") or {}
    for key in ("INITIALIZE", "RETAIN", "REVISE", "ABANDON"):
        if key not in modes:
            raise ValueError(f"A3 V2.1 missing reflection mode definition {key}.")
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(_NAMED_HIDDEN_TERMS):
        if prompt_contains_hidden_term(text, term):
            raise ValueError(f"A3 V2.1 prompt names hidden term {term!r}.")
    for item in payload.get("action_slots") or []:
        if not isinstance(item, Mapping):
            continue
        from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES

        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("A3 V2.1 action_slots expose proxy raw action_key.")


def parse_a3_v2_1_strategic_response(
    text: str,
    *,
    query_index: int,
    residual_m: int,
    require_reflection: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Parse V2.1 strategic response with 512-char hypothesis limit."""
    from attack_lab.archive.contracts.a3_v2_contract import (
        _extract_json_object,
        _normalize_selections,
        _reject_forbidden_keys,
    )

    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        return None, "parse_error"
    forbidden = _reject_forbidden_keys(payload)
    if forbidden:
        return None, forbidden
    if require_reflection:
        reflection = payload.get("reflection_update")
        if not isinstance(reflection, Mapping):
            return None, "missing_reflection_update"
        mode = reflection.get("mode")
        hypothesis = reflection.get("hypothesis")
        if not isinstance(mode, str) or not mode.strip():
            return None, "missing_reflection_mode"
        mode = mode.strip().upper()
        allowed = REFLECTION_MODES_Q1 if int(query_index) <= 1 else REFLECTION_MODES_AFTER
        if mode not in allowed:
            return None, "invalid_reflection_mode"
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            return None, "missing_hypothesis"
        hypothesis = hypothesis.strip()
        if len(hypothesis) > MAX_HYPOTHESIS_CHARS_V2_1:
            return None, "hypothesis_too_long"
        if set(reflection.keys()) - {"mode", "hypothesis"}:
            return None, "schema_error"
    else:
        mode = ""
        hypothesis = ""
    label = payload.get("strategy_label")
    if not isinstance(label, str) or not label.strip():
        return None, "missing_strategy_label"
    selections, status = _normalize_selections(payload.get("selections"))
    if status != "ok" or selections is None:
        return None, status
    if len(selections) < 1:
        return None, "empty_selections"
    if len(selections) > int(residual_m):
        return None, "selection_count_exceeds_residual_m"
    allowed_keys = {"strategy_label", "selections"}
    if require_reflection:
        allowed_keys.add("reflection_update")
    if set(payload.keys()) - allowed_keys:
        return None, "schema_error"
    out: dict[str, Any] = {
        "strategy_label": label.strip(),
        "selections": selections,
    }
    if require_reflection:
        out["reflection_update"] = {"mode": mode, "hypothesis": hypothesis}
    return out, "ok"


def parse_a3_v2_1_repair_selections(
    text: str,
    *,
    residual_m: int,
) -> tuple[dict[str, str] | None, str]:
    return parse_a3_v2_repair_selections(text, residual_m=residual_m)


def render_a3_v2_1_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """V2.1 messages: same structure as V2 with PASS-oriented system framing."""
    system = (
        "You are a fraud-evasion attacker under sparse label-only defender "
        "feedback and a fixed provenance-constrained K-pool action space. "
        f"{ATTACK_OBJECTIVE_V2_1} {REFLECTION_PURPOSE_V2_1} "
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
    if payload.get("local_selection_repair") is not None:
        system = (
            "Local compliance repair only. reflection_update and strategy_label "
            "are already pinned for this real query and must not be re-emitted. "
            "Return ONLY {\"selections\": {...}} using legal action_slot_id -> "
            "choice_id pairs under residual_m. Do not interpret the defender. "
            f"{GENERIC_UNAVAILABLE_NOTICE}"
        )
    user = json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = [
    "ATTACK_OBJECTIVE_V2_1",
    "MAX_HYPOTHESIS_CHARS_V2_1",
    "PROMPT_VERSION_A3_V2_1",
    "REFLECTION_MODE_DEFINITIONS_V2_1",
    "REFLECTION_PURPOSE_V2_1",
    "STATIC_RULE_DISCLOSURE_V2_1",
    "assert_a3_v2_1_prompt_hard_contract",
    "build_a3_v2_1_episode_action_slots",
    "build_a3_v2_1_output_schema",
    "build_a3_v2_1_prompt_payload",
    "build_a3_v2_1_repair_schema",
    "build_v4_choice_catalog",
    "compute_static_cost_and_residual",
    "parse_a3_v2_1_repair_selections",
    "parse_a3_v2_1_strategic_response",
    "public_slot_entries",
    "render_a3_v2_1_messages",
    "resolve_a3_v2_selections",
    "selections_fingerprint",
    "writable_slots_from_episode_map",
]

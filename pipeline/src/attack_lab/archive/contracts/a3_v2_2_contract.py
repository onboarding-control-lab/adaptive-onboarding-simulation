"""A3 V2.2 bounded-cardinality closeout (successor to V2.1).

Preserves V2 / V2.1 historical versions. Fixes only unrecovered
selection_count_exceeds_residual_m by separating strategic-envelope parsing
from selection-compliance repair. residual_m / budget.m_max are dynamic —
never hardcode a literal edit-capacity constant in cardinality logic.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v4_1_contract import (
    GENERIC_UNAVAILABLE_NOTICE,
    ActionSlotCatalog,
)
from attack_lab.archive.contracts.a1_v4_contract import V4ChoiceCatalog
from attack_lab.archive.contracts.a3_v2_1_contract import (
    ATTACK_OBJECTIVE_V2_1,
    MAX_HYPOTHESIS_CHARS_V2_1,
    REFLECTION_MODE_DEFINITIONS_V2_1,
    REFLECTION_PURPOSE_V2_1,
    STATIC_RULE_DISCLOSURE_V2_1,
    build_a3_v2_1_episode_action_slots,
    public_slot_entries,
    writable_slots_from_episode_map,
)
from attack_lab.archive.contracts.a3_v2_contract import (
    REFLECTION_MODES_AFTER,
    REFLECTION_MODES_Q1,
    _FORBIDDEN_OUTPUT_KEYS,
    _NAMED_HIDDEN_TERMS,
    _extract_json_object,
    _normalize_selections,
    _reject_forbidden_keys,
    build_selections_schema,
    compute_static_cost_and_residual,
    prompt_contains_hidden_term,
    resolve_a3_v2_selections,
    selections_fingerprint,
)
from attack_lab.budget import AttackBudget
from attack_lab.types import to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_A3_V2_2 = "a3_episodic_reflective_v2_2_k10_bounded_cardinality"
MAX_HYPOTHESIS_CHARS_V2_2 = MAX_HYPOTHESIS_CHARS_V2_1
ATTACK_OBJECTIVE_V2_2 = ATTACK_OBJECTIVE_V2_1
REFLECTION_PURPOSE_V2_2 = REFLECTION_PURPOSE_V2_1
STATIC_RULE_DISCLOSURE_V2_2 = STATIC_RULE_DISCLOSURE_V2_1
REFLECTION_MODE_DEFINITIONS_V2_2 = dict(REFLECTION_MODE_DEFINITIONS_V2_1)

SELECTIONS_VS_HYPOTHESIS_NOTE = (
    "Your hypothesis may discuss a broader profile strategy, but `selections` "
    "contains ONLY the actual changes submitted on THIS query and must obey "
    "the current residual_m."
)

CARDINALITY_REPAIR_INSTRUCTION = (
    "Your strategic reflection and strategy label are already frozen for this "
    "real query. You proposed too many action selections for the current "
    "mechanical edit capacity residual_m. Return ONLY a compliant selections "
    "object containing between 1 and residual_m choices. Choose yourself which "
    "of your originally proposed legal selections best preserve your frozen "
    "strategy. Do not emit reflection_update or strategy_label. Do not add a "
    "new strategic explanation. Do not interpret defender feedback."
)


def build_a3_v2_2_episode_action_slots(
    catalog: V4ChoiceCatalog,
    *,
    validator: ConstraintValidator,
) -> ActionSlotCatalog:
    return build_a3_v2_1_episode_action_slots(catalog, validator=validator)


def build_a3_v2_2_output_schema(
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
                        "maxLength": MAX_HYPOTHESIS_CHARS_V2_2,
                    },
                },
            },
            "strategy_label": {"type": "string"},
            "selections": build_selections_schema(slots, residual_m=residual_m),
        },
    }


def build_a3_v2_2_repair_schema(
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


def build_a3_v2_2_cardinality_repair_schema(
    *,
    eligible_pairs: Mapping[str, str],
    residual_m: int,
) -> dict[str, Any]:
    """Schema restricted to the model's own mechanically valid proposed pairs.

    Trusted code does not choose the subset; maxProperties follows residual_m.
    """
    r = int(residual_m)
    if r < 1:
        raise ValueError("cardinality repair schema requires residual_m >= 1.")
    properties: dict[str, Any] = {}
    for slot_id, choice_id in sorted(dict(eligible_pairs).items()):
        properties[str(slot_id)] = {
            "type": "string",
            "enum": [str(choice_id)],
        }
    if not properties:
        raise ValueError("cardinality repair requires at least one eligible pair.")
    return {
        "type": "object",
        "required": ["selections"],
        "additionalProperties": False,
        "properties": {
            "selections": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": r,
                "additionalProperties": False,
                "properties": properties,
            }
        },
    }


def filter_mechanically_valid_proposed_pairs(
    selections: Mapping[str, str],
    *,
    slots: ActionSlotCatalog,
    catalog: V4ChoiceCatalog,
) -> dict[str, str]:
    """Keep only proposed pairs that resolve exactly; no ranking or substitution."""
    valid: dict[str, str] = {}
    for slot_id, choice_id in dict(selections).items():
        single, status = resolve_a3_v2_selections(
            {str(slot_id): str(choice_id)},
            slots=slots,
            catalog=catalog,
        )
        # resolve_action_slot_selections returns ("",) on success, not "ok".
        if single is not None and status in {"", "ok"}:
            valid[str(slot_id)] = str(choice_id)
    return valid


def build_a3_v2_2_prompt_payload(
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
    r = int(residual_m)
    payload: dict[str, Any] = {
        "task": (
            f"{ATTACK_OBJECTIVE_V2_2} {REFLECTION_PURPOSE_V2_2} "
            "Choose exactly one query candidate via action_slot_id -> choice_id "
            "selections. At q=1 set reflection_update.mode=INITIALIZE. After a "
            "submitted BLOCK/INVALID, first update reflection_update "
            "(RETAIN/REVISE/ABANDON) using only public label-only history, then "
            "select the next candidate in the SAME response. "
            f"{SELECTIONS_VS_HYPOTHESIS_NOTE}"
        ),
        "attack_objective": ATTACK_OBJECTIVE_V2_2,
        "reflection_purpose": REFLECTION_PURPOSE_V2_2,
        "prompt_version": PROMPT_VERSION_A3_V2_2,
        "budget": {
            "q_max": int(budget.q_max),
            "m_max": int(budget.m_max),
            "q_remaining": int(q_remaining),
            "query_index": int(query_index),
            "static_edit_cost": int(static_edit_cost),
            "residual_m": r,
            "maximum_submitted_action_selections_this_query": r,
            "notes": [
                "1 <= len(selections) <= residual_m.",
                f"current residual_m: {r}",
                f"maximum submitted action selections this query: {r}",
                SELECTIONS_VS_HYPOTHESIS_NOTE,
                "Output only reflection_update, strategy_label, selections.",
                "Never emit raw values, action_key, reference_id, or changes.",
                "You do not know which selected action caused BLOCK.",
                "Hypothesis maxLength is 512 characters; keep it concise.",
            ],
        },
        "episode_static_rules": STATIC_RULE_DISCLOSURE_V2_2,
        "reflection_mode_definitions": dict(REFLECTION_MODE_DEFINITIONS_V2_2),
        "selections_vs_hypothesis": SELECTIONS_VS_HYPOTHESIS_NOTE,
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
        "output_schema": build_a3_v2_2_output_schema(
            slots=slots, residual_m=r, query_index=query_index
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
        repair = dict(local_repair)
        payload["local_selection_repair"] = repair
        eligible = repair.get("eligible_proposed_selections")
        if repair.get("cardinality_repair") and isinstance(eligible, Mapping) and eligible:
            payload["repair_output_schema"] = build_a3_v2_2_cardinality_repair_schema(
                eligible_pairs=eligible, residual_m=r
            )
        else:
            payload["repair_output_schema"] = build_a3_v2_2_repair_schema(
                slots=slots, residual_m=r
            )
    assert_a3_v2_2_prompt_hard_contract(payload)
    return payload


def assert_a3_v2_2_prompt_hard_contract(payload: Mapping[str, Any]) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_A3_V2_2:
        raise ValueError("A3 V2.2 prompt_version mismatch.")
    if ATTACK_OBJECTIVE_V2_2 not in str(payload.get("attack_objective") or ""):
        raise ValueError("A3 V2.2 missing PASS-oriented attack objective.")
    if "obtain PASS" not in str(payload.get("task") or ""):
        raise ValueError("A3 V2.2 task missing PASS objective.")
    if "diagnosing the defender" not in str(payload.get("reflection_purpose") or ""):
        raise ValueError("A3 V2.2 must state reflection is not defender diagnosis.")
    if STATIC_RULE_DISCLOSURE_V2_2 not in str(payload.get("episode_static_rules") or ""):
        raise ValueError("A3 V2.2 missing episode_static_rules disclosure.")
    if SELECTIONS_VS_HYPOTHESIS_NOTE not in str(
        payload.get("selections_vs_hypothesis") or ""
    ):
        raise ValueError("A3 V2.2 missing selections-vs-hypothesis note.")
    budget = payload.get("budget") or {}
    if "residual_m" not in budget:
        raise ValueError("A3 V2.2 budget missing residual_m.")
    if "maximum_submitted_action_selections_this_query" not in budget:
        raise ValueError("A3 V2.2 budget missing maximum selections this query.")
    if int(budget["maximum_submitted_action_selections_this_query"]) != int(
        budget["residual_m"]
    ):
        raise ValueError("A3 V2.2 max selections must equal residual_m.")
    modes = payload.get("reflection_mode_definitions") or {}
    for key in ("INITIALIZE", "RETAIN", "REVISE", "ABANDON"):
        if key not in modes:
            raise ValueError(f"A3 V2.2 missing reflection mode definition {key}.")
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(_NAMED_HIDDEN_TERMS):
        if prompt_contains_hidden_term(text, term):
            raise ValueError(f"A3 V2.2 prompt names hidden term {term!r}.")
    # Cardinality must not hardcode a fixed m=2 capacity string as the rule.
    schema = payload.get("output_schema") or {}
    selections_schema = (schema.get("properties") or {}).get("selections") or {}
    if "maxProperties" in selections_schema:
        if int(selections_schema["maxProperties"]) != int(budget["residual_m"]):
            raise ValueError("A3 V2.2 selections.maxProperties must equal residual_m.")


def parse_a3_v2_2_strategic_response(
    text: str,
    *,
    query_index: int,
    residual_m: int,
    require_reflection: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Two-stage parse: preserve valid strategic envelope under over-cardinality."""
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
        if len(hypothesis) > MAX_HYPOTHESIS_CHARS_V2_2:
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
    allowed_keys = {"strategy_label", "selections"}
    if require_reflection:
        allowed_keys.add("reflection_update")
    if set(payload.keys()) - allowed_keys:
        return None, "schema_error"

    out: dict[str, Any] = {
        "strategy_label": label.strip(),
        "selections": selections,
        "strategic_envelope_valid": True,
        "selection_status": "ok",
    }
    if require_reflection:
        out["reflection_update"] = {"mode": mode, "hypothesis": hypothesis}

    if len(selections) > int(residual_m):
        out["selection_status"] = "selection_count_exceeds_residual_m"
        # Envelope preserved; not a total strategic parse failure.
        return out, "selection_count_exceeds_residual_m"
    return out, "ok"


def parse_a3_v2_2_repair_selections(
    text: str,
    *,
    residual_m: int,
    eligible_pairs: Mapping[str, str] | None = None,
) -> tuple[dict[str, str] | None, str]:
    """Parse selection-only repair; optionally restrict to eligible proposed pairs."""
    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        return None, "parse_error"
    forbidden = _reject_forbidden_keys(payload)
    if forbidden:
        return None, forbidden
    if "reflection_update" in payload:
        return None, "reflection_immutable"
    if "strategy_label" in payload:
        return None, "strategy_label_immutable"
    selections, status = _normalize_selections(payload.get("selections"))
    if status != "ok" or selections is None:
        return None, status
    if len(selections) < 1:
        return None, "empty_selections"
    if len(selections) > int(residual_m):
        return None, "selection_count_exceeds_residual_m"
    if eligible_pairs is not None:
        allowed = {str(k): str(v) for k, v in dict(eligible_pairs).items()}
        for slot_id, choice_id in selections.items():
            if slot_id not in allowed:
                return None, "selection_not_in_eligible_proposed_set"
            if allowed[slot_id] != choice_id:
                return None, "selection_not_in_eligible_proposed_set"
    extra = set(payload.keys()) - {"selections"}
    if extra:
        return None, "schema_error"
    return selections, "ok"


def render_a3_v2_2_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a fraud-evasion attacker under sparse label-only defender "
        "feedback and a fixed provenance-constrained K-pool action space. "
        f"{ATTACK_OBJECTIVE_V2_2} {REFLECTION_PURPOSE_V2_2} "
        f"{SELECTIONS_VS_HYPOTHESIS_NOTE} "
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


__all__ = [
    "ATTACK_OBJECTIVE_V2_2",
    "CARDINALITY_REPAIR_INSTRUCTION",
    "MAX_HYPOTHESIS_CHARS_V2_2",
    "PROMPT_VERSION_A3_V2_2",
    "REFLECTION_MODE_DEFINITIONS_V2_2",
    "REFLECTION_PURPOSE_V2_2",
    "SELECTIONS_VS_HYPOTHESIS_NOTE",
    "STATIC_RULE_DISCLOSURE_V2_2",
    "assert_a3_v2_2_prompt_hard_contract",
    "build_a3_v2_2_cardinality_repair_schema",
    "build_a3_v2_2_episode_action_slots",
    "build_a3_v2_2_output_schema",
    "build_a3_v2_2_prompt_payload",
    "build_a3_v2_2_repair_schema",
    "compute_static_cost_and_residual",
    "filter_mechanically_valid_proposed_pairs",
    "parse_a3_v2_2_repair_selections",
    "parse_a3_v2_2_strategic_response",
    "public_slot_entries",
    "render_a3_v2_2_messages",
    "resolve_a3_v2_selections",
    "selections_fingerprint",
    "writable_slots_from_episode_map",
]

"""A3 V2 hard-action-contract helpers (episodic reflective successor to Stage-B B2).

Preserves historical B2 (`a3_neutral_grounded_structured_reflection_v1`).
Reuses A1 V4.x choice catalogues / action slots / ReferenceSelection resolution.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_1_contract import (
    GENERIC_UNAVAILABLE_NOTICE,
    ActionSlot,
    ActionSlotCatalog,
    prompt_contains_hidden_term,
    resolve_action_slot_selections,
)
from attack_lab.archive.contracts.a1_v4_contract import (
    V4ChoiceCatalog,
    build_v4_choice_catalog,
)
from attack_lab.budget import AttackBudget, compute_edit_metrics
from attack_lab.reference_actions import ReferenceSelection
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import AttackProposal, to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_A3_V2 = "a3_episodic_reflective_v2_k10_hard_contract"

REFLECTION_MODES_Q1 = frozenset({"INITIALIZE"})
REFLECTION_MODES_AFTER = frozenset({"RETAIN", "REVISE", "ABANDON"})
MAX_HYPOTHESIS_CHARS = 240

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "changes",
        "action_key",
        "action_keys",
        "feature",
        "features",
        "reference_id",
        "reference_ids",
        "raw_value",
        "value",
        "adaptation_note",
        "choice_ids",
    }
)
_NAMED_HIDDEN_TERMS = frozenset(
    {
        *PROXY_RAW_FEATURE_NAMES,
        "risk_score",
        "d1_risk_score",
        "feature_importance",
        "shap",
        "d1_threshold",
        "gradients",
        "fraud_bool",
        "true_rejection_reason",
        "proxy_raw_targets_forbidden",
    }
)
def compute_static_cost_and_residual(
    *,
    validator: ConstraintValidator,
    anchor: Mapping[str, Any],
    locked_static_values: Mapping[str, Any],
    m_max: int,
) -> tuple[int, int]:
    """static_edit_cost vs ORIGINAL_ANCHOR and residual_m = m_max - cost."""
    locked = dict(locked_static_values)
    if not locked:
        return 0, int(m_max)
    proposal = AttackProposal(changes=locked)
    projected = validator.project_for_billing(
        anchor, proposal, locked_values=locked
    )
    _, occupied, _, _ = compute_edit_metrics(
        anchor=anchor,
        candidate=projected,
        mutable_feature_names=validator.mutable_feature_names(),
        previous_candidate=None,
    )
    cost = int(occupied)
    return cost, max(0, int(m_max) - cost)


def build_a3_v2_action_slots(
    catalog: V4ChoiceCatalog,
    *,
    validator: ConstraintValidator,
    include_static: bool,
    exclude_action_keys: Sequence[str] = (),
) -> ActionSlotCatalog:
    """Writable action slots for the current query (static optional)."""
    excluded = {str(k) for k in exclude_action_keys}
    choice_ids: list[str] = []
    if include_static:
        choice_ids.extend(catalog.static_choice_ids)
    choice_ids.extend(catalog.per_attempt_choice_ids)

    by_action: dict[str, list[str]] = {}
    for choice_id in choice_ids:
        choice = catalog.get(choice_id)
        if choice is None:
            continue
        if choice.action_key in excluded:
            continue
        rule = validator.policy.field_for_action(choice.action_key)
        if rule is None or not rule.is_mutable:
            continue
        # Mirror A1 V4 catalogue policy: abstract proxy_action_key allowed;
        # trusted proxy raw feature names never become slot action keys.
        if choice.action_key in PROXY_RAW_FEATURE_NAMES:
            continue
        if rule.agent_action_mode == "proxy_action":
            if (
                rule.proxy_action_key is None
                or choice.action_key != rule.proxy_action_key
            ):
                continue
        elif rule.feature in PROXY_RAW_FEATURE_NAMES:
            continue
        by_action.setdefault(choice.action_key, []).append(choice_id)

    ordered: list[ActionSlot] = []
    for index, action_key in enumerate(sorted(by_action), start=1):
        ordered.append(
            ActionSlot(
                action_slot_id=f"action_slot_{index:02d}",
                action_key=action_key,
                allowed_choice_ids=tuple(by_action[action_key]),
            )
        )
    by_id = {slot.action_slot_id: slot for slot in ordered}
    return ActionSlotCatalog(
        slots_by_id=by_id,
        ordered_slot_ids=tuple(slot.action_slot_id for slot in ordered),
    )


def public_slot_entries(
    slots: ActionSlotCatalog,
    *,
    validator: ConstraintValidator,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        rule = validator.policy.field_for_action(slot.action_key)
        category = (
            "episode_static"
            if rule is not None and rule.is_episode_locked
            else "per_attempt"
        )
        out.append(
            {
                "action_slot_id": slot.action_slot_id,
                "action_key": slot.action_key,
                "category": category,
                "allowed_choice_ids": list(slot.allowed_choice_ids),
            }
        )
    return out


def build_selections_schema(
    slots: ActionSlotCatalog, *, residual_m: int
) -> dict[str, Any]:
    """Per-slot enum schema with additionalProperties=false and cardinality."""
    properties: dict[str, Any] = {}
    for slot_id in slots.ordered_slot_ids:
        slot = slots.get(slot_id)
        assert slot is not None
        properties[slot_id] = {
            "type": "string",
            "enum": list(slot.allowed_choice_ids),
        }
    max_props = max(1, int(residual_m)) if int(residual_m) >= 1 else 1
    return {
        "type": "object",
        "minProperties": 1,
        "maxProperties": max_props,
        "additionalProperties": False,
        "properties": properties,
    }


def build_a3_v2_output_schema(
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
                        "maxLength": MAX_HYPOTHESIS_CHARS,
                    },
                },
            },
            "strategy_label": {"type": "string"},
            "selections": build_selections_schema(slots, residual_m=residual_m),
        },
    }


def build_a3_v2_repair_schema(
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


def build_a3_v2_prompt_payload(
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
    local_repair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
            "Choose exactly one query candidate via action_slot_id -> choice_id "
            "selections. At q=1 set reflection_update.mode=INITIALIZE. After a "
            "submitted BLOCK/INVALID, first update reflection_update "
            "(RETAIN/REVISE/ABANDON) using only public label-only history, then "
            "select the next candidate in the SAME response."
        ),
        "prompt_version": PROMPT_VERSION_A3_V2,
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
            ],
        },
        "original_anchor": {
            "case_id": case_id,
            "visible_fields": to_jsonable(safe_visible),
        },
        "current_application": {
            "visible_fields": to_jsonable(safe_current),
        },
        "locked_episode_static": to_jsonable(dict(locked_static_values)),
        "action_slots": list(slot_entries),
        "episodic_memory": list(episodic_memory),
        "output_schema": build_a3_v2_output_schema(
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
        payload["repair_output_schema"] = build_a3_v2_repair_schema(
            slots=slots, residual_m=residual_m
        )
    assert_a3_v2_prompt_hard_contract(payload)
    return payload


def assert_a3_v2_prompt_hard_contract(payload: Mapping[str, Any]) -> None:
    if str(payload.get("prompt_version")) != PROMPT_VERSION_A3_V2:
        raise ValueError("A3 V2 prompt_version mismatch.")
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(_NAMED_HIDDEN_TERMS):
        if prompt_contains_hidden_term(text, term):
            raise ValueError(f"A3 V2 prompt names hidden term {term!r}.")
    for item in payload.get("action_slots") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("A3 V2 action_slots expose proxy raw action_key.")


def _extract_json_object(text: str) -> Any | None:
    raw = str(text).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _reject_forbidden_keys(node: Any) -> str | None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if str(key) in _FORBIDDEN_OUTPUT_KEYS:
                return f"forbidden_output_key:{key}"
            nested = _reject_forbidden_keys(value)
            if nested:
                return nested
    elif isinstance(node, list):
        for item in node:
            nested = _reject_forbidden_keys(item)
            if nested:
                return nested
    return None


def _normalize_selections(raw: Any) -> tuple[dict[str, str] | None, str]:
    if not isinstance(raw, Mapping) or not raw:
        return None, "empty_selections"
    out: dict[str, str] = {}
    for key, value in raw.items():
        slot_id = str(key).strip()
        if not slot_id:
            return None, "unknown_action_slot_id"
        if not isinstance(value, str) or not value.strip():
            return None, "invalid_choice_id_type"
        out[slot_id] = value.strip()
    return out, "ok"


def parse_a3_v2_strategic_response(
    text: str,
    *,
    query_index: int,
    residual_m: int,
    require_reflection: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Parse full strategic response (reflection + selections)."""
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
        if len(hypothesis) > MAX_HYPOTHESIS_CHARS:
            return None, "hypothesis_too_long"
        extra_ref = set(reflection.keys()) - {"mode", "hypothesis"}
        if extra_ref:
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
    extra = set(payload.keys()) - allowed_keys
    if extra:
        return None, "schema_error"
    out: dict[str, Any] = {
        "strategy_label": label.strip(),
        "selections": selections,
    }
    if require_reflection:
        out["reflection_update"] = {"mode": mode, "hypothesis": hypothesis}
    return out, "ok"


def parse_a3_v2_repair_selections(
    text: str,
    *,
    residual_m: int,
) -> tuple[dict[str, str] | None, str]:
    """Parse repair-only response (selections). No new reflection allowed."""
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
    if len(selections) > int(residual_m):
        return None, "selection_count_exceeds_residual_m"
    extra = set(payload.keys()) - {"selections"}
    if extra:
        return None, "schema_error"
    return selections, "ok"


def resolve_a3_v2_selections(
    selections: Mapping[str, str],
    *,
    slots: ActionSlotCatalog,
    catalog: V4ChoiceCatalog,
) -> tuple[dict[str, ReferenceSelection] | None, str]:
    return resolve_action_slot_selections(
        selections, slots=slots, catalog=catalog
    )


def selections_fingerprint(selections: Mapping[str, str]) -> str:
    """Canonical fingerprint of a slot->choice map (local duplicate detection)."""
    encoded = json.dumps(
        {str(k): str(v) for k, v in sorted(dict(selections).items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded


def render_a3_v2_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Single strategic (or repair) message pair for A3 V2."""
    system = (
        "You are a general-purpose sequential adaptive attacker under sparse "
        "label-only defender feedback and a fixed provenance-constrained K-pool "
        "action space. Emit ONLY the JSON object required by the output schema. "
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
    "GENERIC_UNAVAILABLE_NOTICE",
    "MAX_HYPOTHESIS_CHARS",
    "PROMPT_VERSION_A3_V2",
    "REFLECTION_MODES_AFTER",
    "REFLECTION_MODES_Q1",
    "assert_a3_v2_prompt_hard_contract",
    "build_a3_v2_action_slots",
    "build_a3_v2_output_schema",
    "build_a3_v2_prompt_payload",
    "build_a3_v2_repair_schema",
    "build_selections_schema",
    "build_v4_choice_catalog",
    "compute_static_cost_and_residual",
    "parse_a3_v2_repair_selections",
    "parse_a3_v2_strategic_response",
    "prompt_contains_hidden_term",
    "public_slot_entries",
    "render_a3_v2_messages",
    "resolve_a3_v2_selections",
    "selections_fingerprint",
]

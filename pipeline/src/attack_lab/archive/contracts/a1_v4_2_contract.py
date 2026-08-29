"""A1 V4.2 bounded unique-action-slot contract (successor to V4.1).

Minimal cardinality hardening: each candidate's selections object must satisfy
1 <= len(selections) <= selected_plan.residual_m, expressed in schema and
enforced by the local parser before env.step.

Does not modify ``a1_oneshot_v4_1_unique_action_slots`` or earlier versions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_contract import (
    PROMPT_VERSION_V4,
    StaticPlanOption,
    V4ChoiceCatalog,
    build_v4_choice_catalog,
    build_v4_static_plan_options,
    static_locks_and_cost,
    static_plan_by_id,
)
from attack_lab.archive.contracts.a1_v4_1_contract import (
    GENERIC_UNAVAILABLE_NOTICE,
    PROMPT_VERSION_V4_1,
    ActionSlotCatalog,
    build_v4_1_action_slots,
    classify_attacker_visible_term_context,
    resolve_action_slot_selections,
    scan_attacker_visible_hidden_mentions,
)
from attack_lab.budget import AttackBudget
from attack_lab.governance_view import GovernanceView
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_V4_2 = "a1_oneshot_v4_2_bounded_unique_action_slots"

DIVERSIFICATION_PRINCIPLE_V4_2 = (
    "Because the full plan must be created before any feedback is available, "
    "select one feasible static_plan_id and construct a diversified ordered "
    "portfolio of exactly q_max query candidates using only action_slot_id to "
    "choice_id selections allowed for that plan. Each candidate must include "
    "between 1 and that plan's residual_m selections inclusive. Do not invent "
    "action keys, reference ids, or raw values. Do not infer or target any "
    "hidden defender internals."
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
        "choice_ids",
    }
)


def _selections_schema(*, slot_enum: Sequence[str], max_properties: int) -> dict[str, Any]:
    return {
        "type": "object",
        "minProperties": 1,
        "maxProperties": int(max_properties),
        "propertyNames": {"enum": list(slot_enum)},
        "additionalProperties": {"type": "string"},
    }


def _candidate_item_schema(*, slot_enum: Sequence[str], residual_m: int) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["strategy_label", "selections"],
        "additionalProperties": False,
        "properties": {
            "strategy_label": {"type": "string"},
            "selections": _selections_schema(
                slot_enum=slot_enum, max_properties=int(residual_m)
            ),
        },
    }


def build_v4_2_plan_conditioned_output_schema(
    *,
    q_max: int,
    static_plans: Sequence[StaticPlanOption],
    slot_enum: Sequence[str],
) -> dict[str, Any]:
    """oneOf branches: each static_plan_id binds selections.maxProperties to residual_m."""
    branches: list[dict[str, Any]] = []
    for plan in static_plans:
        residual = max(1, int(plan.residual_m)) if int(plan.residual_m) >= 1 else 1
        # residual_m may be 0 only for degenerate q_max=1 plans; still bound to max(residual,1)
        # for schema when residual_m >= 1 is the normal case.
        max_props = int(plan.residual_m) if int(plan.residual_m) >= 1 else 1
        branches.append(
            {
                "type": "object",
                "required": ["static_plan_id", "candidates"],
                "additionalProperties": False,
                "properties": {
                    "static_plan_id": {"const": plan.static_plan_id},
                    "candidates": {
                        "type": "array",
                        "minItems": int(q_max),
                        "maxItems": int(q_max),
                        "items": _candidate_item_schema(
                            slot_enum=slot_enum, residual_m=max_props
                        ),
                    },
                },
            }
        )
    return {"oneOf": branches}


def build_v4_2_repair_output_schema(
    *,
    slot_enum: Sequence[str],
    residual_m: int,
    requested_indices: Sequence[int],
) -> dict[str, Any]:
    """Repair schema with pinned residual_m as selections.maxProperties."""
    max_props = int(residual_m) if int(residual_m) >= 1 else 1
    return {
        "type": "object",
        "required": ["replacements"],
        "additionalProperties": False,
        "properties": {
            "replacements": {
                "type": "array",
                "minItems": len(list(requested_indices)),
                "maxItems": len(list(requested_indices)),
                "items": {
                    "type": "object",
                    "required": ["candidate_index", "strategy_label", "selections"],
                    "additionalProperties": False,
                    "properties": {
                        "candidate_index": {
                            "type": "integer",
                            "enum": [int(i) for i in requested_indices],
                        },
                        "strategy_label": {"type": "string"},
                        "selections": _selections_schema(
                            slot_enum=slot_enum, max_properties=max_props
                        ),
                    },
                },
            }
        },
    }


def build_v4_2_prompt_payload(
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
    """Attacker-public V4.2 payload with plan-conditioned selection cardinality."""
    view = GovernanceView.from_policy(
        validator.policy,
        budget=budget,
        read_only_context_fields=pool.read_only_context_fields,
        enabled_action_keys=validator.enabled_action_keys,
    )
    _ = view
    safe_visible = {
        str(key): value
        for key, value in dict(visible_anchor).items()
        if str(key) not in PROXY_RAW_FEATURE_NAMES
    }
    slot_enum = list(action_slots.ordered_slot_ids)
    plan_cardinality = [
        {
            "static_plan_id": plan.static_plan_id,
            "residual_m": int(plan.residual_m),
            "selections_maxProperties": int(plan.residual_m)
            if int(plan.residual_m) >= 1
            else 1,
        }
        for plan in static_plans
    ]
    payload = {
        "task": (
            "Select one feasible static_plan_id and plan an ordered sequence of "
            "exactly q_max unique query candidates using action_slot selections. "
            "Each candidate must include between 1 and that plan's residual_m "
            "selections. Freeze the full sequence now; no later revision is allowed."
        ),
        "prompt_version": PROMPT_VERSION_V4_2,
        "budget": {
            "q_max": int(q_max),
            "m_max": int(budget.m_max),
            "notes": [
                f"Return exactly {int(q_max)} candidates (q_max={int(q_max)}).",
                "Select exactly one static_plan_id from static_plan_options.",
                "Each candidate selections object maps action_slot_id -> choice_id.",
                "For the chosen static_plan_id, enforce "
                "1 <= len(selections) <= residual_m (see plan_selection_cardinality).",
                "Each action_slot_id may appear at most once (JSON object key).",
                "Each choice_id must belong to that action_slot_id.",
                "Output only static_plan_id, strategy_label, and selections — "
                "never action_key, reference_id, choice_ids arrays, or raw values.",
                "Candidates must be unique.",
                "Local slot repair may replace only invalid candidate indices "
                "before any defender query; it does not consume Q.",
            ],
        },
        "anchor": {
            "case_id": case_id,
            "visible_fields": to_jsonable(safe_visible),
        },
        "action_slots": action_slots.public_slots(),
        "choice_catalogue": catalog.public_choices(
            list(catalog.static_choice_ids) + list(catalog.per_attempt_choice_ids)
        ),
        "static_plan_options": [plan.to_public_dict() for plan in static_plans],
        "allowed_static_plan_ids": [plan.static_plan_id for plan in static_plans],
        "plan_selection_cardinality": plan_cardinality,
        "output_schema": build_v4_2_plan_conditioned_output_schema(
            q_max=q_max, static_plans=static_plans, slot_enum=slot_enum
        ),
        "planning_principle": DIVERSIFICATION_PRINCIPLE_V4_2,
        "hard_contract": {
            "output_may_contain_only": [
                "static_plan_id",
                "candidates",
                "strategy_label",
                "selections",
                "action_slot_id",
                "choice_id",
            ],
            "output_must_not_contain": sorted(_FORBIDDEN_OUTPUT_KEYS),
            "selection_cardinality": (
                "1 <= len(selections) <= selected_static_plan.residual_m"
            ),
        },
        "unavailable_information": GENERIC_UNAVAILABLE_NOTICE,
    }
    assert_v4_2_prompt_hard_contract(payload)
    return payload


def assert_v4_2_prompt_hard_contract(payload: Mapping[str, Any]) -> None:
    """Fail closed: V4.2 version string, no named hidden internals, schema bounds."""
    if str(payload.get("prompt_version")) != PROMPT_VERSION_V4_2:
        raise ValueError("V4.2 payload prompt_version mismatch.")
    if str(payload.get("prompt_version")) in {PROMPT_VERSION_V4, PROMPT_VERSION_V4_1}:
        raise ValueError("V4.2 builder must not emit V4/V4.1 version strings.")
    schema = payload.get("output_schema") or {}
    branches = schema.get("oneOf")
    if not isinstance(branches, list) or not branches:
        raise ValueError("V4.2 output_schema must use plan-conditioned oneOf branches.")
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValueError("V4.2 schema branch must be an object.")
        props = branch.get("properties") or {}
        cands = (props.get("candidates") or {}).get("items") or {}
        selections = ((cands.get("properties") or {}).get("selections") or {})
        if "maxProperties" not in selections:
            raise ValueError("V4.2 selections schema must set maxProperties.")
        if int(selections.get("minProperties") or 0) != 1:
            raise ValueError("V4.2 selections schema must set minProperties=1.")
    for item in payload.get("choice_catalogue") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4.2 choice_catalogue exposes a proxy raw action_key.")
    for item in payload.get("action_slots") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4.2 action_slots expose a proxy raw action_key.")
    visible = ((payload.get("anchor") or {}).get("visible_fields") or {})
    overlap = sorted(set(visible).intersection(PROXY_RAW_FEATURE_NAMES))
    if overlap:
        raise ValueError(f"V4.2 visible_fields expose proxy raw names: {overlap}.")
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(_NAMED_HIDDEN_TERMS):
        if term in text:
            raise ValueError(
                f"V4.2 prompt must not name hidden term {term!r} "
                "(including denylist-style mentions)."
            )


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


def _normalize_selections(
    raw: Any,
) -> tuple[dict[str, str] | None, str]:
    if not isinstance(raw, Mapping):
        return None, "schema_error"
    if not raw:
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


def _enforce_selection_cardinality(
    selections: Mapping[str, str], *, residual_m: int
) -> str | None:
    n = len(selections)
    if n < 1:
        return "empty_selections"
    if n > int(residual_m):
        return "selection_count_exceeds_residual_m"
    return None


def parse_a1_v4_2_plan(
    text: str,
    *,
    q_max: int,
    static_plans: Sequence[StaticPlanOption],
) -> tuple[dict[str, Any] | None, str]:
    """Parse V4.2 plan; retain candidates for slot-level residual_m evaluation.

    Over-cardinality selections are NOT whole-plan parse failures. Portfolio
    evaluation rejects only the offending candidate indices so valid slots stay
    pinned and only those indices enter local repair.
    """
    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        return None, "parse_error"
    forbidden = _reject_forbidden_keys(payload)
    if forbidden:
        return None, forbidden
    static_plan_id = payload.get("static_plan_id")
    if not isinstance(static_plan_id, str) or not static_plan_id.strip():
        return None, "missing_static_plan_id"
    plan = static_plan_by_id(static_plans, static_plan_id)
    if plan is None:
        return None, "unknown_static_plan_id"
    residual_m = int(plan.residual_m)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None, "schema_error"
    if len(candidates) != int(q_max):
        return None, "wrong_candidate_count"
    parsed: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            return None, "schema_error"
        if "choice_ids" in item:
            return None, "forbidden_output_key:choice_ids"
        if "changes" in item or "action_key" in item or "reference_id" in item:
            return None, "forbidden_output_key"
        label = item.get("strategy_label")
        selections_raw = item.get("selections")
        if not isinstance(label, str) or not label.strip():
            return None, "missing_strategy_label"
        selections, status = _normalize_selections(selections_raw)
        if status != "ok" or selections is None:
            return None, status
        # Intentionally do not enforce residual_m here — slot repair requires
        # retaining over-cardinality candidates for candidate-level evaluation.
        extra = set(item.keys()) - {"strategy_label", "selections"}
        if extra:
            return None, "schema_error"
        parsed.append(
            {
                "strategy_label": label.strip(),
                "selections": selections,
            }
        )
    return {
        "static_plan_id": plan.static_plan_id,
        "residual_m": residual_m,
        "candidates": parsed,
    }, "ok"


def parse_a1_v4_2_slot_replacements(
    text: str,
    *,
    requested_indices: Sequence[int],
    residual_m: int,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Parse V4.2 repairs; enforce pinned residual_m on every replacement."""
    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        return None, "parse_error"
    forbidden = _reject_forbidden_keys(payload)
    if forbidden:
        return None, forbidden
    if "static_plan_id" in payload:
        return None, "static_plan_immutable"
    if "candidates" in payload:
        return None, "schema_error"
    replacements = payload.get("replacements")
    if not isinstance(replacements, list):
        return None, "schema_error"
    requested = {int(i) for i in requested_indices}
    parsed: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in replacements:
        if not isinstance(item, Mapping):
            return None, "schema_error"
        if "choice_ids" in item:
            return None, "forbidden_output_key:choice_ids"
        if "changes" in item or "action_key" in item or "reference_id" in item:
            return None, "forbidden_output_key"
        try:
            index = int(item.get("candidate_index"))
        except (TypeError, ValueError):
            return None, "invalid_candidate_index"
        if index not in requested or index in seen:
            return None, "replacement_indices_invalid"
        label = item.get("strategy_label")
        selections, status = _normalize_selections(item.get("selections"))
        if not isinstance(label, str) or not label.strip():
            return None, "missing_strategy_label"
        if status != "ok" or selections is None:
            return None, status
        cardinality = _enforce_selection_cardinality(
            selections, residual_m=int(residual_m)
        )
        if cardinality is not None:
            return None, cardinality
        extra = set(item.keys()) - {
            "candidate_index",
            "strategy_label",
            "selections",
        }
        if extra:
            return None, "schema_error"
        seen.add(index)
        parsed.append(
            {
                "candidate_index": index,
                "strategy_label": label.strip(),
                "selections": selections,
            }
        )
    if seen != requested:
        return None, "replacement_indices_invalid"
    return parsed, "ok"


__all__ = [
    "DIVERSIFICATION_PRINCIPLE_V4_2",
    "GENERIC_UNAVAILABLE_NOTICE",
    "PROMPT_VERSION_V4_2",
    "assert_v4_2_prompt_hard_contract",
    "build_v4_1_action_slots",
    "build_v4_2_plan_conditioned_output_schema",
    "build_v4_2_prompt_payload",
    "build_v4_2_repair_output_schema",
    "build_v4_choice_catalog",
    "build_v4_static_plan_options",
    "classify_attacker_visible_term_context",
    "parse_a1_v4_2_plan",
    "parse_a1_v4_2_slot_replacements",
    "resolve_action_slot_selections",
    "scan_attacker_visible_hidden_mentions",
    "static_locks_and_cost",
    "static_plan_by_id",
]

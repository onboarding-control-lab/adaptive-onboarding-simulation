"""A1 V4.1 unique-action-slot contract (successor to V4).

Preserves the V4 hard contract (opaque choices + static plans) while making
duplicate underlying action selection inside one candidate structurally
impossible via action_slot_id object keys.

Does not modify ``a1_oneshot_v4_hard_contract``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.archive.contracts.a1_v4_contract import (
    PROMPT_VERSION_V4,
    V4ChoiceCatalog,
    build_v4_choice_catalog,
    build_v4_static_plan_options,
    static_locks_and_cost,
    static_plan_by_id,
)
from attack_lab.budget import AttackBudget
from attack_lab.governance_view import GovernanceView
from attack_lab.reference_actions import ReferenceSelection
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_V4_1 = "a1_oneshot_v4_1_unique_action_slots"

DIVERSIFICATION_PRINCIPLE_V4_1 = (
    "Because the full plan must be created before any feedback is available, "
    "select one feasible static_plan_id and construct a diversified ordered "
    "portfolio of exactly q_max query candidates using only action_slot_id to "
    "choice_id selections allowed for that plan. Do not invent action keys, "
    "reference ids, or raw values. Do not infer or target any hidden defender "
    "internals."
)

GENERIC_UNAVAILABLE_NOTICE = (
    "Hidden defender internals and trusted proxy targets are unavailable. "
    "You receive only the declared attacker-visible context and label-only "
    "defender feedback."
)

# Terms that must not appear in V4.1 attacker-visible prompts (even as denylist).
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
# Short tokens that collide with ordinary English require specialised matching.
_BOUNDARY_AWARE_HIDDEN_TERMS = frozenset({"shap"})
# Match bare SHAP / compound identifiers (shap_value, feature_importance_or_shap)
# but not ordinary English continuations (shape, shaping, reshaping).
_SHAP_HIDDEN_TERM_RE = re.compile(r"(?i)(?<![A-Za-z])shap(?![a-z])")


def prompt_contains_hidden_term(text: str, term: str) -> bool:
    """Return True if a forbidden hidden term is present in text.

    Most terms use substring matching. ``shap`` matches case-insensitively as
    a bare token or identifier fragment (e.g. ``SHAP``, ``shap_value``,
    ``feature_importance_or_shap``) but not inside ordinary English words
    such as ``shape``, ``shaping``, or ``reshaping``.
    """
    needle = str(term)
    haystack = str(text)
    if needle in _BOUNDARY_AWARE_HIDDEN_TERMS:
        return _SHAP_HIDDEN_TERM_RE.search(haystack) is not None
    return needle in haystack


def find_hidden_term_span(text: str, term: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the first hidden-term match, if any."""
    needle = str(term)
    haystack = str(text)
    if needle in _BOUNDARY_AWARE_HIDDEN_TERMS:
        match = _SHAP_HIDDEN_TERM_RE.search(haystack)
        if match is None:
            return None
        return match.start(), match.end()
    idx = haystack.find(needle)
    if idx < 0:
        return None
    return idx, idx + len(needle)


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
        "choice_ids",  # V4 array form is rejected; use selections object
    }
)


@dataclass(frozen=True, slots=True)
class ActionSlot:
    """One code-generated action slot with allowed opaque choice IDs."""

    action_slot_id: str
    action_key: str
    allowed_choice_ids: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "action_slot_id": self.action_slot_id,
            "action_key": self.action_key,
            "allowed_choice_ids": list(self.allowed_choice_ids),
        }


@dataclass(frozen=True, slots=True)
class ActionSlotCatalog:
    """Episode-scoped action slots (one underlying action_key each)."""

    slots_by_id: Mapping[str, ActionSlot]
    ordered_slot_ids: tuple[str, ...]

    def get(self, action_slot_id: str) -> ActionSlot | None:
        return self.slots_by_id.get(str(action_slot_id))

    def public_slots(self) -> list[dict[str, Any]]:
        return [
            self.slots_by_id[slot_id].to_public_dict()
            for slot_id in self.ordered_slot_ids
            if slot_id in self.slots_by_id
        ]


def build_v4_1_action_slots(
    catalog: V4ChoiceCatalog,
    *,
    allowed_query_choice_ids: Sequence[str] | None = None,
) -> ActionSlotCatalog:
    """Group per-attempt choices into unique action slots."""
    allowed = (
        set(str(cid) for cid in allowed_query_choice_ids)
        if allowed_query_choice_ids is not None
        else set(catalog.per_attempt_choice_ids)
    )
    by_action: dict[str, list[str]] = {}
    for choice_id in catalog.per_attempt_choice_ids:
        if choice_id not in allowed:
            continue
        choice = catalog.get(choice_id)
        if choice is None:
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


def resolve_action_slot_selections(
    selections: Mapping[str, Any],
    *,
    slots: ActionSlotCatalog,
    catalog: V4ChoiceCatalog,
) -> tuple[dict[str, ReferenceSelection] | None, str]:
    """Resolve action_slot_id -> choice_id map to ReferenceSelection changes."""
    if not isinstance(selections, Mapping) or not selections:
        return None, "empty_selections"
    changes: dict[str, ReferenceSelection] = {}
    for raw_slot_id, raw_choice_id in selections.items():
        slot_id = str(raw_slot_id)
        if not isinstance(raw_choice_id, str) or not raw_choice_id.strip():
            return None, "invalid_choice_id_type"
        choice_id = raw_choice_id.strip()
        slot = slots.get(slot_id)
        if slot is None:
            return None, "unknown_action_slot_id"
        if choice_id not in set(slot.allowed_choice_ids):
            return None, "choice_not_in_action_slot"
        choice = catalog.get(choice_id)
        if choice is None:
            return None, "unknown_choice_id"
        if choice.action_key != slot.action_key:
            return None, "choice_not_in_action_slot"
        # Structurally one slot == one action_key; defensive guard only.
        if slot.action_key in changes:
            return None, "duplicate_action_in_slot"
        changes[slot.action_key] = choice.as_selection()
    return changes, ""


def build_v4_1_prompt_payload(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    budget: AttackBudget,
    q_max: int,
    visible_anchor: Mapping[str, Any],
    case_id: str,
    catalog: V4ChoiceCatalog,
    static_plans: Sequence[Any],
    action_slots: ActionSlotCatalog,
) -> dict[str, Any]:
    """Attacker-public V4.1 payload: action slots + static plans; no named denylist."""
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
    payload = {
        "task": (
            "Select one feasible static_plan_id and plan an ordered sequence of "
            "exactly q_max unique query candidates using action_slot selections. "
            "Freeze the full sequence now; no later revision is allowed."
        ),
        "prompt_version": PROMPT_VERSION_V4_1,
        "budget": {
            "q_max": int(q_max),
            "m_max": int(budget.m_max),
            "notes": [
                f"Return exactly {int(q_max)} candidates (q_max={int(q_max)}).",
                "Select exactly one static_plan_id from static_plan_options.",
                "Each candidate selections object maps action_slot_id -> choice_id.",
                "Include between 1 and residual_m selections from the chosen plan.",
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
        "output_schema": {
            "type": "object",
            "required": ["static_plan_id", "candidates"],
            "additionalProperties": False,
            "properties": {
                "static_plan_id": {
                    "type": "string",
                    "enum": [plan.static_plan_id for plan in static_plans],
                },
                "candidates": {
                    "type": "array",
                    "minItems": int(q_max),
                    "maxItems": int(q_max),
                    "items": {
                        "type": "object",
                        "required": ["strategy_label", "selections"],
                        "additionalProperties": False,
                        "properties": {
                            "strategy_label": {"type": "string"},
                            "selections": {
                                "type": "object",
                                "minProperties": 1,
                                "propertyNames": {"enum": slot_enum},
                                "additionalProperties": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "planning_principle": DIVERSIFICATION_PRINCIPLE_V4_1,
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
        },
        "unavailable_information": GENERIC_UNAVAILABLE_NOTICE,
    }
    assert_v4_1_prompt_hard_contract(payload)
    return payload


def assert_v4_1_prompt_hard_contract(payload: Mapping[str, Any]) -> None:
    """Fail closed: no named hidden internals in attacker-visible V4.1 prompts."""
    if str(payload.get("prompt_version")) != PROMPT_VERSION_V4_1:
        raise ValueError("V4.1 payload prompt_version mismatch.")
    if str(payload.get("prompt_version")) == PROMPT_VERSION_V4:
        raise ValueError("V4.1 builder must not emit the frozen V4 version string.")
    for item in payload.get("choice_catalogue") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4.1 choice_catalogue exposes a proxy raw action_key.")
    for item in payload.get("action_slots") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4.1 action_slots expose a proxy raw action_key.")
    visible = ((payload.get("anchor") or {}).get("visible_fields") or {})
    overlap = sorted(set(visible).intersection(PROXY_RAW_FEATURE_NAMES))
    if overlap:
        raise ValueError(f"V4.1 visible_fields expose proxy raw names: {overlap}.")
    text = json.dumps(to_jsonable(dict(payload)), sort_keys=True)
    for term in sorted(_NAMED_HIDDEN_TERMS):
        if term in text:
            raise ValueError(
                f"V4.1 prompt must not name hidden term {term!r} "
                "(including denylist-style mentions)."
            )


def classify_attacker_visible_term_context(
    text: str, term: str
) -> str:
    """Development classifier for smoke/audit scanners (not a scientific oracle).

    Returns one of:
      A_hidden_value, B_hidden_field_name, C_prohibition_wording, D_not_found
    """
    haystack = str(text)
    needle = str(term)
    span = find_hidden_term_span(haystack, needle)
    if span is None:
        return "D_not_found"
    idx, end = span
    matched_len = end - idx
    # Wide window: V4 denylist keys can sit >80 chars before a listed term.
    window = haystack[max(0, idx - 240) : idx + matched_len + 120].lower()
    prohibition_markers = (
        "unavailable",
        "forbidden",
        "must not",
        "do not",
        "never",
        "explicitly_unavailable",
        "proxy_raw_targets_forbidden",
        "hidden defender",
        "trusted proxy targets are unavailable",
        "unavailable_information",
    )
    if any(marker in window for marker in prohibition_markers):
        return "C_prohibition_wording"
    # Field-name style keys without values.
    if f'"{needle}"' in window or f"'{needle}'" in window:
        if any(v in window for v in (": true", ": false", ": 0", ": 1", " = ")):
            return "A_hidden_value"
        return "B_hidden_field_name"
    return "B_hidden_field_name"


def scan_attacker_visible_hidden_mentions(
    text: str,
) -> list[dict[str, str]]:
    """List named-term mentions with coarse context classes for development audits."""
    findings: list[dict[str, str]] = []
    for term in sorted(_NAMED_HIDDEN_TERMS):
        classification = classify_attacker_visible_term_context(text, term)
        if classification == "D_not_found":
            continue
        findings.append({"term": term, "class": classification})
    return findings


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


def parse_a1_v4_1_plan(
    text: str,
    *,
    q_max: int,
    allowed_static_plan_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, str]:
    """Parse the initial V4.1 planning response (selections object per candidate)."""
    payload = _extract_json_object(text)
    if not isinstance(payload, Mapping):
        return None, "parse_error"
    forbidden = _reject_forbidden_keys(payload)
    if forbidden:
        return None, forbidden
    static_plan_id = payload.get("static_plan_id")
    if not isinstance(static_plan_id, str) or not static_plan_id.strip():
        return None, "missing_static_plan_id"
    if static_plan_id not in set(allowed_static_plan_ids):
        return None, "unknown_static_plan_id"
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
        extra = set(item.keys()) - {"strategy_label", "selections"}
        if extra:
            return None, "schema_error"
        parsed.append(
            {
                "strategy_label": label.strip(),
                "selections": selections,
            }
        )
    return {"static_plan_id": static_plan_id, "candidates": parsed}, "ok"


def parse_a1_v4_1_slot_replacements(
    text: str,
    *,
    requested_indices: Sequence[int],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Parse V4.1 slot-repair replacements (selections only; static plan pinned)."""
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
    "ActionSlot",
    "ActionSlotCatalog",
    "DIVERSIFICATION_PRINCIPLE_V4_1",
    "GENERIC_UNAVAILABLE_NOTICE",
    "PROMPT_VERSION_V4_1",
    "assert_v4_1_prompt_hard_contract",
    "build_v4_1_action_slots",
    "build_v4_1_prompt_payload",
    "build_v4_choice_catalog",
    "build_v4_static_plan_options",
    "classify_attacker_visible_term_context",
    "parse_a1_v4_1_plan",
    "parse_a1_v4_1_slot_replacements",
    "find_hidden_term_span",
    "prompt_contains_hidden_term",
    "resolve_action_slot_selections",
    "scan_attacker_visible_hidden_mentions",
    "static_locks_and_cost",
    "static_plan_by_id",
]

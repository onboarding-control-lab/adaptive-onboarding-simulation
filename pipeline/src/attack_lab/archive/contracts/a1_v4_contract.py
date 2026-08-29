"""A1 V4 hard-contract helpers: opaque choice IDs + explicit static plans.

Historical V1–V3 builders remain in ``a1_planner``.  V4 removes mechanical
rule-following from the LLM: trusted code enumerates legal choices and
feasible episode-static plans; the LLM selects only opaque IDs.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from attack_lab.archive.contracts.a1_v3_contract import PROXY_RAW_FEATURE_NAMES
from attack_lab.budget import AttackBudget, compute_edit_metrics
from attack_lab.candidate_identity import canonical_candidate_fingerprint
from attack_lab.governance_view import GovernanceView
from attack_lab.reference_actions import (
    ReferenceSelection,
    reference_backed_selections_for_action,
)
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import AttackProposal, to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_V4 = "a1_oneshot_v4_hard_contract"

DIVERSIFICATION_PRINCIPLE_V4 = (
    "Because the full plan must be created before any feedback is available, "
    "select one feasible static_plan_id and construct a diversified ordered "
    "portfolio of exactly q_max query candidates using only allowed "
    "choice_ids for that plan. Do not invent action keys, reference ids, or "
    "raw values. Do not infer or target any hidden model score, threshold, "
    "or decision boundary."
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
    }
)


@dataclass(frozen=True, slots=True)
class LegalChoice:
    """One opaque, code-generated legal reference-backed action selection."""

    choice_id: str
    action_key: str
    reference_id: str
    category: str  # "episode_static" | "per_attempt"

    def to_public_dict(self) -> dict[str, Any]:
        """Attacker-visible metadata (no proxy raw values)."""
        return {
            "choice_id": self.choice_id,
            "action_key": self.action_key,
            "reference_id": self.reference_id,
            "category": self.category,
        }

    def as_selection(self) -> ReferenceSelection:
        return ReferenceSelection(reference_id=self.reference_id)


@dataclass(frozen=True, slots=True)
class StaticPlanOption:
    """One feasible episode-static lock plan with residual query budget."""

    static_plan_id: str
    static_choice_ids: tuple[str, ...]
    static_edit_cost: int
    residual_m: int
    allowed_query_choice_ids: tuple[str, ...]
    n_distinct_feasible_candidates: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "static_plan_id": self.static_plan_id,
            "static_choice_ids": list(self.static_choice_ids),
            "static_edit_cost": int(self.static_edit_cost),
            "residual_m": int(self.residual_m),
            "allowed_query_choice_ids": list(self.allowed_query_choice_ids),
            "n_distinct_feasible_candidates": int(self.n_distinct_feasible_candidates),
        }


@dataclass(frozen=True, slots=True)
class V4ChoiceCatalog:
    """Episode-scoped authoritative choice catalogue."""

    choices_by_id: Mapping[str, LegalChoice]
    static_choice_ids: tuple[str, ...]
    per_attempt_choice_ids: tuple[str, ...]

    def get(self, choice_id: str) -> LegalChoice | None:
        return self.choices_by_id.get(str(choice_id))

    def public_choices(self, choice_ids: Sequence[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for choice_id in choice_ids:
            choice = self.get(choice_id)
            if choice is not None:
                out.append(choice.to_public_dict())
        return out


def _values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        import pandas as pd

        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def _is_attacker_facing_catalogue_action(
    action: str,
    rule: Any,
) -> bool:
    """Return True iff ``action`` may appear in A1/A3 opaque choice catalogues.

    Raw trusted proxy *feature* names are never catalogue keys. Governance-
    approved abstract ``proxy_action_key`` values are allowed; trusted
    resolution maps them to the underlying K-pool raw feature.
    """
    if action in PROXY_RAW_FEATURE_NAMES:
        return False
    if rule is None or not rule.is_mutable:
        return False
    if rule.agent_action_mode == "proxy_action":
        return (
            rule.proxy_action_key is not None
            and action == rule.proxy_action_key
        )
    # Non-proxy actions: never surface a trusted proxy raw feature name.
    return rule.feature not in PROXY_RAW_FEATURE_NAMES


def _enabled_actions_by_category(
    validator: ConstraintValidator,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    static: list[str] = []
    per_attempt: list[str] = []
    for action in validator.enabled_action_keys:
        rule = validator.policy.field_for_action(action)
        if not _is_attacker_facing_catalogue_action(action, rule):
            continue
        assert rule is not None  # narrowed by helper
        if rule.is_episode_locked:
            static.append(action)
        else:
            per_attempt.append(action)
    return tuple(static), tuple(per_attempt)


def build_v4_choice_catalog(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    anchor: Mapping[str, Any],
) -> V4ChoiceCatalog:
    """Enumerate opaque choice IDs for all legal reference-backed selections."""
    static_actions, per_attempt_actions = _enabled_actions_by_category(validator)
    ordered: list[LegalChoice] = []
    counter = 0

    def _add(action_key: str, category: str) -> None:
        nonlocal counter
        rule = validator.policy.field_for_action(action_key)
        if rule is None:
            return
        selections = reference_backed_selections_for_action(
            action_key=action_key,
            pool=pool,
            rule=rule,
            anchor_value=anchor.get(rule.feature),
            require_change_from_anchor=True,
        )
        for selection in selections:
            counter += 1
            ordered.append(
                LegalChoice(
                    choice_id=f"choice_{counter:03d}",
                    action_key=action_key,
                    reference_id=selection.reference_id,
                    category=category,
                )
            )

    for action in static_actions:
        _add(action, "episode_static")
    for action in per_attempt_actions:
        _add(action, "per_attempt")

    by_id = {item.choice_id: item for item in ordered}
    return V4ChoiceCatalog(
        choices_by_id=by_id,
        static_choice_ids=tuple(
            item.choice_id for item in ordered if item.category == "episode_static"
        ),
        per_attempt_choice_ids=tuple(
            item.choice_id for item in ordered if item.category == "per_attempt"
        ),
    )


def resolve_choice_ids_to_changes(
    choice_ids: Sequence[str],
    catalog: V4ChoiceCatalog,
) -> tuple[dict[str, ReferenceSelection] | None, str]:
    """Resolve opaque choice IDs to ReferenceSelection changes."""
    changes: dict[str, ReferenceSelection] = {}
    seen_actions: set[str] = set()
    for raw_id in choice_ids:
        choice = catalog.get(str(raw_id))
        if choice is None:
            return None, "unknown_choice_id"
        if choice.action_key in seen_actions:
            return None, "duplicate_action_in_slot"
        seen_actions.add(choice.action_key)
        changes[choice.action_key] = choice.as_selection()
    if not changes:
        return None, "empty_choice_ids"
    return changes, ""


def static_locks_and_cost(
    *,
    validator: ConstraintValidator,
    anchor: Mapping[str, Any],
    catalog: V4ChoiceCatalog,
    static_choice_ids: Sequence[str],
    m_max: int,
) -> tuple[dict[str, Any] | None, int, str]:
    """Build episode locks and static edit cost for a static-choice bundle."""
    changes, reason = resolve_choice_ids_to_changes(static_choice_ids, catalog)
    if static_choice_ids and changes is None:
        return None, 0, reason or "invalid_static_choices"
    proposal = AttackProposal(changes=dict(changes or {}))
    preparation = validator.prepare_episode_locks(anchor, proposal)
    if preparation.errors:
        return None, 0, "static_lock_preparation_failed"
    projected = validator.project_for_billing(
        anchor, proposal, locked_values=preparation.locked_values
    )
    _, distance, _, _ = compute_edit_metrics(
        anchor=anchor,
        candidate=projected,
        mutable_feature_names=validator.mutable_feature_names(),
        previous_candidate=None,
    )
    cost = int(distance)
    if cost > int(m_max):
        return None, cost, "static_budget_exceeded"
    if changes:
        validity = validator.validate(
            anchor,
            proposal,
            locked_values=preparation.locked_values,
            pre_feedback_errors=preparation.errors,
        )
        if not validity.is_valid:
            return None, cost, "static_constraint_failed"
    return dict(preparation.locked_values), cost, ""


def count_feasible_query_candidates(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    anchor: Mapping[str, Any],
    catalog: V4ChoiceCatalog,
    query_choice_ids: Sequence[str],
    locked_values: Mapping[str, Any],
    residual_m: int,
    q_max: int,
    enumerate_cap: int = 400,
) -> int:
    """Count distinct legal fingerprints under residual_m (capped scan)."""
    if residual_m < 1:
        return 0
    choices = [catalog.get(cid) for cid in query_choice_ids]
    choices = [item for item in choices if item is not None]
    if not choices:
        return 0
    fingerprints: set[str] = set()
    scanned = 0
    static_diff = sum(
        1
        for name, value in locked_values.items()
        if name in validator.mutable_feature_names()
        and not _values_equal(value, anchor.get(name))
    )
    for width in range(1, int(residual_m) + 1):
        for combo in itertools.combinations(choices, width):
            action_keys = [item.action_key for item in combo]
            if len(set(action_keys)) != len(action_keys):
                continue
            scanned += 1
            if scanned > enumerate_cap:
                return len(fingerprints)
            changes = {item.action_key: item.as_selection() for item in combo}
            proposal = AttackProposal(changes=changes)
            projected = validator.project_for_billing(
                anchor, proposal, locked_values=locked_values
            )
            _, distance, _, _ = compute_edit_metrics(
                anchor=anchor,
                candidate=projected,
                mutable_feature_names=validator.mutable_feature_names(),
                previous_candidate=None,
            )
            query_cost = int(distance) - int(static_diff)
            if query_cost < 1 or query_cost > int(residual_m):
                continue
            if int(distance) > int(static_diff) + int(residual_m):
                continue
            validity = validator.validate(
                anchor,
                proposal,
                locked_values=locked_values,
                pre_feedback_errors=(),
            )
            if not validity.is_valid:
                continue
            action_fields = [
                name for name in pool.action_fields if name in projected
            ]
            fp = canonical_candidate_fingerprint(
                anchor_id="feasibility",
                projected_candidate=projected,
                action_fields=action_fields,
            )
            fingerprints.add(fp)
            if len(fingerprints) >= int(q_max):
                return len(fingerprints)
    return len(fingerprints)


def build_v4_static_plan_options(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    anchor: Mapping[str, Any],
    catalog: V4ChoiceCatalog,
    m_max: int,
    q_max: int,
) -> tuple[StaticPlanOption, ...]:
    """Enumerate mechanically feasible static plans (no D1 ranking)."""
    plans: list[StaticPlanOption] = []
    by_action: dict[str, list[str]] = {}
    for choice_id in catalog.static_choice_ids:
        choice = catalog.get(choice_id)
        if choice is None:
            continue
        by_action.setdefault(choice.action_key, []).append(choice_id)

    action_keys = sorted(by_action)
    bundles: list[tuple[str, ...]] = [()]
    for action in action_keys:
        for choice_id in by_action[action]:
            bundles.append((choice_id,))
    if int(m_max) >= 2:
        for a1, a2 in itertools.combinations(action_keys, 2):
            for c1 in by_action[a1]:
                for c2 in by_action[a2]:
                    bundles.append((c1, c2))

    seen_signatures: set[tuple[str, ...]] = set()
    plan_idx = 0
    for bundle in bundles:
        signature = tuple(sorted(bundle))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        locks, cost, _reason = static_locks_and_cost(
            validator=validator,
            anchor=anchor,
            catalog=catalog,
            static_choice_ids=bundle,
            m_max=m_max,
        )
        if locks is None:
            continue
        residual = int(m_max) - int(cost)
        if residual < 0:
            continue
        if residual < 1 and int(q_max) > 1:
            continue
        if residual < 1 and int(q_max) == 1:
            if cost < 1:
                continue
            n_feasible = 1
            allowed_query: tuple[str, ...] = ()
        else:
            allowed_query = tuple(catalog.per_attempt_choice_ids)
            n_feasible = count_feasible_query_candidates(
                validator=validator,
                pool=pool,
                anchor=anchor,
                catalog=catalog,
                query_choice_ids=allowed_query,
                locked_values=locks,
                residual_m=residual,
                q_max=q_max,
            )
            if n_feasible < int(q_max):
                continue
        plan_idx += 1
        plans.append(
            StaticPlanOption(
                static_plan_id=f"static_plan_{plan_idx:02d}",
                static_choice_ids=tuple(bundle),
                static_edit_cost=int(cost),
                residual_m=int(residual),
                allowed_query_choice_ids=allowed_query,
                n_distinct_feasible_candidates=int(n_feasible),
            )
        )
    return tuple(plans)


def static_plan_by_id(
    plans: Sequence[StaticPlanOption], static_plan_id: str
) -> StaticPlanOption | None:
    wanted = str(static_plan_id)
    for plan in plans:
        if plan.static_plan_id == wanted:
            return plan
    return None


def build_v4_prompt_payload(
    *,
    validator: ConstraintValidator,
    pool: ReferencePool,
    budget: AttackBudget,
    q_max: int,
    visible_anchor: Mapping[str, Any],
    case_id: str,
    catalog: V4ChoiceCatalog,
    static_plans: Sequence[StaticPlanOption],
) -> dict[str, Any]:
    """Attacker-public V4 planning payload (choice IDs + static plans only)."""
    view = GovernanceView.from_policy(
        validator.policy,
        budget=budget,
        read_only_context_fields=pool.read_only_context_fields,
        enabled_action_keys=validator.enabled_action_keys,
    )
    _ = view  # ensures governance view construction remains fail-closed
    safe_visible = {
        str(key): value
        for key, value in dict(visible_anchor).items()
        if str(key) not in PROXY_RAW_FEATURE_NAMES
    }
    payload = {
        "task": (
            "Select one feasible static_plan_id and plan an ordered sequence of "
            "exactly q_max unique query candidates using only choice_ids. "
            "Freeze the full sequence now; no later revision is allowed."
        ),
        "prompt_version": PROMPT_VERSION_V4,
        "budget": {
            "q_max": int(q_max),
            "m_max": int(budget.m_max),
            "notes": [
                f"Return exactly {int(q_max)} candidates (q_max={int(q_max)}).",
                "Select exactly one static_plan_id from static_plan_options.",
                "Each candidate may include between 1 and residual_m choice_ids "
                "from that plan's allowed_query_choice_ids.",
                "Output only static_plan_id, strategy_label, and choice_ids — "
                "never action_key, reference_id, feature names, or raw values.",
                "Candidates must be unique.",
                "Local slot repair may replace only invalid candidate indices "
                "before any defender query; it does not consume Q.",
            ],
        },
        "anchor": {
            "case_id": case_id,
            "visible_fields": to_jsonable(safe_visible),
        },
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
                        "required": ["strategy_label", "choice_ids"],
                        "additionalProperties": False,
                        "properties": {
                            "strategy_label": {"type": "string"},
                            "choice_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                    },
                },
            },
        },
        "planning_principle": DIVERSIFICATION_PRINCIPLE_V4,
        "hard_contract": {
            "output_may_contain_only": [
                "static_plan_id",
                "candidates",
                "strategy_label",
                "choice_ids",
            ],
            "output_must_not_contain": sorted(_FORBIDDEN_OUTPUT_KEYS),
            "proxy_raw_targets_forbidden": sorted(PROXY_RAW_FEATURE_NAMES),
        },
        "explicitly_unavailable": [
            "d1_risk_score",
            "d1_threshold",
            "feature_importance_or_shap",
            "gradients",
            "true_rejection_reason",
            "fraud_bool",
        ],
    }
    assert_v4_prompt_hard_contract(payload)
    return payload


def assert_v4_prompt_hard_contract(payload: Mapping[str, Any]) -> None:
    """Fail closed if proxy raw names appear as writable attacker vocabulary."""
    for item in payload.get("choice_catalogue") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4 choice_catalogue exposes a proxy raw action_key.")
        if str(item.get("reference_id")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V4 choice_catalogue exposes a proxy raw reference token.")
    visible = ((payload.get("anchor") or {}).get("visible_fields") or {})
    overlap = sorted(set(visible).intersection(PROXY_RAW_FEATURE_NAMES))
    if overlap:
        raise ValueError(f"V4 visible_fields expose proxy raw names: {overlap}.")
    for plan in payload.get("static_plan_options") or []:
        if not isinstance(plan, Mapping):
            continue
        if "ranking" in plan or "expected_asr" in plan or "defender" in plan:
            raise ValueError("V4 static plans must not expose defender ranking.")
    # Explicit denylist may name proxy raw targets; that is documentation only.
    hard = payload.get("hard_contract") or {}
    text_without_denylist = json.dumps(
        to_jsonable(
            {
                key: value
                for key, value in dict(payload).items()
                if key != "hard_contract"
            }
        ),
        sort_keys=True,
    )
    for name in PROXY_RAW_FEATURE_NAMES:
        if name in text_without_denylist:
            raise ValueError(
                f"V4 prompt unexpectedly contains proxy raw name {name!r}."
            )
    _ = hard


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


def parse_a1_v4_plan(
    text: str,
    *,
    q_max: int,
    allowed_static_plan_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, str]:
    """Parse the initial V4 planning response."""
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
        if "changes" in item or "action_key" in item or "reference_id" in item:
            return None, "forbidden_output_key"
        label = item.get("strategy_label")
        choice_ids = item.get("choice_ids")
        if not isinstance(label, str) or not label.strip():
            return None, "missing_strategy_label"
        if not isinstance(choice_ids, list) or not choice_ids:
            return None, "empty_choice_ids"
        if not all(isinstance(cid, str) and cid.strip() for cid in choice_ids):
            return None, "invalid_choice_id_type"
        extra = set(item.keys()) - {"strategy_label", "choice_ids"}
        if extra:
            return None, "schema_error"
        parsed.append(
            {
                "strategy_label": label.strip(),
                "choice_ids": [str(cid).strip() for cid in choice_ids],
            }
        )
    return {"static_plan_id": static_plan_id, "candidates": parsed}, "ok"


def parse_a1_v4_slot_replacements(
    text: str,
    *,
    requested_indices: Sequence[int],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Parse V4 slot-repair replacements (choice_ids only; static plan pinned)."""
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
        if "changes" in item or "action_key" in item or "reference_id" in item:
            return None, "forbidden_output_key"
        try:
            index = int(item.get("candidate_index"))
        except (TypeError, ValueError):
            return None, "invalid_candidate_index"
        if index not in requested or index in seen:
            return None, "replacement_indices_invalid"
        label = item.get("strategy_label")
        choice_ids = item.get("choice_ids")
        if not isinstance(label, str) or not label.strip():
            return None, "missing_strategy_label"
        if not isinstance(choice_ids, list) or not choice_ids:
            return None, "empty_choice_ids"
        if not all(isinstance(cid, str) and cid.strip() for cid in choice_ids):
            return None, "invalid_choice_id_type"
        extra = set(item.keys()) - {"candidate_index", "strategy_label", "choice_ids"}
        if extra:
            return None, "schema_error"
        seen.add(index)
        parsed.append(
            {
                "candidate_index": index,
                "strategy_label": label.strip(),
                "choice_ids": [str(cid).strip() for cid in choice_ids],
            }
        )
    if seen != requested:
        return None, "replacement_indices_invalid"
    return parsed, "ok"


__all__ = [
    "DIVERSIFICATION_PRINCIPLE_V4",
    "LegalChoice",
    "PROMPT_VERSION_V4",
    "PROXY_RAW_FEATURE_NAMES",
    "StaticPlanOption",
    "V4ChoiceCatalog",
    "assert_v4_prompt_hard_contract",
    "build_v4_choice_catalog",
    "build_v4_prompt_payload",
    "build_v4_static_plan_options",
    "count_feasible_query_candidates",
    "parse_a1_v4_plan",
    "parse_a1_v4_slot_replacements",
    "resolve_choice_ids_to_changes",
    "static_locks_and_cost",
    "static_plan_by_id",
    "_enabled_actions_by_category",
    "_is_attacker_facing_catalogue_action",
]

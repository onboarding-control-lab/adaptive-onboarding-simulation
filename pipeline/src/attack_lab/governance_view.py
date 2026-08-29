"""Attacker-public governance view (game rules, not model internals)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from attack_lab.budget import AttackBudget
from attack_lab.governance import CompiledFieldPolicy, CompiledGovernancePolicy
from attack_lab.types import PublicLabel, to_jsonable

FieldCategory = Literal["per_attempt", "episode_static", "forbidden", "not_applicable"]


@dataclass(frozen=True)
class ActionFieldPublicRule:
    """Public rule card for one attacker-mutable action."""

    feature: str
    action_key: str
    category: Literal["per_attempt", "episode_static"]
    data_type: str
    domain_mode: str
    sampling_kind: str
    lower_bound: float | None
    upper_bound: float | None
    allowed_values: tuple[Any, ...]
    observed_support: tuple[Any, ...]
    proxy_action_key: str | None
    proxy_actions: tuple[str, ...]
    sentinel_spec: Mapping[str, Any]
    sentinel_policy: str
    hard_constraints: tuple[Mapping[str, Any], ...]
    counts_toward_edit_budget: bool
    edit_budget_unit: int = 1

    def to_public_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "feature": self.feature,
                "action_key": self.action_key,
                "category": self.category,
                "data_type": self.data_type,
                "domain_mode": self.domain_mode,
                "sampling_kind": self.sampling_kind,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "allowed_values": list(self.allowed_values),
                "observed_support": list(self.observed_support),
                "proxy_action_key": self.proxy_action_key,
                "proxy_actions": list(self.proxy_actions),
                "sentinel_spec": dict(self.sentinel_spec),
                "sentinel_policy": self.sentinel_policy,
                "hard_constraints": [dict(item) for item in self.hard_constraints],
                "counts_toward_edit_budget": self.counts_toward_edit_budget,
                "edit_budget_unit": self.edit_budget_unit,
            }
        )


@dataclass(frozen=True)
class SubmissionHistoryItem:
    """Attacker-visible history of one prior submission."""

    attempt: int
    changes: Mapping[str, Any]
    label: PublicLabel
    candidate_hash: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "changes": to_jsonable(dict(self.changes)),
            "label": self.label,
            "candidate_hash": self.candidate_hash,
        }


@dataclass
class GovernanceView:
    """Read-only public game rules + current episode state for A2.

    Explicitly excludes D1 scores, thresholds, SHAP, model parameters,
    fraud_bool, month-7 data and cross-anchor history.
    """

    policy_version: str
    policy_fingerprint: str
    per_attempt_fields: tuple[str, ...]
    episode_static_fields: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    not_applicable_fields: tuple[str, ...]
    read_only_context_fields: tuple[str, ...]
    action_field_rules: tuple[ActionFieldPublicRule, ...]
    budget: AttackBudget
    attempt_number: int = 1
    queries_used: int = 0
    static_locked: bool = False
    locked_static_values: dict[str, Any] = field(default_factory=dict)
    locked_edit_count: int = 0
    remaining_dynamic_slots: int = 0
    submission_history: tuple[SubmissionHistoryItem, ...] = ()

    @classmethod
    def from_policy(
        cls,
        policy: CompiledGovernancePolicy,
        *,
        budget: AttackBudget,
        read_only_context_fields: Sequence[str],
        enabled_action_keys: Sequence[str] | None = None,
    ) -> "GovernanceView":
        enabled = (
            None
            if enabled_action_keys is None
            else set(enabled_action_keys)
        )
        rules: list[ActionFieldPublicRule] = []
        for action_key in policy.available_action_keys:
            if enabled is not None and action_key not in enabled:
                continue
            rule = policy.field_for_action(action_key)
            if rule is None or not rule.is_mutable:
                continue
            rules.append(_public_rule(action_key, rule))
        return cls(
            policy_version=policy.policy_version,
            policy_fingerprint=policy.policy_fingerprint,
            per_attempt_fields=tuple(policy.per_attempt_fields),
            episode_static_fields=tuple(policy.episode_static_fields),
            forbidden_fields=tuple(policy.forbidden_fields),
            not_applicable_fields=tuple(policy.not_applicable_fields),
            read_only_context_fields=tuple(read_only_context_fields),
            action_field_rules=tuple(rules),
            budget=budget,
            remaining_dynamic_slots=int(budget.m_max),
        )

    def update_episode_state(
        self,
        *,
        attempt_number: int,
        queries_used: int,
        static_locked: bool,
        locked_static_values: Mapping[str, Any],
        locked_edit_count: int,
        remaining_dynamic_slots: int,
        submission_history: Sequence[SubmissionHistoryItem],
    ) -> None:
        self.attempt_number = int(attempt_number)
        self.queries_used = int(queries_used)
        self.static_locked = bool(static_locked)
        self.locked_static_values = dict(locked_static_values)
        self.locked_edit_count = int(locked_edit_count)
        self.remaining_dynamic_slots = int(remaining_dynamic_slots)
        self.submission_history = tuple(submission_history)

    def clear_episode_memory(self) -> None:
        """Episode end: drop failure history and static plan (no cross-anchor)."""
        self.attempt_number = 1
        self.queries_used = 0
        self.static_locked = False
        self.locked_static_values = {}
        self.locked_edit_count = 0
        self.remaining_dynamic_slots = int(self.budget.m_max)
        self.submission_history = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "policy_fingerprint": self.policy_fingerprint,
            "field_roles": {
                "per_attempt_fields": list(self.per_attempt_fields),
                "episode_static_fields": list(self.episode_static_fields),
                "forbidden_fields": list(self.forbidden_fields),
                "not_applicable_fields": list(self.not_applicable_fields),
                "read_only_context_fields": list(self.read_only_context_fields),
            },
            "action_field_rules": [rule.to_public_dict() for rule in self.action_field_rules],
            "budget": self.budget.to_dict(),
            "episode_state": {
                "m_max": self.budget.m_max,
                "q_max": self.budget.q_max,
                "attempt_number": self.attempt_number,
                "queries_used": self.queries_used,
                "static_locked": self.static_locked,
                "locked_static_values": to_jsonable(self.locked_static_values),
                "locked_edit_count": self.locked_edit_count,
                "remaining_dynamic_slots": self.remaining_dynamic_slots,
                "submission_history": [
                    item.to_public_dict() for item in self.submission_history
                ],
            },
            "explicitly_hidden": [
                "d1_risk_score",
                "d1_threshold",
                "xgboost_structure_or_parameters",
                "feature_importance_or_shap",
                "field_causing_block",
                "true_rejection_reason",
                "fraud_bool",
                "month7_data",
                "other_anchor_attack_history",
            ],
        }


def _public_rule(action_key: str, rule: CompiledFieldPolicy) -> ActionFieldPublicRule:
    category: Literal["per_attempt", "episode_static"] = (
        "episode_static" if rule.is_episode_locked else "per_attempt"
    )
    proxy_names = tuple(rule.resolved_proxy_actions.keys())
    return ActionFieldPublicRule(
        feature=rule.feature,
        action_key=action_key,
        category=category,
        data_type=rule.data_type,
        domain_mode=rule.domain_mode,
        sampling_kind=rule.sampling_kind,
        lower_bound=rule.lower_bound,
        upper_bound=rule.upper_bound,
        allowed_values=tuple(rule.allowed_values),
        observed_support=tuple(rule.observed_support),
        proxy_action_key=rule.proxy_action_key,
        proxy_actions=proxy_names,
        sentinel_spec=dict(rule.sentinel_spec),
        sentinel_policy=rule.sentinel_policy,
        hard_constraints=tuple(dict(item) for item in rule.hard_constraints),
        counts_toward_edit_budget=True,
        edit_budget_unit=1,
    )


__all__ = [
    "ActionFieldPublicRule",
    "GovernanceView",
    "SubmissionHistoryItem",
]

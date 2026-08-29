"""A2 — constrained surrogate-guided sequential best-first search.

Mechanism-verification attacker:

- label-only / query-limited / black-box / non-LLM;
- ranks candidates by reference-pool Gower coherence + failure diversification;
- enumerates governance-legal unique remainders (no random-empty false stops);
- reads ``AttackBudget`` for all Q/m decisions (never hard-codes m or Q).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TextIO

from attack_lab.attackers.a0_random import derive_episode_seed
from attack_lab.budget import AttackBudget, compute_edit_metrics
from attack_lab.candidate_identity import canonical_candidate_fingerprint
from attack_lab.constraint_profile import IdentityCompositionProfile
from attack_lab.environment import AttackEnvironment
from attack_lab.governance import CompiledGovernancePolicy
from attack_lab.governance_view import GovernanceView, SubmissionHistoryItem
from attack_lab.gower import (
    GowerFieldSpec,
    build_gower_field_specs,
    mean_gower_to_profiles,
    min_gower_to_set,
)
from attack_lab.public_reference_view import public_safe_gower_field_names
from attack_lab.reference_actions import (
    ReferenceSelection,
    audit_reference_provenance,
    reference_backed_selections_for_action,
    reference_ids_from_changes,
    resolve_reference_selection,
    values_exactly_equal,
)
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import AttackProposal, PublicLabel, to_jsonable
from attack_lab.validator import ConstraintValidator


class A2SearchError(RuntimeError):
    """Raised when A2 cannot operate under the supplied public interfaces."""


# Historical default: Gower over all pool action_fields (incl. proxy raws).
GOWER_POLICY_LEGACY_V1 = "a2_legacy_full_action_gower_v1"
# Corrected public-reference Gower: public-safe fields ∩ action_fields.
GOWER_POLICY_PUBLIC_REFERENCE_V2 = "a2_public_reference_gower_v2"
SUPPORTED_GOWER_POLICIES = frozenset(
    {GOWER_POLICY_LEGACY_V1, GOWER_POLICY_PUBLIC_REFERENCE_V2}
)


@dataclass(frozen=True)
class _LegalCandidate:
    changes: dict[str, Any]
    fingerprint: str
    projected: dict[str, Any]
    distance: int
    locked_edit_count: int
    dynamic_edit_count: int
    remaining_dynamic_slots: int
    edited_fields: tuple[str, ...]
    static_plan: dict[str, Any]


@dataclass
class SurrogateGuidedSearcher:
    """Official A2 mechanism-verification attacker."""

    budget: AttackBudget
    reference_pool: ReferencePool
    experiment_seed: int
    attacker_id: str = "a2"
    stdout: TextIO | None = None
    #: Optional layered eligibility filter shared with A0/environment.
    constraint_profile: IdentityCompositionProfile | None = None
    #: Versioned Gower field policy. Legacy default preserves historical A2.
    gower_policy: str = GOWER_POLICY_LEGACY_V1
    _view: GovernanceView | None = field(default=None, init=False, repr=False)
    _episode_seed: int | None = field(default=None, init=False, repr=False)
    _gower_specs: tuple[GowerFieldSpec, ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _history: list[SubmissionHistoryItem] = field(
        default_factory=list, init=False, repr=False
    )
    _failed_projected: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _seen_fingerprints: set[str] = field(default_factory=set, init=False, repr=False)
    _locked_static_values: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _static_locked: bool = field(default=False, init=False, repr=False)
    _locked_edit_count: int = field(default=0, init=False, repr=False)
    _submission_logs: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _anchor_features: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.reference_pool.K < 1:
            raise A2SearchError("reference_pool.K must be >= 1.")
        if self.budget.m_max < 0 or self.budget.q_max < 1:
            raise A2SearchError("AttackBudget must expose valid q_max and m_max.")
        if self.gower_policy not in SUPPORTED_GOWER_POLICIES:
            raise A2SearchError(
                f"Unsupported gower_policy={self.gower_policy!r}; "
                f"supported={sorted(SUPPORTED_GOWER_POLICIES)}."
            )

    @property
    def governance_view(self) -> GovernanceView | None:
        return self._view

    @property
    def gower_field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._gower_specs)

    @property
    def proxy_raw_geometry_access(self) -> bool:
        from attack_lab.public_reference_view import TRUSTED_PROXY_RAW_TARGETS

        return bool(set(self.gower_field_names) & set(TRUSTED_PROXY_RAW_TARGETS))

    @property
    def submission_logs(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._submission_logs)

    def run(self, env: AttackEnvironment) -> None:
        """Drive online adaptive search until PASS, Q, or action-space exhaustion."""
        self._reset_episode_state(env)
        self._write(
            f"\n=== A2 SurrogateGuidedSearcher "
            f"(experiment_seed={self.experiment_seed}, "
            f"episode_seed={self._episode_seed}, "
            f"case={env.starting_case.case_id}, "
            f"Q={self.budget.q_max}, m={self.budget.m_max}, "
            f"gower_policy={self.gower_policy}, "
            f"pool_fp={self.reference_pool.pool_fingerprint[:12]}) ===\n"
            "Label-only adaptive best-first search with failure diversification; "
            "enumeration remainder required before no_feasible/exhaustion.\n"
        )

        while not env.done:
            if env.ledger.q_remaining < 1:
                env.abort(reason="q_exhausted")
                break

            self._refresh_view(env)
            remaining = self._enumerate_legal_unique(env)
            if not remaining:
                self._write(
                    "Local stop: action_space_exhaustion "
                    "(legal_unique_remaining_candidates=0).\n"
                )
                env.abort(reason="action_space_exhaustion")
                break

            ranked = self._rank_candidates(remaining)
            chosen, rank_index = ranked[0]
            tradeoff_note = (
                f"static_cost={chosen.locked_edit_count}, "
                f"dynamic_slots_remaining={chosen.remaining_dynamic_slots}, "
                f"static_tradeoff="
                f"{'full_m_static_unique_only' if chosen.remaining_dynamic_slots == 0 else 'keeps_dynamic_room'}"
            )
            self._write(
                f"attempt={env.attempts_used + 1}: selecting rank={rank_index} "
                f"hash={chosen.fingerprint[:12]} dist={chosen.distance} "
                f"{tradeoff_note}\n"
            )

            proposal = self._to_proposal(chosen, rank_index=rank_index, remaining_n=len(remaining))
            step = env.step(proposal)
            label: PublicLabel = step.public_feedback.label
            self._record_submission(
                env=env,
                chosen=chosen,
                rank_index=rank_index,
                remaining_n=len(remaining),
                label=label,
                tradeoff_note=tradeoff_note,
            )

            if not self._static_locked:
                self._locked_static_values = dict(env.locked_static_values)
                self._static_locked = True
                self._locked_edit_count = chosen.locked_edit_count

            self._seen_fingerprints.add(chosen.fingerprint)
            self._history.append(
                SubmissionHistoryItem(
                    attempt=step.attempt,
                    changes=dict(chosen.changes),
                    label=label,
                    candidate_hash=chosen.fingerprint,
                )
            )
            if label in {"BLOCK", "INVALID"}:
                self._failed_projected.append(dict(chosen.projected))

            if label == "PASS" or env.done:
                break

        assert self._view is not None
        self._view.clear_episode_memory()
        self._history.clear()
        self._failed_projected.clear()
        self._seen_fingerprints.clear()
        self._locked_static_values = {}
        self._static_locked = False
        self._locked_edit_count = 0
        self._write(
            f"Episode stop observed (success={env.success}); "
            "episode memory cleared (no cross-anchor learning).\n"
        )

    def _reset_episode_state(self, env: AttackEnvironment) -> None:
        if int(env.budget.q_max) != int(self.budget.q_max):
            raise A2SearchError(
                "Environment q_max does not match AttackBudget.q_max; "
                "refuse to run with inconsistent budgets."
            )
        if int(env.budget.m_max) != int(self.budget.m_max):
            raise A2SearchError(
                "Environment m_max does not match AttackBudget.m_max; "
                "refuse to run with inconsistent budgets."
            )
        self._episode_seed = derive_episode_seed(
            self.experiment_seed, env.starting_case.case_id, self.attacker_id
        )
        self._history = []
        self._failed_projected = []
        self._seen_fingerprints = set()
        self._locked_static_values = {}
        self._static_locked = False
        self._locked_edit_count = 0
        self._submission_logs = []
        self._view = GovernanceView.from_policy(
            env.validator.policy,
            budget=self.budget,
            read_only_context_fields=self.reference_pool.read_only_context_fields,
            enabled_action_keys=env.validator.enabled_action_keys,
        )
        self._gower_specs = self._build_gower_specs(env)
        self._refresh_view(env)
        self._anchor_features = dict(env.starting_case.features)

    def _build_gower_specs(self, env: AttackEnvironment) -> tuple[GowerFieldSpec, ...]:
        policy = env.validator.policy
        if self.gower_policy == GOWER_POLICY_PUBLIC_REFERENCE_V2:
            action_fields = [
                name
                for name in public_safe_gower_field_names(self.reference_pool)
                if name in env.starting_case.features
            ]
        else:
            # Historical A2: all pool action_fields present on the anchor.
            action_fields = [
                name
                for name in self.reference_pool.action_fields
                if name in env.starting_case.features
            ]
        data_types = {
            name: policy.fields[name].data_type
            for name in action_fields
            if name in policy.fields
        }
        bounds = {
            name: (policy.fields[name].lower_bound, policy.fields[name].upper_bound)
            for name in action_fields
            if name in policy.fields
        }
        profiles = [dict(profile.fields) for profile in self.reference_pool.profiles]
        return build_gower_field_specs(
            field_names=action_fields,
            data_types=data_types,
            bounds=bounds,
            profiles=profiles,
            anchor=env.starting_case.features,
        )

    def _refresh_view(self, env: AttackEnvironment) -> None:
        assert self._view is not None
        remaining_dynamic = max(0, int(self.budget.m_max) - int(self._locked_edit_count))
        self._view.update_episode_state(
            attempt_number=env.attempts_used + 1 if not env.done else env.attempts_used,
            queries_used=env.attempts_used,
            static_locked=self._static_locked,
            locked_static_values=self._locked_static_values,
            locked_edit_count=self._locked_edit_count,
            remaining_dynamic_slots=remaining_dynamic,
            submission_history=self._history,
        )

    def _action_domains(
        self, validator: ConstraintValidator
    ) -> dict[str, tuple[Any, ...]]:
        """Finite K-pool ReferenceSelection domains only (no governance sources)."""
        domains: dict[str, tuple[Any, ...]] = {}
        anchor = None  # filled per-call via validator callers when needed
        _ = anchor
        for action in validator.enabled_action_keys:
            rule = validator.policy.field_for_action(action)
            if rule is None or not rule.is_mutable:
                continue
            if rule.feature not in self.reference_pool.action_fields:
                domains[action] = ()
                continue
            domains[action] = reference_backed_selections_for_action(
                action_key=action,
                pool=self.reference_pool,
                rule=rule,
                anchor_value=None,
                require_change_from_anchor=False,
            )
        return domains

    def _enumerate_legal_unique(self, env: AttackEnvironment) -> list[_LegalCandidate]:
        validator = env.validator
        policy = validator.policy
        anchor = env.starting_case.features
        domains = self._action_domains(validator)
        static_actions = [
            action
            for action in validator.enabled_action_keys
            if (rule := policy.field_for_action(action)) is not None
            and rule.is_episode_locked
            and domains.get(action)
        ]
        free_actions = [
            action
            for action in validator.enabled_action_keys
            if (rule := policy.field_for_action(action)) is not None
            and rule.is_mutable
            and not rule.is_episode_locked
            and domains.get(action)
        ]
        if self.constraint_profile is not None:
            persona = set(self.constraint_profile.persona_profile_fields)
            contact = set(self.constraint_profile.contact_identity_fields)
            static_actions = [
                action
                for action in static_actions
                if (rule := policy.field_for_action(action)) is not None
                and rule.feature in persona
            ]
            free_actions = [
                action
                for action in free_actions
                if (rule := policy.field_for_action(action)) is not None
                and rule.feature in contact
            ]

        found: dict[str, _LegalCandidate] = {}

        if self._static_locked:
            # Static plan frozen: only vary free actions within remaining slots.
            free_budget = max(0, int(self.budget.m_max) - int(self._locked_edit_count))
            lock_actions = {
                action: self._locked_value_as_action(policy, action)
                for action in static_actions
                if self._locked_value_as_action(policy, action) is not None
            }
            # Always include lock-only if it differs from already-submitted set.
            self._consider_changes(
                env,
                {k: v for k, v in lock_actions.items() if v is not None},
                found,
            )
            if free_budget >= 1:
                for k in range(1, free_budget + 1):
                    if len(free_actions) < k:
                        continue
                    for chosen in itertools.combinations(free_actions, k):
                        for values in itertools.product(
                            *[domains[a] for a in chosen]
                        ):
                            changes = {
                                a: v for a, v in zip(chosen, values, strict=True)
                            }
                            # Include locked static action tokens when they differ.
                            for action, value in lock_actions.items():
                                if value is not None and not _values_equal(
                                    _resolve(policy, action, value, pool=self.reference_pool),
                                    anchor.get(_feature(policy, action)),
                                ):
                                    changes[action] = value
                            self._consider_changes(env, changes, found)
            return sorted(found.values(), key=lambda item: item.fingerprint)

        # Pre-lock: enumerate static plans × free edits with distance <= m.
        m_max = int(self.budget.m_max)
        plan_iter: list[dict[str, Any]] = []
        if self.constraint_profile is not None:
            # Identity-composition requires exactly one persona static edit.
            k_range = (1,) if static_actions else ()
        else:
            plan_iter.append({})
            k_range = range(1, min(len(static_actions), m_max) + 1)
        for k_static in k_range:
            if len(static_actions) < k_static:
                continue
            for chosen in itertools.combinations(static_actions, k_static):
                for values in itertools.product(*[domains[a] for a in chosen]):
                    plan_iter.append(
                        {a: v for a, v in zip(chosen, values, strict=True)}
                    )

        for static_plan in plan_iter:
            lock_cost = _static_edit_cost(policy, anchor, static_plan, pool=self.reference_pool)
            if lock_cost > m_max:
                continue
            free_budget = m_max - lock_cost
            # Static-only candidate.
            if lock_cost >= 1:
                self._consider_changes(env, dict(static_plan), found)
            if free_budget < 1:
                continue
            for k in range(1, free_budget + 1):
                if len(free_actions) < k:
                    continue
                for chosen in itertools.combinations(free_actions, k):
                    for values in itertools.product(*[domains[a] for a in chosen]):
                        changes = dict(static_plan)
                        changes.update(
                            {a: v for a, v in zip(chosen, values, strict=True)}
                        )
                        self._consider_changes(env, changes, found)

        return sorted(found.values(), key=lambda item: item.fingerprint)

    def _locked_value_as_action(
        self, policy: CompiledGovernancePolicy, action: str
    ) -> Any | None:
        rule = policy.field_for_action(action)
        if rule is None:
            return None
        feature = rule.feature
        if feature not in self._locked_static_values:
            return None
        locked = self._locked_static_values[feature]
        # Recover the ReferenceSelection whose resolved raw value matches the lock.
        for profile in sorted(
            self.reference_pool.profiles, key=lambda item: item.profile_id
        ):
            selection = ReferenceSelection(reference_id=profile.profile_id)
            try:
                resolved = resolve_reference_selection(
                    action, selection, self.reference_pool, rule
                )
            except Exception:  # noqa: BLE001
                continue
            if values_exactly_equal(resolved, locked):
                return selection
        return None

    def _consider_changes(
        self,
        env: AttackEnvironment,
        changes: Mapping[str, Any],
        found: dict[str, _LegalCandidate],
    ) -> None:
        item = self._evaluate_candidate(env, changes)
        if item is None:
            return
        if item.fingerprint in self._seen_fingerprints or item.fingerprint in found:
            return
        found[item.fingerprint] = item

    def _evaluate_candidate(
        self, env: AttackEnvironment, changes: Mapping[str, Any]
    ) -> _LegalCandidate | None:
        validator = env.validator
        policy = validator.policy
        anchor = env.starting_case.features
        proposal = AttackProposal(changes=dict(changes))
        if self._static_locked:
            locked = dict(self._locked_static_values)
            pre_errors: tuple[str, ...] = ()
        else:
            preparation = validator.prepare_episode_locks(anchor, proposal)
            locked = dict(preparation.locked_values)
            pre_errors = preparation.errors
        projected = validator.project_for_billing(
            anchor, proposal, locked_values=locked
        )
        edited, distance, _, _ = compute_edit_metrics(
            anchor=anchor,
            candidate=projected,
            mutable_feature_names=validator.mutable_feature_names(),
            previous_candidate=None,
        )
        if distance < 1 or distance > int(self.budget.m_max):
            return None
        validity = validator.validate(
            anchor,
            proposal,
            locked_values=locked,
            pre_feedback_errors=pre_errors,
        )
        if not validity.is_valid:
            return None
        if self.constraint_profile is not None:
            profile_check = self.constraint_profile.check_edited_features(
                edited,
                candidate_features=projected,
                persona_locked=bool(env.profile_persona_locked),
                locked_persona_field=env.profile_persona_field,
                locked_persona_value=env.profile_persona_value,
                forbidden_fields=policy.forbidden_fields,
                read_only_fields=self.reference_pool.read_only_context_fields,
            )
            if not profile_check.is_allowed:
                return None

        static_features = set(policy.episode_static_fields)
        locked_edit_count = sum(
            1
            for name in static_features
            if name in projected
            and name in anchor
            and not _values_equal(projected[name], anchor[name])
        )
        dynamic_edit_count = distance - locked_edit_count
        remaining_dynamic = max(0, int(self.budget.m_max) - locked_edit_count)
        action_fields = [
            name for name in self.reference_pool.action_fields if name in projected
        ]
        fingerprint = canonical_candidate_fingerprint(
            anchor_id=env.starting_case.case_id,
            projected_candidate=projected,
            action_fields=action_fields,
        )
        static_plan = {
            action: value
            for action, value in dict(changes).items()
            if (rule := policy.field_for_action(action)) is not None
            and rule.is_episode_locked
        }
        return _LegalCandidate(
            changes=dict(changes),
            fingerprint=fingerprint,
            projected=dict(projected),
            distance=int(distance),
            locked_edit_count=int(locked_edit_count),
            dynamic_edit_count=int(dynamic_edit_count),
            remaining_dynamic_slots=int(remaining_dynamic),
            edited_fields=tuple(edited),
            static_plan=static_plan,
        )

    def _rank_candidates(
        self, remaining: Sequence[_LegalCandidate]
    ) -> list[tuple[_LegalCandidate, int]]:
        profiles = [dict(p.fields) for p in self.reference_pool.profiles]
        failed = list(self._failed_projected)
        scored: list[tuple[tuple[Any, ...], _LegalCandidate]] = []
        for item in remaining:
            ref_gower = mean_gower_to_profiles(
                item.projected, profiles, self._gower_specs
            )
            if failed:
                overlap = _max_field_value_overlap(item.projected, failed)
                min_fail_gower = min_gower_to_set(
                    item.projected, failed, self._gower_specs
                )
                # Higher diversification first: less overlap, larger min distance.
                key = (
                    0,  # placeholder for non-duplicate (already filtered)
                    overlap,
                    -min_fail_gower,
                    ref_gower,
                    item.locked_edit_count,
                    -item.remaining_dynamic_slots,
                    item.fingerprint,
                )
            else:
                key = (
                    ref_gower,
                    item.locked_edit_count,
                    -item.remaining_dynamic_slots,
                    item.fingerprint,
                )
            scored.append((key, item))
        scored.sort(key=lambda pair: pair[0])
        return [(item, index) for index, (_, item) in enumerate(scored)]

    def _to_proposal(
        self,
        chosen: _LegalCandidate,
        *,
        rank_index: int,
        remaining_n: int,
    ) -> AttackProposal:
        profiles = [dict(p.fields) for p in self.reference_pool.profiles]
        ref_gower = mean_gower_to_profiles(
            chosen.projected, profiles, self._gower_specs
        )
        failed = list(self._failed_projected)
        overlap = _max_field_value_overlap(chosen.projected, failed) if failed else 0
        min_fail = (
            min_gower_to_set(chosen.projected, failed, self._gower_specs)
            if failed
            else None
        )
        # Provenance cares about exact pool match of changed fields.
        provenance = audit_reference_provenance(
            anchor=self._anchor_features,
            candidate=chosen.projected,
            pool=self.reference_pool,
            changed_fields=chosen.edited_fields,
        )
        ref_ids = list(reference_ids_from_changes(chosen.changes))
        meta = {
            "attacker_id": self.attacker_id,
            "generation_method": "a2_surrogate_best_first",
            "candidate_fingerprint": chosen.fingerprint,
            "candidate_hash": chosen.fingerprint,
            "edit_distance_from_anchor": chosen.distance,
            "locked_edit_count": chosen.locked_edit_count,
            "dynamic_edit_count": chosen.dynamic_edit_count,
            "remaining_dynamic_slots": chosen.remaining_dynamic_slots,
            "edited_fields": list(chosen.edited_fields),
            "static_lock_plan": dict(chosen.static_plan),
            "reference_gower": ref_gower,
            "failure_field_value_overlap": overlap,
            "min_gower_to_failures": min_fail,
            "rank_index": rank_index,
            "legal_unique_candidates_remaining_before_submit": remaining_n,
            "m_max": self.budget.m_max,
            "q_max": self.budget.q_max,
            "experiment_seed": self.experiment_seed,
            "episode_seed": self._episode_seed,
            "pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_ids_used": ref_ids,
            "reference_provenance": provenance,
            # Explicitly absent model leakage keys for audits.
            "hidden_from_attacker": [
                "d1_risk_score",
                "d1_threshold",
                "shap",
                "fraud_bool",
            ],
        }
        return AttackProposal(
            changes=dict(chosen.changes),
            raw_command=(
                f"{self.attacker_id}:episode_seed={self._episode_seed}:"
                f"rank={rank_index}"
            ),
            research_meta=meta,
        )

    def _record_submission(
        self,
        *,
        env: AttackEnvironment,
        chosen: _LegalCandidate,
        rank_index: int,
        remaining_n: int,
        label: PublicLabel,
        tradeoff_note: str,
    ) -> None:
        profiles = [dict(p.fields) for p in self.reference_pool.profiles]
        ref_gower = mean_gower_to_profiles(
            chosen.projected, profiles, self._gower_specs
        )
        failed = list(self._failed_projected)
        log = {
            "anchor_id": env.starting_case.case_id,
            "m_max": self.budget.m_max,
            "q_max": self.budget.q_max,
            "attempt": env.attempts_used,
            "static_lock_plan": dict(chosen.static_plan),
            "locked_edit_count": chosen.locked_edit_count,
            "dynamic_edit_count": chosen.dynamic_edit_count,
            "remaining_dynamic_slots": chosen.remaining_dynamic_slots,
            "edited_fields": list(chosen.edited_fields),
            "candidate_hash": chosen.fingerprint,
            "reference_gower": ref_gower,
            "failure_field_value_overlap": (
                _max_field_value_overlap(chosen.projected, failed) if failed else 0
            ),
            "min_gower_to_failures": (
                min_gower_to_set(chosen.projected, failed, self._gower_specs)
                if failed
                else None
            ),
            "rank_index": rank_index,
            "legal_unique_candidates_remaining": max(0, remaining_n - 1),
            "public_label": label,
            "queries_used": env.attempts_used,
            "static_tradeoff_note": tradeoff_note,
        }
        self._submission_logs.append(to_jsonable(log))

    def _write(self, text: str) -> None:
        if self.stdout is not None:
            self.stdout.write(text)
            self.stdout.flush()


def _feature(policy: CompiledGovernancePolicy, action: str) -> str | None:
    rule = policy.field_for_action(action)
    return None if rule is None else rule.feature


def _resolve(
    policy: CompiledGovernancePolicy,
    action: str,
    value: Any,
    *,
    pool: ReferencePool,
) -> Any:
    rule = policy.field_for_action(action)
    if rule is None:
        return value
    if isinstance(value, ReferenceSelection):
        return resolve_reference_selection(action, value, pool, rule)
    if rule.agent_action_mode == "proxy_action":
        return rule.resolved_proxy_actions.get(str(value), value)
    return value


def _static_edit_cost(
    policy: CompiledGovernancePolicy,
    anchor: Mapping[str, Any],
    static_plan: Mapping[str, Any],
    *,
    pool: ReferencePool,
) -> int:
    cost = 0
    for action, value in static_plan.items():
        feature = _feature(policy, action)
        if feature is None:
            continue
        if not _values_equal(
            _resolve(policy, action, value, pool=pool), anchor.get(feature)
        ):
            cost += 1
    return cost


def _max_field_value_overlap(
    candidate: Mapping[str, Any],
    failed: Sequence[Mapping[str, Any]],
) -> int:
    best = 0
    cand_items = set(_field_value_pairs(candidate))
    for other in failed:
        overlap = len(cand_items.intersection(_field_value_pairs(other)))
        if overlap > best:
            best = overlap
    return best


def _field_value_pairs(features: Mapping[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for name, value in features.items():
        pairs.add((str(name), json.dumps(to_jsonable(value), sort_keys=True)))
    return pairs


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


__all__ = [
    "A2SearchError",
    "GOWER_POLICY_LEGACY_V1",
    "GOWER_POLICY_PUBLIC_REFERENCE_V2",
    "SUPPORTED_GOWER_POLICIES",
    "SurrogateGuidedSearcher",
]

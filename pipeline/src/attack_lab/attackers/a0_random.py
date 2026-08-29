"""A0 — frozen constrained-random baseline under the Q,m protocol.

A0 is a non-LLM, non-adaptive synthetic-identity baseline:

- at episode start, generate and freeze up to ``Q`` candidates before any D1
  feedback is observed;
- each candidate satisfies ``edit_distance(candidate, anchor) <= m``;
- edit distance is always relative to the original anchor;
- submit the frozen sequence until PASS or Q is exhausted;
- feedback never influences candidate generation.

Sampling policy (reference-backed revision):

- every changed raw value is a fragment from the current anchor-specific K-pool;
- attackers propose :class:`ReferenceSelection` tokens; trusted code resolves
  raw values (governance judges legality only and is not a value source);
- prefer constrained random draws over K-pool selections for each frozen slot;
- when local random retries are exhausted, deterministically enumerate the
  remaining K-pool-backed unique candidates under the shared lock plan and
  ``m``, then choose one uniformly with a stable seed;
- emit ``no_feasible_candidate`` only when that enumerated remainder is empty.

Shared inputs with A1–A3: anchor, K-reference pool, governance policy, mutable
fields/constraints, and the ``(Q, m)`` budget.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TextIO

import numpy as np

from attack_lab.budget import compute_edit_metrics
from attack_lab.candidate_identity import canonical_candidate_fingerprint
from attack_lab.constraint_profile import IdentityCompositionProfile
from attack_lab.environment import AttackEnvironment
from attack_lab.governance import CompiledGovernancePolicy
from attack_lab.reference_actions import (
    ReferenceSelection,
    audit_reference_provenance,
    reference_backed_selections_for_action,
    reference_ids_from_changes,
    resolve_reference_selection,
)
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import AttackProposal, to_jsonable
from attack_lab.validator import ConstraintValidator


class A0SamplingError(RuntimeError):
    """Raised when governance / pool does not expose a usable sampling domain."""


_MAX_LOCAL_RESAMPLES = 64
_MAX_LOCKED_PLAN_REDRAWS = 32

# Researcher-facing reject labels for random-proposal diagnostics.
REJECT_SAME_AS_ANCHOR = "same_as_anchor"
REJECT_DUPLICATE = "duplicate"
REJECT_BUDGET_EXCEEDED = "budget_exceeded"
REJECT_PROXY_FAILED = "proxy_failed"
REJECT_CONSTRAINT_FAILED = "constraint_failed"
REJECT_OTHER = "other"


@dataclass(frozen=True)
class _FieldDraw:
    action_key: str
    value: Any
    from_reference: bool
    reference_id: str | None


@dataclass(frozen=True)
class _EnumeratedCandidate:
    changes: dict[str, Any]
    fingerprint: str
    distance: int
    edited_fields: tuple[str, ...]
    retained_fields: tuple[str, ...]


def derive_episode_seed(
    experiment_seed: int, anchor_id: str, attacker_name: str
) -> int:
    """Stable per-episode seed: different anchors explore different trajectories."""
    digest = hashlib.sha256(
        f"{int(experiment_seed)}:{anchor_id}:{attacker_name}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def classify_validation_reject(errors: Sequence[str]) -> str:
    """Map validator errors onto the A0 diagnostic reject taxonomy."""
    text = " | ".join(errors).lower()
    if not text:
        return REJECT_OTHER
    if "reference_provenance" in text:
        return REJECT_CONSTRAINT_FAILED
    if "proxy" in text:
        return REJECT_PROXY_FAILED
    if (
        "conditional" in text
        or "coherence" in text
        or "relationship" in text
        or "domain" in text
        or "support" in text
        or "out of" in text
        or "bound" in text
        or "not permitted" in text
        or "locked" in text
    ):
        return REJECT_CONSTRAINT_FAILED
    return REJECT_OTHER


def stable_uniform_index(
    *,
    n_items: int,
    episode_seed: int,
    candidate_index: int,
    remainder_fingerprint: str,
) -> int:
    """Stable uniform index over a sorted remainder (independent of RNG burns)."""
    if n_items < 1:
        raise A0SamplingError("stable_uniform_index requires a non-empty remainder.")
    digest = hashlib.sha256(
        f"{int(episode_seed)}:a0_enum_fallback:{int(candidate_index)}:"
        f"{remainder_fingerprint}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % n_items


@dataclass
class ConstrainedRandomAttacker:
    """Official A0 frozen constrained-random baseline (Q,m protocol)."""

    seed: int
    """Experiment-level seed (not used directly as the episode RNG seed)."""

    reference_pool: ReferencePool
    m_max: int
    """Maximum governance-compiled feature edits relative to the original anchor."""

    attacker_id: str = "a0"
    stdout: TextIO | None = None
    max_local_resamples: int = _MAX_LOCAL_RESAMPLES
    max_locked_plan_redraws: int = _MAX_LOCKED_PLAN_REDRAWS
    #: Optional layered eligibility filter shared with A2/environment.
    constraint_profile: IdentityCompositionProfile | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _episode_seed: int | None = field(default=None, init=False, repr=False)
    _frozen_proposals: list[AttackProposal] = field(
        default_factory=list, init=False, repr=False
    )
    _submit_index: int = field(default=0, init=False, repr=False)
    _sequence_prepared: bool = field(default=False, init=False, repr=False)
    _selected_lock_edits: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock_reference_ids: tuple[str, ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _pending_stop_reason: str | None = field(default=None, init=False, repr=False)
    _reject_counts: Counter[str] = field(
        default_factory=Counter, init=False, repr=False
    )
    _undersample_events: int = field(default=0, init=False, repr=False)
    _enum_fallback_picks: int = field(default=0, init=False, repr=False)
    _termination_audit: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.reference_pool.K < 1:
            raise A0SamplingError("reference_pool.K must be >= 1.")
        if self.m_max < 0:
            raise A0SamplingError("m_max must be >= 0.")
        # Placeholder RNG; replaced when the episode sequence is prepared.
        self._rng = np.random.default_rng(0)

    @property
    def experiment_seed(self) -> int:
        return self.seed

    @property
    def frozen_proposals(self) -> tuple[AttackProposal, ...]:
        return tuple(self._frozen_proposals)

    @property
    def sampling_diagnostics(self) -> dict[str, Any]:
        """Researcher-facing sampling audit (not attacker-public feedback)."""
        return {
            "reject_counts": dict(sorted(self._reject_counts.items())),
            "undersample_events": int(self._undersample_events),
            "enum_fallback_picks": int(self._enum_fallback_picks),
            "lock_plan": dict(self._selected_lock_edits),
            "lock_reference_ids": list(self._lock_reference_ids),
            "m_max": int(self.m_max),
            "reference_pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_pool_K": int(self.reference_pool.K),
            "frozen_count": len(self._frozen_proposals),
            "termination_audit": self._termination_audit,
        }

    def run(self, env: AttackEnvironment) -> None:
        """Freeze the candidate sequence, then submit without feedback adaptation."""
        self.prepare_frozen_sequence(env)
        self._write(
            f"\n=== A0 FrozenConstrainedRandom "
            f"(experiment_seed={self.seed}, episode_seed={self._episode_seed}, "
            f"case={env.starting_case.case_id}, Q={len(self._frozen_proposals)}, "
            f"m={self.m_max}, pool_fp={self.reference_pool.pool_fingerprint[:12]}) ===\n"
            "Pre-feedback frozen candidate sequence; Q/m protocol; "
            "feedback does not alter generation; "
            "enum fallback after random retries.\n"
        )
        while not env.done:
            proposal = self.propose(env)
            if proposal is None:
                reason = self._pending_stop_reason or "no_feasible_candidate"
                self._capture_termination_audit(
                    env,
                    reason=reason,
                    submitted_proposals=self._frozen_proposals[: self._submit_index],
                )
                self._write(f"Local stop: {reason}.\n")
                self._write(
                    "Termination audit: "
                    f"{json.dumps(to_jsonable(self._termination_audit), sort_keys=True)}\n"
                )
                env.abort(reason=reason)
                return
            self._write(
                f"candidate_index={self._submit_index}: submitting "
                f"{sorted(proposal.changes)}\n"
            )
            # Intentionally ignore StepRecord feedback for generation/adaptation.
            env.step(proposal)
            self._submit_index += 1

        self._capture_termination_audit(
            env,
            reason=env.stop_reason or ("success" if env.success else "stopped"),
            submitted_proposals=self._frozen_proposals[: self._submit_index],
        )
        self._write(
            f"Episode stop observed via environment.done "
            f"(success={env.success}).\n"
        )

    def prepare_frozen_sequence(self, env: AttackEnvironment) -> tuple[AttackProposal, ...]:
        """Generate and freeze up to Q candidates before any D1 observation."""
        if self._sequence_prepared:
            return self.frozen_proposals
        if env.done:
            self._sequence_prepared = True
            self._pending_stop_reason = "q_exhausted"
            return ()

        q_max = int(env.budget.q_max)
        if q_max < 1:
            raise A0SamplingError("budget.q_max must be >= 1.")

        self._episode_seed = derive_episode_seed(
            self.seed, env.starting_case.case_id, self.attacker_id
        )
        self._rng = np.random.default_rng(self._episode_seed)
        self._selected_lock_edits = {}
        self._lock_reference_ids = ()
        self._frozen_proposals = []
        self._submit_index = 0
        self._pending_stop_reason = None
        self._reject_counts = Counter()
        self._undersample_events = 0
        self._enum_fallback_picks = 0
        self._termination_audit = None

        if not self._prepare_shared_locked_plan(env):
            self._sequence_prepared = True
            self._pending_stop_reason = "no_feasible_candidate"
            self._capture_termination_audit(env, reason="no_feasible_candidate")
            return ()

        seen_fingerprints: set[str] = set()
        for candidate_index in range(1, q_max + 1):
            proposal = self._draw_next_frozen_candidate(
                env,
                candidate_index=candidate_index,
                seen_fingerprints=seen_fingerprints,
            )
            if proposal is None:
                # Random exhausted and enumeration remainder is empty.
                break
            fingerprint = str(proposal.research_meta.get("candidate_fingerprint", ""))
            if fingerprint:
                seen_fingerprints.add(fingerprint)
            self._frozen_proposals.append(proposal)

        self._sequence_prepared = True
        if not self._frozen_proposals:
            self._pending_stop_reason = (
                "insufficient_edit_budget"
                if self.m_max < 1
                else "no_feasible_candidate"
            )
            self._capture_termination_audit(
                env, reason=self._pending_stop_reason or "no_feasible_candidate"
            )
        return self.frozen_proposals

    def propose(self, env: AttackEnvironment) -> AttackProposal | None:
        """Return the next frozen candidate; never regenerates from feedback."""
        if not self._sequence_prepared:
            self.prepare_frozen_sequence(env)
        self._pending_stop_reason = None
        if env.done:
            return None
        if env.ledger.q_remaining < 1:
            self._pending_stop_reason = "q_exhausted"
            return None
        if self._submit_index >= len(self._frozen_proposals):
            self._pending_stop_reason = (
                "q_exhausted"
                if self._submit_index >= int(env.budget.q_max)
                else "no_feasible_candidate"
            )
            return None
        return self._frozen_proposals[self._submit_index]

    def _draw_next_frozen_candidate(
        self,
        env: AttackEnvironment,
        *,
        candidate_index: int,
        seen_fingerprints: set[str],
    ) -> AttackProposal | None:
        """Random sample first; on exhaustion, enum-fallback pick or None."""
        for _ in range(self.max_local_resamples):
            trial, reject = self._sample_candidate_once(
                env, candidate_index=candidate_index
            )
            if trial is None:
                self._reject_counts[reject or REJECT_OTHER] += 1
                continue
            fingerprint = str(trial.research_meta.get("candidate_fingerprint", ""))
            if fingerprint and fingerprint in seen_fingerprints:
                self._reject_counts[REJECT_DUPLICATE] += 1
                continue
            if not self._locally_feasible(env, trial):
                preparation = env.validator.prepare_episode_locks(
                    env.starting_case.features, trial
                )
                validity = env.validator.validate(
                    env.starting_case.features,
                    trial,
                    locked_values=preparation.locked_values,
                    pre_feedback_errors=preparation.errors,
                )
                self._reject_counts[
                    classify_validation_reject(validity.errors)
                ] += 1
                continue
            distance = int(trial.research_meta.get("edit_distance_from_anchor", -1))
            if distance < 1:
                self._reject_counts[REJECT_SAME_AS_ANCHOR] += 1
                continue
            if distance > self.m_max:
                self._reject_counts[REJECT_BUDGET_EXCEEDED] += 1
                continue
            meta = dict(trial.research_meta)
            meta["generation_method"] = "random"
            return AttackProposal(
                changes=dict(trial.changes),
                raw_command=trial.raw_command,
                research_meta=meta,
            )

        remainder = self._enumerate_legal_unique_candidates(
            env, seen_fingerprints=seen_fingerprints
        )
        if not remainder:
            return None

        # Random returned empty for this slot, but legal unique candidates remain.
        self._undersample_events += 1
        picked = self._stable_pick_enumerated(
            remainder, candidate_index=candidate_index
        )
        self._enum_fallback_picks += 1
        return self._proposal_from_enumerated(
            env, picked, candidate_index=candidate_index
        )

    def _prepare_shared_locked_plan(self, env: AttackEnvironment) -> bool:
        """Sample one episode-static lock plan shared by all frozen candidates."""
        for _ in range(self.max_locked_plan_redraws):
            plan, ref_ids = self._sample_locked_plan_once(env)
            if self.constraint_profile is not None:
                static_features = self._static_features_from_plan(env, plan)
                if not self.constraint_profile.is_compatible_static_lock_plan(
                    static_features
                ):
                    continue
            if plan:
                trial = AttackProposal(changes=plan, raw_command="a0:lock_trial")
                # Under identity-composition, the shared lock is persona-only and
                # intentionally incomplete (contact is added later). Full profile
                # eligibility applies only to complete candidates, so validate
                # governance on the lock trial without the composition check.
                if self.constraint_profile is not None:
                    if not self._governance_feasible(env, trial):
                        continue
                elif not self._locally_feasible(env, trial):
                    continue
                lock_cost = self._edit_distance(env, trial)
                if lock_cost > self.m_max:
                    continue
            elif self.constraint_profile is not None:
                # Profile requires exactly one persona lock edit.
                continue
            self._selected_lock_edits = plan
            self._lock_reference_ids = ref_ids
            return True
        self._selected_lock_edits = {}
        self._lock_reference_ids = ()
        # Unrestricted A0 may freeze with an empty lock plan; the composition
        # profile cannot, because persona must occupy the static lock slot.
        return self.constraint_profile is None

    def _static_features_from_plan(
        self, env: AttackEnvironment, plan: Mapping[str, Any]
    ) -> list[str]:
        anchor = env.starting_case.features
        policy = env.validator.policy
        edited: list[str] = []
        for action, value in plan.items():
            rule = policy.field_for_action(action)
            if rule is None:
                continue
            resolved = _resolve_action_value(policy, action, value, pool=self.reference_pool)
            if not _values_equal(resolved, anchor.get(rule.feature)):
                edited.append(rule.feature)
        return edited

    def _sample_locked_plan_once(
        self, env: AttackEnvironment
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        validator = env.validator
        locked_actions = self._enabled_locked_actions(validator)
        if self.constraint_profile is not None:
            persona = set(self.constraint_profile.persona_profile_fields)
            locked_actions = tuple(
                action
                for action in locked_actions
                if (rule := validator.policy.field_for_action(action)) is not None
                and rule.feature in persona
            )
        if not locked_actions or self.m_max < 1:
            return {}, ()

        if self.constraint_profile is not None:
            # Identity-composition requires exactly one persona lock edit.
            k = 1
        else:
            k_max = min(len(locked_actions), self.m_max)
            k = int(self._rng.integers(0, k_max + 1))
            if k == 0:
                return {}, ()
        chosen = list(
            self._rng.choice(
                np.array(locked_actions, dtype=object), size=k, replace=False
            )
        )
        ordered = _order_actions_for_constraints(validator.policy, chosen)
        working = dict(env.starting_case.features)
        plan: dict[str, Any] = {}
        ref_ids: list[str] = []
        for action_key in ordered:
            draw = self._draw_action_value(
                validator=validator,
                action_key=action_key,
                working=working,
                prefer_change_from=env.starting_case.features,
                force_reference=False,
            )
            if draw is None:
                continue
            plan[action_key] = draw.value
            if draw.from_reference and draw.reference_id is not None:
                ref_ids.append(draw.reference_id)
            feature = _feature_for_action(validator.policy, action_key)
            if feature is not None:
                working[feature] = _resolve_action_value(validator.policy, action_key, draw.value, pool=self.reference_pool)
        return plan, tuple(dict.fromkeys(ref_ids))

    def _sample_candidate_once(
        self, env: AttackEnvironment, *, candidate_index: int
    ) -> tuple[AttackProposal | None, str]:
        validator = env.validator
        policy = validator.policy
        anchor = env.starting_case.features

        lock_changes = dict(self._selected_lock_edits)
        lock_cost = 0
        for action_key, value in lock_changes.items():
            feature = _feature_for_action(policy, action_key)
            if feature is not None and not _values_equal(
                _resolve_action_value(policy, action_key, value, pool=self.reference_pool),
                anchor.get(feature),
            ):
                lock_cost += 1
        if lock_cost > self.m_max:
            return None, REJECT_BUDGET_EXCEEDED

        free_budget = self.m_max - lock_cost
        free_actions = self._enabled_free_actions(validator)
        free_draws = self._sample_free_draws(
            validator=validator,
            free_actions=free_actions,
            anchor=anchor,
            lock_changes=lock_changes,
            max_edits=free_budget,
        )
        changes = {
            **lock_changes,
            **{draw.action_key: draw.value for draw in free_draws},
        }
        if not changes:
            return None, REJECT_SAME_AS_ANCHOR

        composed = self._composition_meta(
            env=env,
            changes=changes,
            free_draws=free_draws,
            lock_reference_ids=self._lock_reference_ids,
            candidate_index=candidate_index,
        )
        if composed is None:
            return None, REJECT_OTHER
        final_changes, meta = composed
        if not final_changes:
            return None, REJECT_SAME_AS_ANCHOR
        distance = int(meta["edit_distance_from_anchor"])
        if distance < 1:
            # Require at least one edit; zero-edit repeats waste Q without progress.
            return None, REJECT_SAME_AS_ANCHOR
        if distance > self.m_max:
            return None, REJECT_BUDGET_EXCEEDED
        return (
            AttackProposal(
                changes=final_changes,
                raw_command=(
                    f"{self.attacker_id}:episode_seed={self._episode_seed}:"
                    f"candidate={candidate_index}"
                ),
                research_meta=meta,
            ),
            "",
        )

    def _sample_free_draws(
        self,
        *,
        validator: ConstraintValidator,
        free_actions: tuple[str, ...],
        anchor: Mapping[str, Any],
        lock_changes: Mapping[str, Any],
        max_edits: int,
    ) -> list[_FieldDraw]:
        if max_edits < 1 or not free_actions:
            return []
        k_max = min(len(free_actions), max_edits)
        k = int(self._rng.integers(1, k_max + 1))
        chosen = list(
            self._rng.choice(
                np.array(free_actions, dtype=object), size=k, replace=False
            )
        )
        ordered = _order_actions_for_constraints(validator.policy, chosen)
        working = dict(anchor)
        for action_key, value in lock_changes.items():
            feature = _feature_for_action(validator.policy, action_key)
            if feature is not None:
                working[feature] = _resolve_action_value(validator.policy, action_key, value, pool=self.reference_pool)
        draws: list[_FieldDraw] = []
        for action_key in ordered:
            draw = self._draw_action_value(
                validator=validator,
                action_key=action_key,
                working=working,
                prefer_change_from=anchor,
                force_reference=False,
            )
            if draw is None:
                continue
            draws.append(draw)
            feature = _feature_for_action(validator.policy, action_key)
            if feature is not None:
                working[feature] = _resolve_action_value(validator.policy, action_key, draw.value, pool=self.reference_pool)
        return draws

    def _composition_meta(
        self,
        *,
        env: AttackEnvironment,
        changes: Mapping[str, Any],
        free_draws: Sequence[_FieldDraw],
        lock_reference_ids: Sequence[str],
        candidate_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        anchor = env.starting_case.features
        working_changes = dict(changes)
        working_draws = list(free_draws)
        # Action generation uses compiled-governance action fields only
        # (not read-only context fields such as bank_months_count).
        action_fields = [
            name
            for name in self.reference_pool.action_fields
            if name in anchor
        ]
        if not action_fields:
            return None

        def _project() -> dict[str, Any]:
            return env.validator.project_for_billing(
                anchor,
                AttackProposal(changes=working_changes),
                locked_values=None,
            )

        def _split(
            projected: Mapping[str, Any],
        ) -> tuple[list[str], list[str]]:
            retained_names = [
                name
                for name in action_fields
                if _values_equal(projected[name], anchor[name])
            ]
            replaced_names = [
                name
                for name in action_fields
                if not _values_equal(projected[name], anchor[name])
            ]
            return retained_names, replaced_names

        candidate = _project()
        retained, replaced = _split(candidate)

        if not retained:
            free_keys = [draw.action_key for draw in working_draws]
            if not free_keys:
                return None
            drop = str(self._rng.choice(np.array(free_keys, dtype=object)))
            working_changes.pop(drop, None)
            working_draws = [draw for draw in working_draws if draw.action_key != drop]
            candidate = _project()
            retained, replaced = _split(candidate)
            if not retained or not replaced:
                return None

        ref_ids = list(lock_reference_ids)
        ref_ids.extend(
            draw.reference_id
            for draw in working_draws
            if draw.from_reference and draw.reference_id is not None
        )
        ref_ids.extend(reference_ids_from_changes(working_changes))
        ref_ids = list(dict.fromkeys(ref_ids))

        # All A0 draws are reference-backed; reject incomplete compositions.
        if not ref_ids:
            return None

        if self._matches_full_reference_profile(env.validator.policy, candidate):
            return None

        trial = AttackProposal(changes=working_changes)
        distance = self._edit_distance(env, trial)
        if distance > self.m_max:
            return None

        edited_fields = self._edited_feature_names(env, trial)
        fingerprint = canonical_candidate_fingerprint(
            anchor_id=env.starting_case.case_id,
            projected_candidate=candidate,
            action_fields=action_fields,
        )
        provenance = audit_reference_provenance(
            anchor=anchor,
            candidate=candidate,
            pool=self.reference_pool,
            changed_fields=edited_fields,
        )
        if provenance["status"] != "PASS":
            return None
        assert self._episode_seed is not None
        meta = {
            "anchor_id": env.starting_case.case_id,
            "candidate_index": candidate_index,
            "candidate_fingerprint": fingerprint,
            "edited_fields": list(edited_fields),
            "reference_ids_used": ref_ids,
            "retained_fields": sorted(retained),
            "edit_distance_from_anchor": distance,
            "generation_seed": self._episode_seed,
            "experiment_seed": self.seed,
            "m_max": self.m_max,
            "pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_pool_fingerprint": self.reference_pool.pool_fingerprint,
            "pool_K": self.reference_pool.K,
            "reference_provenance": provenance,
        }
        return working_changes, meta

    def _matches_full_reference_profile(
        self,
        policy: CompiledGovernancePolicy,
        candidate: Mapping[str, Any],
    ) -> bool:
        mutable_action = [
            name
            for name in self.reference_pool.action_fields
            if name in policy.fields and policy.fields[name].is_mutable
        ]
        for profile in self.reference_pool.profiles:
            fields = [
                name
                for name in mutable_action
                if name in profile.fields and name in candidate
            ]
            if fields and all(
                _values_equal(candidate[name], profile.fields[name]) for name in fields
            ):
                return True
        return False

    def _draw_action_value(
        self,
        *,
        validator: ConstraintValidator,
        action_key: str,
        working: Mapping[str, Any],
        prefer_change_from: Mapping[str, Any] | None,
        force_reference: bool,
    ) -> _FieldDraw | None:
        """Sample one K-pool ReferenceSelection for ``action_key``.

        ``force_reference`` is retained for call-site compatibility; all draws
        are reference-backed.
        """
        _ = working
        _ = force_reference
        rule = validator.policy.field_for_action(action_key)
        if rule is None or not rule.is_mutable:
            return None
        feature = rule.feature
        if feature not in self.reference_pool.action_fields:
            return None
        anchor_value = None if prefer_change_from is None else prefer_change_from.get(feature)
        selections = list(
            reference_backed_selections_for_action(
                action_key=action_key,
                pool=self.reference_pool,
                rule=rule,
                anchor_value=anchor_value,
                require_change_from_anchor=prefer_change_from is not None,
            )
        )
        if not selections:
            return None
        order = list(selections)
        self._rng.shuffle(order)
        for selection in order:
            assert isinstance(selection, ReferenceSelection)
            try:
                resolved = resolve_reference_selection(
                    action_key, selection, self.reference_pool, rule
                )
            except Exception:  # noqa: BLE001
                continue
            if prefer_change_from is not None and _values_equal(
                resolved, prefer_change_from.get(feature)
            ):
                continue
            return _FieldDraw(
                action_key=action_key,
                value=selection,
                from_reference=True,
                reference_id=selection.reference_id,
            )
        return None

    def _edit_distance(self, env: AttackEnvironment, proposal: AttackProposal) -> int:
        edited = self._edited_feature_names(env, proposal)
        return len(edited)

    def _edited_feature_names(
        self, env: AttackEnvironment, proposal: AttackProposal
    ) -> tuple[str, ...]:
        anchor = env.starting_case.features
        preparation = env.validator.prepare_episode_locks(anchor, proposal)
        candidate = env.validator.project_for_billing(
            anchor,
            proposal,
            locked_values=preparation.locked_values,
        )
        edited, _, _, _ = compute_edit_metrics(
            anchor=anchor,
            candidate=candidate,
            mutable_feature_names=env.validator.mutable_feature_names(),
            previous_candidate=None,
        )
        return edited

    def _governance_feasible(
        self, env: AttackEnvironment, proposal: AttackProposal
    ) -> bool:
        """Governance-only local check (no layered constraint profile)."""
        validator = env.validator
        anchor = env.starting_case.features
        preparation = validator.prepare_episode_locks(anchor, proposal)
        result = validator.validate(
            anchor,
            proposal,
            locked_values=preparation.locked_values,
            pre_feedback_errors=preparation.errors,
        )
        return bool(result.is_valid)

    def _locally_feasible(
        self, env: AttackEnvironment, proposal: AttackProposal
    ) -> bool:
        validator = env.validator
        anchor = env.starting_case.features
        preparation = validator.prepare_episode_locks(anchor, proposal)
        result = validator.validate(
            anchor,
            proposal,
            locked_values=preparation.locked_values,
            pre_feedback_errors=preparation.errors,
        )
        if not result.is_valid:
            return False
        if self.constraint_profile is None:
            return True
        projected = validator.project_for_billing(
            anchor, proposal, locked_values=preparation.locked_values
        )
        edited, _, _, _ = compute_edit_metrics(
            anchor=anchor,
            candidate=projected,
            mutable_feature_names=validator.mutable_feature_names(),
            previous_candidate=None,
        )
        check = self.constraint_profile.check_edited_features(
            edited,
            candidate_features=projected,
            persona_locked=False,
            forbidden_fields=validator.policy.forbidden_fields,
            read_only_fields=self.reference_pool.read_only_context_fields,
        )
        return check.is_allowed

    @staticmethod
    def _enabled_locked_actions(validator: ConstraintValidator) -> tuple[str, ...]:
        keys: list[str] = []
        for action in validator.enabled_action_keys:
            rule = validator.policy.field_for_action(action)
            if rule is not None and rule.is_episode_locked:
                keys.append(action)
        return tuple(keys)

    def _enabled_free_actions(self, validator: ConstraintValidator) -> tuple[str, ...]:
        keys: list[str] = []
        for action in validator.enabled_action_keys:
            rule = validator.policy.field_for_action(action)
            if rule is not None and rule.is_mutable and not rule.is_episode_locked:
                keys.append(action)
        if self.constraint_profile is not None:
            contact = set(self.constraint_profile.contact_identity_fields)
            keys = [
                action
                for action in keys
                if (rule := validator.policy.field_for_action(action)) is not None
                and rule.feature in contact
            ]
        return tuple(keys)

    def _action_value_domain(
        self, validator: ConstraintValidator, action_key: str
    ) -> tuple[Any, ...]:
        """Finite K-pool ReferenceSelection catalogue for enum fallback."""
        rule = validator.policy.field_for_action(action_key)
        if rule is None or not rule.is_mutable:
            return ()
        if rule.feature not in self.reference_pool.action_fields:
            return ()
        anchor_value = None  # full legality checked later via validate
        return reference_backed_selections_for_action(
            action_key=action_key,
            pool=self.reference_pool,
            rule=rule,
            anchor_value=anchor_value,
            require_change_from_anchor=False,
        )

    def _evaluate_enumerated_changes(
        self,
        env: AttackEnvironment,
        changes: Mapping[str, Any],
        *,
        exclusion_counts: Counter[str] | None = None,
    ) -> _EnumeratedCandidate | None:
        """Return a legal unique-candidate descriptor, or None if rejected."""
        validator = env.validator
        anchor = env.starting_case.features
        proposal = AttackProposal(changes=dict(changes))
        preparation = validator.prepare_episode_locks(anchor, proposal)
        projected = validator.project_for_billing(
            anchor, proposal, locked_values=preparation.locked_values
        )
        edited, distance, _, _ = compute_edit_metrics(
            anchor=anchor,
            candidate=projected,
            mutable_feature_names=validator.mutable_feature_names(),
            previous_candidate=None,
        )
        counts = exclusion_counts

        def _bump(label: str) -> None:
            if counts is not None:
                counts[label] += 1

        if distance < 1:
            _bump(REJECT_SAME_AS_ANCHOR)
            return None
        if distance > self.m_max:
            _bump(REJECT_BUDGET_EXCEEDED)
            return None
        validity = validator.validate(
            anchor,
            proposal,
            locked_values=preparation.locked_values,
            pre_feedback_errors=preparation.errors,
        )
        if not validity.is_valid:
            _bump(classify_validation_reject(validity.errors))
            return None
        if self.constraint_profile is not None:
            profile_check = self.constraint_profile.check_edited_features(
                edited,
                candidate_features=projected,
                persona_locked=False,
                forbidden_fields=validator.policy.forbidden_fields,
                read_only_fields=self.reference_pool.read_only_context_fields,
            )
            if not profile_check.is_allowed:
                _bump(REJECT_CONSTRAINT_FAILED)
                return None

        action_fields = [
            name for name in self.reference_pool.action_fields if name in projected
        ]
        fingerprint = canonical_candidate_fingerprint(
            anchor_id=env.starting_case.case_id,
            projected_candidate=projected,
            action_fields=action_fields,
        )
        retained = tuple(
            sorted(
                name
                for name in action_fields
                if _values_equal(projected[name], anchor[name])
            )
        )
        return _EnumeratedCandidate(
            changes=dict(changes),
            fingerprint=fingerprint,
            distance=int(distance),
            edited_fields=tuple(edited),
            retained_fields=retained,
        )

    def _enumerate_legal_unique_candidates(
        self,
        env: AttackEnvironment,
        *,
        seen_fingerprints: set[str],
        exclusion_counts: Counter[str] | None = None,
    ) -> list[_EnumeratedCandidate]:
        """Deterministic remainder of governance-legal unique candidates."""
        validator = env.validator
        policy = validator.policy
        anchor = env.starting_case.features
        lock_changes = dict(self._selected_lock_edits)
        counts = exclusion_counts if exclusion_counts is not None else Counter()

        lock_cost = 0
        for action_key, value in lock_changes.items():
            feature = _feature_for_action(policy, action_key)
            if feature is not None and not _values_equal(
                _resolve_action_value(policy, action_key, value, pool=self.reference_pool),
                anchor.get(feature),
            ):
                lock_cost += 1
        if lock_cost > self.m_max:
            counts[REJECT_BUDGET_EXCEEDED] += 1
            return []

        free_budget = self.m_max - lock_cost
        free_actions = [
            action
            for action in self._enabled_free_actions(validator)
            if self._action_value_domain(validator, action)
        ]
        domains = {
            action: self._action_value_domain(validator, action)
            for action in free_actions
        }

        found: dict[str, _EnumeratedCandidate] = {}

        def _consider(changes: dict[str, Any]) -> None:
            item = self._evaluate_enumerated_changes(
                env, changes, exclusion_counts=counts
            )
            if item is None:
                return
            if item.fingerprint in seen_fingerprints or item.fingerprint in found:
                counts[REJECT_DUPLICATE] += 1
                return
            found[item.fingerprint] = item

        # Lock-only candidate when the shared plan already spends budget.
        if lock_changes:
            _consider(dict(lock_changes))

        if free_budget >= 1 and free_actions:
            for k in range(1, free_budget + 1):
                if len(free_actions) < k:
                    continue
                for chosen in itertools.combinations(free_actions, k):
                    value_lists = [domains[action] for action in chosen]
                    for value_tuple in itertools.product(*value_lists):
                        changes = dict(lock_changes)
                        for action, value in zip(chosen, value_tuple, strict=True):
                            changes[action] = value
                        _consider(changes)

        return sorted(found.values(), key=lambda item: item.fingerprint)

    def _stable_pick_enumerated(
        self,
        remainder: Sequence[_EnumeratedCandidate],
        *,
        candidate_index: int,
    ) -> _EnumeratedCandidate:
        assert self._episode_seed is not None
        remainder_fingerprint = hashlib.sha256(
            ",".join(item.fingerprint for item in remainder).encode("utf-8")
        ).hexdigest()
        index = stable_uniform_index(
            n_items=len(remainder),
            episode_seed=self._episode_seed,
            candidate_index=candidate_index,
            remainder_fingerprint=remainder_fingerprint,
        )
        return remainder[index]

    def _proposal_from_enumerated(
        self,
        env: AttackEnvironment,
        item: _EnumeratedCandidate,
        *,
        candidate_index: int,
    ) -> AttackProposal:
        assert self._episode_seed is not None
        projected = env.validator.project_for_billing(
            env.starting_case.features,
            AttackProposal(changes=dict(item.changes)),
            locked_values=None,
        )
        provenance = audit_reference_provenance(
            anchor=env.starting_case.features,
            candidate=projected,
            pool=self.reference_pool,
            changed_fields=item.edited_fields,
        )
        ref_ids = list(self._lock_reference_ids)
        ref_ids.extend(reference_ids_from_changes(item.changes))
        ref_ids = list(dict.fromkeys(ref_ids))
        meta = {
            "anchor_id": env.starting_case.case_id,
            "candidate_index": candidate_index,
            "candidate_fingerprint": item.fingerprint,
            "edited_fields": list(item.edited_fields),
            "reference_ids_used": ref_ids,
            "retained_fields": list(item.retained_fields),
            "edit_distance_from_anchor": item.distance,
            "generation_seed": self._episode_seed,
            "experiment_seed": self.seed,
            "m_max": self.m_max,
            "pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_pool_fingerprint": self.reference_pool.pool_fingerprint,
            "pool_K": self.reference_pool.K,
            "generation_method": "enum_fallback",
            "lock_plan": dict(self._selected_lock_edits),
            "reference_provenance": provenance,
        }
        return AttackProposal(
            changes=dict(item.changes),
            raw_command=(
                f"{self.attacker_id}:episode_seed={self._episode_seed}:"
                f"candidate={candidate_index}:enum_fallback"
            ),
            research_meta=meta,
        )

    def _capture_termination_audit(
        self,
        env: AttackEnvironment,
        *,
        reason: str,
        submitted_proposals: Sequence[AttackProposal] | None = None,
    ) -> None:
        submitted = list(submitted_proposals or ())
        seen = {
            str(item.research_meta.get("candidate_fingerprint", ""))
            for item in submitted
            if item.research_meta.get("candidate_fingerprint")
        }
        enum_exclusions: Counter[str] = Counter()
        remainder = self._enumerate_legal_unique_candidates(
            env,
            seen_fingerprints=seen,
            exclusion_counts=enum_exclusions,
        )
        # After enum fallback, no_feasible with non-empty remainder is a defect.
        undersample = bool(reason == "no_feasible_candidate" and remainder)
        self._termination_audit = {
            "stop_reason": reason,
            "anchor_id": env.starting_case.case_id,
            "anchor_features": to_jsonable(dict(env.starting_case.features)),
            "lock_plan": dict(self._selected_lock_edits),
            "lock_reference_ids": list(self._lock_reference_ids),
            "m_max": int(self.m_max),
            "reference_pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_pool_K": int(self.reference_pool.K),
            "reference_profile_ids": [
                profile.profile_id for profile in self.reference_pool.profiles
            ],
            "submitted_fingerprints": [
                str(item.research_meta.get("candidate_fingerprint", ""))
                for item in submitted
            ],
            "submitted_changes": [dict(item.changes) for item in submitted],
            "random_reject_counts": dict(sorted(self._reject_counts.items())),
            "enum_exclusion_counts": dict(sorted(enum_exclusions.items())),
            "enum_remaining_count": len(remainder),
            "enum_remaining_fingerprints": [item.fingerprint for item in remainder],
            "undersample_confirmed": undersample,
            "undersample_events_during_freeze": int(self._undersample_events),
            "enum_fallback_picks": int(self._enum_fallback_picks),
            "frozen_count": len(self._frozen_proposals),
            "queries_submitted": len(submitted),
        }

    def _write(self, text: str) -> None:
        if self.stdout is not None:
            self.stdout.write(text)
            self.stdout.flush()


def _order_actions_for_constraints(
    policy: CompiledGovernancePolicy, actions: list[str]
) -> list[str]:
    action_set = set(actions)
    dependents: dict[str, set[str]] = {action: set() for action in actions}
    for action in actions:
        rule = policy.field_for_action(action)
        if rule is None:
            continue
        for constraint in rule.hard_constraints:
            if constraint.get("type") != "conditional_train_range":
                continue
            for condition in constraint.get("condition_fields", ()):
                if condition in action_set:
                    dependents[action].add(condition)
                for other in actions:
                    other_rule = policy.field_for_action(other)
                    if other_rule is not None and other_rule.feature == condition:
                        dependents[action].add(other)
    ordered: list[str] = []
    remaining = set(actions)
    while remaining:
        ready = [action for action in remaining if not (dependents[action] & remaining)]
        if not ready:
            ordered.extend(sorted(remaining))
            break
        ready.sort()
        pick = ready[0]
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def _feature_for_action(
    policy: CompiledGovernancePolicy, action_key: str
) -> str | None:
    rule = policy.field_for_action(action_key)
    return None if rule is None else rule.feature


def _resolve_action_value(
    policy: CompiledGovernancePolicy,
    action_key: str,
    value: Any,
    *,
    pool: ReferencePool | None = None,
) -> Any:
    rule = policy.field_for_action(action_key)
    if rule is None:
        return value
    if isinstance(value, ReferenceSelection):
        if pool is None:
            raise A0SamplingError(
                f"ReferenceSelection for {action_key!r} requires a reference pool."
            )
        return resolve_reference_selection(action_key, value, pool, rule)
    # Legacy literal tokens (should not appear on the A0 generation path).
    if rule.agent_action_mode == "proxy_action":
        return rule.resolved_proxy_actions.get(str(value), value)
    return value


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
    "A0SamplingError",
    "ConstrainedRandomAttacker",
    "REJECT_BUDGET_EXCEEDED",
    "REJECT_CONSTRAINT_FAILED",
    "REJECT_DUPLICATE",
    "REJECT_OTHER",
    "REJECT_PROXY_FAILED",
    "REJECT_SAME_AS_ANCHOR",
    "classify_validation_reject",
    "derive_episode_seed",
    "stable_uniform_index",
]

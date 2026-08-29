"""Core attack-episode environment.

AttackEnvironment is the sole entry point that may:

- receive a candidate proposal;
- compute edit cost against the original anchor;
- charge Q and enforce per-candidate m via BudgetLedger;
- run governance validation;
- invoke the frozen D1 defender;
- emit public feedback;
- decide episode stopping.

Attackers must not forge edit costs or call D1 directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from attack_lab.budget import (
    BudgetLedger,
    BudgetSpec,
    compute_edit_metrics,
)
from attack_lab.cases import StartingCase
from attack_lab.constraint_profile import IdentityCompositionProfile
from attack_lab.feedback import FeedbackPolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.types import (
    AttackProposal,
    EpisodeResult,
    Observation,
    PublicFeedback,
    StepRecord,
    ValidityResult,
)
from attack_lab.validator import ConstraintValidator


class _Defender(Protocol):
    name: str
    artefact_id: str
    threshold: float

    def score_application(self, features: dict[str, Any]): ...


class GuardedDefender:
    """Fail-closed wrapper: D1 scoring is allowed only inside env.step."""

    def __init__(self, inner: _Defender) -> None:
        self._inner = inner
        self._allow_score = False

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def artefact_id(self) -> str:
        return self._inner.artefact_id

    @property
    def threshold(self) -> float:
        return self._inner.threshold

    def score_application(self, features: dict[str, Any]):
        if not self._allow_score:
            raise RuntimeError(
                "D1 scoring is only permitted through AttackEnvironment.step; "
                "direct defender calls are blocked."
            )
        return self._inner.score_application(features)


@dataclass
class AttackEnvironment:
    """Interactive episode against a frozen statistical defence.

    Development smoke-test success rule (not the final experiment freeze):
    the starting case must initially be BLOCKED; a valid modified case that
    becomes PASS under the same frozen threshold ends the episode as success.
    Stopping is otherwise governed by the attached BudgetSpec (Q, m).
    """

    starting_case: StartingCase
    defender: _Defender
    validator: ConstraintValidator
    feedback_policy: FeedbackPolicy
    logger: TrajectoryLogger
    budget: BudgetSpec
    max_attempts: int | None = None
    success_rule_label: str = (
        "development_smoke_test: initial BLOCK -> valid PASS under frozen threshold"
    )
    #: Optional layered candidate filter; does not replace governance-v2.
    constraint_profile: IdentityCompositionProfile | None = None
    read_only_context_fields: tuple[str, ...] = ()

    _attempt: int = field(default=0, init=False)
    _done: bool = field(default=False, init=False)
    _success: bool = field(default=False, init=False)
    _stop_reason: str = field(default="", init=False)
    _steps: list[StepRecord] = field(default_factory=list, init=False)
    _last_feedback: PublicFeedback | None = field(default=None, init=False)
    _current_features: dict[str, Any] = field(default_factory=dict, init=False)
    _episode_locks: dict[str, Any] = field(default_factory=dict, init=False)
    _locks_initialised: bool = field(default=False, init=False)
    _profile_persona_locked: bool = field(default=False, init=False)
    _profile_persona_field: str | None = field(default=None, init=False)
    _profile_persona_value: Any | None = field(default=None, init=False)
    _ledger: BudgetLedger = field(init=False)
    _guarded_defender: GuardedDefender = field(init=False)
    _attempts_to_success: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.max_attempts is None:
            self.max_attempts = self.budget.q_max
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if self.budget.q_max < 1:
            raise ValueError("budget.q_max must be >= 1.")
        if self.starting_case.initial_decision != "BLOCK":
            raise ValueError(
                "Starting case must initially be BLOCKED under the frozen defence."
            )
        self._current_features = dict(self.starting_case.features)
        self._ledger = BudgetLedger(spec=self.budget)
        # Replace any raw defender with a guarded wrapper so D1 cannot be
        # invoked outside step(), even if a caller retained an old reference
        # only through this environment attribute.
        if isinstance(self.defender, GuardedDefender):
            self._guarded_defender = self.defender
        else:
            self._guarded_defender = GuardedDefender(self.defender)
            object.__setattr__(self, "defender", self._guarded_defender)
        self.logger.write_governance_manifest(
            self.validator.policy.manifest_payload()
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success(self) -> bool:
        return self._success

    @property
    def stop_reason(self) -> str:
        """Public terminal label; empty while the episode is active."""
        return self._stop_reason

    @property
    def locked_static_values(self) -> dict[str, Any]:
        """Attacker-chosen episode-static values, safe for local planning."""
        return dict(self._episode_locks)

    @property
    def profile_persona_locked(self) -> bool:
        return self._profile_persona_locked

    @property
    def profile_persona_field(self) -> str | None:
        return self._profile_persona_field

    @property
    def profile_persona_value(self) -> Any | None:
        return self._profile_persona_value

    @property
    def artifact_dir(self):
        """Research artefact destination; contains no observation content."""
        return self.logger.run_dir

    @property
    def attempts_used(self) -> int:
        return self._attempt

    @property
    def ledger(self) -> BudgetLedger:
        return self._ledger

    def remaining_attempts(self) -> int:
        return self._ledger.q_remaining

    def observation(self) -> Observation:
        remaining = self.remaining_attempts()
        locked = set(self._episode_locks) if self._locks_initialised else set()
        mutable_fields = tuple(
            field
            for field in self.validator.mutable_fields
            if field not in locked
        )
        # Per-attempt proxy actions remain available after the first submission.
        # Episode-static proxies disappear once their underlying feature is locked.
        proxy_actions = {
            key: actions
            for key, actions in self.validator.proxy_actions.items()
            if (
                (rule := self.validator.policy.field_for_action(key)) is not None
                and rule.feature not in locked
            )
        }
        instructions = (
            "Submit only governance-listed raw actions or abstract proxy actions. "
            "Commands: submit | reset-current-proposal | show | quit. "
            f"Feedback mode={self.feedback_policy.mode}."
        )
        return Observation(
            case_id=self.starting_case.case_id,
            attempt=self._attempt + 1 if not self._done else self._attempt,
            max_attempts=self.max_attempts,
            visible_fields=self.validator.visible_fields(self._current_features),
            mutable_fields=mutable_fields,
            proxy_actions=proxy_actions,
            feedback_mode=self.feedback_policy.mode,
            instructions=instructions,
            remaining_attempts=remaining,
            q_remaining=self._ledger.q_remaining,
            m_max=self._ledger.m_max,
            e_remaining=self._ledger.e_remaining,
            last_feedback=self._last_feedback,
        )

    def step(self, proposal: AttackProposal) -> StepRecord:
        """Unique submission entry: budget gate -> validate -> optional D1."""
        if self._done:
            raise RuntimeError("Episode already finished.")

        t0 = time.perf_counter()
        self._attempt += 1
        attempt = self._attempt

        pre_feedback_errors: tuple[str, ...] = ()
        if not self._locks_initialised:
            preparation = self.validator.prepare_episode_locks(
                self.starting_case.features, proposal
            )
            self._episode_locks = dict(preparation.locked_values)
            self._locks_initialised = True
            pre_feedback_errors = preparation.errors
            self._current_features.update(self._episode_locks)

        # Edit distance is always measured against the original anchor.
        billing_candidate = self.validator.project_for_billing(
            self.starting_case.features,
            proposal,
            locked_values=self._episode_locks,
        )
        edited, edit_cost, transition_fields, transition_count = compute_edit_metrics(
            anchor=self.starting_case.features,
            candidate=billing_candidate,
            mutable_feature_names=self.validator.mutable_feature_names(),
            previous_candidate=self._ledger.previous_candidate,
        )
        check = self._ledger.precheck(
            submitted_edit_cost=edit_cost,
            edited_fields=edited,
            transition_edit_count=transition_count,
            transition_fields=transition_fields,
        )

        if not check.allowed:
            budget_event = self._ledger.record_budget_rejection(
                attempt=attempt, check=check
            )
            public = self.feedback_policy.for_budget_rejected(
                reason=check.reject_reason or "budget_exhausted",
                attempt=attempt,
                remaining_attempts=self._ledger.q_remaining,
                q_remaining=self._ledger.q_remaining,
                m_max=self._ledger.m_max,
                e_remaining=self._ledger.e_remaining,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record = StepRecord(
                attempt=attempt,
                proposed_changes=dict(proposal.changes),
                validity=ValidityResult(
                    False,
                    (f"Budget rejected: {check.reject_reason}.",),
                    None,
                ),
                candidate_case_id=self.starting_case.case_id,
                internal_defence=None,
                public_feedback=public,
                success=False,
                elapsed_ms=elapsed_ms,
                budget_event=budget_event,
                submitted_edit_cost=edit_cost,
                transition_edit_count=transition_count,
                research_meta=dict(proposal.research_meta),
            )
            self._steps.append(record)
            self._last_feedback = public
            self.logger.append_step(record)
            # q_exhausted ends the episode.  m_exceeded is a per-candidate
            # hard reject; fail-closed stop avoids unbounded over-m retries.
            self._finish(
                success=False,
                reason=check.reject_reason or "budget_exhausted",
            )
            return record

        validity = self.validator.validate(
            self.starting_case.features,
            proposal,
            locked_values=self._episode_locks,
            pre_feedback_errors=pre_feedback_errors,
        )
        # Layered profile check after governance, before D1 scoring.
        if validity.is_valid and self.constraint_profile is not None:
            profile_check = self.constraint_profile.check_edited_features(
                edited,
                candidate_features=validity.candidate_features,
                persona_locked=self._profile_persona_locked,
                locked_persona_field=self._profile_persona_field,
                locked_persona_value=self._profile_persona_value,
                forbidden_fields=self.validator.policy.forbidden_fields,
                read_only_fields=self.read_only_context_fields,
            )
            if not profile_check.is_allowed:
                validity = ValidityResult(
                    False,
                    tuple(profile_check.errors),
                    None,
                )
        internal = None
        success = False
        scored = False

        if validity.is_valid:
            assert validity.candidate_features is not None
            self._current_features = dict(validity.candidate_features)
            if (
                self.constraint_profile is not None
                and not self._profile_persona_locked
            ):
                profile_meta = self.constraint_profile.check_edited_features(
                    edited,
                    candidate_features=validity.candidate_features,
                    persona_locked=False,
                    forbidden_fields=self.validator.policy.forbidden_fields,
                    read_only_fields=self.read_only_context_fields,
                )
                if profile_meta.persona_edited:
                    self._profile_persona_locked = True
                    self._profile_persona_field = profile_meta.persona_edited[0]
                    self._profile_persona_value = validity.candidate_features[
                        self._profile_persona_field
                    ]
            self._guarded_defender._allow_score = True  # noqa: SLF001
            try:
                internal = self._guarded_defender.score_application(
                    validity.candidate_features
                )
            finally:
                self._guarded_defender._allow_score = False  # noqa: SLF001
            scored = True
            public = self.feedback_policy.for_scored(
                internal,
                attempt=attempt,
                remaining_attempts=max(0, self._ledger.q_remaining - 1),
                q_remaining=max(0, self._ledger.q_remaining - 1),
                m_max=self._ledger.m_max,
                e_remaining=self._ledger.m_max,
            )
            if internal.decision == "PASS":
                success = True
                self._attempts_to_success = attempt
        else:
            public = self.feedback_policy.for_invalid(
                validity,
                attempt=attempt,
                remaining_attempts=max(
                    0,
                    self._ledger.q_remaining
                    - (1 if self.budget.invalid_charges_q else 0),
                ),
                q_remaining=max(
                    0,
                    self._ledger.q_remaining
                    - (1 if self.budget.invalid_charges_q else 0),
                ),
                m_max=self._ledger.m_max,
                e_remaining=self._ledger.m_max,
            )

        budget_event = self._ledger.charge_submission(
            attempt=attempt,
            check=check,
            is_valid=validity.is_valid,
            scored=scored,
        )
        # Keep previous billing candidate for transition metrics only.
        self._ledger.note_candidate(billing_candidate)

        # Align public remaining figures with post-charge ledger.
        public = PublicFeedback(
            label=public.label,
            message=public.message,
            attempt=public.attempt,
            remaining_attempts=self._ledger.q_remaining,
            q_remaining=self._ledger.q_remaining,
            m_max=self._ledger.m_max,
            e_remaining=self._ledger.e_remaining,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        record = StepRecord(
            attempt=attempt,
            proposed_changes=dict(proposal.changes),
            validity=validity,
            candidate_case_id=self.starting_case.case_id,
            internal_defence=internal,
            public_feedback=public,
            success=success,
            elapsed_ms=elapsed_ms,
            budget_event=budget_event,
            submitted_edit_cost=edit_cost,
            transition_edit_count=transition_count,
            research_meta=dict(proposal.research_meta),
        )
        self._steps.append(record)
        self._last_feedback = public
        self.logger.append_step(record)

        if success and self.budget.stop_on_success:
            self._finish(success=True, reason="success")
        elif self._ledger.q_remaining < 1:
            self._finish(success=False, reason="q_exhausted")
        # Cumulative edit totals never stop the episode under (Q, m).
        return record

    def abort(self, reason: str = "attacker_stopped") -> None:
        """End the episode without a successful bypass."""
        if not self._done:
            # Preserve attacker-classified local stop reasons; only collapse
            # human-quit aliases into the generic attacker_stopped label.
            if reason == "attacker_quit":
                mapped = "attacker_stopped"
            else:
                mapped = reason
            self._finish(success=False, reason=mapped)

    def _finish(self, *, success: bool, reason: str) -> None:
        self._done = True
        self._success = success
        self._stop_reason = reason

    def result(self) -> EpisodeResult:
        if not self._done:
            raise RuntimeError("Episode is still running.")
        episode = EpisodeResult(
            case_id=self.starting_case.case_id,
            success=self._success,
            attempts_used=self._attempt,
            max_attempts=self.max_attempts,
            stop_reason=self._stop_reason,
            steps=tuple(self._steps),
            q_used=self._ledger.q_used,
            total_edits_used=self._ledger.total_edits_used,
            e_used=self._ledger.e_used,
            scored_defender_queries=self._ledger.scored_defender_queries,
            invalid_submissions=self._ledger.invalid_submissions,
            unique_fields_ever_manipulated=tuple(
                sorted(self._ledger.unique_fields_ever_manipulated)
            ),
            attempts_to_success=self._attempts_to_success,
            budget_spec=self.budget.to_dict(),
        )
        self.logger.write_episode_summary(episode)
        return episode

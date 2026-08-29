"""Narrow attacker-facing episode capability.

The environment owns D1, the threshold, researcher diagnostics and the full
starting-case record.  Attackers receive this facade instead.  Python code in
the same process is not a hostile-code sandbox, but the supported API exposes
only threat-model-permitted state and returns a sanitised public step result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from attack_lab.budget import BudgetSpec
from attack_lab.types import AttackProposal, PublicFeedback, ValidityResult
from attack_lab.validator import OpaqueCandidateAssessment


@dataclass(frozen=True)
class AttackerStartingCase:
    """Attacker-visible anchor only; no label, score or initial D1 result."""

    case_id: str
    features: dict[str, Any]
    data_split: str


@dataclass(frozen=True)
class PublicStepRecord:
    """Sanitised result of a submission; contains no internal defence object."""

    attempt: int
    proposed_changes: dict[str, Any]
    validity: ValidityResult
    public_feedback: PublicFeedback
    success: bool
    submitted_edit_cost: int
    transition_edit_count: int


class AttackerBudgetView:
    """Read-only public budget state."""

    __slots__ = ("__env",)

    def __init__(self, env: Any) -> None:
        self.__env = env

    @property
    def q_remaining(self) -> int:
        return int(self.__env.ledger.q_remaining)

    @property
    def m_max(self) -> int:
        return int(self.__env.ledger.m_max)


class AttackerValidator:
    """Governance-only local workbench with hidden full-anchor plumbing."""

    __slots__ = ("__validator", "__anchor", "policy", "enabled_action_keys")

    def __init__(self, validator: Any, anchor: Mapping[str, Any]) -> None:
        self.__validator = validator
        self.__anchor = dict(anchor)
        # Compiled governance is public game configuration, not a D1 object.
        self.policy = validator.policy
        self.enabled_action_keys = tuple(validator.enabled_action_keys)

    @property
    def mutable_fields(self) -> tuple[str, ...]:
        return tuple(self.__validator.mutable_fields)

    @property
    def proxy_actions(self) -> dict[str, tuple[str, ...]]:
        return dict(self.__validator.proxy_actions)

    def mutable_feature_names(self) -> tuple[str, ...]:
        return tuple(self.__validator.mutable_feature_names())

    def visible_fields(self, features: Mapping[str, Any]) -> dict[str, Any]:
        return dict(self.__validator.visible_fields(dict(features)))

    def prepare_episode_locks(self, _anchor: Mapping[str, Any], proposal: AttackProposal):
        return self.__validator.prepare_episode_locks(self.__anchor, proposal)

    def project_for_billing(
        self,
        _anchor: Mapping[str, Any],
        proposal: AttackProposal,
        *,
        locked_values: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        projected = self.__validator.project_for_billing(
            self.__anchor, proposal, locked_values=locked_values
        )
        return dict(self.__validator.visible_fields(projected))

    def validate(
        self,
        _anchor: Mapping[str, Any],
        proposal: AttackProposal,
        *,
        locked_values: Mapping[str, Any] | None,
        pre_feedback_errors: tuple[str, ...] = (),
    ) -> ValidityResult:
        result = self.__validator.validate(
            self.__anchor,
            proposal,
            locked_values=locked_values,
            pre_feedback_errors=pre_feedback_errors,
        )
        visible_candidate = None
        if result.candidate_features is not None:
            visible_candidate = dict(
                self.__validator.visible_fields(result.candidate_features)
            )
        return ValidityResult(
            is_valid=result.is_valid,
            errors=tuple(result.errors),
            candidate_features=visible_candidate,
        )

    def assess_candidate(
        self,
        _anchor: Mapping[str, Any],
        proposal: AttackProposal,
        *,
        locked_values: Mapping[str, Any] | None,
        pre_feedback_errors: tuple[str, ...] = (),
        anchor_id: str,
        m_max: int,
    ) -> OpaqueCandidateAssessment:
        """Return only opaque local-enforcement facts for one candidate."""

        return self.__validator.assess_candidate(
            self.__anchor,
            proposal,
            locked_values=locked_values,
            pre_feedback_errors=pre_feedback_errors,
            anchor_id=anchor_id,
            m_max=m_max,
        )


class AttackerEpisode:
    """Supported episode API passed to every orchestrated attacker."""

    __slots__ = (
        "__env",
        "starting_case",
        "validator",
        "ledger",
        "budget",
        "artifact_dir",
    )

    def __init__(self, env: Any) -> None:
        self.__env = env
        visible_anchor = env.validator.visible_fields(env.starting_case.features)
        self.starting_case = AttackerStartingCase(
            case_id=str(env.starting_case.case_id),
            features=dict(visible_anchor),
            data_split=str(env.starting_case.data_split),
        )
        self.validator = AttackerValidator(
            env.validator, anchor=env.starting_case.features
        )
        self.ledger = AttackerBudgetView(env)
        self.budget: BudgetSpec = env.budget
        self.artifact_dir: Path = env.logger.run_dir

    @property
    def done(self) -> bool:
        return bool(self.__env.done)

    @property
    def success(self) -> bool:
        return bool(self.__env.success)

    @property
    def attempts_used(self) -> int:
        return int(self.__env.attempts_used)

    @property
    def stop_reason(self) -> str:
        return str(self.__env._stop_reason)  # noqa: SLF001

    @property
    def locked_static_values(self) -> dict[str, Any]:
        return dict(self.__env._episode_locks)  # noqa: SLF001

    @property
    def profile_persona_locked(self) -> bool:
        return bool(self.__env._profile_persona_locked)  # noqa: SLF001

    @property
    def profile_persona_field(self) -> str | None:
        return self.__env._profile_persona_field  # noqa: SLF001

    @property
    def profile_persona_value(self) -> Any | None:
        return self.__env._profile_persona_value  # noqa: SLF001

    def observation(self):
        return self.__env.observation()

    def step(self, proposal: AttackProposal) -> PublicStepRecord:
        step = self.__env.step(proposal)
        visible_candidate = None
        if step.validity.candidate_features is not None:
            visible_candidate = self.validator.visible_fields(
                step.validity.candidate_features
            )
        return PublicStepRecord(
            attempt=step.attempt,
            proposed_changes=dict(step.proposed_changes),
            validity=ValidityResult(
                is_valid=step.validity.is_valid,
                errors=tuple(step.validity.errors),
                candidate_features=visible_candidate,
            ),
            public_feedback=step.public_feedback,
            success=step.success,
            submitted_edit_cost=step.submitted_edit_cost,
            transition_edit_count=step.transition_edit_count,
        )

    def abort(self, reason: str = "attacker_stopped") -> None:
        self.__env.abort(reason=reason)


__all__ = [
    "AttackerBudgetView",
    "AttackerEpisode",
    "AttackerStartingCase",
    "AttackerValidator",
    "PublicStepRecord",
]

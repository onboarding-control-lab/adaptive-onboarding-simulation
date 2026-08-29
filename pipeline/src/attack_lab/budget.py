"""Per-attacker Q,m budget accounting for the attack laboratory.

Frozen experiment protocol:

- ``Q_max``: maximum number of candidate submissions in one episode.
- ``m_max``: maximum number of feature edits allowed in each individual
  candidate, measured against the original anchor.

For every candidate:

    edit_distance(candidate, original_anchor) <= m_max

Edit cost is never accumulated across candidates for admission control.
``total_edits_used`` is retained only for reporting / efficiency metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

StopReason = Literal[
    "success",
    "q_exhausted",
    "m_exceeded",
    "e_exhausted",  # legacy alias retained for archived artefacts
    "insufficient_edit_budget",
    "no_feasible_candidate",
    "local_generation_exhausted",
    "action_space_exhaustion",
    "attacker_stopped",
    "invalid_environment",
    "policy_error",
    "bypass_pass",
    "budget_exhausted",
    "attacker_quit",
]


@dataclass(frozen=True)
class AttackBudget:
    """Attacker-facing ``(Q, m)`` budget interface (not hard-coded in A2).

    Core attackers must read ``q_max`` / ``m_max`` from this object (or an
    equivalent BudgetSpec built from it).  Pilot values such as m=2 / Q=5 are
    supplied by CLI/runners only.
    """

    q_max: int
    m_max: int

    def __post_init__(self) -> None:
        if self.q_max < 1:
            raise ValueError("q_max must be >= 1.")
        if self.m_max < 0:
            raise ValueError("m_max must be >= 0.")

    def to_budget_spec(
        self,
        *,
        label: str = "attack_budget_via_interface",
    ) -> "BudgetSpec":
        return BudgetSpec(q_max=int(self.q_max), m_max=int(self.m_max), label=label)

    def to_dict(self) -> dict[str, Any]:
        return {"q_max": int(self.q_max), "m_max": int(self.m_max)}


@dataclass(frozen=True)
class BudgetSpec:
    """Immutable per-attacker ``(Q, m)`` budget configuration.

    Formal scientific ``Q_max`` / ``m_max`` values are not frozen here.
    Tests and development runs must supply explicitly labelled dummy budgets.
    """

    q_max: int
    m_max: int
    invalid_charges_q: bool = True
    invalid_charges_proposed_m: bool = True
    stop_on_success: bool = True
    label: str = "development_dummy_budget_not_final_scientific_freeze"

    def __post_init__(self) -> None:
        if self.q_max < 1:
            raise ValueError("q_max must be >= 1.")
        if self.m_max < 0:
            raise ValueError("m_max must be >= 0.")

    @classmethod
    def development_dummy(
        cls,
        *,
        q_max: int,
        m_max: int | None = None,
        e_max: int | None = None,
        label: str = "development_dummy_budget_not_final_scientific_freeze",
    ) -> "BudgetSpec":
        """Construct an explicitly labelled non-final dummy budget.

        ``e_max`` is accepted only as a deprecated alias for ``m_max`` so that
        archived runners/tests can be updated gradually.
        """
        if m_max is None and e_max is None:
            raise ValueError("development_dummy requires m_max (or legacy e_max).")
        if m_max is not None and e_max is not None and int(m_max) != int(e_max):
            raise ValueError(
                "development_dummy received conflicting m_max and legacy e_max."
            )
        resolved_m = int(m_max if m_max is not None else e_max)  # type: ignore[arg-type]
        return cls(q_max=q_max, m_max=resolved_m, label=label)

    @property
    def e_max(self) -> int:
        """Deprecated alias for ``m_max`` (archived artefact compatibility)."""
        return self.m_max

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep a deprecated mirror key for older summary readers.
        payload["e_max"] = self.m_max
        return payload


@dataclass(frozen=True)
class BudgetCheckResult:
    """Outcome of a pre-submission budget gate."""

    allowed: bool
    submitted_edit_cost: int
    edited_fields: tuple[str, ...]
    transition_edit_count: int
    transition_fields: tuple[str, ...]
    reject_reason: str | None = None


@dataclass(frozen=True)
class BudgetEvent:
    """One ledger event for a charged or rejected submission."""

    attempt: int
    submitted_edit_cost: int
    transition_edit_count: int
    edited_fields: tuple[str, ...]
    transition_fields: tuple[str, ...]
    q_charged: int
    m_charged: int
    q_used: int
    total_edits_used: int
    q_remaining: int
    m_max: int
    scored_defender_query: bool
    invalid_submission: bool
    budget_rejected: bool
    reject_reason: str | None = None

    @property
    def e_charged(self) -> int:
        """Deprecated alias for ``m_charged``."""
        return self.m_charged

    @property
    def e_used(self) -> int:
        """Deprecated alias for ``total_edits_used``."""
        return self.total_edits_used

    @property
    def e_remaining(self) -> int:
        """Deprecated: per-candidate m is not cumulative; reports ``m_max``."""
        return self.m_max


@dataclass
class BudgetLedger:
    """Mutable per-episode ``(Q, m)`` ledger owned only by AttackEnvironment."""

    spec: BudgetSpec
    q_used: int = 0
    total_edits_used: int = 0
    edits_per_candidate: list[int] = field(default_factory=list)
    scored_defender_queries: int = 0
    invalid_submissions: int = 0
    unique_fields_ever_manipulated: set[str] = field(default_factory=set)
    events: list[BudgetEvent] = field(default_factory=list)
    _previous_candidate: dict[str, Any] | None = field(default=None, init=False)

    @property
    def q_remaining(self) -> int:
        return max(0, self.spec.q_max - self.q_used)

    @property
    def m_max(self) -> int:
        return self.spec.m_max

    @property
    def e_used(self) -> int:
        """Deprecated alias for ``total_edits_used`` (reporting only)."""
        return self.total_edits_used

    @property
    def e_remaining(self) -> int:
        """Deprecated: not a cumulative remaining budget; equals ``m_max``."""
        return self.spec.m_max

    def snapshot(self) -> dict[str, Any]:
        return {
            "budget_spec": self.spec.to_dict(),
            "q_used": self.q_used,
            "total_edits_used": self.total_edits_used,
            "edits_per_candidate": list(self.edits_per_candidate),
            "q_remaining": self.q_remaining,
            "m_max": self.spec.m_max,
            # Deprecated mirrors for archived readers.
            "e_used": self.total_edits_used,
            "e_remaining": self.spec.m_max,
            "scored_defender_queries": self.scored_defender_queries,
            "invalid_submissions": self.invalid_submissions,
            "unique_fields_ever_manipulated": sorted(
                self.unique_fields_ever_manipulated
            ),
            "events": [asdict(event) for event in self.events],
        }

    def precheck(
        self,
        *,
        submitted_edit_cost: int,
        edited_fields: tuple[str, ...],
        transition_edit_count: int,
        transition_fields: tuple[str, ...],
    ) -> BudgetCheckResult:
        """Fail-closed gate before validator/D1.  Does not mutate the ledger.

        Rejects only when Q is exhausted or the candidate's anchor-relative
        edit distance exceeds ``m_max``.  Cumulative edit totals never reject.
        """
        if self.q_remaining < 1:
            return BudgetCheckResult(
                allowed=False,
                submitted_edit_cost=submitted_edit_cost,
                edited_fields=edited_fields,
                transition_edit_count=transition_edit_count,
                transition_fields=transition_fields,
                reject_reason="q_exhausted",
            )
        if submitted_edit_cost > self.spec.m_max:
            return BudgetCheckResult(
                allowed=False,
                submitted_edit_cost=submitted_edit_cost,
                edited_fields=edited_fields,
                transition_edit_count=transition_edit_count,
                transition_fields=transition_fields,
                reject_reason="m_exceeded",
            )
        return BudgetCheckResult(
            allowed=True,
            submitted_edit_cost=submitted_edit_cost,
            edited_fields=edited_fields,
            transition_edit_count=transition_edit_count,
            transition_fields=transition_fields,
            reject_reason=None,
        )

    def charge_submission(
        self,
        *,
        attempt: int,
        check: BudgetCheckResult,
        is_valid: bool,
        scored: bool,
    ) -> BudgetEvent:
        """Apply Q charging and record per-candidate edit cost after precheck."""
        if not check.allowed:
            raise RuntimeError("Cannot charge a budget-rejected submission.")

        if is_valid:
            q_charge = 1
            m_charge = check.submitted_edit_cost
        else:
            q_charge = 1 if self.spec.invalid_charges_q else 0
            m_charge = (
                check.submitted_edit_cost
                if self.spec.invalid_charges_proposed_m
                else 0
            )
            self.invalid_submissions += 1

        if q_charge > self.q_remaining:
            raise RuntimeError("Budget charge would exceed remaining Q allowance.")
        if check.submitted_edit_cost > self.spec.m_max:
            raise RuntimeError("Budget charge would exceed per-candidate m_max.")

        self.q_used += q_charge
        if q_charge > 0 or m_charge > 0:
            self.edits_per_candidate.append(int(check.submitted_edit_cost))
            self.total_edits_used += int(m_charge)
        if scored:
            if not is_valid:
                raise RuntimeError(
                    "Invalid submissions must not increment scored_defender_queries."
                )
            self.scored_defender_queries += 1
        self.unique_fields_ever_manipulated.update(check.edited_fields)

        event = BudgetEvent(
            attempt=attempt,
            submitted_edit_cost=check.submitted_edit_cost,
            transition_edit_count=check.transition_edit_count,
            edited_fields=check.edited_fields,
            transition_fields=check.transition_fields,
            q_charged=q_charge,
            m_charged=m_charge,
            q_used=self.q_used,
            total_edits_used=self.total_edits_used,
            q_remaining=self.q_remaining,
            m_max=self.spec.m_max,
            scored_defender_query=scored,
            invalid_submission=not is_valid,
            budget_rejected=False,
            reject_reason=None,
        )
        self.events.append(event)
        return event

    def record_budget_rejection(
        self,
        *,
        attempt: int,
        check: BudgetCheckResult,
    ) -> BudgetEvent:
        """Record a refused submission that did not charge Q or call D1."""
        event = BudgetEvent(
            attempt=attempt,
            submitted_edit_cost=check.submitted_edit_cost,
            transition_edit_count=check.transition_edit_count,
            edited_fields=check.edited_fields,
            transition_fields=check.transition_fields,
            q_charged=0,
            m_charged=0,
            q_used=self.q_used,
            total_edits_used=self.total_edits_used,
            q_remaining=self.q_remaining,
            m_max=self.spec.m_max,
            scored_defender_query=False,
            invalid_submission=False,
            budget_rejected=True,
            reject_reason=check.reject_reason,
        )
        self.events.append(event)
        return event

    def note_candidate(self, candidate: Mapping[str, Any]) -> None:
        self._previous_candidate = dict(candidate)

    @property
    def previous_candidate(self) -> Mapping[str, Any] | None:
        return self._previous_candidate


def values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        import pandas as pd

        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def compute_edit_metrics(
    *,
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mutable_feature_names: tuple[str, ...],
    previous_candidate: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], int, tuple[str, ...], int]:
    """Compare a projected candidate with the anchor and previous candidate.

    ``submitted_edit_cost`` counts mutable fields differing from the original
    anchor.  Transition metrics versus the previous candidate are informational
    only and do not admit or reject under the ``(Q, m)`` protocol.
    """
    edited = tuple(
        sorted(
            name
            for name in mutable_feature_names
            if name in candidate
            and name in anchor
            and not values_equal(candidate[name], anchor[name])
        )
    )
    prior = previous_candidate if previous_candidate is not None else anchor
    transition = tuple(
        sorted(
            name
            for name in mutable_feature_names
            if name in candidate
            and name in prior
            and not values_equal(candidate[name], prior[name])
        )
    )
    return edited, len(edited), transition, len(transition)

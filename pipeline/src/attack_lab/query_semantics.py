"""Explicit transport-retry versus attack-query (Q) semantics.

Transport retries recover API/network/schema transport failures. They are
not strategic attacker observations and must not consume Q.

An attack query is a submitted application that receives PASS, BLOCK, or
INVALID feedback. Transport and local parse/generation retries must not
consume Q. Under the frozen final Q contract A, only a valid candidate
actually submitted to D1 consumes one unit of Q; INVALID is logged and
does not consume Q. BudgetSpec.invalid_charges_q encodes that rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from attack_lab.budget import BudgetLedger

TransportKind = Literal[
    "transport_retry",
    "schema_transport_recovery",
    "parse_local_generation",
    "attack_submission",
    "defence_feedback_regeneration",
    "api_failure_terminal",
]

TRANSPORT_RETRY_REASONS = frozenset({"timeout", "transport_error"})
PARSE_LOCAL_REASONS = frozenset({"empty", "parse_error", "schema_error", "local_validation_failed"})
STRATEGIC_FEEDBACK_LABELS = frozenset({"PASS", "BLOCK", "INVALID"})
API_FAILURE_STOP_REASONS = frozenset(
    {"transport_error", "timeout", "rate_limit", "runner_exception"}
)
ATTACK_FAILURE_STOP_REASONS = frozenset(
    {
        "success",
        "q_exhausted",
        "m_exceeded",
        "no_feasible_candidate",
        "local_generation_exhausted",
        "attacker_stopped",
    }
)


class QuerySemanticsError(RuntimeError):
    """Raised when retry/Q accounting would violate the frozen contract."""


def charges_q(kind: TransportKind) -> bool:
    """Return whether this event consumes a strategic attack query."""
    return kind in {"attack_submission", "defence_feedback_regeneration"}


def is_transport_retry(reason: str | None) -> bool:
    return str(reason or "") in TRANSPORT_RETRY_REASONS


def is_api_failure(stop_reason: str | None) -> bool:
    return str(stop_reason or "") in API_FAILURE_STOP_REASONS


def classify_event(
    *,
    env_step_called: bool,
    defence_feedback_received: bool,
    transport_error: str | None,
    parse_status: str | None,
) -> TransportKind:
    """Classify one LLM/local-generation/submission event."""
    if env_step_called:
        if defence_feedback_received:
            return "defence_feedback_regeneration"
        return "attack_submission"
    if is_transport_retry(parse_status) or transport_error:
        return "transport_retry"
    if parse_status in PARSE_LOCAL_REASONS:
        return "parse_local_generation"
    return "schema_transport_recovery"


def assert_transport_retry_does_not_charge_q(
    ledger_before: BudgetLedger,
    ledger_after: BudgetLedger,
) -> None:
    """Fail closed if a transport/local-generation retry consumed Q."""
    if int(ledger_after.q_used) != int(ledger_before.q_used):
        raise QuerySemanticsError(
            "Transport/local-generation retry consumed Q "
            f"({ledger_before.q_used} -> {ledger_after.q_used})."
        )
    if int(ledger_after.scored_defender_queries) != int(
        ledger_before.scored_defender_queries
    ):
        raise QuerySemanticsError(
            "Transport retry called D1 / scored a defender query."
        )


def assert_submission_charges_q(
    ledger_before: BudgetLedger,
    ledger_after: BudgetLedger,
    *,
    expected_charge: int = 1,
) -> None:
    """Fail closed unless a real submission charged Q."""
    delta = int(ledger_after.q_used) - int(ledger_before.q_used)
    if delta != int(expected_charge):
        raise QuerySemanticsError(
            f"Attack submission Q charge {delta} != expected {expected_charge}."
        )


@dataclass(frozen=True)
class RetryPolicy:
    """Frozen API/transport retry policy for the final runner."""

    max_transport_retries_per_call: int
    timeout_seconds: float
    transport_retry_does_not_charge_q: bool = True
    defence_feedback_regeneration_charges_q: bool = True
    parse_local_generation_does_not_charge_q: bool = True

    def __post_init__(self) -> None:
        if self.max_transport_retries_per_call < 0:
            raise QuerySemanticsError("max_transport_retries_per_call must be >= 0.")
        if self.timeout_seconds <= 0:
            raise QuerySemanticsError("timeout_seconds must be positive.")
        if not self.transport_retry_does_not_charge_q:
            raise QuerySemanticsError(
                "Final protocol requires transport retries not to charge Q."
            )
        if not self.defence_feedback_regeneration_charges_q:
            raise QuerySemanticsError(
                "Final protocol requires post-feedback regeneration to charge Q."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_transport_retries_per_call": int(self.max_transport_retries_per_call),
            "timeout_seconds": float(self.timeout_seconds),
            "transport_retry_does_not_charge_q": True,
            "defence_feedback_regeneration_charges_q": True,
            "parse_local_generation_does_not_charge_q": True,
        }


__all__ = [
    "API_FAILURE_STOP_REASONS",
    "ATTACK_FAILURE_STOP_REASONS",
    "PARSE_LOCAL_REASONS",
    "QuerySemanticsError",
    "RetryPolicy",
    "STRATEGIC_FEEDBACK_LABELS",
    "TRANSPORT_RETRY_REASONS",
    "assert_submission_charges_q",
    "assert_transport_retry_does_not_charge_q",
    "charges_q",
    "classify_event",
    "is_api_failure",
    "is_transport_retry",
]

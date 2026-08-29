"""First successful D1-PASS extraction from saved attack trajectories.

D2 variants operate offline on these extracted submissions. Attackers are
not rerun for D2-S v1.0, D2-S v1.1, or exploratory D1-R.
"""

from __future__ import annotations

from typing import Any, Mapping


class FirstSuccessError(RuntimeError):
    """Raised when a successful episode has no extractable D1-PASS step."""


def extract_first_successful_d1_pass(
    episode: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the first valid D1-PASS candidate, or None if none exists."""
    for step in episode.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        defence = step.get("internal_defence") or {}
        validity = step.get("validity") or {}
        if defence.get("decision") != "PASS":
            continue
        if validity.get("is_valid") is not True:
            continue
        features = validity.get("candidate_features")
        if not isinstance(features, Mapping):
            continue
        return {
            "attempt": step.get("attempt"),
            "features": dict(features),
            "decision": "PASS",
        }
    return None


def require_first_successful_d1_pass(episode: Mapping[str, Any]) -> dict[str, Any]:
    found = extract_first_successful_d1_pass(episode)
    if found is None:
        raise FirstSuccessError("Successful episode has no valid D1-PASS step.")
    return found


__all__ = [
    "FirstSuccessError",
    "extract_first_successful_d1_pass",
    "require_first_successful_d1_pass",
]

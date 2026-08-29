"""Canonical candidate identity shared by A0--A3.

Scientific duplicate identity is defined by the projected attacker-action
feature state for one anchor.  Raw action syntax is deliberately excluded:
two proposals that resolve to the same submitted application are duplicates.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from attack_lab.types import to_jsonable

CANDIDATE_IDENTITY_VERSION = "projected-action-state-v1"


def canonical_candidate_fingerprint(
    *,
    anchor_id: str,
    projected_candidate: Mapping[str, Any],
    action_fields: Iterable[str] | None = None,
) -> str:
    """Return the experiment-wide identity of one projected candidate.

    ``action_fields`` should be supplied by production attackers so read-only
    context and representation-only raw actions cannot affect identity.  The
    optional default is useful for already-projected unit-test fixtures.
    """

    if action_fields is None:
        names = sorted(str(name) for name in projected_candidate)
    else:
        names = sorted(
            str(name) for name in action_fields if str(name) in projected_candidate
        )
    canonical = {name: projected_candidate[name] for name in names}
    payload = {
        "identity_version": CANDIDATE_IDENTITY_VERSION,
        "anchor_id": str(anchor_id),
        "projected_action_state": to_jsonable(canonical),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["CANDIDATE_IDENTITY_VERSION", "canonical_candidate_fingerprint"]

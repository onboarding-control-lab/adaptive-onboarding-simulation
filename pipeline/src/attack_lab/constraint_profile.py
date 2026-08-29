"""Independent candidate constraint profiles layered on attack-governance-v2.

Profiles do not replace governance.  They only further restrict which
governance-legal candidates are eligible under a named experimental condition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack_lab.types import to_jsonable

DEFAULT_IDENTITY_COMPOSITION_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "constraint_profiles"
    / "identity_composition_proxy_v1.json"
)


class ConstraintProfileError(RuntimeError):
    """Raised when a constraint profile cannot be loaded or applied."""


@dataclass(frozen=True)
class ProfileCheckResult:
    """Outcome of an identity-composition (or similar) eligibility check."""

    is_allowed: bool
    errors: tuple[str, ...] = ()
    persona_edited: tuple[str, ...] = ()
    contact_edited: tuple[str, ...] = ()
    other_edited: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "errors": list(self.errors),
            "persona_edited": list(self.persona_edited),
            "contact_edited": list(self.contact_edited),
            "other_edited": list(self.other_edited),
        }


@dataclass(frozen=True)
class IdentityCompositionProfile:
    """identity-composition-proxy candidate eligibility profile."""

    profile_version: str
    label: str
    inherits_governance_version: str
    required_edits: int
    persona_profile_fields: tuple[str, ...]
    contact_identity_fields: tuple[str, ...]
    source_path: str
    profile_fingerprint: str
    description: tuple[str, ...] = ()
    status: str = "experimental"

    @classmethod
    def load(cls, path: Path | None = None) -> "IdentityCompositionProfile":
        profile_path = path or DEFAULT_IDENTITY_COMPOSITION_PROFILE
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        persona = tuple(payload["persona_profile_fields"])
        contact = tuple(payload["contact_identity_fields"])
        if set(persona) & set(contact):
            raise ConstraintProfileError(
                "persona and contact field sets must be disjoint."
            )
        required = int(payload.get("required_edits", 2))
        if required != 2:
            raise ConstraintProfileError(
                "identity-composition-proxy requires required_edits=2."
            )
        canonical = {
            "profile_version": payload["profile_version"],
            "inherits_governance_version": payload["inherits_governance_version"],
            "required_edits": required,
            "persona_profile_fields": list(persona),
            "contact_identity_fields": list(contact),
            "rules": list(payload.get("rules", ())),
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            profile_version=str(payload["profile_version"]),
            label=str(payload.get("label", "identity_composition_proxy")),
            inherits_governance_version=str(payload["inherits_governance_version"]),
            required_edits=required,
            persona_profile_fields=persona,
            contact_identity_fields=contact,
            source_path=str(profile_path),
            profile_fingerprint=fingerprint,
            description=tuple(payload.get("description", ())),
            status=str(payload.get("status", "experimental")),
        )

    def composite_experiment_fingerprint(
        self, *, governance_fingerprint: str
    ) -> str:
        payload = {
            "governance_fingerprint": governance_fingerprint,
            "constraint_profile_fingerprint": self.profile_fingerprint,
            "profile_version": self.profile_version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def public_view(
        self,
        *,
        persona_locked: bool = False,
        locked_persona_field: str | None = None,
        locked_persona_value: Any = None,
        queries_remaining: int | None = None,
        q_max: int | None = None,
        m_max: int | None = None,
        submitted_hashes: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Attacker-visible profile rules and episode lock state."""
        return {
            "profile_version": self.profile_version,
            "profile_fingerprint": self.profile_fingerprint,
            "inherits_governance_version": self.inherits_governance_version,
            "persona_profile_fields": list(self.persona_profile_fields),
            "contact_identity_fields": list(self.contact_identity_fields),
            "required_composition": "exactly_1_persona_and_1_contact",
            "required_edits": self.required_edits,
            "episode_state": {
                "persona_locked": bool(persona_locked),
                "locked_persona_field": locked_persona_field,
                "locked_persona_value": to_jsonable(locked_persona_value),
                "queries_remaining": queries_remaining,
                "q_max": q_max,
                "m_max": m_max,
                "submitted_candidate_hashes": list(submitted_hashes),
            },
            "explicitly_hidden": [
                "d1_risk_score",
                "d1_threshold",
                "model_parameters",
                "shap_or_feature_importance",
                "true_rejection_reason",
                "fraud_bool",
                "month7_data",
                "other_anchor_history",
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "label": self.label,
            "status": self.status,
            "inherits_governance_version": self.inherits_governance_version,
            "required_edits": self.required_edits,
            "persona_profile_fields": list(self.persona_profile_fields),
            "contact_identity_fields": list(self.contact_identity_fields),
            "description": list(self.description),
            "source_path": self.source_path,
            "profile_fingerprint": self.profile_fingerprint,
        }

    def check_edited_features(
        self,
        edited_features: Sequence[str],
        *,
        candidate_features: Mapping[str, Any] | None = None,
        persona_locked: bool = False,
        locked_persona_field: str | None = None,
        locked_persona_value: Any = None,
        forbidden_fields: Sequence[str] = (),
        read_only_fields: Sequence[str] = (),
    ) -> ProfileCheckResult:
        """Validate feature-level edits against the composition profile."""
        edited = tuple(dict.fromkeys(str(name) for name in edited_features))
        persona_set = set(self.persona_profile_fields)
        contact_set = set(self.contact_identity_fields)
        forbidden = set(forbidden_fields)
        read_only = set(read_only_fields)

        persona_edited = tuple(name for name in edited if name in persona_set)
        contact_edited = tuple(name for name in edited if name in contact_set)
        other_edited = tuple(
            name
            for name in edited
            if name not in persona_set and name not in contact_set
        )
        errors: list[str] = []

        if any(name in forbidden for name in edited):
            errors.append("Profile rejected: forbidden field edited.")
        if any(name in read_only for name in edited):
            errors.append("Profile rejected: read-only context field edited.")
        if len(edited) != self.required_edits:
            errors.append(
                f"Profile rejected: expected exactly {self.required_edits} edits, "
                f"got {len(edited)}."
            )
        if len(persona_edited) != 1:
            errors.append(
                "Profile rejected: expected exactly one persona/profile edit, "
                f"got {len(persona_edited)}."
            )
        if len(contact_edited) != 1:
            errors.append(
                "Profile rejected: expected exactly one contact/identity edit, "
                f"got {len(contact_edited)}."
            )
        if other_edited:
            errors.append(
                "Profile rejected: non-profile action fields edited: "
                f"{list(other_edited)}."
            )

        if persona_locked:
            if locked_persona_field is None:
                errors.append(
                    "Profile rejected: persona lock is active but locked field "
                    "is missing."
                )
            else:
                if list(persona_edited) != [locked_persona_field]:
                    errors.append(
                        "Profile rejected: persona field must remain "
                        f"{locked_persona_field!r} after first submission."
                    )
                if (
                    candidate_features is not None
                    and locked_persona_field in candidate_features
                    and not _values_equal(
                        candidate_features[locked_persona_field],
                        locked_persona_value,
                    )
                ):
                    errors.append(
                        "Profile rejected: locked persona value must be retained."
                    )

        return ProfileCheckResult(
            is_allowed=not errors,
            errors=tuple(dict.fromkeys(errors)),
            persona_edited=persona_edited,
            contact_edited=contact_edited,
            other_edited=other_edited,
        )

    def is_compatible_static_lock_plan(
        self, static_edited_features: Sequence[str]
    ) -> bool:
        """A0 shared lock plans under this profile must edit exactly one persona."""
        edited = [str(name) for name in static_edited_features]
        if not edited:
            return False
        persona = [name for name in edited if name in self.persona_profile_fields]
        other = [name for name in edited if name not in self.persona_profile_fields]
        return len(persona) == 1 and not other


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
    "ConstraintProfileError",
    "IdentityCompositionProfile",
    "ProfileCheckResult",
    "DEFAULT_IDENTITY_COMPOSITION_PROFILE",
]

"""A1 V3 prompt/contract helpers: single action_key vocabulary + safe pool view.

Historical V1/V2 builders remain in ``a1_planner``.  This module is V3-only.
Trusted resolution still uses the real ReferencePool; the LLM-facing view is
sanitised and must not expose proxy raw targets as writable names or values.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from attack_lab.governance_view import ActionFieldPublicRule, GovernanceView
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import to_jsonable

PROMPT_VERSION_V3 = "a1_oneshot_v3_k10_single_vocab_slot_repair"

# Underlying proxy raw features: trusted-code-only; never writable in V3 prompts.
PROXY_RAW_FEATURE_NAMES = frozenset(
    {
        "name_email_similarity",
        "phone_home_valid",
        "phone_mobile_valid",
    }
)

DIVERSIFICATION_PRINCIPLE_V3 = (
    "Because the full plan must be created before any feedback is available, "
    "construct a diversified ordered portfolio of exactly q_max candidates. "
    "Where feasible, candidates should test materially different hypotheses by "
    "varying action_key combinations or strategy families. Each candidate must "
    "still satisfy all governance and budget constraints. Do not infer or target "
    "any hidden model score, threshold, or decision boundary."
)

STATIC_LOCK_PRINCIPLE_V3 = (
    "Episode-static action_keys are fixed for the whole portfolio. "
    "Candidate 1 establishes the episode-static lock plan: any episode-static "
    "action_key omitted in candidate 1 locks to the anchor value for all later "
    "candidates. Later candidates must not introduce a different episode-static "
    "value. Use only action_key names from allowed_action_keys."
)


def compact_action_rule_v3(rule: ActionFieldPublicRule) -> dict[str, Any]:
    """Public V3 action card: action_key only (no competing feature name)."""
    payload: dict[str, Any] = {
        "action_key": rule.action_key,
        "category": rule.category,
        "data_type": rule.data_type,
        "counts_toward_edit_budget": rule.counts_toward_edit_budget,
    }
    if rule.lower_bound is not None:
        payload["lower_bound"] = rule.lower_bound
    if rule.upper_bound is not None:
        payload["upper_bound"] = rule.upper_bound
    if rule.allowed_values:
        payload["allowed_values_note"] = (
            "Governance may reject illegal resolved values; "
            "do not emit these as change values — use reference_id only."
        )
    return payload


def action_roles_from_view(
    view: GovernanceView,
    *,
    enabled_action_keys: Sequence[str],
) -> dict[str, Any]:
    """Derive writable action_key roles from governance view + enabled set."""
    enabled = {str(k) for k in enabled_action_keys}
    per_attempt: list[str] = []
    episode_static: list[str] = []
    for rule in view.action_field_rules:
        if rule.action_key not in enabled:
            continue
        if rule.category == "episode_static":
            episode_static.append(rule.action_key)
        else:
            per_attempt.append(rule.action_key)
    return {
        "per_attempt_action_keys": per_attempt,
        "episode_static_action_keys": episode_static,
        "forbidden_fields": list(view.forbidden_fields),
        "read_only_context_fields": list(view.read_only_context_fields),
    }


def safe_a1_v3_reference_pool_view(
    pool: ReferencePool,
    *,
    allowed_visible_fields: Sequence[str],
) -> dict[str, Any]:
    """LLM-safe pool view: stable reference_ids, no proxy raw field values.

    Keeps original profile_id tokens for trusted resolution.  Profile field
    maps retain only attacker-visible allowlisted fields.  Does not rename IDs.
    """
    allowed = {str(name) for name in allowed_visible_fields}
    allowed -= PROXY_RAW_FEATURE_NAMES
    profiles: list[dict[str, Any]] = []
    for profile in pool.profiles:
        fields = {
            str(key): value
            for key, value in dict(profile.fields).items()
            if str(key) in allowed
        }
        profiles.append(
            {
                "profile_id": profile.profile_id,
                "fields": to_jsonable(fields),
            }
        )
    return {
        "K": int(pool.K),
        "anchor_id": pool.anchor_id,
        "pool_fingerprint": pool.pool_fingerprint,
        "allowed_reference_ids": [profile.profile_id for profile in pool.profiles],
        "read_only_context_fields": [
            name for name in pool.read_only_context_fields if name in allowed
        ],
        "profiles": profiles,
    }


def assert_v3_prompt_single_vocab(payload: Mapping[str, Any]) -> None:
    """Fail closed if proxy raw names appear as writable vocabulary."""
    text = str(to_jsonable(dict(payload)))
    action_roles = payload.get("action_roles") or {}
    for key in ("per_attempt_action_keys", "episode_static_action_keys"):
        values = action_roles.get(key) or []
        overlap = sorted(set(values).intersection(PROXY_RAW_FEATURE_NAMES))
        if overlap:
            raise ValueError(
                f"V3 action_roles.{key} exposes proxy raw names: {overlap}."
            )
    allowed = payload.get("allowed_action_keys") or []
    overlap = sorted(set(allowed).intersection(PROXY_RAW_FEATURE_NAMES))
    if overlap:
        raise ValueError(f"V3 allowed_action_keys exposes proxy raw names: {overlap}.")
    for rule in payload.get("action_catalogue") or []:
        if not isinstance(rule, Mapping):
            continue
        if "feature" in rule:
            raise ValueError("V3 action_catalogue must not expose 'feature'.")
        if str(rule.get("action_key")) in PROXY_RAW_FEATURE_NAMES:
            raise ValueError("V3 action_catalogue action_key is a proxy raw name.")
        if "proxy_actions" in rule or "proxy_action_key" in rule:
            raise ValueError(
                "V3 action_catalogue must not expose proxy_actions vocabulary."
            )
    pool = payload.get("reference_pool") or {}
    if "action_fields" in pool:
        raise ValueError("V3 reference_pool must not expose action_fields.")
    for profile in pool.get("profiles") or ():
        if not isinstance(profile, Mapping):
            continue
        fields = profile.get("fields") or {}
        overlap = sorted(set(fields).intersection(PROXY_RAW_FEATURE_NAMES))
        if overlap:
            raise ValueError(
                f"V3 reference profile exposes proxy raw fields: {overlap}."
            )
    for name in PROXY_RAW_FEATURE_NAMES:
        if name in text:
            raise ValueError(
                f"V3 prompt payload unexpectedly contains proxy raw name {name!r}."
            )


def build_v3_candidate_item_schema(
    *,
    allowed_action_keys: Sequence[str],
    allowed_reference_ids: Sequence[str],
    m_max: int,
) -> dict[str, Any]:
    """JSON-schema-like object for one V3 candidate (prompt-facing)."""
    return {
        "type": "object",
        "required": ["strategy_label", "changes"],
        "additionalProperties": False,
        "properties": {
            "strategy_label": {"type": "string"},
            "changes": {
                "type": "object",
                "description": (
                    "Map of action_key -> {\"reference_id\": \"...\"} only. "
                    "action_key must be from allowed_action_keys; "
                    "reference_id must be from allowed_reference_ids."
                ),
                "minProperties": 1,
                "maxProperties": int(m_max),
                "propertyNames": {"enum": list(allowed_action_keys)},
                "additionalProperties": {
                    "type": "object",
                    "required": ["reference_id"],
                    "additionalProperties": False,
                    "properties": {
                        "reference_id": {
                            "type": "string",
                            "enum": list(allowed_reference_ids),
                        }
                    },
                },
            },
        },
    }


def build_v3_budget_notes(*, q_max: int, m_max: int) -> list[str]:
    return [
        f"Return exactly {int(q_max)} candidates (q_max={int(q_max)}), ordered for submission.",
        f"Each candidate must change between 1 and {int(m_max)} action_keys "
        "relative to the original anchor.",
        "Writable vocabulary is allowed_action_keys only (never raw feature names).",
        "Every change value must be {\"reference_id\": ...} from allowed_reference_ids.",
        STATIC_LOCK_PRINCIPLE_V3,
        "Candidates must be unique.",
        "Local slot repair may replace only invalid candidate indices before any "
        "defender query; it does not consume Q.",
    ]


__all__ = [
    "DIVERSIFICATION_PRINCIPLE_V3",
    "PROMPT_VERSION_V3",
    "PROXY_RAW_FEATURE_NAMES",
    "STATIC_LOCK_PRINCIPLE_V3",
    "action_roles_from_view",
    "assert_v3_prompt_single_vocab",
    "build_v3_budget_notes",
    "build_v3_candidate_item_schema",
    "compact_action_rule_v3",
    "safe_a1_v3_reference_pool_view",
]

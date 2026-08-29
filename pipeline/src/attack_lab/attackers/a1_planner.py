"""A1 — non-adaptive one-shot LLM planner under the Q,m protocol.

A1 calls an LLM before any defender feedback is available, freezes an
ordered sequence of at most ``Q`` unique K-pool-backed candidates, then
submits that sequence without revision.  PASS / BLOCK / INVALID never
regenerates, reorders or replaces unsubmitted candidates.

Malformed or rule-violating LLM output is handled as a LOCAL GENERATION
ATTEMPT only: it must not call ``env.step`` / D1, must not consume Q, and
may regenerate within a fixed local-generation cap with a safe error
summary.  V3 performs slot-level repair only before freeze; it is rule
compliance only, not defender adaptation.

Shared inputs with A0–A3: anchor, K-reference pool, governance policy, and
the ``(Q, m)`` budget.  The model never receives D1 scores, thresholds,
feature importance or gradients.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TextIO

from attack_lab.attackers.a0_random import derive_episode_seed
from attack_lab.archive.contracts.a1_v3_contract import (
    DIVERSIFICATION_PRINCIPLE_V3,
    PROMPT_VERSION_V3,
    PROXY_RAW_FEATURE_NAMES,
    STATIC_LOCK_PRINCIPLE_V3,
    action_roles_from_view,
    assert_v3_prompt_single_vocab,
    build_v3_budget_notes,
    build_v3_candidate_item_schema,
    compact_action_rule_v3,
    safe_a1_v3_reference_pool_view,
)
from attack_lab.archive.contracts.a1_v4_contract import (
    PROMPT_VERSION_V4,
    StaticPlanOption,
    V4ChoiceCatalog,
    build_v4_choice_catalog,
    build_v4_prompt_payload,
    build_v4_static_plan_options,
    parse_a1_v4_plan,
    parse_a1_v4_slot_replacements,
    resolve_choice_ids_to_changes,
    static_locks_and_cost,
    static_plan_by_id,
)
from attack_lab.archive.contracts.a1_v4_1_contract import (
    ActionSlotCatalog,
    PROMPT_VERSION_V4_1,
    build_v4_1_action_slots,
    build_v4_1_prompt_payload,
    parse_a1_v4_1_plan,
    parse_a1_v4_1_slot_replacements,
    resolve_action_slot_selections,
)
from attack_lab.archive.contracts.a1_v4_2_contract import (
    PROMPT_VERSION_V4_2,
    build_v4_2_prompt_payload,
    build_v4_2_repair_output_schema,
    parse_a1_v4_2_plan,
    parse_a1_v4_2_slot_replacements,
)
from attack_lab.attackers.a1_v4_3_contract import (
    PROMPT_VERSION_V4_3,
    build_v4_3_prompt_payload,
)
from attack_lab.budget import AttackBudget, compute_edit_metrics
from attack_lab.candidate_identity import canonical_candidate_fingerprint
from attack_lab.environment import AttackEnvironment
from attack_lab.governance_view import GovernanceView
from attack_lab.reference_actions import (
    ReferenceSelection,
    audit_reference_provenance,
    is_reference_selection,
    reference_ids_from_changes,
    resolve_reference_selection,
)
from attack_lab.reference_pool import ReferencePool
from attack_lab.types import AttackProposal, to_jsonable
from attack_lab.validator import ConstraintValidator

PROMPT_VERSION_V1 = "a1_oneshot_v1"
PROMPT_VERSION_V2 = "a1_oneshot_v2_diversified"
# Backward-compatible alias retained for older imports; formal default is v4.
PROMPT_VERSION = PROMPT_VERSION_V4
SUPPORTED_PROMPT_VERSIONS = frozenset(
    {
        PROMPT_VERSION_V1,
        PROMPT_VERSION_V2,
        PROMPT_VERSION_V3,
        PROMPT_VERSION_V4,
        PROMPT_VERSION_V4_1,
        PROMPT_VERSION_V4_2,
        PROMPT_VERSION_V4_3,
    }
)
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 800
# Thinking-enabled requests previously saturated max_tokens=800 with empty
# assistant content.  Raise only the thinking-on default; ThinkOff stays 800.
THINKING_ENABLED_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_PARSE_RETRIES = 2
# Local generation cap (parse + rule-compliance repair).  Equals
# DEFAULT_MAX_PARSE_RETRIES + 1 so existing A1ModelConfig hashes stay stable.
DEFAULT_MAX_LOCAL_GENERATION_ATTEMPTS = DEFAULT_MAX_PARSE_RETRIES + 1
DEFAULT_THINKING_DISABLED = True


def resolve_max_tokens(*, thinking_disabled: bool, max_tokens: int) -> int:
    """API max_tokens policy: ThinkOff keeps the request value; ThinkOn maps 800→2000."""
    mt = int(max_tokens)
    if bool(thinking_disabled):
        return mt
    if mt == DEFAULT_MAX_TOKENS:
        return int(THINKING_ENABLED_MAX_TOKENS)
    return mt

DIVERSIFICATION_PRINCIPLE_V2 = (
    "Because the full plan must be created before any feedback is available, "
    "construct a diversified ordered portfolio of candidates. Where feasible, "
    "the candidates should test materially different hypotheses by varying "
    "field combinations or strategy families, rather than making only small "
    "numerical changes to the same field pair. Each candidate must still "
    "satisfy all governance and budget constraints. Do not infer or target "
    "any hidden model score, threshold, or decision boundary."
)


@dataclass(frozen=True)
class A1ModelConfig:
    """Frozen formal decoding / transport configuration for A1."""

    model: str = DEFAULT_MODEL
    thinking_disabled: bool = DEFAULT_THINKING_DISABLED
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_parse_retries: int = DEFAULT_MAX_PARSE_RETRIES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    prompt_version: str = PROMPT_VERSION_V4
    reasoning_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "thinking_disabled": bool(self.thinking_disabled),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "max_tokens": int(self.max_tokens),
            "max_parse_retries": int(self.max_parse_retries),
            "timeout_seconds": float(self.timeout_seconds),
            "prompt_version": self.prompt_version,
        }
        # Preserve historical hashes when thinking stays disabled.
        if not self.thinking_disabled:
            payload["reasoning_effort"] = str(self.reasoning_effort or "max")
        return payload

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FORMAL_A1_MODEL_CONFIG = A1ModelConfig()

# DeepSeek V4 Flash list prices (USD / 1M tokens), pricing page checked 2026-08-06.
_FLASH_INPUT_CACHE_HIT_PER_MTOK = 0.0028
_FLASH_INPUT_CACHE_MISS_PER_MTOK = 0.14
_FLASH_OUTPUT_PER_MTOK = 0.28

_FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "risk_score",
        "y_score",
        "threshold",
        "fraud_bool",
        "feature_importance",
        "shap",
        "gradient",
        "gradients",
        "d1_risk_score",
        "d1_threshold",
    }
)


class A1PlannerError(RuntimeError):
    """Raised when A1 cannot build a usable frozen plan."""


# Local-generation retries: transport/parse failures AND pre-freeze rule
# failures.  Never retry from D1 PASS/BLOCK/INVALID (A1 is non-adaptive).
RETRYABLE_PARSE_STATUSES = frozenset({"empty", "parse_error", "schema_error"})
RETRYABLE_TRANSPORT_REASONS = frozenset({"timeout", "transport_error"})
RETRYABLE_LOCAL_STATUSES = frozenset({"local_validation_failed"})


@dataclass(frozen=True)
class A1AttemptRecord:
    """One raw LLM attempt in the retry ledger."""

    attempt_index: int
    timestamp: str
    retry_reason: str | None
    parse_status: str
    raw_response_path: str | None
    selected_for_plan: bool
    transport_error: str | None = None
    call_kind: str | None = None
    invalid_candidate_indices: tuple[int, ...] = ()
    preserved_candidate_indices: tuple[int, ...] = ()
    repaired_candidate_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "timestamp": self.timestamp,
            "retry_reason": self.retry_reason,
            "parse_status": self.parse_status,
            "raw_response_path": self.raw_response_path,
            "selected_for_plan": self.selected_for_plan,
            "transport_error": self.transport_error,
            "call_kind": self.call_kind,
            "invalid_candidate_indices": list(self.invalid_candidate_indices),
            "preserved_candidate_indices": list(self.preserved_candidate_indices),
            "repaired_candidate_indices": list(self.repaired_candidate_indices),
        }


@dataclass(frozen=True)
class A1CallRecord:
    """Researcher-facing per-call telemetry (never attacker-public feedback)."""

    model: str
    thinking_disabled: bool
    prompt_version: str
    prompt_hash: str
    config_hash: str
    latency_ms: float
    retry_count: int
    llm_call_count: int
    parse_status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    estimated_cost_usd: float
    raw_response_path: str | None
    parsed_plan_path: str | None
    retry_ledger_path: str | None
    prompt_text_path: str | None
    model_config_path: str | None
    selected_response_index: int | None
    n_raw_candidates: int
    n_frozen_candidates: int
    governance_reject_counts: Mapping[str, int]
    retry_ledger: tuple[A1AttemptRecord, ...] = ()
    q_max: int | None = None
    provisional_candidate_count: int | None = None
    local_repair_count: int = 0
    q_used_before_freeze: int = 0
    d1_calls_before_freeze: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "thinking_disabled": self.thinking_disabled,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "config_hash": self.config_hash,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "llm_call_count": self.llm_call_count,
            "parse_status": self.parse_status,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "raw_response_path": self.raw_response_path,
            "parsed_plan_path": self.parsed_plan_path,
            "retry_ledger_path": self.retry_ledger_path,
            "prompt_text_path": self.prompt_text_path,
            "model_config_path": self.model_config_path,
            "selected_response_index": self.selected_response_index,
            "n_raw_candidates": self.n_raw_candidates,
            "n_frozen_candidates": self.n_frozen_candidates,
            "governance_reject_counts": dict(self.governance_reject_counts),
            "retry_ledger": [item.to_dict() for item in self.retry_ledger],
            "q_max": self.q_max,
            "provisional_candidate_count": self.provisional_candidate_count,
            "local_repair_count": self.local_repair_count,
            "q_used_before_freeze": self.q_used_before_freeze,
            "d1_calls_before_freeze": self.d1_calls_before_freeze,
        }


@dataclass(frozen=True)
class LLMCompletion:
    """Raw model completion plus usage metadata."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    latency_ms: float
    thinking_disabled: bool
    system_fingerprint: str | None = None
    requested_model: str | None = None
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None


class LLMCompletionClient(Protocol):
    """Injectable chat-completion client (real DeepSeek or test double)."""

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        thinking_disabled: bool,
        reasoning_effort: str | None = None,
    ) -> LLMCompletion:
        ...


@dataclass
class DeepSeekPlannerClient:
    """OpenAI-compatible DeepSeek client with explicit thinking control."""

    api_key: str | None = None
    base_url: str | None = None
    default_model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if self.api_key is None or self.base_url is None:
            from deepseek_config import load_deepseek_settings

            settings = load_deepseek_settings()
            if self.api_key is None:
                self.api_key = settings.api_key
            if self.base_url is None:
                self.base_url = settings.base_url
            if self.default_model == DEFAULT_MODEL:
                self.default_model = settings.model

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        thinking_disabled: bool,
        reasoning_effort: str | None = None,
    ) -> LLMCompletion:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(timeout_seconds),
        )
        # Never rely on provider defaults: always send an explicit thinking flag.
        if thinking_disabled:
            extra_body: dict[str, Any] = {"thinking": {"type": "disabled"}}
            effort: str | None = None
        else:
            extra_body = {"thinking": {"type": "enabled"}}
            effort = str(reasoning_effort or "max")

        create_kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [dict(item) for item in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": resolve_max_tokens(
                thinking_disabled=bool(thinking_disabled),
                max_tokens=int(max_tokens),
            ),
            "stream": False,
            "extra_body": extra_body,
        }
        if effort is not None:
            create_kwargs["reasoning_effort"] = effort

        t0 = time.perf_counter()
        response = client.chat.completions.create(**create_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = ""
        if response.choices:
            message = response.choices[0].message
            text = (getattr(message, "content", None) or "").strip()

        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = (
            int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        )
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
        cached_tokens = _extract_cached_tokens(usage)
        reasoning_tokens = _extract_reasoning_tokens(usage)

        requested = str(model or self.default_model)
        returned = str(getattr(response, "model", None) or requested)
        fingerprint = getattr(response, "system_fingerprint", None)
        if fingerprint is not None:
            fingerprint = str(fingerprint)
        return LLMCompletion(
            text=text,
            model=returned,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            cached_tokens=cached_tokens,
            latency_ms=float(latency_ms),
            thinking_disabled=bool(thinking_disabled),
            system_fingerprint=fingerprint,
            requested_model=requested,
            reasoning_effort=effort,
            reasoning_tokens=reasoning_tokens,
        )


def estimate_flash_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimate USD cost for deepseek-v4-flash from token usage."""
    cached = max(0, min(int(cached_tokens), int(prompt_tokens)))
    miss = max(0, int(prompt_tokens) - cached)
    cost = (
        (cached / 1_000_000.0) * _FLASH_INPUT_CACHE_HIT_PER_MTOK
        + (miss / 1_000_000.0) * _FLASH_INPUT_CACHE_MISS_PER_MTOK
        + (int(completion_tokens) / 1_000_000.0) * _FLASH_OUTPUT_PER_MTOK
    )
    return float(cost)


def build_a1_prompt_payload(
    *,
    env: AttackEnvironment,
    reference_pool: ReferencePool,
    budget: AttackBudget,
    q_max: int,
    prompt_version: str = PROMPT_VERSION_V1,
) -> dict[str, Any]:
    """Build the attacker-public planning payload (no D1 internals)."""
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise A1PlannerError(
            f"Unsupported prompt_version={prompt_version!r}; "
            f"supported={sorted(SUPPORTED_PROMPT_VERSIONS)}."
        )
    observation = env.observation()
    view = GovernanceView.from_policy(
        env.validator.policy,
        budget=budget,
        read_only_context_fields=reference_pool.read_only_context_fields,
        enabled_action_keys=env.validator.enabled_action_keys,
    )
    action_catalogue = [
        _compact_action_rule(rule.to_public_dict())
        for rule in view.action_field_rules
    ]
    visible_anchor = to_jsonable(dict(observation.visible_fields))

    if prompt_version == PROMPT_VERSION_V3:
        allowed_action_keys = list(env.validator.enabled_action_keys)
        allowed_visible_fields = sorted(
            set(
                env.validator.visible_fields(env.starting_case.features)
            ).union(reference_pool.read_only_context_fields)
        )
        reference_pool_view = safe_a1_v3_reference_pool_view(
            reference_pool, allowed_visible_fields=allowed_visible_fields
        )
        allowed_reference_ids = [
            profile.profile_id for profile in reference_pool.profiles
        ]
        payload = {
            "task": (
                "Plan an ordered sequence of unique onboarding applications that "
                "differ from the blocked anchor. Freeze the full sequence now; "
                "no later revision is allowed."
            ),
            "prompt_version": prompt_version,
            "budget": {
                "q_max": int(q_max),
                "m_max": int(budget.m_max),
                "notes": build_v3_budget_notes(q_max=q_max, m_max=budget.m_max),
            },
            "allowed_action_keys": allowed_action_keys,
            "action_catalogue": [
                compact_action_rule_v3(rule)
                for rule in view.action_field_rules
                if rule.action_key in set(allowed_action_keys)
            ],
            "action_roles": action_roles_from_view(
                view, enabled_action_keys=allowed_action_keys
            ),
            "anchor": {
                "case_id": observation.case_id,
                "visible_fields": visible_anchor,
            },
            "reference_pool": reference_pool_view,
            "allowed_reference_ids": allowed_reference_ids,
            "output_schema": {
                "type": "object",
                "required": ["candidates"],
                "properties": {
                    "candidates": {
                        "type": "array",
                        "minItems": int(q_max),
                        "maxItems": int(q_max),
                        "items": build_v3_candidate_item_schema(
                            allowed_action_keys=allowed_action_keys,
                            allowed_reference_ids=allowed_reference_ids,
                            m_max=budget.m_max,
                        ),
                    }
                },
            },
            "planning_principle": DIVERSIFICATION_PRINCIPLE_V3,
            "static_lock_principle": STATIC_LOCK_PRINCIPLE_V3,
            "explicitly_unavailable": [
                "d1_risk_score",
                "d1_threshold",
                "feature_importance_or_shap",
                "gradients",
                "true_rejection_reason",
                "fraud_bool",
            ],
        }
        _assert_prompt_safe(payload)
        assert_v3_prompt_single_vocab(payload)
        return payload

    if prompt_version == PROMPT_VERSION_V2:
        candidate_item_schema: dict[str, Any] = {
            "type": "object",
            "required": ["strategy_label", "changes"],
            "additionalProperties": False,
            "properties": {
                "strategy_label": {
                    "type": "string",
                    "description": (
                        "Short descriptive label for the hypothesis or "
                        "strategy family tested by this candidate."
                    ),
                },
                "changes": {
                    "type": "object",
                    "description": (
                        "Map of action_key -> {\"reference_id\": \"ref_XX\"}. "
                        "Each value must select a profile from the provided "
                        "reference_pool. Do not invent free literals, catalogue "
                        "proxy names, or substitute reference_ids."
                    ),
                    "minProperties": 1,
                    "maxProperties": int(budget.m_max),
                },
            },
        }
    else:
        candidate_item_schema = {
            "type": "object",
            "required": ["changes"],
            "properties": {
                "changes": {
                    "type": "object",
                    "description": (
                        "Map of action_key -> {\"reference_id\": \"ref_XX\"} "
                        "from the provided reference_pool only."
                    ),
                }
            },
        }

    payload: dict[str, Any] = {
        "task": (
            "Plan an ordered sequence of unique onboarding applications that "
            "differ from the blocked anchor. Freeze the full sequence now; "
            "no later revision is allowed."
        ),
        "prompt_version": prompt_version,
        "budget": {
            "q_max": int(q_max),
            "m_max": int(budget.m_max),
            "notes": [
                "Return at most q_max candidates, ordered for submission.",
                "Each candidate must change between 1 and m_max mutable features "
                "relative to the original anchor.",
                "Every changed value must be a reference_id from reference_pool "
                "(ReferenceSelection). Governance judges legality only; it is "
                "not a value source.",
                "Episode-static fields must be identical across all candidates.",
                "Candidates must be unique.",
                "Local rule failures regenerate the plan before any defender "
                "query; they do not consume Q.",
            ],
        },
        "anchor": {
            "case_id": observation.case_id,
            "visible_fields": visible_anchor,
        },
        "reference_pool": reference_pool.attacker_view(),
        "action_catalogue": action_catalogue,
        "field_roles": {
            "per_attempt_fields": list(view.per_attempt_fields),
            "episode_static_fields": list(view.episode_static_fields),
            "forbidden_fields": list(view.forbidden_fields),
            "read_only_context_fields": list(view.read_only_context_fields),
        },
        "output_schema": {
            "type": "object",
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": int(q_max),
                    "items": candidate_item_schema,
                }
            },
        },
        "explicitly_unavailable": [
            "d1_risk_score",
            "d1_threshold",
            "feature_importance_or_shap",
            "gradients",
            "true_rejection_reason",
            "fraud_bool",
        ],
    }
    if prompt_version == PROMPT_VERSION_V2:
        payload["planning_principle"] = DIVERSIFICATION_PRINCIPLE_V2
    _assert_prompt_safe(payload)
    return payload


def render_a1_messages(
    payload: Mapping[str, Any],
    *,
    local_generation_feedback: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Render system/user messages for the one-shot planner call."""
    version = str(payload.get("prompt_version", PROMPT_VERSION_V1))
    if version in {PROMPT_VERSION_V4_2, PROMPT_VERSION_V4_3}:
        system = (
            "You are a constrained synthetic-identity application planner in a "
            "sandbox. Reply with JSON only. Do not include markdown fences, "
            "commentary, scores, thresholds or chain-of-thought. "
            "Select only provided action_slot_id -> choice_id pairs. Trusted "
            "code resolves the underlying reference-backed value. Never emit "
            "raw values, action_key, or reference_id."
        )
    else:
        system = (
            "You are a constrained synthetic-identity application planner in a "
            "sandbox. Reply with JSON only. Do not include markdown fences, "
            "commentary, scores, thresholds or chain-of-thought. "
            "Every change must cite a reference_id from the provided pool; "
            "never invent raw values and never ask the system to substitute "
            "a reference_id for you."
        )
    if version in {PROMPT_VERSION_V4_2, PROMPT_VERSION_V4_3}:
        q_max = int((payload.get("budget") or {}).get("q_max", 0))
        repair_mode = bool(
            local_generation_feedback
            and (
                local_generation_feedback.get("mode") == "slot_repair"
                or local_generation_feedback.get("invalid_candidate_indices")
            )
        )
        if repair_mode:
            indices = local_generation_feedback.get("invalid_candidate_indices", [])
            residual = local_generation_feedback.get("residual_m")
            user = (
                "Using only the JSON planning context below, return ONLY a JSON "
                "object with key 'replacements'. Include one object for each listed "
                "candidate_index, with candidate_index, strategy_label, and "
                "selections (action_slot_id -> choice_id). Enforce "
                f"1 <= len(selections) <= residual_m={residual}. Do not emit "
                "action_key, reference_id, choice_ids arrays, changes, or a new "
                "static_plan_id. Repair only the listed invalid slots.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
                "\n\nLOCAL RULE-COMPLIANCE SLOT REPAIR (not defender feedback):\n"
                f"{json.dumps(to_jsonable(dict(local_generation_feedback)), sort_keys=True, indent=2)}"
                f"\nRepair indices: {json.dumps(indices)}. This repair does not consume Q."
            )
        else:
            user = (
                "Using only the JSON planning context below, return a JSON object "
                "with keys 'static_plan_id' and 'candidates' containing exactly "
                f"{q_max} ordered unique query candidates. Each candidate must "
                "include strategy_label and selections only, with "
                "1 <= len(selections) <= the chosen plan residual_m. Follow "
                "planning_principle, hard_contract, and output_schema.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
            )
    elif version == PROMPT_VERSION_V4_1:
        q_max = int((payload.get("budget") or {}).get("q_max", 0))
        repair_mode = bool(
            local_generation_feedback
            and (
                local_generation_feedback.get("mode") == "slot_repair"
                or local_generation_feedback.get("invalid_candidate_indices")
            )
        )
        if repair_mode:
            indices = local_generation_feedback.get("invalid_candidate_indices", [])
            user = (
                "Using only the JSON planning context below, return ONLY a JSON "
                "object with key 'replacements'. Include one object for each listed "
                "candidate_index, with candidate_index, strategy_label, and "
                "selections (action_slot_id -> choice_id). Do not emit action_key, "
                "reference_id, choice_ids arrays, changes, or a new static_plan_id. "
                "Repair only the listed invalid slots.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
                "\n\nLOCAL RULE-COMPLIANCE SLOT REPAIR (not defender feedback):\n"
                f"{json.dumps(to_jsonable(dict(local_generation_feedback)), sort_keys=True, indent=2)}"
                f"\nRepair indices: {json.dumps(indices)}. This repair does not consume Q."
            )
        else:
            user = (
                "Using only the JSON planning context below, return a JSON object "
                "with keys 'static_plan_id' and 'candidates' containing exactly "
                f"{q_max} ordered unique query candidates. Each candidate must "
                "include strategy_label and selections only. Follow "
                "planning_principle and hard_contract.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
            )
    elif version == PROMPT_VERSION_V4:
        q_max = int((payload.get("budget") or {}).get("q_max", 0))
        repair_mode = bool(
            local_generation_feedback
            and (
                local_generation_feedback.get("mode") == "slot_repair"
                or local_generation_feedback.get("invalid_candidate_indices")
            )
        )
        if repair_mode:
            indices = local_generation_feedback.get("invalid_candidate_indices", [])
            user = (
                "Using only the JSON planning context below, return ONLY a JSON "
                "object with key 'replacements'. Include one object for each listed "
                "candidate_index, with candidate_index, strategy_label, and "
                "choice_ids. Do not emit action_key, reference_id, changes, or a "
                "new static_plan_id. Repair only the listed invalid slots.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
                "\n\nLOCAL RULE-COMPLIANCE SLOT REPAIR (not defender feedback):\n"
                f"{json.dumps(to_jsonable(dict(local_generation_feedback)), sort_keys=True, indent=2)}"
                f"\nRepair indices: {json.dumps(indices)}. This repair does not consume Q."
            )
        else:
            user = (
                "Using only the JSON planning context below, return a JSON object "
                "with keys 'static_plan_id' and 'candidates' containing exactly "
                f"{q_max} ordered unique query candidates. Each candidate must "
                "include strategy_label and choice_ids only. Follow "
                "planning_principle and hard_contract.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
            )
    elif version == PROMPT_VERSION_V3:
        q_max = int((payload.get("budget") or {}).get("q_max", 0))
        repair_mode = bool(
            local_generation_feedback
            and (
                local_generation_feedback.get("mode") == "slot_repair"
                or local_generation_feedback.get("invalid_candidate_indices")
            )
        )
        if repair_mode:
            indices = local_generation_feedback.get("invalid_candidate_indices", [])
            user = (
                "Using only the JSON planning context below, return ONLY a JSON "
                "object with key 'replacements'. Include one object for each listed "
                "candidate_index, with candidate_index, strategy_label, and changes. "
                "Repair only the listed invalid slots; do not rewrite preserved "
                "slots and do not return a candidates list.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
                "\n\nLOCAL RULE-COMPLIANCE SLOT REPAIR (not defender feedback):\n"
                f"{json.dumps(to_jsonable(dict(local_generation_feedback)), sort_keys=True, indent=2)}"
                f"\nRepair indices: {json.dumps(indices)}. This repair does not consume Q."
            )
        else:
            user = (
                "Using only the JSON planning context below, return a JSON object "
                "with key 'candidates' containing exactly "
                f"{q_max} ordered unique application variants. Each candidate must "
                "include strategy_label and changes. Follow planning_principle and "
                "static_lock_principle.\n\n"
                f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
            )
    elif version == PROMPT_VERSION_V2:
        user = (
            "Using only the JSON planning context below, return a JSON object "
            "with key 'candidates' containing an ordered list of up to q_max "
            "unique application variants. Each candidate object must include "
            "'strategy_label' and 'changes'. Follow planning_principle when "
            "constructing the portfolio.\n\n"
            f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
        )
    else:
        user = (
            "Using only the JSON planning context below, return a JSON object with "
            "key 'candidates' containing an ordered list of up to q_max unique "
            "application variants.\n\n"
            f"{json.dumps(to_jsonable(dict(payload)), sort_keys=True, indent=2)}"
        )
    if local_generation_feedback and version not in {
        PROMPT_VERSION_V3,
        PROMPT_VERSION_V4,
        PROMPT_VERSION_V4_1,
        PROMPT_VERSION_V4_2,
        PROMPT_VERSION_V4_3,
    }:
        user += (
            "\n\nLOCAL RULE-COMPLIANCE REPAIR (not defender feedback):\n"
            f"{json.dumps(to_jsonable(dict(local_generation_feedback)), sort_keys=True, indent=2)}\n"
            "Correct your own field/reference_id choices and resubmit a full plan. "
            "This repair does not consume the attack budget Q."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def format_a1_prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    """Deterministic full prompt text for hashing and artefact persistence."""
    blocks: list[str] = []
    for item in messages:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        blocks.append(f"### ROLE: {role}\n{content}")
    return "\n\n".join(blocks) + "\n"


def hash_a1_prompt_text(prompt_text: str) -> str:
    """SHA-256 over the exact persisted prompt text."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def parse_a1_candidates(
    raw_text: str,
    *,
    prompt_version: str = PROMPT_VERSION_V1,
    m_max: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Parse candidate objects from a model response.

    Each accepted candidate is
    ``{"changes": {...}, "strategy_label": str | None}``.

    Returns ``(candidates, parse_status)`` where status is one of
    ``ok``, ``empty``, ``parse_error``, ``schema_error``.
    """
    text = (raw_text or "").strip()
    if not text:
        return [], "empty"

    payload = _loads_json_object(text)
    if payload is None:
        return [], "parse_error"

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return [], "schema_error"

    require_label = prompt_version in {PROMPT_VERSION_V2, PROMPT_VERSION_V3}
    edit_cap = None if m_max is None else int(m_max)
    candidates: list[dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, Mapping):
            return [], "schema_error"
        if require_label:
            extra_keys = set(item.keys()) - {"strategy_label", "changes"}
            if extra_keys:
                return [], "schema_error"
        changes = item.get("changes")
        if not isinstance(changes, Mapping):
            return [], "schema_error"
        normalised_changes = {str(key): value for key, value in changes.items()}
        if require_label:
            n_keys = len(normalised_changes)
            if n_keys < 1:
                return [], "schema_error"
            if edit_cap is not None and n_keys > edit_cap:
                return [], "schema_error"
        label_raw = item.get("strategy_label")
        if require_label:
            if not isinstance(label_raw, str) or not label_raw.strip():
                return [], "schema_error"
            strategy_label: str | None = label_raw.strip()
        else:
            strategy_label = (
                label_raw.strip()
                if isinstance(label_raw, str) and label_raw.strip()
                else None
            )
        candidates.append(
            {
                "changes": normalised_changes,
                "strategy_label": strategy_label,
            }
        )
    return candidates, "ok"


def parse_a1_slot_replacements(
    raw_text: str, *, m_max: int
) -> tuple[list[dict[str, Any]], str]:
    """Parse V3 slot replacements without accepting a whole-plan response."""
    text = (raw_text or "").strip()
    if not text:
        return [], "empty"
    payload = _loads_json_object(text)
    if payload is None:
        return [], "parse_error"
    replacements = payload.get("replacements")
    if not isinstance(replacements, list):
        return [], "schema_error"
    parsed: list[dict[str, Any]] = []
    for item in replacements:
        if not isinstance(item, Mapping) or set(item) != {
            "candidate_index", "strategy_label", "changes"
        }:
            return [], "schema_error"
        index = item.get("candidate_index")
        label = item.get("strategy_label")
        changes = item.get("changes")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            return [], "schema_error"
        if not isinstance(label, str) or not label.strip() or not isinstance(changes, Mapping):
            return [], "schema_error"
        normalised = {str(key): value for key, value in changes.items()}
        if not normalised or len(normalised) > int(m_max):
            return [], "schema_error"
        parsed.append(
            {
                "candidate_index": index,
                "strategy_label": label.strip(),
                "changes": normalised,
            }
        )
    return parsed, "ok"


@dataclass
class OneShotLLMPlanner:
    """Official A1 static/non-adaptive LLM planner (Q,m protocol)."""

    experiment_seed: int
    reference_pool: ReferencePool
    budget: AttackBudget
    attacker_id: str = "a1"
    prompt_version: str = PROMPT_VERSION_V4
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_parse_retries: int = DEFAULT_MAX_PARSE_RETRIES
    max_local_generation_attempts: int = DEFAULT_MAX_LOCAL_GENERATION_ATTEMPTS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    thinking_disabled: bool = DEFAULT_THINKING_DISABLED
    reasoning_effort: str | None = None
    llm_client: LLMCompletionClient | None = None
    stdout: TextIO | None = None
    _episode_seed: int | None = field(default=None, init=False, repr=False)
    _frozen_proposals: list[AttackProposal] = field(
        default_factory=list, init=False, repr=False
    )
    _submit_index: int = field(default=0, init=False, repr=False)
    _sequence_prepared: bool = field(default=False, init=False, repr=False)
    _pending_stop_reason: str | None = field(default=None, init=False, repr=False)
    _call_record: A1CallRecord | None = field(default=None, init=False, repr=False)
    _raw_response_text: str = field(default="", init=False, repr=False)
    _prompt_payload: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _prompt_text: str = field(default="", init=False, repr=False)
    _prompt_hash: str = field(default="", init=False, repr=False)
    _model_config: A1ModelConfig | None = field(default=None, init=False, repr=False)
    _config_hash: str = field(default="", init=False, repr=False)
    _governance_reject_counts: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _retry_ledger: list[A1AttemptRecord] = field(
        default_factory=list, init=False, repr=False
    )
    _selected_response_index: int | None = field(default=None, init=False, repr=False)
    _v4_catalog: V4ChoiceCatalog | None = field(default=None, init=False, repr=False)
    _v4_static_plans: tuple[StaticPlanOption, ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _v4_selected_static_plan_id: str | None = field(default=None, init=False, repr=False)
    _v4_selected_locks: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _v4_1_action_slots: ActionSlotCatalog | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.reference_pool.K < 1:
            raise A1PlannerError("reference_pool.K must be >= 1.")
        if self.budget.m_max < 0:
            raise A1PlannerError("budget.m_max must be >= 0.")
        if self.budget.q_max < 1:
            raise A1PlannerError("budget.q_max must be >= 1.")
        if self.max_parse_retries < 0:
            raise A1PlannerError("max_parse_retries must be >= 0.")
        if int(self.max_local_generation_attempts) < 1:
            raise A1PlannerError("max_local_generation_attempts must be >= 1.")
        if self.timeout_seconds <= 0:
            raise A1PlannerError("timeout_seconds must be > 0.")
        if not (0.0 <= float(self.top_p) <= 1.0):
            raise A1PlannerError("top_p must be in [0, 1].")
        if self.prompt_version not in SUPPORTED_PROMPT_VERSIONS:
            raise A1PlannerError(
                f"Unsupported prompt_version={self.prompt_version!r}; "
                f"supported={sorted(SUPPORTED_PROMPT_VERSIONS)}."
            )
        self.max_tokens = resolve_max_tokens(
            thinking_disabled=bool(self.thinking_disabled),
            max_tokens=int(self.max_tokens),
        )
        self._model_config = A1ModelConfig(
            model=self.model,
            thinking_disabled=bool(self.thinking_disabled),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            max_tokens=int(self.max_tokens),
            max_parse_retries=int(self.max_parse_retries),
            timeout_seconds=float(self.timeout_seconds),
            prompt_version=self.prompt_version,
            reasoning_effort=(
                None
                if self.thinking_disabled
                else (self.reasoning_effort or "max")
            ),
        )
        self._config_hash = self._model_config.config_hash()

    @property
    def model_config(self) -> A1ModelConfig:
        assert self._model_config is not None
        return self._model_config

    @property
    def frozen_proposals(self) -> tuple[AttackProposal, ...]:
        return tuple(self._frozen_proposals)

    @property
    def call_record(self) -> A1CallRecord | None:
        return self._call_record

    @property
    def raw_response_text(self) -> str:
        return self._raw_response_text

    def run(self, env: AttackEnvironment) -> None:
        """Freeze the candidate sequence, then submit without feedback adaptation."""
        self.prepare_frozen_sequence(env)
        self._write(
            f"\n=== A1 OneShotLLMPlanner "
            f"(experiment_seed={self.experiment_seed}, "
            f"episode_seed={self._episode_seed}, "
            f"case={env.starting_case.case_id}, "
            f"frozen={len(self._frozen_proposals)}, "
            f"Q={self.budget.q_max}, m={self.budget.m_max}, "
            f"model={self.model}) ===\n"
            "Pre-feedback frozen candidate sequence; feedback does not alter "
            "generation.\n"
        )
        while not env.done:
            proposal = self.propose(env)
            if proposal is None:
                reason = self._pending_stop_reason or "no_feasible_candidate"
                self._write(f"Local stop: {reason}.\n")
                env.abort(reason=reason)
                return
            self._write(
                f"candidate_index={self._submit_index}: submitting "
                f"{sorted(proposal.changes)}\n"
            )
            # Intentionally ignore StepRecord feedback for generation/adaptation.
            env.step(proposal)
            self._submit_index += 1
        self._write(
            f"Episode stop observed via environment.done "
            f"(success={env.success}).\n"
        )

    def prepare_frozen_sequence(self, env: AttackEnvironment) -> tuple[AttackProposal, ...]:
        """Local-generation loop, then freeze before any D1 submission.

        Parse/transport failures and pre-freeze rule failures may regenerate
        within ``max_local_generation_attempts``.  None of those attempts call
        ``env.step`` or consume Q.
        """
        if self.prompt_version in {
            PROMPT_VERSION_V4_2,
            PROMPT_VERSION_V4_3,
        }:
            return self._prepare_v4_2_frozen_sequence(env)
        if self.prompt_version == PROMPT_VERSION_V4_1:
            return self._prepare_v4_1_frozen_sequence(env)
        if self.prompt_version == PROMPT_VERSION_V4:
            return self._prepare_v4_frozen_sequence(env)
        if self.prompt_version == PROMPT_VERSION_V3:
            return self._prepare_v3_frozen_sequence(env)
        if self._sequence_prepared:
            return self.frozen_proposals
        if env.done:
            self._sequence_prepared = True
            self._pending_stop_reason = "q_exhausted"
            return ()

        q_max = min(int(env.budget.q_max), int(self.budget.q_max))
        if q_max < 1:
            raise A1PlannerError("budget.q_max must be >= 1.")

        self._episode_seed = derive_episode_seed(
            self.experiment_seed, env.starting_case.case_id, self.attacker_id
        )
        self._frozen_proposals = []
        self._submit_index = 0
        self._pending_stop_reason = None
        self._call_record = None
        self._raw_response_text = ""
        self._governance_reject_counts = {}
        self._retry_ledger = []
        self._selected_response_index = None

        self._prompt_payload = build_a1_prompt_payload(
            env=env,
            reference_pool=self.reference_pool,
            budget=self.budget,
            q_max=q_max,
            prompt_version=self.prompt_version,
        )
        client = self.llm_client or DeepSeekPlannerClient(default_model=self.model)
        run_dir = _episode_run_dir(env)
        total_latency_ms = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        parse_status = "empty"
        completion: LLMCompletion | None = None
        raw_candidates: list[dict[str, Any]] = []
        selected_model = self.model
        selected_thinking = bool(self.thinking_disabled)
        # Cap local generation; keep A1ModelConfig hash stable via parse retries.
        max_attempts = min(
            int(self.max_local_generation_attempts),
            int(self.max_parse_retries) + 1,
        )
        if max_attempts < 1:
            max_attempts = 1
        local_feedback: dict[str, Any] | None = None
        q_before = int(env.ledger.q_remaining)

        for attempt_idx in range(max_attempts):
            messages = render_a1_messages(
                self._prompt_payload,
                local_generation_feedback=local_feedback,
            )
            self._prompt_text = format_a1_prompt_text(messages)
            self._prompt_hash = hash_a1_prompt_text(self._prompt_text)
            timestamp = datetime.now(timezone.utc).isoformat()
            raw_path: str | None = None
            transport_error: str | None = None
            attempt_text = ""

            try:
                completion = client.complete(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    thinking_disabled=self.thinking_disabled,
                    reasoning_effort=(
                        None
                        if self.thinking_disabled
                        else (self.reasoning_effort or "max")
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — classify then decide retry
                reason = _classify_transport_error(exc)
                transport_error = f"{type(exc).__name__}: {exc}"
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=reason,
                        parse_status=reason,
                        raw_response_path=None,
                        selected_for_plan=False,
                        transport_error=transport_error,
                    )
                )
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1,
                    max_attempts=max_attempts,
                    error_code=reason,
                    details=[{"error_code": reason}],
                )
                if (
                    reason in RETRYABLE_TRANSPORT_REASONS
                    and attempt_idx + 1 < max_attempts
                ):
                    continue
                break

            assert completion is not None
            total_latency_ms += float(completion.latency_ms)
            total_prompt_tokens += int(completion.prompt_tokens)
            total_completion_tokens += int(completion.completion_tokens)
            total_cached_tokens += int(completion.cached_tokens)
            selected_model = completion.model
            selected_thinking = bool(completion.thinking_disabled)
            attempt_text = completion.text
            self._raw_response_text = attempt_text
            raw_path = _persist_raw_attempt(run_dir, attempt_idx, attempt_text)

            raw_candidates, parse_status = parse_a1_candidates(
                attempt_text,
                prompt_version=self.prompt_version,
                m_max=self.budget.m_max,
            )

            if parse_status in RETRYABLE_PARSE_STATUSES:
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=parse_status,
                        parse_status=parse_status,
                        raw_response_path=raw_path,
                        selected_for_plan=False,
                        transport_error=None,
                    )
                )
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1,
                    max_attempts=max_attempts,
                    error_code=parse_status,
                    details=[{"error_code": parse_status}],
                )
                if attempt_idx + 1 < max_attempts:
                    continue
                break

            frozen, reject_counts, reject_details = self._freeze_validated_candidates(
                env,
                raw_candidates,
                q_max=q_max,
                require_all_candidates=True,
            )
            self._governance_reject_counts = dict(reject_counts)

            if not frozen:
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason="local_validation_failed",
                        parse_status="local_validation_failed",
                        raw_response_path=raw_path,
                        selected_for_plan=False,
                        transport_error=None,
                    )
                )
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1,
                    max_attempts=max_attempts,
                    error_code="local_validation_failed",
                    details=reject_details,
                )
                if attempt_idx + 1 < max_attempts:
                    continue
                break

            # Successful local plan: freeze; never call the LLM again.
            self._selected_response_index = attempt_idx
            self._retry_ledger.append(
                A1AttemptRecord(
                    attempt_index=attempt_idx,
                    timestamp=timestamp,
                    retry_reason=None,
                    parse_status="ok",
                    raw_response_path=raw_path,
                    selected_for_plan=True,
                    transport_error=None,
                )
            )
            self._frozen_proposals = list(frozen)
            break

        # Mark sequence prepared before any D1 submission can occur.
        self._sequence_prepared = True
        if int(env.ledger.q_remaining) != q_before:
            raise A1PlannerError(
                "Local generation must not consume Q; ledger changed unexpectedly."
            )
        llm_call_count = len(self._retry_ledger)
        retry_count = max(0, llm_call_count - 1)

        if self._selected_response_index is None:
            self._frozen_proposals = []
            if not self._retry_ledger:
                raise A1PlannerError("LLM client returned no completion attempts.")

        if not self._frozen_proposals:
            if self.budget.m_max < 1:
                self._pending_stop_reason = "insufficient_edit_budget"
            elif any(
                item.parse_status == "local_validation_failed"
                for item in self._retry_ledger
            ):
                self._pending_stop_reason = "local_generation_exhausted"
            else:
                self._pending_stop_reason = "no_feasible_plan"

        final_parse_status = (
            "ok"
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].parse_status if self._retry_ledger else "empty"
            )
        )

        selected_raw_path = None
        if self._selected_response_index is not None:
            selected_raw_path = self._retry_ledger[
                self._selected_response_index
            ].raw_response_path
        elif self._retry_ledger:
            selected_raw_path = self._retry_ledger[-1].raw_response_path

        (
            plan_path,
            call_path,
            ledger_path,
            prompt_text_path,
            model_config_path,
        ) = self._write_artefacts(
            env,
            model=selected_model,
            thinking_disabled=selected_thinking,
            parse_status=final_parse_status,
            retry_count=retry_count,
            llm_call_count=llm_call_count,
            raw_candidates=raw_candidates,
            total_latency_ms=total_latency_ms,
            selected_raw_path=selected_raw_path,
        )
        self._call_record = A1CallRecord(
            model=selected_model,
            thinking_disabled=selected_thinking,
            prompt_version=self.prompt_version,
            prompt_hash=self._prompt_hash,
            config_hash=self._config_hash,
            latency_ms=float(total_latency_ms),
            retry_count=int(retry_count),
            llm_call_count=int(llm_call_count),
            parse_status=final_parse_status,
            prompt_tokens=int(total_prompt_tokens),
            completion_tokens=int(total_completion_tokens),
            total_tokens=int(total_prompt_tokens + total_completion_tokens),
            cached_tokens=int(total_cached_tokens),
            estimated_cost_usd=estimate_flash_cost_usd(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_tokens=total_cached_tokens,
            ),
            raw_response_path=selected_raw_path,
            parsed_plan_path=plan_path,
            retry_ledger_path=ledger_path,
            prompt_text_path=prompt_text_path,
            model_config_path=model_config_path,
            selected_response_index=self._selected_response_index,
            n_raw_candidates=len(raw_candidates),
            n_frozen_candidates=len(self._frozen_proposals),
            governance_reject_counts=dict(self._governance_reject_counts),
            retry_ledger=tuple(self._retry_ledger),
        )
        if call_path is not None and self._call_record is not None:
            Path(call_path).write_text(
                json.dumps(self._call_record.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return self.frozen_proposals

    def _prepare_v4_2_frozen_sequence(
        self, env: AttackEnvironment
    ) -> tuple[AttackProposal, ...]:
        """Freeze V4.2 after bounded local slot repair under bounded unique action slots."""
        if self._sequence_prepared:
            return self.frozen_proposals
        if env.done:
            self._sequence_prepared = True
            self._pending_stop_reason = "q_exhausted"
            return ()
        q_max = min(int(env.budget.q_max), int(self.budget.q_max))
        self._episode_seed = derive_episode_seed(
            self.experiment_seed, env.starting_case.case_id, self.attacker_id
        )
        self._frozen_proposals = []
        self._submit_index = 0
        self._pending_stop_reason = None
        self._call_record = None
        self._raw_response_text = ""
        self._governance_reject_counts = {}
        self._retry_ledger = []
        self._selected_response_index = None
        self._v4_selected_static_plan_id = None
        self._v4_selected_locks = None
        self._v4_1_action_slots = None

        catalog = build_v4_choice_catalog(
            validator=env.validator,
            pool=self.reference_pool,
            anchor=env.starting_case.features,
        )
        static_plans = build_v4_static_plan_options(
            validator=env.validator,
            pool=self.reference_pool,
            anchor=env.starting_case.features,
            catalog=catalog,
            m_max=self.budget.m_max,
            q_max=q_max,
        )
        action_slots = build_v4_1_action_slots(catalog)
        self._v4_catalog = catalog
        self._v4_static_plans = static_plans
        self._v4_1_action_slots = action_slots
        if not static_plans:
            self._sequence_prepared = True
            self._pending_stop_reason = "no_feasible_plan"
            return ()

        observation = env.observation()
        if self.prompt_version == PROMPT_VERSION_V4_3:
            self._prompt_payload = build_v4_3_prompt_payload(
                validator=env.validator,
                pool=self.reference_pool,
                budget=self.budget,
                q_max=q_max,
                visible_anchor=observation.visible_fields,
                case_id=str(observation.case_id),
                catalog=catalog,
                static_plans=static_plans,
                action_slots=action_slots,
            )
        else:
            self._prompt_payload = build_v4_2_prompt_payload(
                validator=env.validator,
                pool=self.reference_pool,
                budget=self.budget,
                q_max=q_max,
                visible_anchor=observation.visible_fields,
                case_id=str(observation.case_id),
                catalog=catalog,
                static_plans=static_plans,
                action_slots=action_slots,
            )
        client = self.llm_client or DeepSeekPlannerClient(default_model=self.model)
        run_dir = _episode_run_dir(env)
        max_attempts = max(
            1,
            min(
                int(self.max_local_generation_attempts),
                int(self.max_parse_retries) + 1,
            ),
        )
        q_before = int(env.ledger.q_remaining)
        steps_before = int(env.attempts_used)
        provisional_raw: list[dict[str, Any]] | None = None
        local_feedback: dict[str, Any] | None = None
        raw_candidates: list[dict[str, Any]] = []
        total_latency_ms = total_prompt_tokens = total_completion_tokens = total_cached_tokens = 0
        selected_model, selected_thinking = self.model, bool(self.thinking_disabled)
        allowed_plan_ids = [plan.static_plan_id for plan in static_plans]

        for attempt_idx in range(max_attempts):
            call_kind = "initial_plan" if provisional_raw is None else "repair"
            messages = render_a1_messages(
                self._prompt_payload, local_generation_feedback=local_feedback
            )
            self._prompt_text = format_a1_prompt_text(messages)
            self._prompt_hash = hash_a1_prompt_text(self._prompt_text)
            timestamp = datetime.now(timezone.utc).isoformat()
            raw_path: str | None = None
            try:
                completion = client.complete(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    thinking_disabled=self.thinking_disabled,
                    reasoning_effort=(
                        None
                        if self.thinking_disabled
                        else (self.reasoning_effort or "max")
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                reason = _classify_transport_error(exc)
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=reason,
                        parse_status=reason,
                        raw_response_path=None,
                        selected_for_plan=False,
                        transport_error=f"{type(exc).__name__}: {exc}",
                        call_kind="transport/parse",
                    )
                )
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1,
                    max_attempts=max_attempts,
                    error_code=reason,
                    details=[{"error_code": reason}],
                )
                continue
            total_latency_ms += float(completion.latency_ms)
            total_prompt_tokens += int(completion.prompt_tokens)
            total_completion_tokens += int(completion.completion_tokens)
            total_cached_tokens += int(completion.cached_tokens)
            selected_model = completion.model
            selected_thinking = bool(completion.thinking_disabled)
            self._raw_response_text = completion.text
            raw_path = _persist_raw_attempt(run_dir, attempt_idx, completion.text)

            repaired_indices: tuple[int, ...] = ()
            if provisional_raw is None:
                parsed, parse_status = parse_a1_v4_2_plan(
                    completion.text,
                    q_max=q_max,
                    static_plans=static_plans,
                )
                if parse_status != "ok" or parsed is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=parse_status,
                            parse_status=parse_status,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                        )
                    )
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        error_code=parse_status,
                        details=[{"error_code": parse_status}],
                    )
                    continue
                self._v4_selected_static_plan_id = str(parsed["static_plan_id"])
                plan = static_plan_by_id(static_plans, self._v4_selected_static_plan_id)
                assert plan is not None
                locks, _cost, lock_reason = static_locks_and_cost(
                    validator=env.validator,
                    anchor=env.starting_case.features,
                    catalog=catalog,
                    static_choice_ids=plan.static_choice_ids,
                    m_max=self.budget.m_max,
                )
                if locks is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=lock_reason,
                            parse_status=lock_reason,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                        )
                    )
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        error_code=lock_reason,
                        details=[{"error_code": lock_reason}],
                    )
                    continue
                self._v4_selected_locks = locks
                provisional_raw = list(parsed["candidates"])
                raw_candidates = list(parsed["candidates"])
            else:
                requested = tuple(
                    int(index)
                    for index in (local_feedback or {}).get(
                        "invalid_candidate_indices", ()
                    )
                )
                pinned_plan = static_plan_by_id(
                    static_plans, str(self._v4_selected_static_plan_id or "")
                )
                pinned_residual = int(pinned_plan.residual_m) if pinned_plan else 0
                replacements, parse_status = parse_a1_v4_2_slot_replacements(
                    completion.text,
                    requested_indices=requested,
                    residual_m=pinned_residual,
                )
                if parse_status != "ok" or replacements is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=parse_status,
                            parse_status=parse_status,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                            invalid_candidate_indices=requested,
                        )
                    )
                    local_feedback = self._v4_2_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        invalid_indices=requested,
                        preserved_indices=(),
                        provisional_raw=provisional_raw,
                        details=[{"error_code": parse_status}],
                    )
                    continue
                for replacement in replacements:
                    index = int(replacement["candidate_index"])
                    provisional_raw[index - 1] = {
                        "strategy_label": replacement["strategy_label"],
                        "selections": dict(replacement["selections"]),
                    }
                    repaired_indices = repaired_indices + (index,)

            accepted, slot_errors, reject_counts = self._evaluate_v4_2_portfolio(
                env, provisional_raw or [], q_max
            )
            for key, value in reject_counts.items():
                self._governance_reject_counts[key] = (
                    self._governance_reject_counts.get(key, 0) + int(value)
                )
            invalid_indices = tuple(
                error["candidate_index"] for error in slot_errors
            )
            preserved_indices = tuple(
                index
                for index, proposal in enumerate(accepted, start=1)
                if proposal is not None
            )
            if not slot_errors and len(accepted) == q_max:
                self._frozen_proposals = [
                    item for item in accepted if item is not None
                ]
                self._selected_response_index = attempt_idx
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=None,
                        parse_status="ok",
                        raw_response_path=raw_path,
                        selected_for_plan=True,
                        call_kind=call_kind,
                        preserved_candidate_indices=preserved_indices,
                        repaired_candidate_indices=repaired_indices,
                    )
                )
                break
            self._retry_ledger.append(
                A1AttemptRecord(
                    attempt_index=attempt_idx,
                    timestamp=timestamp,
                    retry_reason="local_validation_failed",
                    parse_status="local_validation_failed",
                    raw_response_path=raw_path,
                    selected_for_plan=False,
                    call_kind=call_kind,
                    invalid_candidate_indices=invalid_indices,
                    preserved_candidate_indices=preserved_indices,
                    repaired_candidate_indices=repaired_indices,
                )
            )
            local_feedback = self._v4_2_local_feedback(
                attempt=attempt_idx + 1,
                max_attempts=max_attempts,
                invalid_indices=invalid_indices,
                preserved_indices=preserved_indices,
                provisional_raw=provisional_raw or [],
                details=slot_errors,
            )

        self._sequence_prepared = True
        if int(env.ledger.q_remaining) != q_before:
            raise A1PlannerError(
                "Local generation must not consume Q; ledger changed unexpectedly."
            )
        if int(env.attempts_used) != steps_before:
            raise A1PlannerError(
                "Local generation must not call env.step (and therefore must not call D1)."
            )
        q_used_before_freeze = int(env.budget.q_max) - int(env.ledger.q_remaining)
        d1_calls_before_freeze = int(env.attempts_used) - steps_before
        llm_call_count = len(self._retry_ledger)
        retry_count = max(0, llm_call_count - 1)
        if not self._frozen_proposals:
            self._pending_stop_reason = (
                "local_generation_exhausted"
                if provisional_raw is not None
                else "no_feasible_plan"
            )
        final_parse_status = (
            "ok"
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].parse_status
                if self._retry_ledger
                else "empty"
            )
        )
        selected_raw_path = (
            self._retry_ledger[self._selected_response_index].raw_response_path
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].raw_response_path
                if self._retry_ledger
                else None
            )
        )
        plan_path, call_path, ledger_path, prompt_text_path, model_config_path = (
            self._write_artefacts(
                env,
                model=selected_model,
                thinking_disabled=selected_thinking,
                parse_status=final_parse_status,
                retry_count=retry_count,
                llm_call_count=llm_call_count,
                raw_candidates=raw_candidates,
                total_latency_ms=float(total_latency_ms),
                selected_raw_path=selected_raw_path,
            )
        )
        self._call_record = A1CallRecord(
            model=selected_model,
            thinking_disabled=selected_thinking,
            prompt_version=self.prompt_version,
            prompt_hash=self._prompt_hash,
            config_hash=self._config_hash,
            latency_ms=float(total_latency_ms),
            retry_count=retry_count,
            llm_call_count=llm_call_count,
            parse_status=final_parse_status,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            cached_tokens=total_cached_tokens,
            estimated_cost_usd=estimate_flash_cost_usd(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_tokens=total_cached_tokens,
            ),
            raw_response_path=selected_raw_path,
            parsed_plan_path=plan_path,
            retry_ledger_path=ledger_path,
            prompt_text_path=prompt_text_path,
            model_config_path=model_config_path,
            selected_response_index=self._selected_response_index,
            n_raw_candidates=len(raw_candidates),
            n_frozen_candidates=len(self._frozen_proposals),
            governance_reject_counts=dict(self._governance_reject_counts),
            retry_ledger=tuple(self._retry_ledger),
            q_max=q_max,
            provisional_candidate_count=(
                len(provisional_raw) if provisional_raw is not None else None
            ),
            local_repair_count=sum(
                item.call_kind == "repair" for item in self._retry_ledger
            ),
            q_used_before_freeze=q_used_before_freeze,
            d1_calls_before_freeze=d1_calls_before_freeze,
        )
        if call_path is not None:
            Path(call_path).write_text(
                json.dumps(self._call_record.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return self.frozen_proposals


    def _prepare_v4_1_frozen_sequence(
        self, env: AttackEnvironment
    ) -> tuple[AttackProposal, ...]:
        """Freeze V4.1 after bounded local slot repair under unique action slots."""
        if self._sequence_prepared:
            return self.frozen_proposals
        if env.done:
            self._sequence_prepared = True
            self._pending_stop_reason = "q_exhausted"
            return ()
        q_max = min(int(env.budget.q_max), int(self.budget.q_max))
        self._episode_seed = derive_episode_seed(
            self.experiment_seed, env.starting_case.case_id, self.attacker_id
        )
        self._frozen_proposals = []
        self._submit_index = 0
        self._pending_stop_reason = None
        self._call_record = None
        self._raw_response_text = ""
        self._governance_reject_counts = {}
        self._retry_ledger = []
        self._selected_response_index = None
        self._v4_selected_static_plan_id = None
        self._v4_selected_locks = None
        self._v4_1_action_slots = None

        catalog = build_v4_choice_catalog(
            validator=env.validator,
            pool=self.reference_pool,
            anchor=env.starting_case.features,
        )
        static_plans = build_v4_static_plan_options(
            validator=env.validator,
            pool=self.reference_pool,
            anchor=env.starting_case.features,
            catalog=catalog,
            m_max=self.budget.m_max,
            q_max=q_max,
        )
        action_slots = build_v4_1_action_slots(catalog)
        self._v4_catalog = catalog
        self._v4_static_plans = static_plans
        self._v4_1_action_slots = action_slots
        if not static_plans:
            self._sequence_prepared = True
            self._pending_stop_reason = "no_feasible_plan"
            return ()

        observation = env.observation()
        self._prompt_payload = build_v4_1_prompt_payload(
            validator=env.validator,
            pool=self.reference_pool,
            budget=self.budget,
            q_max=q_max,
            visible_anchor=observation.visible_fields,
            case_id=str(observation.case_id),
            catalog=catalog,
            static_plans=static_plans,
            action_slots=action_slots,
        )
        client = self.llm_client or DeepSeekPlannerClient(default_model=self.model)
        run_dir = _episode_run_dir(env)
        max_attempts = max(
            1,
            min(
                int(self.max_local_generation_attempts),
                int(self.max_parse_retries) + 1,
            ),
        )
        q_before = int(env.ledger.q_remaining)
        steps_before = int(env.attempts_used)
        provisional_raw: list[dict[str, Any]] | None = None
        local_feedback: dict[str, Any] | None = None
        raw_candidates: list[dict[str, Any]] = []
        total_latency_ms = total_prompt_tokens = total_completion_tokens = total_cached_tokens = 0
        selected_model, selected_thinking = self.model, bool(self.thinking_disabled)
        allowed_plan_ids = [plan.static_plan_id for plan in static_plans]

        for attempt_idx in range(max_attempts):
            call_kind = "initial_plan" if provisional_raw is None else "repair"
            messages = render_a1_messages(
                self._prompt_payload, local_generation_feedback=local_feedback
            )
            self._prompt_text = format_a1_prompt_text(messages)
            self._prompt_hash = hash_a1_prompt_text(self._prompt_text)
            timestamp = datetime.now(timezone.utc).isoformat()
            raw_path: str | None = None
            try:
                completion = client.complete(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    thinking_disabled=self.thinking_disabled,
                    reasoning_effort=(
                        None
                        if self.thinking_disabled
                        else (self.reasoning_effort or "max")
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                reason = _classify_transport_error(exc)
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=reason,
                        parse_status=reason,
                        raw_response_path=None,
                        selected_for_plan=False,
                        transport_error=f"{type(exc).__name__}: {exc}",
                        call_kind="transport/parse",
                    )
                )
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1,
                    max_attempts=max_attempts,
                    error_code=reason,
                    details=[{"error_code": reason}],
                )
                continue
            total_latency_ms += float(completion.latency_ms)
            total_prompt_tokens += int(completion.prompt_tokens)
            total_completion_tokens += int(completion.completion_tokens)
            total_cached_tokens += int(completion.cached_tokens)
            selected_model = completion.model
            selected_thinking = bool(completion.thinking_disabled)
            self._raw_response_text = completion.text
            raw_path = _persist_raw_attempt(run_dir, attempt_idx, completion.text)

            repaired_indices: tuple[int, ...] = ()
            if provisional_raw is None:
                parsed, parse_status = parse_a1_v4_1_plan(
                    completion.text,
                    q_max=q_max,
                    allowed_static_plan_ids=allowed_plan_ids,
                )
                if parse_status != "ok" or parsed is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=parse_status,
                            parse_status=parse_status,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                        )
                    )
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        error_code=parse_status,
                        details=[{"error_code": parse_status}],
                    )
                    continue
                self._v4_selected_static_plan_id = str(parsed["static_plan_id"])
                plan = static_plan_by_id(static_plans, self._v4_selected_static_plan_id)
                assert plan is not None
                locks, _cost, lock_reason = static_locks_and_cost(
                    validator=env.validator,
                    anchor=env.starting_case.features,
                    catalog=catalog,
                    static_choice_ids=plan.static_choice_ids,
                    m_max=self.budget.m_max,
                )
                if locks is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=lock_reason,
                            parse_status=lock_reason,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                        )
                    )
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        error_code=lock_reason,
                        details=[{"error_code": lock_reason}],
                    )
                    continue
                self._v4_selected_locks = locks
                provisional_raw = list(parsed["candidates"])
                raw_candidates = list(parsed["candidates"])
            else:
                requested = tuple(
                    int(index)
                    for index in (local_feedback or {}).get(
                        "invalid_candidate_indices", ()
                    )
                )
                replacements, parse_status = parse_a1_v4_1_slot_replacements(
                    completion.text, requested_indices=requested
                )
                if parse_status != "ok" or replacements is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=parse_status,
                            parse_status=parse_status,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                            invalid_candidate_indices=requested,
                        )
                    )
                    local_feedback = self._v4_1_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        invalid_indices=requested,
                        preserved_indices=(),
                        provisional_raw=provisional_raw,
                        details=[{"error_code": parse_status}],
                    )
                    continue
                for replacement in replacements:
                    index = int(replacement["candidate_index"])
                    provisional_raw[index - 1] = {
                        "strategy_label": replacement["strategy_label"],
                        "selections": dict(replacement["selections"]),
                    }
                    repaired_indices = repaired_indices + (index,)

            accepted, slot_errors, reject_counts = self._evaluate_v4_1_portfolio(
                env, provisional_raw or [], q_max
            )
            for key, value in reject_counts.items():
                self._governance_reject_counts[key] = (
                    self._governance_reject_counts.get(key, 0) + int(value)
                )
            invalid_indices = tuple(
                error["candidate_index"] for error in slot_errors
            )
            preserved_indices = tuple(
                index
                for index, proposal in enumerate(accepted, start=1)
                if proposal is not None
            )
            if not slot_errors and len(accepted) == q_max:
                self._frozen_proposals = [
                    item for item in accepted if item is not None
                ]
                self._selected_response_index = attempt_idx
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=None,
                        parse_status="ok",
                        raw_response_path=raw_path,
                        selected_for_plan=True,
                        call_kind=call_kind,
                        preserved_candidate_indices=preserved_indices,
                        repaired_candidate_indices=repaired_indices,
                    )
                )
                break
            self._retry_ledger.append(
                A1AttemptRecord(
                    attempt_index=attempt_idx,
                    timestamp=timestamp,
                    retry_reason="local_validation_failed",
                    parse_status="local_validation_failed",
                    raw_response_path=raw_path,
                    selected_for_plan=False,
                    call_kind=call_kind,
                    invalid_candidate_indices=invalid_indices,
                    preserved_candidate_indices=preserved_indices,
                    repaired_candidate_indices=repaired_indices,
                )
            )
            local_feedback = self._v4_1_local_feedback(
                attempt=attempt_idx + 1,
                max_attempts=max_attempts,
                invalid_indices=invalid_indices,
                preserved_indices=preserved_indices,
                provisional_raw=provisional_raw or [],
                details=slot_errors,
            )

        self._sequence_prepared = True
        if int(env.ledger.q_remaining) != q_before:
            raise A1PlannerError(
                "Local generation must not consume Q; ledger changed unexpectedly."
            )
        if int(env.attempts_used) != steps_before:
            raise A1PlannerError(
                "Local generation must not call env.step (and therefore must not call D1)."
            )
        q_used_before_freeze = int(env.budget.q_max) - int(env.ledger.q_remaining)
        d1_calls_before_freeze = int(env.attempts_used) - steps_before
        llm_call_count = len(self._retry_ledger)
        retry_count = max(0, llm_call_count - 1)
        if not self._frozen_proposals:
            self._pending_stop_reason = (
                "local_generation_exhausted"
                if provisional_raw is not None
                else "no_feasible_plan"
            )
        final_parse_status = (
            "ok"
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].parse_status
                if self._retry_ledger
                else "empty"
            )
        )
        selected_raw_path = (
            self._retry_ledger[self._selected_response_index].raw_response_path
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].raw_response_path
                if self._retry_ledger
                else None
            )
        )
        plan_path, call_path, ledger_path, prompt_text_path, model_config_path = (
            self._write_artefacts(
                env,
                model=selected_model,
                thinking_disabled=selected_thinking,
                parse_status=final_parse_status,
                retry_count=retry_count,
                llm_call_count=llm_call_count,
                raw_candidates=raw_candidates,
                total_latency_ms=float(total_latency_ms),
                selected_raw_path=selected_raw_path,
            )
        )
        self._call_record = A1CallRecord(
            model=selected_model,
            thinking_disabled=selected_thinking,
            prompt_version=self.prompt_version,
            prompt_hash=self._prompt_hash,
            config_hash=self._config_hash,
            latency_ms=float(total_latency_ms),
            retry_count=retry_count,
            llm_call_count=llm_call_count,
            parse_status=final_parse_status,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            cached_tokens=total_cached_tokens,
            estimated_cost_usd=estimate_flash_cost_usd(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_tokens=total_cached_tokens,
            ),
            raw_response_path=selected_raw_path,
            parsed_plan_path=plan_path,
            retry_ledger_path=ledger_path,
            prompt_text_path=prompt_text_path,
            model_config_path=model_config_path,
            selected_response_index=self._selected_response_index,
            n_raw_candidates=len(raw_candidates),
            n_frozen_candidates=len(self._frozen_proposals),
            governance_reject_counts=dict(self._governance_reject_counts),
            retry_ledger=tuple(self._retry_ledger),
            q_max=q_max,
            provisional_candidate_count=(
                len(provisional_raw) if provisional_raw is not None else None
            ),
            local_repair_count=sum(
                item.call_kind == "repair" for item in self._retry_ledger
            ),
            q_used_before_freeze=q_used_before_freeze,
            d1_calls_before_freeze=d1_calls_before_freeze,
        )
        if call_path is not None:
            Path(call_path).write_text(
                json.dumps(self._call_record.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return self.frozen_proposals

    def _prepare_v4_frozen_sequence(
        self, env: AttackEnvironment
    ) -> tuple[AttackProposal, ...]:
        """Freeze V4 after bounded local slot repair under the hard contract."""
        if self._sequence_prepared:
            return self.frozen_proposals
        if env.done:
            self._sequence_prepared = True
            self._pending_stop_reason = "q_exhausted"
            return ()
        q_max = min(int(env.budget.q_max), int(self.budget.q_max))
        self._episode_seed = derive_episode_seed(
            self.experiment_seed, env.starting_case.case_id, self.attacker_id
        )
        self._frozen_proposals = []
        self._submit_index = 0
        self._pending_stop_reason = None
        self._call_record = None
        self._raw_response_text = ""
        self._governance_reject_counts = {}
        self._retry_ledger = []
        self._selected_response_index = None
        self._v4_selected_static_plan_id = None
        self._v4_selected_locks = None

        catalog = build_v4_choice_catalog(
            validator=env.validator,
            pool=self.reference_pool,
            anchor=env.starting_case.features,
        )
        static_plans = build_v4_static_plan_options(
            validator=env.validator,
            pool=self.reference_pool,
            anchor=env.starting_case.features,
            catalog=catalog,
            m_max=self.budget.m_max,
            q_max=q_max,
        )
        self._v4_catalog = catalog
        self._v4_static_plans = static_plans
        if not static_plans:
            self._sequence_prepared = True
            self._pending_stop_reason = "no_feasible_plan"
            return ()

        observation = env.observation()
        self._prompt_payload = build_v4_prompt_payload(
            validator=env.validator,
            pool=self.reference_pool,
            budget=self.budget,
            q_max=q_max,
            visible_anchor=observation.visible_fields,
            case_id=str(observation.case_id),
            catalog=catalog,
            static_plans=static_plans,
        )
        client = self.llm_client or DeepSeekPlannerClient(default_model=self.model)
        run_dir = _episode_run_dir(env)
        max_attempts = max(
            1,
            min(
                int(self.max_local_generation_attempts),
                int(self.max_parse_retries) + 1,
            ),
        )
        q_before = int(env.ledger.q_remaining)
        steps_before = int(env.attempts_used)
        provisional_raw: list[dict[str, Any]] | None = None
        local_feedback: dict[str, Any] | None = None
        raw_candidates: list[dict[str, Any]] = []
        total_latency_ms = total_prompt_tokens = total_completion_tokens = total_cached_tokens = 0
        selected_model, selected_thinking = self.model, bool(self.thinking_disabled)
        allowed_plan_ids = [plan.static_plan_id for plan in static_plans]

        for attempt_idx in range(max_attempts):
            call_kind = "initial_plan" if provisional_raw is None else "repair"
            messages = render_a1_messages(
                self._prompt_payload, local_generation_feedback=local_feedback
            )
            self._prompt_text = format_a1_prompt_text(messages)
            self._prompt_hash = hash_a1_prompt_text(self._prompt_text)
            timestamp = datetime.now(timezone.utc).isoformat()
            raw_path: str | None = None
            try:
                completion = client.complete(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    thinking_disabled=self.thinking_disabled,
                    reasoning_effort=(
                        None
                        if self.thinking_disabled
                        else (self.reasoning_effort or "max")
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                reason = _classify_transport_error(exc)
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=reason,
                        parse_status=reason,
                        raw_response_path=None,
                        selected_for_plan=False,
                        transport_error=f"{type(exc).__name__}: {exc}",
                        call_kind="transport/parse",
                    )
                )
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1,
                    max_attempts=max_attempts,
                    error_code=reason,
                    details=[{"error_code": reason}],
                )
                continue
            total_latency_ms += float(completion.latency_ms)
            total_prompt_tokens += int(completion.prompt_tokens)
            total_completion_tokens += int(completion.completion_tokens)
            total_cached_tokens += int(completion.cached_tokens)
            selected_model = completion.model
            selected_thinking = bool(completion.thinking_disabled)
            self._raw_response_text = completion.text
            raw_path = _persist_raw_attempt(run_dir, attempt_idx, completion.text)

            repaired_indices: tuple[int, ...] = ()
            if provisional_raw is None:
                parsed, parse_status = parse_a1_v4_plan(
                    completion.text,
                    q_max=q_max,
                    allowed_static_plan_ids=allowed_plan_ids,
                )
                if parse_status != "ok" or parsed is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=parse_status,
                            parse_status=parse_status,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                        )
                    )
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        error_code=parse_status,
                        details=[{"error_code": parse_status}],
                    )
                    continue
                self._v4_selected_static_plan_id = str(parsed["static_plan_id"])
                plan = static_plan_by_id(static_plans, self._v4_selected_static_plan_id)
                assert plan is not None
                locks, _cost, lock_reason = static_locks_and_cost(
                    validator=env.validator,
                    anchor=env.starting_case.features,
                    catalog=catalog,
                    static_choice_ids=plan.static_choice_ids,
                    m_max=self.budget.m_max,
                )
                if locks is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=lock_reason,
                            parse_status=lock_reason,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                        )
                    )
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        error_code=lock_reason,
                        details=[{"error_code": lock_reason}],
                    )
                    continue
                self._v4_selected_locks = locks
                provisional_raw = list(parsed["candidates"])
                raw_candidates = list(parsed["candidates"])
            else:
                requested = tuple(
                    int(index)
                    for index in (local_feedback or {}).get(
                        "invalid_candidate_indices", ()
                    )
                )
                replacements, parse_status = parse_a1_v4_slot_replacements(
                    completion.text, requested_indices=requested
                )
                if parse_status != "ok" or replacements is None:
                    self._retry_ledger.append(
                        A1AttemptRecord(
                            attempt_index=attempt_idx,
                            timestamp=timestamp,
                            retry_reason=parse_status,
                            parse_status=parse_status,
                            raw_response_path=raw_path,
                            selected_for_plan=False,
                            call_kind=call_kind,
                            invalid_candidate_indices=requested,
                        )
                    )
                    local_feedback = self._v4_local_feedback(
                        attempt=attempt_idx + 1,
                        max_attempts=max_attempts,
                        invalid_indices=requested,
                        preserved_indices=(),
                        provisional_raw=provisional_raw,
                        details=[{"error_code": parse_status}],
                    )
                    continue
                for replacement in replacements:
                    index = int(replacement["candidate_index"])
                    provisional_raw[index - 1] = {
                        "strategy_label": replacement["strategy_label"],
                        "choice_ids": list(replacement["choice_ids"]),
                    }
                    repaired_indices = repaired_indices + (index,)

            accepted, slot_errors, reject_counts = self._evaluate_v4_portfolio(
                env, provisional_raw or [], q_max
            )
            self._governance_reject_counts = reject_counts
            invalid_indices = tuple(
                error["candidate_index"] for error in slot_errors
            )
            preserved_indices = tuple(
                index
                for index, proposal in enumerate(accepted, start=1)
                if proposal is not None
            )
            if not slot_errors and len(accepted) == q_max:
                self._frozen_proposals = [
                    item for item in accepted if item is not None
                ]
                self._selected_response_index = attempt_idx
                self._retry_ledger.append(
                    A1AttemptRecord(
                        attempt_index=attempt_idx,
                        timestamp=timestamp,
                        retry_reason=None,
                        parse_status="ok",
                        raw_response_path=raw_path,
                        selected_for_plan=True,
                        call_kind=call_kind,
                        preserved_candidate_indices=preserved_indices,
                        repaired_candidate_indices=repaired_indices,
                    )
                )
                break
            self._retry_ledger.append(
                A1AttemptRecord(
                    attempt_index=attempt_idx,
                    timestamp=timestamp,
                    retry_reason="local_validation_failed",
                    parse_status="local_validation_failed",
                    raw_response_path=raw_path,
                    selected_for_plan=False,
                    call_kind=call_kind,
                    invalid_candidate_indices=invalid_indices,
                    preserved_candidate_indices=preserved_indices,
                    repaired_candidate_indices=repaired_indices,
                )
            )
            local_feedback = self._v4_local_feedback(
                attempt=attempt_idx + 1,
                max_attempts=max_attempts,
                invalid_indices=invalid_indices,
                preserved_indices=preserved_indices,
                provisional_raw=provisional_raw or [],
                details=slot_errors,
            )

        self._sequence_prepared = True
        if int(env.ledger.q_remaining) != q_before:
            raise A1PlannerError(
                "Local generation must not consume Q; ledger changed unexpectedly."
            )
        if int(env.attempts_used) != steps_before:
            raise A1PlannerError(
                "Local generation must not call env.step (and therefore must not call D1)."
            )
        q_used_before_freeze = int(env.budget.q_max) - int(env.ledger.q_remaining)
        d1_calls_before_freeze = int(env.attempts_used) - steps_before
        llm_call_count = len(self._retry_ledger)
        retry_count = max(0, llm_call_count - 1)
        if not self._frozen_proposals:
            self._pending_stop_reason = (
                "local_generation_exhausted"
                if provisional_raw is not None
                else "no_feasible_plan"
            )
        final_parse_status = (
            "ok"
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].parse_status
                if self._retry_ledger
                else "empty"
            )
        )
        selected_raw_path = (
            self._retry_ledger[self._selected_response_index].raw_response_path
            if self._selected_response_index is not None
            else (
                self._retry_ledger[-1].raw_response_path
                if self._retry_ledger
                else None
            )
        )
        plan_path, call_path, ledger_path, prompt_text_path, model_config_path = (
            self._write_artefacts(
                env,
                model=selected_model,
                thinking_disabled=selected_thinking,
                parse_status=final_parse_status,
                retry_count=retry_count,
                llm_call_count=llm_call_count,
                raw_candidates=raw_candidates,
                total_latency_ms=float(total_latency_ms),
                selected_raw_path=selected_raw_path,
            )
        )
        self._call_record = A1CallRecord(
            model=selected_model,
            thinking_disabled=selected_thinking,
            prompt_version=self.prompt_version,
            prompt_hash=self._prompt_hash,
            config_hash=self._config_hash,
            latency_ms=float(total_latency_ms),
            retry_count=retry_count,
            llm_call_count=llm_call_count,
            parse_status=final_parse_status,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            cached_tokens=total_cached_tokens,
            estimated_cost_usd=estimate_flash_cost_usd(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cached_tokens=total_cached_tokens,
            ),
            raw_response_path=selected_raw_path,
            parsed_plan_path=plan_path,
            retry_ledger_path=ledger_path,
            prompt_text_path=prompt_text_path,
            model_config_path=model_config_path,
            selected_response_index=self._selected_response_index,
            n_raw_candidates=len(raw_candidates),
            n_frozen_candidates=len(self._frozen_proposals),
            governance_reject_counts=dict(self._governance_reject_counts),
            retry_ledger=tuple(self._retry_ledger),
            q_max=q_max,
            provisional_candidate_count=(
                len(provisional_raw) if provisional_raw is not None else None
            ),
            local_repair_count=sum(
                item.call_kind == "repair" for item in self._retry_ledger
            ),
            q_used_before_freeze=q_used_before_freeze,
            d1_calls_before_freeze=d1_calls_before_freeze,
        )
        if call_path is not None:
            Path(call_path).write_text(
                json.dumps(self._call_record.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return self.frozen_proposals

    def _prepare_v3_frozen_sequence(
        self, env: AttackEnvironment
    ) -> tuple[AttackProposal, ...]:
        """Freeze V3 after bounded local slot repair, never after D1 feedback."""
        if self._sequence_prepared:
            return self.frozen_proposals
        if env.done:
            self._sequence_prepared = True
            self._pending_stop_reason = "q_exhausted"
            return ()
        q_max = min(int(env.budget.q_max), int(self.budget.q_max))
        self._episode_seed = derive_episode_seed(
            self.experiment_seed, env.starting_case.case_id, self.attacker_id
        )
        self._frozen_proposals = []
        self._submit_index = 0
        self._pending_stop_reason = None
        self._call_record = None
        self._raw_response_text = ""
        self._governance_reject_counts = {}
        self._retry_ledger = []
        self._selected_response_index = None
        self._prompt_payload = build_a1_prompt_payload(
            env=env, reference_pool=self.reference_pool, budget=self.budget,
            q_max=q_max, prompt_version=self.prompt_version,
        )
        client = self.llm_client or DeepSeekPlannerClient(default_model=self.model)
        run_dir = _episode_run_dir(env)
        max_attempts = max(1, min(
            int(self.max_local_generation_attempts), int(self.max_parse_retries) + 1
        ))
        # AttackerEpisode exposes Q via ledger.q_remaining / budget.q_max and
        # submission/D1 accounting via attempts_used (D1 only runs inside step).
        q_before = int(env.ledger.q_remaining)
        steps_before = int(env.attempts_used)
        provisional_raw: list[dict[str, Any]] | None = None
        local_feedback: dict[str, Any] | None = None
        raw_candidates: list[dict[str, Any]] = []
        total_latency_ms = total_prompt_tokens = total_completion_tokens = total_cached_tokens = 0
        selected_model, selected_thinking = self.model, bool(self.thinking_disabled)

        for attempt_idx in range(max_attempts):
            call_kind = "initial_plan" if provisional_raw is None else "repair"
            messages = render_a1_messages(
                self._prompt_payload, local_generation_feedback=local_feedback
            )
            self._prompt_text = format_a1_prompt_text(messages)
            self._prompt_hash = hash_a1_prompt_text(self._prompt_text)
            timestamp = datetime.now(timezone.utc).isoformat()
            raw_path: str | None = None
            try:
                completion = client.complete(
                    messages, model=self.model, temperature=self.temperature,
                    top_p=self.top_p, max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    thinking_disabled=self.thinking_disabled,
                    reasoning_effort=(
                        None
                        if self.thinking_disabled
                        else (self.reasoning_effort or "max")
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                reason = _classify_transport_error(exc)
                self._retry_ledger.append(A1AttemptRecord(
                    attempt_index=attempt_idx, timestamp=timestamp, retry_reason=reason,
                    parse_status=reason, raw_response_path=None, selected_for_plan=False,
                    transport_error=f"{type(exc).__name__}: {exc}", call_kind="transport/parse",
                ))
                local_feedback = self._safe_local_feedback(
                    attempt=attempt_idx + 1, max_attempts=max_attempts,
                    error_code=reason, details=[{"error_code": reason}],
                )
                continue
            total_latency_ms += float(completion.latency_ms)
            total_prompt_tokens += int(completion.prompt_tokens)
            total_completion_tokens += int(completion.completion_tokens)
            total_cached_tokens += int(completion.cached_tokens)
            selected_model, selected_thinking = completion.model, bool(completion.thinking_disabled)
            self._raw_response_text = completion.text
            raw_path = _persist_raw_attempt(run_dir, attempt_idx, completion.text)

            invalid_indices: tuple[int, ...] = ()
            preserved_indices: tuple[int, ...] = ()
            repaired_indices: tuple[int, ...] = ()
            if provisional_raw is None:
                parsed, parse_status = parse_a1_candidates(
                    completion.text, prompt_version=PROMPT_VERSION_V3, m_max=self.budget.m_max
                )
                if parse_status == "ok":
                    provisional_raw = parsed
                    raw_candidates = parsed
                else:
                    self._retry_ledger.append(A1AttemptRecord(
                        attempt_index=attempt_idx, timestamp=timestamp,
                        retry_reason=parse_status, parse_status=parse_status,
                        raw_response_path=raw_path, selected_for_plan=False,
                        call_kind=call_kind,
                    ))
                    local_feedback = self._safe_local_feedback(
                        attempt=attempt_idx + 1, max_attempts=max_attempts,
                        error_code=parse_status, details=[{"error_code": parse_status}],
                    )
                    continue
            else:
                replacements, parse_status = parse_a1_slot_replacements(
                    completion.text, m_max=self.budget.m_max
                )
                requested = tuple(
                    int(index) for index in (local_feedback or {}).get(
                        "invalid_candidate_indices", ()
                    )
                )
                if parse_status != "ok":
                    invalid_indices = requested
                else:
                    provisional_raw, invalid_indices = self._merge_v3_replacements(
                        provisional_raw, replacements, requested
                    )
                    repaired_indices = tuple(
                        replacement["candidate_index"] for replacement in replacements
                        if replacement["candidate_index"] in requested
                    )
                if parse_status != "ok" or invalid_indices:
                    self._retry_ledger.append(A1AttemptRecord(
                        attempt_index=attempt_idx, timestamp=timestamp,
                        retry_reason=parse_status if parse_status != "ok" else "local_validation_failed",
                        parse_status=parse_status if parse_status != "ok" else "local_validation_failed",
                        raw_response_path=raw_path, selected_for_plan=False, call_kind=call_kind,
                        invalid_candidate_indices=invalid_indices or requested,
                        repaired_candidate_indices=repaired_indices,
                    ))
                    local_feedback = self._v3_local_feedback(
                        attempt=attempt_idx + 1, max_attempts=max_attempts,
                        invalid_indices=invalid_indices or requested, preserved_indices=(),
                        provisional_raw=provisional_raw, details=[
                            {"candidate_index": i, "error_code": "replacement_indices_invalid"}
                            for i in (invalid_indices or requested)
                        ],
                    )
                    continue

            accepted, slot_errors, reject_counts = self._evaluate_v3_portfolio(
                env, provisional_raw or [], q_max
            )
            self._governance_reject_counts = reject_counts
            invalid_indices = tuple(error["candidate_index"] for error in slot_errors)
            preserved_indices = tuple(
                index for index, proposal in enumerate(accepted, start=1)
                if proposal is not None
            )
            if not slot_errors and len(accepted) == q_max:
                self._frozen_proposals = [item for item in accepted if item is not None]
                self._selected_response_index = attempt_idx
                self._retry_ledger.append(A1AttemptRecord(
                    attempt_index=attempt_idx, timestamp=timestamp, retry_reason=None,
                    parse_status="ok", raw_response_path=raw_path, selected_for_plan=True,
                    call_kind=call_kind, preserved_candidate_indices=preserved_indices,
                    repaired_candidate_indices=repaired_indices,
                ))
                break
            self._retry_ledger.append(A1AttemptRecord(
                attempt_index=attempt_idx, timestamp=timestamp,
                retry_reason="local_validation_failed", parse_status="local_validation_failed",
                raw_response_path=raw_path, selected_for_plan=False, call_kind=call_kind,
                invalid_candidate_indices=invalid_indices,
                preserved_candidate_indices=preserved_indices,
                repaired_candidate_indices=repaired_indices,
            ))
            local_feedback = self._v3_local_feedback(
                attempt=attempt_idx + 1, max_attempts=max_attempts,
                invalid_indices=invalid_indices, preserved_indices=preserved_indices,
                provisional_raw=provisional_raw or [], details=slot_errors,
            )

        self._sequence_prepared = True
        if int(env.ledger.q_remaining) != q_before:
            raise A1PlannerError("Local generation must not consume Q; ledger changed unexpectedly.")
        if int(env.attempts_used) != steps_before:
            raise A1PlannerError(
                "Local generation must not call env.step (and therefore must not call D1)."
            )
        q_used_before_freeze = int(env.budget.q_max) - int(env.ledger.q_remaining)
        d1_calls_before_freeze = int(env.attempts_used) - steps_before
        llm_call_count, retry_count = len(self._retry_ledger), max(0, len(self._retry_ledger) - 1)
        if not self._frozen_proposals:
            self._pending_stop_reason = (
                "local_generation_exhausted" if provisional_raw is not None
                else "no_feasible_plan"
            )
        final_parse_status = "ok" if self._selected_response_index is not None else (
            self._retry_ledger[-1].parse_status if self._retry_ledger else "empty"
        )
        selected_raw_path = (
            self._retry_ledger[self._selected_response_index].raw_response_path
            if self._selected_response_index is not None else (
                self._retry_ledger[-1].raw_response_path if self._retry_ledger else None
            )
        )
        plan_path, call_path, ledger_path, prompt_text_path, model_config_path = self._write_artefacts(
            env, model=selected_model, thinking_disabled=selected_thinking,
            parse_status=final_parse_status, retry_count=retry_count,
            llm_call_count=llm_call_count, raw_candidates=raw_candidates,
            total_latency_ms=float(total_latency_ms), selected_raw_path=selected_raw_path,
        )
        self._call_record = A1CallRecord(
            model=selected_model, thinking_disabled=selected_thinking,
            prompt_version=self.prompt_version, prompt_hash=self._prompt_hash,
            config_hash=self._config_hash, latency_ms=float(total_latency_ms),
            retry_count=retry_count, llm_call_count=llm_call_count,
            parse_status=final_parse_status, prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            cached_tokens=total_cached_tokens, estimated_cost_usd=estimate_flash_cost_usd(
                prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens,
                cached_tokens=total_cached_tokens,
            ), raw_response_path=selected_raw_path, parsed_plan_path=plan_path,
            retry_ledger_path=ledger_path, prompt_text_path=prompt_text_path,
            model_config_path=model_config_path,
            selected_response_index=self._selected_response_index,
            n_raw_candidates=len(raw_candidates), n_frozen_candidates=len(self._frozen_proposals),
            governance_reject_counts=dict(self._governance_reject_counts),
            retry_ledger=tuple(self._retry_ledger), q_max=q_max,
            provisional_candidate_count=(len(provisional_raw) if provisional_raw is not None else None),
            local_repair_count=sum(
                item.call_kind == "repair" for item in self._retry_ledger
            ), q_used_before_freeze=q_used_before_freeze,
            d1_calls_before_freeze=d1_calls_before_freeze,
        )
        if call_path is not None:
            Path(call_path).write_text(
                json.dumps(self._call_record.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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

    def _merge_v3_replacements(
        self,
        provisional_raw: list[dict[str, Any]],
        replacements: Sequence[Mapping[str, Any]],
        requested_indices: Sequence[int],
    ) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
        """Merge only requested V3 slots; preserved raw dict identities remain intact."""
        requested = tuple(sorted(set(int(index) for index in requested_indices)))
        by_index: dict[int, Mapping[str, Any]] = {}
        invalid: set[int] = set()
        for replacement in replacements:
            index = int(replacement["candidate_index"])
            if index not in requested or index in by_index:
                invalid.add(index)
            else:
                by_index[index] = replacement
        invalid.update(index for index in requested if index not in by_index)
        if invalid:
            return provisional_raw, tuple(sorted(invalid))
        merged = list(provisional_raw)
        for index in requested:
            replacement = by_index[index]
            item = {
                "strategy_label": replacement["strategy_label"],
                "changes": dict(replacement["changes"]),
            }
            while len(merged) < index:
                merged.append({})
            merged[index - 1] = item
        return merged, ()

    def _evaluate_v3_portfolio(
        self,
        env: AttackEnvironment,
        provisional_raw: Sequence[Mapping[str, Any]],
        q_max: int,
    ) -> tuple[list[AttackProposal | None], list[dict[str, Any]], dict[str, int]]:
        """Locally validate V3 slots while retaining every valid slot position."""
        if len(provisional_raw) != q_max:
            errors = [
                {"candidate_index": index, "error_code": "wrong_candidate_count"}
                for index in range(1, q_max + 1)
            ]
            return [], errors, {"wrong_candidate_count": q_max}
        accepted: list[AttackProposal | None] = [None] * q_max
        errors: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}
        seen: set[str] = set()
        locks: Mapping[str, Any] | None = None
        first_valid = False
        for index, item in enumerate(provisional_raw, start=1):
            raw_changes = item.get("changes") if isinstance(item, Mapping) else None
            strategy_label = item.get("strategy_label") if isinstance(item, Mapping) else None
            if not isinstance(raw_changes, Mapping):
                reason = "empty_changes"
                normalised = None
            else:
                normalised, reason = self._normalise_changes(env.validator, raw_changes)
            if normalised is None:
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            proposal = AttackProposal(
                changes=normalised,
                raw_command=(
                    f"{self.attacker_id}:episode_seed={self._episode_seed}:candidate={index}"
                ),
            )
            # Candidate one establishes locks. If it fails, later candidates are
            # structurally checked without speculative locks, but cannot freeze.
            candidate_locks = locks if (index > 1 and first_valid) else None
            candidate, reason, next_locks = self._validate_candidate(
                env, proposal, locked_values=candidate_locks, seen_fingerprints=seen,
                candidate_index=index,
                strategy_label=str(strategy_label) if strategy_label is not None else None,
            )
            if candidate is None:
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            accepted[index - 1] = candidate
            seen.add(str(candidate.research_meta["candidate_fingerprint"]))
            if index == 1:
                locks = next_locks
                first_valid = True
            elif not first_valid:
                # Candidate 1 still invalid: do not count later candidates as
                # preserveable despite their independent structural validity.
                accepted[index - 1] = None
        if not first_valid:
            # Later slots can never freeze without candidate-one locks.
            for index in range(2, q_max + 1):
                if accepted[index - 1] is None and not any(
                    error["candidate_index"] == index for error in errors
                ):
                    errors.append({"candidate_index": index, "error_code": "static_lock_unset"})
                    reject_counts["static_lock_unset"] = reject_counts.get("static_lock_unset", 0) + 1
        return accepted, errors, reject_counts

    def _evaluate_v4_portfolio(
        self,
        env: AttackEnvironment,
        provisional_raw: Sequence[Mapping[str, Any]],
        q_max: int,
    ) -> tuple[list[AttackProposal | None], list[dict[str, Any]], dict[str, int]]:
        """Validate V4 slots under the pinned static plan + residual_m."""
        catalog = self._v4_catalog
        plan = static_plan_by_id(
            self._v4_static_plans, str(self._v4_selected_static_plan_id or "")
        )
        locks = self._v4_selected_locks
        if catalog is None or plan is None or locks is None:
            return [], [{"candidate_index": 1, "error_code": "missing_static_plan"}], {
                "missing_static_plan": 1
            }
        if len(provisional_raw) != q_max:
            errors = [
                {"candidate_index": index, "error_code": "wrong_candidate_count"}
                for index in range(1, q_max + 1)
            ]
            return [], errors, {"wrong_candidate_count": q_max}

        accepted: list[AttackProposal | None] = [None] * q_max
        errors: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}
        seen: set[str] = set()
        allowed_query = set(plan.allowed_query_choice_ids)
        residual_m = int(plan.residual_m)

        for index, item in enumerate(provisional_raw, start=1):
            choice_ids = item.get("choice_ids") if isinstance(item, Mapping) else None
            strategy_label = (
                item.get("strategy_label") if isinstance(item, Mapping) else None
            )
            if not isinstance(choice_ids, list) or not choice_ids:
                reason = "empty_choice_ids"
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if any(str(cid) not in allowed_query for cid in choice_ids):
                reason = "query_choice_not_allowed"
                errors.append(
                    {
                        "candidate_index": index,
                        "error_code": reason,
                        "choice_ids": list(choice_ids),
                    }
                )
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if len(choice_ids) < 1 or len(choice_ids) > residual_m:
                reason = "budget_exceeded"
                errors.append(
                    {
                        "candidate_index": index,
                        "error_code": reason,
                        "n_choice_ids": len(choice_ids),
                        "residual_m": residual_m,
                    }
                )
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            changes, reason = resolve_choice_ids_to_changes(choice_ids, catalog)
            if changes is None:
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            static_in_query = False
            for action_key in changes:
                rule = env.validator.policy.field_for_action(action_key)
                if rule is not None and rule.is_episode_locked:
                    reason = "static_action_in_query_slot"
                    errors.append(
                        {
                            "candidate_index": index,
                            "error_code": reason,
                            "action_key": action_key,
                        }
                    )
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                    static_in_query = True
                    break
            if static_in_query:
                continue
            proposal = AttackProposal(
                changes=changes,
                raw_command=(
                    f"{self.attacker_id}:episode_seed={self._episode_seed}:candidate={index}"
                ),
            )
            candidate, fail_reason, _ = self._validate_candidate(
                env,
                proposal,
                locked_values=locks,
                seen_fingerprints=seen,
                candidate_index=index,
                strategy_label=str(strategy_label) if strategy_label is not None else None,
            )
            if candidate is None:
                errors.append({"candidate_index": index, "error_code": fail_reason})
                reject_counts[fail_reason] = reject_counts.get(fail_reason, 0) + 1
                continue
            meta = dict(candidate.research_meta)
            meta["prompt_version"] = PROMPT_VERSION_V4
            meta["static_plan_id"] = plan.static_plan_id
            meta["static_edit_cost"] = plan.static_edit_cost
            meta["residual_m"] = plan.residual_m
            meta["choice_ids"] = [str(cid) for cid in choice_ids]
            accepted[index - 1] = AttackProposal(
                changes=candidate.changes,
                raw_command=candidate.raw_command,
                research_meta=meta,
            )
            seen.add(str(meta["candidate_fingerprint"]))
        return accepted, errors, reject_counts

    def _v4_local_feedback(
        self,
        *,
        attempt: int,
        max_attempts: int,
        invalid_indices: Sequence[int],
        preserved_indices: Sequence[int],
        provisional_raw: Sequence[Mapping[str, Any]],
        details: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build V4 repair feedback with pinned static plan and residual_m."""
        plan = static_plan_by_id(
            self._v4_static_plans, str(self._v4_selected_static_plan_id or "")
        )
        preserved = [
            {
                "candidate_index": index,
                "strategy_label": provisional_raw[index - 1].get("strategy_label"),
                "choice_ids": provisional_raw[index - 1].get("choice_ids"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        occupied = [
            {
                "candidate_index": index,
                "choice_ids": provisional_raw[index - 1].get("choice_ids"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        return {
            "local_generation_attempt": int(attempt),
            "max_local_generation_attempts": int(max_attempts),
            "mode": "slot_repair",
            "error_code": "local_validation_failed",
            "static_plan_id": None if plan is None else plan.static_plan_id,
            "residual_m": None if plan is None else plan.residual_m,
            "allowed_query_choice_ids": (
                [] if plan is None else list(plan.allowed_query_choice_ids)
            ),
            "invalid_candidate_indices": list(invalid_indices),
            "preserved_candidate_indices": list(preserved_indices),
            "preserved_candidates_pinned": to_jsonable(preserved),
            "occupied_choice_sets": to_jsonable(occupied),
            "prior_local_rejections": [
                {
                    "candidate_index": item.get("candidate_index"),
                    "error_code": item.get("error_code"),
                }
                for item in details
            ],
            "note": (
                "Repair only invalid indices; static_plan_id is pinned; "
                "do not emit action_key/reference_id/changes. "
                "Local rule-compliance only; consumes no Q; does not call D1."
            ),
        }

    def _evaluate_v4_2_portfolio(
        self,
        env: AttackEnvironment,
        provisional_raw: Sequence[Mapping[str, Any]],
        q_max: int,
    ) -> tuple[list[AttackProposal | None], list[dict[str, Any]], dict[str, int]]:
        """Validate V4.2 slots under pinned static plan + unique action slots."""
        catalog = self._v4_catalog
        slots = self._v4_1_action_slots
        plan = static_plan_by_id(
            self._v4_static_plans, str(self._v4_selected_static_plan_id or "")
        )
        locks = self._v4_selected_locks
        if catalog is None or slots is None or plan is None or locks is None:
            return [], [{"candidate_index": 1, "error_code": "missing_static_plan"}], {
                "missing_static_plan": 1
            }
        if len(provisional_raw) != q_max:
            errors = [
                {"candidate_index": index, "error_code": "wrong_candidate_count"}
                for index in range(1, q_max + 1)
            ]
            return [], errors, {"wrong_candidate_count": q_max}

        accepted: list[AttackProposal | None] = [None] * q_max
        errors: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}
        seen: set[str] = set()
        allowed_query = set(plan.allowed_query_choice_ids)
        residual_m = int(plan.residual_m)

        for index, item in enumerate(provisional_raw, start=1):
            selections = item.get("selections") if isinstance(item, Mapping) else None
            strategy_label = (
                item.get("strategy_label") if isinstance(item, Mapping) else None
            )
            if not isinstance(selections, Mapping) or not selections:
                reason = "empty_selections"
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if len(selections) < 1:
                reason = "empty_selections"
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if len(selections) > residual_m:
                reason = "selection_count_exceeds_residual_m"
                errors.append(
                    {
                        "candidate_index": index,
                        "error_code": reason,
                        "n_selections": len(selections),
                        "residual_m": residual_m,
                    }
                )
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            changes, reason = resolve_action_slot_selections(
                selections, slots=slots, catalog=catalog
            )
            if changes is None:
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if any(
                str(choice_id) not in allowed_query for choice_id in selections.values()
            ):
                reason = "query_choice_not_allowed"
                errors.append(
                    {
                        "candidate_index": index,
                        "error_code": reason,
                        "selections": dict(selections),
                    }
                )
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            static_in_query = False
            for action_key in changes:
                rule = env.validator.policy.field_for_action(action_key)
                if rule is not None and rule.is_episode_locked:
                    reason = "static_action_in_query_slot"
                    errors.append(
                        {
                            "candidate_index": index,
                            "error_code": reason,
                            "action_key": action_key,
                        }
                    )
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                    static_in_query = True
                    break
            if static_in_query:
                continue
            proposal = AttackProposal(
                changes=changes,
                raw_command=(
                    f"{self.attacker_id}:episode_seed={self._episode_seed}:candidate={index}"
                ),
            )
            candidate, fail_reason, _ = self._validate_candidate(
                env,
                proposal,
                locked_values=locks,
                seen_fingerprints=seen,
                candidate_index=index,
                strategy_label=str(strategy_label) if strategy_label is not None else None,
            )
            if candidate is None:
                errors.append({"candidate_index": index, "error_code": fail_reason})
                reject_counts[fail_reason] = reject_counts.get(fail_reason, 0) + 1
                continue
            meta = dict(candidate.research_meta)
            meta["prompt_version"] = self.prompt_version
            meta["static_plan_id"] = plan.static_plan_id
            meta["static_edit_cost"] = plan.static_edit_cost
            meta["residual_m"] = plan.residual_m
            meta["selections"] = {str(k): str(v) for k, v in selections.items()}
            meta["choice_ids"] = [str(v) for v in selections.values()]
            accepted[index - 1] = AttackProposal(
                changes=candidate.changes,
                raw_command=candidate.raw_command,
                research_meta=meta,
            )
            seen.add(str(meta["candidate_fingerprint"]))
        return accepted, errors, reject_counts


    def _v4_2_local_feedback(
        self,
        *,
        attempt: int,
        max_attempts: int,
        invalid_indices: Sequence[int],
        preserved_indices: Sequence[int],
        provisional_raw: Sequence[Mapping[str, Any]],
        details: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build V4.2 repair feedback with pinned static plan and action slots."""
        plan = static_plan_by_id(
            self._v4_static_plans, str(self._v4_selected_static_plan_id or "")
        )
        slots = self._v4_1_action_slots
        preserved = [
            {
                "candidate_index": index,
                "strategy_label": provisional_raw[index - 1].get("strategy_label"),
                "selections": provisional_raw[index - 1].get("selections"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        occupied = [
            {
                "candidate_index": index,
                "selections": provisional_raw[index - 1].get("selections"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        residual = None if plan is None else int(plan.residual_m)
        slot_enum = [] if slots is None else list(slots.ordered_slot_ids)
        repair_schema = None
        if plan is not None and slots is not None and invalid_indices:
            repair_schema = build_v4_2_repair_output_schema(
                slot_enum=slot_enum,
                residual_m=int(plan.residual_m),
                requested_indices=invalid_indices,
            )
        return {
            "local_generation_attempt": int(attempt),
            "max_local_generation_attempts": int(max_attempts),
            "mode": "slot_repair",
            "error_code": "local_validation_failed",
            "static_plan_id": None if plan is None else plan.static_plan_id,
            "residual_m": residual,
            "selections_maxProperties": residual,
            "repair_output_schema": repair_schema,
            "allowed_query_choice_ids": (
                [] if plan is None else list(plan.allowed_query_choice_ids)
            ),
            "action_slots": [] if slots is None else slots.public_slots(),
            "invalid_candidate_indices": list(invalid_indices),
            "preserved_candidate_indices": list(preserved_indices),
            "preserved_candidates_pinned": to_jsonable(preserved),
            "occupied_selection_sets": to_jsonable(occupied),
            "prior_local_rejections": [
                {
                    "candidate_index": item.get("candidate_index"),
                    "error_code": item.get("error_code"),
                }
                for item in details
            ],
            "note": (
                "Repair only invalid indices; static_plan_id is pinned; "
                "emit selections (action_slot_id -> choice_id) only; enforce 1 <= len(selections) <= residual_m; "
                "do not emit action_key/reference_id/choice_ids/changes. "
                "Local rule-compliance only; consumes no Q; does not call D1."
            ),
        }


    def _evaluate_v4_1_portfolio(
        self,
        env: AttackEnvironment,
        provisional_raw: Sequence[Mapping[str, Any]],
        q_max: int,
    ) -> tuple[list[AttackProposal | None], list[dict[str, Any]], dict[str, int]]:
        """Validate V4.1 slots under pinned static plan + unique action slots."""
        catalog = self._v4_catalog
        slots = self._v4_1_action_slots
        plan = static_plan_by_id(
            self._v4_static_plans, str(self._v4_selected_static_plan_id or "")
        )
        locks = self._v4_selected_locks
        if catalog is None or slots is None or plan is None or locks is None:
            return [], [{"candidate_index": 1, "error_code": "missing_static_plan"}], {
                "missing_static_plan": 1
            }
        if len(provisional_raw) != q_max:
            errors = [
                {"candidate_index": index, "error_code": "wrong_candidate_count"}
                for index in range(1, q_max + 1)
            ]
            return [], errors, {"wrong_candidate_count": q_max}

        accepted: list[AttackProposal | None] = [None] * q_max
        errors: list[dict[str, Any]] = []
        reject_counts: dict[str, int] = {}
        seen: set[str] = set()
        allowed_query = set(plan.allowed_query_choice_ids)
        residual_m = int(plan.residual_m)

        for index, item in enumerate(provisional_raw, start=1):
            selections = item.get("selections") if isinstance(item, Mapping) else None
            strategy_label = (
                item.get("strategy_label") if isinstance(item, Mapping) else None
            )
            if not isinstance(selections, Mapping) or not selections:
                reason = "empty_selections"
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if len(selections) < 1 or len(selections) > residual_m:
                reason = "budget_exceeded"
                errors.append(
                    {
                        "candidate_index": index,
                        "error_code": reason,
                        "n_selections": len(selections),
                        "residual_m": residual_m,
                    }
                )
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            changes, reason = resolve_action_slot_selections(
                selections, slots=slots, catalog=catalog
            )
            if changes is None:
                errors.append({"candidate_index": index, "error_code": reason})
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            if any(
                str(choice_id) not in allowed_query for choice_id in selections.values()
            ):
                reason = "query_choice_not_allowed"
                errors.append(
                    {
                        "candidate_index": index,
                        "error_code": reason,
                        "selections": dict(selections),
                    }
                )
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            static_in_query = False
            for action_key in changes:
                rule = env.validator.policy.field_for_action(action_key)
                if rule is not None and rule.is_episode_locked:
                    reason = "static_action_in_query_slot"
                    errors.append(
                        {
                            "candidate_index": index,
                            "error_code": reason,
                            "action_key": action_key,
                        }
                    )
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                    static_in_query = True
                    break
            if static_in_query:
                continue
            proposal = AttackProposal(
                changes=changes,
                raw_command=(
                    f"{self.attacker_id}:episode_seed={self._episode_seed}:candidate={index}"
                ),
            )
            candidate, fail_reason, _ = self._validate_candidate(
                env,
                proposal,
                locked_values=locks,
                seen_fingerprints=seen,
                candidate_index=index,
                strategy_label=str(strategy_label) if strategy_label is not None else None,
            )
            if candidate is None:
                errors.append({"candidate_index": index, "error_code": fail_reason})
                reject_counts[fail_reason] = reject_counts.get(fail_reason, 0) + 1
                continue
            meta = dict(candidate.research_meta)
            meta["prompt_version"] = PROMPT_VERSION_V4_1
            meta["static_plan_id"] = plan.static_plan_id
            meta["static_edit_cost"] = plan.static_edit_cost
            meta["residual_m"] = plan.residual_m
            meta["selections"] = {str(k): str(v) for k, v in selections.items()}
            meta["choice_ids"] = [str(v) for v in selections.values()]
            accepted[index - 1] = AttackProposal(
                changes=candidate.changes,
                raw_command=candidate.raw_command,
                research_meta=meta,
            )
            seen.add(str(meta["candidate_fingerprint"]))
        return accepted, errors, reject_counts

    def _v4_1_local_feedback(
        self,
        *,
        attempt: int,
        max_attempts: int,
        invalid_indices: Sequence[int],
        preserved_indices: Sequence[int],
        provisional_raw: Sequence[Mapping[str, Any]],
        details: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build V4.1 repair feedback with pinned static plan and action slots."""
        plan = static_plan_by_id(
            self._v4_static_plans, str(self._v4_selected_static_plan_id or "")
        )
        slots = self._v4_1_action_slots
        preserved = [
            {
                "candidate_index": index,
                "strategy_label": provisional_raw[index - 1].get("strategy_label"),
                "selections": provisional_raw[index - 1].get("selections"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        occupied = [
            {
                "candidate_index": index,
                "selections": provisional_raw[index - 1].get("selections"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        return {
            "local_generation_attempt": int(attempt),
            "max_local_generation_attempts": int(max_attempts),
            "mode": "slot_repair",
            "error_code": "local_validation_failed",
            "static_plan_id": None if plan is None else plan.static_plan_id,
            "residual_m": None if plan is None else plan.residual_m,
            "allowed_query_choice_ids": (
                [] if plan is None else list(plan.allowed_query_choice_ids)
            ),
            "action_slots": [] if slots is None else slots.public_slots(),
            "invalid_candidate_indices": list(invalid_indices),
            "preserved_candidate_indices": list(preserved_indices),
            "preserved_candidates_pinned": to_jsonable(preserved),
            "occupied_selection_sets": to_jsonable(occupied),
            "prior_local_rejections": [
                {
                    "candidate_index": item.get("candidate_index"),
                    "error_code": item.get("error_code"),
                }
                for item in details
            ],
            "note": (
                "Repair only invalid indices; static_plan_id is pinned; "
                "emit selections (action_slot_id -> choice_id) only; "
                "do not emit action_key/reference_id/choice_ids/changes. "
                "Local rule-compliance only; consumes no Q; does not call D1."
            ),
        }

    def _v3_local_feedback(
        self,
        *,
        attempt: int,
        max_attempts: int,
        invalid_indices: Sequence[int],
        preserved_indices: Sequence[int],
        provisional_raw: Sequence[Mapping[str, Any]],
        details: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build a V3 repair prompt without proxy values or hidden feedback."""
        preserved = [
            {
                "candidate_index": index,
                "strategy_label": provisional_raw[index - 1].get("strategy_label"),
                "changes": provisional_raw[index - 1].get("changes"),
            }
            for index in preserved_indices
            if 0 < index <= len(provisional_raw)
        ]
        return {
            "local_generation_attempt": int(attempt),
            "max_local_generation_attempts": int(max_attempts),
            "mode": "slot_repair",
            "error_code": "local_validation_failed",
            "invalid_candidate_indices": list(invalid_indices),
            "preserved_candidate_indices": list(preserved_indices),
            "preserved_candidates_pinned": to_jsonable(preserved),
            "prior_local_rejections": [
                {
                    "candidate_index": item.get("candidate_index"),
                    "error_code": item.get("error_code"),
                    "action_key": item.get("action_key"),
                }
                for item in details
            ],
            "note": (
                "Repair only invalid indices; do not alter preserved slots. "
                "This is local rule-compliance only, consumes no Q, and does not call D1."
            ),
        }

    def _safe_local_feedback(
        self,
        *,
        attempt: int,
        max_attempts: int,
        error_code: str,
        details: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Attacker-safe local repair summary (no hidden proxy raw values)."""
        return {
            "local_generation_attempt": int(attempt),
            "max_local_generation_attempts": int(max_attempts),
            "error_code": str(error_code),
            "prior_local_rejections": [
                {
                    "candidate_index": item.get("candidate_index"),
                    "error_code": item.get("error_code"),
                    "action_key": item.get("action_key"),
                }
                for item in details
            ],
            "note": (
                "Rule-compliance repair only — not defender feedback. "
                "Correct your own field/reference_id choices. "
                "The system will not substitute a reference_id for you. "
                "Local failures do not consume Q and do not call D1."
            ),
        }

    def _freeze_validated_candidates(
        self,
        env: AttackEnvironment,
        raw_candidates: Sequence[Mapping[str, Any]],
        *,
        q_max: int,
        require_all_candidates: bool = True,
    ) -> tuple[list[AttackProposal], dict[str, int], list[dict[str, Any]]]:
        """Validate every planned candidate locally before freeze.

        When ``require_all_candidates`` is True (default), any local failure
        rejects the whole plan so the LLM can regenerate.  No env.step / D1.
        """
        reject_counts: dict[str, int] = {}
        reject_details: list[dict[str, Any]] = []
        frozen: list[AttackProposal] = []
        seen: set[str] = set()
        locked_values: Mapping[str, Any] | None = None

        planned = list(raw_candidates[:q_max])
        if not planned:
            reject_counts["empty_plan"] = 1
            reject_details.append(
                {"candidate_index": None, "error_code": "empty_plan"}
            )
            return [], reject_counts, reject_details

        for index, raw_item in enumerate(planned, start=1):
            if "changes" in raw_item and isinstance(raw_item.get("changes"), Mapping):
                raw_changes = dict(raw_item["changes"])
                strategy_label = raw_item.get("strategy_label")
            else:
                raw_changes = dict(raw_item)
                strategy_label = None
            normalised, reject = self._normalise_changes(env.validator, raw_changes)
            if normalised is None:
                reject_counts[reject] = reject_counts.get(reject, 0) + 1
                reject_details.append(
                    {"candidate_index": index, "error_code": reject}
                )
                if require_all_candidates:
                    return [], reject_counts, reject_details
                continue

            proposal = AttackProposal(
                changes=normalised,
                raw_command=(
                    f"{self.attacker_id}:episode_seed={self._episode_seed}:"
                    f"candidate={index}"
                ),
            )
            accepted, reason, locked_values = self._validate_candidate(
                env,
                proposal,
                locked_values=locked_values,
                seen_fingerprints=seen,
                candidate_index=len(frozen) + 1,
                strategy_label=(
                    str(strategy_label) if strategy_label is not None else None
                ),
            )
            if accepted is None:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                reject_details.append(
                    {"candidate_index": index, "error_code": reason}
                )
                if require_all_candidates:
                    return [], reject_counts, reject_details
                continue
            seen.add(str(accepted.research_meta["candidate_fingerprint"]))
            frozen.append(accepted)

        if require_all_candidates and len(frozen) != len(planned):
            return [], reject_counts, reject_details
        return frozen, reject_counts, reject_details

    def _normalise_changes(
        self,
        validator: ConstraintValidator,
        raw_changes: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:
        if not raw_changes:
            return None, "empty_changes"
        enabled = set(validator.enabled_action_keys)
        pool_ids = {profile.profile_id for profile in self.reference_pool.profiles}
        normalised: dict[str, Any] = {}
        for key, value in raw_changes.items():
            action_key = str(key)
            if action_key not in enabled:
                return None, "unknown_action"
            rule = validator.policy.field_for_action(action_key)
            if rule is None or not rule.is_mutable:
                return None, "unknown_action"
            selection = _parse_reference_selection_token(value)
            if selection is None:
                return None, "not_reference_selection"
            if selection.reference_id not in pool_ids:
                return None, "unknown_reference_id"
            # Do not substitute another reference_id; only verify resolvability.
            try:
                resolve_reference_selection(
                    action_key, selection, self.reference_pool, rule
                )
            except Exception:  # noqa: BLE001
                return None, "reference_field_incompatible"
            normalised[action_key] = selection
        return normalised, ""

    def _validate_candidate(
        self,
        env: AttackEnvironment,
        proposal: AttackProposal,
        *,
        locked_values: Mapping[str, Any] | None,
        seen_fingerprints: set[str],
        candidate_index: int,
        strategy_label: str | None = None,
    ) -> tuple[AttackProposal | None, str, Mapping[str, Any] | None]:
        validator = env.validator
        anchor = env.starting_case.features

        if locked_values is None:
            preparation = validator.prepare_episode_locks(anchor, proposal)
            next_locks = preparation.locked_values
            pre_errors = preparation.errors
        else:
            next_locks = dict(locked_values)
            pre_errors = ()

        projected = validator.project_for_billing(
            anchor, proposal, locked_values=next_locks
        )
        edited, distance, _, _ = compute_edit_metrics(
            anchor=anchor,
            candidate=projected,
            mutable_feature_names=validator.mutable_feature_names(),
            previous_candidate=None,
        )
        if distance < 1:
            return None, "same_as_anchor", locked_values
        if distance > self.budget.m_max:
            return None, "budget_exceeded", locked_values

        validity = validator.validate(
            anchor,
            proposal,
            locked_values=next_locks,
            pre_feedback_errors=pre_errors,
        )
        if not validity.is_valid:
            joined = " ".join(str(item) for item in validity.errors).lower()
            if "reference_provenance" in joined:
                return None, "reference_provenance_failed", locked_values
            if "episode-locked" in joined or "cannot change after first" in joined:
                return None, "static_lock_inconsistent", locked_values
            return None, "constraint_failed", locked_values

        provenance = audit_reference_provenance(
            anchor=anchor,
            candidate=projected,
            pool=self.reference_pool,
            changed_fields=edited,
        )
        if provenance["status"] != "PASS":
            return None, "reference_provenance_failed", locked_values

        action_fields = [
            name for name in self.reference_pool.action_fields if name in projected
        ]
        fingerprint = canonical_candidate_fingerprint(
            anchor_id=env.starting_case.case_id,
            projected_candidate=projected,
            action_fields=action_fields,
        )
        if fingerprint in seen_fingerprints:
            return None, "duplicate", locked_values

        retained = sorted(
            name
            for name in action_fields
            if _values_equal(projected[name], anchor[name])
        )
        assert self._episode_seed is not None
        meta = {
            "anchor_id": env.starting_case.case_id,
            "candidate_index": candidate_index,
            "candidate_fingerprint": fingerprint,
            "edited_fields": list(edited),
            "retained_fields": retained,
            "edit_distance_from_anchor": int(distance),
            "generation_seed": self._episode_seed,
            "experiment_seed": self.experiment_seed,
            "m_max": self.budget.m_max,
            "pool_fingerprint": self.reference_pool.pool_fingerprint,
            "reference_pool_fingerprint": self.reference_pool.pool_fingerprint,
            "pool_K": self.reference_pool.K,
            "reference_ids_used": list(reference_ids_from_changes(proposal.changes)),
            "reference_provenance": provenance,
            "generation_method": "a1_oneshot_llm",
            "prompt_version": self.prompt_version,
            "prompt_hash": self._prompt_hash,
            "strategy_label": strategy_label,
            "model": self.model,
            "thinking_disabled": self.thinking_disabled,
        }
        accepted = AttackProposal(
            changes=dict(proposal.changes),
            raw_command=proposal.raw_command,
            research_meta=meta,
        )
        return accepted, "", next_locks

    def _write_artefacts(
        self,
        env: AttackEnvironment,
        *,
        model: str,
        thinking_disabled: bool,
        parse_status: str,
        retry_count: int,
        llm_call_count: int,
        raw_candidates: Sequence[Mapping[str, Any]],
        total_latency_ms: float,
        selected_raw_path: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        run_dir = _episode_run_dir(env)
        if run_dir is None:
            return None, None, None, None, None
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)

        prompt_text_path = path / "a1_prompt_full.txt"
        prompt_text_path.write_text(self._prompt_text, encoding="utf-8")
        (path / "a1_prompt_hash.txt").write_text(
            self._prompt_hash + "\n", encoding="utf-8"
        )

        assert self._model_config is not None
        model_config_path = path / "model_config.json"
        model_config_path.write_text(
            json.dumps(
                {
                    **self._model_config.to_dict(),
                    "config_hash": self._config_hash,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (path / "a1_config_hash.txt").write_text(
            self._config_hash + "\n", encoding="utf-8"
        )

        plan_path = path / "a1_parsed_plan.json"
        plan_payload = {
            "parse_status": parse_status,
            "retry_count": retry_count,
            "llm_call_count": llm_call_count,
            "selected_response_index": self._selected_response_index,
            "raw_candidates": to_jsonable(list(raw_candidates)),
            "frozen_candidates": [
                {
                    "changes": to_jsonable(dict(item.changes)),
                    "research_meta": to_jsonable(dict(item.research_meta)),
                }
                for item in self._frozen_proposals
            ],
            "governance_reject_counts": dict(self._governance_reject_counts),
            "prompt_version": self.prompt_version,
            "prompt_hash": self._prompt_hash,
            "config_hash": self._config_hash,
            "model": model,
            "thinking_disabled": thinking_disabled,
            "latency_ms_total": total_latency_ms,
            "selected_raw_response_path": selected_raw_path,
        }
        plan_path.write_text(
            json.dumps(to_jsonable(plan_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        ledger_path = path / "a1_retry_ledger.json"
        ledger_path.write_text(
            json.dumps(
                {
                    "selected_response_index": self._selected_response_index,
                    "retry_count": retry_count,
                    "llm_call_count": llm_call_count,
                    "attempts": [item.to_dict() for item in self._retry_ledger],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if self._prompt_payload is not None:
            (path / "a1_prompt_payload.json").write_text(
                json.dumps(to_jsonable(self._prompt_payload), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

        call_path = path / "a1_llm_call.json"
        return (
            str(plan_path),
            str(call_path),
            str(ledger_path),
            str(prompt_text_path),
            str(model_config_path),
        )

    def _write(self, text: str) -> None:
        if self.stdout is not None:
            self.stdout.write(text)
            self.stdout.flush()


def _episode_run_dir(env: AttackEnvironment) -> Path | None:
    run_dir = getattr(env, "artifact_dir", None)
    if run_dir is None:
        logger = getattr(env, "logger", None)
        run_dir = getattr(logger, "run_dir", None)
    if run_dir is None:
        return None
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persist_raw_attempt(
    run_dir: Path | None, attempt_index: int, text: str
) -> str | None:
    if run_dir is None:
        return None
    path = Path(run_dir) / f"a1_raw_response_attempt_{int(attempt_index)}.txt"
    if path.exists():
        raise A1PlannerError(f"Refusing to overwrite raw attempt file: {path}")
    path.write_text(text, encoding="utf-8")
    return str(path)


def _classify_transport_error(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timeout" in message:
        return "timeout"
    return "transport_error"


def _compact_action_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """Shrink public action rules for the prompt (drop bulky compiled ranges)."""
    hard = []
    for item in rule.get("hard_constraints", ()) or ():
        if not isinstance(item, Mapping):
            continue
        compact = {key: value for key, value in item.items() if key != "compiled_ranges"}
        if "compiled_ranges" in item:
            compact["compiled_ranges_count"] = len(item.get("compiled_ranges") or ())
            compact["note"] = (
                "Conditional train-range constraints apply; invalid values are rejected."
            )
        hard.append(compact)
    support = list(rule.get("observed_support") or ())
    allowed = list(rule.get("allowed_values") or ())
    return {
        "feature": rule.get("feature"),
        "action_key": rule.get("action_key"),
        "category": rule.get("category"),
        "data_type": rule.get("data_type"),
        "domain_mode": rule.get("domain_mode"),
        "sampling_kind": rule.get("sampling_kind"),
        "lower_bound": rule.get("lower_bound"),
        "upper_bound": rule.get("upper_bound"),
        "allowed_values": allowed[:64],
        "observed_support": support[:64],
        "proxy_action_key": rule.get("proxy_action_key"),
        "proxy_actions": list(rule.get("proxy_actions") or ()),
        "hard_constraints": hard,
        "counts_toward_edit_budget": rule.get("counts_toward_edit_budget"),
    }


def _assert_prompt_safe(payload: Mapping[str, Any]) -> None:
    """Ensure D1 internals are not embedded as planning inputs."""
    anchor = payload.get("anchor", {})
    if not isinstance(anchor, Mapping):
        raise A1PlannerError("Prompt anchor block must be a mapping.")
    for block_name in ("anchor", "reference_pool"):
        block = payload.get(block_name, {})
        if not isinstance(block, Mapping):
            continue
        _reject_forbidden_keys(block, path=block_name)

    visible = anchor.get("visible_fields", {})
    if isinstance(visible, Mapping):
        overlap = sorted(set(visible).intersection(_FORBIDDEN_PROMPT_KEYS))
        if overlap:
            raise A1PlannerError(
                f"Anchor visible fields include forbidden keys: {overlap}."
            )


def _reject_forbidden_keys(node: Any, *, path: str) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_s = str(key)
            if key_s in _FORBIDDEN_PROMPT_KEYS:
                raise A1PlannerError(
                    f"Prompt payload unexpectedly contains '{key_s}' at {path}."
                )
            _reject_forbidden_keys(value, path=f"{path}.{key_s}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _reject_forbidden_keys(value, path=f"{path}[{index}]")


def _loads_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _parse_reference_selection_token(value: Any) -> ReferenceSelection | None:
    """Parse LLM change tokens into ReferenceSelection; never invent ids."""
    if is_reference_selection(value):
        return value
    if isinstance(value, Mapping):
        if set(value.keys()) == {"reference_id"} or "reference_id" in value:
            raw = value.get("reference_id")
            if isinstance(raw, str) and raw.strip():
                return ReferenceSelection(reference_id=raw.strip())
            return None
        return None
    if isinstance(value, str) and value.startswith("ref_"):
        return ReferenceSelection(reference_id=value.strip())
    return None


def _coerce_action_value(rule: Any, value: Any) -> Any:
    """Legacy literal coercion retained for parse helpers / older tests only."""
    selection = _parse_reference_selection_token(value)
    if selection is not None:
        return selection
    if rule.agent_action_mode == "proxy_action":
        name = str(value)
        if name not in rule.resolved_proxy_actions:
            raise ValueError(f"Unknown proxy action '{name}'.")
        return name
    if rule.data_type == "categorical":
        return str(value)
    if rule.data_type in {"binary", "integer"}:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return int(value)
    return float(value)


def _extract_cached_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            return int(cached)
    if isinstance(usage, Mapping):
        details_map = usage.get("prompt_tokens_details") or {}
        if isinstance(details_map, Mapping) and details_map.get("cached_tokens") is not None:
            return int(details_map["cached_tokens"])
    return 0


def _extract_reasoning_tokens(usage: Any) -> int | None:
    """Best-effort extraction of provider-reported reasoning token counts."""
    if usage is None:
        return None
    for attr in ("reasoning_tokens", "completion_tokens_details"):
        value = getattr(usage, attr, None)
        if attr == "reasoning_tokens" and value is not None:
            return int(value)
        if attr == "completion_tokens_details" and value is not None:
            nested = getattr(value, "reasoning_tokens", None)
            if nested is not None:
                return int(nested)
    if isinstance(usage, Mapping):
        if usage.get("reasoning_tokens") is not None:
            return int(usage["reasoning_tokens"])
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, Mapping) and details.get("reasoning_tokens") is not None:
            return int(details["reasoning_tokens"])
    return None


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
    "A1AttemptRecord",
    "A1CallRecord",
    "A1ModelConfig",
    "A1PlannerError",
    "DEFAULT_MAX_LOCAL_GENERATION_ATTEMPTS",
    "DEFAULT_MAX_PARSE_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TOP_P",
    "DIVERSIFICATION_PRINCIPLE_V2",
    "DeepSeekPlannerClient",
    "FORMAL_A1_MODEL_CONFIG",
    "LLMCompletion",
    "LLMCompletionClient",
    "OneShotLLMPlanner",
    "PROMPT_VERSION",
    "PROMPT_VERSION_V1",
    "PROMPT_VERSION_V2",
    "PROMPT_VERSION_V3",
    "PROMPT_VERSION_V4",
    "PROMPT_VERSION_V4_1",
    "PROMPT_VERSION_V4_2",
    "PROMPT_VERSION_V4_3",
    "PROXY_RAW_FEATURE_NAMES",
    "RETRYABLE_PARSE_STATUSES",
    "RETRYABLE_TRANSPORT_REASONS",
    "SUPPORTED_PROMPT_VERSIONS",
    "THINKING_ENABLED_MAX_TOKENS",
    "build_a1_prompt_payload",
    "estimate_flash_cost_usd",
    "format_a1_prompt_text",
    "hash_a1_prompt_text",
    "parse_a1_candidates",
    "parse_a1_slot_replacements",
    "parse_a1_v4_plan",
    "parse_a1_v4_slot_replacements",
    "parse_a1_v4_1_plan",
    "parse_a1_v4_1_slot_replacements",
    "resolve_max_tokens",
    "parse_a1_v4_2_plan",
    "parse_a1_v4_2_slot_replacements",
    "render_a1_messages",
]

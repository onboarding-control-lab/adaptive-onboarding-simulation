"""Fail-closed pins for benchmark / formal Attack Lab execution.

Library defaults for historical attackers remain unchanged.  Benchmark and
formal runners must construct attackers with these explicit selections rather
than relying on legacy prompt / Gower / provenance defaults.

Authoritative development pin (active stack):
  A1 ``a1_oneshot_v4_3_public_reference_view``
  A3 ``a3_episodic_reflective_v2_3_public_reference_view``
"""

from __future__ import annotations

from typing import Any, Mapping

from attack_lab.attackers.a1_v4_3_contract import PROMPT_VERSION_V4_3
from attack_lab.attackers.a2_search import GOWER_POLICY_PUBLIC_REFERENCE_V2
from attack_lab.attackers.a3_v2_3_contract import PROMPT_VERSION_A3_V2_3

PINNED_A1_PROMPT_VERSION = PROMPT_VERSION_V4_3
PINNED_A2_GOWER_POLICY = GOWER_POLICY_PUBLIC_REFERENCE_V2
PINNED_A3_PROMPT_VERSION = PROMPT_VERSION_A3_V2_3
PINNED_REQUIRE_REFERENCE_PROVENANCE = True

# Frozen Month-6 D1 / governance fingerprints used by development runners.
PINNED_D1_ARTEFACT_ID = "c1_pipeline_sha256_16=243c851b0c665c9c"
PINNED_GOVERNANCE_FINGERPRINT = (
    "177c7b9fec00f531932528ad4b77d7833a436b9e5705f89bf5045ff576d2ff16"
)

MODEL_FLASH = "deepseek-v4-flash"
MODEL_PRO = "deepseek-v4-pro"
SUPPORTED_LLM_MODELS = frozenset({MODEL_FLASH, MODEL_PRO})
REASONING_EFFORT_MAX = "max"

# Official DeepSeek list prices (USD / 1M tokens), pricing page checked 2026-08-12.
# Flash rates are identical to the historical estimate_flash_cost_usd constants.
_DEEPSEEK_RATES_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    MODEL_FLASH: (0.0028, 0.14, 0.28),  # cache-hit input, cache-miss input, output
    MODEL_PRO: (0.003625, 0.435, 0.87),
}


class BenchmarkPinError(RuntimeError):
    """Raised when a benchmark/formal run is not explicitly pinned."""


def normalize_llm_model(model: str) -> str:
    return str(model or "").strip()


def require_supported_llm_model(model: str) -> str:
    name = normalize_llm_model(model)
    if name not in SUPPORTED_LLM_MODELS:
        raise BenchmarkPinError(
            f"Unsupported llm_model={model!r}; "
            f"supported={sorted(SUPPORTED_LLM_MODELS)}."
        )
    return name


def estimate_deepseek_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimate USD cost for a supported DeepSeek model from token usage.

    Unknown model ids fail closed rather than silently using Flash rates.
    """
    name = require_supported_llm_model(model)
    hit_rate, miss_rate, output_rate = _DEEPSEEK_RATES_USD_PER_MTOK[name]
    cached = max(0, min(int(cached_tokens), int(prompt_tokens)))
    miss = max(0, int(prompt_tokens) - cached)
    return float(
        (cached / 1_000_000.0) * hit_rate
        + (miss / 1_000_000.0) * miss_rate
        + (int(completion_tokens) / 1_000_000.0) * output_rate
    )


def assert_pinned_a1(*, prompt_version: str, require_reference_provenance: bool) -> None:
    if str(prompt_version) != PINNED_A1_PROMPT_VERSION:
        raise BenchmarkPinError(
            f"A1 prompt_version must be {PINNED_A1_PROMPT_VERSION!r}; "
            f"got {prompt_version!r}."
        )
    if not bool(require_reference_provenance):
        raise BenchmarkPinError("A1 require_reference_provenance must be True.")


def assert_pinned_a2(*, gower_policy: str, require_reference_provenance: bool) -> None:
    if str(gower_policy) != PINNED_A2_GOWER_POLICY:
        raise BenchmarkPinError(
            f"A2 gower_policy must be {PINNED_A2_GOWER_POLICY!r}; "
            f"got {gower_policy!r}."
        )
    if not bool(require_reference_provenance):
        raise BenchmarkPinError("A2 require_reference_provenance must be True.")


def assert_pinned_a3(*, prompt_version: str, require_reference_provenance: bool) -> None:
    if str(prompt_version) != PINNED_A3_PROMPT_VERSION:
        raise BenchmarkPinError(
            f"A3 prompt_version must be {PINNED_A3_PROMPT_VERSION!r}; "
            f"got {prompt_version!r}."
        )
    if not bool(require_reference_provenance):
        raise BenchmarkPinError("A3 require_reference_provenance must be True.")


def assert_thinking_cell_config(
    *,
    thinking_disabled: bool,
    reasoning_effort: str | None,
    expect_thinking_disabled: bool,
) -> None:
    """Fail closed unless thinking/reasoning_effort match the registered cell."""
    if bool(thinking_disabled) != bool(expect_thinking_disabled):
        raise BenchmarkPinError(
            f"thinking_disabled mismatch: expected={expect_thinking_disabled!r}, "
            f"got={thinking_disabled!r}."
        )
    if expect_thinking_disabled:
        if reasoning_effort not in {None, ""}:
            raise BenchmarkPinError(
                "thinking-disabled cells must not set reasoning_effort; "
                f"got {reasoning_effort!r}."
            )
        return
    if str(reasoning_effort or "") != REASONING_EFFORT_MAX:
        raise BenchmarkPinError(
            f"thinking-enabled cells require reasoning_effort={REASONING_EFFORT_MAX!r}; "
            f"got {reasoning_effort!r}."
        )


def assert_pinned_defence_identity(
    *,
    d1_artefact_id: str,
    governance_fingerprint: str,
    require_reference_provenance: bool,
    month7_path_fragment: str | None = None,
) -> None:
    if not bool(require_reference_provenance):
        raise BenchmarkPinError("require_reference_provenance must be True.")
    if str(d1_artefact_id) != PINNED_D1_ARTEFACT_ID:
        raise BenchmarkPinError(
            f"D1 artefact id mismatch: expected {PINNED_D1_ARTEFACT_ID!r}, "
            f"got {d1_artefact_id!r}."
        )
    if str(governance_fingerprint) != PINNED_GOVERNANCE_FINGERPRINT:
        raise BenchmarkPinError(
            "Governance fingerprint mismatch: "
            f"expected {PINNED_GOVERNANCE_FINGERPRINT!r}, "
            f"got {governance_fingerprint!r}."
        )
    if month7_path_fragment and (
        "month7" in month7_path_fragment.lower()
        or "month_7" in month7_path_fragment.lower()
    ):
        raise BenchmarkPinError("Month-7 path fragment detected; refusing to proceed.")


def assert_reference_pool_fingerprint(
    *,
    observed: str,
    expected: str,
    anchor_id: str,
) -> None:
    if str(observed) != str(expected):
        raise BenchmarkPinError(
            f"Reference pool fingerprint mismatch for anchor {anchor_id}: "
            f"expected={expected!r}, got={observed!r}."
        )


def assert_benchmark_match_pins(
    *,
    attacker_id: str,
    prompt_version: str | None,
    gower_policy: str | None,
    require_reference_provenance: bool,
    llm_model: str | None = None,
) -> None:
    """Fail closed unless this episode uses the pinned benchmark selections."""
    if not bool(require_reference_provenance):
        raise BenchmarkPinError("require_reference_provenance must be True.")
    aid = str(attacker_id)
    if aid in {"a1", "A1-Flash", "A1-Pro", "A1-Pro-ThinkOff", "A1-Pro-ThinkOn"}:
        assert_pinned_a1(
            prompt_version=str(prompt_version),
            require_reference_provenance=True,
        )
        if llm_model is not None:
            require_supported_llm_model(llm_model)
        return
    if aid in {"a2", "A2"}:
        assert_pinned_a2(
            gower_policy=str(gower_policy),
            require_reference_provenance=True,
        )
        return
    if aid in {"a3", "A3-Flash", "A3-Pro", "A3-Pro-ThinkOff", "A3-Pro-ThinkOn"}:
        assert_pinned_a3(
            prompt_version=str(prompt_version),
            require_reference_provenance=True,
        )
        if llm_model is not None:
            require_supported_llm_model(llm_model)
        return
    if aid in {"a0", "A0"}:
        return
    raise BenchmarkPinError(f"Unknown attacker_id for benchmark pins: {attacker_id!r}.")


def pinned_attacker_summary() -> dict[str, Any]:
    return {
        "a1_prompt_version": PINNED_A1_PROMPT_VERSION,
        "a2_gower_policy": PINNED_A2_GOWER_POLICY,
        "a3_prompt_version": PINNED_A3_PROMPT_VERSION,
        "require_reference_provenance": PINNED_REQUIRE_REFERENCE_PROVENANCE,
        "supported_llm_models": sorted(SUPPORTED_LLM_MODELS),
        "default_flash_model": MODEL_FLASH,
        "pro_model": MODEL_PRO,
        "d1_artefact_id": PINNED_D1_ARTEFACT_ID,
        "governance_fingerprint": PINNED_GOVERNANCE_FINGERPRINT,
        "reasoning_effort_when_thinking_enabled": REASONING_EFFORT_MAX,
    }


def inspect_constructed_attacker(
    attacker: Any,
    *,
    attacker_id: str,
    require_reference_provenance: bool,
    llm_model: str | None = None,
    expect_thinking_disabled: bool | None = None,
) -> None:
    """Inspect a live attacker instance; fail closed on pin drift."""
    prompt_version = getattr(attacker, "prompt_version", None)
    gower_policy = getattr(attacker, "gower_policy", None)
    if llm_model is not None:
        actual_model = getattr(attacker, "model", None)
        if str(actual_model) != str(llm_model):
            raise BenchmarkPinError(
                f"{attacker_id} model drifted: requested={llm_model!r}, "
                f"attacker.model={actual_model!r}."
            )
    assert_benchmark_match_pins(
        attacker_id=attacker_id,
        prompt_version=prompt_version,
        gower_policy=gower_policy,
        require_reference_provenance=require_reference_provenance,
        llm_model=llm_model,
    )
    if expect_thinking_disabled is not None and hasattr(attacker, "thinking_disabled"):
        effort = getattr(attacker, "reasoning_effort", None)
        assert_thinking_cell_config(
            thinking_disabled=bool(attacker.thinking_disabled),
            reasoning_effort=effort,
            expect_thinking_disabled=bool(expect_thinking_disabled),
        )


def condition_manifest(
    *,
    condition_id: str,
    attacker_kind: str,
    prompt_version: str | None,
    model: str | None,
    thinking_disabled: bool | None,
    reasoning_effort: str | None,
    config_hash: str | None,
    prompt_hash: str | None = None,
    gower_policy: str | None = None,
) -> dict[str, Any]:
    """Immutable per-condition metadata recorded before/alongside episodes."""
    return {
        "condition_id": condition_id,
        "attacker_kind": attacker_kind,
        "attacker_version": prompt_version or gower_policy,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "model": model,
        "thinking_disabled": thinking_disabled,
        "thinking_enabled": (
            None if thinking_disabled is None else (not bool(thinking_disabled))
        ),
        "reasoning_effort": reasoning_effort,
        "config_hash": config_hash,
        "gower_policy": gower_policy,
        "pins": pinned_attacker_summary(),
    }


def preflight_formal_payload(payload: Mapping[str, Any]) -> list[str]:
    """Return fail-closed errors for a formal JSON payload (empty = pass)."""
    errors: list[str] = []
    attackers = payload.get("attackers") or {}
    if not isinstance(attackers, Mapping):
        return ["attackers section missing"]
    a1 = attackers.get("a1") or {}
    a2 = attackers.get("a2") or {}
    a3 = attackers.get("a3") or {}
    a1_prompt = a1.get("prompt_version") or (a1.get("model_config") or {}).get(
        "prompt_version"
    )
    a3_prompt = a3.get("prompt_version") or (a3.get("model_config") or {}).get(
        "prompt_version"
    )
    a2_gower = a2.get("gower_policy") or (a2.get("model_config") or {}).get(
        "gower_policy"
    )
    provenance = payload.get("require_reference_provenance")
    if provenance is not True:
        errors.append("require_reference_provenance must be true")
    if str(a1_prompt) != PINNED_A1_PROMPT_VERSION:
        errors.append(
            f"A1 prompt_version must be {PINNED_A1_PROMPT_VERSION!r}; got {a1_prompt!r}"
        )
    if str(a2_gower) != PINNED_A2_GOWER_POLICY:
        errors.append(
            f"A2 gower_policy must be {PINNED_A2_GOWER_POLICY!r}; got {a2_gower!r}"
        )
    if str(a3_prompt) != PINNED_A3_PROMPT_VERSION:
        errors.append(
            f"A3 prompt_version must be {PINNED_A3_PROMPT_VERSION!r}; got {a3_prompt!r}"
        )
    return errors


__all__ = [
    "MODEL_FLASH",
    "MODEL_PRO",
    "PINNED_A1_PROMPT_VERSION",
    "PINNED_A2_GOWER_POLICY",
    "PINNED_A3_PROMPT_VERSION",
    "PINNED_D1_ARTEFACT_ID",
    "PINNED_GOVERNANCE_FINGERPRINT",
    "PINNED_REQUIRE_REFERENCE_PROVENANCE",
    "REASONING_EFFORT_MAX",
    "SUPPORTED_LLM_MODELS",
    "BenchmarkPinError",
    "assert_benchmark_match_pins",
    "assert_pinned_a1",
    "assert_pinned_a2",
    "assert_pinned_a3",
    "assert_pinned_defence_identity",
    "assert_reference_pool_fingerprint",
    "assert_thinking_cell_config",
    "condition_manifest",
    "estimate_deepseek_cost_usd",
    "inspect_constructed_attacker",
    "normalize_llm_model",
    "pinned_attacker_summary",
    "preflight_formal_payload",
    "require_supported_llm_model",
]

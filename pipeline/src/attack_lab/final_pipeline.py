"""Shared final orchestrator for rehearsal and authorised Month-7 execution.

Rehearsal and production use the same attacker classes, frozen D1, first-PASS
extraction, and D2-S v1.0 primary scoring. Only dataset/anchor/LLM transport
injection differs. Attackers are run once; D2 variants are offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
from attack_lab.attackers.a1_planner import OneShotLLMPlanner
from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
from attack_lab.attackers.a3_agent import EpisodicLLMAgent
from attack_lab.benchmark_pins import (
    MODEL_PRO,
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    inspect_constructed_attacker,
)
from attack_lab.budget import AttackBudget, BudgetSpec
from attack_lab.cases import StartingCase
from attack_lab.defender import FrozenXGBoostDefender
from attack_lab.feedback import FeedbackPolicy
from attack_lab.final_anchors import fingerprint_anchor_ids
from attack_lab.final_metrics import FROZEN_D2S_V10_THRESHOLDS, primary_metrics_by_attacker
from attack_lab.final_protocol import FinalProtocolConfig, protocol_role_statement, resolve_repo_path
from attack_lab.final_v11 import verify_v11_artefact
from attack_lab.first_success import extract_first_successful_d1_pass
from attack_lab.governance import CompiledGovernancePolicy
from attack_lab.logger import TrajectoryLogger
from attack_lab.orchestrator import MatchConfig, MatchOrchestrator, ScriptedAttacker
from attack_lab.query_semantics import is_api_failure
from attack_lab.reference_pool import ReferencePool, ReferencePoolConfig, ReferencePoolProvider
from attack_lab.types import to_jsonable
from d2.iforest_v11 import D2SV11IForestAggregator
from d2.scoring import D2SScorer

PipelineMode = Literal["rehearse", "production"]
CONDITION_ORDER: tuple[tuple[str, str], ...] = (
    ("A0", "a0"),
    ("A1", "a1"),
    ("A2", "a2"),
    ("A3", "a3"),
)


class FinalPipelineError(RuntimeError):
    """Fail-closed final pipeline error."""


def instantiate_real_attacker(
    *,
    attacker_kind: str,
    protocol: FinalProtocolConfig,
    pool: ReferencePool,
    llm_client: Any | None,
) -> Any:
    """Construct the production A0/A1/A2/A3 implementation. No ScriptedAttacker."""
    seed = int(protocol.payload["seeds"]["experiment_seed"])
    budget = AttackBudget(q_max=protocol.q_max, m_max=protocol.m_max)
    if attacker_kind == "a0":
        attacker = ConstrainedRandomAttacker(
            seed=seed,
            reference_pool=pool,
            m_max=protocol.m_max,
            attacker_id="a0",
        )
        inspect_constructed_attacker(
            attacker, attacker_id="a0", require_reference_provenance=True
        )
        return attacker
    if attacker_kind == "a1":
        if llm_client is None:
            raise FinalPipelineError("A1 requires an injected or live LLM client.")
        attacker = OneShotLLMPlanner(
            experiment_seed=seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a1",
            prompt_version=PINNED_A1_PROMPT_VERSION,
            model=MODEL_PRO,
            thinking_disabled=True,
            reasoning_effort=None,
            llm_client=llm_client,
            stdout=None,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id="a1",
            require_reference_provenance=True,
            llm_model=MODEL_PRO,
            expect_thinking_disabled=True,
        )
        return attacker
    if attacker_kind == "a2":
        attacker = SurrogateGuidedSearcher(
            budget=budget,
            reference_pool=pool,
            experiment_seed=seed,
            attacker_id="a2",
            gower_policy=PINNED_A2_GOWER_POLICY,
            stdout=None,
        )
        inspect_constructed_attacker(
            attacker, attacker_id="a2", require_reference_provenance=True
        )
        return attacker
    if attacker_kind == "a3":
        if llm_client is None:
            raise FinalPipelineError("A3 requires an injected or live LLM client.")
        attacker = EpisodicLLMAgent(
            experiment_seed=seed,
            reference_pool=pool,
            budget=budget,
            attacker_id="a3",
            prompt_version=PINNED_A3_PROMPT_VERSION,
            model=MODEL_PRO,
            thinking_disabled=True,
            reasoning_effort=None,
            llm_client=llm_client,
            stdout=None,
        )
        inspect_constructed_attacker(
            attacker,
            attacker_id="a3",
            require_reference_provenance=True,
            llm_model=MODEL_PRO,
            expect_thinking_disabled=True,
        )
        return attacker
    raise FinalPipelineError(f"Unknown attacker kind {attacker_kind!r}.")


def _assert_not_scripted(attacker: Any) -> None:
    if isinstance(attacker, ScriptedAttacker):
        raise FinalPipelineError("ScriptedAttacker is forbidden on the final path.")
    name = type(attacker).__name__
    expected = {
        "a0": "ConstrainedRandomAttacker",
        "a1": "OneShotLLMPlanner",
        "a2": "SurrogateGuidedSearcher",
        "a3": "EpisodicLLMAgent",
    }
    kind = getattr(attacker, "attacker_id", "")
    if kind in expected and name != expected[kind]:
        raise FinalPipelineError(
            f"Attacker {kind} must be {expected[kind]}, got {name}."
        )


def score_primary_d2s(
    submissions: Sequence[Mapping[str, Any]],
    protocol: FinalProtocolConfig,
) -> dict[str, Any]:
    artefact = Path(protocol.payload["d2"]["primary"]["artefact_path"])
    expected_fp = str(protocol.payload["d2"]["primary"]["fingerprint"])
    if not artefact.is_file():
        raise FinalPipelineError(f"Missing D2-S v1.0 artefact: {artefact}")
    scorer = D2SScorer.load(artefact)
    if scorer.fingerprint != expected_fp:
        raise FinalPipelineError(
            f"D2-S v1.0 fingerprint drifted: {scorer.fingerprint} != {expected_fp}"
        )
    if scorer.month7_opened:
        raise FinalPipelineError("Refusing a D2-S v1.0 scorer that opened Month 7.")
    rows: list[dict[str, Any]] = []
    for item in submissions:
        features = item.get("features") or {}
        result = scorer.score(features)
        rows.append(
            {
                **dict(item),
                "d2s_v10_score": result["d2_score"],
                "d2s_v10_relationships": result["relationship_scores"],
            }
        )
    return {
        "role": "PRIMARY",
        "id": protocol.payload["d2"]["primary"]["id"],
        "fingerprint": scorer.fingerprint,
        "n_submissions": len(rows),
        "rows": rows,
        "thresholds": dict(FROZEN_D2S_V10_THRESHOLDS),
        "roles": protocol_role_statement(),
    }


def score_optional_v11(
    primary_rows: Sequence[Mapping[str, Any]],
    protocol: FinalProtocolConfig,
) -> dict[str, Any]:
    secondary = protocol.payload["d2"].get("secondary_prespecified") or {}
    path_raw = secondary.get("model_artefact")
    path = resolve_repo_path(path_raw) if path_raw and path_raw != "not_persisted_joblib_at_freeze" else None
    verification = verify_v11_artefact(path, secondary.get("model_sha256"))
    if not verification["enabled"]:
        return {
            "status": "V11_SECONDARY_DISABLED",
            "enabled": False,
            "verification": verification,
            "note": "v1.1 is optional and does not block primary readiness.",
        }
    aggregator = D2SV11IForestAggregator.load(Path(verification["path"]))
    scored: list[dict[str, Any]] = []
    for item in primary_rows:
        relationships = item.get("d2s_v10_relationships")
        if not isinstance(relationships, Mapping):
            scored.append({**dict(item), "d2s_v11_error": "missing_v10_relationships"})
            continue
        import pandas as pd

        frame = pd.DataFrame([dict(relationships)])
        score = float(aggregator.score_relationship_frame(frame)[0])
        scored.append({**dict(item), "d2s_v11_score": score})
    return {
        "status": "V11_SECONDARY_READY",
        "enabled": True,
        "role": "OPTIONAL_SECONDARY_EXPLORATORY",
        "verification": verification,
        "n_submissions": len(scored),
        "rows": scored,
        "note": "Scored offline on the same first D1-PASS submissions. Not primary.",
    }


def run_final_pipeline(
    *,
    protocol: FinalProtocolConfig,
    mode: PipelineMode,
    defender: FrozenXGBoostDefender,
    policy: CompiledGovernancePolicy,
    anchors: Sequence[StartingCase],
    training_frame: Any,
    llm_client: Any | None,
    run_dir: Path,
    write_status: Callable[..., None],
    on_episode: Callable[[dict[str, Any]], None] | None = None,
    live_api_calls: int = 0,
) -> dict[str, Any]:
    """Execute the frozen A0–A3 → D1 → first-PASS → D2-S v1.0 chain."""
    if mode not in {"rehearse", "production"}:
        raise FinalPipelineError(f"Unknown pipeline mode {mode!r}.")
    if not isinstance(defender, FrozenXGBoostDefender):
        raise FinalPipelineError("Final pipeline requires the real FrozenXGBoostDefender.")
    if abs(float(defender.threshold) - float(protocol.payload["d1"]["threshold"])) > 1e-12:
        raise FinalPipelineError("Frozen D1 threshold drifted.")
    if defender.artefact_id != protocol.payload["d1"]["artefact_id"]:
        raise FinalPipelineError("Frozen D1 artefact id drifted.")
    if 7 in set(int(m) for m in training_frame["month"].unique()):
        raise FinalPipelineError("Training/reference frame retained Month 7.")
    if mode == "rehearse" and any(case.data_split == "test_month7" for case in anchors):
        raise FinalPipelineError("Rehearsal must not use Month-7 starting cases.")

    write_status(
        run_dir,
        "RUNNING",
        mode=mode,
        month7_accessed=(mode == "production"),
        live_api_calls=int(live_api_calls),
    )
    seed = int(protocol.payload["seeds"]["experiment_seed"])
    pool_seed = int(protocol.payload["seeds"]["reference_pool_seed"])
    provider = ReferencePoolProvider.from_config(
        ReferencePoolConfig.load(), training_frame=training_frame
    )
    budget = AttackBudget(q_max=protocol.q_max, m_max=protocol.m_max)
    budget_spec = BudgetSpec(
        q_max=protocol.q_max,
        m_max=protocol.m_max,
        invalid_charges_q=False,
        label="final_month7_frozen_budget_contract_A",
    )
    rows: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    constructed_types: dict[str, str] = {}
    q_used_total = 0
    invalid_total = 0
    try:
        for condition_id, attacker_kind in CONDITION_ORDER:
            for case in anchors:
                if case.initial_decision != "BLOCK":
                    raise FinalPipelineError(
                        f"Anchor {case.case_id} is not a D1 BLOCK starting case."
                    )
                pool = provider.get_pool(case.case_id, seed=pool_seed)
                if int(pool.K) != int(protocol.k):
                    raise FinalPipelineError(f"Reference pool K={pool.K} != {protocol.k}.")
                attacker = instantiate_real_attacker(
                    attacker_kind=attacker_kind,
                    protocol=protocol,
                    pool=pool,
                    llm_client=llm_client,
                )
                _assert_not_scripted(attacker)
                constructed_types[condition_id] = type(attacker).__name__
                episode_dir = (
                    run_dir / "trajectories" / condition_id / f"anchor_{case.case_id}"
                )
                episode_dir.mkdir(parents=True, exist_ok=True)
                logger = TrajectoryLogger(
                    run_dir=episode_dir,
                    run_id=f"{condition_id}_{case.case_id}_{mode}",
                )
                match = MatchOrchestrator().run_episode(
                    attacker,
                    MatchConfig(
                        attacker_id=attacker_kind,
                        anchor=case,
                        policy=policy,
                        budget=budget_spec,
                        feedback_policy=FeedbackPolicy(mode="label_only"),
                        defender=defender,
                        seed=seed,
                        logger=logger,
                        reference_pool=pool,
                        require_reference_provenance=True,
                    ),
                )
                episode = to_jsonable(asdict(match.episode))
                (episode_dir / "episode_result.json").write_text(
                    json.dumps(episode, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                first = extract_first_successful_d1_pass(episode)
                row = {
                    "condition_id": condition_id,
                    "attacker_class": type(attacker).__name__,
                    "anchor_id": case.case_id,
                    "success": bool(match.success),
                    "q_used": int(match.q_used),
                    "invalid_submissions": int(match.invalid_submissions),
                    "stop_reason": match.stop_reason,
                    "api_failure": is_api_failure(match.stop_reason),
                    "episode_dir": str(episode_dir),
                    "first_success": first,
                }
                if int(match.q_used) > int(protocol.q_max):
                    raise FinalPipelineError(
                        f"{condition_id} exceeded Q={protocol.q_max}: q_used={match.q_used}."
                    )
                q_used_total += int(match.q_used)
                invalid_total += int(match.invalid_submissions)
                rows.append(row)
                if first:
                    submissions.append(
                        {
                            "condition_id": condition_id,
                            "anchor_id": case.case_id,
                            **first,
                        }
                    )
                if on_episode is not None:
                    on_episode(row)
        (run_dir / "raw_attack_trajectories.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "first_success_d1_pass.json").write_text(
            json.dumps(submissions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        primary = score_primary_d2s(submissions, protocol)
        (run_dir / "d2_primary_v10.json").write_text(
            json.dumps(to_jsonable(primary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        scores_by_attacker: dict[str, list[float | None]] = {
            name: [] for name, _ in CONDITION_ORDER
        }
        for item in primary["rows"]:
            scores_by_attacker[str(item["condition_id"])].append(item.get("d2s_v10_score"))
        metrics = primary_metrics_by_attacker(
            n_anchors=len(anchors),
            scores_by_attacker=scores_by_attacker,
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        optional = score_optional_v11(primary["rows"], protocol)
        (run_dir / "d2_optional_v11.json").write_text(
            json.dumps(to_jsonable(optional), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            "status": "COMPLETE",
            "mode": mode,
            "rows": rows,
            "submissions": submissions,
            "primary_d2": primary,
            "optional_v11": optional,
            "metrics": metrics,
            "constructed_attacker_classes": constructed_types,
            "q_used_total": q_used_total,
            "invalid_total": invalid_total,
            "paired_anchor_fingerprint": fingerprint_anchor_ids(
                [case.case_id for case in anchors]
            ),
            "n_anchors": len(anchors),
            "month7_accessed": mode == "production",
            "live_api_calls": int(live_api_calls),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_status(
            run_dir,
            "COMPLETE",
            mode=mode,
            month7_accessed=(mode == "production"),
            live_api_calls=int(live_api_calls),
        )
        return result
    except Exception:
        finished = { (row["condition_id"], row["anchor_id"]) for row in rows }
        expected = {
            (condition_id, case.case_id)
            for condition_id, _ in CONDITION_ORDER
            for case in anchors
        }
        status = "PARTIAL" if finished and finished != expected else "FAILED"
        write_status(
            run_dir,
            status,
            mode=mode,
            month7_accessed=(mode == "production"),
            live_api_calls=int(live_api_calls),
            n_finished_episodes=len(rows),
        )
        raise


__all__ = [
    "CONDITION_ORDER",
    "FinalPipelineError",
    "instantiate_real_attacker",
    "run_final_pipeline",
    "score_optional_v11",
    "score_primary_d2s",
]

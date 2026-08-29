"""Post-run Month-7 evaluation audit and derived tables.

Scientific execution and reporting are separated. This module reads a
COMPLETE run directory. It never reruns attackers, never opens Month 7,
and never mutates RUN_STATUS.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from attack_lab.final_metrics import FROZEN_D2S_V10_THRESHOLDS, PRIMARY_REVIEW_BUDGET, d2_decision
from attack_lab.final_pipeline import CONDITION_ORDER
from attack_lab.query_semantics import is_api_failure

ATTACKERS = tuple(name for name, _ in CONDITION_ORDER)
EXHAUSTION_REASONS = frozenset(
    {
        "no_feasible_candidate",
        "action_space_exhaustion",
        "local_generation_exhausted",
    }
)
REQUIRED_SCIENTIFIC_FILES = (
    "RUN_STATUS.json",
    "raw_attack_trajectories.json",
    "first_success_d1_pass.json",
    "d2_primary_v10.json",
    "metrics.json",
)
EXECUTION_MANIFEST_NAME = "FINAL_RUN_MANIFEST.json"
REPORTING_CONTEXT_NAME = "REPORTING_PROTOCOL_CONTEXT.json"
NOT_RECORDED = {"status": "not_recorded_at_execution"}
WILSON_METHOD_NOTE = "wilson_95_binomial"
PAIRED_BOOTSTRAP_RESAMPLES = 10_000
ANALYSIS_SEED = 20260822
PLANNED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("A1", "A0"),
    ("A2", "A0"),
    ("A3", "A0"),
    ("A2", "A1"),
    ("A3", "A1"),
    ("A3", "A2"),
)
UNCERTAINTY_METHOD_NOTE = (
    "absolute rates: 95% Wilson; attacker differences: "
    f"anchor-level paired bootstrap 95% CI, B={PAIRED_BOOTSTRAP_RESAMPLES}, "
    f"analysis_seed={ANALYSIS_SEED}; pairing preserved; no p<0.05 threshold"
)


class FinalReportingError(RuntimeError):
    """Reporting-stage failure. Must not invalidate a COMPLETE scientific run."""


def wilson_interval(k: int, n: int, *, z: float = 1.96) -> dict[str, Any]:
    """Descriptive 95% Wilson interval. Not a frozen paired inferential method."""
    if n <= 0:
        return {
            "k": k,
            "n": n,
            "p": None,
            "low": None,
            "high": None,
            "method": WILSON_METHOD_NOTE,
        }
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    margin = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / den
    return {
        "k": k,
        "n": n,
        "p": p,
        "low": max(0.0, centre - margin),
        "high": min(1.0, centre + margin),
        "method": WILSON_METHOD_NOTE,
    }


def write_reporting_status(run_dir: Path, status: str, **extra: Any) -> None:
    if status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise FinalReportingError(f"Invalid reporting status {status!r}.")
    payload = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_run_status_untouched": True,
        **extra,
    }
    reporting_dir = run_dir / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)
    (reporting_dir / "REPORTING_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_reporting_context(
    run_dir: Path, scientific_status: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble reporting pins from frozen protocol + persisted artefacts.

    Used when production execution did not write FINAL_RUN_MANIFEST.json.
    Does not invent unrecorded runtime values (python/platform, live call
    ledgers, retries). Explicitly marked reconstructed_post_run=true.
    """
    from attack_lab.final_protocol import FinalProtocolConfig

    protocol = FinalProtocolConfig.load()
    d2 = _load_json(run_dir / "d2_primary_v10.json")
    expected_fp = str(protocol.payload["d2"]["primary"]["fingerprint"])
    persisted_fp = str(d2.get("fingerprint") or "")
    if persisted_fp != expected_fp:
        raise FinalReportingError(
            "Persisted D2-S fingerprint disagrees with frozen protocol: "
            f"persisted={persisted_fp!r}, protocol={expected_fp!r}."
        )
    anchors_path = Path(str(protocol.payload["paired_anchors"]["production_path"]))
    source_hashes = {
        "frozen_protocol_file_sha256": _sha256_file(protocol.path),
        "RUN_STATUS.json": _sha256_file(run_dir / "RUN_STATUS.json"),
        "raw_attack_trajectories.json": _sha256_file(
            run_dir / "raw_attack_trajectories.json"
        ),
        "first_success_d1_pass.json": _sha256_file(
            run_dir / "first_success_d1_pass.json"
        ),
        "d2_primary_v10.json": _sha256_file(run_dir / "d2_primary_v10.json"),
        "metrics.json": _sha256_file(run_dir / "metrics.json"),
    }
    anchor_fingerprint = None
    anchor_manifest_hash = None
    if anchors_path.is_file():
        source_hashes["immutable_anchor_manifest_file_sha256"] = _sha256_file(
            anchors_path
        )
        issued = _load_json(anchors_path)
        anchor_fingerprint = issued.get("fingerprint")
        anchor_manifest_hash = issued.get("manifest_hash")
    attackers = protocol.payload["attackers"]
    api = protocol.payload["api"]
    d1 = protocol.payload["d1"]
    return {
        "schema_version": "final-month7-reporting-context-v1",
        "reconstructed_post_run": True,
        "not_an_execution_time_artefact": True,
        "provenance_source": (
            "assembled_post_run_from_frozen_protocol_and_persisted_artefacts"
        ),
        "reconstruction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "K": protocol.k,
        "Q": protocol.q_max,
        "m": protocol.m_max,
        "d1_identifier": d1["artefact_id"],
        "d1_threshold": d1["threshold"],
        "d2_v10_fingerprint": persisted_fp,
        "d2_v10_id": d2.get("id"),
        "attacker_pins": {
            "A0": attackers["A0"],
            "A1": attackers["A1"],
            "A2": attackers["A2"],
            "A3": attackers["A3"],
        },
        "model_names": {
            "A1": attackers["A1"]["model"],
            "A3": attackers["A3"]["model"],
        },
        "thinking": api.get("thinking"),
        "system_fingerprints": dict(NOT_RECORDED),
        "call_counts": {
            "live_api_calls": "not_recorded_by_production_runner",
            "run_status_live_api_calls_field": scientific_status.get("live_api_calls"),
            "run_status_live_api_calls_note": (
                "run_execute_final always passes live_api_calls=0; "
                "RUN_STATUS.live_api_calls is not a live call ledger"
            ),
        },
        "retries": dict(NOT_RECORDED),
        "mode": scientific_status.get("mode"),
        "protocol_config_hash": protocol.config_hash,
        "protocol_path": str(protocol.path),
        "anchor_manifest_path": str(anchors_path),
        "paired_anchor_fingerprint": anchor_fingerprint,
        "anchor_manifest_hash": anchor_manifest_hash,
        "source_artefact_hashes": source_hashes,
    }


def load_reporting_context(
    run_dir: Path, scientific_status: Mapping[str, Any]
) -> dict[str, Any]:
    """Prefer an execution-time manifest when present; otherwise assemble."""
    execution_path = Path(run_dir) / EXECUTION_MANIFEST_NAME
    if execution_path.is_file():
        payload = _load_json(execution_path)
        if not isinstance(payload, dict):
            raise FinalReportingError(
                f"{EXECUTION_MANIFEST_NAME} is not a JSON object."
            )
        return {
            **payload,
            "reconstructed_post_run": False,
            "not_an_execution_time_artefact": False,
            "provenance_source": f"execution_time_{EXECUTION_MANIFEST_NAME}",
        }
    return assemble_reporting_context(run_dir, scientific_status)


def persist_reporting_context(run_dir: Path, context: Mapping[str, Any]) -> Path:
    """Write reporting-only provenance. Never writes FINAL_RUN_MANIFEST.json."""
    reporting_dir = Path(run_dir) / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)
    path = reporting_dir / REPORTING_CONTEXT_NAME
    path.write_text(
        json.dumps(dict(context), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_scientific_run(run_dir: Path) -> dict[str, Any]:
    """Confirm the scientific artefacts exist and are COMPLETE. Read-only."""
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED_SCIENTIFIC_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FinalReportingError("Scientific artefacts missing: " + ", ".join(missing))
    status = _load_json(run_dir / "RUN_STATUS.json")
    if status.get("status") != "COMPLETE":
        raise FinalReportingError(
            f"Refusing to report on a non-COMPLETE run ({status.get('status')!r})."
        )
    return status


def _episode_path(row: Mapping[str, Any]) -> Path | None:
    episode_dir = row.get("episode_dir")
    if not episode_dir:
        return None
    path = Path(str(episode_dir)) / "episode_result.json"
    return path if path.is_file() else None


def _count_valid_candidates(episode: Mapping[str, Any] | None) -> int:
    if not episode:
        return 0
    n = 0
    for step in episode.get("steps") or []:
        validity = step.get("validity") or {}
        if validity.get("is_valid") is True:
            n += 1
    return n


def load_run_tables(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    scientific = validate_scientific_run(run_dir)
    trajectories = _load_json(run_dir / "raw_attack_trajectories.json")
    d2 = _load_json(run_dir / "d2_primary_v10.json")
    manifest = load_reporting_context(run_dir, scientific)
    metrics = _load_json(run_dir / "metrics.json")
    d2_by_key = {
        (str(row["condition_id"]), str(row["anchor_id"])): row
        for row in d2.get("rows") or []
    }
    by_attacker: dict[str, list[dict[str, Any]]] = {name: [] for name in ATTACKERS}
    for row in trajectories:
        attacker = str(row["condition_id"])
        if attacker not in by_attacker:
            continue
        by_attacker[attacker].append(row)
    ordered_ids: list[str] | None = None
    for attacker, rows in by_attacker.items():
        ids = [str(item["anchor_id"]) for item in rows]
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise FinalReportingError(
                f"{attacker} anchor order/set disagrees with the paired arena."
            )
    if not ordered_ids:
        raise FinalReportingError("No paired trajectories found.")
    duplicates = sorted({x for x in ordered_ids if ordered_ids.count(x) > 1})
    return {
        "scientific_status": scientific,
        "manifest": manifest,
        "metrics": metrics,
        "d2": d2,
        "d2_by_key": d2_by_key,
        "by_attacker": by_attacker,
        "anchor_ids": ordered_ids,
        "duplicates": duplicates,
        "n_assigned": len(ordered_ids),
    }


def attacker_denominators(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    n = int(bundle["n_assigned"])
    q_max = int(bundle["manifest"].get("Q") or 5)
    threshold = float(FROZEN_D2S_V10_THRESHOLDS[PRIMARY_REVIEW_BUDGET])
    out: dict[str, dict[str, Any]] = {}
    for attacker in ATTACKERS:
        rows = bundle["by_attacker"][attacker]
        n_valid = 0
        n_d1 = 0
        n_d2 = 0
        n_review = 0
        n_clear = 0
        n_exhaust = 0
        n_invalid = 0
        n_api = 0
        n_q_violation = 0
        first_success_queries: list[int] = []
        for row in rows:
            episode = None
            path = _episode_path(row)
            if path is not None:
                episode = _load_json(path)
            n_valid_here = _count_valid_candidates(episode)
            if n_valid_here >= 1:
                n_valid += 1
            n_invalid += int(row.get("invalid_submissions") or 0)
            if int(row.get("q_used") or 0) > q_max:
                n_q_violation += 1
            if row.get("api_failure") or is_api_failure(row.get("stop_reason")):
                n_api += 1
            if str(row.get("stop_reason") or "") in EXHAUSTION_REASONS:
                n_exhaust += 1
            first = row.get("first_success")
            if first:
                n_d1 += 1
                n_d2 += 1
                attempt = first.get("attempt")
                if attempt is not None:
                    first_success_queries.append(int(attempt))
                d2_row = bundle["d2_by_key"].get((attacker, str(row["anchor_id"])))
                score = None if d2_row is None else d2_row.get("d2s_v10_score")
                decision = d2_decision(score, threshold)
                if decision == "REVIEW":
                    n_review += 1
                elif decision == "CLEAR":
                    n_clear += 1
        if n_d1 + (n - n_d1) != n:
            raise FinalReportingError(f"{attacker} D1 PASS/fail split != assigned N.")
        asr = wilson_interval(n_d1, n)
        e2e = wilson_interval(n_clear, n)
        out[attacker] = {
            "assigned_eligible_anchors": n,
            "anchors_with_ge1_valid_candidate": n_valid,
            "anchors_with_d1_pass": n_d1,
            "anchors_without_d1_pass": n - n_d1,
            "anchors_reaching_d2s": n_d2,
            "d2s_review": n_review,
            "d2s_clear": n_clear,
            "exhausted_action_spaces": n_exhaust,
            "invalid_proposal_count": n_invalid,
            "api_or_runner_exceptions": n_api,
            "query_budget_violations": n_q_violation,
            "d1_asr_at_5": asr["p"],
            "d1_asr_at_5_interval": asr,
            "e2e_bypass_at_10pct": e2e["p"],
            "e2e_bypass_at_10pct_interval": e2e,
            "conditional_d2_interception": (n_review / n_d1) if n_d1 else None,
            "mean_first_success_query": (
                sum(first_success_queries) / len(first_success_queries)
                if first_success_queries
                else None
            ),
            "rq1_denominator": n,
            "rq2_denominator": n,
        }
        if out[attacker]["rq1_denominator"] != n or out[attacker]["rq2_denominator"] != n:
            raise FinalReportingError("Primary denominators drifted from assigned N.")
    dens = {out[name]["rq1_denominator"] for name in ATTACKERS}
    dens |= {out[name]["rq2_denominator"] for name in ATTACKERS}
    if dens != {n}:
        raise FinalReportingError("RQ1/RQ2 denominators are not identical across attackers.")
    return out


def paired_first_success_table(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor_id in bundle["anchor_ids"]:
        record: dict[str, Any] = {"anchor_id": anchor_id}
        for attacker in ATTACKERS:
            item = next(
                (
                    row
                    for row in bundle["by_attacker"][attacker]
                    if str(row["anchor_id"]) == str(anchor_id)
                ),
                None,
            )
            if item is None:
                record[attacker] = "missing"
                record[f"{attacker}_exhaustion"] = False
                continue
            first = item.get("first_success")
            record[attacker] = None if not first else int(first.get("attempt") or 0)
            record[f"{attacker}_exhaustion"] = (
                str(item.get("stop_reason") or "") in EXHAUSTION_REASONS
            )
        rows.append(record)
    return rows


def paired_binary_outcomes(
    bundle: Mapping[str, Any], *, metric: str
) -> list[dict[str, Any]]:
    """One row per assigned anchor with paired 0/1 outcomes for A0–A3."""
    if metric not in {"d1_asr", "e2e"}:
        raise FinalReportingError(f"Unsupported paired metric {metric!r}.")
    threshold = float(FROZEN_D2S_V10_THRESHOLDS[PRIMARY_REVIEW_BUDGET])
    rows: list[dict[str, Any]] = []
    for anchor_id in bundle["anchor_ids"]:
        record: dict[str, Any] = {"anchor_id": str(anchor_id)}
        for attacker in ATTACKERS:
            item = next(
                (
                    row
                    for row in bundle["by_attacker"][attacker]
                    if str(row["anchor_id"]) == str(anchor_id)
                ),
                None,
            )
            if item is None:
                raise FinalReportingError(
                    f"Missing paired episode for {attacker} anchor {anchor_id}."
                )
            first = item.get("first_success")
            if metric == "d1_asr":
                record[attacker] = 1 if first else 0
            else:
                if not first:
                    record[attacker] = 0
                    continue
                d2_row = bundle["d2_by_key"].get((attacker, str(anchor_id)))
                score = None if d2_row is None else d2_row.get("d2s_v10_score")
                record[attacker] = 1 if d2_decision(score, threshold) == "CLEAR" else 0
        rows.append(record)
    return rows


def paired_bootstrap_differences(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    seed: int = ANALYSIS_SEED,
    contrasts: Sequence[tuple[str, str]] = PLANNED_CONTRASTS,
) -> list[dict[str, Any]]:
    """Anchor-level paired bootstrap 95% CIs for attacker rate differences.

    Resamples anchors with replacement and keeps all four policy outcomes
    inside each sampled anchor. Does not interpret a p-value threshold.
    """
    if n_resamples < 1:
        raise FinalReportingError("n_resamples must be >= 1.")
    n = len(paired_rows)
    if n < 1:
        raise FinalReportingError("paired bootstrap requires at least one anchor.")
    matrix = {
        name: np.array([int(row[name]) for row in paired_rows], dtype=np.int8)
        for name in ATTACKERS
    }
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for left, right in contrasts:
        observed = float(matrix[left].mean() - matrix[right].mean())
        deltas = np.empty(int(n_resamples), dtype=np.float64)
        for i in range(int(n_resamples)):
            idx = rng.integers(0, n, size=n)
            deltas[i] = float(matrix[left][idx].mean() - matrix[right][idx].mean())
        low, high = np.quantile(deltas, [0.025, 0.975])
        out.append(
            {
                "contrast": f"{left}-{right}",
                "left": left,
                "right": right,
                "observed_difference": observed,
                "ci_low": float(low),
                "ci_high": float(high),
                "n_anchors": n,
                "n_resamples": int(n_resamples),
                "analysis_seed": int(seed),
                "method": "anchor_level_paired_bootstrap_percentile",
            }
        )
    return out


def sensitivity_table(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    n = int(bundle["n_assigned"])
    rows: list[dict[str, Any]] = []
    for attacker in ATTACKERS:
        for budget, threshold in FROZEN_D2S_V10_THRESHOLDS.items():
            n_d1 = 0
            n_review = 0
            n_clear = 0
            for row in bundle["by_attacker"][attacker]:
                if not row.get("first_success"):
                    continue
                n_d1 += 1
                d2_row = bundle["d2_by_key"].get((attacker, str(row["anchor_id"])))
                score = None if d2_row is None else d2_row.get("d2s_v10_score")
                decision = d2_decision(score, threshold)
                if decision == "REVIEW":
                    n_review += 1
                elif decision == "CLEAR":
                    n_clear += 1
            rows.append(
                {
                    "attacker": attacker,
                    "review_budget": budget,
                    "role": "PRIMARY" if abs(budget - 0.10) < 1e-12 else "SENSITIVITY",
                    "threshold": threshold,
                    "assigned_n": n,
                    "d1_pass": n_d1,
                    "d2_review": n_review,
                    "d2_clear": n_clear,
                    "conditional_interception": (n_review / n_d1) if n_d1 else None,
                    "e2e_bypass": (n_clear / n) if n else None,
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_audit_markdown(
    bundle: Mapping[str, Any],
    denominators: Mapping[str, Any],
    *,
    d1_contrasts: Sequence[Mapping[str, Any]] | None = None,
    e2e_contrasts: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    manifest = bundle["manifest"]
    status = bundle["scientific_status"]
    n = bundle["n_assigned"]
    mode = status.get("mode") or manifest.get("mode")
    month7 = bool(status.get("month7_accessed"))
    month7_line = (
        "Month 7 was accessed by this authorised final production run."
        if month7 and mode == "production"
        else "Month 7 was not accessed by this run (rehearsal/dry-run/reporting only)."
    )
    lines = [
        "# Final evaluation audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Scientific status: {status.get('status')}",
        f"Run mode: {mode}",
        "",
        "## A. Protocol integrity",
        "",
        f"- Assigned eligible anchors (this run): **{n}**",
        f"- Reporting protocol context: {manifest.get('provenance_source')}",
        f"- reconstructed_post_run: {manifest.get('reconstructed_post_run')}",
        f"- Frozen protocol hash (reporting assembly): {manifest.get('protocol_config_hash')}",
        f"- Common paired anchor IDs: `{', '.join(str(x) for x in bundle['anchor_ids'])}`",
        f"- Duplicate anchors: {bundle['duplicates'] or 'none'}",
        f"- Missing per-attacker episodes: none (paired order validated)",
        "- Month-7 eligibility: evaluation month 7, fraud_bool=1, original frozen D1 BLOCK, "
        "valid under frozen preprocessing/reference construction; not conditioned on attacker outcomes",
        "- FINAL_N = 100 unique paired eligible anchors; 100×4 is not N=400 independent observations",
        f"- K = {manifest.get('K')}",
        f"- m = {manifest.get('m')}",
        f"- Q = {manifest.get('Q')}",
        f"- D1 identity: {manifest.get('d1_identifier')}",
        f"- D1 threshold: {manifest.get('d1_threshold')}",
        f"- D2-S identity: {manifest.get('d2_v10_fingerprint')} at 10% legitimate-review operating point",
        f"- A0–A3 pins: {json.dumps(manifest.get('attacker_pins'), sort_keys=True)}",
        f"- LLM model: {json.dumps(manifest.get('model_names'), sort_keys=True)}; Thinking {manifest.get('thinking')}",
        f"- System fingerprints: {json.dumps(manifest.get('system_fingerprints'), sort_keys=True)}",
        f"- API / retries: {json.dumps(manifest.get('call_counts'), sort_keys=True)}; {json.dumps(manifest.get('retries'), sort_keys=True)}",
        f"- Query-budget violations (q_used > 5): "
        + str(sum(int(denominators[a]['query_budget_violations']) for a in ATTACKERS)),
        f"- Candidate validity (invalid proposal counts): "
        + str({a: denominators[a]['invalid_proposal_count'] for a in ATTACKERS}),
        f"- {month7_line}",
        "",
        "## B. Evaluation denominators",
        "",
        "Primary RQ1 and RQ2 denominators are the assigned paired eligible-anchor count "
        f"N={n} for every attacker.",
        "",
        "| Attacker | Assigned | ≥1 valid candidate | D1 PASS | no D1 PASS | reach D2-S | REVIEW | CLEAR | exhaustion | invalid proposals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for attacker in ATTACKERS:
        d = denominators[attacker]
        lines.append(
            f"| {attacker} | {d['assigned_eligible_anchors']} | "
            f"{d['anchors_with_ge1_valid_candidate']} | {d['anchors_with_d1_pass']} | "
            f"{d['anchors_without_d1_pass']} | {d['anchors_reaching_d2s']} | "
            f"{d['d2s_review']} | {d['d2s_clear']} | {d['exhausted_action_spaces']} | "
            f"{d['invalid_proposal_count']} |"
        )
    lines += [
        "",
        f"RQ1 denominator identity: {[denominators[a]['rq1_denominator'] for a in ATTACKERS]}",
        f"RQ2 denominator identity: {[denominators[a]['rq2_denominator'] for a in ATTACKERS]}",
        "",
        "## C. Primary outcomes",
        "",
        "No post-hoc primary outcomes are defined.",
        "",
        "### RQ1 — D1 ASR@5",
        "",
        "| Attacker | k/N | ASR@5 | Wilson 95% CI |",
        "|---|---:|---:|---|",
    ]
    for attacker in ATTACKERS:
        d = denominators[attacker]
        interval = d["d1_asr_at_5_interval"]
        lines.append(
            f"| {attacker} | {interval['k']}/{interval['n']} | "
            f"{d['d1_asr_at_5']} | [{interval['low']}, {interval['high']}] |"
        )
    lines += [
        "",
        "### RQ2 — end-to-end bypass at 10% legitimate-review capacity",
        "",
        "| Attacker | CLEAR/N | E2E bypass | Wilson 95% CI |",
        "|---|---:|---:|---|",
    ]
    for attacker in ATTACKERS:
        d = denominators[attacker]
        interval = d["e2e_bypass_at_10pct_interval"]
        lines.append(
            f"| {attacker} | {interval['k']}/{interval['n']} | "
            f"{d['e2e_bypass_at_10pct']} | [{interval['low']}, {interval['high']}] |"
        )
    lines += [
        "",
        f"Uncertainty method: {UNCERTAINTY_METHOD_NOTE}",
        "",
        "### Planned paired contrasts (bootstrap 95% CI)",
        "",
        "Contrasts are the complete A0–A3 pairwise set. Methodology did not enumerate a subset.",
        "This section is written by the reporting stage after a COMPLETE scientific run.",
        "",
        "| Metric | Contrast | Observed difference | Bootstrap 95% CI |",
        "|---|---|---:|---|",
    ]
    for metric, rows in (("D1 ASR@5", d1_contrasts or ()), ("E2E@10%", e2e_contrasts or ())):
        for row in rows:
            lines.append(
                f"| {metric} | {row['contrast']} | {row['observed_difference']} | "
                f"[{row['ci_low']}, {row['ci_high']}] |"
            )
    lines += [
        "",
        "## D. Secondary diagnostics",
        "",
        "| Attacker | Conditional D2 interception | Mean first-success query | Exhaustion | API/runner exceptions |",
        "|---|---:|---:|---:|---:|",
    ]
    for attacker in ATTACKERS:
        d = denominators[attacker]
        lines.append(
            f"| {attacker} | {d['conditional_d2_interception']} | "
            f"{d['mean_first_success_query']} | {d['exhausted_action_spaces']} | "
            f"{d['api_or_runner_exceptions']} |"
        )
    if mode != "production":
        lines += [
            "",
            "## Notice",
            "",
            "This audit was generated from a non-production run. Numeric values are "
            "reporting-pipeline diagnostics, not Month-7 dissertation results.",
        ]
    return "\n".join(lines) + "\n"


def generate_post_run_report(run_dir: Path) -> dict[str, Any]:
    """Build audit, tables and figures for a COMPLETE scientific run.

    Failures raise FinalReportingError after writing REPORTING_STATUS=FAILED.
    RUN_STATUS.json is never modified.
    """
    run_dir = Path(run_dir)
    reporting_dir = run_dir / "reporting"
    tables_dir = reporting_dir / "tables"
    figures_dir = reporting_dir / "figures"
    try:
        write_reporting_status(run_dir, "RUNNING")
        scientific_before = _load_json(run_dir / "RUN_STATUS.json")
        bundle = load_run_tables(run_dir)
        persist_reporting_context(run_dir, bundle["manifest"])
        denominators = attacker_denominators(bundle)
        paired = paired_first_success_table(bundle)
        sensitivity = sensitivity_table(bundle)
        d1_contrasts = paired_bootstrap_differences(
            paired_binary_outcomes(bundle, metric="d1_asr")
        )
        e2e_contrasts = paired_bootstrap_differences(
            paired_binary_outcomes(bundle, metric="e2e")
        )
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / "protocol_integrity.json").write_text(
            json.dumps(
                {
                    "n_assigned": bundle["n_assigned"],
                    "anchor_ids": bundle["anchor_ids"],
                    "duplicates": bundle["duplicates"],
                    "mode": bundle["scientific_status"].get("mode"),
                    "month7_accessed": bundle["scientific_status"].get("month7_accessed"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        denom_rows = [{"attacker": name, **denominators[name]} for name in ATTACKERS]
        for row in denom_rows:
            row["d1_asr_at_5_low"] = row["d1_asr_at_5_interval"]["low"]
            row["d1_asr_at_5_high"] = row["d1_asr_at_5_interval"]["high"]
            row["e2e_low"] = row["e2e_bypass_at_10pct_interval"]["low"]
            row["e2e_high"] = row["e2e_bypass_at_10pct_interval"]["high"]
            row.pop("d1_asr_at_5_interval", None)
            row.pop("e2e_bypass_at_10pct_interval", None)
        _write_csv(tables_dir / "denominators.csv", denom_rows)
        _write_csv(tables_dir / "figure_r1_source.csv", denom_rows)
        _write_csv(tables_dir / "figure_r2_source.csv", denom_rows)
        decomp = [
            {
                "attacker": name,
                "no_d1_bypass": denominators[name]["anchors_without_d1_pass"],
                "d1_bypass_d2_review": denominators[name]["d2s_review"],
                "d1_bypass_d2_clear": denominators[name]["d2s_clear"],
                "assigned_n": denominators[name]["assigned_eligible_anchors"],
            }
            for name in ATTACKERS
        ]
        _write_csv(tables_dir / "figure_r3_source.csv", decomp)
        _write_csv(tables_dir / "paired_first_success.csv", paired)
        _write_csv(tables_dir / "figure_r4_source.csv", paired)
        _write_csv(tables_dir / "figure_r5_source.csv", sensitivity)
        _write_csv(tables_dir / "review_sensitivity.csv", sensitivity)
        _write_csv(tables_dir / "paired_bootstrap_d1_asr.csv", d1_contrasts)
        _write_csv(tables_dir / "paired_bootstrap_e2e.csv", e2e_contrasts)
        audit_text = render_audit_markdown(
            bundle,
            denominators,
            d1_contrasts=d1_contrasts,
            e2e_contrasts=e2e_contrasts,
        )
        (run_dir / "FINAL_EVALUATION_AUDIT.md").write_text(audit_text, encoding="utf-8")
        (reporting_dir / "FINAL_EVALUATION_AUDIT.md").write_text(audit_text, encoding="utf-8")
        from attack_lab.final_figures import render_all_final_figures

        figure_paths = render_all_final_figures(
            figures_dir=figures_dir,
            denominators=denom_rows,
            decomposition=decomp,
            paired=paired,
            sensitivity=sensitivity,
            n_assigned=int(bundle["n_assigned"]),
            mode=str(bundle["scientific_status"].get("mode") or "unknown"),
        )
        scientific_after = _load_json(run_dir / "RUN_STATUS.json")
        if scientific_after != scientific_before:
            raise FinalReportingError("Reporting mutated scientific RUN_STATUS.json.")
        write_reporting_status(
            run_dir,
            "COMPLETE",
            n_assigned=bundle["n_assigned"],
            figures=figure_paths,
        )
        return {
            "status": "COMPLETE",
            "run_dir": str(run_dir),
            "audit": str(run_dir / "FINAL_EVALUATION_AUDIT.md"),
            "figures": figure_paths,
        }
    except Exception as exc:
        write_reporting_status(
            run_dir,
            "FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )
        if isinstance(exc, FinalReportingError):
            raise
        raise FinalReportingError(f"{type(exc).__name__}: {exc}") from exc


def maybe_generate_post_run_report(run_dir: Path) -> dict[str, Any]:
    """Reporting wrapper that never mutates scientific status.

    Returns a COMPLETE or FAILED payload. Callers of a successful scientific
    run should treat FAILED reporting as a separate incident.
    """
    try:
        return generate_post_run_report(run_dir)
    except FinalReportingError as exc:
        return {
            "status": "FAILED",
            "run_dir": str(run_dir),
            "error": str(exc),
            "scientific_run_untouched": True,
        }


__all__ = [
    "ANALYSIS_SEED",
    "FinalReportingError",
    "PAIRED_BOOTSTRAP_RESAMPLES",
    "PLANNED_CONTRASTS",
    "attacker_denominators",
    "assemble_reporting_context",
    "generate_post_run_report",
    "load_reporting_context",
    "load_run_tables",
    "maybe_generate_post_run_report",
    "paired_bootstrap_differences",
    "validate_scientific_run",
    "wilson_interval",
    "write_reporting_status",
]

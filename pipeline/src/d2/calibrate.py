"""Month-6 calibration and security curve for a frozen D2-S scorer.

This module must not refit relationships, change bins, or open Month 7.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from attack_lab.benchmark_pins import (
    PINNED_A1_PROMPT_VERSION,
    PINNED_A2_GOWER_POLICY,
    PINNED_A3_PROMPT_VERSION,
    PINNED_D1_ARTEFACT_ID,
    PINNED_GOVERNANCE_FINGERPRINT,
)
from attack_lab.defender import FrozenArtefactPaths
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR
from d2.contract import CALIBRATION_MONTHS, SEALED_MONTHS
from d2.data import DEFAULT_RAW_PATH, load_month6_applications
from d2.errors import D2DataError
from d2.scoring import D2SScorer

REVIEW_BUDGETS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)

PRIMARY_ATTACKER_DIRS: dict[str, str] = {
    "A0": "A0",
    "A1": "A1-Flash",
    "A2": "A2",
    "A3": "A3-Flash",
}
SUPPLEMENTARY_ATTACKER_DIRS: dict[str, str] = {
    "A1-Pro": "A1-Pro",
    "A3-Pro": "A3-Pro",
}

DEFAULT_BENCHMARK_DIR: Path | None = None
# LEGACY_MONTH6_DEFAULT: historical Month-6 development default. The final
# runner must not inherit this path; it receives explicit artefact locations
# from the frozen final protocol config.


@dataclass(frozen=True)
class ProvenanceReport:
    ok: bool
    warnings: tuple[str, ...]
    details: dict[str, Any]


def _assert_not_month7_path(path: Path) -> None:
    text = str(path).lower()
    if "month7" in text or "month_7" in text:
        raise D2DataError(f"Refusing a Month-7 path: {path}")


def verify_benchmark_provenance(benchmark_root: Path) -> ProvenanceReport:
    """Fail closed on Month 7; collect warnings for pin drift."""
    _assert_not_month7_path(benchmark_root)
    run_config_path = benchmark_root / "run_config.json"
    if not run_config_path.is_file():
        raise D2DataError(f"Benchmark run_config.json missing: {run_config_path}")
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if config.get("month7_opened") is not False:
        raise D2DataError("Benchmark artefact reports month7_opened != false.")
    if str(config.get("d1_artefact_id")) != PINNED_D1_ARTEFACT_ID:
        warnings.append(
            f"D1 artefact id {config.get('d1_artefact_id')!r} != {PINNED_D1_ARTEFACT_ID!r}"
        )
    if str(config.get("governance_fingerprint")) != PINNED_GOVERNANCE_FINGERPRINT:
        warnings.append(
            "Governance fingerprint mismatch: "
            f"{config.get('governance_fingerprint')!r} != {PINNED_GOVERNANCE_FINGERPRINT!r}"
        )
    pins = config.get("pins") or {}
    if pins.get("a1_prompt_version") != PINNED_A1_PROMPT_VERSION:
        warnings.append(f"A1 prompt pin drifted: {pins.get('a1_prompt_version')!r}")
    if pins.get("a2_gower_policy") != PINNED_A2_GOWER_POLICY:
        warnings.append(f"A2 Gower pin drifted: {pins.get('a2_gower_policy')!r}")
    if pins.get("a3_prompt_version") != PINNED_A3_PROMPT_VERSION:
        warnings.append(f"A3 prompt pin drifted: {pins.get('a3_prompt_version')!r}")
    details = {
        "benchmark_root": str(benchmark_root),
        "d1_artefact_id": config.get("d1_artefact_id"),
        "governance_fingerprint": config.get("governance_fingerprint"),
        "Q": config.get("Q"),
        "m": config.get("m"),
        "K": config.get("K"),
        "month7_opened": config.get("month7_opened"),
        "pins": pins,
        "status": config.get("status"),
    }
    return ProvenanceReport(ok=len(warnings) == 0, warnings=tuple(warnings), details=details)


def load_month6_d1_threshold(artefact_dir: Path | None = None) -> float:
    paths = FrozenArtefactPaths.from_dir(artefact_dir or DEFAULT_C1_ARTEFACT_DIR)
    paths.require_present()
    _assert_not_month7_path(paths.threshold_path)
    payload = json.loads(paths.threshold_path.read_text(encoding="utf-8"))
    return float(payload["threshold"])


def load_month6_d1_scores(artefact_dir: Path | None = None) -> pd.DataFrame:
    paths = FrozenArtefactPaths.from_dir(artefact_dir or DEFAULT_C1_ARTEFACT_DIR)
    paths.require_present()
    _assert_not_month7_path(paths.scores_path)
    scores = pd.read_csv(paths.scores_path)
    required = {"row_id", "y_true", "y_score"}
    missing = required - set(scores.columns)
    if missing:
        raise D2DataError(f"D1 score CSV missing {sorted(missing)}")
    return scores


def month6_legitimate_d1_pass(
    raw_path: Path,
    *,
    artefact_dir: Path | None = None,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """Month-6 fraud_bool==0 applications that PASS frozen D1."""
    loaded = load_month6_applications(
        raw_path, fraud_bool=0, verify_hash=verify_hash
    )
    if loaded.month7_opened or set(loaded.months) - set(CALIBRATION_MONTHS):
        raise D2DataError("Month-6 load violated the sealed-month boundary.")
    if loaded.frame["month"].isin(list(SEALED_MONTHS)).any():
        raise D2DataError("Sealed-month rows present in the Month-6 frame.")
    scores = load_month6_d1_scores(artefact_dir)
    threshold = load_month6_d1_threshold(artefact_dir)
    merged = loaded.frame.merge(
        scores,
        left_on="source_row_id",
        right_on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if int((merged["y_true"] != 0).sum()):
        raise D2DataError("D1 score join produced y_true != 0 on a legit Month-6 frame.")
    passed = merged.loc[merged["y_score"] < threshold].copy()
    return passed


def thresholds_for_budgets(scores: np.ndarray, budgets: tuple[float, ...] = REVIEW_BUDGETS) -> pd.DataFrame:
    values = np.asarray(scores, dtype="float64")
    n = int(values.size)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        threshold = float(np.quantile(values, 1.0 - float(budget)))
        n_review = int((values >= threshold).sum())
        rows.append(
            {
                "budget": float(budget),
                "threshold": threshold,
                "n_legitimate_d1_pass": n,
                "n_review": n_review,
                "benign_review_rate": n_review / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def extract_d1_pass_features(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return candidate feature dicts for valid D1-PASS submissions."""
    found: list[dict[str, Any]] = []
    for step in episode.get("steps") or []:
        defence = step.get("internal_defence") or {}
        validity = step.get("validity") or {}
        if defence.get("decision") != "PASS":
            continue
        if validity.get("is_valid") is not True:
            continue
        features = validity.get("candidate_features")
        if isinstance(features, Mapping):
            found.append(dict(features))
    return found


def load_attacker_d1_pass_submissions(
    benchmark_root: Path,
    condition_dir: str,
) -> list[dict[str, Any]]:
    _assert_not_month7_path(benchmark_root)
    root = benchmark_root / "benchmark" / condition_dir
    if not root.is_dir():
        raise D2DataError(f"Attacker condition directory missing: {root}")
    submissions: list[dict[str, Any]] = []
    for episode_path in sorted(root.glob("anchor_*/seed_*/episode_result.json")):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        submissions.extend(extract_d1_pass_features(episode))
    return submissions


def score_submissions(scorer: D2SScorer, submissions: list[dict[str, Any]]) -> np.ndarray:
    if not submissions:
        return np.asarray([], dtype="float64")
    return np.asarray(
        [float(scorer.score(item)["d2_score"]) for item in submissions],
        dtype="float64",
    )


def security_rows(
    *,
    budget_table: pd.DataFrame,
    attacker_scores: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in budget_table.to_dict(orient="records"):
        threshold = float(record["threshold"])
        row: dict[str, Any] = {
            "threshold": threshold,
            "benign_review_rate": float(record["benign_review_rate"]),
            "budget": float(record["budget"]),
        }
        for name, scores in attacker_scores.items():
            n = int(scores.size)
            n_review = int((scores >= threshold).sum()) if n else 0
            interception = n_review / n if n else float("nan")
            row[f"{name}_n_d1_pass"] = n
            row[f"{name}_interception"] = interception
            row[f"{name}_full_bypass"] = (1.0 - interception) if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def curve_table(security: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "threshold",
        "benign_review_rate",
        "A0_interception",
        "A1_interception",
        "A2_interception",
        "A3_interception",
        "A0_full_bypass",
        "A1_full_bypass",
        "A2_full_bypass",
        "A3_full_bypass",
    ]
    return security.loc[:, columns].copy()

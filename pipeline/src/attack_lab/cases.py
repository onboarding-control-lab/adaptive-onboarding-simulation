"""Month-6 starting-case discovery for the development attack laboratory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from attack_lab.defender import FrozenArtefactPaths, FrozenXGBoostDefender
from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR
from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.protocol_access import load_dataset_for_protocol

DEFAULT_RAW_PATH = Path(os.getenv("BAF_BASE_CSV", "Base.csv"))


class AttackLabCaseError(RuntimeError):
    """Raised when a development starting case cannot be resolved."""


FORBIDDEN_SPLIT_NAMES = frozenset({"test", "month7", "month_7", "holdout_test"})


@dataclass(frozen=True)
class StartingCase:
    """One reproducible month-6 true-positive application under frozen D1."""

    case_id: str
    source_row_id: int
    label: int
    features: dict[str, Any]
    initial_score: float
    initial_decision: str
    data_split: str = "dev_month6"


def assert_month6_only(split_name: str) -> None:
    """Refuse any attempt to select the sealed month-7 test split."""
    normalised = split_name.strip().lower()
    if (
        normalised in FORBIDDEN_SPLIT_NAMES
        or "month7" in normalised
        or "month_7" in normalised
    ):
        raise AttackLabCaseError(
            "Month 7 / sealed test split cannot be selected in the attack laboratory."
        )
    if normalised not in {"dev", "dev_month6", "month6", "month_6", "development"}:
        raise AttackLabCaseError(
            f"Unsupported split {split_name!r}; only month-6 development is permitted."
        )


def load_feature_frame_for_protocol(
    raw_path: Path,
    *,
    phase: str,
    month: int,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Load one experimental month via the fail-closed protocol contract."""
    if phase == "development" and int(month) != 6:
        raise AttackLabCaseError(
            "Development case loading may request Month 6 only."
        )
    if phase == "final" and int(month) != 7:
        raise AttackLabCaseError("Final case loading may request Month 7 only.")
    loaded = load_dataset_for_protocol(
        raw_path,
        phase=phase,
        allowed_months=(int(month),),
        config=data_config,
    )
    split_name = {6: "dev", 7: "test"}[int(month)]
    if split_name not in loaded.views:
        raise AttackLabCaseError(
            f"Protocol load for month {month} did not produce split {split_name!r}."
        )
    return loaded.views[split_name].X.copy()


def load_month6_feature_frame(
    raw_path: Path,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Load prepared month-6 features without retaining other months."""
    assert_month6_only("dev_month6")
    return load_feature_frame_for_protocol(
        raw_path,
        phase="development",
        month=6,
        data_config=data_config,
    )


def discover_true_positive_case_ids(
    artefact_dir: Path | None = None,
    *,
    threshold: float | None = None,
) -> list[int]:
    """Return source row_ids that are fraud and BLOCKED under frozen D1.

    Prefers the frozen development score CSV so discovery does not rescore
    unless the CSV is absent.
    """
    paths = FrozenArtefactPaths.from_dir(artefact_dir or DEFAULT_C1_ARTEFACT_DIR)
    paths.require_present()
    scores = pd.read_csv(paths.scores_path)
    required = {"row_id", "y_true", "y_score"}
    missing = required - set(scores.columns)
    if missing:
        raise AttackLabCaseError(
            f"Score CSV missing columns {sorted(missing)}: {paths.scores_path}"
        )
    if threshold is None:
        payload = json.loads(paths.threshold_path.read_text(encoding="utf-8"))
        threshold = float(payload["threshold"])
    mask = (scores["y_true"].astype(int) == 1) & (scores["y_score"] >= threshold)
    row_ids = scores.loc[mask, "row_id"].astype(int).tolist()
    if not row_ids:
        raise AttackLabCaseError("No true-positive BLOCKED cases found on month 6.")
    return row_ids


def load_starting_case(
    case_id: str | int,
    *,
    raw_path: Path,
    defender: FrozenXGBoostDefender | None = None,
    artefact_dir: Path | None = None,
    data_config: DataLayerConfig = FROZEN_CONFIG,
) -> StartingCase:
    """Load a month-6 TP case by stable source row_id."""
    assert_month6_only("dev_month6")
    row_id = int(case_id)
    tp_ids = set(discover_true_positive_case_ids(artefact_dir))
    if row_id not in tp_ids:
        raise AttackLabCaseError(
            f"Case {row_id} is not a frozen-D1 true positive (fraud + initially BLOCKED) "
            "on month 6. Naturally missed fraud cases are not valid starting cases."
        )

    frame = load_month6_feature_frame(raw_path, data_config)
    if row_id not in frame.index:
        raise AttackLabCaseError(
            f"Case row_id {row_id} not found in prepared month-6 feature frame."
        )
    row = frame.loc[row_id]
    features = {name: _to_python(row[name]) for name in data_config.feature_columns}

    if defender is None:
        defender = FrozenXGBoostDefender.from_artefact_dir(artefact_dir, data_config)
    internal = defender.score_application(features)
    if internal.decision != "BLOCK":
        raise AttackLabCaseError(
            f"Case {row_id} did not BLOCK under the loaded frozen defender "
            f"(decision={internal.decision}, score={internal.risk_score})."
        )
    return StartingCase(
        case_id=str(row_id),
        source_row_id=row_id,
        label=1,
        features=features,
        initial_score=internal.risk_score,
        initial_decision=internal.decision,
        data_split="dev_month6",
    )


def load_starting_case_for_protocol(
    case_id: str | int,
    *,
    phase: str,
    raw_path: Path,
    defender: FrozenXGBoostDefender,
    artefact_dir: Path | None = None,
    data_config: DataLayerConfig = FROZEN_CONFIG,
    y_true: Mapping[int, int] | None = None,
) -> StartingCase:
    """Load a blocked fraud starting case for an explicit experimental phase.

    Development uses the frozen Month-6 score CSV. Final scoring uses the
    frozen D1 defender on the requested Month-7 row only after the protocol
    loader has retained Month 7. Callers must not invoke the final branch
    during pre-Month-7 hardening.
    """
    if phase == "development":
        return load_starting_case(
            case_id,
            raw_path=raw_path,
            defender=defender,
            artefact_dir=artefact_dir,
            data_config=data_config,
        )
    if phase != "final":
        raise AttackLabCaseError(f"Unsupported case-loading phase {phase!r}.")
    if y_true is None:
        raise AttackLabCaseError(
            "Final eligibility requires an explicit fraud_bool map; "
            "y_true must be supplied before any Month-7 row is scored."
        )

    row_id = int(case_id)
    if int(y_true.get(row_id, 0)) != 1:
        raise AttackLabCaseError(
            f"Case {row_id} is not labelled fraud on the final split."
        )
    frame = load_feature_frame_for_protocol(
        raw_path, phase="final", month=7, data_config=data_config
    )
    if row_id not in frame.index:
        raise AttackLabCaseError(
            f"Case row_id {row_id} not found in prepared Month-7 feature frame."
        )
    row = frame.loc[row_id]
    features = {name: _to_python(row[name]) for name in data_config.feature_columns}
    internal = defender.score_application(features)
    if internal.decision != "BLOCK":
        raise AttackLabCaseError(
            f"Case {row_id} did not BLOCK under the loaded frozen defender "
            f"(decision={internal.decision}, score={internal.risk_score})."
        )
    return StartingCase(
        case_id=str(row_id),
        source_row_id=row_id,
        label=1,
        features=features,
        initial_score=internal.risk_score,
        initial_decision=internal.decision,
        data_split="test_month7",
    )


def _to_python(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    return value

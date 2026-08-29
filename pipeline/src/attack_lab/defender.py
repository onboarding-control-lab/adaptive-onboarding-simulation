"""Frozen D1 statistical defender (read-only inference)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from attack_lab.paths import DEFAULT_C1_ARTEFACT_DIR
from attack_lab.types import DefenceDecision, InternalDefenceResult
from baf_data.config import FROZEN_CONFIG, DataLayerConfig


class AttackLabDefenderError(RuntimeError):
    """Raised when the frozen defender cannot be loaded or scored."""


class Defender(Protocol):
    """Minimal scoring interface used by the attack environment."""

    name: str
    artefact_id: str
    threshold: float

    def score_application(
        self, features: Mapping[str, Any]
    ) -> InternalDefenceResult: ...


@dataclass(frozen=True)
class FrozenArtefactPaths:
    """Locations of the serialised C1 pipeline and threshold record."""

    artefact_dir: Path
    pipeline_path: Path
    threshold_path: Path
    config_path: Path
    scores_path: Path

    @classmethod
    def from_dir(cls, artefact_dir: Path) -> "FrozenArtefactPaths":
        return cls(
            artefact_dir=artefact_dir,
            pipeline_path=artefact_dir / "fitted_pipeline.joblib",
            threshold_path=artefact_dir / "development_month6_threshold_selection.json",
            config_path=artefact_dir / "config.json",
            scores_path=artefact_dir / "development_month6_scores.csv",
        )

    def require_present(self) -> None:
        missing = [
            str(path)
            for path in (
                self.pipeline_path,
                self.threshold_path,
                self.config_path,
                self.scores_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise AttackLabDefenderError(
                "Frozen C1 artefacts incomplete; missing: " + ", ".join(missing)
            )


class FrozenXGBoostDefender:
    """Load-once, inference-only wrapper around the frozen C1 pipeline.

    This class intentionally exposes no training or refit API.
    """

    name: str = "d1_frozen_c1_xgboost"

    def __init__(
        self,
        pipeline: Pipeline,
        threshold: float,
        *,
        artefact_id: str,
        feature_columns: tuple[str, ...],
        artefact_dir: Path,
        config_payload: Mapping[str, Any],
    ) -> None:
        if threshold <= 0.0 or threshold > 1.0:
            raise AttackLabDefenderError(f"Invalid frozen threshold: {threshold}")
        self._pipeline = pipeline
        self.threshold = float(threshold)
        self.artefact_id = artefact_id
        self.feature_columns = feature_columns
        self.artefact_dir = artefact_dir
        self.config_payload = dict(config_payload)

    @classmethod
    def from_artefact_dir(
        cls,
        artefact_dir: Path | None = None,
        data_config: DataLayerConfig = FROZEN_CONFIG,
    ) -> "FrozenXGBoostDefender":
        """Load the serialised C1 pipeline and selected threshold."""
        paths = FrozenArtefactPaths.from_dir(artefact_dir or DEFAULT_C1_ARTEFACT_DIR)
        paths.require_present()
        pipeline = joblib.load(paths.pipeline_path)
        if not isinstance(pipeline, Pipeline):
            raise AttackLabDefenderError(
                f"Expected sklearn Pipeline, got {type(pipeline)!r}."
            )
        if hasattr(pipeline, "fit") and not hasattr(pipeline, "predict_proba"):
            raise AttackLabDefenderError("Loaded object cannot score applications.")
        threshold_payload = json.loads(paths.threshold_path.read_text(encoding="utf-8"))
        if "threshold" not in threshold_payload:
            raise AttackLabDefenderError(
                f"Threshold file missing 'threshold': {paths.threshold_path}"
            )
        config_payload = json.loads(paths.config_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(paths.pipeline_path.read_bytes()).hexdigest()[:16]
        return cls(
            pipeline=pipeline,
            threshold=float(threshold_payload["threshold"]),
            artefact_id=f"c1_pipeline_sha256_16={digest}",
            feature_columns=tuple(data_config.feature_columns),
            artefact_dir=paths.artefact_dir,
            config_payload=config_payload,
        )

    # Explicitly absent training surface.
    def fit(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        raise AttackLabDefenderError(
            "FrozenXGBoostDefender has no training/refit path."
        )

    def refit(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        raise AttackLabDefenderError(
            "FrozenXGBoostDefender has no training/refit path."
        )

    def score_application(self, features: Mapping[str, Any]) -> InternalDefenceResult:
        """Score one complete application feature vector.

        Accepts only feature values. Caller identity, history and budgets
        must not be supplied here and are ignored if present under other keys.
        """
        row = {name: features[name] for name in self.feature_columns}
        frame = pd.DataFrame([row], columns=list(self.feature_columns))
        t0 = time.perf_counter()
        score = float(self._pipeline.predict_proba(frame)[0, 1])
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        decision: DefenceDecision = "BLOCK" if score >= self.threshold else "PASS"
        return InternalDefenceResult(
            risk_score=score,
            threshold=self.threshold,
            decision=decision,
            runtime_ms=runtime_ms,
            defender_name=self.name,
            artefact_id=self.artefact_id,
        )

    def categorical_vocabularies(self) -> dict[str, tuple[Any, ...]]:
        """Return fitted one-hot categories keyed by raw categorical field."""
        preprocessor = self._pipeline.named_steps["preprocessing"]
        cat_pipe = preprocessor.named_transformers_["categorical"]
        encoder = cat_pipe.named_steps["onehot"]
        cat_columns: list[str] | None = None
        for name, _trans, columns in preprocessor.transformers_:
            if name == "categorical":
                cat_columns = [str(c) for c in columns]
                break
        if cat_columns is None:
            raise AttackLabDefenderError(
                "Could not recover categorical column names from frozen preprocessor."
            )
        vocab: dict[str, tuple[Any, ...]] = {}
        for column, categories in zip(cat_columns, encoder.categories_):
            vocab[column] = tuple(categories.tolist())
        return vocab

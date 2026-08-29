"""D2-S v1.1 Isolation Forest aggregator.

Learns only the multivariate structure of frozen D2-S v1.0 relationship
scores on Months 0–5 legitimate applications.  This is not a fraud
classifier, does not consume raw application fields, D1 scores, fraud
labels, or attacker outcomes, and does not modify D2-S v1.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from inspect import getsource
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from d2.contract import FORBIDDEN_INFERENCE_KEYS, RELATIONSHIP_IDS
from d2.errors import D2ContractError, D2FitError
from d2.scoring import D2SScorer

SCORE_CONTRACT_ID_V11 = "d2s-v1.1.0-isolation-forest-20260816"
V10_SCORE_CONTRACT_ID = "d2s-v1.0.0-pairwise8-20260816"
FROZEN_D2S_V10_FINGERPRINT = (
    "cfd5330f096dabb1749be447ee4da4d5f498d2599f4f22c24a0b706e570bfd94"
)

COLLAPSED_RELATIONSHIPS: tuple[str, str] = ("C01", "C14")
PASSTHROUGH_RELATIONSHIPS: tuple[str, ...] = (
    "C13",
    "C09",
    "C03",
    "C10",
    "C11",
    "C15",
)
IFOREST_FEATURE_IDS: tuple[str, ...] = ("payment_channel", *PASSTHROUGH_RELATIONSHIPS)

FIXED_IFOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_samples": 256,
    "max_features": 1.0,
    "bootstrap": False,
    "contamination": "auto",
    "random_state": 20260816,
    "n_jobs": -1,
}


def _assert_relationship_frame(frame: pd.DataFrame) -> None:
    missing = [rid for rid in RELATIONSHIP_IDS if rid not in frame.columns]
    if missing:
        raise D2ContractError(f"Relationship-score frame missing {missing}.")
    leaked = sorted(set(frame.columns).intersection(FORBIDDEN_INFERENCE_KEYS))
    if leaked:
        raise D2ContractError(
            f"Forbidden inference keys present on the relationship frame: {leaked}."
        )


def collapse_to_iforest_features(relationship_scores: pd.DataFrame) -> pd.DataFrame:
    """Map the eight v1.0 scores onto the seven Isolation Forest channels."""
    _assert_relationship_frame(relationship_scores)
    c01 = relationship_scores["C01"].to_numpy(dtype="float64")
    c14 = relationship_scores["C14"].to_numpy(dtype="float64")
    out = pd.DataFrame(index=relationship_scores.index)
    out["payment_channel"] = np.maximum(c01, c14)
    for rid in PASSTHROUGH_RELATIONSHIPS:
        out[rid] = relationship_scores[rid].to_numpy(dtype="float64")
    values = out.to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise D2ContractError("Isolation Forest features contain NaN.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise D2ContractError("Isolation Forest features must lie in [0, 1].")
    if list(out.columns) != list(IFOREST_FEATURE_IDS):
        raise D2ContractError("Isolation Forest feature order drifted.")
    return out


def relationship_scores_from_scorer(
    scorer: D2SScorer, applications: pd.DataFrame
) -> pd.DataFrame:
    """Score applications with frozen D2-S v1.0; return the eight relationships."""
    if scorer.month7_opened:
        raise D2FitError("Refusing a D2-S v1.0 scorer that opened Month 7.")
    scored = scorer.score_many(applications)
    return scored.loc[:, list(RELATIONSHIP_IDS)].copy()


@dataclass
class D2SV11IForestAggregator:
    """Continuous Isolation Forest aggregator over seven consistency channels."""

    model: IsolationForest
    n_train: int
    fitted_utc: str
    sklearn_version: str
    v10_fingerprint: str
    month7_opened: bool
    params: dict[str, Any]

    def score_features(self, features: pd.DataFrame) -> np.ndarray:
        """Return d2s_v11_anomaly_score = -score_samples(X). Higher = more anomalous."""
        if list(features.columns) != list(IFOREST_FEATURE_IDS):
            raise D2ContractError(
                f"Expected features {list(IFOREST_FEATURE_IDS)}; got {list(features.columns)}."
            )
        matrix = features.to_numpy(dtype="float64")
        return -self.model.score_samples(matrix)

    def score_relationship_frame(self, relationship_scores: pd.DataFrame) -> np.ndarray:
        return self.score_features(collapse_to_iforest_features(relationship_scores))

    def to_config(self) -> dict[str, Any]:
        return {
            "score_contract_id": SCORE_CONTRACT_ID_V11,
            "parent_score_contract_id": V10_SCORE_CONTRACT_ID,
            "v10_fingerprint": self.v10_fingerprint,
            "n_train": self.n_train,
            "fitted_utc": self.fitted_utc,
            "sklearn_version": self.sklearn_version,
            "month7_opened": self.month7_opened,
            "iforest_feature_ids": list(IFOREST_FEATURE_IDS),
            "collapsed_relationships": {
                "payment_channel": "max(C01, C14)",
                "from": list(COLLAPSED_RELATIONSHIPS),
            },
            "passthrough_relationships": list(PASSTHROUGH_RELATIONSHIPS),
            "raw_application_features_used": False,
            "fraud_labels_used": False,
            "d1_score_used": False,
            "attacker_outcomes_used": False,
            "predict_used_as_operating_rule": False,
            "anomaly_score_definition": "d2s_v11_anomaly_score = -IsolationForest.score_samples(X)",
            "params": dict(self.params),
        }

    def save(self, model_path: Path, config_path: Path | None = None) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_train": self.n_train,
                "fitted_utc": self.fitted_utc,
                "sklearn_version": self.sklearn_version,
                "v10_fingerprint": self.v10_fingerprint,
                "month7_opened": self.month7_opened,
                "params": dict(self.params),
                "feature_ids": list(IFOREST_FEATURE_IDS),
                "score_contract_id": SCORE_CONTRACT_ID_V11,
            },
            model_path,
        )
        if config_path is not None:
            Path(config_path).write_text(
                json.dumps(self.to_config(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, model_path: Path) -> "D2SV11IForestAggregator":
        payload = joblib.load(Path(model_path))
        if payload.get("score_contract_id") != SCORE_CONTRACT_ID_V11:
            raise D2ContractError(
                f"Model contract {payload.get('score_contract_id')!r} "
                f"!= {SCORE_CONTRACT_ID_V11!r}."
            )
        if list(payload.get("feature_ids") or []) != list(IFOREST_FEATURE_IDS):
            raise D2ContractError("Loaded Isolation Forest feature contract drifted.")
        if payload.get("month7_opened"):
            raise D2FitError("Refusing a v1.1 artefact that opened Month 7.")
        return cls(
            model=payload["model"],
            n_train=int(payload["n_train"]),
            fitted_utc=str(payload["fitted_utc"]),
            sklearn_version=str(payload["sklearn_version"]),
            v10_fingerprint=str(payload["v10_fingerprint"]),
            month7_opened=False,
            params=dict(payload["params"]),
        )


def fit_iforest_aggregator(
    relationship_scores: pd.DataFrame,
    *,
    v10_fingerprint: str,
    month7_opened: bool = False,
    params: Mapping[str, Any] | None = None,
) -> D2SV11IForestAggregator:
    """Fit Isolation Forest on legitimate v1.0 relationship scores only."""
    if month7_opened:
        raise D2FitError("Refusing to fit D2-S v1.1 after Month 7 was opened.")
    if v10_fingerprint != FROZEN_D2S_V10_FINGERPRINT:
        raise D2ContractError(
            f"D2-S v1.0 fingerprint {v10_fingerprint!r} != {FROZEN_D2S_V10_FINGERPRINT!r}."
        )
    work = relationship_scores.copy()
    if "fraud_bool" in work.columns:
        fraud_values = set(int(v) for v in work["fraud_bool"].dropna().unique())
        if fraud_values - {0}:
            raise D2FitError("Training relationship scores include non-legitimate rows.")
    leaked_to_drop = [c for c in work.columns if c in FORBIDDEN_INFERENCE_KEYS]
    if leaked_to_drop:
        work = work.drop(columns=leaked_to_drop)
    features = collapse_to_iforest_features(work)
    frozen_params = dict(FIXED_IFOREST_PARAMS if params is None else params)
    if frozen_params != FIXED_IFOREST_PARAMS:
        raise D2FitError(
            "D2-S v1.1 uses one pre-specified Isolation Forest configuration; "
            f"got {frozen_params}."
        )
    model = IsolationForest(**frozen_params)
    model.fit(features.to_numpy(dtype="float64"))
    import sklearn

    return D2SV11IForestAggregator(
        model=model,
        n_train=int(len(features)),
        fitted_utc=datetime.now(timezone.utc).isoformat(),
        sklearn_version=str(sklearn.__version__),
        v10_fingerprint=v10_fingerprint,
        month7_opened=False,
        params=frozen_params,
    )


def assert_predict_not_used_in_score() -> None:
    source = getsource(D2SV11IForestAggregator.score_features)
    if ".predict(" in source:
        raise D2ContractError("v1.1 scoring must not call IsolationForest.predict().")

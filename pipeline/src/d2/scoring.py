"""D2-S scorer: fit on Months 0–5 legitimate data, score one application."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from baf_data.config import FROZEN_CONFIG
from d2.aggregation import (
    aggregate_equal_mean,
    aggregation_payload,
    redundancy_matrix,
)
from d2.contract import (
    FORBIDDEN_INFERENCE_KEYS,
    RELATIONSHIP_IDS,
    REQUIRED_APPLICATION_FIELDS,
    SCORE_CONTRACT_ID,
    score_contract_payload,
)
from d2.data import apply_official_sentinels
from d2.errors import D2ContractError, D2FitError
from d2.relationships import (
    BinningSpec,
    ConditionalTable,
    fit_all_relationships,
    fit_binning,
    score_relationship_series,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def application_to_frame(features: Mapping[str, Any]) -> pd.DataFrame:
    """Build a one-row frame from application fields; drop leakage keys."""
    missing = [name for name in REQUIRED_APPLICATION_FIELDS if name not in features]
    if missing:
        raise D2ContractError(f"Application missing required fields: {missing}")
    row = {name: features[name] for name in REQUIRED_APPLICATION_FIELDS}
    frame = pd.DataFrame([row], columns=list(REQUIRED_APPLICATION_FIELDS))
    return apply_official_sentinels(frame, FROZEN_CONFIG)


def _assert_reference_population(frame: pd.DataFrame) -> None:
    if "month" not in frame.columns or "fraud_bool" not in frame.columns:
        raise D2FitError("Reference frame must include month and fraud_bool.")
    months = set(int(m) for m in frame["month"].unique())
    if months - {0, 1, 2, 3, 4, 5}:
        raise D2FitError(f"Reference frame contains non-training months: {sorted(months)}.")
    fraud = set(int(v) for v in frame["fraud_bool"].unique())
    if fraud - {0}:
        raise D2FitError("Reference frame must contain only fraud_bool == 0 rows.")
    if len(frame) == 0:
        raise D2FitError("Reference frame is empty.")


@dataclass
class D2SScorer:
    """Frozen statistical consistency reviewer."""

    bins: BinningSpec
    tables: dict[str, ConditionalTable]
    reference_n: int
    reference_sha256: str
    reference_months: tuple[int, ...]
    redundancy: dict[str, Any]
    fitted_utc: str
    month7_opened: bool
    fingerprint: str

    def score(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Score one application. Attacked and untouched rows use this function."""
        _ = FORBIDDEN_INFERENCE_KEYS  # documented ignore-set; values are never read
        frame = application_to_frame(features)
        relationship_scores = {
            rid: float(score_relationship_series(frame, rid, self.tables[rid], self.bins)[0])
            for rid in RELATIONSHIP_IDS
        }
        for rid, value in relationship_scores.items():
            if not (0.0 <= value <= 1.0):
                raise D2ContractError(f"{rid} produced score {value} outside [0, 1].")
        return {
            "relationship_scores": relationship_scores,
            "d2_score": aggregate_equal_mean(relationship_scores),
        }

    def score_many(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Vectorised scoring for calibration / redundancy inspection."""
        required = [c for c in REQUIRED_APPLICATION_FIELDS if c not in frame.columns]
        if required:
            raise D2ContractError(f"Frame missing required fields: {required}")
        work = apply_official_sentinels(
            frame.loc[:, list(REQUIRED_APPLICATION_FIELDS)].copy(),
            FROZEN_CONFIG,
        )
        out: dict[str, np.ndarray] = {}
        for rid in RELATIONSHIP_IDS:
            out[rid] = score_relationship_series(work, rid, self.tables[rid], self.bins)
        result = pd.DataFrame(out, index=frame.index)
        result["d2_score"] = result[list(RELATIONSHIP_IDS)].mean(axis=1)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_contract": score_contract_payload(),
            "bins": self.bins.to_dict(),
            "tables": {rid: self.tables[rid].to_dict() for rid in RELATIONSHIP_IDS},
            "reference_n": self.reference_n,
            "reference_sha256": self.reference_sha256,
            "reference_months": list(self.reference_months),
            "redundancy": self.redundancy,
            "aggregation": aggregation_payload(),
            "fitted_utc": self.fitted_utc,
            "month7_opened": self.month7_opened,
            "fingerprint": self.fingerprint,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "D2SScorer":
        contract_id = payload.get("score_contract", {}).get("score_contract_id")
        if contract_id != SCORE_CONTRACT_ID:
            raise D2ContractError(
                f"Artefact contract {contract_id!r} != {SCORE_CONTRACT_ID!r}."
            )
        tables = {
            rid: ConditionalTable.from_dict(payload["tables"][rid])
            for rid in RELATIONSHIP_IDS
        }
        if set(tables) != set(RELATIONSHIP_IDS):
            raise D2ContractError("Artefact does not contain exactly the eight relationships.")
        return cls(
            bins=BinningSpec.from_dict(payload["bins"]),
            tables=tables,
            reference_n=int(payload["reference_n"]),
            reference_sha256=str(payload["reference_sha256"]),
            reference_months=tuple(int(m) for m in payload["reference_months"]),
            redundancy=dict(payload.get("redundancy") or {}),
            fitted_utc=str(payload["fitted_utc"]),
            month7_opened=bool(payload["month7_opened"]),
            fingerprint=str(payload["fingerprint"]),
        )

    @classmethod
    def load(cls, path: Path) -> "D2SScorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fit_d2s_scorer(
    reference: pd.DataFrame,
    *,
    raw_sha256: str,
    month7_opened: bool = False,
) -> D2SScorer:
    """Fit bins, conditionals, rarity CDFs and the redundancy matrix."""
    if month7_opened:
        raise D2FitError("Refusing to fit a scorer after Month 7 was opened.")
    _assert_reference_population(reference)
    prepared = apply_official_sentinels(reference, FROZEN_CONFIG)
    bins = fit_binning(prepared)
    tables = fit_all_relationships(prepared, bins)
    scored = {}
    work = prepared.loc[:, list(REQUIRED_APPLICATION_FIELDS)].copy()
    for rid in RELATIONSHIP_IDS:
        scored[rid] = score_relationship_series(work, rid, tables[rid], bins)
    redundancy = redundancy_matrix(scored)
    fitted_utc = datetime.now(timezone.utc).isoformat()
    months = tuple(sorted(int(m) for m in reference["month"].unique()))
    draft = {
        "score_contract_id": SCORE_CONTRACT_ID,
        "bins": bins.to_dict(),
        "tables": {rid: tables[rid].to_dict() for rid in RELATIONSHIP_IDS},
        "reference_n": int(len(reference)),
        "reference_sha256": raw_sha256,
        "reference_months": list(months),
        "aggregation": aggregation_payload(),
        "month7_opened": False,
    }
    fingerprint = _fingerprint(draft)
    return D2SScorer(
        bins=bins,
        tables=tables,
        reference_n=int(len(reference)),
        reference_sha256=raw_sha256,
        reference_months=months,
        redundancy=redundancy,
        fitted_utc=fitted_utc,
        month7_opened=False,
        fingerprint=fingerprint,
    )

"""Redundancy inspection and the frozen V1 global aggregate.

Aggregation is a transparent equal-weight mean of the eight standardized
relationship scores.  Editability is not a weight.  Fraud labels and
attacker outcomes are not used.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from d2.contract import (
    AGGREGATION_FORMULA,
    AGGREGATION_METHOD_ID,
    REDUNDANCY_REVIEW_SPEARMAN_FLAG,
    RELATIONSHIP_IDS,
)
from d2.errors import D2ContractError


def relationship_score_matrix(scores: Mapping[str, np.ndarray]) -> pd.DataFrame:
    missing = [rid for rid in RELATIONSHIP_IDS if rid not in scores]
    if missing:
        raise D2ContractError(f"Redundancy inspect missing relationships: {missing}")
    data = {rid: np.asarray(scores[rid], dtype="float64") for rid in RELATIONSHIP_IDS}
    lengths = {rid: arr.size for rid, arr in data.items()}
    if len(set(lengths.values())) != 1:
        raise D2ContractError(f"Score length mismatch: {lengths}")
    return pd.DataFrame(data)


def redundancy_matrix(scores: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Spearman and Pearson correlations among the eight standardized scores."""
    frame = relationship_score_matrix(scores)
    spearman = frame.corr(method="spearman")
    pearson = frame.corr(method="pearson")
    off = spearman.to_numpy().copy()
    np.fill_diagonal(off, np.nan)
    max_abs = float(np.nanmax(np.abs(off))) if off.size else 0.0
    flagged: list[list[object]] = []
    ids = list(RELATIONSHIP_IDS)
    for i, left in enumerate(ids):
        for j, right in enumerate(ids):
            if j <= i:
                continue
            value = float(spearman.iloc[i, j])
            if abs(value) >= REDUNDANCY_REVIEW_SPEARMAN_FLAG:
                flagged.append([left, right, value])
    return {
        "method": ["spearman", "pearson"],
        "spearman": spearman.round(6).to_dict(),
        "pearson": pearson.round(6).to_dict(),
        "max_abs_spearman_offdiag": max_abs,
        "flag_threshold": REDUNDANCY_REVIEW_SPEARMAN_FLAG,
        "flagged_pairs": flagged,
        "n_rows": int(len(frame)),
    }


def aggregate_equal_mean(relationship_scores: Mapping[str, float]) -> float:
    """d2_score = mean of the eight standardized relationship scores."""
    missing = [rid for rid in RELATIONSHIP_IDS if rid not in relationship_scores]
    extra = [k for k in relationship_scores if k not in RELATIONSHIP_IDS]
    if missing or extra:
        raise D2ContractError(
            f"Aggregation expects exactly {list(RELATIONSHIP_IDS)}; "
            f"missing={missing}, extra={extra}."
        )
    values = [float(relationship_scores[rid]) for rid in RELATIONSHIP_IDS]
    if any(not (0.0 <= v <= 1.0) for v in values):
        raise D2ContractError("Relationship scores must lie in [0, 1] before aggregation.")
    return float(sum(values) / len(values))


def aggregation_payload() -> dict[str, object]:
    return {
        "method_id": AGGREGATION_METHOD_ID,
        "formula": AGGREGATION_FORMULA,
        "n_signals": len(RELATIONSHIP_IDS),
        "weights": {rid: 1.0 / len(RELATIONSHIP_IDS) for rid in RELATIONSHIP_IDS},
        "editability_used_as_weight": False,
        "fraud_trained": False,
        "attacker_outcomes_used": False,
    }

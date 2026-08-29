"""Empirical pairwise conditionals for the eight qualified D2-S relationships.

No three-way or four-way tables are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from d2.contract import (
    CURRENT_ADDRESS_N_BINS,
    DOB_EMAIL_N_BINS,
    LAPLACE_ALPHA,
    RELATIONSHIP_IDS,
)
from d2.errors import D2FitError

MISSING_BIN = "MISSING"
UNSEEN_CONDITIONER_SCORE = 1.0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def presence_label(value: Any) -> str:
    return "0" if _is_missing(value) else "1"


def missingness_label(value: Any) -> str:
    return "1" if _is_missing(value) else "0"


def category_label(value: Any) -> str:
    if _is_missing(value):
        return MISSING_BIN
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> tuple[float, ...]:
    """Strictly increasing edges from the supplied finite values only."""
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise D2FitError("Cannot derive quantile bins from an empty numeric series.")
    raw = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges: list[float] = [float(raw[0])]
    for item in raw[1:]:
        value = float(item)
        if value > edges[-1]:
            edges.append(value)
    if len(edges) < 2:
        edges.append(edges[0] + 1.0)
    return tuple(edges)


def assign_quantile_bin(value: Any, edges: tuple[float, ...]) -> str:
    if _is_missing(value):
        return MISSING_BIN
    number = float(value)
    if not np.isfinite(number):
        return MISSING_BIN
    clipped = min(max(number, edges[0]), edges[-1])
    # Inner edges; digitize with right=True → bins 0 .. n-1.
    inner = np.asarray(edges[1:-1], dtype="float64")
    if inner.size == 0:
        idx = 0
    else:
        idx = int(np.digitize(clipped, inner, right=True))
    return f"Q{idx + 1}"


@dataclass
class ConditionalTable:
    """Laplace-smoothed P(Y|X) plus the reference rarity CDF."""

    relationship_id: str
    n_xy: dict[str, dict[str, int]] = field(default_factory=dict)
    n_x: dict[str, int] = field(default_factory=dict)
    y_levels: tuple[str, ...] = ()
    rarity_values: tuple[float, ...] = ()
    rarity_cdf_less: tuple[float, ...] = ()
    n_reference: int = 0

    def probability(self, x: str, y: str) -> float:
        n_x = self.n_x.get(x)
        if n_x is None:
            return 0.0
        k = len(self.y_levels)
        extra = 0 if y in self.y_levels else 1
        count = self.n_xy.get(x, {}).get(y, 0)
        return (count + LAPLACE_ALPHA) / (n_x + LAPLACE_ALPHA * (k + extra))

    def raw_rarity(self, x: str, y: str) -> float:
        if x not in self.n_x:
            return 1.0
        return 1.0 - float(self.probability(x, y))

    def standardized_score(self, x: str, y: str) -> float:
        if x not in self.n_x:
            return UNSEEN_CONDITIONER_SCORE
        rarity = self.raw_rarity(x, y)
        return self.cdf_strict_less(rarity)

    def cdf_strict_less(self, rarity: float) -> float:
        """s = P_ref(R < rarity).  Most ordinary → 0; above max → 1."""
        if not self.rarity_values:
            return 1.0
        values = np.asarray(self.rarity_values, dtype="float64")
        cdfs = np.asarray(self.rarity_cdf_less, dtype="float64")
        idx = int(np.searchsorted(values, rarity, side="left"))
        if idx == 0:
            if rarity <= values[0]:
                return 0.0
        if idx >= len(values):
            return 1.0
        return float(cdfs[idx])

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "n_xy": self.n_xy,
            "n_x": self.n_x,
            "y_levels": list(self.y_levels),
            "rarity_values": list(self.rarity_values),
            "rarity_cdf_less": list(self.rarity_cdf_less),
            "n_reference": self.n_reference,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditionalTable":
        return cls(
            relationship_id=str(payload["relationship_id"]),
            n_xy={
                str(x): {str(y): int(c) for y, c in ymap.items()}
                for x, ymap in dict(payload["n_xy"]).items()
            },
            n_x={str(x): int(n) for x, n in dict(payload["n_x"]).items()},
            y_levels=tuple(str(y) for y in payload["y_levels"]),
            rarity_values=tuple(float(v) for v in payload["rarity_values"]),
            rarity_cdf_less=tuple(float(v) for v in payload["rarity_cdf_less"]),
            n_reference=int(payload["n_reference"]),
        )


def _fit_conditional(relationship_id: str, x: pd.Series, y: pd.Series) -> ConditionalTable:
    if len(x) != len(y) or len(x) == 0:
        raise D2FitError(f"{relationship_id}: empty or misaligned conditional sample.")
    n_xy: dict[str, dict[str, int]] = {}
    n_x: dict[str, int] = {}
    y_set: set[str] = set()
    x_labels = x.astype(str).to_numpy()
    y_labels = y.astype(str).to_numpy()
    for xv, yv in zip(x_labels, y_labels, strict=True):
        n_x[xv] = n_x.get(xv, 0) + 1
        bucket = n_xy.setdefault(xv, {})
        bucket[yv] = bucket.get(yv, 0) + 1
        y_set.add(yv)
    y_levels = tuple(sorted(y_set))
    table = ConditionalTable(
        relationship_id=relationship_id,
        n_xy=n_xy,
        n_x=n_x,
        y_levels=y_levels,
        n_reference=len(x),
    )
    rarities = np.array(
        [table.raw_rarity(xv, yv) for xv, yv in zip(x_labels, y_labels, strict=True)],
        dtype="float64",
    )
    unique, counts = np.unique(np.round(rarities, decimals=12), return_counts=True)
    order = np.argsort(unique)
    unique = unique[order]
    counts = counts[order]
    n = float(len(rarities))
    cdf_less = np.concatenate([[0.0], np.cumsum(counts[:-1]) / n])
    table.rarity_values = tuple(float(v) for v in unique)
    table.rarity_cdf_less = tuple(float(v) for v in cdf_less)
    return table


@dataclass(frozen=True)
class BinningSpec:
    current_address_edges: tuple[float, ...]
    dob_email_edges: tuple[float, ...]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "current_address_edges": list(self.current_address_edges),
            "dob_email_edges": list(self.dob_email_edges),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BinningSpec":
        return cls(
            current_address_edges=tuple(float(v) for v in payload["current_address_edges"]),
            dob_email_edges=tuple(float(v) for v in payload["dob_email_edges"]),
        )


def fit_binning(frame: pd.DataFrame) -> BinningSpec:
    current = pd.to_numeric(frame["current_address_months_count"], errors="coerce")
    dob = pd.to_numeric(frame["date_of_birth_distinct_emails_4w"], errors="coerce")
    return BinningSpec(
        current_address_edges=quantile_bin_edges(
            current.to_numpy(), CURRENT_ADDRESS_N_BINS
        ),
        dob_email_edges=quantile_bin_edges(dob.to_numpy(), DOB_EMAIL_N_BINS),
    )


def _presence_series(values: pd.Series) -> pd.Series:
    return pd.Series(np.where(values.isna(), "0", "1"), index=values.index)


def _missingness_series(values: pd.Series) -> pd.Series:
    return pd.Series(np.where(values.isna(), "1", "0"), index=values.index)


def _category_series(values: pd.Series) -> pd.Series:
    out = values.astype("string")
    out = out.where(~values.isna(), MISSING_BIN)
    # Integer-like floats (0.0 / 1.0) become "0" / "1".
    numeric = pd.to_numeric(values, errors="coerce")
    integer_mask = numeric.notna() & np.isclose(numeric % 1, 0)
    out = out.mask(integer_mask, numeric.round().astype("Int64").astype("string"))
    return out.astype(str)


def _bin_series(values: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    n_bins = max(1, len(edges) - 1)
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    clipped = numeric.clip(lower=edges[0], upper=edges[-1])
    binned = pd.cut(
        clipped,
        bins=list(edges),
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    return pd.Series(
        np.where(numeric.isna(), MISSING_BIN, binned.astype("string")),
        index=values.index,
    ).astype(str)


def pair_labels(
    frame: pd.DataFrame,
    relationship_id: str,
    bins: BinningSpec,
) -> tuple[pd.Series, pd.Series]:
    """Return (X, Y) string labels for one pairwise relationship."""
    if relationship_id == "C01":
        return _category_series(frame["payment_type"]), _presence_series(
            frame["bank_months_count"]
        )
    if relationship_id == "C14":
        return _category_series(frame["payment_type"]), _presence_series(
            frame["intended_balcon_amount"]
        )
    if relationship_id == "C13":
        return (
            _bin_series(frame["current_address_months_count"], bins.current_address_edges),
            _missingness_series(frame["prev_address_months_count"]),
        )
    if relationship_id == "C09":
        return (
            _category_series(frame["housing_status"]),
            _bin_series(frame["current_address_months_count"], bins.current_address_edges),
        )
    if relationship_id == "C03":
        return (
            _category_series(frame["customer_age"]),
            _bin_series(frame["date_of_birth_distinct_emails_4w"], bins.dob_email_edges),
        )
    if relationship_id == "C10":
        return (
            _category_series(frame["housing_status"]),
            _category_series(frame["customer_age"]),
        )
    if relationship_id == "C11":
        return (
            _category_series(frame["employment_status"]),
            _category_series(frame["customer_age"]),
        )
    if relationship_id == "C15":
        return (
            _category_series(frame["phone_home_valid"]),
            _category_series(frame["phone_mobile_valid"]),
        )
    raise D2FitError(f"Unknown pairwise relationship: {relationship_id!r}")


def fit_all_relationships(
    frame: pd.DataFrame, bins: BinningSpec
) -> dict[str, ConditionalTable]:
    tables: dict[str, ConditionalTable] = {}
    for relationship_id in RELATIONSHIP_IDS:
        x, y = pair_labels(frame, relationship_id, bins)
        tables[relationship_id] = _fit_conditional(relationship_id, x, y)
    return tables


def score_relationship_series(
    frame: pd.DataFrame,
    relationship_id: str,
    table: ConditionalTable,
    bins: BinningSpec,
) -> np.ndarray:
    x, y = pair_labels(frame, relationship_id, bins)
    lookup: dict[tuple[str, str], float] = {}
    for xv, ymap in table.n_xy.items():
        known_y = set(ymap) | set(table.y_levels)
        for yv in known_y:
            lookup[(xv, yv)] = table.standardized_score(xv, yv)
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    scores = np.empty(len(x_arr), dtype="float64")
    for i, (xv, yv) in enumerate(zip(x_arr, y_arr, strict=True)):
        cached = lookup.get((str(xv), str(yv)))
        scores[i] = (
            cached if cached is not None else table.standardized_score(str(xv), str(yv))
        )
    return scores

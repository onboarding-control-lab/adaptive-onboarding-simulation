"""Gower distance for A2 surrogate ranking (attacker-visible fields only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GowerFieldSpec:
    """One surrogate field with a finite numeric range or categorical flag."""

    name: str
    kind: str  # "numeric" | "categorical"
    low: float | None = None
    high: float | None = None


def build_gower_field_specs(
    *,
    field_names: Sequence[str],
    data_types: Mapping[str, str],
    bounds: Mapping[str, tuple[float | None, float | None]],
    profiles: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
) -> tuple[GowerFieldSpec, ...]:
    """Build Gower specs from governance types and observed ranges."""
    specs: list[GowerFieldSpec] = []
    for name in field_names:
        dtype = str(data_types.get(name, "categorical"))
        if dtype in {"float", "integer", "binary"}:
            low, high = bounds.get(name, (None, None))
            values: list[float] = []
            if name in anchor:
                try:
                    values.append(float(anchor[name]))
                except (TypeError, ValueError):
                    pass
            for profile in profiles:
                if name not in profile:
                    continue
                try:
                    values.append(float(profile[name]))
                except (TypeError, ValueError):
                    continue
            if values:
                obs_low = min(values)
                obs_high = max(values)
                low = obs_low if low is None else min(float(low), obs_low)
                high = obs_high if high is None else max(float(high), obs_high)
            if low is None or high is None or float(high) <= float(low):
                specs.append(GowerFieldSpec(name=name, kind="categorical"))
            else:
                specs.append(
                    GowerFieldSpec(
                        name=name, kind="numeric", low=float(low), high=float(high)
                    )
                )
        else:
            specs.append(GowerFieldSpec(name=name, kind="categorical"))
    return tuple(specs)


def gower_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    specs: Sequence[GowerFieldSpec],
) -> float:
    """Standard Gower distance in ``[0, 1]`` over the supplied specs."""
    if not specs:
        return 0.0
    total = 0.0
    weight = 0.0
    for spec in specs:
        if spec.name not in left or spec.name not in right:
            continue
        a = left[spec.name]
        b = right[spec.name]
        if spec.kind == "numeric":
            assert spec.low is not None and spec.high is not None
            try:
                fa = float(a)
                fb = float(b)
            except (TypeError, ValueError):
                continue
            span = float(spec.high) - float(spec.low)
            if span <= 0:
                continue
            total += min(1.0, abs(fa - fb) / span)
            weight += 1.0
        else:
            total += 0.0 if _equal(a, b) else 1.0
            weight += 1.0
    if weight <= 0:
        return 0.0
    return total / weight


def mean_gower_to_profiles(
    candidate: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    specs: Sequence[GowerFieldSpec],
) -> float:
    """Mean Gower distance from candidate to each reference profile."""
    if not profiles:
        return 1.0
    distances = [gower_distance(candidate, profile, specs) for profile in profiles]
    return sum(distances) / len(distances)


def min_gower_to_set(
    candidate: Mapping[str, Any],
    others: Sequence[Mapping[str, Any]],
    specs: Sequence[GowerFieldSpec],
) -> float:
    if not others:
        return 1.0
    return min(gower_distance(candidate, other, specs) for other in others)


def _equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        import pandas as pd

        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


__all__ = [
    "GowerFieldSpec",
    "build_gower_field_specs",
    "gower_distance",
    "mean_gower_to_profiles",
    "min_gower_to_set",
]

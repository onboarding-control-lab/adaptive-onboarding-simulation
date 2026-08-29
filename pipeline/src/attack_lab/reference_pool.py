"""Synthetic identity reference-pool infrastructure.

The reference pool is a bounded, reproducible attacker-information budget: a
synthetic proxy for limited identity fragments.  It is not an attack result,
not a training leak of labels/scores, and not owned by any single attacker.

Field roles under attack-governance-v2.0.0:

- ``context_fields`` (20): background identity fragments shown in profiles;
- ``action_fields`` (18): only these may enter candidate modification;
- ``read_only_context_fields`` (2): context-only; never sampled as actions.

``eligible_fields`` is retained as a deprecated alias of ``context_fields`` and
must not be read as implying that all twenty fields are mutable.

All attackers (A0–A3) must obtain pools from :class:`ReferencePoolProvider`
so that the same ``(anchor_id, seed)`` yields an identical pool.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from attack_lab.types import to_jsonable
from baf_data.config import FROZEN_CONFIG, DataLayerConfig
from baf_data.protocol_access import load_dataset_for_protocol

DEFAULT_REFERENCE_POOL_CONFIG = (
    Path(__file__).resolve().parents[3] / "config" / "reference_pool_config.json"
)
DEFAULT_RAW_PATH = Path(os.getenv("BAF_BASE_CSV", "Base.csv"))

_FORBIDDEN_SOURCE_MONTHS = frozenset({6, 7})
_HARD_EXCLUSIONS = frozenset(
    {
        "fraud_bool",
        "month",
        "y_score",
        "risk_score",
        "threshold",
        "score",
    }
)
_DEFAULT_READ_ONLY = ("bank_months_count", "has_other_cards")


class ReferencePoolError(RuntimeError):
    """Raised when a reference pool cannot be constructed safely."""


@dataclass(frozen=True)
class ReferencePoolConfig:
    """Experiment-level reference-pool configuration (not attacker-local)."""

    K: int
    seed: int
    context_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    read_only_context_fields: tuple[str, ...]
    excluded_fields: tuple[str, ...]
    label: str = "synthetic_identity_reference_pool_config"
    source_path: str | None = None

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ReferencePoolError("K must be >= 1.")
        for label, values in (
            ("context_fields", self.context_fields),
            ("action_fields", self.action_fields),
            ("read_only_context_fields", self.read_only_context_fields),
        ):
            if len(values) != len(set(values)):
                raise ReferencePoolError(f"{label} must be unique.")
        context = set(self.context_fields)
        action = set(self.action_fields)
        read_only = set(self.read_only_context_fields)
        if action & read_only:
            raise ReferencePoolError(
                "action_fields and read_only_context_fields must be disjoint; "
                f"overlap={sorted(action & read_only)}."
            )
        if action | read_only != context:
            raise ReferencePoolError(
                "context_fields must equal action_fields ∪ read_only_context_fields."
            )
        overlap = context.intersection(self.excluded_fields)
        if overlap:
            raise ReferencePoolError(
                f"Fields listed as both context and excluded: {sorted(overlap)}."
            )
        banned = sorted(context.intersection(_HARD_EXCLUSIONS))
        if banned:
            raise ReferencePoolError(
                f"Hard-excluded fields cannot be reference context: {banned}."
            )
        if len(self.context_fields) != 20:
            raise ReferencePoolError(
                f"Expected 20 context_fields; got {len(self.context_fields)}."
            )
        if len(self.action_fields) != 18:
            raise ReferencePoolError(
                f"Expected 18 action_fields; got {len(self.action_fields)}."
            )
        if set(self.read_only_context_fields) != set(_DEFAULT_READ_ONLY):
            raise ReferencePoolError(
                "read_only_context_fields must be exactly "
                f"{list(_DEFAULT_READ_ONLY)}; got {list(self.read_only_context_fields)}."
            )

    @property
    def eligible_fields(self) -> tuple[str, ...]:
        """Deprecated alias of ``context_fields`` (context-only, not mutability)."""
        return self.context_fields

    @classmethod
    def load(cls, path: Path | None = None) -> "ReferencePoolConfig":
        config_path = path or DEFAULT_REFERENCE_POOL_CONFIG
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if "context_fields" in payload:
            context_fields = tuple(payload["context_fields"])
        elif "eligible_fields" in payload:
            # Legacy configs: eligible_fields meant the 20-field context set.
            context_fields = tuple(payload["eligible_fields"])
        else:
            raise ReferencePoolError(
                "Reference pool config requires context_fields "
                "(or legacy eligible_fields)."
            )
        read_only = tuple(
            payload.get("read_only_context_fields", list(_DEFAULT_READ_ONLY))
        )
        if "action_fields" in payload:
            action_fields = tuple(payload["action_fields"])
        else:
            action_fields = tuple(
                name for name in context_fields if name not in set(read_only)
            )
        return cls(
            K=int(payload["K"]),
            seed=int(payload["seed"]),
            context_fields=context_fields,
            action_fields=action_fields,
            read_only_context_fields=read_only,
            excluded_fields=tuple(payload["excluded_fields"]),
            label=str(payload.get("label", "synthetic_identity_reference_pool_config")),
            source_path=str(config_path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "K": self.K,
            "seed": self.seed,
            "context_fields": list(self.context_fields),
            "action_fields": list(self.action_fields),
            "read_only_context_fields": list(self.read_only_context_fields),
            # Deprecated mirror for archived readers.
            "eligible_fields": list(self.context_fields),
            "excluded_fields": list(self.excluded_fields),
            "source_path": self.source_path,
        }

    def validate_against_governance(self, action_fields: Sequence[str]) -> None:
        """Fail closed if configured action_fields disagree with governance."""
        expected = tuple(action_fields)
        if set(expected) != set(self.action_fields) or len(expected) != len(
            self.action_fields
        ):
            raise ReferencePoolError(
                "reference-pool action_fields disagree with compiled governance "
                f"action_fields; pool={sorted(self.action_fields)}; "
                f"governance={sorted(expected)}."
            )


@dataclass(frozen=True)
class ReferenceProfile:
    """One synthetic identity fragment bundle exposed to attackers."""

    profile_id: str
    fields: Mapping[str, Any]
    generation_seed: int

    def attacker_view(self) -> dict[str, Any]:
        """Attacker-visible content only (no source-row provenance)."""
        return {
            "profile_id": self.profile_id,
            "fields": dict(self.fields),
            "generation_seed": self.generation_seed,
        }


@dataclass(frozen=True)
class ReferencePool:
    """Fixed-K reference identity pool for one anchor and seed."""

    anchor_id: str
    K: int
    generation_seed: int
    pool_fingerprint: str
    context_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    read_only_context_fields: tuple[str, ...]
    profiles: tuple[ReferenceProfile, ...]
    #: Research-log only; never returned by :meth:`attacker_view`.
    source_row_ids: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.profiles) != self.K:
            raise ReferencePoolError("Profile count does not match K.")
        if len(self.source_row_ids) != self.K:
            raise ReferencePoolError("source_row_ids length does not match K.")

    @property
    def eligible_fields(self) -> tuple[str, ...]:
        """Deprecated alias of ``context_fields``."""
        return self.context_fields

    def attacker_view(self) -> dict[str, Any]:
        """Public view shared by A0–A3; omits source-row identifiers."""
        return {
            "anchor_id": self.anchor_id,
            "K": self.K,
            "generation_seed": self.generation_seed,
            "pool_fingerprint": self.pool_fingerprint,
            "context_fields": list(self.context_fields),
            "action_fields": list(self.action_fields),
            "read_only_context_fields": list(self.read_only_context_fields),
            # Deprecated mirror: context-only, not a mutability claim.
            "eligible_fields": list(self.context_fields),
            "profiles": [profile.attacker_view() for profile in self.profiles],
        }

    def research_log(self) -> dict[str, Any]:
        """Researcher-only record including source-row provenance."""
        payload = self.attacker_view()
        payload["source_row_ids"] = list(self.source_row_ids)
        payload["profiles_with_source"] = [
            {
                **profile.attacker_view(),
                "source_row_id": source_row_id,
            }
            for profile, source_row_id in zip(
                self.profiles, self.source_row_ids, strict=True
            )
        ]
        return payload


@dataclass
class ReferencePoolProvider:
    """Build reproducible reference pools from months 0–5 only.

    Month 6 and month 7 rows are never retained for sampling.  Attackers must
    not construct their own pools; they call :meth:`get_pool`.
    """

    config: ReferencePoolConfig
    _frame: pd.DataFrame
    _source_split: str = "train_months_0_5"

    @classmethod
    def from_config(
        cls,
        config: ReferencePoolConfig | None = None,
        *,
        raw_path: Path | None = None,
        data_config: DataLayerConfig = FROZEN_CONFIG,
        training_frame: pd.DataFrame | None = None,
    ) -> "ReferencePoolProvider":
        """Construct a provider from config and an explicit train-only frame.

        Prefer ``training_frame`` in unit tests.  When loading from disk, only
        the months-0–5 train split is retained; month 6/7 handles are dropped.
        """
        cfg = config or ReferencePoolConfig.load()
        if training_frame is not None:
            frame = _prepare_reference_frame(training_frame, cfg, data_config)
            return cls(config=cfg, _frame=frame)

        path = raw_path or DEFAULT_RAW_PATH
        loaded = load_dataset_for_protocol(
            path,
            phase="development",
            allowed_months=(0, 1, 2, 3, 4, 5),
            config=data_config,
        )
        train = loaded.frame.copy()
        frame = _prepare_reference_frame(train, cfg, data_config)
        return cls(config=cfg, _frame=frame)

    @property
    def K(self) -> int:
        return self.config.K

    @property
    def context_fields(self) -> tuple[str, ...]:
        return self.config.context_fields

    @property
    def action_fields(self) -> tuple[str, ...]:
        return self.config.action_fields

    @property
    def read_only_context_fields(self) -> tuple[str, ...]:
        return self.config.read_only_context_fields

    @property
    def eligible_fields(self) -> tuple[str, ...]:
        """Deprecated alias of ``context_fields``."""
        return self.config.context_fields

    def get_pool(
        self,
        anchor_id: str,
        *,
        seed: int | None = None,
        attacker_ids: Sequence[str] | None = None,
    ) -> ReferencePool:
        """Return the shared pool for ``anchor_id`` under a fixed seed.

        ``attacker_ids`` is accepted only to document that A0–A3 must call the
        same provider; it does not alter sampling.
        """
        _ = attacker_ids  # intentional no-op: pool is attacker-independent
        generation_seed = int(self.config.seed if seed is None else seed)
        rng_seed = _derive_rng_seed(anchor_id, generation_seed)
        rng = np.random.default_rng(rng_seed)

        if len(self._frame) < self.config.K:
            raise ReferencePoolError(
                f"Reference source has {len(self._frame)} rows; need at least K={self.config.K}."
            )

        positions = rng.choice(len(self._frame), size=self.config.K, replace=False)
        positions = np.sort(positions)
        chosen = self._frame.iloc[positions]
        source_row_ids = tuple(int(idx) for idx in chosen.index.to_list())

        profiles: list[ReferenceProfile] = []
        for offset, (row_id, row) in enumerate(chosen.iterrows()):
            fields = {
                name: _native(row[name]) for name in self.config.context_fields
            }
            # Hard safety: never leak excluded / label / split columns.
            for banned in self.config.excluded_fields:
                fields.pop(banned, None)
            fields.pop("fraud_bool", None)
            fields.pop("month", None)
            profiles.append(
                ReferenceProfile(
                    profile_id=f"ref_{offset:02d}",
                    fields=fields,
                    generation_seed=generation_seed,
                )
            )

        fingerprint = _pool_fingerprint(
            anchor_id=str(anchor_id),
            generation_seed=generation_seed,
            K=self.config.K,
            context_fields=self.config.context_fields,
            action_fields=self.config.action_fields,
            read_only_context_fields=self.config.read_only_context_fields,
            source_row_ids=source_row_ids,
            profiles=profiles,
        )
        return ReferencePool(
            anchor_id=str(anchor_id),
            K=self.config.K,
            generation_seed=generation_seed,
            pool_fingerprint=fingerprint,
            context_fields=self.config.context_fields,
            action_fields=self.config.action_fields,
            read_only_context_fields=self.config.read_only_context_fields,
            profiles=tuple(profiles),
            source_row_ids=source_row_ids,
        )


def _prepare_reference_frame(
    frame: pd.DataFrame,
    config: ReferencePoolConfig,
    data_config: DataLayerConfig,
) -> pd.DataFrame:
    if "month" not in frame.columns:
        raise ReferencePoolError(
            "Reference source frame must include month for split isolation."
        )
    months = set(int(value) for value in frame["month"].unique())
    if months & _FORBIDDEN_SOURCE_MONTHS:
        raise ReferencePoolError(
            "Reference pool source includes forbidden months "
            f"{sorted(months & _FORBIDDEN_SOURCE_MONTHS)}; only months 0-5 are allowed."
        )
    if not months.issubset(set(range(6))):
        raise ReferencePoolError(
            f"Reference pool source months {sorted(months)} are not within 0-5."
        )

    missing = [name for name in config.context_fields if name not in frame.columns]
    if missing:
        raise ReferencePoolError(
            f"Reference source missing context fields: {missing}."
        )

    # Keep only context columns; drop labels/split/hidden aggregates entirely.
    out = frame.loc[:, list(config.context_fields)].copy()
    # Preserve stable source-row identifiers from the original index.
    out.index = frame.index
    if out.index.has_duplicates:
        raise ReferencePoolError("Reference source index must be unique.")
    # Ensure excluded columns never remain attached.
    for banned in set(config.excluded_fields) | _HARD_EXCLUSIONS:
        if banned in out.columns:
            raise ReferencePoolError(
                f"Excluded field {banned!r} leaked into the reference frame."
            )
    _ = data_config  # reserved for future schema checks against FROZEN_CONFIG
    return out


def _derive_rng_seed(anchor_id: str, generation_seed: int) -> int:
    digest = hashlib.sha256(
        f"{generation_seed}:{anchor_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _pool_fingerprint(
    *,
    anchor_id: str,
    generation_seed: int,
    K: int,
    context_fields: Sequence[str],
    action_fields: Sequence[str],
    read_only_context_fields: Sequence[str],
    source_row_ids: Sequence[int],
    profiles: Sequence[ReferenceProfile],
) -> str:
    payload = {
        "anchor_id": anchor_id,
        "generation_seed": generation_seed,
        "K": K,
        "context_fields": list(context_fields),
        "action_fields": list(action_fields),
        "read_only_context_fields": list(read_only_context_fields),
        "source_row_ids": list(source_row_ids),
        "profiles": [to_jsonable(asdict(profile)) for profile in profiles],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _native(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


__all__ = [
    "DEFAULT_REFERENCE_POOL_CONFIG",
    "ReferencePool",
    "ReferencePoolConfig",
    "ReferencePoolError",
    "ReferencePoolProvider",
    "ReferenceProfile",
]

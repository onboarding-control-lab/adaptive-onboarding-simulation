"""Machine-executable attacker feature governance.

The CSV is the feature-specific policy source.  This module deliberately
contains no per-feature permission or proxy special cases: it validates the
schema, compiles train-only support, and exposes one policy object used by every
attacker through :class:`AttackEnvironment`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class GovernanceError(RuntimeError):
    """Raised when governance cannot be compiled safely."""


_MUTABILITIES = {
    "allowed",
    "allowed_if_episode_locked",
    "forbidden",
    "not_applicable",
}
_VISIBILITIES = {"yes", "no", "proxy_only"}
_ACTION_MODES = {"raw_value", "proxy_action", "none"}
_DATA_TYPES = {"binary", "integer", "float", "categorical"}
_DOMAIN_MODES = {
    "bounded_train_support",
    "categorical_train_support",
    "fixed_set",
    "proxy_action",
    "preserve_anchor",
    "excluded",
}
_SAMPLING_KINDS = {"discrete_support", "continuous_bounds"}
#: Unique-value threshold for treating a numeric train support as discrete.
#: This is a compilation rule, not a per-feature hard-coded list.
_LOW_CARDINALITY_UNIQUE_MAX = 64
_FROZEN_GOVERNANCE_STATUSES = {
    "frozen_attacker_rule",
    "frozen_project_rule",
}
_REQUIRED_COLUMNS = {
    "feature",
    "attacker_visible",
    "agent_mutability",
    "governance_status",
    "policy_version",
    "data_type",
    "domain_mode",
    "lower_bound",
    "upper_bound",
    "allowed_values_json",
    "sentinel_spec_json",
    "sentinel_policy",
    "hard_constraints_json",
    "agent_action_mode",
    "proxy_action_key",
    "proxy_mapping_json",
    "support_split",
}


@dataclass(frozen=True)
class GovernanceRule:
    """One validated row from ``attacker_feature_governance.csv``.

    The legacy preprocessing ``status`` column is intentionally not represented
    here, so it cannot influence attacker visibility or permission.
    """

    feature: str
    attacker_visible: str
    agent_mutability: str
    governance_status: str
    policy_version: str
    data_type: str
    domain_mode: str
    lower_bound: float | None
    upper_bound: float | None
    allowed_values: tuple[Any, ...]
    sentinel_spec: Mapping[str, Any]
    sentinel_policy: str
    hard_constraints: tuple[Mapping[str, Any], ...]
    agent_action_mode: str
    proxy_action_key: str | None
    proxy_mapping: Mapping[str, Any]
    support_split: str

    @property
    def is_episode_locked(self) -> bool:
        return self.agent_mutability == "allowed_if_episode_locked"

    @property
    def is_mutable(self) -> bool:
        return self.agent_mutability in {"allowed", "allowed_if_episode_locked"}


@dataclass(frozen=True)
class CompiledFieldPolicy:
    feature: str
    attacker_visible: str
    agent_mutability: str
    governance_status: str
    data_type: str
    domain_mode: str
    lower_bound: float | None
    upper_bound: float | None
    allowed_values: tuple[Any, ...]
    observed_support: tuple[Any, ...]
    sampling_kind: str
    sentinel_spec: Mapping[str, Any]
    sentinel_policy: str
    hard_constraints: tuple[Mapping[str, Any], ...]
    agent_action_mode: str
    proxy_action_key: str | None
    resolved_proxy_actions: Mapping[str, Any]
    proxy_seed: int | None

    @property
    def is_episode_locked(self) -> bool:
        return self.agent_mutability == "allowed_if_episode_locked"

    @property
    def is_mutable(self) -> bool:
        return self.agent_mutability in {"allowed", "allowed_if_episode_locked"}


@dataclass(frozen=True)
class CompiledGovernancePolicy:
    policy_version: str
    source_path: str
    source_sha256: str
    support_split: str
    support_months: tuple[int, ...]
    support_rows: int
    support_fingerprint: str
    fields: Mapping[str, CompiledFieldPolicy]
    policy_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "support_split": self.support_split,
            "support_months": list(self.support_months),
            "support_rows": self.support_rows,
            "support_fingerprint": self.support_fingerprint,
            "fields": {
                name: _jsonable(asdict(rule))
                for name, rule in self.fields.items()
            },
            "policy_fingerprint": self.policy_fingerprint,
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CompiledGovernancePolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = {
            name: CompiledFieldPolicy(
                feature=item["feature"],
                attacker_visible=item["attacker_visible"],
                agent_mutability=item["agent_mutability"],
                governance_status=item["governance_status"],
                data_type=item["data_type"],
                domain_mode=item["domain_mode"],
                lower_bound=item["lower_bound"],
                upper_bound=item["upper_bound"],
                allowed_values=tuple(item["allowed_values"]),
                observed_support=tuple(
                    item.get(
                        "observed_support",
                        item.get("allowed_values", ()),
                    )
                ),
                sampling_kind=str(
                    item.get(
                        "sampling_kind",
                        (
                            "discrete_support"
                            if item.get("allowed_values")
                            or item.get("data_type")
                            in {"categorical", "binary"}
                            else "continuous_bounds"
                        ),
                    )
                ),
                sentinel_spec=dict(item["sentinel_spec"]),
                sentinel_policy=item["sentinel_policy"],
                hard_constraints=tuple(item["hard_constraints"]),
                agent_action_mode=item["agent_action_mode"],
                proxy_action_key=item["proxy_action_key"],
                resolved_proxy_actions=dict(item["resolved_proxy_actions"]),
                proxy_seed=item["proxy_seed"],
            )
            for name, item in payload["fields"].items()
        }
        for rule in fields.values():
            if rule.sampling_kind not in _SAMPLING_KINDS:
                raise GovernanceError(
                    f"{rule.feature}: invalid sampling_kind {rule.sampling_kind!r}."
                )
        policy = cls(
            policy_version=payload["policy_version"],
            source_path=payload["source_path"],
            source_sha256=payload["source_sha256"],
            support_split=payload["support_split"],
            support_months=tuple(payload["support_months"]),
            support_rows=int(payload["support_rows"]),
            support_fingerprint=payload["support_fingerprint"],
            fields=fields,
            policy_fingerprint=payload["policy_fingerprint"],
        )
        if policy.support_months != tuple(range(6)):
            raise GovernanceError(
                "Compiled governance was not built from exactly months 0-5."
            )
        expected = hashlib.sha256(
            json.dumps(
                {
                    "policy_version": policy.policy_version,
                    "source_sha256": policy.source_sha256,
                    "support_split": policy.support_split,
                    "support_months": policy.support_months,
                    "support_rows": policy.support_rows,
                    "support_fingerprint": policy.support_fingerprint,
                    "fields": {
                        name: _jsonable(asdict(rule))
                        for name, rule in policy.fields.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if expected != policy.policy_fingerprint:
            raise GovernanceError("Compiled governance fingerprint mismatch.")
        return policy

    @property
    def raw_action_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rule in self.fields.items()
            if rule.is_mutable and rule.agent_action_mode == "raw_value"
        )

    @property
    def proxy_action_keys(self) -> tuple[str, ...]:
        return tuple(
            rule.proxy_action_key
            for rule in self.fields.values()
            if rule.is_mutable
            and rule.agent_action_mode == "proxy_action"
            and rule.proxy_action_key is not None
        )

    @property
    def available_action_keys(self) -> tuple[str, ...]:
        return self.raw_action_fields + self.proxy_action_keys

    @property
    def locked_fields(self) -> tuple[str, ...]:
        return tuple(
            name for name, rule in self.fields.items() if rule.is_episode_locked
        )

    @property
    def action_fields(self) -> tuple[str, ...]:
        """Mutable action features (allowed + allowed_if_episode_locked)."""
        return tuple(name for name, rule in self.fields.items() if rule.is_mutable)

    @property
    def per_attempt_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rule in self.fields.items()
            if rule.agent_mutability == "allowed"
        )

    @property
    def episode_static_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rule in self.fields.items()
            if rule.agent_mutability == "allowed_if_episode_locked"
        )

    @property
    def forbidden_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rule in self.fields.items()
            if rule.agent_mutability == "forbidden"
        )

    @property
    def not_applicable_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, rule in self.fields.items()
            if rule.agent_mutability == "not_applicable"
        )

    def field_for_action(self, action_key: str) -> CompiledFieldPolicy | None:
        direct = self.fields.get(action_key)
        if (
            direct is not None
            and direct.is_mutable
            and direct.agent_action_mode == "raw_value"
        ):
            return direct
        for rule in self.fields.values():
            if (
                rule.is_mutable
                and rule.agent_action_mode == "proxy_action"
                and rule.proxy_action_key == action_key
            ):
                return rule
        return None

    def visible_fields(self, features: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: features[name]
            for name, rule in self.fields.items()
            if rule.attacker_visible == "yes" and name in features
        }

    def proxy_action_catalogue(
        self, *, include_locked: bool = True
    ) -> dict[str, tuple[str, ...]]:
        catalogue: dict[str, tuple[str, ...]] = {}
        for rule in self.fields.values():
            if (
                rule.agent_action_mode == "proxy_action"
                and rule.proxy_action_key is not None
                and rule.is_mutable
                and (include_locked or not rule.is_episode_locked)
            ):
                catalogue[rule.proxy_action_key] = tuple(
                    rule.resolved_proxy_actions.keys()
                )
        return catalogue

    def manifest_payload(self) -> dict[str, Any]:
        proxy_mappings = {}
        for rule in self.fields.values():
            if rule.agent_action_mode == "proxy_action":
                proxy_mappings[rule.proxy_action_key or rule.feature] = {
                    "underlying_feature": rule.feature,
                    "seed": rule.proxy_seed,
                    "actions": dict(rule.resolved_proxy_actions),
                    "support_split": self.support_split,
                }
        return {
            "governance": {
                "policy_version": self.policy_version,
                "policy_fingerprint": self.policy_fingerprint,
                "source_path": self.source_path,
                "source_sha256": self.source_sha256,
                "support_split": self.support_split,
                "support_months": list(self.support_months),
                "support_rows": self.support_rows,
                "support_fingerprint": self.support_fingerprint,
                "permission_columns": [
                    "governance_status",
                    "agent_mutability",
                ],
                "legacy_preprocessing_status_used": False,
                "proxy_mappings": proxy_mappings,
            }
        }


class GovernanceLoader:
    """Load and validate governance rows without consulting preprocessing status."""

    @classmethod
    def load_csv(cls, path: Path) -> tuple[GovernanceRule, ...]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise GovernanceError(
                    f"Governance CSV missing machine columns: {sorted(missing)}."
                )
            raw_rows = list(reader)
        if not raw_rows:
            raise GovernanceError("Governance CSV contains no feature rows.")

        seen: set[str] = set()
        rules: list[GovernanceRule] = []
        versions: set[str] = set()
        for row_number, row in enumerate(raw_rows, start=2):
            feature = row["feature"].strip()
            if not feature or feature in seen:
                raise GovernanceError(
                    f"Row {row_number} has blank or duplicate feature {feature!r}."
                )
            seen.add(feature)
            mutability = row["agent_mutability"].strip()
            visibility = row["attacker_visible"].strip()
            governance_status = row["governance_status"].strip()
            action_mode = row["agent_action_mode"].strip()
            data_type = row["data_type"].strip()
            domain_mode = row["domain_mode"].strip()
            if mutability not in _MUTABILITIES:
                raise GovernanceError(f"{feature}: invalid agent_mutability.")
            if visibility not in _VISIBILITIES:
                raise GovernanceError(f"{feature}: invalid attacker_visible.")
            if governance_status not in _FROZEN_GOVERNANCE_STATUSES:
                raise GovernanceError(
                    f"{feature}: governance_status is not frozen; attack use is blocked."
                )
            if action_mode not in _ACTION_MODES:
                raise GovernanceError(f"{feature}: invalid agent_action_mode.")
            if data_type not in _DATA_TYPES:
                raise GovernanceError(f"{feature}: invalid data_type.")
            if domain_mode not in _DOMAIN_MODES:
                raise GovernanceError(f"{feature}: invalid domain_mode.")

            allowed_values = _json_tuple(row["allowed_values_json"], feature)
            sentinel_spec = _json_object(row["sentinel_spec_json"], feature)
            hard_constraints = _json_tuple(
                row["hard_constraints_json"], feature, require_objects=True
            )
            proxy_mapping = _json_object(row["proxy_mapping_json"], feature)
            proxy_key = row["proxy_action_key"].strip() or None
            lower = _optional_float(row["lower_bound"], feature)
            upper = _optional_float(row["upper_bound"], feature)
            version = row["policy_version"].strip()
            versions.add(version)

            if mutability in {"forbidden", "not_applicable"} and action_mode != "none":
                raise GovernanceError(
                    f"{feature}: non-mutable field must have agent_action_mode=none."
                )
            if action_mode == "proxy_action":
                if visibility != "proxy_only" or not proxy_key or not proxy_mapping:
                    raise GovernanceError(
                        f"{feature}: proxy action requires proxy_only visibility, "
                        "an action key, and a mapping specification."
                    )
            elif proxy_key or proxy_mapping:
                raise GovernanceError(
                    f"{feature}: non-proxy field contains proxy configuration."
                )
            if mutability == "allowed_if_episode_locked" and not any(
                item.get("type") == "episode_lock_on_first_submission"
                for item in hard_constraints
            ):
                raise GovernanceError(
                    f"{feature}: locked mutability lacks executable lock constraint."
                )
            if mutability == "allowed" and any(
                item.get("type") == "episode_lock_on_first_submission"
                for item in hard_constraints
            ):
                raise GovernanceError(
                    f"{feature}: per-attempt allowed field must not carry "
                    "episode_lock_on_first_submission."
                )

            rules.append(
                GovernanceRule(
                    feature=feature,
                    attacker_visible=visibility,
                    agent_mutability=mutability,
                    governance_status=governance_status,
                    policy_version=version,
                    data_type=data_type,
                    domain_mode=domain_mode,
                    lower_bound=lower,
                    upper_bound=upper,
                    allowed_values=allowed_values,
                    sentinel_spec=sentinel_spec,
                    sentinel_policy=row["sentinel_policy"].strip(),
                    hard_constraints=hard_constraints,
                    agent_action_mode=action_mode,
                    proxy_action_key=proxy_key,
                    proxy_mapping=proxy_mapping,
                    support_split=row["support_split"].strip(),
                )
            )

        if len(versions) != 1 or "" in versions:
            raise GovernanceError("All governance rows must share one policy_version.")
        action_keys = [
            rule.proxy_action_key for rule in rules if rule.proxy_action_key is not None
        ]
        if len(action_keys) != len(set(action_keys)):
            raise GovernanceError("Proxy action keys must be unique.")
        if set(action_keys) & seen:
            raise GovernanceError("Proxy action keys must not collide with field names.")
        return tuple(rules)


class PolicyCompiler:
    """Compile governance against an explicitly train-only support frame."""

    @classmethod
    def compile(
        cls,
        rules: Iterable[GovernanceRule],
        training_frame: pd.DataFrame,
        *,
        source_path: Path,
    ) -> CompiledGovernancePolicy:
        rules = tuple(rules)
        if not rules:
            raise GovernanceError("Cannot compile an empty governance policy.")
        _assert_train_months_only(training_frame)
        support_months = tuple(
            sorted(int(value) for value in training_frame["month"].unique())
        )
        support_split_values = {rule.support_split for rule in rules}
        if support_split_values != {"train_months_0_5"}:
            raise GovernanceError(
                "All governance rules must use support_split=train_months_0_5."
            )

        compiled_fields: dict[str, CompiledFieldPolicy] = {}
        for rule in rules:
            if rule.feature not in training_frame.columns:
                raise GovernanceError(
                    f"Training support is missing governed feature {rule.feature!r}."
                )
            support = training_frame[rule.feature]
            non_sentinel = _without_sentinel(support, rule.sentinel_spec)
            allowed_values = cls._compile_allowed_values(rule, non_sentinel)
            lower, upper = cls._compile_bounds(rule, non_sentinel)
            observed_support, sampling_kind = cls._compile_observed_support(
                rule, non_sentinel, allowed_values
            )
            constraints = cls._compile_constraints(
                rule, training_frame, non_sentinel
            )
            resolved_proxy, proxy_seed = cls._compile_proxy(
                rule, non_sentinel, lower, upper
            )
            compiled_fields[rule.feature] = CompiledFieldPolicy(
                feature=rule.feature,
                attacker_visible=rule.attacker_visible,
                agent_mutability=rule.agent_mutability,
                governance_status=rule.governance_status,
                data_type=rule.data_type,
                domain_mode=rule.domain_mode,
                lower_bound=lower,
                upper_bound=upper,
                allowed_values=allowed_values,
                observed_support=observed_support,
                sampling_kind=sampling_kind,
                sentinel_spec=dict(rule.sentinel_spec),
                sentinel_policy=rule.sentinel_policy,
                hard_constraints=constraints,
                agent_action_mode=rule.agent_action_mode,
                proxy_action_key=rule.proxy_action_key,
                resolved_proxy_actions=resolved_proxy,
                proxy_seed=proxy_seed,
            )

        source_sha = _sha256_file(source_path)
        support_fingerprint = _frame_fingerprint(training_frame, tuple(compiled_fields))
        base_payload = {
            "policy_version": rules[0].policy_version,
            "source_sha256": source_sha,
            "support_split": "train_months_0_5",
            "support_months": support_months,
            "support_rows": len(training_frame),
            "support_fingerprint": support_fingerprint,
            "fields": {
                name: _jsonable(asdict(rule))
                for name, rule in compiled_fields.items()
            },
        }
        policy_fingerprint = hashlib.sha256(
            json.dumps(base_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CompiledGovernancePolicy(
            policy_version=rules[0].policy_version,
            source_path=str(source_path),
            source_sha256=source_sha,
            support_split="train_months_0_5",
            support_months=support_months,
            support_rows=len(training_frame),
            support_fingerprint=support_fingerprint,
            fields=compiled_fields,
            policy_fingerprint=policy_fingerprint,
        )

    @staticmethod
    def _compile_allowed_values(
        rule: GovernanceRule, support: pd.Series
    ) -> tuple[Any, ...]:
        if rule.domain_mode in {"categorical_train_support", "fixed_set"}:
            observed = tuple(_sorted_unique(support.dropna().tolist()))
            if rule.allowed_values:
                allowed = tuple(
                    value
                    for value in rule.allowed_values
                    if _contains_value(observed, value)
                )
                if len(allowed) != len(rule.allowed_values):
                    raise GovernanceError(
                        f"{rule.feature}: configured allowed value absent from "
                        "months 0-5 support."
                    )
                return allowed
            return observed
        if rule.data_type == "binary":
            observed = tuple(
                _coerce_compiled_value(value, "binary")
                for value in _sorted_unique(support.dropna().tolist())
            )
            if rule.allowed_values:
                return tuple(
                    value
                    for value in rule.allowed_values
                    if _contains_value(observed, value)
                )
            return observed
        return tuple(rule.allowed_values)

    @staticmethod
    def _compile_observed_support(
        rule: GovernanceRule,
        support: pd.Series,
        allowed_values: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], str]:
        """Persist months 0-5 support and decide discrete vs continuous sampling.

        The discrete/continuous decision is driven only by data_type, domain_mode
        and observed cardinality — never by hard-coded feature names.
        """
        cleaned = support.dropna()
        if rule.domain_mode in {"categorical_train_support", "fixed_set"}:
            observed = allowed_values or tuple(
                _sorted_unique(cleaned.tolist())
            )
            return observed, "discrete_support"
        if rule.data_type == "binary":
            observed = allowed_values or tuple(
                _coerce_compiled_value(value, "binary")
                for value in _sorted_unique(cleaned.tolist())
            )
            return observed, "discrete_support"
        if rule.data_type in {"integer", "float"}:
            numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
            observed = tuple(
                _coerce_compiled_value(value, rule.data_type)
                for value in _sorted_unique(numeric.tolist())
            )
            if (
                rule.domain_mode in {"bounded_train_support", "proxy_action"}
                and 0 < len(observed) <= _LOW_CARDINALITY_UNIQUE_MAX
            ):
                return observed, "discrete_support"
            # High-cardinality continuous: keep empty discrete support; A0 uses bounds.
            return (), "continuous_bounds"
        return allowed_values, "discrete_support" if allowed_values else "continuous_bounds"

    @staticmethod
    def _compile_bounds(
        rule: GovernanceRule, support: pd.Series
    ) -> tuple[float | None, float | None]:
        if rule.data_type == "categorical" or rule.domain_mode in {
            "excluded",
            "preserve_anchor",
        }:
            return rule.lower_bound, rule.upper_bound
        numeric = pd.to_numeric(support, errors="coerce").dropna()
        if numeric.empty and rule.is_mutable:
            raise GovernanceError(f"{rule.feature}: no usable train support.")
        if rule.domain_mode in {"bounded_train_support", "proxy_action"} and not numeric.empty:
            observed_lower = float(numeric.min())
            observed_upper = float(numeric.max())
            lower = (
                observed_lower
                if rule.lower_bound is None
                else max(rule.lower_bound, observed_lower)
            )
            upper = (
                observed_upper
                if rule.upper_bound is None
                else min(rule.upper_bound, observed_upper)
            )
            if lower > upper:
                raise GovernanceError(
                    f"{rule.feature}: configured domain has no train-supported values."
                )
            return lower, upper
        return rule.lower_bound, rule.upper_bound

    @staticmethod
    def _compile_constraints(
        rule: GovernanceRule,
        training_frame: pd.DataFrame,
        non_sentinel: pd.Series,
    ) -> tuple[Mapping[str, Any], ...]:
        output: list[Mapping[str, Any]] = []
        for item in rule.hard_constraints:
            if item.get("type") != "conditional_train_range":
                output.append(dict(item))
                continue
            conditions = tuple(item.get("condition_fields", ()))
            binning = dict(item.get("condition_binning", {}))
            if not conditions or any(name not in training_frame for name in conditions):
                raise GovernanceError(
                    f"{rule.feature}: invalid conditional constraint fields."
                )
            valid_mask = non_sentinel.index
            work = training_frame.loc[
                valid_mask, [*conditions, rule.feature]
            ].dropna()
            compiled_ranges: list[dict[str, Any]] = []
            grouped_keys: list[str] = []
            for condition in conditions:
                width = float(binning.get(condition, 0))
                key = f"__bin_{condition}"
                if width <= 0:
                    work[key] = work[condition]
                else:
                    work[key] = (
                        np.floor(pd.to_numeric(work[condition]) / width) * width
                    )
                grouped_keys.append(key)
            for group_values, group in work.groupby(grouped_keys, dropna=False):
                if not isinstance(group_values, tuple):
                    group_values = (group_values,)
                compiled_ranges.append(
                    {
                        "conditions": {
                            name: _native(value)
                            for name, value in zip(conditions, group_values)
                        },
                        "min": _native(pd.to_numeric(group[rule.feature]).min()),
                        "max": _native(pd.to_numeric(group[rule.feature]).max()),
                    }
                )
            output.append(
                {
                    **dict(item),
                    "compiled_ranges": compiled_ranges,
                }
            )
        return tuple(output)

    @staticmethod
    def _compile_proxy(
        rule: GovernanceRule,
        support: pd.Series,
        lower: float | None,
        upper: float | None,
    ) -> tuple[Mapping[str, Any], int | None]:
        if rule.agent_action_mode != "proxy_action":
            return {}, None
        spec = dict(rule.proxy_mapping)
        strategy = spec.get("strategy")
        seed = int(spec.get("seed"))
        actions = spec.get("actions")
        if not isinstance(actions, dict) or not actions:
            raise GovernanceError(f"{rule.feature}: proxy actions are missing.")
        numeric = pd.to_numeric(support, errors="coerce").dropna()
        if lower is not None:
            numeric = numeric[numeric >= lower]
        if upper is not None:
            numeric = numeric[numeric <= upper]
        values = np.array(sorted(set(_native(value) for value in numeric)), dtype=float)
        if values.size == 0:
            raise GovernanceError(f"{rule.feature}: proxy has no train support.")
        resolved: dict[str, Any] = {}
        for action_name, action_spec in actions.items():
            if strategy == "fixed_supported_value":
                requested = action_spec.get("value")
                matches = [value for value in values if _values_equal(value, requested)]
                if not matches:
                    raise GovernanceError(
                        f"{rule.feature}: proxy value {requested!r} lacks train support."
                    )
                value = matches[0]
            elif strategy == "train_quantile_nearest":
                quantile = float(action_spec.get("quantile"))
                if not 0 <= quantile <= 1:
                    raise GovernanceError(
                        f"{rule.feature}: proxy quantile must be in [0,1]."
                    )
                target = float(numeric.quantile(quantile))
                distances = np.abs(values - target)
                candidates = values[distances == distances.min()]
                if len(candidates) == 1:
                    value = candidates[0]
                else:
                    digest = hashlib.sha256(
                        f"{seed}:{rule.proxy_action_key}:{action_name}".encode()
                    ).digest()
                    value = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
            else:
                raise GovernanceError(
                    f"{rule.feature}: unsupported proxy strategy {strategy!r}."
                )
            resolved[str(action_name)] = _coerce_compiled_value(value, rule.data_type)
        return resolved, seed


def _assert_train_months_only(frame: pd.DataFrame) -> None:
    if "month" not in frame.columns:
        raise GovernanceError(
            "Policy compilation requires an explicit month column for leakage checks."
        )
    if frame.empty:
        raise GovernanceError("Policy compilation support frame is empty.")
    months = set(pd.to_numeric(frame["month"], errors="raise").astype(int).unique())
    if not months <= set(range(6)):
        raise GovernanceError(
            f"Policy compilation accepts months 0-5 only; received {sorted(months)}."
        )
    if months != set(range(6)):
        raise GovernanceError(
            f"Policy compilation requires all training months 0-5; received {sorted(months)}."
        )


def _without_sentinel(series: pd.Series, spec: Mapping[str, Any]) -> pd.Series:
    if not spec:
        return series.dropna()
    kind = spec.get("kind")
    if kind == "values":
        values = spec.get("values", [])
        return series[~series.isin(values)].dropna()
    if kind == "predicate":
        value = spec.get("value")
        operator = spec.get("operator")
        if operator == "lt":
            return series[~(pd.to_numeric(series, errors="coerce") < value)].dropna()
        raise GovernanceError(f"Unsupported sentinel predicate {operator!r}.")
    raise GovernanceError(f"Unsupported sentinel specification kind {kind!r}.")


def is_sentinel(value: Any, spec: Mapping[str, Any]) -> bool:
    if not spec or pd.isna(value):
        return False
    if spec.get("kind") == "values":
        return any(_values_equal(value, item) for item in spec.get("values", []))
    if spec.get("kind") == "predicate" and spec.get("operator") == "lt":
        return float(value) < float(spec["value"])
    return False


def _json_tuple(
    raw: str, feature: str, *, require_objects: bool = False
) -> tuple[Any, ...]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise GovernanceError(f"{feature}: invalid JSON list.") from exc
    if not isinstance(value, list):
        raise GovernanceError(f"{feature}: expected a JSON list.")
    if require_objects and any(not isinstance(item, dict) for item in value):
        raise GovernanceError(f"{feature}: constraints must be JSON objects.")
    return tuple(value)


def _json_object(raw: str, feature: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise GovernanceError(f"{feature}: invalid JSON object.") from exc
    if not isinstance(value, dict):
        raise GovernanceError(f"{feature}: expected a JSON object.")
    return value


def _optional_float(raw: str, feature: str) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise GovernanceError(f"{feature}: invalid numeric bound {raw!r}.") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    ordered = frame.loc[:, [*columns, "month"]]
    hashes = pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
    return hashlib.sha256(hashes).hexdigest()


def _sorted_unique(values: Iterable[Any]) -> list[Any]:
    native = [_native(value) for value in values]
    return sorted(set(native), key=lambda value: (type(value).__name__, str(value)))


def _contains_value(values: Iterable[Any], target: Any) -> bool:
    return any(_values_equal(value, target) for value in values)


def _values_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def _coerce_compiled_value(value: Any, data_type: str) -> Any:
    if data_type in {"binary", "integer"}:
        return int(value)
    if data_type == "float":
        return float(value)
    return str(value)


def _native(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return _native(value)


__all__ = [
    "CompiledFieldPolicy",
    "CompiledGovernancePolicy",
    "GovernanceError",
    "GovernanceLoader",
    "GovernanceRule",
    "PolicyCompiler",
    "is_sentinel",
]

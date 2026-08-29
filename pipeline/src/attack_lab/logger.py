"""Episode trajectory and public-transcript logging."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from attack_lab.paths import new_run_directory
from attack_lab.types import EpisodeResult, StepRecord, to_jsonable


RESEARCHER_ONLY_DIAGNOSTICS_FILENAME = (
    "RESEARCHER_ONLY_defence_diagnostics.json"
)


@dataclass
class TrajectoryLogger:
    """Write researcher-internal and attacker-public artefacts for one episode."""

    run_dir: Path
    run_id: str
    _steps_written: int = field(default=0, init=False, repr=False)

    @classmethod
    def create(
        cls,
        run_id: str | None = None,
        *,
        parent: Path | None = None,
    ) -> "TrajectoryLogger":
        run_dir = new_run_directory(run_id, parent=parent)
        return cls(run_dir=run_dir, run_id=run_dir.name)

    @property
    def trajectory_path(self) -> Path:
        return self.run_dir / "trajectory.jsonl"

    @property
    def public_transcript_path(self) -> Path:
        return self.run_dir / "public_transcript.txt"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def governance_manifest_path(self) -> Path:
        return self.run_dir / "governance_manifest.json"

    @property
    def researcher_only_diagnostics_path(self) -> Path:
        return self.run_dir / RESEARCHER_ONLY_DIAGNOSTICS_FILENAME

    def append_step(self, step: StepRecord) -> None:
        if self._steps_written == 0:
            existing = [
                path
                for path in (self.trajectory_path, self.public_transcript_path)
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Refusing to append to existing experimental trajectory: "
                    + ", ".join(str(path) for path in existing)
                )
        payload = to_jsonable(asdict(step))
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        # Attacker-public transcript deliberately omits research_meta /
        # reference provenance (anchor composition detail is researcher-only).
        public_lines = [
            f"attempt={step.attempt}",
            f"proposed_changes={json.dumps(to_jsonable(step.proposed_changes), sort_keys=True)}",
            f"valid={step.validity.is_valid}",
        ]
        public_lines.append(f"public_feedback={step.public_feedback.label}")
        public_lines.append(f"public_message={step.public_feedback.message}")
        public_lines.append(f"success={step.success}")
        public_lines.append(f"elapsed_ms={step.elapsed_ms:.3f}")
        public_lines.append("---")
        with self.public_transcript_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(public_lines) + "\n")
        self._steps_written += 1

    def write_governance_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Record compiled policy provenance outside the attacker transcript."""
        self._write_new_json(
            self.governance_manifest_path,
            to_jsonable(dict(manifest)),
        )

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            **dict(manifest),
        }
        # Enrich with package versions when readily available.
        for package in ("numpy", "pandas", "sklearn", "xgboost", "joblib"):
            try:
                module = __import__(package if package != "sklearn" else "sklearn")
                payload["software"][package] = getattr(module, "__version__", "unknown")
            except Exception:  # noqa: BLE001
                payload["software"][package] = "unavailable"
        self._write_new_json(self.manifest_path, to_jsonable(payload))

    def write_episode_summary(self, episode: EpisodeResult) -> None:
        path = self.run_dir / "episode_result.json"
        self._write_new_json(path, to_jsonable(asdict(episode)))
        self._write_new_json(
            self.researcher_only_diagnostics_path,
            {
                "access": "RESEARCHER_ONLY",
                "attacker_observation_prohibition": (
                    "Never load this file or any value from it into an attacker "
                    "observation, prompt, memory, repair message or action policy."
                ),
                "case_id": episode.case_id,
                "termination_reason": episode.stop_reason,
                "success": episode.success,
                "q_used": episode.q_used,
                "steps": [
                    {
                        "attempt": step.attempt,
                        "query_index": step.research_meta.get(
                            "query_index", step.attempt
                        ),
                        "proposed_governed_edits": to_jsonable(
                            step.proposed_changes
                        ),
                        "submitted_candidate_fingerprint": step.research_meta.get(
                            "candidate_fingerprint"
                        ),
                        "local_validation_outcome": (
                            "valid" if step.validity.is_valid else "invalid"
                        ),
                        "environment_validation_errors": list(step.validity.errors),
                        "public_decision": step.public_feedback.label,
                        "budget_event": to_jsonable(step.budget_event),
                        "internal_defence": to_jsonable(step.internal_defence),
                        "research_meta": to_jsonable(step.research_meta),
                    }
                    for step in episode.steps
                ],
            },
        )

    @staticmethod
    def _write_new_json(path: Path, payload: Any) -> None:
        """Write one immutable JSON artefact; never replace an existing file."""
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artefact: {path}")
        path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

"""Output-path guards for the attack laboratory.

Default writes go to package-relative ``runs/scratch/``. Formal dissertation
results may be written under ``runs/experiments/`` only when callers pass
``stage="experiments"`` with an explicit parent in that tree.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]

RUNS_ROOT = REPO_ROOT / "runs"
OUTPUTS_ROOT = RUNS_ROOT

EXPERIMENTS_ROOT = OUTPUTS_ROOT / "experiments"
SCRATCH_ROOT = OUTPUTS_ROOT / "scratch"
ARCHIVE_ROOT = OUTPUTS_ROOT / "archive"

SCRATCH_SMOKE_ROOT = SCRATCH_ROOT / "smoke"
SCRATCH_CALIBRATION_ROOT = SCRATCH_ROOT / "calibration"
SCRATCH_DEBUG_ROOT = SCRATCH_ROOT / "debug"

# Default exploratory run root (debug ad-hoc episodes).
DEFAULT_RUN_ROOT = SCRATCH_DEBUG_ROOT

# Backward-compatible alias used by older notes/tests: exploratory root.
ATTACK_LAB_ROOT = SCRATCH_ROOT

DEFAULT_C1_ARTEFACT_DIR = REPO_ROOT / "artifacts" / "d1"
DEFAULT_D2S_ARTEFACT_DIR = REPO_ROOT / "artifacts" / "d2s"
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

PROTECTED_PREFIXES = (
    REPO_ROOT / "artifacts",
    REPO_ROOT / "results",
    REPO_ROOT / "config",
)

OutputStage = Literal["scratch", "experiments"]


class AttackLabPathError(RuntimeError):
    """Raised when an output path would overwrite protected artefacts."""


def new_run_directory(
    run_id: str | None = None,
    *,
    parent: Path | None = None,
    stage: OutputStage = "scratch",
) -> Path:
    """Create a uniquely named run directory.

    Defaults to ``runs/scratch/debug/``. Writing under
    ``runs/experiments/`` requires ``stage="experiments"`` and an
    explicit ``parent`` inside that tree. Existing paths are never overwritten.
    """
    if stage not in {"scratch", "experiments"}:
        raise AttackLabPathError(
            f"Unsupported stage {stage!r}; use 'scratch' or 'experiments'."
        )
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"attack_lab_{stamp}_{uuid.uuid4().hex[:8]}"

    if parent is None:
        if stage == "experiments":
            raise AttackLabPathError(
                "Writing under experiments/ requires stage='experiments' "
                "and an explicit parent path under runs/experiments/."
            )
        root = DEFAULT_RUN_ROOT
    else:
        root = Path(parent)
        if stage == "experiments":
            assert_under_root(root, EXPERIMENTS_ROOT, label="experiments")
        else:
            assert_under_root(root, SCRATCH_ROOT, label="scratch")

    root.mkdir(parents=True, exist_ok=True)
    path = root / run_id
    if path.exists():
        raise AttackLabPathError(f"Refusing to overwrite existing run directory: {path}")
    assert_not_protected(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def assert_under_root(path: Path, root: Path, *, label: str) -> None:
    """Ensure ``path`` equals ``root`` or lives beneath it."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise AttackLabPathError(
            f"Output parent must live under {label} root {root_resolved}; got {resolved}"
        )


def assert_under_attack_lab(path: Path) -> None:
    """Backward-compatible alias: parents must live under scratch/."""
    assert_under_root(path, SCRATCH_ROOT, label="scratch")


def assert_not_protected(path: Path) -> None:
    """Refuse writes into existing experimental artefact trees."""
    resolved = path.resolve()
    for prefix in PROTECTED_PREFIXES:
        root = prefix.resolve()
        if resolved == root or root in resolved.parents:
            raise AttackLabPathError(
                f"Refusing to write under protected artefact tree {root}: {resolved}"
            )

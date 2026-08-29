#!/usr/bin/env python3
"""Fit the frozen D2-S reference on Months 0–5 legitimate rows only.

Does not open Month 6 for fitting, does not open Month 7, and does not
call any LLM/API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "pipeline" / "src"
sys.path.insert(0, str(SRC))

from attack_lab.paths import SCRATCH_CALIBRATION_ROOT, new_run_directory  # noqa: E402
from d2.contract import SCORE_CONTRACT_ID, score_contract_payload  # noqa: E402
from d2.data import DEFAULT_RAW_PATH, load_reference_legitimate  # noqa: E402
from d2.scoring import fit_d2s_scorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--raw", dest="raw", type=Path, default=None, help="Path to Base.csv")
    parser.add_argument("--output-parent", type=Path, default=SCRATCH_CALIBRATION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = args.raw or Path(os.getenv("BAF_BASE_CSV", "Base.csv"))
    if not raw_path.is_file():
        print(
            f"ERROR: Raw BAF dataset not found at {raw_path}. "
            "Pass --data /path/to/Base.csv or set BAF_BASE_CSV environment variable.",
            file=sys.stderr,
        )
        return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"d2s_reference_{SCORE_CONTRACT_ID}_{stamp}",
        parent=args.output_parent,
        stage="scratch",
    )
    loaded = load_reference_legitimate(raw_path, verify_hash=True)
    if loaded.month7_opened:
        raise SystemExit("Month 7 was opened during reference load; aborting.")
    scorer = fit_d2s_scorer(
        loaded.frame,
        raw_sha256=loaded.raw_sha256,
        month7_opened=loaded.month7_opened,
    )
    artefact_path = run_dir / "d2s_reference.json"
    scorer.save(artefact_path)
    summary = {
        "score_contract": score_contract_payload(),
        "reference_n": scorer.reference_n,
        "reference_sha256": scorer.reference_sha256,
        "reference_months": list(scorer.reference_months),
        "fingerprint": scorer.fingerprint,
        "month7_opened": scorer.month7_opened,
        "n_rows_read": loaded.n_rows_read,
        "n_sealed_rows_skipped": loaded.n_sealed_rows_skipped,
        "redundancy": {
            "max_abs_spearman_offdiag": scorer.redundancy.get("max_abs_spearman_offdiag"),
            "flagged_pairs": scorer.redundancy.get("flagged_pairs"),
            "flag_threshold": scorer.redundancy.get("flag_threshold"),
        },
        "artefact_path": str(artefact_path),
    }
    (run_dir / "FIT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

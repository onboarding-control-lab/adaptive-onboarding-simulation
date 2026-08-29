#!/usr/bin/env python3
"""Month-6 D2-S calibration and security curve.

Requires a frozen D2-S reference artefact. Does not refit relationships,
does not open Month 7, and does not call any LLM/API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "pipeline" / "src"
sys.path.insert(0, str(SRC))

from attack_lab.paths import SCRATCH_CALIBRATION_ROOT, new_run_directory  # noqa: E402
from d2.calibrate import (  # noqa: E402
    DEFAULT_BENCHMARK_DIR,
    PRIMARY_ATTACKER_DIRS,
    SUPPLEMENTARY_ATTACKER_DIRS,
    curve_table,
    load_attacker_d1_pass_submissions,
    month6_legitimate_d1_pass,
    score_submissions,
    security_rows,
    thresholds_for_budgets,
    verify_benchmark_provenance,
)
from d2.contract import SCORE_CONTRACT_ID  # noqa: E402
from d2.data import DEFAULT_RAW_PATH  # noqa: E402
from d2.plotting import plot_security_curve  # noqa: E402
from d2.scoring import D2SScorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Path to d2s_reference.json")
    parser.add_argument("--data", "--raw", dest="raw", type=Path, default=None, help="Path to Base.csv")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_DIR, help="Path to dev benchmark")
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
    if not args.reference.is_file():
        print(f"ERROR: Reference artefact not found at {args.reference}.", file=sys.stderr)
        return 1
    scorer = D2SScorer.load(args.reference)
    if scorer.month7_opened:
        raise SystemExit("Refusing to calibrate a scorer that opened Month 7.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = new_run_directory(
        f"d2s_month6_calibration_{SCORE_CONTRACT_ID}_{stamp}",
        parent=args.output_parent,
        stage="scratch",
    )

    legit = month6_legitimate_d1_pass(raw_path, verify_hash=True)
    legit_scores = scorer.score_many(legit)["d2_score"].to_numpy()
    budget_table = thresholds_for_budgets(legit_scores)

    attacker_scores = {}
    attacker_counts = {}
    if args.benchmark and args.benchmark.is_dir():
        provenance = verify_benchmark_provenance(args.benchmark)
        for label, directory in {**PRIMARY_ATTACKER_DIRS, **SUPPLEMENTARY_ATTACKER_DIRS}.items():
            submissions = load_attacker_d1_pass_submissions(args.benchmark, directory)
            attacker_scores[label] = score_submissions(scorer, submissions)
            attacker_counts[label] = {
                "n_d1_pass_submissions": len(submissions),
                "condition_dir": directory,
            }
        security = security_rows(budget_table=budget_table, attacker_scores=attacker_scores)
        primary = curve_table(security)
        plot_path = plot_security_curve(primary, run_dir / "d2s_month6_security_curve.png")
        primary.to_csv(run_dir / "security_curve_primary.csv", index=False)
        security.to_csv(run_dir / "security_curve_full.csv", index=False)
        prov_ok = provenance.ok
    else:
        prov_ok = True

    budget_table.to_csv(run_dir / "benign_review_thresholds.csv", index=False)
    pd.DataFrame({"d2_score": legit_scores}).to_csv(
        run_dir / "month6_legit_d1_pass_d2_scores.csv", index=False
    )

    summary = {
        "score_contract_id": SCORE_CONTRACT_ID,
        "reference_path": str(args.reference),
        "reference_fingerprint": scorer.fingerprint,
        "reference_n": scorer.reference_n,
        "month6_legitimate_d1_pass_n": int(len(legit)),
        "month7_opened": False,
        "months_scored": [6],
        "primary_attacker_mapping": PRIMARY_ATTACKER_DIRS,
        "supplementary_attacker_mapping": SUPPLEMENTARY_ATTACKER_DIRS,
        "attacker_counts": attacker_counts,
        "provenance": {
            "ok": prov_ok,
        },
        "review_budgets_calibrated": [float(b) for b in budget_table["review_budget"]],
    }
    (run_dir / "CALIBRATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sealed final Month-7 experiment runner.

Default mode is fail-closed.

``--dry-run``: orchestration-shell check (fixtures; no Month 7; no API).
``--rehearse-final``: real A0/A1/A2/A3 + frozen D1, mock LLM transport only.
``--execute-final``: authorised Month-7 execution. Refuses until production
anchors are issued. Requires explicit BAF data path and API credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "pipeline" / "src"
sys.path.insert(0, str(SRC))

from dotenv import load_dotenv  # noqa: E402

from attack_lab.final_anchors import AnchorManifestError  # noqa: E402
from attack_lab.final_experiment import (  # noqa: E402
    FORBIDDEN_EXECUTE_WITHOUT_FLAG,
    FinalRunnerError,
    run_dry_run,
    run_execute_final,
    run_rehearse_final,
)
from attack_lab.final_pipeline import FinalPipelineError  # noqa: E402
from attack_lab.final_protocol import DEFAULT_PROTOCOL_PATH, FinalProtocolConfig  # noqa: E402
from attack_lab.final_reporting import maybe_generate_post_run_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Path to the frozen final protocol JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Orchestration-shell dry-run. No Month 7, no API.",
    )
    parser.add_argument(
        "--rehearse-final",
        action="store_true",
        help="Real-class rehearsal: frozen D1 and real A0-A3, mock LLM only. No Month 7.",
    )
    parser.add_argument(
        "--execute-final",
        action="store_true",
        help="Authorised Month-7 execution. Requires --data and API credentials.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to raw BAF Base.csv dataset file.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file containing DEEPSEEK_API_KEY.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=None,
        help="Optional parent directory under runs/experiments/.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate audit/tables/figures from a COMPLETE run. Does not rerun Month 7.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="COMPLETE run directory for --report-only.",
    )
    return parser.parse_args(argv)


def _print_reporting(payload: dict) -> None:
    print(json.dumps({"reporting": payload}, default=str))
    if payload.get("status") != "COMPLETE":
        print(
            "REPORTING FAILED (scientific run untouched): "
            + str(payload.get("error") or payload.get("status")),
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env_file and args.env_file.is_file():
        load_dotenv(dotenv_path=args.env_file, override=False)
    if args.data:
        os.environ["BAF_BASE_CSV"] = str(args.data.resolve())

    try:
        if args.report_only:
            if args.run_dir is None:
                raise FinalRunnerError("--report-only requires --run-dir.")
            payload = maybe_generate_post_run_report(args.run_dir)
            _print_reporting(payload)
            return 0 if payload.get("status") == "COMPLETE" else 2
        protocol = FinalProtocolConfig.load(args.protocol)
        selected = [bool(args.dry_run), bool(args.rehearse_final), bool(args.execute_final)]
        if sum(selected) > 1:
            raise FinalRunnerError(
                "Pass only one of --dry-run, --rehearse-final, or --execute-final."
            )
        if args.dry_run:
            result = run_dry_run(protocol=protocol, output_parent=args.output_parent)
            print(json.dumps({"status": result["status"], "run_dir": result["run_dir"]}))
            return 0
        if args.rehearse_final:
            result = run_rehearse_final(
                protocol=protocol, output_parent=args.output_parent
            )
            print(json.dumps({"status": result["status"], "run_dir": result["run_dir"]}))
            if result.get("status") == "COMPLETE":
                _print_reporting(maybe_generate_post_run_report(Path(result["run_dir"])))
            return 0
        if args.execute_final:
            result = run_execute_final(protocol, output_parent=args.output_parent)
            print(json.dumps({"status": result["status"], "run_dir": result["run_dir"]}))
            if result.get("status") == "COMPLETE":
                _print_reporting(maybe_generate_post_run_report(Path(result["run_dir"])))
            return 0
        raise FinalRunnerError(FORBIDDEN_EXECUTE_WITHOUT_FLAG)
    except (FinalRunnerError, FinalPipelineError, AnchorManifestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

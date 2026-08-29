#!/usr/bin/env python3
"""Unified CLI entry point for the reproducibility package.

Commands:
  verify           Offline structural and cryptographic integrity check.
  inspect-results  Offline inspection of packaged Month-7 experimental results.
  run-final        Guarded execution entry point (requires BAF data & API credentials).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "pipeline" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# 1. verify command
# -----------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    """Offline structural and cryptographic integrity validation."""
    print("=" * 72)
    print("Controlled Financial Onboarding Simulation — Offline Integrity Verification")
    print("=" * 72)

    errors = 0
    checks_passed = 0

    # 1. Structural file checks
    required_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "RELEASE_MANIFEST.md",
        REPO_ROOT / "config" / "final_month7_protocol.json",
        REPO_ROOT / "config" / "final_month7_dryrun_anchors.json",
        REPO_ROOT / "config" / "feature_handling.csv",
        REPO_ROOT / "config" / "attacker_feature_governance.csv",
        REPO_ROOT / "config" / "attacker_compiled_governance.json",
        REPO_ROOT / "config" / "reference_pool_config.json",
        REPO_ROOT / "artifacts" / "d1" / "fitted_pipeline.joblib",
        REPO_ROOT / "artifacts" / "d1" / "development_month6_threshold_selection.json",
        REPO_ROOT / "artifacts" / "d1" / "config.json",
        REPO_ROOT / "artifacts" / "d2s" / "d2s_reference.json",
        REPO_ROOT / "artifacts" / "d2s" / "benign_review_thresholds.csv",
        REPO_ROOT / "artifacts" / "d2s" / "v11" / "D2S_V11_IFOREST_MODEL.joblib",
        REPO_ROOT / "artifacts" / "d2s" / "v11" / "D2S_V11_IFOREST_CONFIG.json",
        REPO_ROOT / "results" / "RUN_STATUS.json",
        REPO_ROOT / "results" / "metrics.json",
        REPO_ROOT / "results" / "tables" / "rq1_results.csv",
        REPO_ROOT / "results" / "tables" / "rq2_results.csv",
        REPO_ROOT / "results" / "tables" / "paired_bootstrap_d1_asr.csv",
        REPO_ROOT / "results" / "tables" / "paired_bootstrap_e2e.csv",
        REPO_ROOT / "results" / "tables" / "review_sensitivity.csv",
        REPO_ROOT / "pipeline" / "scripts" / "run_final_month7_experiment.py",
        REPO_ROOT / "pipeline" / "scripts" / "issue_final_month7_anchors.py",
        REPO_ROOT / "pipeline" / "scripts" / "run_d2s_fit_reference.py",
        REPO_ROOT / "pipeline" / "scripts" / "run_d2s_month6_calibration.py",
    ]

    print("\n[1/4] Checking required package files...")
    missing = [str(p.relative_to(REPO_ROOT)) for p in required_paths if not p.is_file()]
    if missing:
        print(f"  FAIL: Missing {len(missing)} required files:")
        for m in missing:
            print(f"    - {m}")
        errors += len(missing)
    else:
        print(f"  PASS: All {len(required_paths)} required structural files present.")
        checks_passed += 1

    # 2. Cryptographic checksum verification of frozen artefacts
    pinned_hashes = {
        "artifacts/d1/fitted_pipeline.joblib": "243c851b0c665c9c9ddad88941004eba6a30f98a8f446961c310f7521fb5fe4a",
        "artifacts/d1/development_month6_threshold_selection.json": "e3920e64df96a984433da99f963860bdc3656214edd1a2f5dbc9a83468809455",
        "artifacts/d1/config.json": "6a8c0e7d101ee8c95a30cb17c365ab447108fd8abd84c344498c79556b0bfb81",
        "artifacts/d2s/d2s_reference.json": "1ca5677687ab45418a98f2bbab6237ca1ce18c00abd6485a4fb565be6e9ea26e",
        "artifacts/d2s/v11/D2S_V11_IFOREST_MODEL.joblib": "7d72a2bbf4fc61dffdd0b4aa35148835b0f74d4b48149c645f71777fd0874d9b",
        "artifacts/d2s/v11/D2S_V11_IFOREST_CONFIG.json": "0ff835fc9b593234ed649db46fbe50698aa3ab17674f05f1ebc17ffc0b959079",
        "config/final_month7_dryrun_anchors.json": "f801b4734a77f00c4ae98e1d1e6ceda79fa9140b5a3be15abd2ce10781adee5b",
        "config/attacker_compiled_governance.json": "fcf1d3d1396bff92c0527696e29e64039a2766b6404567ffa3581fa8275a8b8c",
        "config/final_month7_protocol.json": "0b7b62fc2033f17312eb7cf8840b94e3f1081dd664ab37e38a69b878b05b3158",
    }

    print("\n[2/4] Verifying SHA-256 cryptographic signatures...")
    hash_mismatches = 0
    for rel, expected in pinned_hashes.items():
        file_path = REPO_ROOT / rel
        if not file_path.is_file():
            print(f"  FAIL: {rel} (missing)")
            hash_mismatches += 1
            continue
        actual = sha256_file(file_path)
        if actual != expected:
            print(f"  FAIL: {rel} (expected {expected[:16]}..., got {actual[:16]}...)")
            hash_mismatches += 1
        else:
            print(f"  PASS: {rel} ({actual[:16]}...)")

    if hash_mismatches == 0:
        checks_passed += 1
    else:
        errors += hash_mismatches

    # 3. Protocol configuration & model contract verification
    print("\n[3/4] Validating frozen protocol contracts and artefact bindings...")
    try:
        from attack_lab.final_protocol import FinalProtocolConfig, verify_frozen_artefacts

        protocol = FinalProtocolConfig.load()
        artefact_errs = verify_frozen_artefacts(protocol)
        if artefact_errs:
            print("  FAIL: Protocol artefact validation errors:")
            for e in artefact_errs:
                print(f"    - {e}")
            errors += len(artefact_errs)
        else:
            print(f"  PASS: Protocol configuration valid (config_hash={protocol.config_hash[:16]}...).")
            print("  PASS: Frozen D1, D2-S, and governance bindings match specification.")
            checks_passed += 1
    except Exception as exc:
        print(f"  FAIL: Protocol validation exception: {exc}")
        errors += 1

    # 4. Path portability verification (no hardcoded absolute local paths in active code)
    print("\n[4/4] Checking path portability of executable code and active configuration...")
    import re

    user_path_re = re.compile(r"/(?:Users|home|Volumes)/[a-zA-Z0-9_.-]+/")
    dev_tree_re = re.compile(r"0[1-5]_[a-zA-Z0-9_-]+")
    active_dirs = [REPO_ROOT / "pipeline", REPO_ROOT / "config"]
    portability_violations = []

    for d in active_dirs:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".json", ".csv", ".yaml", ".yml", ".sh"}:
                try:
                    text = p.read_text(encoding="utf-8")
                    m1 = user_path_re.search(text)
                    if m1:
                        portability_violations.append((str(p.relative_to(REPO_ROOT)), m1.group(0)))
                    m2 = dev_tree_re.search(text)
                    if m2:
                        portability_violations.append((str(p.relative_to(REPO_ROOT)), m2.group(0)))
                except Exception:
                    pass

    if portability_violations:
        print(f"  FAIL: Found {len(portability_violations)} active non-portable path references:")
        for rel_file, pat in portability_violations:
            print(f"    - {rel_file}: contains '{pat}'")
        errors += len(portability_violations)
    else:
        print("  PASS: Zero active hardcoded local paths found in pipeline code and configuration.")
        checks_passed += 1

    print("\n" + "-" * 72)
    if errors == 0:
        print(f"VERIFICATION SUCCESSFUL: {checks_passed}/4 check suites passed. Bundle is intact and portable.")
        print("-" * 72)
        return 0
    else:
        print(f"VERIFICATION FAILED: {errors} error(s) detected across checks.")
        print("-" * 72)
        return 1


# -----------------------------------------------------------------------------
# 2. inspect-results command
# -----------------------------------------------------------------------------

def cmd_inspect_results(args: argparse.Namespace) -> int:
    """Offline inspection and formatting of included Month-7 results."""
    print("=" * 78)
    print("Controlled Financial Onboarding Simulation — Confirmatory Evaluation Results")
    print("=" * 78)

    status_file = REPO_ROOT / "results" / "RUN_STATUS.json"
    metrics_file = REPO_ROOT / "results" / "metrics.json"
    rq1_file = REPO_ROOT / "results" / "tables" / "rq1_results.csv"
    rq2_file = REPO_ROOT / "results" / "tables" / "rq2_results.csv"
    sens_file = REPO_ROOT / "results" / "tables" / "review_sensitivity.csv"

    if not status_file.is_file() or not metrics_file.is_file():
        print("ERROR: Packaged results not found in results/", file=sys.stderr)
        return 1

    status_data = json.loads(status_file.read_text(encoding="utf-8"))
    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))

    print("\n1. EXECUTION METADATA")
    print(f"  Evaluation Phase : Final Month 7 (Evaluation Month = 7)")
    print(f"  Protocol ID      : {status_data.get('protocol_id', 'final-month7-a0-a3-d1-d2s-20260817')}")
    print(f"  Execution Status : {status_data.get('status')} (mode={status_data.get('mode')})")
    print(f"  Month 7 Accessed : {status_data.get('month7_accessed')} (single-pass production evaluation)")
    print(f"  Live API Calls   : {status_data.get('live_api_calls', 0)} (recorded ledger)")
    print(f"  Anchors Evaluated: N = 100 paired eligible true-positive fraud anchors")
    print(f"  Budget Contract  : K = 10 reference records, m = 2 edits/attempt, Q = 5 submissions")

    # Table 1: RQ1 Results
    print("\n2. RQ1: ATTACKER ADAPTATION VS FIRST-LINE STATISTICAL DEFENCE (D1 XGBoost)")
    print("   Frozen D1 threshold = 0.047245666 (Month-6 calibration enforcing FPR <= 5%)\n")
    print(f"  {'Attacker Policy':<10} | {'Description':<32} | {'Pass / N':<10} | {'D1 ASR (%)':<12} | {'95% Bootstrap CI'}")
    print("  " + "-" * 88)

    policy_descriptions = {
        "A0": "Constrained Random Baseline",
        "A1": "One-Shot LLM Planner (DeepSeek)",
        "A2": "Gower-Guided Searcher",
        "A3": "Episodic Reflective Agent",
    }

    if rq1_file.is_file():
        with open(rq1_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                att = row.get("attacker", "")
                desc = policy_descriptions.get(att, row.get("description", ""))
                pass_count = row.get("anchors_with_d1_pass", row.get("pass_count", ""))
                denom = row.get("assigned_eligible_anchors", row.get("denominator", "100"))
                asr = float(row.get("d1_asr_at_5", row.get("asr", 0.0))) * 100
                ci_low = float(row.get("d1_asr_at_5_low", row.get("ci_lower", 0.0))) * 100
                ci_high = float(row.get("d1_asr_at_5_high", row.get("ci_upper", 0.0))) * 100
                print(f"  {att:<10} | {desc:<32} | {f'{pass_count}/{denom}':<10} | {asr:>6.1f}%     | [{ci_low:0.1f}%, {ci_high:0.1f}%]")

    # Table 2: RQ2 Results
    print("\n3. RQ2: LAYERED DEFENCE EVALUATION (D1 XGBoost + D2-S Pairwise Consistency Review)")
    print("   Primary review capacity = 10% legitimate onboarding capacity (equal-weight mean aggregation)\n")
    print(f"  {'Attacker Policy':<10} | {'D1 ASR (%)':<12} | {'D2-S Interception':<18} | {'E2E Bypass (%)':<15} | {'95% Bootstrap CI'}")
    print("  " + "-" * 88)

    if rq2_file.is_file():
        with open(rq2_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                att = row.get("attacker", "")
                d1_asr = float(row.get("d1_asr_at_5", 0.0)) * 100
                inter = float(row.get("conditional_d2_interception", 0.0)) * 100
                e2e = float(row.get("e2e_bypass_at_10pct", 0.0)) * 100
                ci_low = float(row.get("e2e_low", 0.0)) * 100
                ci_high = float(row.get("e2e_high", 0.0)) * 100
                print(f"  {att:<10} | {d1_asr:>6.1f}%     | {inter:>10.1f}%        | {e2e:>6.1f}%         | [{ci_low:0.1f}%, {ci_high:0.1f}%]")

    # Table 3: Sensitivity Results
    if sens_file.is_file():
        print("\n4. D2-S REVIEW CAPACITY SENSITIVITY (5%, 10%, 15% REVIEW BUDGETS)\n")
        print(f"  {'Review Capacity':<18} | {'A0 Bypass':<12} | {'A1 Bypass':<12} | {'A2 Bypass':<12} | {'A3 Bypass'}")
        print("  " + "-" * 75)
        budget_map = {"0.05": "5% Capacity", "0.1": "10% (Primary)", "0.15": "15% Capacity"}
        b_data: dict[str, dict[str, float]] = {"0.05": {}, "0.1": {}, "0.15": {}}
        with open(sens_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                att = row.get("attacker", "")
                b_str = str(row.get("review_budget", ""))
                e2e_val = float(row.get("e2e_bypass", 0.0)) * 100
                if b_str in b_data:
                    b_data[b_str][att] = e2e_val

        for b_str, label in budget_map.items():
            a0_v = b_data[b_str].get("A0", 0.0)
            a1_v = b_data[b_str].get("A1", 0.0)
            a2_v = b_data[b_str].get("A2", 0.0)
            a3_v = b_data[b_str].get("A3", 0.0)
            print(f"  {label:<18} | {a0_v:>6.1f}%      | {a1_v:>6.1f}%      | {a2_v:>6.1f}%      | {a3_v:>6.1f}%")

    print("\n" + "=" * 78)
    print("NOTE: Values reflect the frozen empirical findings evaluated on Month 7.")
    print("Full theoretical derivations and discussion are provided in the dissertation.")
    print("=" * 78)
    return 0


# -----------------------------------------------------------------------------
# 3. run-final command
# -----------------------------------------------------------------------------

def cmd_run_final(args: argparse.Namespace) -> int:
    """Guarded pipeline execution wrapper."""
    script_path = REPO_ROOT / "pipeline" / "scripts" / "run_final_month7_experiment.py"
    if not script_path.is_file():
        print(f"ERROR: Execution script not found at {script_path}", file=sys.stderr)
        return 1

    # Pass through to run_final_month7_experiment main
    from pipeline.scripts.run_final_month7_experiment import main as runner_main

    argv = []
    if args.dry_run:
        argv.append("--dry-run")
    if args.rehearse_final:
        argv.append("--rehearse-final")
    if args.execute_final:
        argv.append("--execute-final")
    if args.data:
        argv.extend(["--data", str(args.data)])
    if args.env_file:
        argv.extend(["--env-file", str(args.env_file)])
    if args.output_parent:
        argv.extend(["--output-parent", str(args.output_parent)])
    if args.protocol:
        argv.extend(["--protocol", str(args.protocol)])
    if args.report_only:
        argv.append("--report-only")
    if args.run_dir:
        argv.extend(["--run-dir", str(args.run_dir)])

    if not any([args.dry_run, args.rehearse_final, args.execute_final, args.report_only]):
        print(
            "ERROR: Pass one execution mode flag:\n"
            "  --dry-run         : Safe dry run (fixtures only; no Month 7, no API)\n"
            "  --rehearse-final  : Rehearsal with mock LLM (no Month 7, no live API)\n"
            "  --execute-final   : Full Month-7 execution (requires --data and API credentials)\n"
            "  --report-only     : Post-run report from existing run directory",
            file=sys.stderr,
        )
        return 1

    return runner_main(argv)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. verify
    parser_verify = subparsers.add_parser(
        "verify",
        help="Offline structural and cryptographic integrity check of the bundle.",
    )
    parser_verify.set_defaults(func=cmd_verify)

    # 2. inspect-results
    parser_inspect = subparsers.add_parser(
        "inspect-results",
        help="Offline inspection and formatting of included Month-7 results.",
    )
    parser_inspect.set_defaults(func=cmd_inspect_results)

    # 3. run-final
    parser_run = subparsers.add_parser(
        "run-final",
        help="Guarded pipeline execution entry point (requires BAF data & API credentials).",
    )
    parser_run.add_argument("--dry-run", action="store_true", help="Safe dry run check.")
    parser_run.add_argument("--rehearse-final", action="store_true", help="Rehearsal with mock LLM.")
    parser_run.add_argument("--execute-final", action="store_true", help="Authorised Month-7 execution.")
    parser_run.add_argument("--data", type=Path, default=None, help="Path to Base.csv.")
    parser_run.add_argument("--env-file", type=Path, default=None, help="Path to .env.")
    parser_run.add_argument("--output-parent", type=Path, default=None, help="Output directory under runs/.")
    parser_run.add_argument("--protocol", type=Path, default=REPO_ROOT / "config" / "final_month7_protocol.json")
    parser_run.add_argument("--report-only", action="store_true", help="Post-run reporting.")
    parser_run.add_argument("--run-dir", type=Path, default=None, help="Run directory for reporting.")
    parser_run.set_defaults(func=cmd_run_final)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

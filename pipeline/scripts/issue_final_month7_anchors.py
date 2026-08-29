#!/usr/bin/env python3
"""Issue the immutable Month-7 paired-anchor manifest.

Eligibility and selection only. No attacker APIs.
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

from attack_lab.final_anchor_issuance import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    issue_production_anchor_manifest,
)
from attack_lab.final_protocol import DEFAULT_PROTOCOL_PATH, FinalProtocolConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--data", type=Path, default=None, help="Path to Base.csv")
    parser.add_argument("--dest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data:
        os.environ["BAF_BASE_CSV"] = str(args.data.resolve())
    protocol = FinalProtocolConfig.load(args.protocol)
    result = issue_production_anchor_manifest(
        protocol=protocol,
        dest=args.dest,
        overwrite=args.overwrite,
        raw_path_override=args.data,
    )
    print(json.dumps({k: result[k] for k in (
        "manifest_path",
        "eligible_universe_size",
        "selected_n",
        "anchor_fingerprint",
        "manifest_sha256",
        "month7_accessed",
        "attack_outcomes_inspected",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Source-file integrity and write-path guards.

Protects the immutable raw file: SHA-256 verification before and after
every pipeline run, a writability probe, and a guard that refuses any
output path inside (or above) the raw data directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from baf_data.errors import OutputPathError, RawSourceIntegrityError

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_source(raw_path: Path, expected_sha256: str) -> str:
    """Verify that the raw file exists and matches the expected hash.

    Returns the observed digest. Raises :class:`RawSourceIntegrityError`
    if the file is missing or the digest differs; the pipeline must
    refuse to run in that case.
    """
    if not raw_path.is_file():
        raise RawSourceIntegrityError(
            f"Raw source file not found: {raw_path}. "
            "Check that the external drive is mounted."
        )
    observed = sha256_of_file(raw_path)
    if observed != expected_sha256:
        raise RawSourceIntegrityError(
            f"SHA-256 mismatch for {raw_path}: expected {expected_sha256}, "
            f"observed {observed}. Refusing to continue."
        )
    logger.info("Raw source hash verified: %s", observed)
    return observed


def raw_file_is_writable(raw_path: Path) -> bool:
    """Report whether the current process could write to the raw file."""
    return os.access(raw_path, os.W_OK)


def ensure_output_path_allowed(output_dir: Path, raw_path: Path) -> None:
    """Refuse output locations inside, equal to, or above the raw directory.

    Raises :class:`OutputPathError` when writing to ``output_dir`` could
    pollute the raw data directory or its parents.
    """
    resolved_output = output_dir.resolve()
    raw_dir = raw_path.resolve().parent
    if resolved_output == raw_dir or raw_dir in resolved_output.parents:
        raise OutputPathError(
            f"Output directory {resolved_output} is inside the raw data "
            f"directory {raw_dir}. Choose a separate location."
        )
    if resolved_output in raw_dir.parents:
        raise OutputPathError(
            f"Output directory {resolved_output} is an ancestor of the raw "
            f"data directory {raw_dir}. Choose a separate location."
        )

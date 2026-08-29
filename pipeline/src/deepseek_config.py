"""Load DeepSeek API settings from environment variables or a specified .env file.

This module is intentionally separate from A0/A2 attack-lab code. It only
exposes local configuration for API connectivity checks and LLM execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class DeepSeekSettings:
    """Resolved DeepSeek client settings (API key is never printed by helpers)."""

    api_key: str
    base_url: str
    model: str
    env_path: Path


def _mask_secret(value: str) -> str:
    """Return a non-reversible hint that a secret is present (never the full key)."""
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-2:]}"


def load_deepseek_settings(env_path: Path | None = None) -> DeepSeekSettings:
    """Load DeepSeek settings from environment variables or .env.

    Raises:
        RuntimeError: if ``DEEPSEEK_API_KEY`` is missing or blank.
    """
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if path.is_file():
        load_dotenv(dotenv_path=path, override=False)
    else:
        load_dotenv(override=False)

    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Provide DEEPSEEK_API_KEY via environment "
            f"variable or supply an env file (checked: {path})."
        )

    base_url = (
        os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    ).strip()
    model = (os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro").strip()

    return DeepSeekSettings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        env_path=path,
    )


def describe_settings(settings: DeepSeekSettings) -> str:
    """Human-readable settings summary that never includes the full API key."""
    return (
        f"env_path={settings.env_path} "
        f"base_url={settings.base_url} "
        f"model={settings.model} "
        f"api_key={_mask_secret(settings.api_key)}"
    )

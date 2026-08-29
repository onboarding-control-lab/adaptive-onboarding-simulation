"""Month-6 D2-S calibration/security curve figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_security_curve(table: pd.DataFrame, path: Path) -> Path:
    """X = benign review rate; Y = residual attack interception. One curve per A0–A3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    series = (
        ("A0_interception", "A0"),
        ("A1_interception", "A1"),
        ("A2_interception", "A2"),
        ("A3_interception", "A3"),
    )
    for column, label in series:
        ax.plot(
            table["benign_review_rate"],
            table[column],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Benign review rate")
    ax.set_ylabel("Residual attack interception rate")
    ax.set_title("D2-S Month-6 calibration / security curve (development only)")
    ax.set_xlim(0.0, max(0.22, float(table["benign_review_rate"].max()) + 0.01))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

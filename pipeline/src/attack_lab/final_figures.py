"""Publication figures for the final evaluation reporting stage.

Quantitative figures use matplotlib only. The evaluation-framework schematic
is stored as Graphviz DOT (editable source) and also rendered with matplotlib
because Graphviz may be absent at runtime.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplconfig_final_reporting")))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ATTACKERS = ("A0", "A1", "A2", "A3")
FRAMEWORK_DOT = '''digraph FinalEvaluationFramework {
  graph [rankdir=TB, fontname="Helvetica", fontsize=11, bgcolor="white"];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];
  edge [fontname="Helvetica", fontsize=9];

  anchors [label="N paired eligible anchors\\n(same ordered set for A0–A3)", fillcolor="#F4F1EA"];
  policies [label="A0 / A1 / A2 / A3", fillcolor="#E8EEF4"];
  queries [label="Up to Q=5 defended submissions\\n(m≤2; PASS / BLOCK / INVALID)", fillcolor="#E8EEF4"];
  rq1fail [label="No D1 PASS\\nRQ1 failure", fillcolor="#E6E6E6"];
  rq1ok [label="D1 PASS\\nRQ1 success", fillcolor="#D9E8D3"];
  d2 [label="D2-S (primary 10%\\nlegitimate-review point)", fillcolor="#F7E6C7"];
  review [label="REVIEW", fillcolor="#F0D9D9"];
  clear [label="CLEAR", fillcolor="#D9E8D3"];
  rq2 [label="RQ2 end-to-end bypass", fillcolor="#D9E8D3"];
  diag [shape=note, label="Side diagnostics\\n(not primary outcomes):\\nquery efficiency\\ncandidate validity\\naction-space exhaustion", fillcolor="#F7F7F7"];

  anchors -> policies -> queries;
  queries -> rq1fail;
  queries -> rq1ok -> d2;
  d2 -> review;
  d2 -> clear -> rq2;
  queries -> diag [style=dashed];
}
'''


def _render_dot(dot_path: Path, stem: Path) -> dict[str, str] | None:
    binary = shutil.which("dot") or str(Path(sys.executable).parent / "dot")
    if not Path(binary).is_file():
        return None
    outputs = {}
    for fmt in ("svg", "pdf", "png"):
        dest = stem.with_suffix(f".{fmt}")
        subprocess.run(
            [binary, f"-T{fmt}", "-o", str(dest), str(dot_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs[fmt] = str(dest)
    outputs["dot"] = str(dot_path)
    return outputs


def _save_current(path_stem: Path) -> dict[str, str]:
    svg = path_stem.with_suffix(".svg")
    pdf = path_stem.with_suffix(".pdf")
    png = path_stem.with_suffix(".png")
    plt.savefig(svg, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close()
    return {"svg": str(svg), "pdf": str(pdf), "png": str(png)}


def _maybe_rehearsal_title(title: str, mode: str) -> str:
    if mode == "production":
        return title
    return f"{title} (reporting rehearsal — not Month-7 results)"


def render_framework_figure(figures_dir: Path, *, n_assigned: int, mode: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    dot_path = figures_dir / "fig_eval_framework.dot"
    dot_path.write_text(FRAMEWORK_DOT, encoding="utf-8")
    graphviz_paths = _render_dot(dot_path, figures_dir / "fig_eval_framework")
    if graphviz_paths is not None:
        return graphviz_paths
    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title(
        _maybe_rehearsal_title("Final evaluation framework", mode),
        fontsize=12,
        pad=8,
    )

    def box(x, y, w, h, text, fc):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=fc,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="black",
            )
        )

    n_label = "N paired eligible anchors"
    box(2.4, 10.4, 5.2, 1.1, n_label + f"\n(this run N={n_assigned})", "#F4F1EA")
    box(2.4, 8.7, 5.2, 1.0, "A0 / A1 / A2 / A3", "#E8EEF4")
    box(1.6, 6.9, 6.8, 1.1, "Up to Q=5 defended submissions\n(m≤2; PASS / BLOCK / INVALID)", "#E8EEF4")
    box(0.3, 4.8, 3.4, 1.1, "No D1 PASS\nRQ1 failure", "#E6E6E6")
    box(6.3, 4.8, 3.4, 1.1, "D1 PASS\nRQ1 success", "#D9E8D3")
    box(6.3, 3.2, 3.4, 1.0, "D2-S\n10% legitimate-review point", "#F7E6C7")
    box(4.6, 1.5, 2.4, 0.9, "REVIEW", "#F0D9D9")
    box(7.6, 1.5, 2.4, 0.9, "CLEAR", "#D9E8D3")
    box(7.2, 0.2, 3.2, 0.9, "RQ2 E2E bypass", "#D9E8D3")
    box(0.3, 0.2, 3.8, 1.8, "Side diagnostics\nquery efficiency\nvalidity\nexhaustion", "#F7F7F7")
    arrow(5.0, 10.4, 5.0, 9.7)
    arrow(5.0, 8.7, 5.0, 8.0)
    arrow(3.2, 6.9, 2.0, 5.9)
    arrow(6.8, 6.9, 8.0, 5.9)
    arrow(8.0, 4.8, 8.0, 4.2)
    arrow(7.2, 3.2, 5.8, 2.4)
    arrow(8.8, 3.2, 8.8, 2.4)
    arrow(8.8, 1.5, 8.8, 1.1)
    paths = _save_current(figures_dir / "fig_eval_framework")
    paths["dot"] = str(dot_path)
    return paths


def render_r1(figures_dir: Path, denominators: Sequence[Mapping[str, Any]], mode: str) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = list(range(len(ATTACKERS)))
    ys = [float(row["d1_asr_at_5"] or 0.0) for row in denominators]
    lows = [float(row["d1_asr_at_5_low"] or 0.0) for row in denominators]
    highs = [float(row["d1_asr_at_5_high"] or 0.0) for row in denominators]
    yerr = [
        [max(0.0, y - lo) for y, lo in zip(ys, lows)],
        [max(0.0, hi - y) for y, hi in zip(ys, highs)],
    ]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=4, color="black")
    ax.set_xticks(xs, ATTACKERS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("D1 ASR@5")
    ax.set_title(_maybe_rehearsal_title("RQ1: D1 ASR@5", mode))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    return _save_current(figures_dir / "fig_r1_d1_asr")


def render_r2(figures_dir: Path, denominators: Sequence[Mapping[str, Any]], mode: str) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = list(range(len(ATTACKERS)))
    ys = [float(row["e2e_bypass_at_10pct"] or 0.0) for row in denominators]
    lows = [float(row["e2e_low"] or 0.0) for row in denominators]
    highs = [float(row["e2e_high"] or 0.0) for row in denominators]
    yerr = [
        [max(0.0, y - lo) for y, lo in zip(ys, lows)],
        [max(0.0, hi - y) for y, hi in zip(ys, highs)],
    ]
    ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=4, color="black")
    ax.set_xticks(xs, ATTACKERS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("End-to-end bypass (10% review)")
    ax.set_title(_maybe_rehearsal_title("RQ2: E2E bypass at 10% legitimate-review", mode))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    return _save_current(figures_dir / "fig_r2_e2e_bypass")


def render_r3(figures_dir: Path, decomposition: Sequence[Mapping[str, Any]], mode: str) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = list(range(len(ATTACKERS)))
    none = [int(row["no_d1_bypass"]) for row in decomposition]
    review = [int(row["d1_bypass_d2_review"]) for row in decomposition]
    clear = [int(row["d1_bypass_d2_clear"]) for row in decomposition]
    ax.bar(xs, none, label="No D1 bypass", color="#B0B0B0")
    ax.bar(xs, review, bottom=none, label="D1 PASS → D2 REVIEW", color="#8C8C8C")
    ax.bar(
        xs,
        clear,
        bottom=[a + b for a, b in zip(none, review)],
        label="D1 PASS → D2 CLEAR",
        color="#4A4A4A",
    )
    ax.set_xticks(xs, ATTACKERS)
    ax.set_ylabel("Assigned eligible anchors")
    ax.set_title(_maybe_rehearsal_title("Outcome decomposition (denominator = N)", mode))
    ax.legend(frameon=False, fontsize=8)
    return _save_current(figures_dir / "fig_r3_outcome_decomposition")


def render_r4(figures_dir: Path, paired: Sequence[Mapping[str, Any]], mode: str) -> dict[str, str]:
    n = len(paired)
    fig_h = max(3.5, min(11.0, 1.6 + 0.18 * n))
    fig, ax = plt.subplots(figsize=(6.2, fig_h))
    grid = np.full((n, 4), np.nan)
    exhaust = np.zeros((n, 4), dtype=bool)
    for i, row in enumerate(paired):
        for j, attacker in enumerate(ATTACKERS):
            value = row.get(attacker)
            if value not in {None, "missing"}:
                grid[i, j] = float(value)
            exhaust[i, j] = bool(row.get(f"{attacker}_exhaustion"))
    cmap = matplotlib.colormaps["Greys"].copy().with_extremes(bad="#D9D9D9")
    im = ax.imshow(grid, aspect="auto", vmin=1, vmax=5, cmap=cmap)
    for i in range(n):
        for j in range(4):
            if exhaust[i, j]:
                ax.scatter(j, i, marker="x", color="black", s=18)
    ax.set_xticks(range(4), ATTACKERS)
    ax.set_yticks(range(n), [str(row["anchor_id"]) for row in paired], fontsize=7)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Paired eligible anchors")
    ax.set_title(
        _maybe_rehearsal_title(
            "Paired-anchor first D1 PASS query (grey = none by Q=5; x = exhaustion)",
            mode,
        ),
        fontsize=9,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("First valid D1 PASS query")
    ax.text(
        0.0,
        -0.12,
        "Descriptive paired-anchor heterogeneity. Not a causal interpretability claim.",
        transform=ax.transAxes,
        fontsize=7,
    )
    return _save_current(figures_dir / "fig_r4_paired_first_success")


def render_r5(figures_dir: Path, sensitivity: Sequence[Mapping[str, Any]], mode: str) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    markers = {"A0": "o", "A1": "s", "A2": "^", "A3": "D"}
    for attacker in ATTACKERS:
        rows = [row for row in sensitivity if row["attacker"] == attacker]
        xs = [float(row["review_budget"]) for row in rows]
        ys = [float(row["e2e_bypass"] or 0.0) for row in rows]
        ax.plot(xs, ys, marker=markers[attacker], color="black", label=attacker, linewidth=1.0)
    ax.axvline(0.10, color="black", linestyle="--", linewidth=0.8, label="Primary 10%")
    ax.set_xlabel("Legitimate-review capacity")
    ax.set_ylabel("End-to-end bypass")
    ax.set_ylim(0, 1)
    ax.set_xticks([0.05, 0.10, 0.15])
    ax.set_title(_maybe_rehearsal_title("D2-S review-capacity sensitivity", mode))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    return _save_current(figures_dir / "fig_r5_review_sensitivity")


def render_all_final_figures(
    *,
    figures_dir: Path,
    denominators: Sequence[Mapping[str, Any]],
    decomposition: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    n_assigned: int,
    mode: str,
) -> dict[str, Any]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "framework": render_framework_figure(figures_dir, n_assigned=n_assigned, mode=mode),
        "r1": render_r1(figures_dir, denominators, mode),
        "r2": render_r2(figures_dir, denominators, mode),
        "r3": render_r3(figures_dir, decomposition, mode),
        "r4": render_r4(figures_dir, paired, mode),
        "r5": render_r5(figures_dir, sensitivity, mode),
    }


def copy_framework_source_to_figure_library(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(FRAMEWORK_DOT, encoding="utf-8")
    return dest


__all__ = ["FRAMEWORK_DOT", "copy_framework_source_to_figure_library", "render_all_final_figures"]

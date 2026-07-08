#!/usr/bin/env python3
"""Render publication-oriented figures from plot_mimo_results.jl summaries."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mimopotts")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


SUITE_TITLES = {
    "core": "Core Rayleigh benchmarks",
    "certml": "Small-system benchmarks",
    "lowber": "Low-BER benchmarks",
    "robustness": "Robustness benchmarks",
    "scaling": "Scaling benchmarks",
    "other": "Other benchmarks",
}
SUITE_COLORS = {
    "core": "#4477AA",
    "certml": "#66CCEE",
    "lowber": "#228833",
    "robustness": "#CC6677",
    "scaling": "#AA3377",
    "other": "#777777",
}
METHODS = [
    ("pottsBer", "MIMO Potts", "#0072B2", "o", "-"),
    ("mmseBer", "MMSE", "#D55E00", "s", "--"),
    ("zfBer", "ZF", "#777777", "^", ":"),
]


def pretty_scenario(name: str) -> str:
    prefixes = ("scale_overloaded_", "robust_overloaded_", "robust_", "lowber_", "certml_", "core_", "scale_")
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    replacements = {
        "rayleigh": "Rayleigh",
        "corr03": r"corr. $\rho=0.3$",
        "corr07": r"corr. $\rho=0.7$",
        "corr09": r"corr. $\rho=0.9$",
        "ricianK10": "Rician K=10",
        "illcond100": r"$\kappa=100$",
        "qpsk": "QPSK",
        "qam": "-QAM",
        "Rx_": " Rx, ",
        "Users_": " users, ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("_", ", ")


def positive_rate(values: np.ndarray, denominators: np.ndarray) -> np.ndarray:
    floor = 0.5 / np.maximum(denominators, 1)
    return np.maximum(values, floor)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_suite_curves(summary: pd.DataFrame, suite: str, outdir: Path) -> None:
    data = summary[summary.suite == suite]
    scenarios = sorted(data.scenario.unique())
    if not scenarios:
        return
    ncols = min(3, len(scenarios))
    nrows = math.ceil(len(scenarios) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.25 * ncols, 3.35 * nrows), squeeze=False)
    legend_handles = None
    for ax, scenario in zip(axes.flat, scenarios):
        group = data[data.scenario == scenario].sort_values("ebnodb")
        x = group.ebnodb.to_numpy()
        for column, label, color, marker, linestyle in METHODS:
            denominator = group.pottsBits if column == "pottsBer" else group.baselineBits
            raw = group[column].to_numpy()
            y = positive_rate(raw, denominator.to_numpy())
            ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=4.5)
            zero = raw == 0
            if zero.any():
                ax.scatter(x[zero], y[zero], facecolors="none", edgecolors=color, marker=marker, s=42, zorder=5)
            if column == "pottsBer":
                se = group.pottsBerSE.to_numpy()
                lower = positive_rate(np.maximum(raw - 1.96 * se, 0), denominator.to_numpy())
                upper = positive_rate(raw + 1.96 * se, denominator.to_numpy())
                ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
        ax.set_yscale("log")
        ax.set_title(pretty_scenario(scenario), fontsize=10)
        ax.set_xlabel(r"$E_b/N_0$ (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.22)
        legend_handles = ax.get_legend_handles_labels()
    for ax in axes.flat[len(scenarios) :]:
        ax.remove()
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(SUITE_TITLES.get(suite, suite), y=1.045, fontsize=14)
    fig.text(0.5, -0.005, "Open markers denote zero observed errors and are shown at 0.5 / tested bits.", ha="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, outdir / f"ber_{suite}")


def plot_overview(summary: pd.DataFrame, outdir: Path) -> None:
    records = []
    for scenario, group in summary.groupby("scenario", sort=True):
        group = group.sort_values("ebnodb")
        for rank, row in enumerate(group.itertuples(), start=1):
            # Use one common detection floor so unequal Potts/baseline trial counts
            # do not manufacture an apparent gain when both observed zero errors.
            floor = 0.5 / max(row.baselineBits, 1)
            potts = max(row.pottsBer, floor)
            mmse = max(row.mmseBer, floor)
            records.append((scenario, row.suite, rank, row.ebnodb, math.log10(mmse / potts)))
    frame = pd.DataFrame(records, columns=["scenario", "suite", "rank", "ebnodb", "gain"])
    ordered = []
    for suite in ("core", "certml", "lowber", "robustness", "scaling", "other"):
        ordered.extend(sorted(frame.loc[frame.suite == suite, "scenario"].unique()))
    max_rank = int(frame["rank"].max())
    matrix = np.full((len(ordered), max_rank), np.nan)
    annotations = np.full(matrix.shape, "", dtype=object)
    for i, scenario in enumerate(ordered):
        for row in frame[frame.scenario == scenario].itertuples():
            matrix[i, row.rank - 1] = row.gain
            annotations[i, row.rank - 1] = f"{row.ebnodb:g} dB"
    bound = max(0.5, np.nanpercentile(np.abs(matrix), 95))
    fig, ax = plt.subplots(figsize=(8.0, max(8.0, 0.31 * len(ordered))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set_xticks(range(max_rank), range(1, max_rank + 1))
    ax.set_xlabel("SNR point (actual Eb/N0 shown in cells)")
    ax.set_yticks(range(len(ordered)), [pretty_scenario(x) for x in ordered], fontsize=8)
    for i in range(len(ordered)):
        for j in range(max_rank):
            if annotations[i, j]:
                ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=6.5, color="black")
    ax.set_title(r"BER improvement over MMSE: $\log_{10}(\mathrm{BER}_{MMSE}/\mathrm{BER}_{Potts})$")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("decades (blue = MIMO Potts better)")
    fig.tight_layout()
    save_figure(fig, outdir / "overview_mmse_ber_gain")


def plot_scaling(backend: pd.DataFrame, outdir: Path) -> None:
    data = backend[backend.suite == "scaling"]
    if data.empty:
        return
    reduced = data.groupby(["scenario", "backend", "nt", "nr"], as_index=False).agg(
        time=("medianSteadyTime", "median"), throughput=("medianStepsPerSecond", "median"), n=("nRuns", "sum")
    )
    scenario_order = sorted(reduced.scenario.unique(), key=lambda s: (reduced.loc[reduced.scenario == s, "nt"].iloc[0], s))
    labels = [pretty_scenario(s) for s in scenario_order]
    y = np.arange(len(scenario_order))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    styles = {"cpu": ("#D55E00", "s"), "cuda": ("#0072B2", "o"), "unknown": ("#777777", "^")}
    for backend_name, group in reduced.groupby("backend"):
        color, marker = styles.get(backend_name, styles["unknown"])
        lookup = group.set_index("scenario")
        positions = [i for i, scenario in enumerate(scenario_order) if scenario in lookup.index]
        axes[0].scatter([lookup.loc[scenario, "time"] for scenario in scenario_order if scenario in lookup.index], positions,
                        color=color, marker=marker, s=48, label=backend_name.upper(), zorder=3)
        axes[1].scatter([lookup.loc[scenario, "throughput"] for scenario in scenario_order if scenario in lookup.index], positions,
                        color=color, marker=marker, s=48, label=backend_name.upper(), zorder=3)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yticks(y, labels, fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_xlabel("Steady-state solve time (s)")
    axes[0].set_title("Runtime scaling")
    axes[1].set_xlabel("Potts steps / second")
    axes[1].set_title("Update throughput")
    fig.suptitle("Scaling suite (median across instances and SNRs)")
    fig.text(0.5, -0.01, "The first timed trial of each job is excluded to remove compilation and GPU initialization.", ha="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, outdir / "scaling_runtime")


def plot_diagnostics(scenarios: pd.DataFrame, outdir: Path) -> None:
    data = scenarios.sort_values(["suite", "nt", "scenario"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for suite, group in data.groupby("suite"):
        color = SUITE_COLORS.get(suite, "#777777")
        x = np.maximum(2 * group.nt.to_numpy(), 1)
        label = SUITE_TITLES.get(suite, suite).replace(" benchmarks", "")
        axes[0].scatter(x, group.medianSteadyTime, color=color, label=label, alpha=0.85)
        axes[1].scatter(x, group.medianDistanceImprovement, color=color, label=label, alpha=0.85)
        axes[2].scatter(x, group.initialBestFraction, color=color, label=label, alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median steady-state time (s)")
    axes[1].set_ylabel("Median relative distance improvement")
    axes[2].set_ylabel("Fraction retaining initial solution")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Real transmit dimensions (2 Nt)")
        ax.grid(True, alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Search and runtime diagnostics across all test scenarios")
    fig.tight_layout()
    save_figure(fig, outdir / "search_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary, sep="\t")
    backend = pd.read_csv(args.backend, sep="\t")
    scenarios = pd.read_csv(args.scenarios, sep="\t")

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.facecolor": "white",
    })
    for suite in summary.suite.unique():
        plot_suite_curves(summary, suite, outdir)
    plot_overview(summary, outdir)
    plot_scaling(backend, outdir)
    plot_diagnostics(scenarios, outdir)
    print(f"Rendered figures in {outdir}")


if __name__ == "__main__":
    main()

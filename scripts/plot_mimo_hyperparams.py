#!/usr/bin/env python3
"""Plot BER sensitivity and tuning regret for the MIMO Potts HP sweep."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mimopotts")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIZE_COLORS = {"small": "#66CCEE", "medium": "#4477AA", "large": "#AA3377"}


def pretty_scenario(name: str) -> str:
    for prefix in ("scale_overloaded_", "robust_overloaded_", "robust_", "lowber_", "certml_", "core_", "scale_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    replacements = {
        "rayleigh": "Rayleigh", "corr03": r"corr. $\rho=0.3$",
        "corr07": r"corr. $\rho=0.7$", "corr09": r"corr. $\rho=0.9$",
        "ricianK10": "Rician K=10", "illcond100": r"$\kappa=100$",
        "qpsk": "QPSK", "qam": "-QAM", "Rx_": " Rx, ", "Users_": " users, ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("_", ", ")


def scenario_label(name: str) -> str:
    if name.startswith("certml_"):
        suite = "Small"
    elif name.startswith("core_"):
        suite = "Core"
    elif name.startswith("lowber_"):
        suite = "Low-BER"
    elif name.startswith("robust"):
        suite = "Robust"
    elif name.startswith("scale"):
        suite = "Scale"
    else:
        suite = "Other"
    return f"{suite} — {pretty_scenario(name)}"


def selected_row(selected: pd.DataFrame, size_class: str) -> pd.Series:
    rows = selected[selected.sizeClass == size_class]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected row for sizeClass={size_class}, found {len(rows)}")
    return rows.iloc[0]


def is_selected(frame: pd.DataFrame, selected: pd.Series) -> pd.Series:
    return (
        np.isclose(frame.noiseRatio, selected.noiseRatio)
        & np.isclose(frame.cyclesScaler, selected.cyclesScaler)
        & (frame.freeDims == int(selected.freeDims))
    )


def minimum_profile(frame: pd.DataFrame, parameter: str, metric: str) -> pd.DataFrame:
    """For each parameter value, retain the row minimizing metric over all other HPs."""
    indices = frame.groupby(parameter, sort=True)[metric].idxmin()
    return frame.loc[indices].sort_values(parameter)


def plot_profile(ax: plt.Axes, frame: pd.DataFrame, parameter: str, metric: str,
                 selected: pd.Series, ylabel: str, use_log: bool, show_error: bool) -> None:
    profile = minimum_profile(frame, parameter, metric)
    x = profile[parameter].to_numpy()
    y = profile[metric].to_numpy()
    ax.plot(x, y, color="#0072B2", marker="o", linewidth=1.8, label="profile minimum")
    if show_error and "berSE" in profile:
        ax.errorbar(x, y, yerr=1.96 * profile.berSE.to_numpy(), fmt="none",
                    ecolor="#0072B2", alpha=0.45, capsize=2, linewidth=0.8)

    selected_x = float(selected[parameter])
    ax.axvline(selected_x, color="#D55E00", linestyle="--", linewidth=1.2,
               label="selected value")
    chosen = frame[is_selected(frame, selected)]
    if not chosen.empty:
        chosen_y = float(chosen.iloc[0][metric])
        ax.scatter(selected_x, chosen_y, marker="*", s=180, facecolor="none",
                   edgecolor="#D55E00", linewidth=1.8, zorder=5,
                   label="selected configuration")

    best = frame.loc[frame[metric].idxmin()]
    ax.scatter(float(best[parameter]), float(best[metric]), marker="X", s=75,
               color="#009E73", zorder=5, label="global minimum")
    if use_log and np.all(y > 0):
        ax.set_yscale("log")
    ax.set_xlabel(parameter)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Minimize over other HPs; fix {parameter}")
    ax.grid(True, which="both", alpha=0.22)


def plot_scenario(frame: pd.DataFrame, selected: pd.Series, outdir: Path) -> None:
    scenario = frame.scenario.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), squeeze=False)
    for ax, parameter in zip(axes.flat, ("noiseRatio", "cyclesScaler", "freeDims")):
        plot_profile(ax, frame, parameter, "meanBer", selected, "Mean BER", True, True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.91), fontsize=8)
    fig.suptitle(f"Hyperparameter sensitivity: {pretty_scenario(scenario)}", fontsize=14, y=0.98)
    fig.text(0.5, 0.01, "At each x value, BER is minimized over the other two hyperparameters; bars show 95% normal intervals.",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.07, right=0.98, wspace=0.30)
    fig.savefig(outdir / f"hp_{scenario}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"hp_{scenario}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def normalized_size_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    best = data.groupby("scenario").meanBer.transform("min")
    positive = data.loc[data.meanBer > 0, "meanBer"]
    floor = max(float(positive.min()) / 2 if len(positive) else 1e-12, 1e-12)
    data["relativeBer"] = np.maximum(data.meanBer, floor) / np.maximum(best, floor)
    keys = ["noiseRatio", "cyclesScaler", "freeDims"]
    return data.groupby(keys, as_index=False).agg(
        relativeBer=("relativeBer", "mean"),
        meanBer=("meanBer", "mean"),
        medianCudaTime=("medianCudaTime", "median"),
        scenarios=("scenario", "nunique"),
    )


def plot_size_class(frame: pd.DataFrame, selected: pd.Series, outdir: Path) -> pd.DataFrame:
    size_class = frame.sizeClass.iloc[0]
    aggregate = normalized_size_data(frame)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), squeeze=False)
    for ax, parameter in zip(axes.flat, ("noiseRatio", "cyclesScaler", "freeDims")):
        plot_profile(ax, aggregate, parameter, "relativeBer", selected,
                     "Mean BER / scenario-best BER", False, False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.91), fontsize=8)
    fig.suptitle(f"{size_class.capitalize()}-class hyperparameter sensitivity", fontsize=14, y=0.98)
    fig.text(0.5, 0.01, "At each x value, normalized BER is minimized over the other two hyperparameters.", ha="center", fontsize=8)
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.07, right=0.98, wspace=0.30)
    fig.savefig(outdir / f"hp_size_{size_class}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"hp_size_{size_class}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    aggregate.insert(0, "sizeClass", size_class)
    return aggregate


def selection_diagnostics(summary: pd.DataFrame, selected: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    records = []
    for scenario, frame in summary.groupby("scenario"):
        size_class = frame.sizeClass.iloc[0]
        choice = selected_row(selected, size_class)
        chosen = frame[is_selected(frame, choice)]
        if chosen.empty:
            continue
        chosen = chosen.iloc[0]
        best = frame.loc[frame.meanBer.idxmin()]
        floor = max(frame.loc[frame.meanBer > 0, "meanBer"].min() / 2, 1e-12) if (frame.meanBer > 0).any() else 1e-12
        same_hp = (
            np.isclose(chosen.noiseRatio, best.noiseRatio)
            and np.isclose(chosen.cyclesScaler, best.cyclesScaler)
            and int(chosen.freeDims) == int(best.freeDims)
        )
        combined_se = 0.0 if same_hp else math.sqrt(chosen.berSE**2 + best.berSE**2)
        records.append({
            "sizeClass": size_class, "scenario": scenario,
            "selectedBer": chosen.meanBer, "bestBer": best.meanBer,
            "selectedToBestBer": max(chosen.meanBer, floor) / max(best.meanBer, floor),
            "ratioApproxSE": combined_se / max(best.meanBer, floor),
            "berDifferenceZ": (chosen.meanBer - best.meanBer) / combined_se if combined_se > 0 else 0.0,
            "selectedNoiseRatio": choice.noiseRatio, "selectedCyclesScaler": choice.cyclesScaler,
            "selectedFreeDims": int(choice.freeDims),
            "bestNoiseRatio": best.noiseRatio, "bestCyclesScaler": best.cyclesScaler,
            "bestFreeDims": int(best.freeDims),
        })
    diagnostics = pd.DataFrame(records).sort_values("selectedToBestBer")
    fig, ax = plt.subplots(figsize=(8.5, max(7.0, 0.27 * len(diagnostics))))
    y = np.arange(len(diagnostics))
    colors = [SIZE_COLORS.get(x, "#777777") for x in diagnostics.sizeClass]
    ax.barh(y, diagnostics.selectedToBestBer, color=colors)
    ax.errorbar(diagnostics.selectedToBestBer, y, xerr=diagnostics.ratioApproxSE,
                fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax.axvline(1, color="black", linewidth=1)
    ax.set_yticks(y, [scenario_label(x) for x in diagnostics.scenario], fontsize=8)
    ax.set_xlabel("Selected size-class BER / scenario-best grid BER")
    ax.set_title("Hyperparameter selection regret by scenario")
    ax.grid(True, axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(outdir / "hp_selection_regret.pdf", bbox_inches="tight")
    fig.savefig(outdir / "hp_selection_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    scenario_dir = outdir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary, sep="\t")
    selected = pd.read_csv(args.selected, sep="\t")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120, "savefig.facecolor": "white"})
    for _, frame in summary.groupby("scenario", sort=True):
        plot_scenario(frame, selected_row(selected, frame.sizeClass.iloc[0]), scenario_dir)
    aggregates = []
    for size_class, frame in summary.groupby("sizeClass", sort=True):
        aggregates.append(plot_size_class(frame, selected_row(selected, size_class), outdir))
    pd.concat(aggregates, ignore_index=True).to_csv(outdir / "size_class_summary.tsv", sep="\t", index=False)
    diagnostics = selection_diagnostics(summary, selected, outdir)
    diagnostics.to_csv(outdir / "selection_diagnostics.tsv", sep="\t", index=False)
    print(f"Rendered hyperparameter figures in {outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot every current-energy trial separately for the Potts trace experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mimopotts")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.points, sep="\t")

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "savefig.facecolor": "white",
    })
    cmap = plt.get_cmap("turbo", 32)

    group_columns = ["scenario", "ebnodb", "snrIndex", "optimizer", "noiseRatio"]
    figure_count = 0
    for key, group in points.groupby(group_columns, sort=True):
        scenario, ebnodb, snr_index, optimizer, noise_ratio = key
        instances = sorted(group.instance.unique())
        fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), squeeze=False)
        for ax, instance in zip(axes.flat, instances):
            instance_data = group[group.instance == instance]
            for trial, trace in instance_data.groupby("trial", sort=True):
                ax.plot(trace.step, trace.energy, color=cmap(int(trial) - 1),
                        linewidth=0.75, alpha=0.72)
            ax.set_title(f"Instance {instance}")
            ax.set_xlabel("Step")
            ax.set_ylabel(r"Current objective $||y-Hx||^2$")
            ax.grid(True, alpha=0.18)
        for ax in axes.flat[len(instances):]:
            ax.remove()

        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 32))
        scalar_map.set_array([])
        fig.colorbar(scalar_map, ax=list(axes.flat[:len(instances)]), label="Trial", pad=0.015, fraction=0.025)
        fig.suptitle(
            f"{scenario} | Eb/N0={ebnodb:g} dB | {optimizer} | noiseRatio={noise_ratio:g}",
            fontsize=13,
        )
        fig.subplots_adjust(top=0.91, bottom=0.08, left=0.07, right=0.91, wspace=0.30, hspace=0.34)
        stem = f"trace_{scenario}_snr{int(snr_index)}_{optimizer}_noise{safe_name(noise_ratio)}"
        fig.savefig(outdir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        figure_count += 1

    print(f"Rendered {figure_count} per-trial energy figures in {outdir}")


if __name__ == "__main__":
    main()

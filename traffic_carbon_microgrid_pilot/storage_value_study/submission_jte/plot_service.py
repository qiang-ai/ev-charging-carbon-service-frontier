"""Generate print-ready figures from the paired two-year charging experiments."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT.parents[1] / ".deps"
if LOCAL.is_dir():
    sys.path.insert(0, str(LOCAL))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analyze_service import HOLDOUT, read
from figure_style import use_editable_times


OUT = ROOT / "figures"
YEARS = (2023, 2024)
PALETTE = ("#343434", "#237894", "#ae531c")


def save(fig, number):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"Fig_{number}.tif", dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"Fig_{number}.png", dpi=210)
    fig.savefig(OUT / f"Fig_{number}.svg")
    plt.close(fig)


def paired_windows():
    rows = read(HOLDOUT / "analysis_common_pv" / "matched_interaction_changes.csv")
    primary = [row for row in rows if float(row["pv_scale"]) == .2 and float(row["battery_fraction"]) == .5]
    x = np.array([-float(row["interaction_g_per_kwh_2023"]) for row in primary])
    y = np.array([-float(row["interaction_g_per_kwh_2024"]) for row in primary])
    lims = [min(-5, min(x), min(y)) - 2, max(max(x), max(y)) + 4]
    fig, ax = plt.subplots(figsize=(5.4, 4.9), layout="constrained")
    ax.plot(lims, lims, color="0.65", lw=.9, ls="--", label="1:1")
    ax.axhline(1, lw=.75, color="0.55", ls=":")
    ax.axvline(1, lw=.75, color="0.55", ls=":")
    for month, marker, color, name in ((1, "o", PALETTE[1], "January"), (7, "^", PALETTE[2], "July")):
        mask = np.array([int(row["month"]) == month for row in primary])
        ax.scatter(x[mask], y[mask], s=28, color=color, marker=marker, label=name, alpha=.78)
    ax.set(xlim=lims, ylim=lims, xlabel="2023 decrease (g CO₂/kWh)",
           ylabel="2024 decrease (g CO₂/kWh)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    save(fig, 8)


def readiness_frontier(summary):
    fig, ax = plt.subplots(figsize=(7.1, 4.25), layout="constrained")
    targets = (.5, .75, .9, 1.)
    x = [100 * ratio for ratio in targets]
    storage = []
    for year, color, marker in zip(YEARS, PALETTE[1:], ("o", "^"), strict=True):
        for minute, style in ((30, "-"), (60, "--")):
            half = summary["summaries"][f"{year}|main|{minute}|0.2|0.5"]
            points = [half["shortfall_no_battery_pct"], *[
                summary["frontier"][f"{year}|{minute}|0.2|{ratio}|0.0"]["pooled_avoidable_shortfall_pct"]
                for ratio in targets[1:]]]
            ax.plot(x, points, color=color, marker=marker, lw=1.22, ls=style, markersize=4.5,
                    label=f"No battery, {year}, {minute} min")
            for f in (.125, .5):
                at_half = summary["summaries"][f"{year}|main|{minute}|0.2|{f}"]["shortfall_battery_pct"]
                storage.append([at_half, *[
                    summary["frontier"][f"{year}|{minute}|0.2|{ratio}|{f}"]["pooled_avoidable_shortfall_pct"]
                    for ratio in targets[1:]]])
    ax.plot(x, np.max(storage, axis=0), color=PALETTE[0], marker="s", lw=1.25, ls=":",
            markersize=4.5, label="Maximum with storage")
    ax.legend(ncol=2, loc="upper left", columnspacing=1.4, handletextpad=.5, fontsize=10.1)
    ax.set(xticks=[50, 60, 70, 80, 90, 100], xlabel="Targeted share of no-battery reduction (%)",
           ylabel="Avoidable early shortfall (% of requested energy)")
    ax.grid(axis="y", linewidth=.5, color=".87")
    ax.set_axisbelow(True)
    save(fig, 10)


def carbon_by_pv(summary):
    fig, ax = plt.subplots(figsize=(7.1, 3.7), layout="constrained")
    groups = ((.125, "12.5 kWh", PALETTE[1], "s"), (.5, "50 kWh", PALETTE[2], "^"))
    for k, year in enumerate(YEARS):
        for i, (f, _, color, marker) in enumerate(groups):
            dist = [summary["summaries"][f"{year}|main|30|{p}|{f}"]["readiness_floor_carbon_gain_g_per_kwh"]
                    for p in (0., .2, .4)]
            medians = np.array([x["median"] for x in dist])
            error = [medians - [x["q25"] for x in dist], [x["q75"] for x in dist] - medians]
            ax.errorbar(np.arange(3) + (2 * i + k - 1.5) * .135, medians, yerr=error, marker=marker,
                        lw=1.2, capsize=3, ls="none", color=color, markersize=5.5,
                        markerfacecolor="white" if year == 2023 else color)
    keys = [Line2D([], [], color=color, ls="none", marker=mark,
                   markerfacecolor="white" if year == 2023 else color, label=f"{label}, {year}")
            for _, label, color, mark in groups for year in YEARS]
    ax.legend(handles=keys, ncol=2, loc="upper left", fontsize=10.5)
    ax.set(xticks=range(3), xticklabels=["0", "20", "40"], xlim=(-.36, 2.38),
           xlabel="PV capacity (kWp per 100 kWh/day)", ylabel="Storage carbon benefit (g CO₂/kWh)")
    ax.grid(axis="y", linewidth=.5, color=".87")
    ax.set_axisbelow(True)
    save(fig, 9)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white",
                         "legend.framealpha": 1, "legend.edgecolor": "none"})
    use_editable_times()
    data = json.loads((ROOT / "results" / "service_summary.json").read_text(encoding="utf-8"))
    paired_windows()
    readiness_frontier(data)
    carbon_by_pv(data)

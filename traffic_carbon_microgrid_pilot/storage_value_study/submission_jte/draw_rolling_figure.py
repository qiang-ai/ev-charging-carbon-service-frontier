"""Draw the rolling controller timeline."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT.parents[1] / ".deps"
if LOCAL.is_dir():
    sys.path.insert(0, str(LOCAL))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from figure_style import use_editable_times


INK = "#22292e"
EXECUTE = "#2b5766"
PLAN = "#e4eaec"
LINE = "#858e91"


def arrow(ax, start, end, *, style="-", color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 shrinkA=0, shrinkB=0, linewidth=.85,
                                 linestyle=style, color=color))


def setup(ax, title):
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.set_axis_off()
    ax.text(0, 99, title, va="top", fontsize=12.2, fontweight="bold")


def car(ax, x, y):
    outline = [(x, y), (x, y + 2.2), (x + 1.25, y + 2.2), (x + 2.05, y + 4.7),
               (x + 4.95, y + 4.7), (x + 5.85, y + 2.2), (x + 6.75, y + 2.2),
               (x + 6.75, y)]
    ax.add_patch(Polygon(outline, closed=True, facecolor="white", edgecolor=INK, lw=.85))
    ax.scatter([x + 1.4, x + 5.35], [y, y], s=10, c=INK, linewidths=0, zorder=3)


def battery(ax, x, y):
    ax.add_patch(Rectangle((x, y), 4.6, 4.6, facecolor="white", edgecolor=INK, lw=.85))
    ax.add_patch(Rectangle((x + 4.6, y + 1.4), .55, 1.8, facecolor=INK, edgecolor="none"))
    ax.add_patch(Rectangle((x + .45, y + .62), 2, 3.35, facecolor=EXECUTE, edgecolor="none"))


def rolling_timeline(ax):
    setup(ax, "Receding-horizon dispatch")
    t0, hour = 23.0, 2.85
    step = 2 * hour
    first_end = t0 + 24 * hour

    for x, color, label in ((58, EXECUTE, "Execute 2 h"), (79, PLAN, "Planned")):
        ax.add_patch(Rectangle((x, 91), 4.4, 5, facecolor=color, edgecolor=LINE, lw=.55))
        ax.text(x + 5.5, 93.5, label, fontsize=10.2, va="center")

    ax.hlines(76, t0, first_end, color=LINE, lw=.9)
    ax.vlines([t0, first_end], 72.5, 79.5, color=LINE, lw=.9)
    ax.text((t0 + first_end) / 2, 81.5, "24 h planning horizon", ha="center", fontsize=10.8)

    for start, y, label in ((t0, 60, "First decision"), (t0 + step, 22, "Second decision")):
        ax.text(0, y + 4, label, va="center", fontsize=10.8)
        ax.add_patch(Rectangle((start, y), 24 * hour, 8, facecolor=PLAN, edgecolor=LINE, lw=.7))
        ax.add_patch(Rectangle((start, y), step, 8, facecolor=EXECUTE, edgecolor=EXECUTE, lw=.7))
        for mark in range(4, 24, 2):
            ax.vlines(start + mark * hour, y + 1.1, y + 6.9, color="#bdc6ca", lw=.55)

    car(ax, 0.5, 43.7)
    ax.text(8.8, 45.9, "New plug-in\nCharge now", fontsize=10.0, va="center", linespacing=1.12)
    arrow(ax, (21.4, 46.5), (t0 + step * .48, 59.1))
    arrow(ax, (t0 + step, 58.9), (t0 + step, 31.4), style="--", color=LINE)
    battery(ax, 33, 42.8)
    ax.text(39.8, 45.1, "Carry battery state and delivered energy", fontsize=10.3, va="center")

    ax.hlines(13, t0, t0 + 26 * hour, color=LINE, lw=.7)
    ax.vlines([t0, t0 + step, first_end, first_end + step], 10.6, 15.4, color=LINE, lw=.7)
    for x, label, align in ((t0, "t", "center"), (t0 + step, "t + 2 h", "left"),
                            (first_end, "t + 24 h", "right")):
        ax.text(x, 8.4, label, va="top", ha=align, fontsize=10.2)


def save_figure(output_dir, number, height, draw):
    fig = plt.figure(figsize=(7.15, height))
    draw(fig.add_axes((.045, .07, .91, .89)))
    output = output_dir / f"Fig_{number}"
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".tif"), dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"text.color": INK, "savefig.facecolor": "white"})
    use_editable_times()
    save_figure(output_dir, 7, 2.3, rolling_timeline)


if __name__ == "__main__":
    main()

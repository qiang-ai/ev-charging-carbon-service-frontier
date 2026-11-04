"""Render the cross-year cohort and evaluation diagrams."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
LOCAL_DEPS = HERE.parents[1] / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from figure_style import use_editable_times


WIDTH = 7.0
INK = "#202020"
BODY = "#2c2c2c"
EDGE = "#545454"
LEFT, RIGHT, BOX_WIDTH = 0.29, 3.96, 2.75
LEFT_CENTER, RIGHT_CENTER = LEFT + BOX_WIDTH / 2, RIGHT + BOX_WIDTH / 2


class Diagram:
    def __init__(self, height):
        use_editable_times()
        self.figure = plt.figure(figsize=(WIDTH, height), dpi=160, facecolor="white")
        self.ax = self.figure.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(0, height))
        self.ax.set_axis_off()
        self.boxes = []

    def label(self, x, y, text, *, bold=False, align="center", size=11.2, **kwargs):
        return self.ax.text(
            x, y, text, ha=align, va="center", color=INK if bold else BODY,
            fontsize=size, fontweight="bold" if bold else "normal", zorder=4, **kwargs,
        )

    def box(self, x, y, width, height, lines, *, face="white", edge=EDGE, text_x=None):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=face, edgecolor=edge, linewidth=1.05, zorder=2,
        )
        self.ax.add_patch(patch)
        texts = [
            self.label(
                x + width / 2 if text_x is None else text_x, y + height - offset,
                text, bold=bold, size=size, align="center" if text_x is None else "left",
            )
            for offset, text, bold, size in lines
        ]
        self.boxes.append((patch, texts))

    def arrow(self, x0, y0, x1, y1):
        self.ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
            linewidth=1.15, color=EDGE, shrinkA=0, shrinkB=0, zorder=1,
        ))

    def line(self, *points, zorder=3):
        x, y = zip(*points)
        self.ax.plot(x, y, color=EDGE, lw=1.05, solid_capstyle="round", zorder=zorder)

    def rect(self, x, y, width, height, *, fill=False):
        self.ax.add_patch(Rectangle(
            (x, y), width, height, linewidth=1.05, edgecolor=EDGE,
            facecolor="white" if fill else "none", zorder=3,
        ))

    def circle(self, x, y, radius):
        self.ax.add_patch(Circle((x, y), radius, edgecolor=EDGE, facecolor="none", lw=1.05, zorder=3))

    def calendar(self, x, y):
        self.rect(x - .11, y - .09, .22, .18, fill=True)
        self.line((x - .11, y + .03), (x + .11, y + .03))
        for dx in (-.055, .055):
            self.line((x + dx, y + .06), (x + dx, y + .12))

    def checked_figure(self):
        self.figure.canvas.draw()
        renderer = self.figure.canvas.get_renderer()
        figure_bounds = self.figure.bbox.padded(-5)
        text_extents = []
        for text in self.ax.texts:
            extent = text.get_window_extent(renderer)
            if not (figure_bounds.contains(extent.x0, extent.y0) and figure_bounds.contains(extent.x1, extent.y1)):
                raise ValueError(f"Figure label exceeds the canvas: {text.get_text()}")
            if any(extent.padded(2).overlaps(previous) for previous in text_extents):
                raise ValueError(f"Figure labels overlap near: {text.get_text()}")
            text_extents.append(extent)
        for patch, texts in self.boxes:
            bounds = patch.get_window_extent(renderer).padded(-5)
            for text in texts:
                extent = text.get_window_extent(renderer)
                if not (bounds.contains(extent.x0, extent.y0) and bounds.contains(extent.x1, extent.y1)):
                    raise ValueError(f"Figure label exceeds its box: {text.get_text()}")
        return self.figure


def draw_cohort():
    diagram = Diagram(2.70)
    diagram.label(3.5, 2.47, "Edinburgh and Glasgow · January and July", bold=True, size=12)

    inputs = (0.93, 2.64, 4.36, 5.99)
    y = 2.02
    x = inputs[0]
    diagram.rect(x - .12, y - .16, .24, .32, fill=True)
    diagram.rect(x - .07, y + .12, .14, .065, fill=True)
    for dy in (-.08, -.01, .06):
        diagram.line((x - .065, y + dy), (x + .065, y + dy))

    x = inputs[1]
    diagram.line((x - .16, y + .12), (x - .16, y - .13), (x + .16, y - .13))
    diagram.line((x - .12, y - .07), (x - .045, y + .035), (x + .025, y - .015), (x + .145, y + .13))
    diagram.line((x + .08, y + .13), (x + .145, y + .13), (x + .145, y + .065))

    x = inputs[2]
    diagram.label(
        x, y, "CO₂", bold=True, size=15,
        bbox={"boxstyle": "round,pad=.24,rounding_size=.45", "facecolor": "white", "edgecolor": EDGE},
    )

    x = inputs[3]
    diagram.line((x - .17, y - .13), (x + .09, y - .13), (x + .03, y + .02), (x - .11, y + .02), (x - .17, y - .13))
    diagram.line((x - .14, y - .055), (x + .06, y - .055))
    diagram.line((x - .04, y - .13), (x - .04, y + .02))
    diagram.circle(x + .12, y + .08, .058)
    diagram.line((x + .12, y + .16), (x + .12, y + .19))
    diagram.line((x + .20, y + .08), (x + .23, y + .08))

    for x, name in zip(inputs, ("Sessions and chargers", "Elexon price", "NESO carbon", "Common ERA5 PV model")):
        diagram.label(x, 1.72, name, size=10.7)
        diagram.line((x, 1.60), (x, 1.50), zorder=1)
    diagram.line((inputs[0], 1.50), (inputs[-1], 1.50), zorder=1)
    diagram.arrow(LEFT_CENTER, 1.50, LEFT_CENTER, 1.25)
    diagram.arrow(RIGHT_CENTER, 1.50, RIGHT_CENTER, 1.25)

    diagram.box(LEFT, .19, BOX_WIDTH, 1.02, [
        (.22, "2023 anchor", True, 12),
        (.49, "12 sites · PV and battery sizing", False, 11.2),
        (.77, "82 site-weeks · 3,506 sessions", True, 11.2),
    ])
    diagram.box(RIGHT, .19, BOX_WIDTH, 1.02, [
        (.22, "2024 holdout", True, 12),
        (.49, "Same sites and sizes", False, 11.2),
        (.77, "81 site-weeks · 2,956 sessions", True, 11.2),
    ], face="#f1f1f1")
    diagram.label(3.5, .85, "Freeze", bold=True, size=10.6)
    diagram.arrow(LEFT + BOX_WIDTH + .05, .68, RIGHT - .05, .68)
    return diagram.checked_figure()


def draw_evaluation():
    diagram = Diagram(2.45)
    diagram.box(1.75, 1.58, 3.50, .69, [
        (.20, "70 matched site-week pairs", True, 12),
        (.47, "Site · month · complete UTC week rank", False, 11.0),
    ], edge=INK, text_x=2.49)
    diagram.calendar(2.07, 1.94)
    diagram.calendar(2.27, 1.85)

    diagram.line((3.50, 1.55), (3.50, 1.36), zorder=1)
    diagram.line((LEFT_CENTER, 1.36), (RIGHT_CENTER, 1.36), zorder=1)
    diagram.arrow(LEFT_CENTER, 1.36, LEFT_CENTER, 1.12)
    diagram.arrow(RIGHT_CENTER, 1.36, RIGHT_CENTER, 1.12)

    diagram.box(LEFT, .18, BOX_WIDTH, .91, [
        (.22, "Offline · 2023 and 2024", True, 12),
        (.49, "Fixed service or fixed carbon", False, 11.2),
        (.72, "Storage comparisons", False, 11.2),
    ], text_x=.93)
    diagram.rect(.47, .53, .30, .18)
    diagram.rect(.77, .575, .035, .09)
    diagram.line((.64, .68), (.58, .61), (.65, .61), (.60, .56))

    diagram.box(RIGHT, .18, BOX_WIDTH, .91, [
        (.22, "Rolling dispatch · 2024", True, 12),
        (.49, "2023-trained forecasts", False, 11.2),
        (.72, "Replan every 2 h", False, 11.2),
    ], face="#f1f1f1", text_x=4.60)
    diagram.circle(4.29, .625, .13)
    diagram.line((4.29, .70), (4.29, .625), (4.36, .59))
    return diagram.checked_figure()


def save_figure(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    tiff = base.with_suffix(".tif")
    fig.savefig(tiff, dpi=1200, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(tiff) as src:
        grayscale = src.convert("L")
    grayscale.save(tiff, compression="tiff_lzw", dpi=(1200, 1200))
    grayscale.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(draw_cohort(), args.output_dir / "Fig_1")
    save_figure(draw_evaluation(), args.output_dir / "Fig_2")


if __name__ == "__main__":
    main()

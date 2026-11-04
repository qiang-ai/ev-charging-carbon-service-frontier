"""Draw the station model and the mechanisms tested by the offline comparisons."""
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
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np

from figure_style import use_editable_times

OUT = ROOT / "figures"
INK = "#20272d"
BLUE = "#24586b"
TEAL = "#dfedf0"
ORANGE = "#98440f"
WARM = "#f4dfce"
PALE = "#f4f5f5"
MID = "#849198"


def pictogram(ax, kind, x, y, size, color=INK):
    drawing = DrawingArea(size, size, 0, 0)
    scale = lambda value: value * size
    width = 1.15 if size >= 25 else .95

    def line(points):
        u, v = zip(*points, strict=True)
        drawing.add_artist(Line2D([scale(a) for a in u], [scale(b) for b in v],
                                  color=color, lw=width, solid_capstyle="round", solid_joinstyle="round"))

    def circle(x0, y0, radius):
        drawing.add_artist(Circle((scale(x0), scale(y0)), scale(radius), facecolor="white", edgecolor=color, lw=width))

    def polygon(points, fill="white"):
        drawing.add_artist(Polygon([(scale(u), scale(v)) for u, v in points],
                                   closed=True, facecolor=fill, edgecolor=color, lw=width, joinstyle="round"))

    bolt = [(.55, .88), (.27, .46), (.47, .46), (.42, .15), (.74, .58), (.55, .58)]
    if kind == "grid":
        line([(.16, .11), (.50, .91), (.84, .11)])
        for y0, left, right in ((.27, .22, .78), (.53, .29, .71), (.72, .34, .66)):
            line([(left, y0), (right, y0)])
        line([(.26, .28), (.62, .53), (.36, .53), (.73, .27)])
    elif kind == "pv":
        polygon([(.07, .30), (.76, .30), (.93, .79), (.24, .79)], TEAL)
        line([(.16, .55), (.845, .55)])
        line([(.30, .30), (.47, .79)])
        line([(.53, .30), (.70, .79)])
        line([(.49, .30), (.49, .12), (.30, .12), (.69, .12)])
    elif kind == "battery":
        polygon([(.12, .24), (.85, .24), (.85, .40), (.95, .40), (.95, .62), (.85, .62),
                 (.85, .77), (.12, .77)])
        polygon([(.48, .69), (.34, .47), (.47, .47), (.43, .33), (.63, .56), (.50, .56)], TEAL)
    elif kind == "ev":
        polygon([(.11, .27), (.89, .27), (.90, .53), (.77, .56), (.64, .74), (.34, .74),
                 (.22, .56), (.13, .51)], "white")
        line([(.25, .56), (.75, .56)])
        circle(.28, .26, .085)
        circle(.74, .26, .085)
    elif kind == "charge":
        polygon(bolt, "white")
    elif kind == "lock":
        drawing.add_artist(Arc((scale(.50), scale(.57)), scale(.43), scale(.51),
                               theta1=0, theta2=180, color=color, lw=width))
        polygon([(.22, .16), (.78, .16), (.78, .58), (.22, .58)], TEAL)
        line([(.50, .33), (.50, .42)])
    elif kind == "bars":
        polygon([(.16, .20), (.37, .20), (.37, .66), (.16, .66)], WARM)
        polygon([(.58, .20), (.79, .20), (.79, .43), (.58, .43)], TEAL)
        line([(.10, .18), (.87, .18)])
    else:
        raise ValueError(kind)
    ax.add_artist(AnnotationBbox(drawing, (x, y), frameon=False, pad=0, box_alignment=(.5, .5), zorder=5))


def arrow(ax, start, end, *, color=INK, curve=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, shrinkA=0, shrinkB=0,
                                 color=color, lw=1.1, connectionstyle=f"arc3,rad={curve}", zorder=4))


def clean_axes(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()


def title(ax, text):
    ax.text(.005, .99, text, fontsize=13, va="top", fontweight="bold")


def save(fig, number):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"Fig_{number}.tif", dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"Fig_{number}.png", dpi=210)
    fig.savefig(OUT / f"Fig_{number}.svg")
    plt.close(fig)


def station(ax):
    clean_axes(ax)
    title(ax, "Station energy balance")
    for x, kind, label in ((.12, "grid", "Grid"), (.372, "pv", "PV"),
                           (.623, "battery", "Battery"), (.872, "ev", "EVs")):
        pictogram(ax, kind, x, .744, 34, color=BLUE if kind == "pv" else INK)
        ax.text(x, .584, label, ha="center", va="center", fontsize=12)
    ax.plot([.05, .948], [.335, .335], color=INK, lw=1.45)
    for x, label in ((.12, r"$g_t$"), (.372, r"$v_t$")):
        arrow(ax, (x, .513), (x, .342))
        ax.text(x + .015, .437, label, va="center", fontsize=11.6)
    arrow(ax, (.596, .343), (.596, .510))
    arrow(ax, (.649, .511), (.649, .342))
    ax.text(.581, .437, r"$b_t^+$", ha="right", va="center", fontsize=11.6)
    ax.text(.665, .437, r"$b_t^-$", va="center", fontsize=11.6)
    arrow(ax, (.872, .342), (.872, .511))
    ax.text(.888, .437, r"$\sum_i x_{it}$", va="center", fontsize=11.6)
    ax.text(.5, .226, r"$g_t+v_t+b_t^-=\sum_i x_{it}+b_t^+$",
            ha="center", va="center", fontsize=13.5)
    ax.text(.5, .069, r"$C=\sum_t c_tg_t$" + "       " +
            r"$K=\sum_t[p_tg_t+\omega(b_t^++b_t^-)]$",
            ha="center", va="center", fontsize=12.4)


def checkpoint(ax):
    a, d, end, delayed = 8., 75., 105., 27.
    rate, total = 8 / 60, 10.
    t = np.linspace(0, end, 841)
    immediate = np.clip((t - a) * rate, 0, total)
    deferred = np.clip((t - delayed) * rate, 0, total)
    r, q = (d - a) * rate, (d - delayed) * rate
    ax.set_title("Early service checkpoint", loc="left", fontsize=13, fontweight="bold", pad=7)
    ax.axvspan(60, d, color=TEAL, zorder=0)
    ax.axvspan(d, 90, color=WARM, alpha=.58, zorder=0)
    for grid in (0, 30, 60, 90):
        ax.axvline(grid, linewidth=.55, color="#cdd6db", zorder=0)
    for x, weight in ((a, .7), (d, .95), (end, .7)):
        ax.axvline(x, lw=weight, color=INK if x == d else MID, ls=":")
    ax.hlines(total, 0, end, lw=.65, color=MID, ls=":")
    ax.plot(t, immediate, color=BLUE, lw=1.9, label="Immediate")
    ax.plot(t, deferred, color=ORANGE, lw=1.75, ls="--", label="Deferred")
    ax.scatter([d, d], [r, q], s=22, color=[BLUE, ORANGE], zorder=5)
    ax.text(d - 2.7, r + .26, r"$r_i$", color=BLUE, va="bottom", ha="right", fontsize=11.7)
    ax.text(d + 2.5, q - .16, r"$q_i$", color=ORANGE, va="top", fontsize=11.7)
    ax.scatter([end], [total], s=18, color=INK, zorder=5)
    ax.text(67.5, 11.29, r"$x_{it}^{\mathrm{pre}}$", ha="center", va="center", fontsize=11.7)
    ax.text(82.5, 11.29, r"$x_{it}^{\mathrm{post}}$", ha="center", va="center", fontsize=11.7)
    ax.hlines([q, r], d, 112, colors=MID, linestyles=":", linewidth=.58, zorder=1)
    ax.hlines(total, end, 112, colors=MID, linestyles=":", linewidth=.58)
    for lower, upper, label, color in ((r, total, r"$E_i-r_i$", BLUE), (q, r, r"$r_i-q_i$", ORANGE)):
        ax.annotate("", xy=(113, upper), xytext=(113, lower),
                    arrowprops={"arrowstyle": "|-|", "color": color, "linewidth": .95, "shrinkA": 0, "shrinkB": 0})
        ax.text(117, (lower + upper) / 2, label, color=color, va="center", fontsize=11.8)
    ax.set(xlim=(0, 143), ylim=(0, 12.25), xticks=[a, d, end],
           xticklabels=["Arrival\n" + r"$a_i$", "Checkpoint\n" + r"$d_i$", "Proxy end\n" + r"$e_i$"],
           yticks=[0, total], yticklabels=["0", r"$E_i$"], ylabel="Energy delivered (kWh)")
    ax.legend(loc="upper left", fontsize=10.2, handlelength=1.65, borderpad=.3)


def site_and_service():
    fig, ax = plt.subplots(figsize=(7.12, 2.75), layout="constrained")
    station(ax)
    save(fig, 3)
    fig, ax = plt.subplots(figsize=(7.12, 3.4), layout="constrained")
    checkpoint(ax)
    save(fig, 4)


def regime_track(ax, left, base, *, late, battery_shift=False):
    w = .092
    for j, fill in enumerate((TEAL, WARM, TEAL)):
        ax.add_patch(Rectangle((left + j * w, base), w, .115, edgecolor=MID if j == 1 else "white",
                               linewidth=.6, facecolor=fill, hatch="//" if j == 1 else None))
    ax.add_patch(Rectangle((left, base), 3 * w, .115, edgecolor=MID, linewidth=.65, facecolor="none"))
    arrival = left + w
    ax.vlines(arrival, base - .021, base + .137, colors=INK, linestyles=":", linewidth=.9)
    charge_at = left + w * (2.5 if late else 1.5)
    pictogram(ax, "charge", charge_at, base + .0575, 16, color=ORANGE if not late and not battery_shift else BLUE)
    if battery_shift:
        pv_at, storage_at = left + w * .21, left + w * .73
        pictogram(ax, "pv", pv_at, base + .056, 14, color=BLUE)
        pictogram(ax, "battery", storage_at, base + .056, 16, color=BLUE)
        arrow(ax, (pv_at + .012, base + .056), (storage_at - .015, base + .056), color=BLUE)
        arrow(ax, (storage_at + .013, base + .093), (charge_at - .016, base + .092), color=BLUE, curve=-.47)


def temporal_mechanism(ax):
    clean_axes(ax)
    title(ax, "How flexible charging can substitute for a battery")
    for x, label in ((.353, "No battery"), (.689, "With battery"), (.932, "Battery gain")):
        ax.text(x, .823, label, ha="center", fontsize=11.85, fontweight="bold")
    for base, label, late, symbol in ((.565, "Immediate", False, r"$\Delta_{\mathrm{ASAP}}$"),
                                      (.322, "Flexible", True, r"$\Delta_{\mathrm{full}}$")):
        ax.text(.015, base + .057, label, va="center", fontsize=12.5, fontweight="bold")
        regime_track(ax, .216, base, late=late)
        regime_track(ax, .551, base, late=late, battery_shift=not late)
        ax.text(.933, base + .057, symbol, ha="center", va="center", fontsize=13, color=BLUE)
    for x, color, name in ((.02, TEAL, "Lower carbon"), (.26, WARM, "Higher carbon")):
        ax.add_patch(Rectangle((x, .123), .036, .056, linewidth=.6, facecolor=color, edgecolor=MID,
                               hatch="//" if color == WARM else None))
        ax.text(x + .046, .151, name, va="center", fontsize=10.95)
    pictogram(ax, "charge", .514, .151, 14, color=INK)
    ax.text(.542, .151, "EV charge", va="center", fontsize=10.95)
    ax.vlines(.725, .115, .187, colors=INK, linestyles=":", linewidth=.9)
    ax.text(.743, .151, "Arrival", va="center", fontsize=10.95)
    ax.text(.987, .046, "Schematic", ha="right", fontsize=9.85, color="#57646c")


def matching_objectives(ax):
    clean_axes(ax)
    title(ax, "Two controlled comparisons")
    for x, fill, name, constraint, result in (
        (.013, TEAL, "Fix early service", r"$q_i\geq r_i$", r"$\min C$"),
        (.519, WARM, "Fix carbon", r"$C\leq C_\lambda$", r"$\min D$"),
    ):
        ax.add_patch(Rectangle((x, .275), .466, .53, linewidth=.8, edgecolor=MID, facecolor=fill))
        ax.text(x + .233, .704, name, ha="center", fontsize=12.5, fontweight="bold")
        pictogram(ax, "lock", x + .10, .523, 24)
        pictogram(ax, "bars", x + .365, .523, 24)
        arrow(ax, (x + .19, .523), (x + .273, .523))
        ax.text(x + .10, .362, constraint, ha="center", fontsize=12.4)
        ax.text(x + .365, .362, result, ha="center", fontsize=12.4)
    ax.text(.5, .124, r"Common constraints    $\sum_t x_{it}=E_i$    and    $K\leq B$",
            ha="center", fontsize=11.8)


def offline_comparisons():
    fig, ax = plt.subplots(figsize=(7.12, 2.85), layout="constrained")
    temporal_mechanism(ax)
    save(fig, 5)
    fig, ax = plt.subplots(figsize=(7.12, 2.4), layout="constrained")
    matching_objectives(ax)
    save(fig, 6)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.framealpha": 1, "legend.edgecolor": "none", "savefig.facecolor": "white"})
    use_editable_times()
    site_and_service()
    offline_comparisons()

"""Keep plot and equation glyphs in the same editable publication font."""
import matplotlib
from matplotlib import font_manager


FONT = "Times New Roman"


def use_editable_times():
    for style, weight in (("normal", "normal"), ("normal", "bold"), ("italic", "normal"), ("italic", "bold")):
        font_manager.findfont(font_manager.FontProperties(family=FONT, style=style, weight=weight),
                              fallback_to_default=False)
    matplotlib.rcParams.update({
        "font.family": FONT,
        "svg.fonttype": "none",
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "mathtext.bfit": f"{FONT}:italic:bold",
        "mathtext.fallback": None,
    })

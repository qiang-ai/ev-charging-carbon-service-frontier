"""Transfer verified manuscript content to the ASCE Journal LaTeX template."""
from __future__ import annotations

import re
from types import SimpleNamespace

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

import build_manuscript as manuscript


ROOT = manuscript.ROOT
TEMPLATE = ROOT / "asce"
TITLE = "Carbon and Charging Service Tradeoffs of Stationary Storage at Public EV Charging Sites"
EQUATIONS = (
    r"g_t + v_t + b_t^- = \sum_i x_{it} + b_t^+, \qquad \sum_t x_{it} = E_i",
    r"S_{t+1} = S_t + \eta b_t^+ - b_t^-/\eta",
    r"C = \sum_t c_tg_t,\qquad K = \sum_t [p_tg_t+0.01(b_t^+ + b_t^-)] \le B",
    r"D = \frac{\sum_i (r_i-q_i)}{\sum_i E_i}",
)
REPLACEMENTS = {
    "CO₂": r"CO\textsubscript{2}", "m²": r"m\textsuperscript{2}",
    "xᵢₜ": "$x_{it}$", "gₜ": "$g_t$", "vₜ": "$v_t$", "bₜ⁺": "$b_t^+$",
    "bₜ⁻": "$b_t^-$", "Eᵢ": "$E_i$", "Pᵢ": "$P_i$", "pₜ": "$p_t$", "cₜ": "$c_t$",
    "aᵢ": "$a_i$", "eᵢ": "$e_i$", "dᵢ": "$d_i$", "qᵢ": "$q_i$", "rᵢ": "$r_i$",
    "C₀ − λ(C₀ − Cₘᵢₙ)": r"$C_0-\lambda(C_0-C_{\min})$",
    "qᵢ ≥ rᵢ": r"$q_i \ge r_i$", "Eᵢ − rᵢ": "$E_i-r_i$", "rᵢ − qᵢ": "$r_i-q_i$",
    "Cₘᵢₙ": r"$C_{\min}$", "C₀": "$C_0$", "η": r"$\eta$", "ω": r"$\omega$",
    "λ": r"$\lambda$", "©": r"\textcopyright{}", "£": r"\pounds{}",
    "°": r"\textdegree{}", "–": "--", "−": "$-$",
    "%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#",
}
TOKEN = re.compile(r'https?://[^\s]+|' + "|".join(re.escape(x) for x in sorted(REPLACEMENTS, key=len, reverse=True)))


def escape(value):
    def replace(match):
        word = match.group()
        if word.startswith(("https://", "http://")):
            url = word.rstrip(".")
            return r"\url{" + url + "}" + word[len(url):]
        return REPLACEMENTS[word]

    return TOKEN.sub(replace, str(value))


class Manuscript:
    def __init__(self):
        self.lines = []
        self.mode = "normal"
        self.core_properties = SimpleNamespace(title=TITLE)
        self.paragraph_format = SimpleNamespace()
        self.ref_count = 0

    def add_heading(self, title, level=1):
        if title == "Abstract":
            self.lines.extend((r"\maketitle", r"\begin{abstract}"))
            self.mode = "abstract"
        elif title == "References":
            self.lines.append(r"\begin{thebibliography}{99}")
            self.mode = "reference"
        else:
            if title == "Figure Captions":
                self.lines.extend((r"\end{thebibliography}", r"\clearpage"))
                self.mode = "normal"
            if title == "Tables":
                self.lines.append(r"\clearpage")
                return self
            section = "section" if level == 1 else "subsection"
            self.lines.append("\\" + section + "{" + escape(title) + "}")
        return self

    def add_paragraph(self, text, style=None):
        if style == "Title":
            self.lines.append(r"\title{\MakeUppercase{" + escape(text) + "}}")
        elif self.mode == "abstract":
            self.lines.extend((escape(text), r"\end{abstract}"))
            self.mode = "normal"
        elif text.startswith("Keywords  "):
            self.lines.append(r"\KeyWords{" + escape(text.removeprefix("Keywords  ")) + "}")
        elif self.mode == "reference":
            self.ref_count += 1
            author, rest = text.split('. "', 1)
            title, after = rest.split('." ', 1)
            venue = re.match(r"(.*?)(?= \d+[\s:(])", after)
            if venue and not after.startswith(("Accessed", "Data set", "In Proc.")):
                name = venue.group(1)
                after = r"\emph{" + escape(name) + "}" + escape(after[len(name):])
            else:
                after = escape(after)
            item = rf"\bibitem[]{{reference{self.ref_count}}} "
            self.lines.append(item + escape(author) + r". \textquotedblleft{}" + escape(title) +
                              r".\textquotedblright{} " + after)
        else:
            self.lines.extend(("", escape(text)))
        return self

    def add_byline(self):
        for name, affiliations in manuscript.AUTHORS:
            marks = [str(number) for number in affiliations]
            if name == manuscript.CORRESPONDING_AUTHOR:
                marks.append("*")
            self.lines.append(r"\author[" + ",".join(marks) + "]{" + escape(name) + "}")
        for number, affiliation in enumerate(manuscript.AFFILIATIONS, start=1):
            self.lines.append(rf"\affil[{number}]{{{escape(affiliation)}}}")
        name = escape(manuscript.CORRESPONDING_AUTHOR)
        email = escape(manuscript.CORRESPONDING_EMAIL)
        self.lines.append(rf"\affil[*]{{Corresponding author: {name}. Email: \href{{mailto:{email}}}{{{email}}}}}")

    def add_equation(self, _expression, number):
        self.lines.extend((r"\begin{linenomath}", r"\begin{equation}", EQUATIONS[number - 1],
                           r"\end{equation}", r"\end{linenomath}"))

    def add_table(self, caption, headers, rows, widths, *, start_page=False):
        if start_page:
            self.lines.append(r"\clearpage")
        self.lines.append(r"\noindent\textbf{" + escape(caption) + "}")
        self.lines.extend((r"\par\smallskip", r"\noindent"))
        fractions = [width / sum(widths) for width in widths]
        padding = 12 * (len(headers) - 1)
        spec = "@{}" + "".join(
            r"p{\dimexpr" + f"{fraction:.4f}" + r"\linewidth" +
            f"-{fraction * padding + 0.1:.3f}pt" + r"\relax}" for fraction in fractions
        ) + "@{}"
        self.lines.append(r"\begin{tabular}{" + spec + "}")
        self.lines.append(r"\hline")
        self.lines.append(" & ".join(r"\textbf{" + escape(x) + "}" for x in headers) + r" \\ \hline")
        for row in rows:
            self.lines.append(" & ".join(escape(x) for x in row) + r" \\")
        self.lines.extend((r"\hline", r"\end{tabular}", r"\par\medskip"))

    def add_figures(self):
        for n in range(1, 11):
            self.lines.extend((
                r"\clearpage", r"\begin{center}", rf"\textbf{{Fig. {n}}}\par",
                rf"\includegraphics[width=\linewidth,height=.75\textheight,keepaspectratio]{{../figures/Fig_{n}.pdf}}",
                r"\end{center}",
            ))


def build_figures():
    for n in range(1, 11):
        path = ROOT / "figures" / f"Fig_{n}.pdf"
        with Image.open(path.with_suffix(".tif")) as source:
            img = source.convert("RGB")
            width = 7 * 72
            height = width * img.height / img.width
            doc = Canvas(str(path), pagesize=(width, height), pageCompression=1)
            doc.setTitle(f"Figure {n}")
            doc.drawImage(ImageReader(img), 0, 0, width=width, height=height)
            doc.showPage()
            doc.save()


def main():
    source = (TEMPLATE / "ascexmpl-new.tex").read_text(encoding="utf-8")
    preamble = source.split(r"\begin{document}", 1)[0]
    preamble = preamble.replace("[Journal,letterpaper]", "[Journal,letterpaper,10pt,InsideFigs,NoLists]")
    surname = manuscript.AUTHORS[0][0].rsplit(" ", 1)[-1]
    preamble = preamble.replace(r"\NameTag{AuthorOneLastName, \today}", rf"\NameTag{{{surname}}}")
    preamble = preamble.replace("citecolor=red,linkcolor=black", "citecolor=black,linkcolor=black,urlcolor=black")
    preamble += r"\urlstyle{same}" + "\n"
    pdf_authors = ", ".join(name for name, _ in manuscript.AUTHORS)
    preamble += rf"\hypersetup{{pdfauthor={{{escape(pdf_authors)}}}}}" + "\n"
    doc = Manuscript()
    manuscript.h = lambda d, title, level=1: d.add_heading(title, level)
    manuscript.p = lambda d, text: d.add_paragraph(text)
    manuscript.byline = lambda d: d.add_byline()
    manuscript.equation = lambda d, expression, number: d.add_equation(expression, number)
    manuscript.table = lambda d, *args, **kwargs: d.add_table(*args, **kwargs)
    for part in (manuscript.front, manuscript.background, manuscript.design, manuscript.methods,
                 manuscript.results, manuscript.end):
        part(doc)
    doc.add_figures()
    manuscript_path = TEMPLATE / "JTE_Part_A_Technical_Paper.tex"
    manuscript_path.write_text(preamble + "\\begin{document}\n\n" + "\n".join(doc.lines) +
                               "\\end{document}\n", encoding="utf-8")
    build_figures()


if __name__ == "__main__":
    main()

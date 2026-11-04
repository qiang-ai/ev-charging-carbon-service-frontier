"""Build the double-spaced Word working manuscript from verified case summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median, quantiles

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT / "results" / "service_summary.json").read_text(encoding="utf-8"))
CONTROL = json.loads((ROOT / "results" / "controllers" / "controller_summary.json").read_text(encoding="utf-8"))
ENDPOINT = json.loads((ROOT / "results" / "endpoint_sensitivity_audit.json").read_text(encoding="utf-8"))
ONLINE = json.loads((ROOT / "results" / "online_forecast" / "controller_summary.json").read_text(encoding="utf-8"))
PROXY = json.loads((ROOT / "results" / "proxy_window_sensitivity" / "summary.json").read_text(encoding="utf-8"))
REGIONAL = json.loads((ROOT / "results" / "m2_regional" / "summary.json").read_text(encoding="utf-8"))
BLOCKS = json.loads((ROOT / "results" / "block_uncertainty.json").read_text(encoding="utf-8"))
WEAR = json.loads((ROOT / "results" / "wear_sensitivity_summary.json").read_text(encoding="utf-8"))
REFS = json.loads((ROOT / "references.json").read_text(encoding="utf-8"))
LITERATURE = json.loads((ROOT / "literature_evidence.json").read_text(encoding="utf-8"))
YEARS = (2023, 2024)
BATTERIES = (.125, .5)
PV_SCALES = (0., .2, .4)
CHECKPOINTS = (15, 30, 60)
AUTHORS = (
    ("Qiang Ai", (1, 2)),
    ("Zhijie Wang", (1,)),
    ("Kexiang Huang", (3,)),
)
AFFILIATIONS = (
    "School of Instrument Science and Technology, Xi'an Jiaotong University, Xi'an, Shaanxi, China",
    "School of Computer, Qinghai Normal University, Xining, Qinghai, China",
    "School of Computer Science and Technology, Xidian University, Xi'an, Shaanxi, China",
)
CORRESPONDING_AUTHOR = "Qiang Ai"
CORRESPONDING_EMAIL = "qiang.ai@outlook.com"


def main_summary(year, minute=30, pv=.2, battery=.5, design="main"):
    return SUMMARY["summaries"][f"{year}|{design}|{minute}|{pv}|{battery}"]


def strict(year, fraction, minute=30, pv=.2, target=1.):
    return SUMMARY["frontier"][f"{year}|{minute}|{pv}|{target}|{fraction}"]


def carbon(year, minute=30, pv=.2, battery=.5, design="main"):
    return main_summary(year, minute, pv, battery, design)["readiness_floor_carbon_gain_g_per_kwh"]["median"]


def control_effect(comparator, fraction):
    return CONTROL["comparisons"][f"{comparator}|{fraction}"]["carbon_saving_g_per_kwh_quartiles"][1]


def online_effect(comparator, fraction):
    return ONLINE["comparisons"][f"carbon_vs_{comparator}|{fraction}"]["carbon_saving_g_per_kwh_quartiles"][1]


def regional_gain(year, factor):
    return REGIONAL["results"][str(year)]["storage_effects"][f"{factor}|0.5"]["gain_g_per_kwh_q25_q50_q75"][1]


def pooled_shortfall(year, minute, pv, target, battery):
    if target == .5:
        summary = main_summary(year, minute, pv, battery or BATTERIES[0])
        return summary["shortfall_battery_pct" if battery else "shortfall_no_battery_pct"]
    if target in (.95, .99):
        key = f"{year}|{minute}|{pv}|{target}|{battery}"
        return ENDPOINT["near_endpoint_groups"][key]["pooled_avoidable_shortfall_pct"]
    return strict(year, battery, minute, pv, target)["pooled_avoidable_shortfall_pct"]


def storage_value_rows():
    def interval(center, lower, upper):
        return f"{center:.1f} ({lower:.1f}–{upper:.1f})"

    path = ROOT.parent / "cross_year_2024" / "analysis_common_pv" / "matched_interaction_changes.csv"
    with path.open(newline="", encoding="utf-8") as source:
        matches = list(csv.DictReader(source))
    rows = []
    for pv in PV_SCALES:
        cells = []
        for year in YEARS:
            for battery in BATTERIES:
                decreases = [-float(r[f"interaction_g_per_kwh_{year}"]) for r in matches
                             if float(r["pv_scale"]) == pv and float(r["battery_fraction"]) == battery
                             and r[f"both_alpha_pairs_valid_{year}"] == "True"]
                if len(decreases) != SUMMARY["matched_site_weeks_per_year"]:
                    raise ValueError(f"Incomplete window comparison for {year}, PV {pv}, battery {battery}")
                q25, _, q75 = quantiles(decreases, method="inclusive")
                cells.append(interval(median(decreases), q25, q75))
        rows.append(("Window effect", round(100 * pv), *cells))
    for minute in CHECKPOINTS:
        for pv in PV_SCALES:
            cells = []
            for year in YEARS:
                for battery in BATTERIES:
                    dist = main_summary(year, minute, pv, battery)["readiness_floor_carbon_gain_g_per_kwh"]
                    cells.append(interval(dist["median"], dist["q25"], dist["q75"]))
            rows.append((f"{minute}-min service", round(100 * pv), *cells))
    return rows


def set_element(parent, tag, **attrs):
    element = OxmlElement(tag)
    for key, value in attrs.items():
        element.set(qn(f"w:{key}"), str(value))
    parent.append(element)
    return element


def setup():
    d = Document()
    section = d.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
    section.footer_distance = Inches(.5)
    for name in ("Normal", "Title", "Heading 1", "Heading 2", "Caption"):
        s = d.styles[name]
        s.font.name = "Times New Roman"
        s.font.size = Pt(11)
        s.font.color.rgb = RGBColor(0, 0, 0)
        fonts = s.element.find(".//" + qn("w:rFonts"))
        if fonts is not None:
            for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
                fonts.attrib.pop(qn("w:" + attr), None)
        s.paragraph_format.line_spacing = 2
        s.paragraph_format.space_after = Pt(0)
        s.paragraph_format.space_before = Pt(0)
    d.styles["Normal"].paragraph_format.space_after = Pt(2)
    d.styles["Title"].font.size = Pt(16)
    d.styles["Title"].font.bold = True
    d.styles["Title"].paragraph_format.keep_with_next = True
    d.styles["Title"].paragraph_format.line_spacing = 1.25
    title_props = d.styles["Title"].element.get_or_add_pPr()
    border = title_props.find(qn("w:pBdr"))
    if border is not None:
        title_props.remove(border)
    for name in ("Heading 1", "Heading 2"):
        d.styles[name].font.bold = True
        d.styles[name].paragraph_format.space_before = Pt(9)
    d.styles["Caption"].font.italic = False
    d.styles["Caption"].paragraph_format.keep_with_next = True
    props = section._sectPr
    set_element(props, "w:lnNumType", countBy=1, distance=250, restart="continuous")
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.line_spacing = 1
    set_element(footer._p, "w:fldSimple", instr="PAGE")
    d.core_properties.title = "Carbon and Charging Service Tradeoffs of Stationary Storage at Public EV Charging Sites"
    d.core_properties.subject = "Technical Paper for Journal of Transportation Engineering Part A Systems"
    d.core_properties.author = ", ".join(name for name, _ in AUTHORS)
    d.core_properties.keywords = "Public charging, Stationary storage, Carbon intensity, Charging deadlines, Photovoltaics"
    return d


def h(d, title, level=1):
    d.add_heading(title, level)


def p(d, paragraph):
    return d.add_paragraph(paragraph)


def equation(d, expression, number):
    paragraph = d.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    run = paragraph.add_run(expression)
    run.font.name = "Cambria Math"
    paragraph.add_run(f"    ({number})")


def table(d, caption, headers, rows, widths, *, start_page=False):
    title = d.add_paragraph(caption, "Caption")
    if start_page:
        title.paragraph_format.page_break_before = True
    t = d.add_table(rows=1, cols=len(headers))
    t.autofit = False
    properties = t._tbl.tblPr
    borders = set_element(properties, "w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        set_element(borders, f"w:{edge}", val="single", sz=4, color="D9D9D9")
    for col, width in zip(t.columns, widths, strict=True):
        col.width = Inches(width)
    for idx, values in enumerate([headers, *rows]):
        row = t.rows[0] if idx == 0 else t.add_row()
        row_props = row._tr.get_or_add_trPr()
        set_element(row_props, "w:cantSplit")
        if idx == 0:
            set_element(row_props, "w:tblHeader")
        for j, (cell, value, width) in enumerate(zip(row.cells, values, widths, strict=True)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cp = cell.paragraphs[0]
            cp.paragraph_format.line_spacing = 1.2
            cp.paragraph_format.keep_with_next = idx < len(rows)
            cp.paragraph_format.space_before = cp.paragraph_format.space_after = Pt(3)
            if j > 0:
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run(str(value))
            run.bold = idx == 0
            run.font.size = Pt(11)
    spacer = d.add_paragraph()
    spacer.paragraph_format.line_spacing = 1
    spacer.paragraph_format.space_after = Pt(8)
    spacer.paragraph_format.keep_with_next = True


def byline(d):
    authors = d.add_paragraph()
    authors.paragraph_format.keep_with_next = True
    for index, (name, affiliations) in enumerate(AUTHORS):
        if index:
            authors.add_run(", ")
        authors.add_run(name)
        markers = [str(number) for number in affiliations]
        if name == CORRESPONDING_AUTHOR:
            markers.append("*")
        authors.add_run(",".join(markers)).font.superscript = True
    for number, affiliation in enumerate(AFFILIATIONS, start=1):
        line = d.add_paragraph()
        line.paragraph_format.keep_with_next = True
        line.add_run(str(number)).font.superscript = True
        line.add_run(" " + affiliation)
    p(d, f"* Corresponding author: {CORRESPONDING_AUTHOR}. Email: {CORRESPONDING_EMAIL}")


def front(d):
    d.add_paragraph(d.core_properties.title, "Title")
    byline(d)
    h(d, "Abstract")
    g23, g24 = [carbon(year) for year in YEARS]
    q23, q24 = [strict(year, 0.)["pooled_avoidable_shortfall_pct"] for year in YEARS]
    p(d, f"Public charging stations can shift electricity use toward lower-carbon hours, but this can reduce the energy "
         f"available if a driver needs to leave early. We quantify when stationary storage can reduce the attributed carbon "
         f"footprint of grid purchases while preserving charging service. We replayed 6,462 recorded AC charging sessions "
         f"at 12 sites in Edinburgh and Glasgow in January and July of 2023 and 2024. An offline dispatch model held "
         f"session energy and the weekly operating-cost ceiling fixed while varying charging flexibility, photovoltaic "
         f"capacity, and storage. On 70 matched site-weeks per year (5,933 sessions), allowing the full proxy charging window reduced "
         f"the incremental storage benefit by more than 1 g CO₂ per delivered kWh in 67 cases in 2023 and 68 in 2024. "
         f"For a checkpoint 30 min before the recorded session-end proxy, the larger battery lowered the median footprint "
         f"by {g23:.1f} and {g24:.1f} g CO₂/kWh, respectively, while preserving the energy that immediate charging could deliver "
         f"by that checkpoint. At a moderate carbon target, avoidable early energy shortfall was negligible with or without "
         f"storage. In an exploratory scenario at the minimum-carbon target attainable without storage, "
         f"avoidable no-storage shortfalls reached {q23:.1f}% "
         f"and {q24:.1f}% of requested energy. Endpoint perturbations preserved the direction of the main fixed-service "
         f"result, while modeled regional intensity factors yielded smaller gains. In the 2024 rolling replay with an "
         f"online-updated forecast, carbon control saved a median {online_effect('immediate', 0.):.1f} and "
         f"{online_effect('immediate', .5):.1f} g CO₂/kWh relative to immediate charging without and with storage, "
         f"respectively. Its advantage over tariff-only rolling control was small and varied by site-week.")
    h(d, "Practical Applications")
    p(d, "Public charging operators can use their own session records and electricity data to decide when a battery "
         "could help meet a carbon target while protecting the amount of energy available at an early service checkpoint. An operator can first "
         "calculate how much energy each vehicle would have received by a selected checkpoint if it started charging "
         "immediately. This gives a service benchmark that accounts for sessions that are too short to complete at that "
         "checkpoint. The operator can then test whether a charging schedule meets both that benchmark and the carbon "
         "target within the operating budget. In the Scottish case study, flexible charging alone generally preserved the "
         "benchmark at a moderate target. As the target approached the lowest footprint achievable without a battery, the "
         "same strategy could leave vehicles with less energy at the checkpoint. Adding a modest battery in the primary "
         "solar scenario recovered the missing energy in the 30-minute test. In the rolling replay, updating the "
         "forecasts improved their accuracy, but carbon control gave little additional benefit over price-based control. Procurement should consider "
         "local tariffs, connection capacity, equipment cost, and the signals available during operation.")
    p(d, "Keywords  Public EV charging, stationary storage, charging service, carbon accounting, photovoltaics")


def background(d):
    h(d, "Introduction")
    p(d, "The carbon footprint of public electric vehicle charging depends on where drivers plug in and when "
         "stations buy electricity. Powell et al. (2022) distinguish infrastructure access, which shapes where and when "
         "vehicles connect, from control, which shapes electricity use within sessions. UK public AC records show that "
         "connected time beyond active charging can make load shifting possible (Bayram et al. 2023). Utilization "
         "varies across Scottish locations and charger types (Davies et al. 2024), across German area and power "
         "categories (Hecht et al. 2022), and with local spatial factors across three British regions "
         "(Feng et al. 2026). Fleet duty schedules can also concentrate charging opportunities, as observed for "
         "Shenzhen taxis (Lei et al. 2022). These differences motivate examining storage and flexibility on the same "
         "recorded sessions at each public site.")
    p(d, "Charging access has several dimensions. Yang et al. (2026) study where to allocate charging capacity. "
         "Rempel et al. (2024) test whether public fast chargers can initiate and sustain a charge, and Zhao et al. "
         "(2025) study how idle occupancy and unusable equipment constrain turnover. Here the concern is how much "
         "energy an already connected driver can obtain before the modeled end of a session. Winschermann et al. "
         "(2026) discuss minimum-energy-by-time guarantees in online charging, and Winschermann et al. (2023) show "
         "how access to session information changes the tradeoff between peak load and unserved energy at an office site. "
         "At a public site, delaying charge can leave less energy available at an early checkpoint even when the recorded "
         "request is fully delivered by the end.")
    p(d, "Price and carbon offer different signals for choosing when to charge. Li and Jenn (2022), Daneshzand et al. "
         "(2023), and Romano et al. (2025) report that emissions outcomes depend on the tariff, carbon signal, or "
         "available charging time in their respective settings. Li et al. (2023) evaluate a joint price and "
         "marginal-carbon objective with public charging records, while Cheng et al. (2022) show how relaxing total "
         "delivery can change the carbon gains in a station scheduling model. Emission-factor choice also matters. "
         "Powell et al. (2024) distinguish factors used to account for emissions from those used to guide demand "
         "response, Fungyai et al. (2025) show sensitivity to temporal resolution and factor choice, and Will et al. "
         "(2024) and Martin et al. (2025) show how load shifting or matching can yield different results for allocated "
         "emissions and power-system emissions. We use the national average intensity to compare the operating "
         "footprint attributed to electricity purchases.")
    p(d, "With photovoltaics (PV) and stationary storage, a station can also shift grid purchases or carry "
         "solar energy forward while keeping the time of vehicle charging fixed. Xin et al. (2025) show that the "
         "modeled carbon effect of adding storage in Shenzhen depends on the charging strategy. Hull et al. (2024) "
         "and ur Rehman et al. (2022) evaluate PV and storage sizing under grid or demand constraints. Yang et al. "
         "(2024) couple infrastructure sizing with predictive station operation, and Gumus (2026) finds that direct "
         "alignment of daytime demand and solar output can weaken the economic case for storage at a campus site. "
         "These results motivate examining the incremental value of storage at a fixed budget across multiple solar sizes "
         "and charging windows, with early charging service measured for each vehicle.")
    p(d, "Prior controllers already address uncertainty and limited information. Cabrera-Tobar et al. (2022, 2024) "
         "develop predictive control for PV and battery stations with carbon objectives and uncertain inputs, "
         "and Saleem et al. (2024) coordinate residential charging, PV, and storage with stochastic predictive "
         "control. Li et al. (2026) compare tariff and carbon supervisory strategies for a PV and battery station "
         "using designed carbon profiles. Kang et al. (2025) compare rolling carbon-aware charging with an offline "
         "benchmark under distribution constraints. This study uses historical signals to test whether a carbon objective "
         "improves on a tariff-only controller using the same forecasts, equipment, and session requirements.")
    p(d, "We ask when stationary storage adds measurable carbon value at public AC charging sites under a common "
         "operating budget and matched charging service. The first comparison fixes each session's early energy at "
         "the level immediate charging can achieve and estimates the additional carbon reduction from storage. The "
         "second fixes the carbon ceiling and measures any early energy shortfall. A same-site cross-year comparison "
         "retains the equipment and service rules, and a rolling 2024 replay compares achievable control with the "
         "offline frontier under explicit information limits. The proxy charging window runs from recorded start to "
         "start plus transaction duration. Readiness is energy available at an early checkpoint. A matched market-week block "
         "groups all sites with the same month and week rank across the two years.")


def design(d):
    h(d, "Data and Study Design")
    h(d, "Charging sessions and study sites", 2)
    p(d, "Charging transactions were obtained from the public GridCharge archive (Batic 2025), sourced from "
         "ChargePlace Scotland (2026). The original transaction files contain a charger identifier, connector, "
         "start time, transaction duration, and delivered energy. All 6,462 retained records were cross-checked "
         "against the original monthly transaction files using charger and connector identifiers, local start, duration, "
         "and energy. Start times were converted from Europe/London to UTC. Transport Scotland (n.d.) distinguishes "
         "plug-in and unplug from charge start and end in its network guidance. The public transaction reports do not "
         "resolve that distinction. We treat duration as a proxy and perturb it in a sensitivity test.")
    p(d, "Equipment was joined to a frozen infrastructure inventory by charger and connector identifiers. Co-located "
         "equipment records were grouped into geographic sites using the normalized address and mapped coordinates. "
         "The analysis selected connectors rated at no more than 22 kW in the inventory. In the original operator "
         "workbooks, the retained records were labeled AC or Type 2. The model uses the smaller "
         "of the connector rating and a 22-kW vehicle power ceiling. It excludes duplicate records, nonpositive energy, "
         "durations outside 0–24 h, records inconsistent with the power bound, overlapping windows on one connector, and "
         "transactions crossing a study-week boundary. The modeled request for each session is its recorded delivered "
         "energy. The sample therefore describes recorded sessions rather than latent demand for charging.")
    p(d, "Six sites per city were selected from the 2023 eligible sites at evenly spaced ranks of recorded daily "
         "energy. Selection required at least three qualifying site-weeks in each study month. A qualifying site-week was "
         "a complete Monday–Sunday UTC week within January or July with complete market and weather inputs and at least "
         "five qualifying sessions. The 2024 replay kept those 12 sites without replacement (Fig. 1). The main comparison pairs "
         "the same site, calendar month, and ordinal complete UTC week across years (Fig. 2). Table 1 reports the coverage. The "
         "paired panel represents seven matched market-week blocks, each shared by several sites. Complete signal "
         "coverage ended before January 26 and July 31 in 2023 and spans both full months in 2024.")
    h(d, "Energy inputs and equipment scenarios", 2)
    p(d, "We used half-hourly actual national grid carbon intensity for Great Britain (NESO 2026) and the Elexon "
         "APXMIDP half-hourly wholesale price index (Elexon 2026). The price indexes modeled energy purchases. "
         "The attributional footprint assigns grid imports the national average carbon intensity. "
         "The offline model uses realized historical signals to estimate the operating frontier.")
    p(d, "Both years use hourly ERA5 global tilted irradiance accessed through Open-Meteo (Hersbach et al. 2023; "
         "Open-Meteo 2026), sampled at one reference point per city for a south-facing array at a 35° tilt. "
         "Each hour-ending value is assigned to the preceding two half-hours. Available output per kWp equals "
         "0.86 times irradiance in kW/m², a lumped derate using the 14% PVGIS default loss assumption (JRC n.d.). "
         "The nameplates are scenarios normalized by each site's mean eligible daily "
         "charging energy in 2023. The photovoltaic capacities are 0, 20, and 40 kWp and the storage capacities "
         "are 0, 12.5, and 50 kWh per 100 kWh/day of that reference demand. The nonzero battery power rating "
         "equals one-half of its energy rating in kW, a two-hour dispatch scenario. Each site retains the same physical capacities in both years.")


def methods(d):
    h(d, "Dispatch and Charging Service Methods")
    p(d, "The modeled system connects three decisions. The public charger allocates energy to sessions subject to connector "
         "power and service windows. The station energy controller balances charging load, photovoltaics, storage, and grid "
         "purchases. The settlement layer evaluates grid imports against the half-hourly price and carbon signals (Fig. 3). "
         "The offline and rolling models share the physical constraints. Their cost rules reflect the information available at dispatch.")
    h(d, "Dispatch model and cost ceiling", 2)
    p(d, "The dispatch model uses one half-hourly energy balance for each site. Let xᵢₜ denote charge delivered to "
         "session i, gₜ grid imports, vₜ utilized PV, and bₜ⁺ and bₜ⁻ battery charging and discharging energy in interval t. "
         "Every session receives its recorded energy Eᵢ. Its interval allocation is bounded by the effective power "
         "ceiling Pᵢ times the exact overlap in hours between the interval and its allowed window. Imported energy "
         "is nonnegative, and PV use can be curtailed. The model permits grid charging of storage, excludes export "
         "and vehicle-to-grid operation, and assumes the site resources share the modeled bus.")
    equation(d, "gₜ + vₜ + bₜ⁻ = ∑ᵢ xᵢₜ + bₜ⁺,     ∑ₜ xᵢₜ = Eᵢ", 1)
    p(d, "The main battery scenario assumes 90% round-trip efficiency, with symmetric charge and discharge efficiencies η "
         "equal to the square root of 0.90. State of charge S stays between zero and nameplate energy and starts and "
         "ends the week at 50%. Charging and discharging each respect the power rating, and a binary mode prevents "
         "simultaneous flows. The site balance and battery power operate on half-hour energy totals. No site import "
         "limit is imposed in the main design.")
    equation(d, "Sₜ₊₁ = Sₜ + ηbₜ⁺ − bₜ⁻/η", 2)
    p(d, "The budget covers grid purchases and a battery throughput charge. Cycling contributes to degradation "
         "(Vermeer et al. 2022). The illustrative £0.01 per kWh charged or discharged corresponds to £100/kWh "
         "of replacement value spread over 5,000 full cycles, counting both directions and ignoring losses in that "
         "conversion. This is a scenario parameter. The common ceiling B is the cost of immediate grid-only charging "
         "at the effective session power limits. Retail, demand and capacity charges, capital and fixed costs are excluded. "
         "Let pₜ be the price in GBP/kWh and cₜ the attributed carbon factor in kg CO₂/kWh. The carbon objective C "
         "and the common cost constraint are")
    equation(d, "C = ∑ₜ cₜgₜ,     K = ∑ₜ [pₜgₜ + 0.01(bₜ⁺ + bₜ⁻)] ≤ B.", 3)
    h(d, "Flexible windows and early charging service", 2)
    p(d, "The service experiment retains the full proxy window for final delivery (Fig. 4). For arrival aᵢ and "
         "proxy end eᵢ, an earlier checkpoint dᵢ is the later of aᵢ and eᵢ less 15, 30, or 60 min. Let qᵢ be the energy allocated "
         "before dᵢ. The model splits a session's allocation in any half-hour intersected by dᵢ into two bounded "
         "segments. Energy that could be delivered by immediate charging by dᵢ, denoted rᵢ, is the smaller of Eᵢ "
         "and Pᵢ times the time from arrival to dᵢ. The intrinsic shortfall at the checkpoint is Eᵢ − rᵢ. "
         "The avoidable shortfall from a schedule is rᵢ − qᵢ. These are modeled service metrics under hypothetical "
         "early cutoffs.")
    equation(d, "D = ∑ᵢ (rᵢ − qᵢ) / ∑ᵢ Eᵢ", 4)
    p(d, "The storage interaction compares two availability settings (Fig. 5). Immediate charging starts at recorded arrival "
         "and charges at the effective power ceiling until the request is complete. Full-window charging can shift "
         "energy anywhere within the proxy connection window. For each setting, storage benefit is the no-storage "
         "minus storage footprint divided by total session energy. The interaction is the full-window benefit "
         "minus the immediate-charging benefit. A negative interaction means that flexibility reduced the incremental "
         "carbon value of storage.")
    p(d, "For the fixed-service comparison we minimize C with qᵢ ≥ rᵢ for every session (Fig. 6). For the fixed-carbon "
         "comparison we minimize D, equivalently maximizing the sum of qᵢ, at the same site-week budget and "
         "common carbon ceiling with or without storage. The site-specific ceiling is C₀ − λ(C₀ − Cₘᵢₙ), where "
         "C₀ is the minimum footprint with fixed immediate-charging load, the specified PV, and no battery, "
         "allowing PV curtailment. Cₘᵢₙ is the full-window minimum without storage or an early service rule. "
         "Both use the same cost ceiling. This anchor makes the common target feasible in both "
         "configurations and puts λ on a comparable scale across cases. A separate fixed target requires 10% "
         "less carbon than C₀. Because the service objective maximizes aggregate energy, the fully served session "
         "share describes the computed schedules and is not optimized separately.")
    p(d, "The primary configuration uses a 30-min checkpoint, 20 kWp of PV, and 50 kWh of battery per 100 "
         "kWh/day of 2023 reference demand. The primary PV size corresponds to 3.33–125.72 kWp across sites. "
         "The primary common target uses λ = 0.50. We subsequently "
         "evaluated λ = 0.75, 0.90, and 1.00. After observing a steep rise between 0.90 and 1.00, we added 0.95 "
         "(the midpoint) and 0.99 (one percentage point below the endpoint) as exploratory checks. We also evaluated how far no-storage "
         "operation could move from C₀ toward Cₘᵢₙ while preserving the immediate-charging benchmark, using the "
         "fixed-service carbon optimum directly. Additional checks cap half-hour grid imports at the site-week's "
         "immediate-charging peak and compare 11- and 22-kW ceilings on the same sessions individually feasible "
         "at 11 kW. The benchmark and budget are recomputed for each power ceiling. The import cap likewise "
         "changes the relevant no-storage carbon optimum.")
    p(d, "We offset all proxy ends by 0, ±15, ±30, or ±60 min and recomputed the checkpoint and its floor 30 min "
         "before the adjusted end. Shortening stops at the earliest feasible full delivery. Extension stops at the "
         "next retained session on the connector or week end. We keep all 5,933 matched sessions, energy, and budgets. "
         "The strict-carbon test uses both the adjusted no-storage minimum and the original absolute carbon cap.")
    p(d, "Additional fixed-service runs replace GB actual-average intensity with archived GB and South Scotland "
         "forecast factors (NESO 2026). They hold PV, session, budget and equipment settings fixed. Equipment checks vary "
         "throughput charges (£0, £0.01 and £0.03/kWh) and test an 85%-efficient four-hour battery informed by NREL (2024).")
    p(d, "HiGHS via SciPy solved the models. Audits checked energy, cost, carbon and battery constraints. For 95% "
         "percentile intervals we sampled seven whole matched market-week blocks with replacement 10,000 times, "
         "using the same draw in both years. These intervals describe variation over the observed weeks at the same sites. "
         "With seven blocks, they offer limited precision for broader inference.")
    h(d, "Historical Rolling Controller Benchmark", 2)
    p(d, "The operational replay uses the 70 matched 2024 site-weeks, the primary PV size, and either no battery or "
         "the primary battery. All policies preserve the 30-min immediate-charging benchmark and the final recorded "
         "energy requirement. At plug-in, the controller is assumed to learn the recorded energy and proxy connection end "
         "as if they had been declared by the user. It has no access to subsequent arrivals. The frozen forecast uses "
         "2023 medians by city, month, UTC half-hour and weekday or weekend. The online variant retains frozen PV and updates "
         "price and carbon using the same slot one day earlier, or two days earlier if needed to impose a two-hour buffer "
         "after slot end. It falls back to the frozen profile at the month boundary. This replay uses archived values with "
         "an assumed availability delay.")
    p(d, "Every 2 h the controller solves a 24-h plan, truncated at the end of the study week (Fig. 7). Sessions last "
         "at most 24 h, so the planning window contains the remaining service checkpoints and end times of all active "
         "sessions. The controller subtracts prior delivery from the session's checkpoint and total energy requirements and "
         "updates the battery state. Each plan returns the battery to its initial weekly state of charge by the planning "
         "horizon. The controller executes the first 2 h. Sessions that arrive between decisions begin charging immediately "
         "and can enter the flexible schedule at the next decision. Simulated realized PV balances the bus at execution "
         "using the plan's PV utilization fraction. The realized wholesale price and grid carbon factor are applied afterward.")
    p(d, "For an implementable cost reference at each decision, the controller carries forward the unexecuted part of the "
         "previous feasible plan and adds immediate charging for newly active sessions. The forecast operating cost of "
         "this plan supplies the local ceiling. With unrestricted grid imports, new sessions can be added while preserving "
         "the prior session and battery constraints. The carbon controller minimizes forecast carbon under this ceiling. "
         "The tariff-only benchmark minimizes forecast operating cost with the same forecasts, horizon, equipment, "
         "service constraints, and rule for constructing the ceiling. Each policy carries forward its own prior plan. "
         "The immediate benchmark fixes vehicle charging at the power ceiling until delivery, uses "
         "PV first without storage, and applies the same forecast carbon dispatch to the battery when present.")
    p(d, "The whole-week grid-only immediate-charging budget is available only during scoring. We report realized costs, "
         "service errors, and footprints for every policy. An offline optimum uses all arrivals and realized signals with the "
         "same service rule and caps expenditure at the operating cost incurred in that replay. A separate hindsight "
         "diagnostic replaces the price, PV, and carbon forecasts together with realized values within each horizon "
         "and continues to withhold future arrivals (Table 2). An audit changed future arrivals and signals and "
         "confirmed the earlier decisions were identical.")


def results(d):
    a, b = [main_summary(year) for year in YEARS]
    h(d, "Results")
    h(d, "Charging flexibility reduces the incremental storage benefit", 2)
    p(d, "With the primary equipment sizes, the median fall in incremental storage benefit when changing from "
         "immediate charging to full-window charging was 11.46 g CO₂/kWh in 2023 and 11.55 g CO₂/kWh "
         "in 2024. Solver bounds certified a fall greater than the descriptive threshold of 1 g CO₂/kWh in 67 and 68 of 70 paired "
         "site-weeks, respectively, and in 65 site-week pairs in both years (Fig. 8). Table 3 shows that the median "
         "decline was smaller without PV for both battery sizes and years. Flexibility shifts when vehicles charge, "
         "while storage shifts when the site buys electricity or uses PV.")
    h(d, "Storage carbon value at matched charging service", 2)
    p(d, f"The immediate-charging reference left {a['physical_shortfall_pct']:.4f}% and "
         f"{b['physical_shortfall_pct']:.4f}% of energy undelivered at the 30-min checkpoint in the two years. "
         f"About {a['early_full_asap_pct']:.1f}% and {b['early_full_asap_pct']:.1f}% of sessions could be fully "
         "served by the checkpoint even with immediate charging. The service rule preserves that achievable "
         "energy for each session. Under the rule, the primary battery reduced the median carbon footprint "
         f"by {carbon(2023):.2f} g CO₂/kWh in 2023 and {carbon(2024):.2f} g CO₂/kWh in 2024. Gains met both "
         "5 g CO₂/kWh and 5% relative in 70 and 68 site-weeks, respectively. These dual cutoffs "
         "are descriptive measures of the modeled effect size.")
    p(d, "Both batteries gained more at the 30-min checkpoint with the primary PV size than without PV "
         "(Table 3 and Fig. 9). Higher PV changes both the baseline footprint and the opportunities for storing surplus output.")
    h(d, "The carbon target determines whether storage improves readiness", 2)
    p(d, f"At λ = 0.50, the pooled avoidable shortfall under the primary configuration rounded to zero "
         f"with and without storage in both years. Without a battery, the median share of the unconstrained "
         f"carbon-reduction potential that could be achieved while preserving immediate-charging readiness was "
         f"{100 * a['median_compatible_potential_no_battery']['median']:.1f}% and "
         f"{100 * b['median_compatible_potential_no_battery']['median']:.1f}% in 2023 and 2024, respectively. "
         "An operator working below this boundary may meet both objectives through charging control alone. "
         "The boundary varies by site-week and is calculated from the site's own service benchmark.")
    p(d, "At λ = 0.50, the carbon ceiling represented a median reduction from immediate charging of 10.41% in "
         "2023 and 9.12% in 2024. The no-storage case met the early benchmark within 0.00001 kWh in 139 of "
         "140 site-weeks. The remaining case fell short by 0.047 kWh.")
    p(d, f"At λ = 1.00, schedules without a battery had pooled avoidable shortfalls of "
         f"{strict(2023, 0.)['pooled_avoidable_shortfall_pct']:.2f}% and "
         f"{strict(2024, 0.)['pooled_avoidable_shortfall_pct']:.2f}% of requested energy at the 30-min "
         "checkpoint (Table 4 and Fig. 10). With either battery, summed avoidable shortfall was below "
         "0.00001 kWh in every matched site-week at this checkpoint and PV size. "
         f"At the 60-min checkpoint, the no-storage shortfalls rose to "
         f"{strict(2023, 0., 60)['pooled_avoidable_shortfall_pct']:.2f}% and "
         f"{strict(2024, 0., 60)['pooled_avoidable_shortfall_pct']:.2f}%. The smaller battery left "
         f"{strict(2024, .125, 60)['pooled_avoidable_shortfall_pct']:.2f}% in 2024. "
         "All these schedules deliver full energy by the original proxy end.")
    p(d, "Intermediate targets show the service loss growing toward the no-storage minimum. At a separate 10% "
         "reduction from immediate charging, full delivery by the proxy end was feasible in 57/70 and 51/70 site-weeks "
         "without storage, 69/70 and 68/70 with the smaller battery, and 70/70 in both years with the larger battery. "
         "That test uses final energy, the operating budget and the carbon cap.")
    h(d, "Robustness of the planning comparisons", 2)
    shifts = PROXY["summary"].values()
    window = [r["primary_interaction_fall_g_per_kwh"]["median"] for r in shifts]
    service = [r["fixed_service_battery_gain_g_per_kwh"]["0.5"]["median"] for r in shifts]
    p(d, f"Across the endpoint perturbations, the full-window reduction in storage value remained "
         f"{min(window):.2f}–{max(window):.2f} g CO₂/kWh and the larger battery's fixed-service gain remained "
         f"{min(service):.2f}–{max(service):.2f} g CO₂/kWh. At the recomputed strict cap, the larger battery kept "
         "avoidable shortfall below 0.00001 kWh per site-week. At the original absolute cap, all three shortening "
         "scenarios made no-storage operation infeasible, whereas the largest extension restored feasibility in "
         "both years with pooled shortfall below 0.03%. The 30-min checkpoint moves with the end in each scenario "
         "(Table S1).")
    p(d, f"The larger battery's median fixed-service gain was {regional_gain(2023, 'gb_forecast'):.2f} and "
         f"{regional_gain(2024, 'gb_forecast'):.2f} g CO₂/kWh with GB forecasts, but "
         f"{regional_gain(2023, 'south_scotland_forecast'):.2f} and {regional_gain(2024, 'south_scotland_forecast'):.2f} "
         "with South Scotland forecasts. The share meeting both effect-size cutoffs fell under the regional factors "
         "(Table S2). Table 5 reports block intervals and checks on power, wear, efficiency, and sampling. "
         "The two city-specific estimates differ substantially in 2024.")
    h(d, "Operating Policy Benchmarks", 2)
    mae = ONLINE["forecast_mae"]["by_city_month"]
    jan, july = [mae[f"edinburgh|{month}"]["first_two_hours"] for month in (1, 7)]
    p(d, f"On matched weeks, updating forecasts reduced the first-2-h carbon MAE from {jan['frozen']['carbon']:.2f} "
         f"to {jan['online']['carbon']:.2f} g CO₂/kWh in January and from {july['frozen']['carbon']:.2f} to "
         f"{july['online']['carbon']:.2f} in July. All 840 forecast-based policy replays met both service constraints "
         "and the common realized weekly budget within numerical tolerance. Relative to tariff-only control, median "
         f"carbon savings with online updates were {online_effect('tariff', 0.):.2f} without storage and "
         f"{online_effect('tariff', .5):.2f} g CO₂/kWh with storage. Both interquartile ranges spanned zero. "
         "Energy-weighted effects had opposite signs with and without storage (Table 6 and Table S3).")
    gaps = [CONTROL["groups"][f"carbon|{f}"]["equal_realized_cost_oracle_gap_g_per_kwh_quartiles"][1]
            for f in (0., .5)]
    gain = [control_effect("perfect_vs_frozen", f) for f in (0., .5)]
    p(d, f"The carbon controller's median gap to the offline benchmark capped at replay cost was {gaps[0]:.2f} and "
         f"{gaps[1]:.2f} g CO₂/kWh without and with storage. Replacing the frozen forecasts with future realized "
         f"signals in the otherwise identical rolling controller reduced the median footprint by {gain[0]:.2f} and "
         f"{gain[1]:.2f} g CO₂/kWh, respectively. The diagnostic continues to hide future arrivals and tests "
         "the joint effect of replacing all three forecast signals in hindsight.")


def end(d):
    h(d, "Discussion")
    p(d, "The service and carbon results identify different reasons for considering storage. Without storage, "
         "the primary moderate carbon target and immediate-charging readiness were jointly feasible in nearly "
         "every site-week. This is consistent "
         "with the observed substitution between long charging windows and batteries. At a strict carbon target, a "
         "battery lets the station deliver more energy early using electricity acquired at other times. This "
         "mechanism explains why storage can add value even when flexible charging can meet the same carbon target "
         "over the full modeled connection window.")
    p(d, "The two forecast models yielded small and heterogeneous incremental gains from carbon control. "
         "The offline benchmark uses longer horizons and advance information about arrivals and signals. "
         "Its gap from rolling control is an opportunity bound under the model.")
    p(d, "The comparison suggests an operational screen. Estimate the immediate-charging energy by a locally "
         "relevant checkpoint. Compute the minimum footprint of schedules meeting that service rule without "
         "storage. If this footprint exceeds the operator's target, repeat the calculation with candidate sizes. "
         "The difference between the two feasible frontiers measures the carbon and service value attributable to "
         "storage under the modeled budget. The 60-min case illustrates why sizing depends on the chosen service "
         "requirement. Investment appraisal requires retail and demand charges, connection costs and capital and lifetime costs.")
    p(d, "The window test changes planning inputs consistently. A rolling controller facing unobserved errors "
         "could miss its service commitment. Understating requested energy reduces the planned delivery and can "
         "lower the readiness floor. Overstating energy can force unnecessary early charging or an infeasible request. "
         "A deadline declared too late can defer energy past actual unplugging. A deadline declared too early can "
         "reduce flexibility and can become infeasible at the connector power limit. The replays measure conditional "
         "performance under exact declarations. Deployment would require a way to obtain or update those inputs.")
    p(d, "The dataset contains no independent unplug timestamps, charging-power traces, queues or unmet requests. "
         "Shared cabinet limits could further restrict the modeled schedules. The retained sites and months define the "
         "scope of the repeated-week comparisons. Carbon is attributed to grid purchases. The regional sensitivity uses "
         "modeled average forecasts and gives no estimate of marginal emissions or lifecycle impact.")
    h(d, "Conclusions")
    p(d, "The incremental operating carbon value of stationary storage at public charging sites depends on how "
         "much flexibility the operator can use and what energy must be available before the end of the connection "
         "window. At 12 Scottish sites observed in two years, full-window flexibility generally reduced the extra "
         "carbon benefit of storage. At fixed early service, the gain remained positive in the tested window scenarios "
         "but was smaller under modeled South Scotland factors. The reported main magnitudes apply to GB "
         "national-average attribution. Early-service benefits at a fixed absolute cap depended on the proxy endpoint. "
         "Under both forecast models, rolling carbon control reduced the footprint relative to immediate charging "
         "and showed little consistent additional gain over tariff-only control.")
    h(d, "Data Availability Statement")
    p(d, "Some or all data, models, or code that support the findings of this study are available from the "
         "corresponding author upon reasonable request. Original inputs are publicly accessible through ChargePlace "
         "Scotland (2026), Batic (2025), NESO (2026), Elexon (2026), and Open-Meteo (2026). "
         "The reproducibility archive identifies source URLs, snapshot checksums, preparation steps, solver "
         "inputs, outputs, and independent checks. Redistribution of source charging transactions follows the "
         "source providers' terms.")
    h(d, "Acknowledgments")
    p(d, "Contains BMRS data © Elexon Limited copyright and database right 2023 and 2024.")
    p(d, "OpenAI Codex assisted with code development, computational checks, literature verification, "
         "and drafting and editing the manuscript.")
    h(d, "Supplemental Materials")
    p(d, "Tables S1–S3 report the endpoint perturbations, alternative carbon factors, and paired controller comparisons. "
         "The accompanying reproducibility archive includes their case-level results, input manifests, and audit outputs.")
    h(d, "References")
    sources = ("Batic2025", "ChargePlaceScotland2026", "Elexon2026", "Hersbach2023", "JRC",
               "NESO2026", "NREL2024", "OpenMeteo2026", "TransportScotland", "Vermeer2022")
    references = [REFS[key] for key in sources] + [study["reference"] for study in LITERATURE]
    for reference in sorted(references, key=lambda ref: ref.lower().removeprefix("ur ")):
        ref = p(d, reference)
        ref.paragraph_format.left_indent = Inches(.27)
        ref.paragraph_format.first_line_indent = Inches(-.27)
    d.add_heading("Figure Captions", 1).paragraph_format.page_break_before = True
    p(d, "Fig. 1. Construction of the 2023 anchor and 2024 holdout cohorts. Both years use their own irradiance "
         "values and the same PV conversion.")
    p(d, "Fig. 2. Cross-year pairing and the two evaluation samples. Offline comparisons use both years. "
         "Rolling dispatch uses the 2024 members of the pairs.")
    p(d, "Fig. 3. Station energy balance. Arrows show energy in a half-hour interval. Cost K includes "
         "battery throughput wear.")
    p(d, "Fig. 4. Illustrative charge delivery at the early checkpoint. Shading splits the intersected slot. "
         "The two brackets show intrinsic and avoidable shortfall.")
    p(d, "Fig. 5. Substitution mechanism. A battery can save earlier PV surplus for immediate charging, "
         "while flexible charging can use a later low-carbon interval within the session window.")
    p(d, "Fig. 6. Offline comparisons at fixed early service or a fixed carbon target. Both share a weekly "
         "cost ceiling and require delivery of all recorded energy.")
    p(d, "Fig. 7. Rolling dispatch with up to 24 h of planning and a 2-h execution step. New plug-ins charge "
         "immediately between decisions.")
    p(d, "Fig. 8. Immediate minus full-window storage benefit for 70 matched pairs. Dotted lines are at "
         "x = 1 and y = 1 g CO₂/kWh. The diagonal marks equality between years.")
    p(d, "Fig. 9. Median and interquartile range of the storage benefit at fixed 30-min readiness. "
         "Open and filled markers indicate 2023 and 2024. Points are offset within each PV category. "
         "Capacities are per 100 kWh/day of reference demand.")
    p(d, "Fig. 10. Pooled avoidable early shortfall at 20 kWp of PV per 100 kWh/day. Lines show the no-battery "
         "cases. The storage curve gives the maximum across both years, checkpoints, and battery sizes at each shared "
         "target. The horizontal axis is linear. Table 4 reports each configuration and the extra 30-min targets.")
    d.add_heading("Tables", 1).paragraph_format.page_break_before = True
    table(d, "Table 1. Panel coverage by year", ("Cohort", "Year", "Sites", "Site-weeks", "Sessions", "MWh"),
          [("All qualifying site-weeks", 2023, 12, 82, "3,506", "67.71"),
           ("All qualifying site-weeks", 2024, 12, 81, "2,956", "57.39"),
           *[("Matched site-weeks", y, 12, 70, f"{main_summary(y)['sessions']:,}",
              f"{main_summary(y)['energy_kwh'] / 1000:.2f}") for y in YEARS]],
          (2.36, .65, .55, .94, 1.02, .98))
    table(d, "Table 2. Information available at planning time",
          ("Planning case", "Price, PV, and carbon inputs", "Session information", "Planning span"),
          [("Forecast replay", "Frozen 2023 profiles", "Arrived sessions", "Up to 24 h"),
           ("Online replay", "Lagged price and carbon, frozen PV", "Arrived sessions", "Up to 24 h"),
           ("Hindsight replay", "Realized signals", "Arrived sessions", "Up to 24 h"),
           ("Offline benchmark", "Realized signals", "All sessions", "Whole week")],
          (1.55, 2.02, 1.68, 1.25))
    table(d, "Table 3. Charging-window and fixed-service storage effects (g CO₂/kWh)",
          ("Comparison", "PV", "2023, 12.5", "2023, 50", "2024, 12.5", "2024, 50"),
          storage_value_rows(), (1.12, .43, 1.2375, 1.2375, 1.2375, 1.2375), start_page=True)
    p(d, "Cells report the median (25th–75th percentile) of paired effects over 70 site-weeks per year. "
         "Window effect is the storage benefit under immediate charging minus its benefit under full-window charging. "
         "Fixed-service rows give the no-battery minus battery footprint with the specified early-service floor. "
         "PV is in kWp and battery capacity (in column headings) is in kWh, both per 100 kWh/day of reference demand.")
    frontier_cases = [(20, 30, target) for target in (.50, .75, .90, .95, .99, 1.)]
    frontier_cases += [(20, 15, 1.), (20, 60, 1.), (0, 30, 1.), (40, 30, 1.)]
    rows = [(pv, minute, f"{target:.2f}", *
             [f"{max(0., pooled_shortfall(year, minute, pv / 100, target, battery)):.2f}"
              for year in YEARS for battery in (0., *BATTERIES)])
            for pv, minute, target in frontier_cases]
    table(d, "Table 4. Avoidable early energy shortfall at fixed carbon targets (%)",
          ("PV", "Lead (min)", "λ", "2023, 0", "2023, 12.5", "2023, 50",
           "2024, 0", "2024, 12.5", "2024, 50"),
          rows, (.42, .66, .49, .80, .86, .80, .80, .86, .81), start_page=True)
    p(d, "Cells are total avoidable shortfall divided by requested energy across 70 site-weeks per year. "
         "Lead is time before the proxy session end. The carbon reduction potential used for λ is calculated "
         "separately for each PV size and site-week. PV and battery capacities use the units in Table 3. "
         "The 0.95 and 0.99 targets are post hoc. Values are rounded to two decimal places. "
         "All displayed cases meet final energy and modeled budget.")
    influence = SUMMARY["influence"]
    rows = []
    for label, name in (("Omit one market-week block", "leave_one_market_week"), ("Omit one city", "leave_one_city")):
        ranges = [influence[str(y)][name]["median_carbon_gain_g_per_kwh"] for y in YEARS]
        rows.append((label, *[f"{lo:.2f}–{hi:.2f}" for lo, hi in ranges]))
    checks = []
    for label, metric in (("Window effect (g CO₂/kWh)", "window_decrease_g_per_kwh"),
                          ("Fixed-service gain (g CO₂/kWh)", "fixed_service_gain_g_per_kwh"),
                          ("Strict shortfall reduction (pp)", "strict_shortfall_reduction_pp")):
        intervals = [BLOCKS["years"][str(y)][metric] for y in YEARS]
        checks.append((label, *[f"{r['estimate']:.2f} [{r['lower']:.2f}, {r['upper']:.2f}]" for r in intervals]))
    wear = WEAR["median_fixed_service_gain_g_per_kwh"]
    table(d, "Table 5. Block uncertainty and sensitivity of the primary comparisons",
          ("Estimate or check", "2023", "2024"),
          [*checks,
           ("Imports limited to immediate charging peak", *[f"{carbon(y, design='grid_peak'):.2f}" for y in YEARS]),
           ("Common cohort at 11 kW", *[f"{carbon(y, design='common_11kw'):.2f}" for y in YEARS]),
           ("Common cohort at 22 kW", *[f"{carbon(y, design='common_22kw'):.2f}" for y in YEARS]),
           ("Throughput charge £0 or £0.03/kWh", *[f"{wear[f'{y}|wear_0.03']:.2f}" for y in YEARS]),
           ("85% efficiency, four-hour battery", *[f"{wear[f'{y}|85pct_4h']:.2f}" for y in YEARS]), *rows],
          (3.20, 1.65, 1.65), start_page=True)
    p(d, "Brackets give 95% market-week bootstrap intervals. Remaining rows show fixed-service gains in g CO₂/kWh. "
         "The omission rows give the range of medians. The two alternative throughput charges produced the same yearly medians.")
    policies = (("immediate", "Immediate charging"), ("tariff", "Tariff-only rolling"),
                ("carbon", "Carbon rolling"), ("perfect_signals", "Carbon rolling, hindsight"))
    rows = []
    for f, capacity in ((0., 0), (.5, 50)):
        variants = [(CONTROL, key, label) for key, label in policies]
        variants += [(ONLINE, key, label + ", online") for key, label in policies[:3]]
        for source, key, label in variants:
            group = source["groups"][f"{key}|{f}"]
            rows.append((capacity, label, f"{group['pooled_carbon_g_per_kwh']:.2f}",
                         f"{group['pooled_cost_gbp_per_kwh']:.4f}",
                         f"{group['equal_realized_cost_oracle_gap_g_per_kwh_quartiles'][1]:.2f}",
                         f"{group['within_budget_weeks']}/70"))
    table(d, "Table 6. Operating policy benchmarks on 70 matched site-weeks in 2024",
          ("Battery", "Policy", "Carbon", "Cost", "Oracle gap", "Within cap"), rows,
          (.65, 2.15, .85, .8, 1.05, 1.0), start_page=True)
    p(d, "Battery in kWh per 100 kWh/day of reference demand. Carbon and cost are pooled per delivered kWh "
         "in g CO₂ and GBP. Oracle gap is the median excess in g CO₂/kWh above the replay-cost-capped offline optimum. "
         "Forecasts are frozen except in the online and hindsight rows. See Table S3 for paired contrasts.")


def main():
    doc = setup()
    for part in (front, background, design, methods, results, end):
        part(doc)
    doc.save(ROOT / "JTE_Part_A_Technical_Paper.docx")


if __name__ == "__main__":
    main()

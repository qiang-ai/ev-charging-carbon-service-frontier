"""Summarize paired service experiments without independent-row inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT.parent / "cross_year_2024"
YEARS = (2023, 2024)
SCALES = (0.0, 0.2, 0.4)
MINUTES = (15, 30, 60)
FRACTIONS = (0.125, 0.5)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def panel_key(row):
    return row["site_id"], int(row["month"]), int(row["week_rank"])


def summarize(values):
    values = np.asarray(values)
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return dict(q25=float(q25), median=float(median), q75=float(q75))


def assemble(rows):
    names = ("year", "month", "week_rank", "checkpoint_min", "pv_scale", "battery_fraction")
    table = {}
    cohorts = {year: set() for year in YEARS}
    for row in rows:
        year, month, rank, minute = [int(row[key]) for key in names[:4]]
        scale, fraction = [float(row[key]) for key in names[4:]]
        site, design, objective = (row[key] for key in ("site_id", "design", "objective"))
        key = year, (site, month, rank), design, minute, scale, fraction, objective
        if key in table:
            raise ValueError(f"Repeated scenario {key}")
        table[key] = row
        if design == "main":
            cohorts[year].add(key[1])
    return table, cohorts


def calculate(table, keys, year, design, minute, scale, fraction):
    pairs = []
    for key in sorted(keys):
        prefix = year, key, design, minute, scale
        if (*prefix, 0., "common_carbon") not in table:
            continue
        control, treatment = [table[*prefix, f, "common_carbon"] for f in (0., fraction)]
        service_control, service_treatment = [table[*prefix, f, "preserve_readiness"] for f in (0., fraction)]
        energy = float(control["energy_kwh"])
        control_c = float(service_control["carbon_kg"])
        treatment_c = float(service_treatment["carbon_kg"])
        early0 = float(control["avoidable_shortfall_kwh"])
        early1 = float(treatment["avoidable_shortfall_kwh"])
        pairs.append({
            "year": year, "case_id": control["case_id"], "site_id": key[0], "month": key[1], "week_rank": key[2],
            "city": control["city"], "design": design, "checkpoint_min": minute, "pv_scale": scale,
            "battery_fraction": fraction, "sessions": int(control["n_sessions"]), "energy_kwh": energy,
            "physical_kwh": float(control["physical_early_shortfall_kwh"]),
            "shortfall_no_battery_kwh": early0, "shortfall_battery_kwh": early1,
            "readiness_gain_pp": 100 * (early0 - early1) / energy,
            "early_full_no_battery": int(control["early_full_count"]),
            "early_full_battery": int(treatment["early_full_count"]),
            "early_full_asap": int(control["asap_early_full_count"]),
            "readiness_floor_carbon_no_battery_kg": control_c,
            "readiness_floor_carbon_battery_kg": treatment_c,
            "carbon_gain_g_per_kwh": (control_c - treatment_c) * 1000 / energy,
            "carbon_gain_pct": 100 * (control_c - treatment_c) / control_c if control_c > 1e-8 else 0,
            "preserve_no_storage_target_share": (float(np.clip(
                (float(control["fixed_carbon_kg"]) - control_c) /
                (float(control["fixed_carbon_kg"]) - float(control["best_no_battery_carbon_kg"])), 0, 1))
                if float(control["fixed_carbon_kg"]) - float(control["best_no_battery_carbon_kg"]) > 1e-6 else 1),
            "preserve_storage_target_share": (float(np.clip(
                (float(control["fixed_carbon_kg"]) - treatment_c) /
                (float(control["fixed_carbon_kg"]) - float(control["best_no_battery_carbon_kg"])), 0, 1))
                if float(control["fixed_carbon_kg"]) - float(control["best_no_battery_carbon_kg"]) > 1e-6 else 1),
            "target_reduction_pct": (100 * (1 - float(control["carbon_cap_kg"]) / float(control["fixed_carbon_kg"]))
                                     if float(control["fixed_carbon_kg"]) > 1e-8 else 0),
        })
    return pairs


def aggregate(pairs):
    total = lambda key: sum(row[key] for row in pairs)
    energy = total("energy_kwh")
    count = total("sessions")
    return {
        "site_weeks": len(pairs), "sessions": count, "energy_kwh": energy,
        "physical_shortfall_pct": 100 * total("physical_kwh") / energy,
        "shortfall_no_battery_pct": 100 * total("shortfall_no_battery_kwh") / energy,
        "shortfall_battery_pct": 100 * total("shortfall_battery_kwh") / energy,
        "readiness_gain_pp_pooled": 100 * (total("shortfall_no_battery_kwh") - total("shortfall_battery_kwh")) / energy,
        "readiness_gain_pp": summarize([row["readiness_gain_pp"] for row in pairs]),
        "positive_readiness_cases": sum(row["readiness_gain_pp"] > 1e-4 for row in pairs),
        "readiness_gain_at_least_half_pp": sum(row["readiness_gain_pp"] >= .5 for row in pairs),
        "early_full_no_battery_pct": 100 * total("early_full_no_battery") / count,
        "early_full_battery_pct": 100 * total("early_full_battery") / count,
        "early_full_asap_pct": 100 * total("early_full_asap") / count,
        "readiness_floor_carbon_gain_g_per_kwh": summarize([row["carbon_gain_g_per_kwh"] for row in pairs]),
        "carbon_gain_above_dual_threshold": sum(row["carbon_gain_g_per_kwh"] >= 5 and row["carbon_gain_pct"] >= 5
                                                for row in pairs),
        "target_reduction_pct": summarize([row["target_reduction_pct"] for row in pairs]),
        "full_potential_without_readiness_loss_no_battery": sum(row["preserve_no_storage_target_share"] >= .999999
                                                               for row in pairs),
        "full_potential_without_readiness_loss_battery": sum(row["preserve_storage_target_share"] >= .999999
                                                            for row in pairs),
        "median_compatible_potential_no_battery": summarize([row["preserve_no_storage_target_share"] for row in pairs]),
    }


def main():
    output = ROOT / "results"
    rows = read(output / "service_outcomes.csv")
    table, cohorts = assemble(rows)
    matched = cohorts[2023] & cohorts[2024]
    summary, pair_rows = {}, []
    absolute, influence, persistence = {}, {}, {}
    main_pairs = {}
    for year in YEARS:
        for design in ("main", "grid_peak", "common_11kw", "common_22kw"):
            combinations = ((m, p, b) for m in MINUTES for p in SCALES for b in FRACTIONS) if design == "main" else [(30, .2, .5)]
            for minute, scale, fraction in combinations:
                pairs = calculate(table, matched, year, design, minute, scale, fraction)
                key = f"{year}|{design}|{minute}|{scale}|{fraction}"
                summary[key] = aggregate(pairs)
                pair_rows.extend(pairs)
                if (design, minute, scale, fraction) == ("main", 30, .2, .5):
                    main_pairs[year] = pairs
        counts = {str(f): sum(table[year, k, "main", 30, .2, f, "fixed_10_percent"]["status"] == "optimal"
                              for k in matched) for f in (0., .125, .5)}
        absolute[year] = {**counts, "denominator": len(matched)}
        year_pairs = main_pairs[year]
        block_keys = {(row["month"], row["week_rank"]) for row in year_pairs}
        leave_block = [aggregate([row for row in year_pairs if (row["month"], row["week_rank"]) != block])
                       for block in sorted(block_keys)]
        leave_city = [aggregate([row for row in year_pairs if row["city"] != city])
                      for city in sorted({row["city"] for row in year_pairs})]
        influence[year] = {
            name: {metric: [min(values), max(values)] for metric, values in (
                ("readiness_gain_pp_pooled", [s["readiness_gain_pp_pooled"] for s in series]),
                ("median_carbon_gain_g_per_kwh", [s["readiness_floor_carbon_gain_g_per_kwh"]["median"] for s in series]))
            } for name, series in (("leave_one_market_week", leave_block), ("leave_one_city", leave_city))}
    for label, key, cut in (("both_positive_readiness", "readiness_gain_pp", .0001),
                            ("both_half_pp_readiness", "readiness_gain_pp", .5),
                            ("both_carbon_gain_5g", "carbon_gain_g_per_kwh", 5)):
        persistence[label] = sum(a[key] >= cut and b[key] >= cut for a, b in zip(main_pairs[2023], main_pairs[2024], strict=True))
    frontier = {}
    frontier_path = output / "frontier_outcomes.csv"
    if frontier_path.exists():
        grouped = {}
        for row in read(frontier_path):
            if panel_key(row) not in matched:
                continue
            key = (int(row["year"]), int(row["checkpoint_min"]), float(row["pv_scale"]),
                   float(row["target_share"]), float(row["battery_fraction"]))
            grouped.setdefault(key, []).append(row)
        for (year, minute, scale, target, fraction), values in sorted(grouped.items()):
            pooled = sum(float(r["avoidable_shortfall_kwh"]) for r in values) / sum(float(r["energy_kwh"]) for r in values)
            frontier[f"{year}|{minute}|{scale}|{target}|{fraction}"] = {
                "site_weeks": len(values), "pooled_avoidable_shortfall_pct": 100 * pooled,
                "site_weeks_with_avoidable_shortfall_above_half_pp": sum(float(r["shortfall_pct"]) >= .5 for r in values),
                "pooled_early_full_pct": 100 * sum(int(r["early_full_count"]) for r in values) / sum(int(r["n_sessions"]) for r in values),
                "site_week_shortfall_pct": summarize([float(r["shortfall_pct"]) for r in values]),
            }
    outcomes = {"matched_site_weeks_per_year": len(matched), "all_site_weeks": {y: len(cohorts[y]) for y in YEARS},
                "outcomes": len(rows), "status_counts": {k: sum(r["status"] == k for r in rows)
                                                         for k in sorted({r["status"] for r in rows})},
                "summaries": summary, "fixed_ten_percent_feasibility": absolute,
                "influence": influence, "persistence": persistence, "frontier": frontier}
    (output / "service_summary.json").write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    with (output / "matched_service_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    print(json.dumps({str(y): summary[f"{y}|main|30|0.2|0.5"] for y in YEARS}, indent=2))


if __name__ == "__main__":
    main()

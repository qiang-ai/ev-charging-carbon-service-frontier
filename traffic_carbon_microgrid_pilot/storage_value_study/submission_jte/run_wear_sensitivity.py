"""Repeat the primary fixed-service comparison at alternative wear and equipment settings."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from statistics import median

from run_service import HOLDOUT, PROTOCOL, ROOT, load_cases, make_model, read_csv, settings_for, sha256, solve, write_csv


WEAR = (0., .01, .03)
CASES = [(f"wear_{charge:.2f}", charge, .90, 2) for charge in WEAR] + [("85pct_4h", .01, .85, 4)]


def solve_week(task):
    case, sessions, signals, protocol, control = task
    cap = protocol["power_cap_kw"]
    baseline = float(control["carbon_kg"])
    budget = float(control["budget_gbp"])
    energy = float(control["energy_kwh"])
    rows = []
    for scenario, charge, efficiency, duration in CASES:
        settings = settings_for(case, protocol, .2, .5)
        settings = replace(settings, battery_wear=charge, roundtrip_efficiency=efficiency,
                           battery_kw=settings.battery_kwh / duration)
        model, before, asap = make_model(sessions, signals, cap, settings, 30)
        info, vector = solve(model, "carbon", budget, (-before, -asap), (case["case_id"], charge))
        if vector is None:
            raise RuntimeError(f"Unexpected fixed-service infeasibility: {case['case_id']}, {charge}")
        carbon = float(model.carbon @ vector)
        realized_cost = float(model.cost @ vector)
        service_error = max(0., float(max(asap - before @ vector)))
        if max(carbon - baseline, realized_cost - budget, service_error) > 1e-5:
            raise ValueError(f"Independent cost, service, or nesting check failed: {case['case_id']}, {charge}")
        rows.append({"year": case["year"], "case_id": case["case_id"], "scenario": scenario,
                     "wear_gbp_per_kwh": charge, "roundtrip_efficiency": efficiency, "duration_h": duration,
                     "energy_kwh": energy, "carbon_no_battery_kg": baseline, "carbon_battery_kg": carbon,
                     "cost_gbp": realized_cost, "budget_gbp": budget, "max_readiness_deficit_kwh": service_error,
                     "gain_g_per_kwh": 1000 * (baseline - carbon) / energy, **info})
    if any(a["carbon_battery_kg"] > b["carbon_battery_kg"] + 1e-5 for a, b in zip(rows[:2], rows[1:3])):
        raise ValueError(f"Nonmonotone cost sensitivity: {case['case_id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    root = ROOT / "results"
    pair_file = root / "matched_service_pairs.csv"
    service_file = root / "service_outcomes.csv"
    pairs = {r["case_id"]: r for r in read_csv(pair_file) if r["design"] == "main"
             and int(r["checkpoint_min"]) == 30 and float(r["pv_scale"]) == .2 and float(r["battery_fraction"]) == .5}
    controls = {r["case_id"]: r for r in read_csv(service_file)
                if r["case_id"] in pairs and r["objective"] == "preserve_readiness" and r["design"] == "main"
                and int(r["checkpoint_min"]) == 30 and float(r["pv_scale"]) == .2 and float(r["battery_fraction"]) == 0}
    assert len(controls) == len(pairs) == 140
    tasks = []
    for year, source in ((2023, "data_harmonized_2023"), (2024, "data")):
        _, cases, _ = load_cases(HOLDOUT / source, protocol, year)
        tasks.extend((case, sessions, signals, protocol, controls[case["case_id"]])
                     for case, sessions, signals, _, _ in cases if case["case_id"] in controls)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(solve_week, tasks), 1):
            rows.extend(result)
            if index % 20 == 0:
                print(f"[{index}/{len(tasks)}] wear sensitivity", flush=True)
    assert len(rows) == 140 * len(CASES)
    for row in rows:
        if row["scenario"] == "wear_0.01":
            expected = float(pairs[row["case_id"]]["readiness_floor_carbon_battery_kg"])
            if abs(row["carbon_battery_kg"] - expected) > 1e-5:
                raise ValueError(f"Main result did not reproduce: {row['case_id']}")
    output = root / "wear_sensitivity_outcomes.csv"
    write_csv(output, rows)
    summary = {f"{year}|{scenario}": median(r["gain_g_per_kwh"] for r in rows
                                           if r["year"] == year and r["scenario"] == scenario)
               for year in (2023, 2024) for scenario, *_ in CASES}
    index = {(r["case_id"], r["scenario"]): r["gain_g_per_kwh"] for r in rows}
    result = {"outcomes": len(rows), "median_fixed_service_gain_g_per_kwh": summary,
              "max_gain_change_0_to_3p_g_per_kwh": max(abs(index[case, "wear_0.00"] - index[case, "wear_0.03"])
                                                     for case in pairs),
              "max_readiness_deficit_kwh": max(r["max_readiness_deficit_kwh"] for r in rows),
              "sha256": {p.name: sha256(p) for p in (pair_file, service_file, output)}}
    (root / "wear_sensitivity_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

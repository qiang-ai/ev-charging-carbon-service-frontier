"""Bootstrap the paired station results by whole matched market-week blocks."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from statistics import median, quantiles

from analyze_service import HOLDOUT, ROOT, read


def main():
    results = ROOT / "results"
    primary = [r for r in read(results / "matched_service_pairs.csv")
               if r["design"] == "main" and int(r["checkpoint_min"]) == 30
               and float(r["pv_scale"]) == .2 and float(r["battery_fraction"]) == .5]
    windows = [r for r in read(HOLDOUT / "analysis_common_pv" / "matched_interaction_changes.csv")
               if float(r["pv_scale"]) == .2 and float(r["battery_fraction"]) == .5]
    frontier = {(r["case_id"], float(r["battery_fraction"])): r
                for r in read(results / "frontier_outcomes.csv")
                if int(r["checkpoint_min"]) == 30 and float(r["pv_scale"]) == .2
                and float(r["target_share"]) == 1.}
    blocks = defaultdict(lambda: {2023: [], 2024: []})
    pairs = {(int(r["year"]), r["case_id"]): r for r in primary}
    for window in windows:
        key = (int(window["month"]), int(window["week_rank"]))
        for year in (2023, 2024):
            case = window[f"case_id_{year}"]
            row = pairs[year, case]
            no_battery, battery = [frontier[case, f] for f in (0., .5)]
            blocks[key][year].append((
                -float(window[f"interaction_g_per_kwh_{year}"]),
                float(row["carbon_gain_g_per_kwh"]),
                float(no_battery["avoidable_shortfall_kwh"]) - float(battery["avoidable_shortfall_kwh"]),
                float(row["energy_kwh"]),
            ))
    assert len(blocks) == 7 and len(windows) == 70 and len(pairs) == 140

    def estimates(selected):
        return (median(r[0] for r in selected), median(r[1] for r in selected),
                100 * sum(r[2] for r in selected) / sum(r[3] for r in selected))

    keys = sorted(blocks)
    years = (2023, 2024)
    rng = random.Random(20261104)
    draws = 10000
    metrics = ("window_decrease_g_per_kwh", "fixed_service_gain_g_per_kwh", "strict_shortfall_reduction_pp")
    replicates = {y: [] for y in years}
    for _ in range(draws):
        sample = rng.choices(keys, k=len(keys))
        for year in years:
            replicates[year].append(estimates([r for key in sample for r in blocks[key][year]]))
    out = {}
    for year in years:
        observed = estimates([r for key in keys for r in blocks[key][year]])
        out[str(year)] = {}
        for metric, point, values in zip(metrics, observed, zip(*replicates[year]), strict=True):
            cutpoints = quantiles(values, n=40, method="inclusive")
            out[str(year)][metric] = {"estimate": point, "lower": cutpoints[0], "upper": cutpoints[-1]}
    data = {"draws": draws, "seed": 20261104, "market_week_blocks": len(keys),
            "site_weeks_per_block": {f"{m}|{w}": len(blocks[m, w][2023]) for m, w in keys},
            "interval": "95% percentile, same block draws in both years", "years": out}
    (results / "block_uncertainty.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

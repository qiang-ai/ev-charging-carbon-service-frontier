"""Stress-test readiness as the common carbon cap approaches the no-storage optimum."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from run_service import (
    ROOT, HOLDOUT, SERVICE, PROTOCOL, FEAS_TOL, csr_matrix, load_cases, make_model, np,
    read_csv, settings_for, sha256, solve, write_csv,
)


TARGETS = (0.75, 0.90, 1.00)
ENDPOINT_TARGETS = (0.95, 0.99)


def solve_week(task):
    case, sessions, sig, protocol, plan, anchors, targets = task
    energies = np.array([s.energy for s in sessions])
    total = float(energies.sum())
    budget = case["budget_gbp"]
    cap = protocol["power_cap_kw"]
    out = []
    for scale, (fixed, best) in anchors.items():
        saving = max(0, fixed - best)
        for minute in plan["checkpoints_minutes_before_proxy_end"]:
            for fraction in plan["battery_fractions"]:
                model, before, asap = make_model(sessions, sig, cap, settings_for(case, protocol, scale, fraction), minute)
                objective = -np.asarray(before.sum(axis=0)).ravel()
                matrix = csr_matrix(model.carbon.reshape(1, -1))
                for ratio in targets:
                    target = fixed - ratio * saving
                    label = case["case_id"], scale, minute, fraction, ratio
                    info, x = solve(model, objective, budget, (matrix, np.array([target + FEAS_TOL])), label)
                    if x is None:
                        raise RuntimeError(f"Carbon target must be feasible: {label}")
                    readiness = before @ x
                    avoidable = float(np.maximum(asap - readiness, 0).sum())
                    if float(np.max(readiness - asap)) > 2 * FEAS_TOL:
                        raise ValueError(f"Checkpoint exceeds physical upper bound: {label}")
                    out.append({
                        "case_id": case["case_id"], "year": case["year"], "city": case["city"],
                        "site_id": case["site_id"], "month": case["month"], "week_rank": case["week_rank"],
                        "pv_scale": scale, "checkpoint_min": minute, "battery_fraction": fraction,
                        "target_share": ratio, "carbon_cap_kg": target, "carbon_kg": float(model.carbon @ x),
                        "fixed_carbon_kg": fixed, "best_no_battery_carbon_kg": best, "energy_kwh": total,
                        "avoidable_shortfall_kwh": avoidable, "shortfall_pct": 100 * avoidable / total,
                        "operating_cost_gbp": float(model.cost @ x), "budget_gbp": budget,
                        "early_full_count": int((energies - readiness <= 1e-5).sum()),
                        "n_sessions": len(sessions), **info,
                    })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--near-endpoint", action="store_true",
                        help="Run the post hoc .95/.99 check at the primary PV and checkpoint")
    args = parser.parse_args()
    plan = json.loads(SERVICE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    targets = ENDPOINT_TARGETS if args.near_endpoint else TARGETS
    if args.near_endpoint:
        plan["checkpoints_minutes_before_proxy_end"] = [plan["primary_checkpoint_minutes"]]
        plan["pv_scales"] = [plan["primary_pv_scale"]]
    service = ROOT / "results" / "service_outcomes.csv"
    anchors = {(r["case_id"], float(r["pv_scale"])):
               (float(r["fixed_carbon_kg"]), float(r["best_no_battery_carbon_kg"]))
               for r in read_csv(service) if r["design"] == "main" and r["checkpoint_min"] == "30"
               and r["objective"] == "common_carbon" and float(r["battery_fraction"]) == 0}
    tasks = []
    for year, source in ((2023, "data_harmonized_2023"), (2024, "data")):
        _, cases, _ = load_cases(HOLDOUT / source, protocol, year)
        for case, sessions, sig, _, _ in cases:
            design = {scale: anchors[case["case_id"], scale] for scale in plan["pv_scales"]}
            tasks.append((case, sessions, sig, protocol, plan, design, targets))
    results = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for i, rows in enumerate(pool.map(solve_week, tasks) if pool else map(solve_week, tasks), 1):
            results.extend(rows)
            if i % 16 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {len(results)} frontier scenarios", flush=True)
    finally:
        if pool:
            pool.shutdown()
    stem = "endpoint_sensitivity" if args.near_endpoint else "frontier"
    output = ROOT / "results" / f"{stem}_outcomes.csv"
    write_csv(output, results)
    meta = {"site_weeks": len(tasks), "outcomes": len(results), "target_shares": targets,
            "pv_scales": plan["pv_scales"], "checkpoints_min": plan["checkpoints_minutes_before_proxy_end"],
            "service_outcomes_sha256": sha256(service), "frontier_outcomes_sha256": sha256(output),
            "role": ("Post hoc near-endpoint sensitivity" if args.near_endpoint else
                     "Exploratory extension following the planned half-potential target")}
    (output.parent / f"{stem}_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

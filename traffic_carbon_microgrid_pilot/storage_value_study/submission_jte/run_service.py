"""Readiness and carbon frontiers for the two-year public charging panel."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT.parent / "cross_year_2024"
sys.path.insert(0, str(HOLDOUT))
from run_holdout import PROTOCOL, load_cases, read_csv, sha256, write_csv
from model import FEAS_TOL, Model, Session, Settings, Signals, _solve, early_reference
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack

SERVICE = ROOT / "service_protocol.json"


def make_model(sessions, sig, cap, settings, minutes=None):
    cutoff = ({s.session_id: max(s.start, s.end - 60 * minutes) for s in sessions}
              if minutes is not None else None)
    config = "smart_pv_battery" if settings.battery_kwh > 0 else "smart_pv"
    model = Model(sessions, sig, cap, config, settings, force_mip=settings.battery_kwh > 0,
                  checkpoints=cutoff)
    if cutoff is None:
        return model, None, None
    pairs = [(i, c) for i, s in enumerate(sessions) for c in model.before_checkpoint[s.session_id]]
    row, col = zip(*pairs) if pairs else ([], [])
    before = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(sessions), model.nv)).tocsr()
    asap = np.array([min(s.energy, min(s.rated, cap) * (cutoff[s.session_id] - s.start) / 3600)
                     for s in sessions])
    return model, before, asap


def settings_for(case, protocol, scale, fraction):
    daily = case["site_daily_energy_kwh"]
    capacity = daily * fraction
    return Settings(pv_kwp=daily * scale, battery_kwh=capacity, battery_kw=capacity / protocol["battery_hours"],
                    roundtrip_efficiency=protocol["battery_roundtrip_efficiency"],
                    initial_soc_fraction=protocol["initial_soc_fraction"],
                    battery_wear=protocol["battery_wear_gbp_per_kwh_throughput"],
                    time_limit=45, mip_rel_gap=protocol["mip_rel_gap"])


def grid_rows(model, peak):
    cols = model.grid.ravel()
    return (coo_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)),
                       shape=(len(cols), model.nv)).tocsr(), np.full(len(cols), peak))


def append_cap(model, peak, rows):
    if peak is None:
        return rows
    matrix, rhs = grid_rows(model, peak)
    if rows is None:
        return matrix, rhs
    return vstack([matrix, rows[0]], format="csc"), np.concatenate([rhs, rows[1]])


def solve(model, objective, budget, extra, label):
    info, x = _solve(model, objective, budget, extra=extra)
    if info["status"] not in {"optimal", "infeasible"} or (info["status"] == "optimal" and x is None):
        raise RuntimeError(f"{label}: {info}")
    return info, x


def carbon_anchors(case, sessions, sig, cap, settings, budget, peak, cached):
    if cached is not None:
        return cached
    fixed = Model(sessions, sig, cap, "fixed_pv", settings)
    smart, _, _ = make_model(sessions, sig, cap, settings)
    values = []
    for model in (fixed, smart):
        _, x = solve(model, "carbon", budget, append_cap(model, peak, None), case["case_id"])
        if x is None:
            raise RuntimeError(f"Infeasible reference in {case['case_id']}")
        values.append(float(model.carbon @ x))
    return values


def describe(model, before, asap, energies, info, x, carbon_cap, store_solution):
    summary = {**info, "carbon_cap_kg": carbon_cap}
    if x is None:
        return summary
    readiness = np.asarray(before @ x)
    avoidable = np.maximum(asap - readiness, 0)
    full = energies - readiness <= 1e-5
    eligible = energies - asap <= 1e-5
    summary.update(
        carbon_kg=float(model.carbon @ x), operating_cost_gbp=float(model.cost @ x),
        grid_kwh=float(x[model.grid].sum()), grid_peak_kw=float(x[model.grid].max() * 2),
        early_energy_kwh=float(readiness.sum()), avoidable_shortfall_kwh=float(avoidable.sum()),
        early_full_count=int(full.sum()), asap_early_full_count=int(eligible.sum()),
        early_full_on_asap_feasible_count=int((full & eligible).sum()),
        max_early_reference_excess_kwh=float(max(0, np.max(readiness - asap))),
    )
    if store_solution:
        return summary, x
    return summary


def run_design(case, sessions, sig, protocol, plan, design, cap, scales, minutes, use_peak, anchors):
    energies = np.array([s.energy for s in sessions])
    immediate, _ = early_reference(sessions, sig, cap, [case["site_id"]])
    budget = float(immediate[0] @ sig.price / 1000)
    peak = float(immediate[0].max()) if use_peak else None
    for scale in scales:
        settings = settings_for(case, protocol, scale, 0)
        fixed_c, best_c = carbon_anchors(case, sessions, sig, cap, settings, budget, peak, anchors.get(scale))
        if best_c > fixed_c + FEAS_TOL:
            raise ValueError(f"Flexible operation must nest immediate operation in {case['case_id']}")
        mid = fixed_c - plan["common_target_share_of_available_no_storage_reduction"] * max(0, fixed_c - best_c)
        for minute in minutes:
            for fraction in plan["battery_fractions"]:
                if design != "main" and fraction not in (0, plan["primary_battery_fraction"]):
                    continue
                settings = settings_for(case, protocol, scale, fraction)
                model, before, asap = make_model(sessions, sig, cap, settings, minute)
                primary = (design == "main" and minute == plan["primary_checkpoint_minutes"]
                           and scale == plan["primary_pv_scale"])
                label = (case["case_id"], design, scale, minute, fraction)
                row = {key: case[key] for key in ("case_id", "year", "month", "week_rank", "city", "site_id")}
                row.update(design=design, checkpoint_min=minute, pv_scale=scale, battery_fraction=fraction,
                           power_ceiling_kw=cap, n_sessions=len(sessions), energy_kwh=float(energies.sum()),
                           budget_gbp=budget, peak_import_limit_kw=None if peak is None else 2 * peak,
                           pv_kwp=settings.pv_kwp, battery_kwh=settings.battery_kwh,
                           fixed_carbon_kg=fixed_c, best_no_battery_carbon_kg=best_c,
                           asap_checkpoint_kwh=float(asap.sum()),
                           physical_early_shortfall_kwh=float((energies - asap).sum()))
                c_early = -np.asarray(before.sum(axis=0)).ravel()
                # Separate outcomes: preserve the immediate-charging readiness floor, or minimize
                # energy shortfall under a common carbon cap. Every plan delivers E by proxy end.
                solves = [("preserve_readiness", "carbon", (-before, -asap), None),
                          ("common_carbon", c_early, (csr_matrix(model.carbon.reshape(1, -1)),
                                                      np.array([mid + FEAS_TOL])), mid)]
                if primary:
                    absolute = fixed_c * (1 - plan["fixed_reduction_sensitivity"])
                    solves.append(("fixed_10_percent", c_early, (csr_matrix(model.carbon.reshape(1, -1)),
                                                                 np.array([absolute + FEAS_TOL])), absolute))
                for objective_label, objective, rows, carbon_cap in solves:
                    info, x = solve(model, objective, budget, append_cap(model, peak, rows), (*label, objective_label))
                    if x is None and objective_label != "fixed_10_percent":
                        raise RuntimeError(f"Known-feasible case reported infeasible: {label}, {objective_label}")
                    store = primary and objective_label != "fixed_10_percent"
                    result = describe(model, before, asap, energies, info, x, carbon_cap, store)
                    if store:
                        result, vector = result
                    else:
                        vector = None
                    yield {**row, "objective": objective_label, **result}, vector


def solve_week(task):
    case, sessions, sig, protocol, plan, anchors, run_sensitivity = task
    main = list(run_design(case, sessions, sig, protocol, plan, "main", protocol["power_cap_kw"],
                           plan["pv_scales"], plan["checkpoints_minutes_before_proxy_end"], False, anchors))
    sensitivity = []
    if run_sensitivity:
        scale = [plan["primary_pv_scale"]]
        minute = [plan["primary_checkpoint_minutes"]]
        power = protocol["power_cap_kw"]
        sensitivity.extend(run_design(case, sessions, sig, protocol, plan, "grid_peak", power,
                                      scale, minute, True, {}))
        common = [s for s in sessions if s.energy <= min(s.rated, 11) * (s.end - s.start) / 3600 + FEAS_TOL]
        if common:
            for cap in (11, power):
                sensitivity.extend(run_design(case, common, sig, protocol, plan, f"common_{cap}kw", cap,
                                              scale, minute, False, {}))
    return case, main + sensitivity


def self_test():
    sig = Signals(np.array([0., 1800., 3600.]), np.array([500., 200., 50.]),
                  np.full(3, 100.), np.zeros(3))
    sessions = [Session("example", "site", "plug", 300, 4500, 1.5, 2)]
    model, before, asap = make_model(sessions, sig, 2, Settings(pv_kwp=0, battery_kwh=0), 35)
    assert np.allclose(asap, 7 / 6)
    assert np.allclose(sorted(model.upper[i] for i in model.before_checkpoint["example"]), [1 / 3, 5 / 6])
    _, x = solve(model, "carbon", 0.15, (-before, -asap), "test")
    assert np.allclose((before @ x)[0], 7 / 6)
    assert np.allclose(model.carbon @ x, (5 / 6) * .5 + (1 / 3) * .2 + (1 / 3) * .05)
    target = .25
    c = -np.asarray(before.sum(axis=0)).ravel()
    _, x = solve(model, c, .15, (csr_matrix(model.carbon.reshape(1, -1)), np.array([target])), "test")
    assert (before @ x)[0] < asap[0] and math.isclose(model.carbon @ x, target)
    _, x = solve(model, c, .15, (csr_matrix(model.carbon.reshape(1, -1)), np.array([.01])), "test")
    assert x is None
    print("PASS: exact intra-slot cutoff, ASAP shortfall floor, common carbon cap and infeasibility")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    plan = json.loads(SERVICE.read_text(encoding="utf-8"))
    tasks, manifests = [], {}
    for year, source, old in ((2023, "data_harmonized_2023", "results_harmonized_2023"),
                              (2024, "data", "results")):
        _, cases, _ = load_cases(HOLDOUT / source, protocol, year)
        anchor_dir = HOLDOUT / old
        manifests[year] = {name: sha256(anchor_dir / name) for name in ("cases.csv", "outcomes.csv", "manifest.json")}
        prior = {(r["case_id"], float(r["pv_scale"]), float(r["alpha"])): float(r["carbon_kg"])
                 for r in read_csv(anchor_dir / "outcomes.csv") if float(r["battery_fraction"]) == 0}
        prior_cases = {r["case_id"]: r for r in read_csv(anchor_dir / "cases.csv")}
        if {case["case_id"] for case, *_ in cases} != set(prior_cases):
            raise ValueError(f"Case set changed for {year}")
        for case, sessions, sig, _, _ in cases:
            key = case["case_id"]
            for field in ("n_sessions", "energy_kwh", "budget_gbp"):
                if abs(case[field] - float(prior_cases[key][field])) > FEAS_TOL:
                    raise ValueError(f"{field} changed for {key}")
            anchors = {p: (prior[key, p, 0.], prior[key, p, 1.]) for p in plan["pv_scales"]}
            tasks.append((case, sessions, sig, protocol, plan, anchors, True))
    if args.limit:
        tasks = tasks[:args.limit]
    out = args.output_dir
    vectors = out / "primary_solutions"
    vectors.mkdir(parents=True, exist_ok=True)
    solved = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for index, (case, entries) in enumerate(pool.map(solve_week, tasks) if pool else map(solve_week, tasks), 1):
            primary = {}
            for row, x in entries:
                solved.append(row)
                if x is not None:
                    primary[f'{row["objective"]}_{row["battery_fraction"]}'] = x
            safe = case["case_id"].replace("|", "__")
            np.savez_compressed(vectors / f"{safe}.npz", **primary)
            if index % 8 == 0 or index == len(tasks):
                print(f"[{index}/{len(tasks)}] {len(solved)} scenarios", flush=True)
    finally:
        if pool:
            pool.shutdown()
    write_csv(out / "service_outcomes.csv", solved)
    manifest = {"service_protocol_sha256": sha256(SERVICE), "holdout_protocol_sha256": sha256(PROTOCOL),
                "prior_results_sha256": manifests, "site_weeks": len(tasks), "outcomes": len(solved),
                "outcome_sha256": sha256(out / "service_outcomes.csv")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

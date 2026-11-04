"""Compare causal public charging controls with the full-information carbon optimum."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from run_service import (
    FEAS_TOL, HOLDOUT, PROTOCOL, ROOT, Model, Session, Signals, early_reference, load_cases,
    make_model, np, read_csv, settings_for, sha256, solve, write_csv,
)
from optimize import STEP, read_inputs


PLAN = ROOT / "controller_protocol.json"
TOL = 5e-6


def time_key(stamp):
    value = datetime.fromtimestamp(float(stamp), timezone.utc)
    return value.weekday() >= 5, 2 * value.hour + value.minute // 30


def forecast_for(case, sig, training):
    profile = training[case["city"], case["month"]]
    matrices = np.array([profile[time_key(t)] for t in sig.ts])
    return Signals(sig.ts.copy(), *matrices.T)


def train(year, cities, months):
    result = {}
    for city in cities:
        for month in months:
            _, signals, _ = read_inputs(HOLDOUT / "data_harmonized_2023" / f"{city}_{year}_{month:02d}")
            groups = {}
            for t, carbon, price, pv in zip(signals.ts, signals.carbon, signals.price, signals.pv, strict=True):
                groups.setdefault(time_key(t), []).append((carbon, price, pv))
            if len(groups) != 96:
                raise ValueError(f"Incomplete forecast training in {city} {month}")
            result[city, month] = {k: np.median(v, axis=0) for k, v in groups.items()}
    return result


def asap_chunks(s, left, right, times, cap, checkpoint, remaining):
    """Energy by slot, with an exact split at the service checkpoint."""
    begin = max(s.start, left)
    p = min(cap, s.rated)
    end = min(right, s.end, begin + remaining * 3600 / p)
    start = np.maximum(times, begin)
    stop = np.minimum(times + STEP, end)
    early_stop = np.minimum(stop, checkpoint)
    return p * np.maximum(0, stop - start) / 3600, p * np.maximum(0, early_stop - start) / 3600


def plan_window(case, active, new, sig, settings, cap, minute, soc, carry, policy):
    ts = sig.ts
    length = len(ts)
    zeros = np.zeros(length)
    e_prev, in_prev, out_prev = carry
    has_bess = settings.battery_kwh > 0
    target = settings.initial_soc_fraction * settings.battery_kwh
    if not active and abs(soc - target) <= FEAS_TOL:
        return {}, zeros, zeros, zeros, np.ones(length), 0., 0.
    sessions = [record for record, _, _ in active]
    # The inherited model requires a site even if storage recovery is the only task.
    model_sessions = sessions or [Session("recovery", case["site_id"], "recovery", ts[0], ts[0] + STEP, 0., cap)]
    by_id = {s.session_id: (i, threshold) for s, i, threshold in active}
    if has_bess:
        settings = replace(settings, initial_soc_fraction=float(np.clip(soc / settings.battery_kwh, 0, 1)))
    pv_limit = settings.pv_kwp * sig.pv * .5
    if policy == "immediate":
        config = "fixed_pv_battery" if has_bess else "fixed_pv"
        model = Model(model_sessions, sig, cap, config, settings, force_mip=has_bess)
        before = None
        base_load = model.fixed_load[0]
    else:
        # Shift the session's start to the current decision time but retain the original checkpoint.
        model, before, _ = make_model(model_sessions, sig, cap, settings, minute)
        base_load = e_prev + new
    if has_bess:
        end_var = int(model.soc[0, -1])
        model.lower[end_var] = model.upper[end_var] = target
    base_net = base_load + in_prev - out_prev
    if np.min(base_net) < -TOL:
        raise ValueError(f"Carry-forward plan violates site balance in {case['case_id']}")
    base_net = np.maximum(base_net, 0)
    if policy == "immediate" and not has_bess:
        baseline_cost = float((base_net - np.minimum(pv_limit, base_net)) @ sig.price / 1000)
        return {}, base_load, zeros, zeros, np.ones(length), baseline_cost, baseline_cost
    base_pv = np.where(sig.price < 0, 0., np.minimum(pv_limit, base_net))
    budget = float((base_net - base_pv) @ sig.price / 1000
                   + settings.battery_wear * (in_prev + out_prev).sum())
    extra = None
    if before is not None:
        readiness = np.array([by_id.get(s.session_id, (0, 0.))[1] for s in model_sessions])
        if np.any(readiness > FEAS_TOL):
            extra = -before, -readiness
    objective = "cost" if policy == "tariff" else "carbon"
    _, x = solve(model, objective, budget + FEAS_TOL, extra, (case["case_id"], policy, ts[0]))
    if x is None:
        raise ValueError(f"Feasible carry-forward schedule was lost: {case['case_id']} {policy} {ts[0]}")
    ev_plan = {}
    if model.smart:
        ev_load = zeros.copy()
        for s in sessions:
            flow = zeros.copy()
            early_flow = zeros.copy()
            early_cols = set(model.before_checkpoint[s.session_id])
            for col, slot, _ in model.session_columns[s.session_id]:
                energy = max(0., x[col])
                flow[slot] += energy
                if col in early_cols:
                    early_flow[slot] += energy
            ev_load += flow
            ev_plan[by_id[s.session_id][0]] = flow, early_flow
    else:
        ev_load = base_load
    charge = np.maximum(0, x[model.charge].ravel()) if has_bess else zeros
    discharge = np.maximum(0, x[model.discharge].ravel()) if has_bess else zeros
    max_pv = np.minimum(pv_limit, ev_load + charge - discharge)
    use_fraction = np.divide(x[model.pv_use].ravel(), max_pv, out=np.ones(length), where=max_pv > TOL)
    return ev_plan, ev_load, charge, discharge, np.clip(use_fraction, 0, 1), budget, float(model.cost @ x)


def replay(case, sessions, forecast, actual, settings, plan, policy, *, stop_at=None, forecast_at=None):
    n = len(forecast.ts)
    ticks = plan["decision_interval_slots"]
    horizon = plan["horizon_slots"]
    minute = plan["early_checkpoint_min"]
    cap = case["power_cap_kw"]
    capacity = settings.battery_kwh
    target = capacity * settings.initial_soc_fraction
    eta = np.sqrt(settings.roundtrip_efficiency)
    ts = forecast.ts
    energy = np.zeros((len(sessions), n))
    early = np.zeros_like(energy)
    delivered = np.zeros(len(sessions))
    checkpoints = np.array([max(s.start, s.end - 60 * minute) for s in sessions])
    energy_due = np.array([s.energy for s in sessions])
    readiness = np.minimum(energy_due, np.array([min(cap, s.rated) * (d - s.start) / 3600
                                                 for s, d in zip(sessions, checkpoints, strict=True)]))
    records = {name: np.zeros(n) for name in ("ev_plan", "charge_plan", "discharge_plan", "pv", "grid", "charge", "discharge")}
    soc = np.full(n + 1, np.nan)
    soc[0] = target
    decisions = []
    seen = set()
    last = n if stop_at is None else stop_at
    for k in range(0, last, ticks):
        end = min(k + horizon, n)
        step_end = min(k + ticks, last)
        t = ts[k]
        execute = slice(k, step_end)
        dt = step_end - k
        stop = ts[step_end - 1] + STEP
        fc = (forecast_at(k, end) if forecast_at is not None else
              Signals(*(getattr(forecast, name)[k:end] for name in ("ts", "carbon", "price", "pv"))))
        new_load = np.zeros(end - k)
        active = []
        for i, original in enumerate(sessions):
            remaining = energy_due[i] - delivered[i]
            if original.start <= t < original.end and remaining > TOL:
                residual = replace(original, start=t, energy=remaining)
                threshold = max(0., readiness[i] - delivered[i])
                if threshold > TOL and checkpoints[i] <= t:
                    raise ValueError(f"Missed readiness checkpoint in {case['case_id']}: {original.session_id}")
                active.append((residual, i, threshold))
                if i not in seen:
                    seen.add(i)
                    new_load += asap_chunks(residual, t, ts[end - 1] + STEP, fc.ts, cap, checkpoints[i], remaining)[0]
        plan_names = "ev_plan", "charge_plan", "discharge_plan"
        carry = tuple(records[name][k:end].copy() for name in plan_names)
        solution = plan_window(case, active, new_load, fc, settings, cap, minute, soc[k], carry, policy)
        ev_plan, future, charge, discharge, fraction, cap_gbp, cost_gbp = solution
        for name, value in zip(plan_names, (future, charge, discharge), strict=True):
            records[name][k:end] = value
        for i, original in enumerate(sessions):
            left = max(original.start, t)
            remaining = energy_due[i] - delivered[i]
            if left >= min(stop, original.end) or remaining <= TOL:
                continue
            if i in ev_plan:
                energy[i, execute], early[i, execute] = [v[:dt] for v in ev_plan[i]]
            elif policy == "immediate" or original.start > t:
                energy[i, execute], early[i, execute] = asap_chunks(
                    original, left, stop, ts[execute], cap, checkpoints[i], remaining)
            else:
                raise ValueError(f"Known session missing from plan in {case['case_id']}")
        delivered += energy[:, execute].sum(axis=1)
        records["charge"][execute] = charge[:dt]
        records["discharge"][execute] = discharge[:dt]
        net = energy[:, execute].sum(axis=0) + charge[:dt] - discharge[:dt]
        if np.min(net) < -TOL:
            raise ValueError(f"Negative realized site demand in {case['case_id']}")
        net = np.maximum(net, 0)
        pv = np.minimum(actual.pv[execute] * settings.pv_kwp * .5, net) * fraction[:dt]
        records["pv"][execute] = pv
        records["grid"][execute] = net - pv
        soc[k + 1:step_end + 1] = soc[k] + np.cumsum(eta * charge[:dt] - discharge[:dt] / eta)
        decisions.append((k, len(active), cap_gbp, cost_gbp))
    return {**records, "energy": energy, "early": early, "soc": soc, "readiness": readiness,
            "decisions": np.asarray(decisions)}


def score(case, sessions, actual, settings, trace):
    energy = np.array([s.energy for s in sessions])
    delivered = trace["energy"].sum(axis=1)
    early = trace["early"].sum(axis=1)
    grid, charge, discharge, soc = [trace[k] for k in ("grid", "charge", "discharge", "soc")]
    residuals = (
        np.max(np.abs(delivered - energy)), np.max(np.maximum(trace["readiness"] - early, 0)),
        abs(soc[-1] - soc[0]), max(0, -np.min(soc), np.max(soc) - settings.battery_kwh),
        np.max(np.minimum(charge, discharge)),
    )
    if max(residuals) > 1e-4:
        raise ValueError(f"Service or physical check failed for {case['case_id']}: {residuals}")
    carbon = float(grid @ actual.carbon / 1000)
    cost = float(grid @ actual.price / 1000 + settings.battery_wear * (charge + discharge).sum())
    return {
        "carbon_kg": carbon, "carbon_g_per_kwh": 1000 * carbon / energy.sum(),
        "cost_gbp": cost, "within_week_budget": cost <= case["budget_gbp"] + 1e-5,
        "budget_excess_gbp": max(0., cost - case["budget_gbp"]),
        "max_session_delivery_error_kwh": residuals[0], "max_avoidable_checkpoint_error_kwh": residuals[1],
        "terminal_soc_error_kwh": residuals[2], "max_soc_bound_error_kwh": residuals[3],
        "max_simultaneous_kwh": residuals[4],
        "max_forecast_budget_excess_gbp": float(max(0, np.max(trace["decisions"][:, 3] - trace["decisions"][:, 2]))),
    }


def solve_case(task, *, forecast_at=None):
    case, sessions, actual, fc, original, plan, anchor = task
    scale, cap = plan["pv_scale"], case["power_cap_kw"]
    rows, traces = [], {}
    for fraction in plan["battery_fractions"]:
        settings = settings_for(case, original, scale, fraction)
        oracle, before, asap = make_model(sessions, actual, cap, settings, plan["early_checkpoint_min"])
        for policy in plan["policies"]:
            trace = replay(case, sessions, fc, actual, settings, plan, policy, forecast_at=forecast_at)
            scores = score(case, sessions, actual, settings, trace)
            _, equal_cost = solve(oracle, "carbon", scores["cost_gbp"] + FEAS_TOL, (-before, -asap),
                                  (case["case_id"], fraction, policy, "equal_cost_oracle"))
            if equal_cost is None:
                raise ValueError(f"The online schedule certifies a feasible equal-cost oracle for {case['case_id']}")
            same_cost_carbon = float(oracle.carbon @ equal_cost)
            rows.append({key: case[key] for key in ("case_id", "site_id", "city", "year", "month", "week_rank",
                                                     "energy_kwh", "n_sessions", "budget_gbp")} |
                        {"policy": policy, "battery_fraction": fraction, **scores,
                         "oracle_shared_budget_carbon_kg": anchor[fraction],
                         "oracle_equal_realized_cost_carbon_kg": same_cost_carbon})
            for key in ("energy", "early", "soc", "pv", "grid", "charge", "discharge", "decisions"):
                traces[f"{policy}_{fraction}_{key}"] = trace[key]
    return case["case_id"], rows, traces


def prefix_test(case, sessions, fc, actual, original, plan):
    settings = settings_for(case, original, plan["pv_scale"], plan["battery_fractions"][-1])
    start = max(sessions[0].start, fc.ts[0])
    stride = plan["decision_interval_slots"]
    stop = min(len(fc.ts) - stride, int(np.ceil((start - fc.ts[0]) / (STEP * stride))) * stride + 2 * stride)
    boundary = fc.ts[stop]
    current = [s for s in sessions if s.start < boundary]
    future = [s for s in sessions if s.start >= boundary]
    assert current and future
    changed = [*current, *(replace(s, energy=s.energy * .5) for s in future)]
    signals = Signals(actual.ts, actual.carbon + 55, actual.price - 30, actual.pv.copy())
    signals.pv[stop:] = 2 * signals.pv[stop:]
    a = replay(case, sessions, fc, actual, settings, plan, "carbon", stop_at=stop)
    b = replay(case, changed, fc, signals, settings, plan, "carbon", stop_at=stop)
    for key in ("grid", "pv", "charge", "discharge", "decisions"):
        np.testing.assert_array_equal(a[key], b[key])
    old_order = {s.session_id: i for i, s in enumerate(sessions)}
    for i, s in enumerate(current):
        for key in ("energy", "early"):
            np.testing.assert_array_equal(a[key][old_order[s.session_id]], b[key][i])
    print(f"PASS: changing future arrivals and realized signals leaves the {stop}-slot decision prefix unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--perfect-signals", action="store_true",
                        help="Diagnostic only: give the carbon planner future realized signals but hide future arrivals")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if args.perfect_signals:
        plan["policies"] = ["carbon"]
    original = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _, cases, _ = load_cases(HOLDOUT / "data", original, plan["evaluation_year"])
    matched = {r["case_id_2024"] for r in read_csv(HOLDOUT / "analysis_common_pv" / "matched_interaction_changes.csv")}
    training = (None if args.perfect_signals else
                train(plan["training_year"], {r[0]["city"] for r in cases}, original["months"]))
    rows = read_csv(ROOT / "results" / "service_outcomes.csv")
    anchor = {(r["case_id"], float(r["battery_fraction"])): float(r["carbon_kg"]) for r in rows
              if r["year"] == "2024" and r["design"] == "main" and r["objective"] == "preserve_readiness"
              and int(r["checkpoint_min"]) == plan["early_checkpoint_min"] and float(r["pv_scale"]) == plan["pv_scale"]}
    tasks = [(case, sessions, sig, sig if training is None else forecast_for(case, sig, training), original, plan,
              {f: anchor[case["case_id"], f] for f in plan["battery_fractions"]})
             for case, sessions, sig, _, _ in cases if case["case_id"] in matched]
    assert len(tasks) == len(matched) == 70
    if training is not None:
        case, sessions, actual, fc, *_ = tasks[0]
        prefix_test(case, sessions, fc, actual, original, plan)
    tasks = tasks[:args.limit] if args.limit else tasks
    out = args.output_dir or ROOT / "results" / ("signal_diagnostic" if args.perfect_signals else "controllers")
    vectors = out / "traces"
    vectors.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    solved = []
    try:
        for i, (case_id, result, traces) in enumerate(pool.map(solve_case, tasks) if pool else map(solve_case, tasks), 1):
            solved.extend(result)
            np.savez_compressed(vectors / (case_id.replace("|", "__") + ".npz"), **traces)
            if i % 5 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {len(solved)} controller weeks", flush=True)
    finally:
        if pool:
            pool.shutdown()
    write_csv(out / "controller_outcomes.csv", solved)
    source = HOLDOUT / "data_harmonized_2023"
    training_hashes = {p.parent.name: sha256(p) for p in sorted(source.glob("*/signals.csv"))} if training is not None else {}
    meta = {"cases": len(tasks), "outcomes": len(solved), "protocol_sha256": sha256(PLAN),
            "training_signal_sha256": training_hashes,
            "outcomes_sha256": sha256(out / "controller_outcomes.csv"),
            "information_prefix_test": "pass" if training is not None else "not_applicable_future_signals_supplied",
            "planner_signals": "realized_diagnostic" if args.perfect_signals else "2023_frozen_medians"}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

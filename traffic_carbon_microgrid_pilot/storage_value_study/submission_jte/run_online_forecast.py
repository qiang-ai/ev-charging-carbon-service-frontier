"""Replay the matched 2024 weeks with a past-observation forecast benchmark."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import json
from pathlib import Path

from analyze_controllers import compare, score_group
from audit_controllers import TRACE_FIELDS, case_arrays, check_trace
from audit_service import Audit, load_frozen_inputs, stamp
from run_controllers import (
    HOLDOUT, PLAN, PROTOCOL, ROOT, STEP, Signals, forecast_for, load_cases,
    np, read_csv, read_inputs, replay, settings_for, sha256, solve_case, train, write_csv,
)


DEFAULT_OUTPUT = ROOT / "results" / "online_forecast"
RELEASE_DELAY_SECONDS = 2 * 3600
DAY_SECONDS = 24 * 3600
FIELDS = ("carbon", "price", "pv")


@dataclass
class DailyLag:
    frozen: Signals
    history: Signals

    def sources(self, start, end):
        targets = self.frozen.ts[start:end]
        # A historical slot is available only after its end and the release buffer.
        latest_start = targets[0] - STEP - RELEASE_DELAY_SECONDS
        lag = np.where(targets - DAY_SECONDS <= latest_start, DAY_SECONDS, 2 * DAY_SECONDS)
        stamps = targets - lag
        positions = np.searchsorted(self.history.ts, stamps)
        if np.any(positions >= len(self.history.ts)):
            raise ValueError("The requested historical month does not cover the decision grid")
        present = self.history.ts[positions] == stamps
        return np.where(present, positions, -1)

    def __call__(self, start, end):
        sources = self.sources(start, end)
        valid = sources >= 0
        carbon = self.frozen.carbon[start:end].copy()
        price = self.frozen.price[start:end].copy()
        carbon[valid] = self.history.carbon[sources[valid]]
        price[valid] = self.history.price[sources[valid]]
        return Signals(self.frozen.ts[start:end], carbon, price, self.frozen.pv[start:end])


def solve_online_case(task_and_forecast):
    task, online = task_and_forecast
    return solve_case(task, forecast_at=online)


def select_tasks():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _, cases, _ = load_cases(HOLDOUT / "data", protocol, plan["evaluation_year"])
    matched = {r["case_id_2024"] for r in read_csv(HOLDOUT / "analysis_common_pv" / "matched_interaction_changes.csv")}
    groups = {(case["city"], case["month"]) for case, *_ in cases}
    training = train(plan["training_year"], {city for city, _ in groups}, protocol["months"])
    histories = {
        (city, month): read_inputs(HOLDOUT / "data" / f"{city}_{plan['evaluation_year']}_{month:02d}")[1]
        for city, month in groups
    }
    anchors = {(r["case_id"], float(r["battery_fraction"])): float(r["carbon_kg"])
               for r in read_csv(ROOT / "results" / "service_outcomes.csv")
               if r["case_id"] in matched and r["design"] == "main" and r["objective"] == "preserve_readiness"
               and int(r["checkpoint_min"]) == plan["early_checkpoint_min"] and float(r["pv_scale"]) == plan["pv_scale"]}
    tasks = []
    for case, sessions, actual, _, _ in cases:
        if case["case_id"] in matched:
            frozen = forecast_for(case, actual, training)
            shared = {f: anchors[case["case_id"], f] for f in plan["battery_fractions"]}
            tasks.append(((case, sessions, actual, frozen, protocol, plan, shared),
                          DailyLag(frozen, histories[case["city"], case["month"]])))
    if {task[0][0]["case_id"] for task in tasks} != matched or len(tasks) != 70:
        raise ValueError("Expected all 70 distinct original matched holdout cases")
    return plan, protocol, tasks


def information_audit(task, online):
    case, sessions, actual, frozen, protocol, plan, _ = task
    stride = plan["decision_interval_slots"]
    first_active = int(np.searchsorted(frozen.ts, min(s.start for s in sessions)))
    boundary = min(len(frozen.ts) - stride, max(2 * stride, (first_active // stride + 4) * stride))
    boundary_time = frozen.ts[boundary]
    current = [s for s in sessions if s.start < boundary_time]
    future = [s for s in sessions if s.start >= boundary_time]
    if not current or not future:
        raise ValueError("The information test needs sessions before and after the boundary")
    future_sessions = [*current, *(replace(s, energy=s.energy * .55) for s in future)]

    def perturb(signals):
        after = signals.ts >= boundary_time
        return Signals(signals.ts, np.where(after, signals.carbon + 130, signals.carbon),
                       np.where(after, signals.price - 70, signals.price),
                       np.where(after, signals.pv * 2, signals.pv))

    rewritten = DailyLag(frozen, perturb(online.history))
    case_actual = perturb(actual)
    for k in range(0, boundary, stride):
        end = min(k + plan["horizon_slots"], len(frozen.ts))
        for field in FIELDS:
            np.testing.assert_array_equal(getattr(online(k, end), field), getattr(rewritten(k, end), field))
    for f in plan["battery_fractions"]:
        settings = settings_for(case, protocol, plan["pv_scale"], f)
        for policy in plan["policies"]:
            a = replay(case, sessions, frozen, actual, settings, plan, policy, stop_at=boundary, forecast_at=online)
            b = replay(case, future_sessions, frozen, case_actual, settings, plan, policy,
                       stop_at=boundary, forecast_at=rewritten)
            for field in ("grid", "pv", "charge", "discharge", "decisions"):
                np.testing.assert_array_equal(a[field], b[field])
            keep = [i for i, s in enumerate(sessions) if s.start < boundary_time]
            for field in ("energy", "early"):
                np.testing.assert_array_equal(a[field][keep], b[field][:len(current)])
    return {"case_id": case["case_id"], "prefix_slots": boundary, "combinations_checked": 6,
            "perturbations": ["future arrivals' requested energy", "future national carbon", "future price", "future PV"],
            "all_decision_prefixes_identical": True}


def forecast_diagnostics(tasks):
    unique = {}
    for task, online in tasks:
        case, _, actual, _, _, plan, _ = task
        unique[case["city"], case["month"], case["week_rank"]] = (actual, online)
    metrics = defaultdict(list)
    counts = defaultdict(Counter)
    shortest_age = float("inf")
    for (city, month, _), (actual, online) in sorted(unique.items()):
        history = {t: (ci, price) for t, ci, price in
                   zip(online.history.ts, online.history.carbon, online.history.price, strict=True)}
        frozen = online.frozen
        group = f"{city}|{month}"
        stride = plan["decision_interval_slots"]
        for k in range(0, len(actual.ts), stride):
            end = min(k + plan["horizon_slots"], len(actual.ts))
            prediction = online(k, end)
            decision = actual.ts[k]
            source_indices = online.sources(k, end)
            for offset, t in enumerate(actual.ts[k:end]):
                expected_source = next((t - lag for lag in (DAY_SECONDS, 2 * DAY_SECONDS)
                                        if t - lag + STEP + RELEASE_DELAY_SECONDS <= decision
                                        and t - lag in history), None)
                observed_source = None if source_indices[offset] < 0 else online.history.ts[source_indices[offset]]
                if observed_source != expected_source:
                    raise AssertionError("Independent historical source audit failed")
                if observed_source is None:
                    reference = frozen.carbon[k + offset], frozen.price[k + offset]
                    category = "frozen_fallback"
                else:
                    reference = history[observed_source]
                    category = "prior_day" if t - observed_source == DAY_SECONDS else "two_days_prior"
                    shortest_age = min(shortest_age, decision - observed_source - STEP)
                np.testing.assert_array_equal([prediction.carbon[offset], prediction.price[offset]], reference)
                for horizon in ("full_horizon", "first_two_hours") if offset < stride else ("full_horizon",):
                    counts[group, horizon][category] += 1
                    metrics[group, horizon].append([
                        [abs(getattr(actual, f)[k + offset] - getattr(prediction, f)[offset]) for f in FIELDS],
                        [abs(getattr(actual, f)[k + offset] - getattr(frozen, f)[k + offset]) for f in FIELDS],
                    ])
    results = {}
    for (group, horizon), errors in metrics.items():
        sums = np.mean(errors, axis=0)
        results.setdefault(group, {})[horizon] = {
            "targets": len(errors), "source_counts": counts[group, horizon],
            "online": dict(zip(FIELDS, map(float, sums[0]), strict=True)),
            "frozen": dict(zip(FIELDS, map(float, sums[1]), strict=True)),
        }
    return {"by_city_month": results, "unique_city_weeks": len(unique),
            "minimum_source_age_after_slot_end_seconds": shortest_age}


def check_results(out, rows, plan, protocol):
    audit = Audit()
    source = ROOT / "results" / "service_outcomes.csv"
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    audit.require(manifest["outcome_sha256"] == sha256(source), "offline oracle archive checksum")
    cases, sessions, _, inputs = load_frozen_inputs(manifest, protocol, audit)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["case_id"]].append(r)
    anchors = {(r["case_id"], float(r["battery_fraction"])): r for r in read_csv(source)
               if r["case_id"] in grouped and r["design"] == "main" and r["objective"] == "preserve_readiness"
               and int(r["checkpoint_min"]) == plan["early_checkpoint_min"] and float(r["pv_scale"]) == plan["pv_scale"]}
    for case_id, policies in grouped.items():
        case, records = cases[case_id], sessions[case_id]
        args = case_arrays(records, stamp(case["start_utc"]), stamp(case["end_utc_exclusive"]),
                           inputs[case["group"]][1], float(case["power_cap_kw"]), plan["early_checkpoint_min"])
        with np.load(out / "traces" / (case_id.replace("|", "__") + ".npz"), allow_pickle=False) as raw:
            for row in policies:
                p, f = row["policy"], float(row["battery_fraction"])
                audit.context = f"{case_id}/{p}/{f}"
                trace = {k: raw[f"{p}_{f}_{k}"] for k in TRACE_FIELDS}
                check_trace(audit, trace, row, args, float(case["site_daily_energy_kwh"]), plan, protocol,
                            anchors[case_id, f])
    return {"pass": True, "weeks": len(grouped), "replays": len(rows), "max_residuals": dict(audit.maxima)}


def summarize(out, rows, plan):
    frozen_rows = read_csv(ROOT / "results" / "controllers" / "controller_outcomes.csv")
    cases = {r["case_id"] for r in rows}
    def group(items, p, f):
        return sorted((r for r in items if r["case_id"] in cases and r["policy"] == p
                       and float(r["battery_fraction"]) == f), key=lambda r: r["case_id"])
    stats, comparisons, pairs = {}, {}, []
    for f in plan["battery_fractions"]:
        for p in plan["policies"]:
            stats[f"{p}|{f}"] = score_group(group(rows, p, f))
            summary, pair = compare(group(rows, p, f), group(frozen_rows, p, f), f"frozen_{p}")
            comparisons[f"{p}_vs_frozen|{f}"] = summary
            pairs.extend(pair)
        for p in ("tariff", "immediate"):
            summary, pair = compare(group(rows, "carbon", f), group(rows, p, f), f"online_{p}")
            comparisons[f"carbon_vs_{p}|{f}"] = summary
            pairs.extend(pair)
    write_csv(out / "controller_pairs.csv", pairs)
    return {"cases": len(cases), "outcomes": len(rows), "groups": stats, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Smoke-test prefix only, requires a separate output directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 1 or args.output_dir.resolve() == DEFAULT_OUTPUT.resolve()):
        parser.error("--limit needs a positive count and a separate --output-dir")
    plan, protocol, tasks = select_tasks()
    tasks = tasks[:args.limit]
    unique_weeks = {}
    for task, online in tasks:
        if online.sources(0, 1)[0] >= 0:
            unique_weeks.setdefault(task[0]["month"], (task, online))
    sample = list(unique_weeks.values()) or [tasks[0]]
    information = [information_audit(task, online) for task, online in sample]
    forecast = forecast_diagnostics(tasks)
    print(f"Forecast audit passed on {forecast['unique_city_weeks']} distinct city-weeks", flush=True)
    out = args.output_dir
    vectors = out / "traces"
    vectors.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    rows = []
    try:
        results = pool.map(solve_online_case, tasks) if pool else map(solve_online_case, tasks)
        for i, (case_id, result, traces) in enumerate(results, 1):
            rows.extend(result)
            np.savez_compressed(vectors / (case_id.replace("|", "__") + ".npz"), **traces)
            if i % 5 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {len(rows)} online controller replays", flush=True)
    finally:
        if pool:
            pool.shutdown()
    output = out / "controller_outcomes.csv"
    write_csv(output, rows)
    reread = read_csv(output)
    results = summarize(out, reread, plan)
    results["forecast_mae"] = forecast
    results["physical_audit"] = check_results(out, reread, plan, protocol)
    results["information_audit"] = information
    summary = out / "controller_summary.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    sources = [*sorted((HOLDOUT / "data").glob("*/signals.csv")),
               *sorted((HOLDOUT / "data_harmonized_2023").glob("*/signals.csv")),
               HOLDOUT / "analysis_common_pv" / "matched_interaction_changes.csv",
               ROOT / "results" / "service_outcomes.csv",
               ROOT / "results" / "controllers" / "controller_outcomes.csv", PLAN, PROTOCOL,
               ROOT / "run_controllers.py", Path(__file__).resolve()]
    manifest = {"cases": len(tasks), "outcomes": len(reread),
                "forecast": "Prior-day UTC slot persistence for carbon and price with 2 h post-slot release buffer. "
                            "Use the prior two days if necessary and the frozen 2023 profile when history is absent. "
                            "PV always uses the frozen profile.",
                "as_of_assumption": "Archived realized carbon and price are proxies for observations available after "
                                    "the release buffer; publication and revision vintages are not observed.",
                "input_sha256": {p.relative_to(HOLDOUT.parent).as_posix(): sha256(p) for p in sources},
                "outcome_sha256": sha256(output), "summary_sha256": sha256(summary),
                "trace_sha256": {p.name: sha256(p) for p in sorted(vectors.glob("*.npz"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(reread)} outcomes and full trace audit. {summary}")


if __name__ == "__main__":
    main()

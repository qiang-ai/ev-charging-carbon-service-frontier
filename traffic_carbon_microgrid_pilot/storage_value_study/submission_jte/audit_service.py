"""Read-only physical and design checks of the saved service solutions.

The decision vector column order is a file-format contract. All feasibility
and score checks below are reconstructed from the original time intervals and
signals without calling the optimization model or its postsolve checker.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT.parent / "cross_year_2024"
DEPS = ROOT.parents[1] / ".deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))

import numpy as np

STEP = 1800
TOL = 3e-6
MODEL_TOL = 1e-6
KEY_FIELDS = ("case_id", "design", "pv_scale", "checkpoint_min", "battery_fraction", "objective")
OBJECTIVES = ("preserve_readiness", "common_carbon")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def truth(text):
    return str(text).lower() == "true"


def scenario_key(row):
    return (row["case_id"], row["design"], float(row["pv_scale"]), int(row["checkpoint_min"]),
            float(row["battery_fraction"]), row["objective"])


class Audit:
    def __init__(self):
        self.maxima = defaultdict(float)
        self.context = "input"

    def require(self, condition, label):
        if not condition:
            raise ValueError(f"{self.context}: {label}")

    def error(self, error, label, limit=TOL):
        error = np.asarray(error, dtype=float)
        self.require(np.isfinite(error).all(), f"non-finite {label}")
        largest = max(0.0, float(error.max())) if error.size else 0.0
        self.maxima[label] = max(self.maxima[label], largest)
        self.require(largest <= limit, f"{label} residual {largest:.8g} exceeds {limit:.8g}")

    def close(self, actual, expected, label, limit=TOL):
        self.error(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)), label, limit)

    def le(self, actual, ceiling, label, limit=TOL):
        self.error(np.asarray(actual, dtype=float) - np.asarray(ceiling, dtype=float), label, limit)


def window_energy(session, starts, cap, end):
    left = np.maximum(starts, float(session["start"]))
    right = np.minimum(starts + STEP, end)
    return np.maximum(right - left, 0) * min(cap, float(session["rated"])) / 3600


def immediate(sessions, starts, cap):
    loads = []
    for s in sessions:
        end = float(s["start"]) + float(s["energy"]) * 3600 / min(cap, float(s["rated"]))
        loads.append(window_energy(s, starts, cap, end))
    return np.sum(loads, axis=0)


def cutoff_energy(sessions, cap, minutes):
    return np.array([min(float(s["energy"]), min(cap, float(s["rated"])) *
                         max(0, float(s["end"]) - float(s["start"]) - 60 * minutes) / 3600) for s in sessions])


def fits(s, cap):
    return float(s["energy"]) <= min(cap, float(s["rated"])) * (float(s["end"]) - float(s["start"])) / 3600 + MODEL_TOL


def fixed_carbon(load, pv_available, prices, carbon, budget):
    # Fixed EV load admits a scalar PV-curtailment problem. Using all PV gives
    # lowest carbon. Any remaining budget deficit can only be offset by
    # curtailment in negative-price slots, ordered by carbon cost per pound.
    pv = np.minimum(load, pv_available).copy()
    deficit = float((load - pv) @ prices) - budget
    if deficit > 1e-9:
        negative = np.flatnonzero((prices < 0) & (pv > 0))
        for j in sorted(negative, key=lambda j: carbon[j] / -prices[j]):
            reduction = min(pv[j], deficit / -prices[j])
            pv[j] -= reduction
            deficit -= reduction * -prices[j]
            if deficit <= 1e-9:
                break
    if deficit > TOL:
        raise ValueError("Independent fixed charging reference exceeds its budget")
    return float((load - pv) @ carbon)


def load_frozen_inputs(manifest, protocol, audit):
    cases, sessions, anchors, inputs = {}, defaultdict(list), {}, {}
    for year, data_name, result_name in ((2023, "data_harmonized_2023", "results_harmonized_2023"),
                                         (2024, "data", "results")):
        folder = HOLDOUT / result_name
        for name, expected in manifest["prior_results_sha256"][str(year)].items():
            audit.require(sha(folder / name) == expected, f"frozen hash mismatch for {year} {name}")
        source_data = HOLDOUT / data_name
        for case in read(folder / "cases.csv"):
            case_id = case["case_id"]
            audit.context = case_id
            audit.require(case_id not in cases and int(case["year"]) == year, "duplicate or wrong-year frozen case")
            a, b = stamp(case["start_utc"]), stamp(case["end_utc_exclusive"])
            start = datetime.fromtimestamp(a, timezone.utc)
            audit.require(start.weekday() == 0 and int(case["month"]) == start.month and b - a == 7 * 24 * 3600,
                          "case does not span a complete UTC week")
            audit.require((start.day - 1) // 7 + 1 == int(case["week_rank"]), "incorrect UTC calendar-week rank")
            audit.close(case["site_daily_energy_kwh"], protocol["sites"][case["site_id"]]["site_daily_energy_kwh"], "frozen equipment basis")
            cases[case_id] = case
            group = case["group"]
            if group not in inputs:
                raw = read(source_data / group / "sessions.csv")
                raw_map = {s["session_id"]: s for s in raw}
                audit.require(len(raw_map) == len(raw), "duplicate prepared session ID")
                sig = read(source_data / group / "signals.csv")
                inputs[group] = (raw_map, {stamp(r["timestamp_utc"]): r for r in sig})
        for row in read(folder / "case_sessions.csv"):
            case_id = row["case_id"]
            case = cases[case_id]
            audit.context = case_id
            source = inputs[case["group"]][0][row["session_id"]]
            audit.require(row["station_id"] == source["station_id"] == case["site_id"] and
                          row["connector_id"] == source["connector_id"], "source station/connector mismatch")
            audit.close([row["start"], row["end"], row["energy"], row["rated"]],
                        [stamp(source["start_utc"]), stamp(source["end_proxy_utc"]), source["energy_kwh"], source["power_kw"]],
                        "prepared-session lineage")
            sessions[case_id].append(row)
        for r in read(folder / "outcomes.csv"):
            if float(r["battery_fraction"]) == 0:
                anchors[r["case_id"], float(r["pv_scale"]), float(r["alpha"])] = float(r["carbon_kg"])
    return cases, sessions, anchors, inputs


def design_rows(case, sessions, plan, protocol):
    cap = protocol["power_cap_kw"]
    primary_pv = plan["primary_pv_scale"]
    primary_minutes = plan["primary_checkpoint_minutes"]
    primary_fraction = plan["primary_battery_fraction"]
    for pv in plan["pv_scales"]:
        for minute in plan["checkpoints_minutes_before_proxy_end"]:
            for fraction in plan["battery_fractions"]:
                objectives = list(OBJECTIVES)
                if pv == primary_pv and minute == primary_minutes:
                    objectives.append("fixed_10_percent")
                for objective in objectives:
                    yield (case, "main", pv, minute, fraction, objective), cap, sessions
    common = [s for s in sessions if fits(s, 11)]
    for design, limit, cohort in (("grid_peak", cap, sessions), ("common_11kw", 11, common), ("common_22kw", cap, common)):
        if not cohort:
            continue
        for fraction in (0, primary_fraction):
            for objective in OBJECTIVES:
                yield (case, design, primary_pv, primary_minutes, fraction, objective), limit, cohort


def split_values(x, starts, sessions, cap, minute, offset, audit):
    """Read session variables in storage order and integrate each sub-slot."""
    ev = np.zeros(len(starts))
    ready = []
    for s in sessions:
        a, b = float(s["start"]), float(s["end"])
        cutoff = max(a, b - minute * 60)
        p = min(cap, float(s["rated"]))
        pre = 0.0
        total = 0.0
        for j in np.flatnonzero((starts < b) & (starts + STEP > a)):
            left, right = max(a, starts[j]), min(b, starts[j] + STEP)
            # Compute both time intersections directly, from source timestamps.
            early_seconds = max(0.0, min(right, cutoff) - left)
            late_seconds = max(0.0, right - max(left, cutoff))
            for seconds, is_early in ((early_seconds, True), (late_seconds, False)):
                if seconds == 0:
                    continue
                delivered = x[offset]
                offset += 1
                audit.le(delivered, p * seconds / 3600, "EV sub-slot power bound kWh")
                total += delivered
                ev[j] += delivered
                if is_early:
                    pre += delivered
        audit.close(total, s["energy"], "per-session energy kWh")
        ready.append(pre)
    audit.require(offset == len(x), "unexpected decision vector size")
    return ev, np.array(ready)


def physical_vector(x, row, starts, sessions, cap, pv, price, intensity, protocol, audit):
    n = len(starts)
    audit.require(x.ndim == 1 and np.isfinite(x).all(), "invalid decision vector")
    audit.le(0, x, "negative decision value")
    grid, solar = x[:n], x[n:2*n]
    offset = 2*n
    audit.le(solar, pv, "PV power bound kWh")
    capacity = float(row["battery_kwh"])
    charge, discharge = np.zeros(n), np.zeros(n)
    if capacity > 0:
        charge, discharge = x[offset:offset+n], x[offset+n:offset+2*n]
        soc, modes = x[offset+2*n:offset+3*n+1], x[offset+3*n+1:offset+4*n+1]
        offset += 4*n+1
        limit = capacity / protocol["battery_hours"] / 2
        eta = math.sqrt(protocol["battery_roundtrip_efficiency"])
        initial = protocol["initial_soc_fraction"] * capacity
        audit.le(soc, capacity, "SOC capacity kWh")
        audit.close([soc[0], soc[-1]], initial, "initial and terminal SOC kWh")
        audit.close(np.diff(soc), eta * charge - discharge / eta, "SOC transition kWh")
        audit.close(modes, modes.round(), "storage mode integrality")
        audit.le(modes, 1, "storage mode upper bound")
        audit.le(charge, limit * modes, "storage charging power kWh")
        audit.le(discharge, limit * (1 - modes), "storage discharging power kWh")
        audit.le(np.minimum(charge, discharge), 0, "simultaneous charge/discharge kWh")
    minute = int(row["checkpoint_min"])
    ev, ready = split_values(x, starts, sessions, cap, minute, offset, audit)
    audit.close(grid + solar + discharge, ev + charge, "half-hour station balance kWh")
    target = cutoff_energy(sessions, cap, minute)
    energy = np.array([float(s["energy"]) for s in sessions])
    cost = grid @ price + protocol["battery_wear_gbp_per_kwh_throughput"] * (charge.sum() + discharge.sum())
    carbon = grid @ intensity
    objective = row["objective"]
    audit.le(cost, float(row["budget_gbp"]), "operating budget excess GBP")
    audit.le(ready, target, "readiness above physical maximum kWh")
    if objective == "preserve_readiness":
        audit.close(ready, target, "session-level ASAP floor kWh")
    else:
        audit.le(carbon, float(row["carbon_cap_kg"]) + MODEL_TOL, "shared carbon cap excess kg")
    eligible = energy - target <= 1e-5
    full = energy - ready <= 1e-5
    metrics = {"early_energy_kwh": ready.sum(), "avoidable_shortfall_kwh": np.maximum(target - ready, 0).sum(),
               "asap_early_full_count": eligible.sum(), "early_full_count": full.sum(),
               "early_full_on_asap_feasible_count": (eligible & full).sum(),
               "max_early_reference_excess_kwh": max(0, np.max(ready - target)), "operating_cost_gbp": cost,
               "carbon_kg": carbon, "grid_kwh": grid.sum(), "grid_peak_kw": 2 * grid.max(),
               "objective_value": carbon if objective == "preserve_readiness" else -ready.sum()}
    for name, value in metrics.items():
        audit.close(row[name], value, f"primary reconstructed {name}")


def nesting(index, audit, plan):
    # Use objective bounds, not a forced ordering of finite-gap incumbents.
    def no_worse(relaxed, restricted, reason):
        audit.le(float(relaxed["objective_lower_bound"]), float(restricted["objective_value"]), reason)

    for key, row in index.items():
        case, design, pv, minute, frac, obj = key
        audit.context = key
        if not truth(row["feasible"]):
            if frac and (other := index.get((case, design, pv, minute, 0.0, obj))) and truth(other["feasible"]):
                raise ValueError(f"{key}: storage made a feasible no-storage case infeasible")
            if obj == "fixed_10_percent":
                common = index[case, design, pv, minute, frac, "common_carbon"]
                audit.require(float(row["carbon_cap_kg"]) + TOL < float(common["carbon_cap_kg"]),
                              "looser fixed cap declared infeasible")
            continue
        for prior in (0.0, 0.125):
            other = index.get((case, design, pv, minute, prior, obj))
            if prior < frac and other is not None and truth(other["feasible"]):
                no_worse(row, other, "storage capacity nesting")
        if design == "main" and obj == "preserve_readiness":
            for prev_pv in plan["pv_scales"]:
                if prev_pv < pv:
                    no_worse(row, index[case, design, prev_pv, minute, frac, obj], "PV capacity nesting")
        if design == "grid_peak" and obj == "preserve_readiness":
            no_worse(index[case, "main", pv, minute, frac, obj], row, "grid-peak restriction nesting")
        if design == "main" and obj == "common_carbon":
            for earlier in plan["checkpoints_minutes_before_proxy_end"]:
                if earlier > minute:
                    no_worse(row, index[case, design, pv, earlier, frac, obj], "later checkpoint readiness nesting")
        if obj == "fixed_10_percent":
            common = index[case, design, pv, minute, frac, "common_carbon"]
            if float(row["carbon_cap_kg"]) < float(common["carbon_cap_kg"]):
                no_worse(common, row, "stricter carbon cap readiness nesting")
            else:
                no_worse(row, common, "looser carbon cap readiness nesting")


def audit_matched_panel(cases, audit):
    def panel_key(case):
        return tuple(case[key] for key in ("city", "site_id", "month", "week_rank"))

    year_cases = [{panel_key(c): c["case_id"] for c in cases.values() if c["year"] == str(year)} for year in (2023, 2024)]
    audit.require(sum(map(len, year_cases)) == len(cases), "duplicate frozen site/month/week key")
    common = set(year_cases[0]).intersection(year_cases[1])
    cross_year = read(HOLDOUT / "analysis_common_pv" / "week_ordinal_matches.csv")
    listed = [row for row in cross_year if row["match_status"] == "matched"]
    audit.require(len(listed) == len(common) == len({panel_key(r) for r in listed}), "matched-panel count or duplicate")
    audit.require({(r["case_id_2023"], r["case_id_2024"]) for r in listed} ==
                  {(year_cases[0][k], year_cases[1][k]) for k in common}, "matched-panel members changed")
    return len(common)


def run(directory, allow_subset):
    manifest = json_file(directory / "manifest.json")
    protocol = json_file(HOLDOUT / "protocol.json")
    plan = json_file(ROOT / "service_protocol.json")
    audit = Audit()
    audit.require(sha(ROOT / "service_protocol.json") == manifest["service_protocol_sha256"], "service protocol hash")
    audit.require(sha(HOLDOUT / "protocol.json") == manifest["holdout_protocol_sha256"], "holdout protocol hash")
    audit.require(sha(directory / "service_outcomes.csv") == manifest["outcome_sha256"], "outcome file hash")
    cases, session_map, anchors, inputs = load_frozen_inputs(manifest, protocol, audit)
    audit.context = "cross-year panel"
    matched_count = audit_matched_panel(cases, audit)
    n = manifest["site_weeks"]
    audit.require(n == len(cases) or (allow_subset and 0 < n < len(cases)), "incomplete run; use --allow-subset for a smoke test")
    selected = list(cases)[:n]
    all_rows = read(directory / "service_outcomes.csv")
    index = {scenario_key(r): r for r in all_rows}
    audit.require(len(index) == len(all_rows) == manifest["outcomes"], "duplicate or incomplete service rows")
    expected_vectors = {f"{case.replace('|', '__')}.npz" for case in selected}
    found = {p.name for p in (directory / "primary_solutions").glob("*.npz")}
    audit.require(found == expected_vectors, "missing or surplus primary vector files")
    physical_count = 0
    designs = Counter()
    status = Counter()
    position = 0
    for case_id in selected:
        audit.context = case_id
        case = cases[case_id]
        sessions = session_map[case_id]
        audit.require(len(sessions) == int(case["n_sessions"]) == len({s["session_id"] for s in sessions}), "session count or duplicate")
        starts = np.arange(stamp(case["start_utc"]), stamp(case["end_utc_exclusive"]), STEP)
        signal_map = inputs[case["group"]][1]
        signal_rows = [signal_map[t] for t in starts]
        price = np.array([float(r["price_gbp_per_mwh"]) / 1000 for r in signal_rows])
        intensity = np.array([float(r["carbon_g_per_kwh"]) / 1000 for r in signal_rows])
        pv_base = np.array([float(r["pv_kw_per_kwp"]) / 2 for r in signal_rows])
        vec_path = directory / "primary_solutions" / f"{case_id.replace('|', '__')}.npz"
        cache = {}
        seen = set()
        with np.load(vec_path, allow_pickle=False) as vectors:
            for key, cap, cohort in design_rows(case_id, sessions, plan, protocol):
                audit.context = key
                audit.require(position < len(all_rows) and scenario_key(all_rows[position]) == key, "unexpected scenario order or missing cell")
                row = all_rows[position]
                position += 1
                _, design, pv_scale, minute, fraction, objective = key
                designs[design] += 1
                status[row["status"]] += 1
                is_optimal = row["status"] == "optimal"
                audit.require(is_optimal == truth(row["optimal"]) == truth(row["feasible"]), "inconsistent solver flags")
                audit.require(is_optimal or (row["status"] == "infeasible" and objective == "fixed_10_percent"), "unexpected solver status")
                audit.require(all(row[field] == case[field] for field in ("city", "site_id", "year", "month", "week_rank")), "case labels changed")
                if design not in cache:
                    load = immediate(cohort, starts, cap)
                    cache[design] = load, float(load @ price), sum(float(s["energy"]) for s in cohort)
                load, budget, energy = cache[design]
                audit.close(row["n_sessions"], len(cohort), "cohort count")
                audit.close(row["energy_kwh"], energy, "cohort energy kWh")
                audit.close(load.sum(), energy, "immediate reference energy kWh")
                audit.close(row["power_ceiling_kw"], cap, "vehicle cap kW")
                audit.close(row["budget_gbp"], budget, "recomputed shared budget GBP")
                if design == "grid_peak":
                    audit.close(row["peak_import_limit_kw"], 2 * load.max(), "peak proxy kW")
                else:
                    audit.require(not row["peak_import_limit_kw"], "unexpected peak restriction")
                daily = float(case["site_daily_energy_kwh"])
                audit.close(row["pv_kwp"], daily * pv_scale, "PV nameplate kWp")
                audit.close(row["battery_kwh"], daily * fraction, "battery nameplate kWh")
                pv = pv_base * daily * pv_scale
                fixed = fixed_carbon(load, pv, price, intensity, budget)
                flex = float(row["best_no_battery_carbon_kg"])
                audit.close(row["fixed_carbon_kg"], fixed, "independent fixed carbon kg")
                audit.le(flex, fixed, "flexible anchor carbon kg")
                if design == "main":
                    audit.close(flex, anchors[case_id, pv_scale, 1.0], "frozen flexible carbon kg")
                if objective == "common_carbon":
                    alpha = plan["common_target_share_of_available_no_storage_reduction"]
                    audit.close(row["carbon_cap_kg"], fixed - alpha * max(0, fixed - flex), "common carbon target kg")
                elif objective == "fixed_10_percent":
                    audit.close(row["carbon_cap_kg"], fixed * (1-plan["fixed_reduction_sensitivity"]), "absolute target kg")
                early = cutoff_energy(cohort, cap, minute)
                audit.close(row["asap_checkpoint_kwh"], early.sum(), "independent ASAP checkpoint kWh")
                audit.close(row["physical_early_shortfall_kwh"], energy - early.sum(), "physical shortfall kWh")
                if is_optimal:
                    cost, carbon = float(row["operating_cost_gbp"]), float(row["carbon_kg"])
                    ready = float(row["early_energy_kwh"])
                    audit.le(cost, budget, "CSV operating budget GBP")
                    audit.le(0, carbon, "CSV carbon nonnegativity kg")
                    audit.le(ready, early.sum(), "CSV physical early maximum kWh")
                    audit.close(row["avoidable_shortfall_kwh"], early.sum() - ready, "CSV avoidable shortfall kWh")
                    audit.close(row["objective_value"], carbon if objective == "preserve_readiness" else -ready, "CSV objective")
                    value, bound = float(row["objective_value"]), float(row["objective_lower_bound"])
                    audit.le(bound, value, "objective lower-bound ordering")
                    audit.close(row["objective_abs_gap"], max(0, value - bound), "objective bound gap")
                    if objective == "preserve_readiness":
                        audit.close(ready, early.sum(), "CSV total ASAP readiness floor kWh")
                    else:
                        audit.le(carbon, float(row["carbon_cap_kg"]) + MODEL_TOL, "CSV target exceedance kg")
                    if design == "grid_peak":
                        audit.le(float(row["grid_peak_kw"]), 2 * load.max(), "CSV peak exceedance kW")
                if design == "main" and pv_scale == plan["primary_pv_scale"] and minute == plan["primary_checkpoint_minutes"] and objective in OBJECTIVES:
                    name = f"{objective}_{row['battery_fraction']}"
                    audit.require(name in vectors and name not in seen, "missing or repeated primary decision vector")
                    seen.add(name)
                    physical_vector(vectors[name], row, starts, cohort, cap, pv, price, intensity, protocol, audit)
                    physical_count += 1
            audit.require(seen == set(vectors.files), "unused decision vector in primary file")
    audit.require(position == len(all_rows), "extra rows outside frozen design")
    nesting(index, audit, plan)
    return {"all_pass": True, "site_weeks": n, "frozen_site_weeks": len(cases),
            "cross_year_matched_site_weeks": matched_count, "outcomes_checked": len(all_rows),
            "primary_decision_vectors_recomputed": physical_count, "design_counts": dict(designs),
            "solver_status_counts": dict(status), "max_absolute_residuals": dict(sorted(audit.maxima.items())),
            "check_scope": "independent source, design, primary-vector physics and aggregate consistency; no independent full reoptimization"}


def audit_frontier(directory, *, near_endpoint=False):
    audit = Audit()
    service_path = directory / "service_outcomes.csv"
    stem = "endpoint_sensitivity" if near_endpoint else "frontier"
    frontier_path = directory / f"{stem}_outcomes.csv"
    manifest = json_file(directory / f"{stem}_manifest.json")
    service_manifest = json_file(directory / "manifest.json")
    plan = json_file(ROOT / "service_protocol.json")
    protocol = json_file(HOLDOUT / "protocol.json")
    audit.require(sha(service_path) == manifest["service_outcomes_sha256"] == service_manifest["outcome_sha256"], "frontier service hash")
    audit.require(sha(frontier_path) == manifest["frontier_outcomes_sha256"], "frontier output hash")
    audit.require(sha(ROOT / "service_protocol.json") == service_manifest["service_protocol_sha256"], "frontier protocol hash")
    cases, _, _, _ = load_frozen_inputs(service_manifest, protocol, audit)
    audit.context = "frontier panel"
    matched_count = audit_matched_panel(cases, audit)
    panel = read(HOLDOUT / "analysis_common_pv" / "week_ordinal_matches.csv")
    matched_ids = {r[f"case_id_{year}"] for r in panel if r["match_status"] == "matched" for year in (2023, 2024)}
    shares = tuple(manifest["target_shares"])
    audit.require(shares == ((0.95, 0.99) if near_endpoint else (0.75, 0.9, 1.0)), "unexpected target shares")
    if near_endpoint:
        plan["pv_scales"] = [plan["primary_pv_scale"]]
        plan["checkpoints_minutes_before_proxy_end"] = [plan["primary_checkpoint_minutes"]]
        base = directory / "frontier_outcomes.csv"
        audit.require(sha(base) == json_file(directory / "frontier_manifest.json")["frontier_outcomes_sha256"],
                      "original frontier hash")
        bounds = {(r["case_id"], float(r["pv_scale"]), int(r["checkpoint_min"]),
                   float(r["battery_fraction"]), float(r["target_share"])): r for r in read(base)
                  if float(r["target_share"]) in (0.90, 1.00)}
    original_share = plan["common_target_share_of_available_no_storage_reduction"]
    original = {}
    for r in read(service_path):
        if r["design"] == "main" and r["objective"] == "common_carbon":
            key = (r["case_id"], float(r["pv_scale"]), int(r["checkpoint_min"]), float(r["battery_fraction"]))
            audit.require(key not in original, "duplicate service anchor")
            original[key] = r
    rows = read(frontier_path)
    expected = [(case_id, pv, minute, fraction, share) for case_id in cases for pv in plan["pv_scales"]
                for minute in plan["checkpoints_minutes_before_proxy_end"] for fraction in plan["battery_fractions"] for share in shares]
    audit.require(len(rows) == len(expected) == manifest["outcomes"] and len(cases) == manifest["site_weeks"], "frontier grid incomplete")
    index = {}
    groups = defaultdict(list)
    for row, key in zip(rows, expected, strict=True):
        case_id, pv, minute, fraction, share = key
        audit.context = key
        actual = (row["case_id"], float(row["pv_scale"]), int(row["checkpoint_min"]),
                  float(row["battery_fraction"]), float(row["target_share"]))
        audit.require(actual == key and actual not in index, "out of order or duplicate frontier scenario")
        index[key] = row
        anchor = original[key[:-1]]
        audit.require(row["status"] == "optimal" and truth(row["optimal"]) and truth(row["feasible"]), "frontier solution missing or nonoptimal")
        audit.require(all(row[name] == cases[case_id][name] for name in ("year", "city", "site_id", "month", "week_rank")), "frontier case labels")
        for name in ("budget_gbp", "energy_kwh", "n_sessions", "fixed_carbon_kg", "best_no_battery_carbon_kg"):
            audit.close(row[name], anchor[name], f"frontier anchor {name}")
        fixed, best = float(anchor["fixed_carbon_kg"]), float(anchor["best_no_battery_carbon_kg"])
        reduction = max(0, fixed - best)
        cap = fixed - share * reduction
        audit.close(row["carbon_cap_kg"], cap, "frontier recomputed carbon cap kg")
        audit.close(anchor["carbon_cap_kg"], fixed - original_share * reduction, "main recomputed carbon cap kg")
        audit.le(float(row["operating_cost_gbp"]), float(anchor["budget_gbp"]), "frontier budget excess GBP")
        carbon = float(row["carbon_kg"])
        audit.le(carbon, cap + MODEL_TOL, "frontier carbon excess kg")
        audit.le(0, carbon, "frontier nonnegative carbon")
        if fraction == 0:
            audit.le(best, carbon, "frontier no-storage carbon relative to minimum")
        value, bound = float(row["objective_value"]), float(row["objective_lower_bound"])
        asap = float(anchor["asap_checkpoint_kwh"])
        shortfall = asap + value
        audit.le(-value, asap, "frontier physical early energy maximum kWh")
        audit.le(0, -value, "frontier nonnegative early energy kWh")
        audit.close(row["avoidable_shortfall_kwh"], shortfall, "frontier recalculated avoidable shortfall kWh")
        audit.close(row["shortfall_pct"], 100 * shortfall / float(row["energy_kwh"]), "frontier recalculated shortfall percent")
        audit.le(int(row["early_full_count"]), int(anchor["asap_early_full_count"]), "frontier physically possible full count")
        audit.le(bound, value, "frontier objective lower-bound ordering")
        audit.close(row["objective_abs_gap"], max(0, value - bound), "frontier objective gap")
        reported = [float(row[name]) for name in row if name.startswith("max_") and row[name]]
        audit.le(reported, MODEL_TOL, "frontier reported residual above tolerance")
        audit.le(float(anchor["objective_lower_bound"]), value, "frontier tighter than main half-potential cap")
        if case_id in matched_ids:
            groups[int(row["year"]), minute, float(pv), float(share), float(fraction)].append(row)

    def no_worse(relaxed, restricted, reason):
        audit.le(float(relaxed["objective_lower_bound"]), float(restricted["objective_value"]), reason)

    for key, row in index.items():
        case, pv, minute, fraction, share = key
        audit.context = key
        for looser in shares:
            if looser < share:
                no_worse(index[case, pv, minute, fraction, looser], row, "frontier carbon-target nesting")
        if near_endpoint:
            no_worse(bounds[(*key[:-1], .90)], row, "endpoint nesting against .90")
            no_worse(row, bounds[(*key[:-1], 1.)], "endpoint nesting against 1.00")
        for smaller in plan["battery_fractions"]:
            if smaller < fraction:
                no_worse(row, index[case, pv, minute, smaller, share], "frontier storage capacity nesting")
        for earlier in plan["checkpoints_minutes_before_proxy_end"]:
            if earlier > minute:
                # The unnormalized cumulative energy is monotone. Its gap
                # relative to the changing ASAP target need not be monotone.
                no_worse(row, index[case, pv, earlier, fraction, share], "frontier later-checkpoint readiness nesting")

    recorded = None if near_endpoint else json_file(directory / "service_summary.json")["frontier"]
    if recorded is not None:
        audit.require(len(recorded) == len(groups), "frontier summary groups do not match")
    independent = {}
    endpoint_groups = {}
    for key, group in groups.items():
        name = "|".join(map(str, key))
        audit.context = name
        pooled = 100 * sum(float(r["avoidable_shortfall_kwh"]) for r in group) / sum(float(r["energy_kwh"]) for r in group)
        full = 100 * sum(int(r["early_full_count"]) for r in group) / sum(int(r["n_sessions"]) for r in group)
        half_pp = sum(float(r["shortfall_pct"]) >= 0.5 for r in group)
        audit.require(len(group) == matched_count, "incorrect frontier matched denominator")
        if recorded is None:
            endpoint_groups[name] = {"pooled_avoidable_shortfall_pct": pooled, "site_weeks": len(group),
                                     "site_weeks_with_shortfall_at_least_half_pp": half_pp,
                                     "site_weeks_with_shortfall_gt_1e_5_kwh": sum(
                                         float(r["avoidable_shortfall_kwh"]) > 1e-5 for r in group)}
        else:
            expected_group = recorded[name]
            audit.require(len(group) == expected_group["site_weeks"], "incorrect recorded matched denominator")
            audit.require(half_pp == expected_group["site_weeks_with_avoidable_shortfall_above_half_pp"], "incorrect practical shortfall count")
            audit.close(pooled, expected_group["pooled_avoidable_shortfall_pct"], "frontier pooled matched shortfall percent")
            audit.close(full, expected_group["pooled_early_full_pct"], "frontier pooled matched full-session percent")
            quartiles = np.quantile([float(r["shortfall_pct"]) for r in group], (0.25, 0.5, 0.75))
            audit.close(quartiles, [expected_group["site_week_shortfall_pct"][x] for x in ("q25", "median", "q75")],
                        "frontier site-week shortfall percent quartiles")
        independent[key] = pooled
    primary = []
    for year in (2023, 2024):
        for share in shares:
            key = (year, plan["primary_checkpoint_minutes"], plan["primary_pv_scale"], share)
            no_storage = independent[(*key, 0.0)]
            storage = independent[(*key, plan["primary_battery_fraction"])]
            primary.append({"year": year, "target_share": share, "pooled_shortfall_no_storage_pct": no_storage,
                            "pooled_shortfall_main_storage_pct": storage, "pooled_improvement_percentage_points": no_storage - storage})
    return {"all_pass": True, "outcomes_checked": len(rows), "site_weeks": len(cases),
            "matched_site_weeks_per_year": matched_count, "matched_summary_groups_checked": len(groups),
            "exploratory_primary_effects": primary, "max_absolute_residuals": dict(sorted(audit.maxima.items())),
            "check_scope": "CSV and recorded solver summaries, anchors and monotonicity; no frontier decision vectors were saved",
            **({"near_endpoint_groups": endpoint_groups} if near_endpoint else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--frontier-only", action="store_true", help="Audit the exploratory frontier aggregate output")
    parser.add_argument("--near-endpoint", action="store_true", help="Audit the .95 and .99 sensitivity against the original frontier")
    parser.add_argument("--report", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = (audit_frontier(args.results, near_endpoint=args.near_endpoint)
              if args.frontier_only or args.near_endpoint else run(args.results, args.allow_subset))
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

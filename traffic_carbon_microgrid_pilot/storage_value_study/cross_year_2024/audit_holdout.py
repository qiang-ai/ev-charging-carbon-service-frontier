"""Independently audit the 2024 temporal holdout against the frozen 2023 design."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
sys.path.insert(0, str(STUDY))
from audit_study import TOL, bool_value, fixed_pv_footprint, physical_asap, read_group

ALPHAS = (0.0, 1.0)
PV_SCALES = (0.0, 0.2, 0.4)
BATTERY_FRACTIONS = (0.0, 0.125, 0.5)
CITIES = ("edinburgh", "glasgow")
MONTHS = (1, 7)
KEY_COLUMNS = ("site_id", "case_id", "session_id", "station_id", "connector_id", "group")


def read_table(path):
    return pd.read_csv(path, dtype={column: str for column in KEY_COLUMNS})


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def month_number(value):
    return int(str(value).replace("-", "_").split("_")[-1])


def complete_calendar_weeks(year, month):
    first = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    month_end = first + pd.offsets.MonthBegin(1)
    monday = first + pd.Timedelta(days=(-first.weekday()) % 7)
    rank = 1
    while monday + pd.Timedelta(days=7) <= month_end:
        yield rank, monday, monday + pd.Timedelta(days=7)
        monday += pd.Timedelta(days=7)
        rank += 1


def case_key(row):
    return row.city, row.site_id, month_number(row.month), int(row.week_rank)


def primary_matched_changes(reference_outcomes, reference_cases, original_keys, holdout_effects, matched_keys, endpoint):
    selected = reference_outcomes[reference_outcomes.pv_scale.eq(endpoint["pv_scale"])
                                  & reference_outcomes.alpha.isin(ALPHAS)
                                  & reference_outcomes.battery_fraction.isin((0, endpoint["battery_fraction"]))].copy()
    selected = selected[selected.feasible.map(bool_value)]
    carbon = selected.set_index(["case_id", "alpha", "battery_fraction"]).carbon_kg
    energy = reference_cases.set_index("case_id").energy_kwh
    changes = []
    effects = []
    for key in matched_keys & set(holdout_effects):
        case_id = original_keys[key]
        lookup = [(case_id, alpha, fraction) for alpha, fraction in
                  product(ALPHAS, (0, endpoint["battery_fraction"]))]
        if any(cell not in carbon.index for cell in lookup):
            continue
        no_fixed, yes_fixed, no_flex, yes_flex = (carbon.at[cell] for cell in lookup)
        effect = 1000 * ((no_flex - yes_flex) - (no_fixed - yes_fixed)) / energy.at[case_id]
        effects.append(effect)
        changes.append(holdout_effects[key][0] - effect)
    return {
        "pairs": len(changes),
        "reference_median_interaction_g_per_delivered_kwh": float(np.median(effects)) if effects else None,
        "median_within_pair_change_g_per_delivered_kwh": float(np.median(changes)) if changes else None,
    }


def check_pv_source(data_dir, groups, require, close):
    raw_manifest = json.loads((data_dir / "raw_manifest.json").read_text(encoding="utf-8"))
    radiation_field = "global_tilted_irradiance"
    for group in groups:
        city, year, month = group.split("_")
        name = f"era5_gti_{city}_{year}_{month}.json"
        source = data_dir / "raw" / name
        provenance = raw_manifest["files"][name]
        require("era5_irradiance_raw_sha256", file_hash(source) == provenance["sha256"], name)
        query = parse_qs(urlsplit(provenance["url"]).query)
        require("common_era5_pv_parameters", all(query.get(key) == [value] for key, value in (
            ("models", "era5"), ("tilt", "35"), ("azimuth", "0"), ("timezone", "GMT"), ("hourly", radiation_field),
        )), name)
        payload = json.loads(source.read_bytes())
        require("utc_radiation_watts_per_square_meter",
                payload["utc_offset_seconds"] == 0 and payload["hourly_units"][radiation_field] == "W/m²", name)
        hourly = pd.Series(payload["hourly"][radiation_field],
                           index=pd.to_datetime(payload["hourly"]["time"], utc=True), dtype=float)
        signals = read_table(data_dir / group / "signals.csv")
        # Open-Meteo's timestamp labels the end of the preceding-hour average.
        hour_ending = pd.to_datetime(signals.timestamp_utc, utc=True).dt.floor("h") + pd.Timedelta(hours=1)
        calculated = 0.86 * hourly.reindex(hour_ending).to_numpy() / 1000
        close("era5_preceding_hour_pv_signal_kw_per_kwp",
              np.abs(signals.pv_kw_per_kwp.to_numpy() - calculated).max(), 0, group, 1e-8)


def expected_site_weeks(data_dir, frozen, year, min_sessions, power_cap):
    """Use source membership and the calendar, without reading study outputs."""
    expected = {}
    for city, month in product(CITIES, MONTHS):
        group = f"{city}_{year}_{month:02d}"
        sessions, signals = read_group(data_dir, group)
        for rank, begin, end in complete_calendar_weeks(year, month):
            slots = np.arange(begin.timestamp(), end.timestamp(), 1800)
            signal_slice = signals[signals.seconds.ge(slots[0]) & signals.seconds.lt(end.timestamp())]
            signal_values = signal_slice[["price_gbp_per_mwh", "carbon_g_per_kwh", "pv_kw_per_kwp"]].to_numpy()
            complete = np.array_equal(signal_slice.seconds.to_numpy(), slots) and np.isfinite(signal_values).all()
            for row in frozen[frozen.city.eq(city)].itertuples(index=False):
                cohort = sessions[sessions.station_id.eq(row.site_id) & sessions.start_seconds.ge(slots[0])
                                  & sessions.end_seconds.le(end.timestamp()) & sessions.power_kw.le(22.0 + 1e-6)].copy()
                feasible_energy = np.minimum(cohort.power_kw, power_cap) * (cohort.end_seconds - cohort.start_seconds) / 3600
                cohort = cohort[cohort.energy_kwh.le(feasible_energy + 1e-6)]
                key = city, row.site_id, month, rank
                expected[key] = {
                    "group": group, "start": begin, "end": end, "slots": slots, "complete": complete,
                    "included": complete and len(cohort) >= min_sessions, "signals": signal_slice, "sessions": cohort,
                }
    return expected


def audit(data_dir, result_dir, protocol_path, baseline_cases_path, year, analysis_dir, reference_results_dir):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = read_table(baseline_cases_path)
    cases = read_table(result_dir / "cases.csv")
    coverage = read_table(result_dir / "coverage.csv")
    outcomes = read_table(result_dir / "outcomes.csv")
    membership = read_table(result_dir / "case_sessions.csv")
    frozen = baseline[["city", "site_id", "site_daily_energy_kwh"]].drop_duplicates()
    failures, coverage_reasons = Counter(), Counter()
    examples, maxima = [], {}

    def require(name, passed, detail):
        if not bool(passed):
            failures[name] += 1
            if len(examples) < 35:
                examples.append(f"{name}: {detail}")

    def close(name, observed, expected, detail, tolerance=TOL):
        error = abs(float(observed) - float(expected))
        if np.isfinite(error):
            maxima[name] = max(error, maxima.get(name, 0.0))
        require(name, np.isfinite(error) and error <= tolerance, f"{detail}, observed={observed}, expected={expected}")

    require("protocol_hash", file_hash(protocol_path) == manifest["protocol_sha256"], str(protocol_path))
    require("completed_run", manifest["completed"] and manifest["year"] == year, str(result_dir))
    for name, digest in protocol["baseline_sha256"].items():
        path = baseline_cases_path if name == "results/cases.csv" else baseline_cases_path.parent.parent / name
        require("frozen_2023_artifact_hash", file_hash(path) == digest, str(path))
    for name, digest in manifest["input_sha256"].items():
        require("input_table_hash", file_hash(data_dir / name) == digest, name)
    check_pv_source(data_dir, cases.group.unique(), require, close)
    for name, actual in (("n_case_weeks", len(cases)), ("n_sessions", len(membership)),
                         ("n_coverage_cells", len(coverage)), ("n_solutions", len(outcomes))):
        close("manifest_" + name, manifest[name], actual, str(result_dir), 0)
    for name in ("site_mapping.csv", "site_catalog.csv"):
        candidate = data_dir / name
        if candidate.exists():
            original = STUDY / "data" / name
            require("frozen_infrastructure_mapping", read_table(candidate).equals(read_table(original)), name)
    require("frozen_factorial_levels", tuple(protocol["alphas"]) == ALPHAS and tuple(protocol["pv_scales"]) == PV_SCALES
            and tuple(protocol["battery_fractions"]) == BATTERY_FRACTIONS, "protocol")
    require("frozen_holdout_definition", protocol["baseline_year"] == 2023 and protocol["holdout_year"] == 2024
            and tuple(protocol["months"]) == MONTHS and not protocol["site_replacements_allowed"]
            and protocol["min_complete_sessions_per_week"] == 5, "protocol")
    require("frozen_battery_parameters", protocol["battery_hours"] == 2 and protocol["battery_roundtrip_efficiency"] == 0.9
            and protocol["initial_soc_fraction"] == 0.5 and protocol["battery_wear_gbp_per_kwh_throughput"] == 0.01, "protocol")
    require("frozen_thresholds", protocol["substantial_threshold_g_per_kwh"] == 5
            and protocol["substantial_threshold_relative_percent"] == 5 and protocol["interaction_threshold_g_per_kwh"] == 1,
            "protocol")
    require("unique_frozen_site_scale", frozen.site_id.is_unique and len(frozen) == 12, "frozen baseline sites")
    expected_scale = frozen.set_index("site_id").site_daily_energy_kwh.to_dict()
    expected_city = frozen.set_index("site_id").city.to_dict()
    require("frozen_site_set", set(protocol["sites"]) == set(expected_scale), "protocol site set")
    for site in set(protocol["sites"]) & set(expected_scale):
        close("protocol_frozen_scale", protocol["sites"][site]["site_daily_energy_kwh"], expected_scale[site], site, 1e-9)
        require("protocol_frozen_site_city", protocol["sites"][site]["city"] == expected_city[site], site)
    power_cap = float(baseline.power_cap_kw.iloc[0])
    require("frozen_power_cap", baseline.power_cap_kw.eq(power_cap).all() and protocol["power_cap_kw"] == power_cap, "baseline")
    expected = expected_site_weeks(data_dir, frozen, year, 5, power_cap)
    expected_included = {key for key, entry in expected.items() if entry["included"]}
    seen_coverage = {case_key(row): row for row in coverage.itertuples(index=False)}
    seen_cases = {case_key(row): row for row in cases.itertuples(index=False)}
    require("coverage_calendar_grid", len(seen_coverage) == len(coverage) and set(seen_coverage) == set(expected), "coverage table")
    require("all_and_only_rule_eligible_cases", len(seen_cases) == len(cases) and set(seen_cases) == expected_included, "case table")
    for key, entry in expected.items():
        if not entry["complete"]:
            reason = "incomplete_signals"
        elif not entry["included"]:
            reason = "too_few_sessions"
        else:
            reason = "included"
        coverage_reasons[reason] += 1
        if key not in seen_coverage:
            continue
        row = seen_coverage[key]
        require("coverage_year_and_group", int(row.year) == year and row.group == entry["group"], key)
        require("coverage_calendar_window", pd.Timestamp(row.start_utc) == entry["start"]
                and pd.Timestamp(row.end_utc_exclusive) == entry["end"], key)
        close("coverage_signal_slot_count", row.signal_slots, len(entry["signals"]), key, 0)
        require("coverage_inclusion_rule", row.coverage_status == reason, key)
        close("coverage_eligible_session_count", row.n_sessions, len(entry["sessions"]), key, 0)

    require("duplicate_holdout_case_id", cases.case_id.is_unique, "cases")
    require("duplicate_holdout_case_session", not membership.duplicated(["case_id", "session_id"]).any(), "membership")
    require("complete_case_membership", set(cases.case_id) == set(membership.case_id), "membership")
    require("complete_outcome_case_set", set(cases.case_id) == set(outcomes.case_id), "outcomes")
    groups = dict(tuple(outcomes.groupby("case_id")))
    member_groups = dict(tuple(membership.groupby("case_id")))
    grid = set(product(ALPHAS, PV_SCALES, BATTERY_FRACTIONS))
    retained_ids = set()
    by_status = Counter()
    interaction_counts = Counter()
    practical_counts = Counter()
    primary_effects = {}
    residual_columns = (
        "max_energy_error_kwh", "max_balance_error_kwh", "max_bound_error_kwh",
        "max_inequality_error_kwh", "max_integrality_error", "max_simultaneous_kwh", "max_cost_budget_error_gbp",
    )
    for key in set(seen_cases) & set(expected):
        case, entry = seen_cases[key], expected[key]
        label = case.case_id
        require("case_year", int(case.year) == year, label)
        require("case_group", case.group == entry["group"], label)
        require("case_calendar_window", pd.Timestamp(case.start_utc) == entry["start"]
                and pd.Timestamp(case.end_utc_exclusive) == entry["end"], label)
        close("case_scale_frozen_to_2023", case.site_daily_energy_kwh, expected_scale[case.site_id], label, 1e-9)
        close("case_nominal_battery_capacity", case.battery_kwh, 0.5 * expected_scale[case.site_id], label, 1e-9)
        close("case_nominal_battery_power", case.battery_kw, 0.25 * expected_scale[case.site_id], label, 1e-9)
        close("case_power_cap_frozen", case.power_cap_kw, power_cap, label)
        if label not in member_groups or label not in groups or not entry["complete"]:
            continue
        member, source = member_groups[label], entry["sessions"]
        require("all_eligible_session_ids_retained", set(member.session_id) == set(source.index), label)
        if not set(member.session_id) <= set(source.index):
            continue
        original = source.loc[member.session_id]
        for field in ("station_id", "connector_id"):
            require("session_" + field, np.array_equal(member[field], original[field]), label)
        for field, name in (("start", "start_seconds"), ("end", "end_seconds"),
                            ("energy", "energy_kwh"), ("rated", "power_kw")):
            close("session_source_" + field, np.abs(member[field].to_numpy() - original[name].to_numpy()).max(), 0, label)
        ids = {(case.group, session) for session in member.session_id}
        require("session_reused_in_different_weeks", not retained_ids.intersection(ids), label)
        retained_ids.update(ids)
        close("case_session_count", case.n_sessions, len(member), label, 0)
        close("case_session_energy_kwh", case.energy_kwh, member.energy.sum(), label)
        charging_power = np.minimum(member.rated.to_numpy(), power_cap)
        slack_hours = np.maximum(0.0, (member.end - member.start) / 3600 - member.energy / charging_power)
        close("case_weighted_slack_min", case.energy_weighted_slack_min,
              60 * np.average(slack_hours, weights=member.energy), label)
        signals = entry["signals"]
        price = signals.price_gbp_per_mwh.to_numpy() / 1000
        carbon = signals.carbon_g_per_kwh.to_numpy() / 1000
        pv_unit = signals.pv_kw_per_kwp.to_numpy() * 0.5
        asap = physical_asap(member, entry["slots"], power_cap)
        close("recomputed_asap_energy_kwh", asap.sum(), case.energy_kwh, label)
        close("recomputed_asap_budget_gbp", asap @ price, case.budget_gbp, label)
        close("recomputed_reference_cost_gbp", asap @ price, case.reference_cost_gbp, label)
        close("recomputed_asap_footprint_kg", asap @ carbon, case.reference_carbon_kg, label)
        case_outcomes = groups[label]
        actual_grid = set(zip(case_outcomes.alpha, case_outcomes.pv_scale, case_outcomes.battery_fraction))
        require("complete_18_cell_factorial", len(case_outcomes) == len(grid) and actual_grid == grid, label)
        values = {}
        for row in case_outcomes.itertuples(index=False):
            cell = (row.alpha, row.pv_scale, row.battery_fraction)
            desc = f"{label}/{cell}"
            by_status[row.status] += 1
            require("outcome_year", int(row.year) == year, desc)
            require("outcome_site", row.site_id == case.site_id, desc)
            require("outcome_storage_configuration", bool_value(row.has_battery) == (row.battery_fraction > 0), desc)
            close("outcome_budget_gbp", row.operating_budget_gbp, case.budget_gbp, desc)
            close("outcome_energy_requirement_kwh", row.energy_required_kwh, case.energy_kwh, desc)
            close("outcome_session_count", row.n_sessions, case.n_sessions, desc, 0)
            close("outcome_power_cap", row.power_cap_kw, power_cap, desc)
            close("outcome_allowed_slack_hours", row.allowed_slack_hours, row.alpha * slack_hours.sum(), desc)
            scale = expected_scale[case.site_id]
            pv = pv_unit * row.pv_scale * scale
            battery_kwh = row.battery_fraction * scale
            close("outcome_pv_capacity_frozen_kwp", row.pv_kwp, row.pv_scale * scale, desc, 1e-9)
            close("outcome_pv_availability_kwh", row.pv_available_kwh, pv.sum(), desc)
            close("outcome_pv_yield_ratio", row.pv_to_ev, pv.sum() / case.energy_kwh, desc)
            close("outcome_pv_asap_overlap_ratio", row.asap_pv_overlap_fraction,
                  np.minimum(pv, asap).sum() / case.energy_kwh, desc)
            close("outcome_battery_capacity_frozen_kwh", row.battery_kwh, battery_kwh, desc, 1e-9)
            close("outcome_battery_power_frozen_kw", row.battery_kw, battery_kwh / 2, desc, 1e-9)
            feasible = bool_value(row.feasible)
            require("baseline_constructively_feasible", feasible and row.status != "infeasible", desc)
            if not feasible:
                continue
            close("outcome_energy_delivered_kwh", row.delivered_kwh, case.energy_kwh, desc)
            close("outcome_energy_balance_kwh", row.grid_import_kwh + row.pv_used_kwh + row.battery_discharge_kwh,
                  row.delivered_kwh + row.battery_charge_kwh, desc)
            close("outcome_pv_split_kwh", row.pv_used_kwh + row.pv_curtailed_kwh, row.pv_available_kwh, desc)
            close("outcome_battery_losses_kwh", row.battery_discharge_kwh, 0.9 * row.battery_charge_kwh, desc)
            close("outcome_initial_soc_kwh", row.initial_soc_kwh, battery_kwh / 2, desc)
            close("outcome_final_soc_kwh", row.final_soc_kwh, battery_kwh / 2, desc)
            close("outcome_battery_wear_gbp", row.battery_wear_gbp, 0.01 * (row.battery_charge_kwh + row.battery_discharge_kwh), desc)
            close("outcome_cost_identity_gbp", row.operating_cost_gbp, row.grid_energy_cost_gbp + row.battery_wear_gbp, desc)
            close("outcome_cost_budget_slack_gbp", row.budget_slack_gbp, case.budget_gbp - row.operating_cost_gbp, desc)
            require("outcome_operating_cost_cap", row.operating_cost_gbp <= case.budget_gbp + TOL, desc)
            for residual in residual_columns:
                close("reported_" + residual, getattr(row, residual), 0, desc)
            close("outcome_carbon_objective_kg", row.carbon_kg, row.objective_value, desc)
            require("outcome_valid_objective_bound", np.isfinite(row.objective_lower_bound)
                    and row.objective_lower_bound <= row.carbon_kg + TOL, desc)
            if row.alpha == 0 and row.battery_fraction == 0:
                analytic_carbon, _ = fixed_pv_footprint(asap, pv, price, carbon, case.budget_gbp)
                if analytic_carbon is None:
                    require("recomputed_fixed_pv_feasible", False, desc)
                else:
                    close("recomputed_fixed_pv_optimum_kg", row.carbon_kg, analytic_carbon, desc)
            values[cell] = row

        for strict, relaxed in product(values, values):
            # Zero-to-positive storage is nested. Positive-capacity cases
            # share a 50% initial SOC, so an added constant SOC offset proves
            # nesting when both capacity and charge/discharge power increase.
            if strict != relaxed and all(lo <= hi for lo, hi in zip(strict, relaxed)):
                require("nested_objective_bounds", values[relaxed].objective_lower_bound <= values[strict].carbon_kg + TOL,
                        f"{label}/{strict}/{relaxed}")
        for pv, fraction in product(PV_SCALES, BATTERY_FRACTIONS[1:]):
            intervals = {}
            for alpha in ALPHAS:
                control, battery = values.get((alpha, pv, 0.0)), values.get((alpha, pv, fraction))
                if control is None or battery is None:
                    practical_counts["unresolved"] += 1
                    continue
                lo = 1000 * (control.objective_lower_bound - battery.carbon_kg) / case.energy_kwh
                hi = 1000 * (control.carbon_kg - battery.objective_lower_bound) / case.energy_kwh
                rel_lo = 1 - battery.carbon_kg / control.objective_lower_bound if control.objective_lower_bound > TOL else -np.inf
                rel_hi = 1 - battery.objective_lower_bound / control.carbon_kg if control.carbon_kg > TOL else np.inf
                point = 1000 * (control.carbon_kg - battery.carbon_kg) / case.energy_kwh
                intervals[alpha] = lo, hi, point
                if lo >= 5 and rel_lo >= 0.05:
                    practical_counts["substantial"] += 1
                elif hi < 5 or rel_hi < 0.05:
                    practical_counts["below"] += 1
                else:
                    practical_counts["unresolved"] += 1
            interaction = "unresolved"
            if 0 in intervals and 1 in intervals:
                lo = intervals[1][0] - intervals[0][1]
                hi = intervals[1][1] - intervals[0][0]
                if hi < -1:
                    interaction = "substitution"
                elif lo > 1:
                    interaction = "complement"
                elif lo >= -1 and hi <= 1:
                    interaction = "within_threshold"
                if pv == protocol["primary_endpoint"]["pv_scale"] and fraction == protocol["primary_endpoint"]["battery_fraction"]:
                    primary_effects[key] = (intervals[1][2] - intervals[0][2], interaction)
            interaction_counts[interaction] += 1

    original_keys = {}
    for row in baseline.itertuples(index=False):
        start = pd.Timestamp(row.start_utc)
        rank = next(rank for rank, left, _ in complete_calendar_weeks(2023, start.month) if left == start)
        original_keys[row.city, row.site_id, start.month, rank] = row.case_id
    common_keys = set(original_keys) & expected_included
    missing_original = set(original_keys) - expected_included
    if year == 2023:
        require("harmonized_2023_exact_anchor_keys", set(original_keys) == set(seen_cases), "2023 replay")
        for key in common_keys & set(seen_cases):
            require("harmonized_2023_exact_case_id", original_keys[key] == seen_cases[key].case_id, key)
        for group in cases.group.unique():
            group_path = data_dir / group
            reference_path = STUDY / "data" / group
            require("harmonized_2023_identical_prepared_sessions",
                    file_hash(group_path / "sessions.csv") == file_hash(reference_path / "sessions.csv"), group)
            current_signals = read_table(group_path / "signals.csv").drop(columns=["pv_kw_per_kwp"])
            reference_signals = read_table(reference_path / "signals.csv").drop(columns=["pv_kw_per_kwp"])
            require("harmonized_2023_identical_non_pv_signals", current_signals.equals(reference_signals), group)
    match_path = analysis_dir / "week_ordinal_matches.csv"
    checked_matches = match_path.exists() and year == 2024
    if checked_matches:
        matches = read_table(match_path)
        match_keys = {case_key(row): row for row in matches.itertuples(index=False)}
        require("analysis_strict_union_keys", len(match_keys) == len(matches)
                and set(match_keys) == set(original_keys) | expected_included, str(match_path))
        for key, row in match_keys.items():
            if key in original_keys:
                require("analysis_matched_2023_case_id", row.case_id_2023 == original_keys[key], key)
            if key in seen_cases:
                require("analysis_matched_2024_case_id", row.case_id_2024 == seen_cases[key].case_id, key)
            status = "matched" if key in common_keys else "2023_only" if key in original_keys else "2024_only"
            require("analysis_strict_match_status", row.match_status == status, key)

    matched_changes = None
    if year == 2024 and (reference_results_dir / "outcomes.csv").exists():
        matched_changes = primary_matched_changes(
            read_table(reference_results_dir / "outcomes.csv"), baseline, original_keys,
            primary_effects, common_keys, protocol["primary_endpoint"],
        )
        summary_path = analysis_dir / "holdout_summary.json"
        if summary_path.exists():
            analysis = json.loads(summary_path.read_text(encoding="utf-8"))
            close("analysis_primary_strict_matched_count", analysis["primary_matched_valid_case_pairs"],
                  matched_changes["pairs"], str(summary_path), 0)
            if matched_changes["pairs"]:
                close("analysis_primary_strict_matched_change",
                      analysis["primary_matched_median_within_pair_interaction_change_g"],
                      matched_changes["median_within_pair_change_g_per_delivered_kwh"], str(summary_path))

    def summarize_primary(keys):
        values = [primary_effects[key] for key in keys if key in primary_effects]
        return {"cases": len(values), "substitution_cases": sum(label == "substitution" for _, label in values),
                "median_g_per_delivered_kwh": float(np.median([point for point, _ in values])) if values else None}

    return {
        "all_pass": not failures, "year": year, "case_count": len(cases), "outcome_count": len(outcomes),
        "retained_sessions": len(membership), "frozen_sites": len(frozen), "observed_sites": int(cases.site_id.nunique()),
        "observed_shared_weeks": int(cases.start_utc.nunique()), "coverage_counts": dict(coverage_reasons),
        "strict_common_site_month_calendar_week_rank": len(common_keys) if year == 2024 else None,
        "same_year_frozen_anchor_count": len(common_keys) if year == 2023 else None,
        "additional_year_eligible_site_weeks": len(expected_included - set(original_keys)) if year == 2024 else None,
        "unmatched_2023_site_weeks": len(missing_original) if year == 2024 else None,
        "unmatched_2023_keys": [list(key) for key in sorted(missing_original)] if year == 2024 else None,
        "outcome_status_counts": dict(by_status), "practical_storage_counts": dict(practical_counts),
        "endpoint_interaction_counts": dict(interaction_counts), "failure_counts": dict(failures),
        "primary_endpoint_all_eligible": summarize_primary(expected_included),
        "primary_endpoint_strict_matched": summarize_primary(common_keys) if year == 2024 else None,
        "analysis_match_table_checked": checked_matches,
        "primary_cross_year_matched_changes": matched_changes,
        "failure_examples": examples, "max_absolute_errors": maxima,
        "scope": "Independent frozen sample, coverage, source records, ASAP budget, physical aggregate balances, and fixed-PV control optimum. Optimized half-hour trajectories use solver-reported residuals.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2023, 2024), default=2024)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path, help="Optional folder containing the strict match table.")
    parser.add_argument("--reference-results-dir", type=Path, default=ROOT / "results_harmonized_2023",
                        help="2023 common-PV outcome table for independent paired cross-year calculations.")
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--baseline-cases", type=Path, default=STUDY / "results" / "cases.csv")
    args = parser.parse_args()
    data_dir = args.data_dir or ROOT / ("data" if args.year == 2024 else "data_harmonized_2023")
    results_dir = args.results_dir or ROOT / ("results" if args.year == 2024 else "results_harmonized_2023")
    report = audit(data_dir, results_dir, args.protocol, args.baseline_cases, args.year,
                   args.analysis_dir or results_dir, args.reference_results_dir)
    (results_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key not in ("max_absolute_errors", "unmatched_2023_keys")},
                     indent=2, ensure_ascii=False))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

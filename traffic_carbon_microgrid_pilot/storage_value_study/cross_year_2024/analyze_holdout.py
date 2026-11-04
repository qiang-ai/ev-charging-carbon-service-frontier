"""Apply the frozen 2023 definitions to the 2024 storage study holdout."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
sys.path.insert(0, str(STUDY))
from analyze import interactions, make_pairs, summarize

import numpy as np
import pandas as pd

YEARS = (2023, 2024)
ALPHAS = (0.0, 1.0)
PV_SCALES = (0.0, 0.2, 0.4)
CAPACITIES = (0.0, 0.5, 0.125)
DESIGN = ["alpha", "pv_scale", "battery_fraction"]
SITE = ["city", "site_id"]
SITE_SEASON = [*SITE, "season"]
WEEK = [*SITE, "month", "week_rank"]
LEVELS = (("total", []), ("city", ["city"]), ("season", ["season"]), ("city_season", ["city", "season"]))


def table(folder, filename):
    return pd.read_csv(folder / filename, dtype={"site_id": str, "case_id": str})


def require_close(observed, expected, label, *, atol=1e-6):
    lhs = np.asarray(observed, dtype=float)
    rhs = np.asarray(expected, dtype=float)
    if not np.allclose(lhs, rhs, rtol=0, atol=atol, equal_nan=False):
        raise ValueError(f"{label} differs from the frozen protocol")


def read_cases(folder, year):
    cases = table(folder, "cases.csv")
    if cases.empty or cases.case_id.duplicated().any():
        raise ValueError(f"Cases must be nonempty and unique in {folder}")
    starts = pd.to_datetime(cases.start_utc, format="mixed", utc=True)
    ends = pd.to_datetime(cases.end_utc_exclusive, format="mixed", utc=True)
    rank = (starts.dt.day - 1) // 7 + 1
    expected = {"year": starts.dt.year, "month": starts.dt.month, "week_rank": rank}
    if (not starts.dt.year.eq(year).all() or not starts.dt.month.isin((1, 7)).all()
            or not starts.dt.dayofweek.eq(0).all() or not starts.eq(starts.dt.normalize()).all()
            or not ends.sub(starts).eq(pd.Timedelta(days=7)).all()
            or not (ends - pd.Timedelta(seconds=1)).dt.month.eq(starts.dt.month).all()):
        raise ValueError(f"Expected full Monday–Sunday UTC weeks in January or July {year}")
    if not cases.season.eq(np.where(starts.dt.month.eq(1), "winter", "summer")).all():
        raise ValueError("Season does not match the UTC month")
    for key, value in expected.items():
        if key in cases:
            require_close(cases[key], value, f"{year} {key}", atol=0)
        cases[key] = value
    if cases.duplicated(WEEK).any():
        raise ValueError(f"Repeated site/calendar-month/week ordinal in {year}")
    return cases


def core(frame):
    return frame.loc[frame.alpha.isin(ALPHAS) & frame.pv_scale.isin(PV_SCALES)].copy()


def read_reference(main_dir, small_dir):
    cases = read_cases(main_dir, 2023)
    small_cases = read_cases(small_dir, 2023)
    keys = ["case_id", "site_id", "start_utc", "n_sessions", "energy_kwh", "site_daily_energy_kwh", "budget_gbp"]
    left = cases[keys].sort_values("case_id").reset_index(drop=True)
    right = small_cases[keys].sort_values("case_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    main = core(table(main_dir, "outcomes.csv"))
    small = core(table(small_dir, "outcomes.csv"))
    pair_keys = ["case_id", "alpha", "pv_scale"]
    columns = pair_keys + ["carbon_kg", "operating_cost_gbp", "objective_lower_bound"]
    reference_controls = main.loc[~main.has_battery, columns]
    small_controls = small.loc[~small.has_battery, columns]
    check = reference_controls.merge(small_controls, on=pair_keys, how="outer", validate="one_to_one",
                                     indicator=True, suffixes=("_main", "_small"))
    if not check._merge.eq("both").all():
        raise ValueError("2023 main and small-capacity controls do not cover the same cells")
    for name in columns[len(pair_keys):]:
        require_close(check[f"{name}_main"], check[f"{name}_small"], f"2023 {name} controls", atol=1e-5)
    main["battery_fraction"] = np.where(main.has_battery, 0.5, 0.0)
    small = small.loc[small.has_battery].assign(battery_fraction=0.125)
    outcomes = pd.concat((main, small), ignore_index=True).assign(year=2023)
    return cases, outcomes


def read_harmonized_reference(directory, frozen_dir):
    replay_manifest = directory / "manifest.json"
    if replay_manifest.exists() and not json.loads(replay_manifest.read_text(encoding="utf-8-sig")).get("completed", True):
        raise ValueError("Complete the harmonized 2023 replay before comparison")
    replay = read_cases(directory, 2023)
    frozen = read_cases(frozen_dir, 2023)
    keys = ["case_id", "city", "site_id", "start_utc", "n_sessions", "energy_kwh", "site_daily_energy_kwh", "budget_gbp"]
    ordered_replay = replay[keys].sort_values("case_id").reset_index(drop=True)
    ordered_frozen = frozen[keys].sort_values("case_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered_replay, ordered_frozen, check_dtype=False, rtol=0, atol=1e-6)
    return replay, core(table(directory, "outcomes.csv"))


def frozen_scales(reference, holdout):
    frozen = reference.groupby(SITE, as_index=False).site_daily_energy_kwh.agg(["min", "max"])
    require_close(frozen["min"], frozen["max"], "2023 within-site equipment scale")
    frozen = frozen[SITE + ["min"]].rename(columns={"min": "frozen_daily_energy_kwh"})
    checked = holdout.merge(frozen, on=SITE, how="left", validate="many_to_one")
    if checked.frozen_daily_energy_kwh.isna().any():
        raise ValueError("2024 includes a replacement or unknown site")
    require_close(checked.site_daily_energy_kwh, checked.frozen_daily_energy_kwh, "2024 equipment scale")
    return frozen


def validate_outcomes(cases, outcomes, year):
    if not outcomes.battery_fraction.isin(CAPACITIES).all():
        raise ValueError("Unexpected battery fraction in core outcomes")
    if not outcomes.year.eq(year).all() or not outcomes.has_battery.eq(outcomes.battery_fraction.ne(0)).all():
        raise ValueError(f"Battery mode or year mismatch in {year} outcomes")
    design_meta = cases[["case_id", "site_daily_energy_kwh", "budget_gbp", "energy_kwh", "power_cap_kw"]]
    checked = outcomes.merge(design_meta, on="case_id", how="left", validate="many_to_one", suffixes=("", "_case"))
    if checked.site_daily_energy_kwh.isna().any():
        raise ValueError(f"Outcome without a case in {year}")
    require_close(checked.pv_kwp, checked.pv_scale * checked.site_daily_energy_kwh, f"{year} PV nameplate")
    require_close(checked.battery_kwh, checked.battery_fraction * checked.site_daily_energy_kwh, f"{year} BESS capacity")
    require_close(checked.battery_kw, checked.battery_kwh / 2, f"{year} BESS power")
    require_close(checked.power_cap_kw, checked.power_cap_kw_case, f"{year} power cap")
    require_close(checked.operating_budget_gbp, checked.budget_gbp, f"{year} shared operating budget")
    require_close(checked.energy_required_kwh, checked.energy_kwh, f"{year} required energy")
    feasible = checked.loc[checked.feasible]
    require_close(feasible.delivered_kwh, feasible.energy_kwh, f"{year} delivered energy", atol=1e-5)
    anchor = feasible.loc[feasible.alpha.eq(0) & feasible.pv_scale.eq(0) & feasible.battery_fraction.eq(0)]
    require_close(anchor.operating_cost_gbp, anchor.budget_gbp, f"{year} grid-only immediate budget anchor")


def fill_design(cases, outcomes):
    """Retain every expected design cell, including absent or infeasible solves."""
    factors = pd.DataFrame(product(ALPHAS, PV_SCALES, CAPACITIES), columns=DESIGN)
    expected = cases[["case_id", "year"]].merge(factors, how="cross")
    keys = ["case_id", "year", *DESIGN]
    joined = expected.merge(outcomes, on=keys, how="left", validate="one_to_one", indicator=True)
    missing = joined._merge.eq("left_only")
    joined.loc[missing, "status"] = "missing_outcome"
    for name in ("feasible", "optimal"):
        joined[name] = joined[name].eq(True)
    joined["has_battery"] = joined.battery_fraction.ne(0)
    coverage = joined.assign(missing_outcome=missing).groupby(["year", *DESIGN], as_index=False).agg(
        expected_outcomes=("case_id", "size"), missing_outcomes=("missing_outcome", "sum"),
        feasible_outcomes=("feasible", "sum"), optimal_outcomes=("optimal", "sum"),
    )
    coverage["available_outcomes"] = coverage.expected_outcomes - coverage.missing_outcomes
    return joined.drop(columns="_merge"), coverage


def paired_results(cases, outcomes):
    pairs = []
    effects = []
    context = cases[["case_id", "year", "month", "week_rank"]]
    for capacity in CAPACITIES[1:]:
        selected = outcomes.loc[outcomes.battery_fraction.isin((0, capacity))]
        contrasts = make_pairs(cases, selected).assign(battery_fraction=capacity)
        contrasts["missing_outcome"] = contrasts.status_no.eq("missing_outcome") | contrasts.status_yes.eq("missing_outcome")
        pairs.append(contrasts)
        difference = interactions(contrasts).merge(context, on="case_id", validate="many_to_one")
        difference["both_alpha_pairs_valid"] = difference.pair_valid_fixed & difference.pair_valid_flex
        effects.append(difference.assign(battery_fraction=capacity))
    return pd.concat(pairs, ignore_index=True), pd.concat(effects, ignore_index=True)


def build_week_matches(cases):
    fields = WEEK + ["case_id", "start_utc", "n_sessions", "energy_kwh"]
    reference = cases.loc[cases.year.eq(2023), fields]
    holdout = cases.loc[cases.year.eq(2024), fields]
    joined = reference.merge(holdout, on=WEEK, how="outer", validate="one_to_one", indicator=True,
                             suffixes=("_2023", "_2024"))
    joined["match_status"] = joined._merge.map({"both": "matched", "left_only": "2023_only", "right_only": "2024_only"})
    joined["match_id"] = joined[WEEK].astype(str).agg("|".join, axis=1)
    return joined.drop(columns="_merge").sort_values(WEEK)


def analysis_scopes(cases, matches):
    present = cases.groupby(SITE_SEASON).year.nunique().rename("n_years").reset_index()
    common = cases.merge(present.loc[present.n_years.eq(2), SITE_SEASON], on=SITE_SEASON, validate="many_to_one")
    panel = matches.loc[matches.match_status.eq("matched")]
    matched_ids = set(panel.case_id_2023) | set(panel.case_id_2024)
    return {
        "frozen_sites_observed": set(cases.case_id),
        "common_site_seasons": set(common.case_id),
        "common_site_week_ordinal": matched_ids,
    }


def pair_summary(pairs):
    summaries = []
    for label, fields in LEVELS:
        groups = ["year", "battery_fraction", "alpha", "pv_scale", *fields]
        result = summarize(pairs, groups)
        counts = pairs.assign(
            feasible_without=pairs.feasible_no, feasible_with=pairs.feasible_yes,
            only_storage_feasible=pairs.feasibility_class.eq("only_storage_feasible"),
            both_infeasible=pairs.feasibility_class.eq("both_infeasible"),
            below_threshold=pairs.practical_class.eq("below"),
        ).groupby(groups, as_index=False).agg(
            missing_pairs=("missing_outcome", "sum"), feasible_without=("feasible_without", "sum"),
            feasible_with=("feasible_with", "sum"), only_storage_feasible=("only_storage_feasible", "sum"),
            both_infeasible=("both_infeasible", "sum"), below_threshold=("below_threshold", "sum"),
        )
        summaries.append(result.merge(counts, on=groups, validate="one_to_one").assign(summary_level=label))
    return pd.concat(summaries, ignore_index=True).fillna({"city": "", "season": ""})


def interaction_summary(effects):
    rows = []
    for label, fields in LEVELS:
        groups = ["year", "battery_fraction", "pv_scale", *fields]
        for keys, sample in effects.groupby(groups, sort=True):
            valid = sample.loc[sample.both_alpha_pairs_valid, "interaction_g_per_kwh"]
            counts = sample.interaction_class.value_counts()
            n_sub = int(counts.get("substitution", 0))
            rows.append({**dict(zip(groups, keys)), "summary_level": label,
                         "site_weeks": len(sample), "sites": sample.site_id.nunique(),
                         "distinct_gb_weeks": sample.start_utc.nunique(), "valid_pairs": len(valid),
                         "substitution_pairs": n_sub, "substitution_percent": 100 * n_sub / len(sample),
                         "within_1g_pairs": int(counts.get("within_1g", 0)),
                         "complement_pairs": int(counts.get("complement", 0)),
                         "unresolved_pairs": int(counts.get("unresolved", 0)),
                         "median_interaction_g": valid.median(), "q25_interaction_g": valid.quantile(0.25),
                         "q75_interaction_g": valid.quantile(0.75)})
    return pd.DataFrame(rows).fillna({"city": "", "season": ""})


def compare_year_summaries(summary, dimensions, differences):
    keys = ["scope", "summary_level", "city", "season", "battery_fraction", "pv_scale", *dimensions]
    a = summary.loc[summary.year.eq(2023)].drop(columns="year")
    b = summary.loc[summary.year.eq(2024)].drop(columns="year")
    joined = a.merge(b, on=keys, how="outer", validate="one_to_one", suffixes=("_2023", "_2024"))
    joined["matches_calendar_week_rank"] = joined.scope.eq("common_site_week_ordinal")
    for column in differences:
        joined[f"change_{column}_2024_minus_2023"] = joined[f"{column}_2024"] - joined[f"{column}_2023"]
    return joined


def paired_year_changes(frame, matches, dimensions, metric, columns):
    panel = matches.loc[matches.match_status.eq("matched"), ["match_id", *WEEK, "case_id_2023", "case_id_2024"]]
    years = []
    for year in YEARS:
        cases = panel[["match_id", f"case_id_{year}"]].rename(columns={f"case_id_{year}": "case_id"})
        subset = frame.loc[frame.year.eq(year), ["case_id", "start_utc", *dimensions, metric, *columns]]
        years.append(subset.merge(cases, on="case_id", how="inner", validate="many_to_one"))
    combined = years[0].merge(years[1], on=["match_id", *dimensions], validate="one_to_one", suffixes=("_2023", "_2024"))
    combined[f"change_{metric}_2024_minus_2023"] = combined[f"{metric}_2024"] - combined[f"{metric}_2023"]
    return combined.merge(panel[["match_id", *WEEK]], on="match_id", validate="many_to_one")


def coverage_tables(cases, frozen, scopes):
    expected = frozen.merge(pd.DataFrame(product(YEARS, ("winter", "summer")), columns=["year", "season"]), how="cross")
    site_rows = []
    totals = []
    for scope, ids in scopes.items():
        subset = cases.loc[cases.case_id.isin(ids)]
        fields = ["year", *SITE_SEASON]
        observed = subset.groupby(fields, as_index=False).agg(
            site_weeks=("case_id", "size"), sessions=("n_sessions", "sum"), energy_kwh=("energy_kwh", "sum"),
        )
        site_rows.append(expected.merge(observed, on=fields, how="left", validate="one_to_one")
                         .fillna({"site_weeks": 0, "sessions": 0, "energy_kwh": 0}).assign(scope=scope))
        for year in YEARS:
            kept = subset.loc[subset.year.eq(year)]
            source = cases.loc[cases.year.eq(year)]
            totals.append({"scope": scope, "year": year, "frozen_sites": len(frozen), "retained_sites": kept.site_id.nunique(),
                           "retained_site_weeks": len(kept), "excluded_site_weeks_vs_observed": len(source) - len(kept),
                           "retained_sessions": int(kept.n_sessions.sum()),
                           "excluded_sessions_vs_observed": int(source.n_sessions.sum() - kept.n_sessions.sum()),
                           "retained_energy_kwh": float(kept.energy_kwh.sum()), "distinct_gb_weeks": kept.start_utc.nunique(),
                           "matches_calendar_week_rank": scope == "common_site_week_ordinal"})
    return pd.DataFrame(totals), pd.concat(site_rows, ignore_index=True)


def run(reference_dir, small_dir, holdout_dir, output_dir, harmonized_baseline=None):
    holdout_manifest = holdout_dir / "manifest.json"
    if holdout_manifest.exists() and not json.loads(holdout_manifest.read_text(encoding="utf-8-sig")).get("completed", True):
        raise ValueError("Complete the holdout run before comparison")
    if harmonized_baseline is None:
        reference, reference_outcomes = read_reference(reference_dir, small_dir)
    else:
        reference, reference_outcomes = read_harmonized_reference(harmonized_baseline, reference_dir)
    holdout = read_cases(holdout_dir, 2024)
    holdout_outcomes = core(table(holdout_dir, "outcomes.csv"))
    frozen = frozen_scales(reference, holdout)
    validate_outcomes(reference, reference_outcomes, 2023)
    validate_outcomes(holdout, holdout_outcomes, 2024)
    cases = pd.concat((reference, holdout), ignore_index=True)
    complete, design_coverage = fill_design(cases, pd.concat((reference_outcomes, holdout_outcomes), ignore_index=True))
    pairs, effects = paired_results(cases, complete)
    matches = build_week_matches(cases)
    scopes = analysis_scopes(cases, matches)
    coverage, sites = coverage_tables(cases, frozen, scopes)
    summary = pd.concat([pair_summary(pairs.loc[pairs.case_id.isin(ids)]).assign(scope=scope)
                         for scope, ids in scopes.items() if ids], ignore_index=True)
    effects_summary = pd.concat([interaction_summary(effects.loc[effects.case_id.isin(ids)]).assign(scope=scope)
                                 for scope, ids in scopes.items() if ids], ignore_index=True)
    comparison = compare_year_summaries(summary, ["alpha"], ["median_saved_g", "substantial_percent"])
    effects_comparison = compare_year_summaries(effects_summary, [], ["median_interaction_g", "substitution_percent"])
    pair_changes = paired_year_changes(pairs, matches, ["battery_fraction", "alpha", "pv_scale"], "saved_g_per_kwh",
                                       ["pair_valid", "practical_class", "saved_g_lower", "saved_g_upper"])
    interaction_changes = paired_year_changes(effects, matches, ["battery_fraction", "pv_scale"], "interaction_g_per_kwh",
                                              ["both_alpha_pairs_valid", "interaction_class", "interaction_lower_g", "interaction_upper_g"])
    primary = effects_comparison.loc[effects_comparison.summary_level.eq("total") & effects_comparison.battery_fraction.eq(0.5)
                                     & effects_comparison.pv_scale.eq(0.2)].copy()
    primary_matched = interaction_changes.loc[interaction_changes.battery_fraction.eq(0.5) & interaction_changes.pv_scale.eq(0.2)
                                              & interaction_changes.both_alpha_pairs_valid_2023 & interaction_changes.both_alpha_pairs_valid_2024]
    values = primary_matched.change_interaction_g_per_kwh_2024_minus_2023
    out = {
        "storage_pairs": pairs, "interactions": effects, "paired_summary": summary,
        "interaction_summary": effects_summary, "cross_year_comparison": comparison,
        "cross_year_interaction_comparison": effects_comparison, "cohort_coverage": coverage,
        "site_coverage": sites, "week_ordinal_matches": matches, "design_coverage": design_coverage,
        "matched_pair_changes": pair_changes, "matched_interaction_changes": interaction_changes,
        "primary_comparison": primary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    def records(frame):
        return json.loads(frame.to_json(orient="records", double_precision=12))

    result = {
        "design": {"years": list(YEARS), "alphas": list(ALPHAS), "pv_scales": list(PV_SCALES), "battery_fractions": list(CAPACITIES),
                   "reference_analysis_source": "harmonized_2023_replay" if harmonized_baseline else "original_frozen_2023_results",
                   "pv_method_harmonized": harmonized_baseline is not None,
                   "battery_and_pv_nameplate_basis": "2023 site_daily_energy_kwh fixed in both years",
                   "budget_rule": "site-week grid-only ASAP electricity cost, fixed across design",
                   "substantial_threshold": "both >=5 g CO2 per delivered kWh and >=5 percent, certified by objective bounds",
                   "substitution_threshold": "upper bound on full-window minus fixed storage gain < -1 g CO2 per delivered kWh",
                   "week_match": "same site, UTC calendar month, and first-through-last full Monday–Sunday week ordinal",
                   "interpretation": "paired ex-post purchased-electricity carbon footprints under proxy windows"},
        "cohort_coverage": records(coverage),
        "primary": records(primary),
        "primary_matched_valid_case_pairs": len(primary_matched),
        "primary_matched_median_within_pair_interaction_change_g": float(values.median()) if len(values) else None,
        "week_match_counts": matches.match_status.value_counts().to_dict(),
        "missing_design_outcomes": int(design_coverage.missing_outcomes.sum()),
    }
    (output_dir / "holdout_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(coverage.to_string(index=False))
    print(primary[["scope", "substitution_pairs_2023", "site_weeks_2023", "substitution_pairs_2024", "site_weeks_2024",
                   "median_interaction_g_2023", "median_interaction_g_2024"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=STUDY / "results")
    parser.add_argument("--reference-small-dir", type=Path, default=STUDY / "results_small_storage")
    parser.add_argument("--harmonized-baseline", type=Path,
                        help="2023 replay directory with shared-PV-method cases and combined battery-fraction outcomes")
    parser.add_argument("--holdout-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, help="Defaults to holdout-dir")
    args = parser.parse_args()
    run(args.reference_dir, args.reference_small_dir, args.holdout_dir, args.output_dir or args.holdout_dir,
        args.harmonized_baseline)


if __name__ == "__main__":
    main()

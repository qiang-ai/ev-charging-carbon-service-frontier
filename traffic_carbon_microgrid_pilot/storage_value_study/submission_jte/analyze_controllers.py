"""Summarize paired operating policies under the 2024 holdout."""
import json

from run_controllers import HOLDOUT, PLAN, ROOT, forecast_for, np, read_csv, read_inputs, train, write_csv


def median(values):
    return float(np.median(values))


def percentile(values):
    return [float(v) for v in np.percentile(values, [25, 50, 75])]


def score_group(rows):
    energy = np.array([float(r["energy_kwh"]) for r in rows])
    co2 = np.array([float(r["carbon_kg"]) for r in rows])
    cost = np.array([float(r["cost_gbp"]) for r in rows])
    budget = np.array([float(r["budget_gbp"]) for r in rows])
    equal = np.array([float(r["oracle_equal_realized_cost_carbon_kg"]) for r in rows])
    shared = np.array([float(r["oracle_shared_budget_carbon_kg"]) for r in rows])
    compliance = np.array([r["within_week_budget"] == "True" for r in rows])
    return {
        "weeks": len(rows), "sessions": sum(int(r["n_sessions"]) for r in rows),
        "energy_kwh": float(energy.sum()), "weekly_carbon_g_per_kwh_quartiles": percentile(1000 * co2 / energy),
        "pooled_carbon_g_per_kwh": float(1000 * co2.sum() / energy.sum()),
        "pooled_cost_gbp_per_kwh": float(cost.sum() / energy.sum()),
        "within_budget_weeks": int(compliance.sum()),
        "median_budget_excess_gbp_among_over_budget": median(cost[~compliance] - budget[~compliance])
        if not all(compliance) else 0.,
        "equal_realized_cost_oracle_gap_g_per_kwh_quartiles": percentile(1000 * (co2 - equal) / energy),
        "common_budget_oracle_gap_among_compliant_quartiles": percentile(1000 * (co2[compliance] - shared[compliance])
                                                                       / energy[compliance])
        if any(compliance) else [],
    }


def compare(candidate, comparator, label=None):
    diffs, joint = [], []
    for new, old in zip(candidate, comparator, strict=True):
        assert new["case_id"] == old["case_id"]
        saving = (float(old["carbon_kg"]) - float(new["carbon_kg"])) * 1000 / float(new["energy_kwh"])
        ok = new["within_week_budget"] == old["within_week_budget"] == "True"
        diffs.append({
            "case_id": new["case_id"], "city": new["city"], "month": int(new["month"]),
            "week_rank": int(new["week_rank"]), "battery_fraction": float(new["battery_fraction"]),
            "comparator": label or old["policy"], "carbon_saving_g_per_kwh": saving,
            "cost_change_gbp": float(new["cost_gbp"]) - float(old["cost_gbp"]),
            "energy_kwh": float(new["energy_kwh"]), "jointly_within_budget": ok,
        })
        if ok:
            joint.append(saving)
    all_saving = [d["carbon_saving_g_per_kwh"] for d in diffs]
    summary = {"carbon_saving_g_per_kwh_quartiles": percentile(all_saving),
               "pooled_carbon_saving_g_per_kwh": float(np.average(all_saving, weights=[d["energy_kwh"] for d in diffs])),
               "cases_saving_at_least_1g": sum(x >= 1 for x in all_saving),
               "cases_increasing_by_at_least_1g": sum(x <= -1 for x in all_saving),
               "median_cost_change_gbp": median([d["cost_change_gbp"] for d in diffs]),
               "jointly_within_budget": len(joint),
               "jointly_within_budget_carbon_saving_g_per_kwh_quartiles": percentile(joint) if joint else [],
               "jointly_within_budget_saving_at_least_1g": sum(x >= 1 for x in joint)}
    return summary, diffs


def forecast_errors(plan):
    cities, months = ("edinburgh", "glasgow"), (1, 7)
    training = train(plan["training_year"], cities, months)
    results = {}
    for city in cities:
        for month in months:
            folder = HOLDOUT / "data" / f"{city}_{plan['evaluation_year']}_{month:02d}"
            _, actual, _ = read_inputs(folder)
            forecast = forecast_for({"city": city, "month": month}, actual, training)
            results[f"{city}|{month}"] = {
                field: float(np.abs(getattr(actual, field) - getattr(forecast, field)).mean())
                for field in ("carbon", "price", "pv")}
    return results


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    root = ROOT / "results" / "controllers"
    rows = read_csv(root / "controller_outcomes.csv")
    diagnostic_path = root.parent / "signal_diagnostic" / "controller_outcomes.csv"
    diagnostic = read_csv(diagnostic_path) if diagnostic_path.exists() else []
    stats, comparisons, pairwise = {}, {}, []
    for fraction in plan["battery_fractions"]:
        groups = {p: sorted((r for r in rows if r["policy"] == p and float(r["battery_fraction"]) == fraction),
                            key=lambda r: r["case_id"]) for p in plan["policies"]}
        for policy, group in groups.items():
            stats[f"{policy}|{fraction}"] = score_group(group)
        for comparator in ("immediate", "tariff"):
            summary, pairs = compare(groups["carbon"], groups[comparator])
            comparisons[f"{comparator}|{fraction}"] = summary
            pairwise.extend(pairs)
        if diagnostic:
            group = sorted((r for r in diagnostic if float(r["battery_fraction"]) == fraction), key=lambda r: r["case_id"])
            stats[f"perfect_signals|{fraction}"] = score_group(group)
            summary, pairs = compare(group, groups["carbon"], "frozen_signals")
            comparisons[f"perfect_vs_frozen|{fraction}"] = summary
            pairwise.extend(pairs)
    residuals = {field: max(float(r[field]) for r in rows) for field in
                 ("max_session_delivery_error_kwh", "max_avoidable_checkpoint_error_kwh", "terminal_soc_error_kwh",
                  "max_soc_bound_error_kwh", "max_simultaneous_kwh", "max_forecast_budget_excess_gbp")}
    result = {"outcomes": len(rows), "diagnostic_outcomes": len(diagnostic), "groups": stats, "comparisons": comparisons,
              "maximum_residuals": residuals, "forecast_mae": forecast_errors(plan)}
    (root / "controller_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(root / "controller_pairs.csv", pairwise)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

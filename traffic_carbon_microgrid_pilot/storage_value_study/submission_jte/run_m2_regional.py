"""Replay the primary fixed-service experiment with archived NESO forecast factors."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from run_service import (
    FEAS_TOL, HOLDOUT, PROTOCOL, ROOT, SERVICE, Signals, describe, load_cases,
    make_model, np, read_csv, settings_for, sha256, solve, write_csv,
)
from scipy.stats import spearmanr


MATCHES = HOLDOUT / "analysis_common_pv" / "week_ordinal_matches.csv"
NATIONAL_BASELINE = ROOT / "results" / "service_outcomes.csv"
REGIONAL_API = "https://api.carbonintensity.org.uk/regional/intensity"
REGION_ID = 2
REGION = "South Scotland"
DNO = "SP Distribution"
STEP = 1800
SOURCES = {2023: "data_harmonized_2023", 2024: "data"}
GB_RAW = {2023: ROOT.parent / "data" / "raw", 2024: HOLDOUT / "data" / "raw"}
SITE_MAPPING = ROOT.parent / "data" / "site_mapping.csv"


def timestamp(label):
    return int(datetime.fromisoformat(label.replace("Z", "+00:00")).timestamp())


def week_request(start):
    start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end = start + timedelta(days=7)
    left, right = (d.strftime("%Y-%m-%dT%H:%MZ") for d in (start, end))
    return f"{REGIONAL_API}/{left}/{right}/regionid/{REGION_ID}"


def read_forecasts(data, source):
    values = {}
    for row in data:
        start = timestamp(row["from"])
        forecast = float(row["intensity"]["forecast"])
        if timestamp(row["to"]) - start != STEP or not np.isfinite(forecast) or forecast < 0:
            raise ValueError(f"Invalid carbon forecast interval at {source}: {row['from']}")
        if start in values and forecast != values[start]:
            raise ValueError(f"Conflicting duplicate carbon forecasts at {source}: {row['from']}")
        values[start] = forecast
    return values


def regional_week(raw, start, fetch):
    path = raw / f"south_scotland_{start[:10]}.json"
    url = week_request(start)
    if fetch and not path.is_file():
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=45) as response:
            content = response.read()
    else:
        content = path.read_bytes()
    regional = json.loads(content)["data"]
    actual_region = regional["regionid"], regional["shortname"], regional["dnoregion"]
    if actual_region != (REGION_ID, REGION, DNO):
        raise ValueError(f"Unexpected region {actual_region} from {url}")
    if any("actual" in row["intensity"] for row in regional["data"]):
        raise ValueError("The NESO regional payload has changed semantics")
    values = read_forecasts(regional["data"], path.name)
    left = timestamp(start)
    expected = set(range(left, left + 7 * 24 * 3600, STEP))
    if expected - values.keys():
        raise ValueError(f"Missing {len(expected - values.keys())} South Scotland slots in {start}")
    if not path.exists():
        path.write_bytes(content)
    metadata = {"source_url": url, "sha256": sha256(path), "region_id": REGION_ID,
                "matched_slots": len(expected), "boundary_slots_excluded": len(values.keys() - expected)}
    return start, {t: values[t] for t in expected}, path.name, metadata


def gb_forecasts(year):
    snapshots = sorted(GB_RAW[year].glob("neso_*.json"))
    records = [row for p in snapshots for row in json.loads(p.read_bytes())["data"]]
    return read_forecasts(records, f"GB {year}"), {p.name: sha256(p) for p in snapshots}


def checked_sites(protocol):
    mapped = read_csv(SITE_MAPPING)
    sites = {}
    for site_id in protocol["sites"]:
        connectors = [r for r in mapped if r["station_id"] == site_id]
        if not connectors or {r["region_id"] for r in connectors} != {str(REGION_ID)}:
            raise ValueError(f"Unconfirmed NESO region for {site_id}")
        sites[site_id] = {"postcodes": sorted({r["postcode"] for r in connectors}),
                          "inventory_connectors": len(connectors), "region_id": REGION_ID}
    return sites


def signal_rows(year, week_start, national, regional, gb):
    if len(national.ts) != 336 or not np.array_equal(national.ts, np.arange(national.ts[0],
                                                                          national.ts[0] + 336 * STEP, STEP)):
        raise ValueError(f"Incomplete UTC time grid in {week_start}")
    return [{"year": year, "week_start_utc": week_start,
             "start_utc": datetime.fromtimestamp(t, tz=timezone.utc).isoformat(),
             "gb_actual_g_per_kwh": float(carbon), "gb_forecast_g_per_kwh": gb[t],
             "south_scotland_forecast_g_per_kwh": regional[t]}
            for t, carbon in zip(national.ts, national.carbon, strict=True)]


def solve_case(task):
    case, sessions, sig, protocol, plan, factors, source = task
    signal = Signals(sig.ts, np.asarray(factors, dtype=float), sig.price, sig.pv)
    cap = protocol["power_cap_kw"]
    scale = plan["primary_pv_scale"]
    minutes = plan["primary_checkpoint_minutes"]
    energy = np.array([s.energy for s in sessions])
    results = []
    for fraction in plan["battery_fractions"]:
        settings = settings_for(case, protocol, scale, fraction)
        model, before, asap = make_model(sessions, signal, cap, settings, minutes)
        budget = case["budget_gbp"]
        info, x = solve(model, "carbon", budget, (-before, -asap), (case["case_id"], source, fraction))
        if x is None:
            raise ValueError(f"Unchanged service and budget should remain feasible: {case['case_id']}")
        result = describe(model, before, asap, energy, info, x, None, False)
        recomputed = float(np.dot(x[model.grid].ravel(), signal.carbon) / 1000)
        if (abs(result["carbon_kg"] - recomputed) > FEAS_TOL
                or result["operating_cost_gbp"] > budget + 2 * FEAS_TOL
                or np.max(asap - before @ x) > 2 * FEAS_TOL):
            raise ValueError(f"Carbon, budget or service check failed: {case['case_id']}, {source}, {fraction}")
        results.append({"signal": source, **{k: case[k] for k in ("case_id", "year", "city", "month", "week_rank",
                                                                "site_id", "n_sessions", "energy_kwh", "budget_gbp")},
                        "pv_scale": scale, "checkpoint_min": minutes, "battery_fraction": fraction, **result})
    return results


def summarize(rows, signal_data, protocol):
    summary, pairs = {}, []
    for year in SOURCES:
        year_data = [r for r in signal_data if r["year"] == year]
        actual = [r["gb_actual_g_per_kwh"] for r in year_data]
        summary[str(year)] = {"unique_utc_half_hours": len(year_data), "factors": {}, "storage_effects": {}}
        for source in ("gb_actual", "gb_forecast", "south_scotland_forecast"):
            values = [r[f"{source}_g_per_kwh"] for r in year_data]
            summary[str(year)]["factors"][source] = {"mean_g_per_kwh": float(np.mean(values)),
                "spearman_vs_gb_actual": float(spearmanr(actual, values).statistic)}
            base = {r["case_id"]: r for r in rows if int(r["year"]) == year and r["signal"] == source
                    and float(r["battery_fraction"]) == 0.0}
            if len(base) != 70:
                raise ValueError(f"Unexpected matched sample for {year} {source}: {len(base)}")
            for fraction in (0.125, 0.5):
                alt = {r["case_id"]: r for r in rows if int(r["year"]) == year and r["signal"] == source
                       and float(r["battery_fraction"]) == fraction}
                if base.keys() != alt.keys():
                    raise ValueError(f"Missing capacity pairs for {year} {source} {fraction}")
                gains, relative, certified = [], [], []
                for key in base:
                    a, b = base[key], alt[key]
                    n, kwh, s = float(a["carbon_kg"]), float(a["energy_kwh"]), float(b["carbon_kg"])
                    gain = 1000 * (n - s) / kwh
                    pct = 100 * (n - s) / n if n > FEAS_TOL else 0.0
                    bound = 1000 * (float(a["objective_lower_bound"]) - s) / kwh
                    gains.append(gain)
                    relative.append(pct)
                    certified.append(bound)
                    pairs.append({"year": year, "signal": source, "case_id": key, "battery_fraction": fraction,
                                  "gain_g_per_kwh": gain, "gain_pct": pct, "gain_lower_bound_g_per_kwh": bound})
                summary[str(year)]["storage_effects"][f"{source}|{fraction}"] = {
                    "site_weeks": len(base), "gain_g_per_kwh_q25_q50_q75": np.percentile(gains, [25, 50, 75]).tolist(),
                    "median_gain_pct": float(np.median(relative)),
                    "certified_gain_above_1g_weeks": sum(g > 1 for g in certified),
                    "gain_above_dual_threshold_weeks": sum(g >= protocol["substantial_threshold_g_per_kwh"]
                        and p >= protocol["substantial_threshold_relative_percent"]
                        for g, p in zip(gains, relative, strict=True)),
                }
    return summary, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing public South Scotland snapshots")
    parser.add_argument("--fetch-only", action="store_true", help="Validate and archive data without optimizing")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "m2_regional")
    args = parser.parse_args()
    out = args.output_dir
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    matched = [r for r in read_csv(MATCHES) if r["match_status"] == "matched"]
    if len(matched) != 70:
        raise ValueError("The matched 2023/24 panel has changed")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    starts = sorted({r[f"start_utc_{year}"] for year in SOURCES for r in matched})
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(lambda start: regional_week(raw, start, args.fetch), starts))
    regional = {start: values for start, values, _, _ in downloads}
    manifest = {"regional_field": "intensity.forecast", "regional_semantics": "Archived modeled average intensity forecast",
                "region_id": REGION_ID, "region": REGION, "dno": DNO,
                "dno_coverage_url": "https://www.spenergynetworks.co.uk/pages/our_distribution_network.aspx",
                "site_mapping_sha256": sha256(SITE_MAPPING), "checked_sites": checked_sites(protocol),
                "matched_panel_sha256": sha256(MATCHES), "regional_snapshots": {n: m for _, _, n, m in downloads}}
    print(f"Verified {len(starts)} distinct UTC weeks and {len(starts) * 336} South Scotland half-hours", flush=True)
    if args.fetch_only:
        (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return
    plan = json.loads(SERVICE.read_text(encoding="utf-8"))
    scope = {r[f"case_id_{year}"] for year in SOURCES for r in matched}
    tasks, signal_data, gb_hashes = [], {}, {}
    cohort = {}
    for year, folder in SOURCES.items():
        _, cases, _ = load_cases(HOLDOUT / folder, protocol, year)
        gb, gb_hashes[year] = gb_forecasts(year)
        for case, sessions, sig, _, _ in cases:
            if case["case_id"] not in scope:
                continue
            cohort[case["case_id"]] = case
            start = case["start_utc"]
            if start not in signal_data:
                signal_data[start] = signal_rows(year, start, sig, regional[start], gb)
            for source, mapping in (("gb_forecast", gb), ("south_scotland_forecast", regional[start])):
                tasks.append((case, sessions, sig, protocol, plan, [mapping[t] for t in sig.ts], source))
    if scope != cohort.keys() or len(tasks) != 2 * len(scope):
        raise ValueError("The regional experiment changed the matched panel")
    archive = [{**r, "signal": "gb_actual"} for r in read_csv(NATIONAL_BASELINE)
               if r["case_id"] in scope and r["design"] == "main" and r["objective"] == "preserve_readiness"
               and int(r["checkpoint_min"]) == plan["primary_checkpoint_minutes"]
               and float(r["pv_scale"]) == plan["primary_pv_scale"]]
    for row in archive:
        case = cohort[row["case_id"]]
        for field in ("n_sessions", "energy_kwh", "budget_gbp"):
            if abs(float(row[field]) - case[field]) > FEAS_TOL:
                raise ValueError(f"The comparison changed {field} in {row['case_id']}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        solved = []
        for i, result in enumerate(pool.map(solve_case, tasks), 1):
            solved.extend(result)
            if i % 40 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] solved alternative-factor site-weeks", flush=True)
    signals = [r for _, week in sorted(signal_data.items()) for r in week]
    summary, pairs = summarize(archive + solved, signals, protocol)
    write_csv(out / "outcomes.csv", solved)
    write_csv(out / "paired_effects.csv", pairs)
    write_csv(out / "aligned_carbon_factors.csv", signals)
    manifest.update({"gb_forecast_field": "intensity.forecast", "gb_raw_snapshot_sha256": gb_hashes,
                     "holdout_protocol_sha256": sha256(PROTOCOL), "service_protocol_sha256": sha256(SERVICE),
                     "gb_actual_service_outcomes_sha256": sha256(NATIONAL_BASELINE),
                     "outcomes_sha256": sha256(out / "outcomes.csv"), "case_weeks": len(scope),
                     "alternative_outcomes": len(solved), "results": summary})
    (out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Paired site-week study of incremental storage carbon footprint under a shared budget."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from optimize import Settings, Signals, early_reference, iso, np, read_inputs, select_sessions
from model import run_case

ALPHAS = (0, .25, .5, 1)
PV_SCALES = (0, .05, .10, .20, .40)
GROUPS = ('edinburgh_2023_01', 'edinburgh_2023_07', 'glasgow_2023_01', 'glasgow_2023_07')
WEEK_SLOTS = 336


def read_site_names(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['station_id']: row.get('site_name', row.get('address', '')) for row in csv.DictReader(stream)}


def sample_site_weeks(data_dir, group_names, max_sites, min_sessions, min_weeks, power_cap,
                      battery_fraction, budget_reduction):
    windows = defaultdict(list)
    preparation = []
    for group in group_names:
        directory = data_dir / group
        sessions, signals, invalid = read_inputs(directory)
        eligible, excluded = select_sessions(sessions, signals, power_cap, max_stations=10**6)
        site_names = read_site_names(directory / 'sessions.csv')
        placed = set()
        for start in range(0, len(signals.ts) - WEEK_SLOTS + 1, 48):
            first = datetime.fromtimestamp(signals.ts[start], timezone.utc)
            if first.weekday() != 0 or first.hour != 0:
                continue
            stop = start + WEEK_SLOTS
            block = Signals(*(getattr(signals, field)[start:stop] for field in ('ts','carbon','price','pv')))
            lower, upper = block.ts[0], block.ts[-1] + 1800
            by_site = defaultdict(list)
            for session in eligible:
                if session.start >= lower and session.end <= upper:
                    by_site[session.station_id].append(session)
                    placed.add(session.session_id)
            for site, records in sorted(by_site.items()):
                if len(records) >= min_sessions:
                    city = group.split('_')[0]
                    season = 'winter' if first.month == 1 else 'summer'
                    case = {'case_id': f'{group}|{first:%Y-%m-%d}|{site}', 'city': city,
                            'group': group, 'season': season, 'site_id': site,
                            'site_name': site_names.get(site, ''), 'start_utc': iso(lower),
                            'end_utc_exclusive': iso(upper), 'n_sessions': len(records),
                            'energy_kwh': sum(s.energy for s in records)}
                    windows[city, site].append((case, records, block))
        preparation.append({'group': group, 'prepared_sessions': len(sessions),
                            'eligible_ac_sessions': len(eligible), 'invalid': len(invalid),
                            'model_exclusions': dict(Counter(r['reason'] for r in excluded)),
                            'complete_utc_week_sessions': len(placed)})
    available = defaultdict(list)
    for (city, site), blocks in windows.items():
        by_season = Counter(case['season'] for case, _, _ in blocks)
        if min(by_season['winter'], by_season['summer']) >= min_weeks:
            average = sum(case['energy_kwh'] for case, _, _ in blocks) / (7 * len(blocks))
            available[city].append((average, site))
    chosen = {}
    tasks = []
    for city in sorted(available):
        ranked = sorted(available[city])
        indexes = np.rint(np.linspace(0, len(ranked) - 1, min(max_sites, len(ranked)))).astype(int)
        chosen[city] = {'eligible_sites': len(ranked), 'selected_sites': [ranked[i][1] for i in indexes]}
        for index in indexes:
            average, site = ranked[index]
            for case, records, block in windows[city, site]:
                case['site_daily_energy_kwh'] = average
                case['battery_kwh'] = battery_fraction * average
                case['battery_kw'] = case['battery_kwh'] / 2
                case['power_cap_kw'] = power_cap
                immediate, _ = early_reference(records, block, power_cap, [site])
                load = immediate[0]
                reference_cost = float(load @ block.price / 1000)
                case['reference_cost_gbp'] = reference_cost
                case['budget_gbp'] = reference_cost - budget_reduction * abs(reference_cost)
                case['reference_carbon_kg'] = float(load @ block.carbon / 1000)
                slack = np.array([max(0, (s.end - s.start) / 3600 - s.energy / min(s.rated, power_cap)) for s in records])
                energy = np.array([s.energy for s in records])
                case['energy_weighted_slack_min'] = float(60 * np.average(slack, weights=energy))
                case['median_slack_min'] = float(60 * np.median(slack))
                case['share_energy_slack_at_least_60min'] = float(energy[slack >= 1].sum() / energy.sum())
                tasks.append((case, records, block, load))
    return sorted(tasks, key=lambda entry: entry[0]['case_id']), preparation, chosen


def solve_week(task):
    case, records, signals, immediate, pv_scales, time_limit = task
    outcomes = []
    for scale in pv_scales:
        pv_kwp = scale * case['site_daily_energy_kwh']
        config = Settings(pv_kwp=pv_kwp, battery_kwh=case['battery_kwh'], battery_kw=case['battery_kw'],
                          roundtrip_efficiency=.9, initial_soc_fraction=.5, battery_wear=.01,
                          time_limit=time_limit, mip_rel_gap=1e-4)
        available = .5 * signals.pv * pv_kwp
        pv_yield = float(available.sum())
        overlap = float(np.minimum(immediate, available).sum())
        values = run_case(records, signals, case['power_cap_kw'], config,
                          alphas=ALPHAS, budget_gbp=case['budget_gbp'])
        for value in values:
            outcomes.append({**value, 'case_id': case['case_id'], 'pv_scale': scale, 'pv_kwp': pv_kwp,
                             'pv_available_kwh': pv_yield, 'pv_to_ev': pv_yield / case['energy_kwh'],
                             'asap_pv_overlap_fraction': overlap / case['energy_kwh']})
    return case['case_id'], outcomes


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'Cannot write empty table {path.name}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--max-sites', type=int, default=6)
    parser.add_argument('--min-sessions', type=int, default=5)
    parser.add_argument('--min-weeks', type=int, default=3)
    parser.add_argument('--power-cap', type=float, default=22)
    parser.add_argument('--battery-fraction', type=float, default=.5)
    parser.add_argument('--budget-reduction', type=float, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--time-limit', type=float, default=15)
    parser.add_argument('--limit-cases', type=int)
    parser.add_argument('--groups', nargs='+', default=GROUPS)
    args = parser.parse_args()
    if args.battery_fraction <= 0 or not 0 <= args.budget_reduction < 1:
        parser.error('Use a positive battery fraction and budget reduction in [0, 1).')
    tasks, preparation, chosen = sample_site_weeks(args.data_dir, args.groups, args.max_sites,
                                                   args.min_sessions, args.min_weeks, args.power_cap,
                                                   args.battery_fraction, args.budget_reduction)
    if args.limit_cases:
        tasks = tasks[:args.limit_cases]
    if not tasks:
        raise SystemExit('No paired site-weeks pass the prespecified sample filters.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / 'cases.csv', (case for case, _, _, _ in tasks))
    write_csv(args.out_dir / 'case_sessions.csv', ({'case_id':case['case_id'], **asdict(s)}
                for case, records, _, _ in tasks for s in records))
    inputs = {}
    for group in args.groups:
        for name in ('sessions.csv', 'signals.csv'):
            with (args.data_dir / group / name).open('rb') as stream:
                inputs[f'{group}/{name}'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {'data_groups': list(args.groups), 'input_sha256': inputs, 'sites': chosen,
                'preparation': preparation, 'max_sites_per_city': args.max_sites,
                'minimum_complete_sessions_per_week': args.min_sessions, 'minimum_weeks_per_season': args.min_weeks,
                'power_cap_kw': args.power_cap, 'weekly_blocks': 'complete Monday-Sunday UTC weeks',
                'n_case_weeks': len(tasks), 'n_sessions': sum(case['n_sessions'] for case, _, _, _ in tasks),
                'n_physical_site_proxies': len({(case['city'],case['site_id']) for case, _, _, _ in tasks}),
                'alphas': ALPHAS, 'pv_scales_kwp_per_mean_daily_ev_kwh': PV_SCALES,
                'budget': 'case-level grid-only ASAP cost minus budget_reduction times its absolute value, held constant across the grid',
                'budget_reduction': args.budget_reduction,
                'battery_kwh_per_mean_daily_ev_kwh': args.battery_fraction,
                'battery_kw_per_mean_daily_ev_kwh': args.battery_fraction/2,
                'battery_roundtrip_efficiency': .9, 'battery_wear_gbp_per_kwh_throughput': .01,
                'min_relevant_relative_percent':5, 'min_relevant_g_per_delivered_kwh':5,
                'time_limit_s_per_solve': args.time_limit, 'mip_rel_gap':1e-4,
                'completed':False}
    manifest_path = args.out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key:manifest[key] for key in ('sites','n_case_weeks','n_sessions')}, ensure_ascii=True), flush=True)
    jobs = [(*task, PV_SCALES, args.time_limit) for task in tasks]
    all_outcomes = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = pool.map(solve_week, jobs) if pool else map(solve_week, jobs)
        for number, (case_id, outcomes) in enumerate(results, 1):
            all_outcomes.extend(outcomes)
            if number == 1 or number % 8 == 0 or number == len(tasks):
                write_csv(args.out_dir / 'outcomes.csv', all_outcomes)
                print(f'[{number}/{len(tasks)}] {case_id} | {len(all_outcomes)} solves', flush=True)
    finally:
        if pool:
            pool.shutdown()
    manifest['completed'] = True
    manifest['n_solutions'] = len(all_outcomes)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

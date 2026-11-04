"""Reproduce the frozen 2023 study on 2024 charging sessions and signals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
sys.path.insert(0, str(STUDY))
from run_study import Settings, Signals, early_reference, iso, np, read_inputs, select_sessions, write_csv
from model import run_case

PROTOCOL = ROOT / 'protocol.json'
ANCHORS = ('results/cases.csv', 'results/manifest.json', 'results/outcomes.csv',
           'results_small_storage/outcomes.csv')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def freeze_protocol():
    if PROTOCOL.exists():
        raise SystemExit('The protocol is already frozen.')
    sites = {}
    for row in read_csv(STUDY / ANCHORS[0]):
        site = row['site_id']
        details = {'city':row['city'], 'site_name':row['site_name'],
                   'site_daily_energy_kwh':float(row['site_daily_energy_kwh'])}
        if site in sites and sites[site] != details:
            raise ValueError(f'Inconsistent frozen values for {site}')
        sites[site] = details
    anchor = json.loads((STUDY / ANCHORS[1]).read_text(encoding='utf-8'))
    if len(sites) != 12 or not anchor['completed']:
        raise ValueError('The anchor must be the audited 12-site 2023 study.')
    protocol = {
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'baseline_year':2023, 'holdout_year':2024, 'months':[1,7],
        'baseline_sha256':{name:sha256(STUDY/name) for name in ANCHORS},
        'sites':dict(sorted(sites.items())), 'site_replacements_allowed':False,
        'power_cap_kw':anchor['power_cap_kw'], 'min_complete_sessions_per_week':5,
        'week_selection':'Full calendar Monday-Sunday UTC weeks within January or July. Retain every frozen site-week with complete signals and >=5 eligible complete sessions. Missing weeks keep their calendar rank.',
        'matched_comparison':'Exact intersection of frozen site_id, calendar month and full UTC week rank in 2023 and 2024. Weeks in different years are different market observations.',
        'alphas':[0,1], 'pv_scales':[0,.2,.4], 'battery_fractions':[0,.125,.5],
        'battery_hours':2, 'battery_roundtrip_efficiency':.9,
        'initial_soc_fraction':.5, 'battery_wear_gbp_per_kwh_throughput':.01,
        'budget_rule':'Compute the immediate charging, grid-only operating cost from the actual sessions and prices in each site-week. Use that same cap for every configuration in the site-week.',
        'substantial_threshold_g_per_kwh':5, 'substantial_threshold_relative_percent':5,
        'interaction_threshold_g_per_kwh':1,
        'primary_endpoint':{'pv_scale':.2,'battery_fraction':.5,'contrast':'incremental storage saving at alpha=1 minus incremental storage saving at alpha=0'},
        'time_limit_s_per_solve':15,'mip_rel_gap':1e-4,
    }
    dump(PROTOCOL, protocol)


def weeks_of_month(year, month):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year, month+1, 1, tzinfo=timezone.utc)
    monday = start + timedelta(days=(-start.weekday())%7)
    rank = 1
    while monday+timedelta(days=7) <= end:
        yield rank, monday.timestamp(), (monday+timedelta(days=7)).timestamp()
        rank += 1
        monday += timedelta(days=7)


def load_cases(data_dir, protocol, year):
    coverage, cases, prepared = [], [], []
    cities = sorted({site['city'] for site in protocol['sites'].values()})
    cap = protocol['power_cap_kw']
    for city in cities:
        sites = {k:v for k,v in protocol['sites'].items() if v['city']==city}
        for month in protocol['months']:
            group = f'{city}_{year}_{month:02d}'
            source_sessions, sig, invalid = read_inputs(data_dir / group)
            eligible, rejected = select_sessions(source_sessions, sig, cap, len(sites), station_ids=list(sites))
            by_site = defaultdict(list)
            for session in eligible:
                by_site[session.station_id].append(session)
            prepared.append({'group':group,'prepared_sessions':len(source_sessions),
                             'eligible_frozen_sessions':len(eligible),'invalid':len(invalid),
                             'excluded_by_model':len(rejected)})
            for rank, start, end in weeks_of_month(year,month):
                week_mask = (sig.ts >= start) & (sig.ts < end)
                slots = int(week_mask.sum())
                complete = slots == 336 and np.array_equal(sig.ts[week_mask], np.arange(start,end,1800))
                week = Signals(*(getattr(sig, key)[week_mask] for key in ('ts','carbon','price','pv')))
                for site_id, site in sites.items():
                    records = [s for s in by_site[site_id] if s.start>=start and s.end<=end]
                    admitted = complete and len(records)>=protocol['min_complete_sessions_per_week']
                    state = 'included' if admitted else ('incomplete_signals' if not complete else 'too_few_sessions')
                    row = {'year':year,'month':month,'week_rank':rank,'city':city,
                           'season':'winter' if month==1 else 'summer','group':group,
                           'site_id':site_id,'site_name':site['site_name'],
                           'start_utc':iso(start),'end_utc_exclusive':iso(end),
                           'n_sessions':len(records),'signal_slots':slots,'coverage_status':state}
                    coverage.append(row)
                    if not admitted:
                        continue
                    daily = site['site_daily_energy_kwh']
                    immediate,_ = early_reference(records,week,cap,[site_id])
                    load = immediate[0]
                    energy = np.array([s.energy for s in records])
                    slack = np.array([max(0,(s.end-s.start)/3600-s.energy/min(s.rated,cap)) for s in records])
                    case = {**row,'case_id':f'{group}|{iso(start)[:10]}|{site_id}',
                            'energy_kwh':float(energy.sum()),'site_daily_energy_kwh':daily,
                            'battery_kwh':.5*daily,'battery_kw':.25*daily,'power_cap_kw':cap,
                            'budget_gbp':float(load@week.price/1000),
                            'reference_cost_gbp':float(load@week.price/1000),
                            'reference_carbon_kg':float(load@week.carbon/1000),
                            'energy_weighted_slack_min':float(60*np.average(slack,weights=energy)),
                            'median_slack_min':float(60*np.median(slack)),
                            'share_energy_slack_at_least_60min':float(energy[slack>=1].sum()/energy.sum())}
                    cases.append((case,records,week,load,protocol))
    return coverage, cases, prepared


def solve_week(task):
    case, sessions, signals, immediate, protocol = task
    output = []
    daily = case['site_daily_energy_kwh']
    for scale in protocol['pv_scales']:
        pv_kwp = scale*daily
        production = .5*signals.pv*pv_kwp
        total_pv = float(production.sum())
        overlap = float(np.minimum(immediate,production).sum())
        for fraction in (f for f in protocol['battery_fractions'] if f>0):
            capacity = daily*fraction
            settings = Settings(pv_kwp=pv_kwp,battery_kwh=capacity,battery_kw=capacity/protocol['battery_hours'],
                                roundtrip_efficiency=protocol['battery_roundtrip_efficiency'],
                                initial_soc_fraction=protocol['initial_soc_fraction'],
                                battery_wear=protocol['battery_wear_gbp_per_kwh_throughput'],
                                time_limit=protocol['time_limit_s_per_solve'],mip_rel_gap=protocol['mip_rel_gap'])
            for value in run_case(sessions,signals,case['power_cap_kw'],settings,
                                  alphas=tuple(protocol['alphas']),budget_gbp=case['budget_gbp']):
                if not value['has_battery'] and fraction!=.5:
                    continue
                output.append({**value,'case_id':case['case_id'],'year':case['year'],
                               'battery_fraction':fraction if value['has_battery'] else 0,
                               'pv_scale':scale,'pv_kwp':pv_kwp,'pv_available_kwh':total_pv,
                               'pv_to_ev':total_pv/case['energy_kwh'],
                               'asap_pv_overlap_fraction':overlap/case['energy_kwh']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--harmonized-2023',action='store_true',help='Replay original 2023 site-weeks with a common PV model.')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--results-dir',type=Path)
    args = parser.parse_args()
    if args.freeze_only:
        freeze_protocol()
        print(PROTOCOL)
        return
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for name, digest in protocol['baseline_sha256'].items():
        if sha256(STUDY/name) != digest:
            raise ValueError(f'The 2023 reference was changed: {name}')
    year = protocol['baseline_year'] if args.harmonized_2023 else protocol['holdout_year']
    source = args.data_dir or ROOT/('data_harmonized_2023' if args.harmonized_2023 else 'data')
    out = args.results_dir or ROOT/('results_harmonized_2023' if args.harmonized_2023 else 'results')
    coverage,cases,prepared = load_cases(source,protocol,year)
    if not cases:
        raise SystemExit('No frozen site-week has sufficient complete observations.')
    if args.harmonized_2023:
        anchors = {row['case_id']:row for row in read_csv(STUDY/ANCHORS[0])}
        if {case['case_id'] for case,*_ in cases} != set(anchors):
            raise ValueError('The common-PV replay must use exactly the original 2023 site-weeks.')
        for case,*_ in cases:
            for field in ('n_sessions','energy_kwh','budget_gbp','reference_carbon_kg'):
                if abs(case[field]-float(anchors[case['case_id']][field]))>1e-7:
                    raise ValueError(f'Non-PV baseline input changed in {case["case_id"]}: {field}')
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'coverage.csv',coverage)
    write_csv(out/'cases.csv',(case for case, *_ in cases))
    write_csv(out/'case_sessions.csv',({'case_id':case['case_id'],**asdict(s)} for case,records,*_ in cases for s in records))
    input_hashes = {f'{row["group"]}/{name}':sha256(source/row['group']/name)
                    for row in prepared for name in ('sessions.csv','signals.csv')}
    manifest = {'study':'2023 common-PV replay' if args.harmonized_2023 else 'frozen 2023 design, 2024 holdout',
                'year':year,'protocol_sha256':sha256(PROTOCOL),
                'input_sha256':input_hashes,'preparation':prepared,'n_case_weeks':len(cases),
                'n_sessions':sum(case['n_sessions'] for case,*_ in cases),
                'n_coverage_cells':len(coverage),'completed':False}
    dump(out/'manifest.json',manifest)
    print(json.dumps({key:manifest[key] for key in ('n_case_weeks','n_sessions','n_coverage_cells')}),flush=True)
    outcomes = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    try:
        batches = pool.map(solve_week,cases) if pool else map(solve_week,cases)
        for number, rows in enumerate(batches,1):
            outcomes.extend(rows)
            if number%12==0 or number==len(cases):
                print(f'[{number}/{len(cases)}] {len(outcomes)} unique scenarios',flush=True)
    finally:
        if pool:
            pool.shutdown()
    write_csv(out/'outcomes.csv',outcomes)
    manifest['completed']=True
    manifest['n_solutions']=len(outcomes)
    dump(out/'manifest.json',manifest)


if __name__=='__main__':
    main()

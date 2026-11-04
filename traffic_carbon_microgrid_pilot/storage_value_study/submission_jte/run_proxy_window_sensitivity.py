"""Stress-test the unobserved session-end proxy without dropping requests."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

from run_service import (
    FEAS_TOL, HOLDOUT, Model, PROTOCOL, ROOT, SERVICE, csr_matrix,
    early_reference, load_cases, make_model, np, read_csv, settings_for, sha256, solve, write_csv,
)


OFFSETS = (-60, -30, -15, 0, 15, 30, 60)
FRACTIONS = (0.0, 0.125, 0.5)
STEP = 1800


def perturb(sessions, cap, stop, offset):
    """Preserve service capacity and non-overlap among retained connector sessions."""
    following = {}
    groups = defaultdict(list)
    for session in sessions:
        groups[session.connector_id].append(session)
    for group in groups.values():
        ordered = sorted(group, key=lambda s: (s.start, s.session_id))
        for current, next_session in zip(ordered, ordered[1:]):
            following[current.session_id] = next_session.start
    adjusted, audit = [], []
    for session in sessions:
        earliest = session.start + 3600 * session.energy / min(session.rated, cap)
        latest = min(stop, following.get(session.session_id, stop))
        if not earliest - 1e-6 <= session.end <= latest + 1e-6:
            raise ValueError(f'Original interval is outside perturbation bounds: {session.session_id}')
        requested = session.end + 60 * offset
        if offset < 0:
            end = min(session.end, max(earliest, requested))
            limit = 'energy' if end > requested + 1e-6 else 'none'
        elif offset > 0:
            end = min(latest, requested)
            limit = ('connector' if latest < stop else 'week_end') if end < requested - 1e-6 else 'none'
        else:
            end, limit = session.end, 'none'
        if (end > latest + 1e-6 or end <= session.start or
                min(session.rated, cap) * (end - session.start) / 3600 < session.energy - FEAS_TOL):
            raise ValueError(f'Adjusted interval is infeasible: {session.session_id}')
        adjusted.append(replace(session, end=end))
        audit.append({
            'session_id': session.session_id, 'connector_id': session.connector_id, 'start_utc_seconds': session.start,
            'original_end_utc_seconds': session.end, 'adjusted_end_utc_seconds': end, 'energy_kwh': session.energy,
            'power_kw': min(session.rated, cap), 'actual_offset_min': (end - session.end) / 60,
            'clipping_reason': limit,
        })
    return adjusted, audit


def solve_record(model, objective, budget, extra, context, before=None, asap=None, carbon_cap=None):
    info, x = solve(model, objective, budget, extra, context)
    record = {'status': info['status'], 'carbon_cap_kg': carbon_cap, 'carbon_kg': None, 'cost_gbp': None,
              'objective_lower_bound': info['objective_lower_bound'], 'objective_gap': info['objective_abs_gap'],
              'intrinsic_shortfall_kwh': None, 'avoidable_shortfall_kwh': None}
    if before is not None:
        record['intrinsic_shortfall_kwh'] = float(sum(s.energy for s in model.sessions) - asap.sum())
    if x is None:
        return record
    record['carbon_kg'] = float(model.carbon @ x)
    record['cost_gbp'] = float(model.cost @ x)
    if before is not None:
        readiness = before @ x
        if float(max(readiness - asap)) > 2 * FEAS_TOL:
            raise ValueError(f'Checkpoint energy exceeds physical limit: {context}')
        record['avoidable_shortfall_kwh'] = float(np.maximum(asap - readiness, 0).sum())
    return record


def solve_week(task):
    case, sessions, signals, protocol, service, frozen_anchor = task
    cap = protocol['power_cap_kw']
    scale = service['primary_pv_scale']
    minutes = service['primary_checkpoint_minutes']
    stop = signals.ts[-1] + STEP
    immediate, _ = early_reference(sessions, signals, cap, [case['site_id']])
    budget = float(immediate[0] @ signals.price / 1000)
    if abs(budget - case['budget_gbp']) > FEAS_TOL:
        raise ValueError(f"Reference budget changed: {case['case_id']}")
    site = {k: case[k] for k in ('case_id', 'year', 'month', 'week_rank', 'city', 'site_id', 'energy_kwh', 'n_sessions')}
    settings = {f: settings_for(case, protocol, scale, f) for f in FRACTIONS}
    fixed = {}
    for f in (0.0, 0.5):
        model = Model(sessions, signals, cap, 'fixed_pv_battery' if f else 'fixed_pv', settings[f],
                      force_mip=f > 0)
        fixed[f] = solve_record(model, 'carbon', budget, None, (case['case_id'], 'asap', f))
        if fixed[f]['status'] != 'optimal':
            raise ValueError(f"Unexpected ASAP infeasibility: {case['case_id']}")
    if abs(fixed[0.0]['carbon_kg'] - frozen_anchor[0]) > 2 * FEAS_TOL:
        raise ValueError(f"Original fixed carbon anchor changed: {case['case_id']}")
    outcomes, windows = [], []
    for offset in OFFSETS:
        changed, changes = perturb(sessions, cap, stop, offset)
        windows.extend({'case_id': case['case_id'], 'year': case['year'], 'requested_offset_min': offset, **row}
                       for row in changes)
        labels = {**site, 'requested_offset_min': offset, 'budget_gbp': budget}
        full = {}
        for f in (0.0, 0.5):
            model, _, _ = make_model(changed, signals, cap, settings[f])
            full[f] = solve_record(model, 'carbon', budget, None, (case['case_id'], offset, 'full', f))
            if full[f]['status'] != 'optimal':
                raise ValueError(f"Unexpected full-window infeasibility: {case['case_id']}, {offset}")
        recalculated_cap = full[0.0]['carbon_kg']
        if offset == 0 and abs(recalculated_cap - frozen_anchor[1]) > 2 * FEAS_TOL:
            raise ValueError(f"Original full-window carbon anchor changed: {case['case_id']}")
        if recalculated_cap > fixed[0.0]['carbon_kg'] + 2 * FEAS_TOL:
            raise ValueError(f"Immediate window does not nest: {case['case_id']}, {offset}")
        labels.update(c0_kg=fixed[0.0]['carbon_kg'], cmin_recalculated_kg=recalculated_cap,
                      cmin_original_kg=frozen_anchor[1])

        for objective, configs in (('asap_carbon', fixed), ('full_carbon', full)):
            outcomes.extend({**labels, 'objective': objective, 'battery_fraction': f, **record}
                            for f, record in configs.items())
        for f, config in settings.items():
            model, before, asap = make_model(changed, signals, cap, config, minutes)
            record = solve_record(model, 'carbon', budget, (-before, -asap),
                                  (case['case_id'], offset, 'service', f), before, asap)
            if record['status'] != 'optimal':
                raise ValueError(f"Unexpected fixed-service infeasibility: {case['case_id']}, {offset}, {f}")
            outcomes.append({**labels, 'objective': 'fixed_service', 'battery_fraction': f, **record})
            early_objective = -np.asarray(before.sum(axis=0)).ravel()
            carbon_row = csr_matrix(model.carbon.reshape(1, -1))
            for objective, target in (('strict_recalculated', recalculated_cap), ('strict_original', frozen_anchor[1])):
                if objective == 'strict_original' and abs(target - recalculated_cap) <= 1e-9:
                    record = {**strict_recalculated, 'carbon_cap_kg': target}
                else:
                    record = solve_record(model, early_objective, budget,
                                          (carbon_row, np.array([target + FEAS_TOL])),
                                          (case['case_id'], offset, objective, f), before, asap, target)
                if objective == 'strict_recalculated':
                    if record['status'] != 'optimal':
                        raise ValueError(f"Recalculated carbon anchor infeasible: {case['case_id']}, {offset}, {f}")
                    strict_recalculated = record
                outcomes.append({**labels, 'objective': objective, 'battery_fraction': f, **record})
    return outcomes, windows


def quartiles(values):
    if len(values) == 0:
        return None
    return dict(zip(('q25', 'median', 'q75'), map(float, np.quantile(values, (.25, .5, .75))), strict=True))


def summarize(outcomes, windows):
    groups, session_groups = defaultdict(list), defaultdict(list)
    for row in outcomes:
        groups[row['year'], row['requested_offset_min']].append(row)
    for row in windows:
        session_groups[row['year'], row['requested_offset_min']].append(row)
    report, effects = {}, []
    for (year, offset), rows in sorted(groups.items()):
        cases = {r['case_id'] for r in rows}
        index = {(r['case_id'], r['objective'], r['battery_fraction']): r for r in rows}
        if len(index) != 13 * len(cases):
            raise ValueError(f'Incomplete or duplicate result grid: {year}, {offset}')
        details = session_groups[year, offset]
        change = np.asarray([r['actual_offset_min'] for r in details])
        clipping = {k: sum(r['clipping_reason'] == k for r in details)
                    for k in ('energy', 'connector', 'week_end')}
        storage = {str(f): [] for f in FRACTIONS[1:]}
        fall, fall_certified, intrinsic = [], [], []
        for case_id in sorted(cases):
            a, b, c, d = [index[case_id, obj, f] for obj in ('asap_carbon', 'full_carbon') for f in (0.0, 0.5)]
            energy = a['energy_kwh']
            interaction_fall = (a['carbon_kg'] - b['carbon_kg'] - c['carbon_kg'] + d['carbon_kg']) * 1000 / energy
            certified = (a['objective_lower_bound'] - b['carbon_kg'] - c['carbon_kg'] +
                         d['objective_lower_bound']) * 1000 / energy > 1
            fall.append(interaction_fall)
            fall_certified.append(certified)
            control = index[case_id, 'fixed_service', 0.0]
            intrinsic.append(control['intrinsic_shortfall_kwh'])
            for f in FRACTIONS[1:]:
                battery = index[case_id, 'fixed_service', f]
                gain = (control['carbon_kg'] - battery['carbon_kg']) * 1000 / energy
                storage[str(f)].append(gain)
                effects.append({'case_id': case_id, 'year': year, 'requested_offset_min': offset,
                                'battery_fraction': f, 'energy_kwh': energy,
                                'interaction_fall_primary_g_per_kwh': interaction_fall,
                                'interaction_fall_primary_certified_above_1': certified,
                                'fixed_service_battery_gain_g_per_kwh': gain})
        denominator = sum(r['energy_kwh'] for r in rows if r['objective'] == 'asap_carbon' and r['battery_fraction'] == 0)
        strict = {}
        for objective in ('strict_recalculated', 'strict_original'):
            strict[objective] = {}
            for f in FRACTIONS:
                sample = [index[c, objective, f] for c in cases]
                feasible = [r for r in sample if r['status'] == 'optimal']
                complete = [r for r in feasible if index[r['case_id'], objective, 0.0]['status'] == 'optimal']
                paired_energy = sum(r['energy_kwh'] for r in complete)
                paired_control = sum(index[r['case_id'], objective, 0.0]['avoidable_shortfall_kwh'] for r in complete)
                paired_battery = sum(r['avoidable_shortfall_kwh'] for r in complete)
                strict[objective][str(f)] = {
                    'feasible_weeks': len(feasible), 'infeasible_weeks': len(cases) - len(feasible),
                    'pooled_shortfall_pct_among_feasible_weeks': (100 * sum(r['avoidable_shortfall_kwh'] for r in feasible)
                                                               / sum(r['energy_kwh'] for r in feasible)) if feasible else None,
                    'feasible_pair_weeks_with_no_battery': len(complete),
                    'paired_control_shortfall_pct': 100 * paired_control / paired_energy if complete else None,
                    'paired_battery_shortfall_pct': 100 * paired_battery / paired_energy if complete else None,
                }
        report[f'{year}|{offset}'] = {
            'site_weeks': len(cases), 'sessions': len(details), 'energy_kwh': denominator,
            'actual_end_offset_minutes': quartiles(change),
            'mean_actual_end_offset_minutes': float(np.mean(change)),
            'unshifted_sessions': int(np.count_nonzero(np.abs(change) <= 1e-6)),
            'clipped_sessions_by_reason': clipping,
            'pooled_intrinsic_shortfall_pct': 100 * sum(intrinsic) / denominator,
            'primary_interaction_fall_g_per_kwh': quartiles(fall),
            'primary_interaction_fall_certified_above_1g_weeks': int(sum(fall_certified)),
            'fixed_service_battery_gain_g_per_kwh': {f: quartiles(values) for f, values in storage.items()},
            'strict_targets': strict,
        }
    return report, effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, help='Optional small smoke run')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'proxy_window_sensitivity')
    args = parser.parse_args()
    protocol, service = [json.loads(p.read_text(encoding='utf-8')) for p in (PROTOCOL, SERVICE)]
    matches = HOLDOUT / 'analysis_common_pv' / 'week_ordinal_matches.csv'
    matched = [r for r in read_csv(matches) if r['match_status'] == 'matched']
    if len(matched) != 70:
        raise ValueError('The frozen matched panel must have 70 pairs')
    tasks, sources = [], {}
    for year, source, result in ((2023, 'data_harmonized_2023', 'results_harmonized_2023'),
                                 (2024, 'data', 'results')):
        included = {r[f'case_id_{year}'] for r in matched}
        anchor_path = HOLDOUT / result / 'outcomes.csv'
        sources[year] = sha256(anchor_path)
        anchors = {(r['case_id'], float(r['alpha'])): float(r['carbon_kg'])
                   for r in read_csv(anchor_path) if r['case_id'] in included and
                   float(r['pv_scale']) == service['primary_pv_scale'] and float(r['battery_fraction']) == 0}
        _, cohort, _ = load_cases(HOLDOUT / source, protocol, year)
        chosen = [(case, sessions, sig) for case, sessions, sig, _, _ in cohort if case['case_id'] in included]
        if {case['case_id'] for case, *_ in chosen} != included:
            raise ValueError(f'Missing matched site-weeks in {year}')
        for case, sessions, signals in chosen:
            original = tuple(anchors[case['case_id'], a] for a in (0.0, 1.0))
            tasks.append((case, sessions, signals, protocol, service, original))
    if args.limit:
        tasks = tasks[:args.limit]
    outcomes, windows = [], []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        iterator = pool.map(solve_week, tasks) if pool else map(solve_week, tasks)
        for i, (solved, changes) in enumerate(iterator, 1):
            outcomes.extend(solved)
            windows.extend(changes)
            if i % 10 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] {len(outcomes)} objective rows', flush=True)
    finally:
        if pool:
            pool.shutdown()
    report, effects = summarize(outcomes, windows)
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    files = {'outcomes.csv': outcomes, 'session_windows.csv': windows, 'paired_effects.csv': effects}
    for name, rows in files.items():
        write_csv(directory / name, rows)
    meta = {'site_weeks': len(tasks), 'outcomes': len(outcomes), 'session_windows': len(windows),
            'nominal_signed_offsets_minutes': OFFSETS, 'checkpoint_minutes_before_adjusted_end': service['primary_checkpoint_minutes'],
            'pv_scale': service['primary_pv_scale'], 'battery_fractions': FRACTIONS,
            'holdout_protocol_sha256': sha256(PROTOCOL), 'service_protocol_sha256': sha256(SERVICE),
            'original_carbon_anchor_outcomes_sha256': sources, 'matched_panel_sha256': sha256(matches),
            'output_sha256': {name: sha256(directory / name) for name in files},
            'design_note': 'All retained session starts, delivered energy, connector ratings, equipment, signals and budgets are unchanged. Extensions respect week end and the next retained connector session. Both early checkpoints and immediate-charging readiness at the checkpoint are recomputed after each endpoint change.',
            'summary': report}
    (directory / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

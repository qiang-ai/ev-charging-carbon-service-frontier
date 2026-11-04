"""Independently check the proxy-window cohort, bounds and outcome nesting."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from statistics import median

from audit_service import Audit, HOLDOUT, ROOT, json_file, read, sha, stamp


def main():
    audit = Audit()
    directory = ROOT / 'results' / 'proxy_window_sensitivity'
    summary = json_file(directory / 'summary.json')
    offsets = tuple(summary['nominal_signed_offsets_minutes'])
    audit.require(offsets == (-60, -30, -15, 0, 15, 30, 60), 'offset design')
    for name, expected in summary['output_sha256'].items():
        audit.require(sha(directory / name) == expected, 'output checksum for ' + name)
    match_path = HOLDOUT / 'analysis_common_pv' / 'week_ordinal_matches.csv'
    audit.require(sha(match_path) == summary['matched_panel_sha256'], 'matched panel checksum')
    matches = [r for r in read(match_path)
               if r['match_status'] == 'matched']
    original, week_ends, cohort = {}, {}, set()
    for year, result in ((2023, 'results_harmonized_2023'), (2024, 'results')):
        active = {row[f'case_id_{year}'] for row in matches}
        cohort.update(active)
        audit.require(sha(HOLDOUT / result / 'outcomes.csv') == summary['original_carbon_anchor_outcomes_sha256'][str(year)],
                      'original carbon anchor checksum')
        week_ends.update({r['case_id']: stamp(r['end_utc_exclusive']) for r in read(HOLDOUT / result / 'cases.csv')
                          if r['case_id'] in active})
        original.update({(r['case_id'], r['session_id']): r for r in read(HOLDOUT / result / 'case_sessions.csv')
                         if r['case_id'] in active})
    audit.require(len(cohort) == summary['site_weeks'] == 140, 'matched panel size')
    audit.require(len(original) == 5933, 'matched session count')
    windows = read(directory / 'session_windows.csv')
    audit.require(len(windows) == summary['session_windows'] == len(original) * len(offsets), 'session outcome count')
    connector = defaultdict(list)
    for (case, _), row in original.items():
        connector[case, row['connector_id']].append(row)
    next_start = {}
    for (case, _), group in connector.items():
        ordered = sorted(group, key=lambda row: float(row['start']))
        for a, b in zip(ordered, ordered[1:]):
            next_start[case, a['session_id']] = float(b['start'])
    covered = set()
    clipping = Counter()
    for row in windows:
        key = row['case_id'], row['session_id']
        shift = int(row['requested_offset_min'])
        audit.context = f'{key}/{shift}'
        audit.require((key, shift) not in covered, 'duplicate session-window row')
        covered.add((key, shift))
        source = original[key]
        start, end, energy, rated = (float(source[k]) for k in ('start', 'end', 'energy', 'rated'))
        power = min(22, rated)
        last = min(week_ends[key[0]], next_start.get(key, float('inf')))
        candidate = end + shift * 60
        minimum_end = start + energy / power * 3600
        target = min(last, max(minimum_end, candidate))
        actual = float(row['adjusted_end_utc_seconds'])
        audit.require(row['connector_id'] == source['connector_id'], 'connector lineage')
        audit.close([row[k] for k in ('start_utc_seconds', 'original_end_utc_seconds', 'energy_kwh', 'power_kw')],
                    [start, end, energy, power], 'session lineage', 1e-8)
        audit.close(actual, target, 'endpoint adjustment', 3e-6)
        audit.close(row['actual_offset_min'], (actual - end) / 60, 'logged actual shift')
        audit.le(energy, (actual - start) * power / 3600, 'energy feasible after perturbation')
        audit.le(actual, last, 'connector and study-week boundary')
        label = ('energy' if shift < 0 and target > candidate + 1e-6 else
                 ('connector' if last < week_ends[key[0]] else 'week_end') if shift > 0 and target < candidate - 1e-6 else 'none')
        audit.require(row['clipping_reason'] == label, 'clipping flag')
        clipping[int(row['year']), shift, label] += 1
    audit.require(covered == {(key, shift) for key in original for shift in offsets}, 'exact session retention')
    for key, group in summary['summary'].items():
        year, shift = map(int, key.split('|'))
        for label, count in group['clipped_sessions_by_reason'].items():
            audit.require(clipping[year, shift, label] == count, 'reported clipping counts')
    service = read(directory / 'outcomes.csv')
    index = {(r['case_id'], int(r['requested_offset_min']), r['objective'], float(r['battery_fraction'])): r
             for r in service}
    audit.require(len(index) == len(service) == summary['outcomes'] == 140 * 7 * 13, 'objective row count')
    for key, row in index.items():
        audit.context = str(key)
        status = row['status']
        audit.require(status == 'optimal' or (status == 'infeasible' and row['objective'] == 'strict_original'),
                      'solver status')
        if status == 'optimal':
            audit.le(float(row['cost_gbp']), float(row['budget_gbp']), 'cost ceiling', 2e-6)
            if row['carbon_cap_kg']:
                audit.le(float(row['carbon_kg']), float(row['carbon_cap_kg']), 'carbon ceiling', 2e-6)
        if row['objective'] == 'fixed_service':
            audit.le(float(row['avoidable_shortfall_kwh']), 0, 'fixed service residual', 5e-6)
    for case in cohort:
        for f in (0.0, 0.5):
            full = [float(index[case, s, 'full_carbon', f]['carbon_kg']) for s in offsets]
            audit.le(full[1:], full[:-1], 'nested window full-carbon optimum', 3e-6)
        for shift in offsets:
            audit.context = f'{case}/{shift}'
            for objective, metric in (('fixed_service', 'carbon_kg'), ('strict_recalculated', 'avoidable_shortfall_kwh'),
                                      ('strict_original', 'avoidable_shortfall_kwh')):
                values = [index[case, shift, objective, f][metric] for f in (0.0, .125, .5)]
                for low, high in zip(values, values[1:]):
                    if low:
                        audit.require(bool(high), 'larger storage broke feasibility')
                        audit.le(float(high), float(low), 'larger-storage optimum', 3e-6)
    baseline = json_file(ROOT / 'results' / 'service_summary.json')
    source_interaction = read(HOLDOUT / 'analysis_common_pv' / 'matched_interaction_changes.csv')
    primary = [r for r in source_interaction if float(r['pv_scale']) == .2 and float(r['battery_fraction']) == .5]
    audit.require(len(primary) == 70, 'original interaction cohort count')
    for year in (2023, 2024):
        audit.context = f'{year}/0'
        entry = summary['summary'][f'{year}|0']
        audit.close(entry['primary_interaction_fall_g_per_kwh']['median'],
                    median(-float(r[f'interaction_g_per_kwh_{year}']) for r in primary), 'zero-offset interaction replication')
        for f in (.125, .5):
            old = baseline['summaries'][f'{year}|main|30|0.2|{f}']
            audit.close(entry['fixed_service_battery_gain_g_per_kwh'][str(f)]['median'],
                        old['readiness_floor_carbon_gain_g_per_kwh']['median'], 'zero-offset fixed-service replication')
        for f in (0., .125, .5):
            old = baseline['frontier'][f'{year}|30|0.2|1.0|{f}']
            audit.close(entry['strict_targets']['strict_recalculated'][str(f)]['pooled_shortfall_pct_among_feasible_weeks'],
                        old['pooled_avoidable_shortfall_pct'], 'zero-offset strict-cap replication')
    output = {'pass': True, 'site_weeks': len(cohort), 'unique_sessions': len(original), 'windows': len(windows),
              'outcomes': len(service), 'max_absolute_residuals': dict(audit.maxima),
              'scope': 'Original session lineage, exact perturbation and occupancy caps, objective output bounds, nesting, and reproduction of the archived zero-offset results. Individual schedule vectors are checked by the solver at run time and are not archived.'}
    (directory / 'audit_summary.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(windows)} session windows and {len(service)} objective rows')


if __name__ == '__main__':
    main()

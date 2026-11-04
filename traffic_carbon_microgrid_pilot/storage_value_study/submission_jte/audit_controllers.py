"""Recompute controller outcomes from archived actions and frozen input intervals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_service import (
    Audit, HOLDOUT, ROOT, STEP, audit_matched_panel, json_file, load_frozen_inputs,
    np, read, sha, stamp, truth,
)


PHYSICAL_TOL = 5e-6
ORACLE_TOL = 1e-4
SIGNALS = ('carbon_g_per_kwh', 'price_gbp_per_mwh', 'pv_kw_per_kwp')
TRACE_FIELDS = ('energy', 'early', 'soc', 'pv', 'grid', 'charge', 'discharge', 'decisions')


def overlap(left, right, ticks):
    return np.maximum(0, np.minimum(right[:, None], ticks + STEP) - np.maximum(left[:, None], ticks)) / 3600


def case_arrays(records, first, last, raw_signals, cap, minutes):
    ticks = np.arange(first, last, STEP)
    factors = np.array([[float(raw_signals[t][column]) for t in ticks] for column in SIGNALS])
    begin, end, demand, rated = np.array([[float(s[k]) for s in records]
                                         for k in ('start', 'end', 'energy', 'rated')])
    power = np.minimum(cap, rated)
    cutoff = np.maximum(begin, end - minutes * 60)
    asap_end = begin + demand * 3600 / power
    early_max = power[:, None] * overlap(begin, cutoff, ticks)
    late_max = power[:, None] * overlap(cutoff, end, ticks)
    asap = power[:, None] * overlap(begin, asap_end, ticks)
    asap_early = power[:, None] * overlap(begin, np.minimum(asap_end, cutoff), ticks)
    return ticks, factors, begin, end, demand, early_max, late_max, asap, asap_early


def check_trace(audit, trace, row, arrays, daily, plan, protocol, anchor):
    ticks, (carbon, price, irradiance), begin, end, demand, early_max, late_max, asap, asap_early = arrays
    energy, early, soc, pv, grid, charge, discharge, decisions = [trace[k] for k in TRACE_FIELDS]
    policy, fraction = row['policy'], float(row['battery_fraction'])
    n = len(ticks)
    stride = plan['decision_interval_slots']
    steps = np.arange(0, n, stride)
    audit.require(energy.shape == early.shape == asap.shape and soc.shape == (n + 1,) and
                  all(a.shape == (n,) for a in (pv, grid, charge, discharge)) and decisions.shape == (len(steps), 4),
                  'trace dimensions')
    audit.require(all(np.isfinite(a).all() for a in trace.values()), 'non-finite trace')
    audit.close(decisions[:, 0], steps, 'decision cadence', 0)
    elapsed = np.column_stack((np.zeros(len(demand)), np.cumsum(energy, axis=1)))[:, steps]
    decision_times = ticks[steps]
    active = (begin[:, None] <= decision_times) & (decision_times < end[:, None]) & (demand[:, None] - elapsed > 5e-6)
    audit.close(decisions[:, 1], active.sum(axis=0), 'arrived unfinished session count', 0)
    first_decision = ticks[0] + stride * STEP * np.ceil((begin - ticks[0]) / (stride * STEP))
    fallback = ticks < first_decision[:, None]
    if policy == 'immediate':
        fallback = np.ones_like(fallback)
    audit.close(energy[fallback], asap[fallback], 'immediate and between-decision delivery', PHYSICAL_TOL)
    audit.close(early[fallback], asap_early[fallback], 'immediate and between-decision early delivery', PHYSICAL_TOL)
    audit.le(0, early, 'early nonnegativity')
    audit.le(early, early_max, 'early overlap bound', PHYSICAL_TOL)
    audit.le(0, energy - early, 'late nonnegativity')
    audit.le(energy - early, late_max, 'late overlap bound', PHYSICAL_TOL)
    delivered_error = np.max(np.abs(energy.sum(axis=1) - demand))
    readiness_error = np.max(np.maximum(0, asap_early.sum(axis=1) - early.sum(axis=1)))
    audit.error(delivered_error, 'recorded energy conservation', PHYSICAL_TOL)
    audit.error(readiness_error, 'ASAP checkpoint floor', PHYSICAL_TOL)
    audit.le(pv, irradiance * daily * plan['pv_scale'] * STEP / 3600, 'available PV bound')
    audit.le(0, np.array([pv, grid, charge, discharge]), 'site flow nonnegativity')
    audit.close(grid + pv + discharge, energy.sum(axis=0) + charge, 'site bus energy balance')
    capacity = daily * fraction
    target = capacity * protocol['initial_soc_fraction']
    limit = capacity / protocol['battery_hours'] * STEP / 3600
    eta = np.sqrt(protocol['battery_roundtrip_efficiency'])
    soc_bound_error = max(0., -min(soc), max(soc) - capacity)
    simultaneous = max(np.minimum(charge, discharge))
    audit.close(soc[[0, -1]], target, 'initial and terminal SOC', PHYSICAL_TOL)
    audit.error(soc_bound_error, 'SOC bounds', PHYSICAL_TOL)
    audit.close(np.diff(soc), eta * charge - discharge / eta, 'SOC energy recursion')
    audit.le(np.array([charge, discharge]), limit, 'battery power bound')
    audit.error(simultaneous, 'battery mode exclusivity', PHYSICAL_TOL)
    cost = float(grid @ price / 1000 + protocol['battery_wear_gbp_per_kwh_throughput'] * (charge + discharge).sum())
    footprint = float(grid @ carbon / 1000)
    budget = float(asap.sum(axis=0) @ price / 1000)
    within = cost <= budget + 1e-5
    audit.require(within == truth(row['within_week_budget']), 'budget compliance flag')
    forecast_excess = max(0., max(decisions[:, 3] - decisions[:, 2]))
    audit.error(forecast_excess, 'logged forecast budget residual', PHYSICAL_TOL)
    for column, expected in {
        'carbon_kg': footprint, 'carbon_g_per_kwh': 1000 * footprint / demand.sum(), 'cost_gbp': cost,
        'budget_gbp': budget, 'budget_excess_gbp': max(0., cost - budget), 'energy_kwh': demand.sum(),
        'n_sessions': len(demand), 'max_session_delivery_error_kwh': delivered_error,
        'max_avoidable_checkpoint_error_kwh': readiness_error, 'terminal_soc_error_kwh': abs(soc[-1] - soc[0]),
        'max_soc_bound_error_kwh': soc_bound_error, 'max_simultaneous_kwh': simultaneous,
        'max_forecast_budget_excess_gbp': forecast_excess,
    }.items():
        audit.close(row[column], expected, 'recomputed ' + column)
    shared = float(row['oracle_shared_budget_carbon_kg'])
    equal = float(row['oracle_equal_realized_cost_carbon_kg'])
    audit.close(anchor['budget_gbp'], budget, 'shared oracle budget lineage')
    audit.close(shared, anchor['carbon_kg'], 'shared oracle carbon lineage')
    shared_lower = float(anchor['objective_lower_bound'])
    audit.le(shared_lower, shared, 'shared oracle lower bound versus incumbent')
    audit.le(equal, footprint, 'equal-cost oracle incumbent versus online', ORACLE_TOL)
    if within:
        audit.le(shared_lower, footprint, 'shared-budget lower bound versus complying online', ORACLE_TOL)
        audit.le(shared, footprint, 'shared-budget oracle incumbent versus complying online',
                 ORACLE_TOL + shared - shared_lower)
        audit.le(shared_lower, equal, 'shared-budget lower bound versus tighter equal-cost oracle', ORACLE_TOL)
    else:
        audit.le(equal, shared, 'looser equal-cost oracle incumbent versus shared-budget oracle', ORACLE_TOL)
    return within


def independent_information_test(audit, selected, plan, protocol, diagnostic):
    # The numerical audit never calls the control implementation. Here only,
    # replay it under deliberately incompatible future data to test causality.
    from run_controllers import Session, Signals, replay, settings_for

    case, records, arrays = selected
    ticks, factors, begin, _, _, *_ = arrays
    actual = Signals(ticks, *factors)
    forecast = actual if diagnostic else training_forecast(case, plan, ticks, Signals)
    sessions = [Session(s['session_id'], s['station_id'], s['connector_id'], *map(float, (s['start'], s['end'], s['energy'], s['rated'])))
                for s in records]
    stride = plan['decision_interval_slots']
    boundary_slot = min(len(ticks) - stride, (int(np.ceil((begin.min() - ticks[0]) / (STEP * stride))) + 3) * stride)
    boundary = ticks[boundary_slot]
    current = begin < boundary
    audit.require(current.any() and not current.all(), 'information test needs arrivals on both sides of the boundary')
    changed = [s if keep else replace(s, energy=s.energy / 3) for s, keep in zip(sessions, current, strict=True)]
    rewritten = actual
    changed_inputs = ['future arrivals energy']
    if not diagnostic:
        pv = actual.pv.copy()
        pv[boundary_slot:] = 5 * pv[boundary_slot:] + 1
        rewritten = Signals(ticks, -actual.carbon - 123, -actual.price + 999, pv)
        changed_inputs.extend(['future physical PV', 'all realized prices', 'all realized carbon'])
    typed = {**case, 'site_daily_energy_kwh': float(case['site_daily_energy_kwh']),
             'power_cap_kw': float(case['power_cap_kw'])}
    # Replay does not need or receive any of the realized weekly quantities.
    for name in ('budget_gbp', 'reference_cost_gbp', 'reference_carbon_kg', 'energy_kwh', 'n_sessions'):
        typed.pop(name, None)
    comparisons = 0
    for fraction in plan['battery_fractions']:
        settings = settings_for(typed, protocol, plan['pv_scale'], fraction)
        for policy in plan['policies']:
            a = replay(typed, sessions, forecast, actual, settings, plan, policy, stop_at=boundary_slot)
            b = replay(typed, changed, forecast, rewritten, settings, plan, policy, stop_at=boundary_slot)
            for name in ('ev_plan', 'charge_plan', 'discharge_plan', 'grid', 'pv', 'charge', 'discharge', 'decisions'):
                audit.close(a[name], b[name], 'causal prefix under rewritten future information', 0)
            for name in ('energy', 'early'):
                audit.close(a[name][current], b[name][current], 'causal session allocation prefix', 0)
            comparisons += 1
    code_review = ('The diagnostic gives the rolling planner realized signals for its horizon. It still receives only active sessions.'
                   if diagnostic else 'Forecasting uses the frozen training profiles and the evaluation time grid. The planner receives only active sessions and forecast signals. Realized PV enters execution balance after the dispatch decisions, while realized prices and carbon enter scoring and oracle evaluation.')
    return {'case_id': case['case_id'], 'prefix_slots': boundary_slot, 'policies_by_capacity': comparisons,
            'mutated': changed_inputs, 'weekly_outcomes_withheld': True, 'code_review': code_review}


def training_forecast(case, plan, ticks, signals_type):
    train_group = f"{case['city']}_{plan['training_year']}_{int(case['month']):02}"
    profiles = defaultdict(list)

    def key(t):
        dt = datetime.fromtimestamp(t, timezone.utc)
        return dt.weekday() >= 5, dt.hour, dt.minute

    for row in read(HOLDOUT / 'data_harmonized_2023' / train_group / 'signals.csv'):
        profiles[key(stamp(row['timestamp_utc']))].append([float(row[col]) for col in SIGNALS])
    medians = {k: np.median(values, axis=0) for k, values in profiles.items()}
    return signals_type(ticks, *np.array([medians[key(t)] for t in ticks]).T)


def run(directory, diagnostic):
    audit = Audit()
    plan_path = ROOT / 'controller_protocol.json'
    plan = json_file(plan_path)
    if diagnostic:
        plan['policies'] = ['carbon']
    protocol = json_file(HOLDOUT / 'protocol.json')
    manifest = json_file(directory / 'manifest.json')
    results = directory / 'controller_outcomes.csv'
    audit.require(manifest['protocol_sha256'] == sha(plan_path), 'controller protocol hash')
    audit.require(manifest['outcomes_sha256'] == sha(results), 'controller outcome hash')
    expected_test = 'not_applicable_future_signals_supplied' if diagnostic else 'pass'
    expected_signals = 'realized_diagnostic' if diagnostic else '2023_frozen_medians'
    audit.require(manifest['information_prefix_test'] == expected_test and manifest['planner_signals'] == expected_signals,
                  'runner signal design and information test')
    training_hashes = {} if diagnostic else {
        path.parent.name: sha(path) for path in (HOLDOUT / 'data_harmonized_2023').glob('*/signals.csv')
    }
    audit.require(manifest['training_signal_sha256'] == training_hashes, 'frozen training signal hashes')
    source_manifest = json_file(ROOT / 'results' / 'manifest.json')
    source = ROOT / 'results' / 'service_outcomes.csv'
    audit.require(source_manifest['outcome_sha256'] == sha(source), 'source oracle outcome hash')
    cases, sessions, _, inputs = load_frozen_inputs(source_manifest, protocol, audit)
    panel_count = audit_matched_panel(cases, audit)
    expected = {r['case_id_2024'] for r in read(HOLDOUT / 'analysis_common_pv' / 'week_ordinal_matches.csv')
                if r['match_status'] == 'matched'}
    rows = read(results)
    by_case = defaultdict(list)
    for r in rows:
        by_case[r['case_id']].append(r)
    audit.require(len(by_case) == len(expected) == panel_count == manifest['cases'] == 70 and set(by_case) == expected,
                  'full matched evaluation panel coverage')
    expected_count = 140 if diagnostic else 420
    audit.require(len(rows) == manifest['outcomes'] == expected_count, 'full evaluation outcome count')
    audit.require({p.stem for p in (directory / 'traces').glob('*.npz')} == {c.replace('|', '__') for c in expected},
                  'missing or surplus trace files')
    anchor = {(r['case_id'], float(r['battery_fraction'])): r for r in read(source)
              if r['case_id'] in expected and r['design'] == 'main' and r['objective'] == 'preserve_readiness' and
              int(r['checkpoint_min']) == plan['early_checkpoint_min'] and float(r['pv_scale']) == plan['pv_scale']}
    combos = {(policy, fraction) for policy in plan['policies'] for fraction in plan['battery_fractions']}
    summary = defaultdict(Counter)
    allocation_count = 0
    info_sample = None
    for case_id, policies in by_case.items():
        case, records = cases[case_id], sessions[case_id]
        cap = float(case['power_cap_kw'])
        arrays = case_arrays(records, stamp(case['start_utc']), stamp(case['end_utc_exclusive']),
                             inputs[case['group']][1], cap, plan['early_checkpoint_min'])
        if info_sample is None and len(records) > 5:
            info_sample = case, records, arrays
        audit.context = case_id
        audit.require(len(policies) == len(combos) and {(r['policy'], float(r['battery_fraction'])) for r in policies} == combos,
                      'complete per-case policies with distinct capacities')
        file = directory / 'traces' / (case_id.replace('|', '__') + '.npz')
        with np.load(file, allow_pickle=False) as raw:
            audit.require(set(raw.files) == {f'{p}_{f}_{k}' for p, f in combos for k in TRACE_FIELDS}, 'trace keys')
            for row in policies:
                policy, fraction = row['policy'], float(row['battery_fraction'])
                audit.context = f'{case_id}/{policy}/{fraction}'
                audit.require(all(row[k] == case[k] for k in ('city', 'site_id', 'month', 'week_rank', 'year')), 'case metadata')
                trace = {k: raw[f'{policy}_{fraction}_{k}'] for k in TRACE_FIELDS}
                audit.require(truth(anchor[case_id, fraction]['optimal']), 'shared oracle solver status')
                within = check_trace(audit, trace, row, arrays, float(case['site_daily_energy_kwh']),
                                     plan, protocol, anchor[case_id, fraction])
                summary[f'{policy}/{fraction}']['cases'] += 1
                summary[f'{policy}/{fraction}']['within_shared_budget'] += within
                allocation_count += len(records)
    audit.context = 'independent information test'
    information = independent_information_test(audit, info_sample, plan, protocol, diagnostic)
    return {'pass': True, 'planner_signals': expected_signals, 'weeks': len(by_case), 'outcomes': len(rows),
            'session_policy_allocations': allocation_count,
            'within_budget_by_policy_capacity': summary, 'largest_residuals': dict(sorted(audit.maxima.items())),
            'independent_information_test': information,
            'oracle_scope': 'Verified archived shared-budget certified bounds and both oracle objective orderings. Equal-cost oracle vectors and dual certificates are not archived.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--diagnostic', action='store_true', help='Audit the perfect-signal diagnostic with hidden arrivals')
    args = parser.parse_args()
    directory = args.directory or ROOT / 'results' / ('signal_diagnostic' if args.diagnostic else 'controllers')
    summary = run(directory, args.diagnostic)
    output = directory / 'audit_controller_summary.json'
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"PASS: {summary['outcomes']} controller policies across {summary['weeks']} site-weeks; {output}")


if __name__ == '__main__':
    main()

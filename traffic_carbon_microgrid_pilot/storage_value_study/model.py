"""One-site, one-window storage value study using the pilot's physical model.

The objective is attributed CO2 from grid imports under a common operating
cost cap. Sessions retain their complete requested energy. Future signals and
proxy connection windows are scenario inputs for an ex-post assessment.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))

from optimize import Model, STEP, Session, Settings, Signals, early_reference

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, vstack

FEAS_TOL = 1e-6
CARBON_TIE_TOL = 1e-7
STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}


def _validate(sessions, sig, power_cap_kw):
    if not sessions or len({s.station_id for s in sessions}) != 1:
        raise ValueError("Supply nonempty sessions from exactly one station")
    if not math.isfinite(power_cap_kw) or power_cap_kw <= 0:
        raise ValueError("power_cap_kw must be positive")
    n = len(sig.ts)
    if not n or any(len(a) != n or not np.isfinite(a).all() for a in (sig.ts, sig.carbon, sig.price, sig.pv)):
        raise ValueError("Signals must be nonempty, aligned and finite")
    if not np.allclose(np.diff(sig.ts), STEP, rtol=0, atol=1e-6):
        raise ValueError("Expected contiguous 30-minute signals")
    if np.min(sig.carbon) < 0 or np.min(sig.pv) < 0:
        raise ValueError("Carbon intensity and PV profile must be nonnegative")
    seen = set()
    last_end = {}
    for s in sorted(sessions, key=lambda row: row.start):
        if s.session_id in seen:
            raise ValueError(f"Duplicate session ID {s.session_id}")
        seen.add(s.session_id)
        if not all(math.isfinite(v) for v in (s.start, s.end, s.energy, s.rated)) or min(s.energy, s.rated) <= 0:
            raise ValueError(f"Invalid session data for {s.session_id}")
        if s.start < sig.ts[0] - FEAS_TOL or s.end > sig.ts[-1] + STEP + FEAS_TOL:
            raise ValueError(f"Full session {s.session_id} must lie inside the horizon")
        if s.end <= s.start or s.energy > min(s.rated, power_cap_kw) * (s.end - s.start) / 3600 + FEAS_TOL:
            raise ValueError(f"Session {s.session_id} cannot deliver full energy under its power cap")
        if s.start < last_end.get(s.connector_id, -math.inf) - FEAS_TOL:
            raise ValueError(f"Overlapping connector records at session {s.session_id}")
        last_end[s.connector_id] = s.end


def _scaled_sessions(sessions, power_cap_kw, alpha):
    """Scale only time beyond minimum feasible charging, not the energy."""
    scaled = []
    for s in sessions:
        earliest_completion = s.start + 3600 * s.energy / min(s.rated, power_cap_kw)
        slack = max(0.0, s.end - earliest_completion)
        scaled.append(replace(s, end=min(s.end, earliest_completion + alpha * slack)))
    return scaled


def reference_cost(sessions, sig, power_cap_kw, settings):
    """Cost of immediate capped-power charging with specified PV and no BESS.

    PV serves simultaneous charging when grid prices are nonnegative. At a
    negative price the cost reference chooses grid imports and may curtail PV.
    Pass settings with pv_kwp=0 to anchor every PV scenario to the same budget.
    """
    _validate(sessions, sig, power_cap_kw)
    if not math.isfinite(settings.pv_kwp) or settings.pv_kwp < 0:
        raise ValueError("PV kWp must be nonnegative and finite")
    return _reference_cost(sessions, sig, power_cap_kw, settings)


def _reference_cost(sessions, sig, power_cap_kw, settings):
    load, _ = early_reference(sessions, sig, power_cap_kw, [sessions[0].station_id])
    pv_used = np.where(sig.price < 0, 0.0, np.minimum(load[0], settings.pv_kwp * sig.pv * 0.5))
    return float((load[0] - pv_used) @ (sig.price / 1000.0))


def _constraints(model, budget_gbp):
    rows = [model.aub]
    rhs = [np.asarray(model.ub_rhs)]
    if budget_gbp is not None:
        rows.append(csr_matrix(model.cost.reshape(1, -1)))
        rhs.append(np.array([budget_gbp]))
    return vstack(rows, format="csc"), np.concatenate(rhs)


def _physical_solution(model, x, budget_gbp):
    """A noncycling relaxed point can be completed to a MILP point."""
    if x is None:
        return None, {}
    x = np.asarray(x).copy()
    if model.has_binary:
        # An idle slot can use either mode. Keep the larger flow's mode when
        # the smaller flow is only a numerical residual.
        x[model.binary] = (x[model.charge] > x[model.discharge]).astype(float)
    check = model.check(x, None)
    check.pop("max_carbon_budget_error_kg", None)
    check["max_cost_budget_error_gbp"] = (max(0.0, float(model.cost @ x) - budget_gbp)
                                          if budget_gbp is not None else 0.0)
    return (x if max(check.values()) <= FEAS_TOL else None), check


def _solve(model, objective, budget_gbp, *, extra=None):
    """Try LP, then limited integrality, then full mutual exclusion if needed.

    Every relaxation provides a lower bound for the exact problem. A relaxed
    optimum passing the full physical test (including mode assignment and
    budget) attains that bound up to numerical tolerance. A partial-MILP
    bound remains valid even if cycling survives in its relaxed slots.
    """
    start = time.perf_counter()
    aub, bub = _constraints(model, budget_gbp)
    is_carbon = isinstance(objective, str) and objective == "carbon"
    c = (model.carbon if is_carbon else model.cost) if isinstance(objective, str) else objective
    if extra is not None:
        a_extra, b_extra = extra
        aub = vstack([aub, a_extra], format="csc")
        bub = np.concatenate([bub, b_extra])
    limit = model.cfg.time_limit
    lower = np.array(model.lower)
    upper = np.array(model.upper)
    counters = {"n_lp_solves": 0, "n_mip_solves": 0}
    bound = None
    trial = None

    def remaining():
        return max(0.001, limit - (time.perf_counter() - start))

    def lp(coefs, a, b):
        counters["n_lp_solves"] += 1
        return linprog(coefs, A_ub=a if len(b) else None, b_ub=b if len(b) else None,
                       A_eq=model.aeq, b_eq=model.eq_rhs, bounds=np.column_stack((lower, upper)),
                       method="highs", options={"time_limit": remaining()})

    def physical(x):
        valid, check = _physical_solution(model, x, budget_gbp)
        if valid is not None and extra is not None:
            err = float(np.max(np.maximum(a_extra @ valid - b_extra, 0)))
            check["max_extra_constraint_error"] = err
            if err > FEAS_TOL:
                valid = None
        return valid, check

    def result(res, solver, valid_x, check, *, status=None):
        state = status or STATUS.get(res.status, "solver_error")
        objective_value = float(c @ valid_x) if valid_x is not None else None
        absolute_gap = max(0.0, objective_value - bound) if objective_value is not None and bound is not None else None
        info = {"status": state, "optimal": state == "optimal", "feasible": valid_x is not None,
                "solver": solver, "solver_message": str(res.message), "solve_seconds": time.perf_counter() - start,
                "objective_value": objective_value, "objective_lower_bound": bound, "objective_abs_gap": absolute_gap,
                "mip_gap": float(res.mip_gap) if getattr(res, "mip_gap", None) is not None and math.isfinite(res.mip_gap) else None,
                **counters, **check}
        return info, valid_x

    trial = lp(c, aub, bub)
    mode = "HiGHS_LP_relaxation" if model.has_binary else "HiGHS_LP"
    valid, check = physical(trial.x)
    if trial.status == 0:
        bound = float(trial.fun)
    if valid is not None or trial.x is None:
        return result(trial, mode, valid, check)
    if not model.has_binary:
        return result(trial, mode, None, check, status="failed_postsolve_check")

    if trial.status == 0 and (is_carbon or extra is not None) and time.perf_counter() - start < limit:
        # Some LP optima dissipate surplus through simultaneous flows only
        # because carbon minimization is indifferent to that throughput.
        refined = lp(model.throughput, vstack([aub, csr_matrix(c.reshape(1, -1))], format="csc"),
                     np.append(bub, bound + CARBON_TIE_TOL))
        valid, check = physical(refined.x)
        if valid is not None and float(c @ valid) <= bound + CARBON_TIE_TOL + FEAS_TOL:
            return result(refined, "HiGHS_LP_throughput_refinement", valid, check, status="optimal")

    base_constraints = [LinearConstraint(model.aeq, model.eq_rhs, model.eq_rhs)]
    if len(bub):
        base_constraints.append(LinearConstraint(aub, np.full(len(bub), -math.inf), bub))
    simultaneous = np.minimum(trial.x[model.charge], trial.x[model.discharge]).ravel() > FEAS_TOL
    partial = (model.sig.price < 0) | simultaneous
    masks = [partial, np.ones(model.nt, dtype=bool)] if not partial.all() else [partial]
    for mask in masks:
        if time.perf_counter() - start >= limit:
            return result(trial, mode, None, check, status="limit_reached")
        integ = np.zeros(model.nv, dtype=np.int8)
        integ[model.binary.ravel()[mask]] = 1
        mode = "HiGHS_full_MILP" if mask.all() else "HiGHS_partial_MILP"
        counters["n_mip_solves"] += 1
        trial = milp(c, integrality=integ, bounds=Bounds(lower, upper), constraints=base_constraints,
                     options={"time_limit": remaining(), "mip_rel_gap": model.cfg.mip_rel_gap})
        dual_bound = getattr(trial, "mip_dual_bound", None)
        if dual_bound is not None and math.isfinite(dual_bound):
            bound = max(bound, float(dual_bound)) if bound is not None else float(dual_bound)
        valid, check = physical(trial.x)
        if valid is not None or trial.status in (2, 3, 4):
            return result(trial, mode, valid, check)
    state = "failed_postsolve_check" if trial.status == 0 else STATUS.get(trial.status, "solver_error")
    return result(trial, mode, None, check, status=state)


def run_case(sessions, sig, power_cap_kw, settings, *, alphas=(0.0, 0.25, 0.5, 1.0), budget_gbp=None):
    """Return one minimum-carbon result per availability fraction × storage.

    Use the same explicit budget for every PV and flexibility comparison that
    belongs to a site-week. Default budget is this PV case's no-storage fixed
    charging reference. All costs include the configured BESS throughput wear.
    """
    sessions = list(sessions)
    _validate(sessions, sig, power_cap_kw)
    numeric = (settings.pv_kwp, settings.battery_kw, settings.battery_kwh, settings.battery_wear,
               settings.time_limit, settings.mip_rel_gap, settings.roundtrip_efficiency, settings.initial_soc_fraction)
    if not all(math.isfinite(v) and v >= 0 for v in numeric):
        raise ValueError("Settings must be finite and nonnegative")
    if not 0 < settings.roundtrip_efficiency <= 1 or not 0 <= settings.initial_soc_fraction <= 1 or settings.time_limit <= 0:
        raise ValueError("Invalid efficiency, initial SOC or time limit")
    fixed_cost = _reference_cost(sessions, sig, power_cap_kw, settings)
    budget = fixed_cost if budget_gbp is None else float(budget_gbp)
    if not math.isfinite(budget):
        raise ValueError("Cost budget must be finite")
    required = sum(s.energy for s in sessions)
    full_slack = sum(max(0.0, (s.end - s.start) / 3600 - s.energy / min(s.rated, power_cap_kw)) for s in sessions)
    out = []
    for alpha in alphas:
        alpha = float(alpha)
        if not 0 <= alpha <= 1:
            raise ValueError("Availability fraction must lie in [0, 1]")
        # Fixed reference is immediate charging, independent of unused time.
        cohort = _scaled_sessions(sessions, power_cap_kw, alpha) if alpha else sessions
        for battery in (False, True):
            config = ("smart" if alpha else "fixed") + "_pv" + ("_battery" if battery else "")
            model = Model(cohort, sig, power_cap_kw, config, settings, force_mip=battery)
            info, x = _solve(model, "carbon", budget)
            row = {"site_id": sessions[0].station_id, "station_id": sessions[0].station_id,
                   "alpha": alpha, "has_battery": battery, "configuration": config, "power_cap_kw": power_cap_kw,
                   "pv_kwp": settings.pv_kwp, "battery_kwh": settings.battery_kwh if battery else 0.0,
                   "battery_kw": settings.battery_kw if battery else 0.0, "n_sessions": len(sessions),
                   "energy_required_kwh": required, "proxy_slack_hours": full_slack,
                   "allowed_slack_hours": alpha * full_slack, "operating_budget_gbp": budget,
                   "fixed_pv_reference_cost_gbp": fixed_cost, **info}
            if x is None and info["status"] == "infeasible":
                diagnostic, cost_x = _solve(model, "cost", None)
                row.update({"minimum_cost_status": diagnostic["status"],
                            "minimum_cost_gbp": float(model.cost @ cost_x) if cost_x is not None else None,
                            "minimum_cost_lower_bound_gbp": diagnostic["objective_lower_bound"]})
            if x is not None:
                grid = x[model.grid].ravel()
                pv_use = float(x[model.pv_use].sum())
                charge = float(x[model.charge].sum()) if battery else 0.0
                discharge = float(x[model.discharge].sum()) if battery else 0.0
                actual_cost = float(model.cost @ x)
                row.update({"carbon_kg": float(model.carbon @ x), "operating_cost_gbp": actual_cost,
                            "cost_gbp": actual_cost, "budget_slack_gbp": budget - actual_cost,
                            "grid_energy_cost_gbp": float(grid @ (sig.price / 1000.0)),
                            "battery_wear_gbp": (charge + discharge) * settings.battery_wear,
                            "delivered_kwh": float(sum(x[c] for parts in model.session_columns.values() for c, _, _ in parts)) if model.smart else float(model.fixed_load.sum()),
                            "grid_import_kwh": float(grid.sum()), "pv_available_kwh": float(model.pv_available.sum()),
                            "pv_used_kwh": pv_use, "pv_curtailed_kwh": float(model.pv_available.sum()) - pv_use,
                            "battery_charge_kwh": charge, "battery_discharge_kwh": discharge,
                            "initial_soc_kwh": float(x[model.soc][0, 0]) if battery else 0.0,
                            "final_soc_kwh": float(x[model.soc][0, -1]) if battery else 0.0})
            out.append(row)
    return out


def self_test():
    ts = np.array([0.0, STEP])
    settings = Settings(pv_kwp=0, initial_soc_fraction=0, time_limit=20)
    sig = Signals(ts, np.array([500.0, 50.0]), np.array([100.0, 100.0]), np.zeros(2))
    sessions = [Session("s", "site", "plug", 0.0, 2 * STEP, 1.0, 2.0)]
    budget = reference_cost(sessions, sig, 2.0, settings)
    rows = run_case(sessions, sig, 2.0, settings, alphas=(0, 0.5, 1), budget_gbp=budget)
    assert all(row["optimal"] and abs(row["delivered_kwh"] - 1) < FEAS_TOL for row in rows)
    no_bess = [row for row in rows if not row["has_battery"]]
    assert np.allclose([row["carbon_kg"] for row in no_bess], [0.5, 0.275, 0.05])
    assert all(row["operating_budget_gbp"] == budget and row["budget_slack_gbp"] >= -FEAS_TOL for row in rows)
    assert all(row["max_simultaneous_kwh"] < FEAS_TOL and abs(row["final_soc_kwh"] - row["initial_soc_kwh"]) < FEAS_TOL for row in rows)

    # At a negative-price slot with no connected EV, a loss loop could
    # manufacture revenue while returning to zero SOC. The MILP forbids it.
    sig2 = Signals(ts, np.array([100.0, 1.0]), np.array([100.0, -500.0]), np.zeros(2))
    fixed = [replace(sessions[0], end=STEP)]
    rows = run_case(fixed, sig2, 2.0, settings, alphas=(0,), budget_gbp=0.05)
    assert all(row["status"] == "infeasible" and abs(row["minimum_cost_gbp"] - 0.1) < FEAS_TOL for row in rows)
    assert rows[-1]["n_mip_solves"] > 0, "Negative-price loss loop must trigger integer enforcement"
    print("PASS: same service and shared cost cap, scaled-window carbon response, cyclic SOC, negative-price loss-loop rejection and minimum-cost diagnostic")


if __name__ == "__main__":
    self_test()

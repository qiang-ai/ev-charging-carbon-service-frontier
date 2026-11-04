"""Fixed-service, ex-post charging / PV / battery pilot.

This is an attributional purchased-electricity carbon-footprint experiment,
not an estimate of causally avoided power-system emissions. Connection end
times are proxies, and smart schedules are an ideal-information upper bound.
The reference charges immediately at a stated constant power cap; it is NOT
a measured uncontrolled load profile. No grid export or vehicle discharge.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / ".deps").is_dir():
    sys.path.insert(0, str(HERE / ".deps"))

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

STEP = 1800.0
TOL = 1e-6


def stamp(value):
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).timestamp()


def iso(value):
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class Session:
    session_id: str
    station_id: str
    connector_id: str
    start: float
    end: float
    energy: float
    rated: float


@dataclass
class Signals:
    ts: np.ndarray
    carbon: np.ndarray
    price: np.ndarray
    pv: np.ndarray


@dataclass
class Settings:
    pv_kwp: float = 10.0
    battery_kwh: float = 20.0
    battery_kw: float = 10.0
    roundtrip_efficiency: float = 0.9
    initial_soc_fraction: float = 0.5
    battery_wear: float = 0.01  # GBP per kWh charge PLUS discharge throughput.
    time_limit: float = 120.0
    mip_rel_gap: float = 0.001


def read_inputs(directory, session_path=None):
    with (directory / "signals.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: stamp(r["timestamp_utc"]))
    sig = Signals(
        np.array([stamp(r["timestamp_utc"]) for r in rows]),
        np.array([float(r["carbon_g_per_kwh"]) for r in rows]),
        np.array([float(r["price_gbp_per_mwh"]) for r in rows]),
        np.array([float(r["pv_kw_per_kwp"]) for r in rows]),
    )
    if not len(rows) or not np.allclose(np.diff(sig.ts), STEP, atol=0.01, rtol=0):
        raise ValueError("signals.csv must have unique, contiguous 30-minute UTC rows")
    for a in (sig.carbon, sig.price, sig.pv):
        if not np.isfinite(a).all():
            raise ValueError("Signal contains NaN/inf")
    if np.min(sig.carbon) < 0 or np.min(sig.pv) < 0:
        raise ValueError("Carbon intensity and PV profile must be nonnegative")
    sessions, rejected, seen = [], [], set()
    with (session_path or directory / "sessions.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            sid = str(r["session_id"])
            try:
                s = Session(sid, str(r["station_id"]), str(r["connector_id"]),
                            stamp(r["start_utc"]), stamp(r["end_proxy_utc"]),
                            float(r["energy_kwh"]), float(r["power_kw"]))
                if sid in seen:
                    raise ValueError("duplicate_session_id")
                seen.add(sid)
                if not all(math.isfinite(v) for v in (s.start, s.end, s.energy, s.rated)):
                    raise ValueError("nonfinite_value")
                if s.end <= s.start or s.energy <= 0 or s.rated <= 0:
                    raise ValueError("invalid_duration_energy_or_power")
                sessions.append(s)
            except (ValueError, TypeError) as e:
                rejected.append({"session_id": sid, "reason": str(e)})
    return sessions, sig, rejected


def intersections(session, sig):
    """Exactly integrate available connection hours in each 30-minute slot."""
    left = max(0, int(math.floor((session.start - sig.ts[0]) / STEP)))
    right = min(len(sig.ts), int(math.ceil((session.end - sig.ts[0]) / STEP)))
    return [(j, max(0.0, min(session.end, sig.ts[j] + STEP) -
                         max(session.start, sig.ts[j])) / 3600.0)
            for j in range(left, right) if min(session.end, sig.ts[j] + STEP) >
            max(session.start, sig.ts[j])]


def select_sessions(sessions, sig, power_cap, max_stations, station_ids=None):
    eligible, excluded = [], []
    for s in sessions:
        reason = None
        if s.rated > 22.0 + TOL:
            reason = "not_AC_power_proxy_rated_above_22kW"
        elif s.start < sig.ts[0] - TOL or s.end > sig.ts[-1] + STEP + TOL:
            reason = "full_session_not_within_signal_horizon"
        elif s.energy > min(s.rated, power_cap) * (s.end - s.start) / 3600.0 + TOL:
            reason = "energy_infeasible_at_stated_power_cap"
        if reason:
            excluded.append({"session_id": s.session_id, "reason": reason})
        else:
            eligible.append(s)
    counts = {}
    for s in eligible:
        counts[s.station_id] = counts.get(s.station_id, 0) + 1
    chosen = set(station_ids) if station_ids else set(sorted(counts, key=lambda k: (-counts[k], k))[:max_stations])
    kept = []
    for s in eligible:
        if s.station_id not in chosen:
            excluded.append({"session_id": s.session_id, "reason": "station_limit"})
        else:
            kept.append(s)
    # Same-connector overlapping windows cannot be treated as independent plugs.
    # Reject later overlapping records, without changing any retained energy.
    clean, last_end = [], {}
    for s in sorted(kept, key=lambda q: (q.station_id, q.connector_id, q.start, q.session_id)):
        key = (s.station_id, s.connector_id)
        if s.start < last_end.get(key, -math.inf) - TOL:
            excluded.append({"session_id": s.session_id, "reason": "overlapping_connector_record"})
        else:
            clean.append(s)
            last_end[key] = s.end
    return clean, excluded


def early_reference(sessions, sig, power_cap, stations):
    load = np.zeros((len(stations), len(sig.ts)))
    si = {s: i for i, s in enumerate(stations)}
    allocations = []
    for s in sessions:
        remaining = s.energy
        for j, hours in intersections(s, sig):
            e = min(remaining, min(power_cap, s.rated) * hours)
            load[si[s.station_id], j] += e
            allocations.append((s.session_id, s.station_id, j, e))
            remaining -= e
        if remaining > TOL:
            raise ValueError(f"Reference cannot supply full session {s.session_id}")
    return load, allocations


class Model:
    def __init__(self, sessions, sig, power_cap, configuration, settings, force_mip=False, checkpoints=None):
        self.sessions, self.sig, self.cap = sessions, sig, power_cap
        self.configuration, self.cfg = configuration, settings
        self.checkpoints = checkpoints or {}
        self.stations = sorted({s.station_id for s in sessions})
        self.ns, self.nt = len(self.stations), len(sig.ts)
        if not self.ns:
            raise ValueError("No eligible sessions")
        self.smart = configuration.startswith("smart")
        self.has_pv = "pv" in configuration
        self.has_battery = "battery" in configuration
        self.has_binary = self.has_battery and (force_mip or np.min(sig.price) < 0)
        self.fixed_load, self.fixed_allocations = early_reference(sessions, sig, power_cap, self.stations)
        self.fixed_by_session = {}
        for sid, station, j, energy in self.fixed_allocations:
            self.fixed_by_session.setdefault(sid, []).append((j, energy))
        self.si = {s: i for i, s in enumerate(self.stations)}
        self.lower, self.upper = [], []
        self.groups = {}
        self.eq_rows, self.eq_cols, self.eq_vals, self.eq_rhs = [], [], [], []
        self.ub_rows, self.ub_cols, self.ub_vals, self.ub_rhs = [], [], [], []
        self.session_columns = {}
        self.before_checkpoint = {}
        self.nst = self.ns * self.nt
        self.grid = self.group("grid", self.nst, 0, math.inf).reshape(self.ns, self.nt)
        self.pv_available = np.tile(sig.pv * settings.pv_kwp * 0.5, (self.ns, 1)) if self.has_pv else np.zeros((self.ns, self.nt))
        self.pv_use = self.group("pv", self.nst, 0, self.pv_available.ravel()).reshape(self.ns, self.nt) if self.has_pv else None
        if self.has_battery:
            self.charge = self.group("charge", self.nst, 0, settings.battery_kw * 0.5).reshape(self.ns, self.nt)
            self.discharge = self.group("discharge", self.nst, 0, settings.battery_kw * 0.5).reshape(self.ns, self.nt)
            self.soc = self.group("soc", self.ns * (self.nt + 1), 0, settings.battery_kwh).reshape(self.ns, self.nt + 1)
            initial = settings.initial_soc_fraction * settings.battery_kwh
            for i in range(self.ns):
                for j in (0, self.nt):
                    self.lower[self.soc[i, j]] = self.upper[self.soc[i, j]] = initial
            if self.has_binary:
                self.binary = self.group("binary_charge", self.nst, 0, 1).reshape(self.ns, self.nt)
        # Session energy is never reduced; each session is one cross-day equation.
        slot_ev = [[] for _ in range(self.nst)]
        if self.smart:
            for s in sessions:
                items = []
                before = []
                checkpoint = self.checkpoints.get(s.session_id, s.start)
                power = min(s.rated, power_cap)
                for j, hours in intersections(s, sig):
                    left = max(s.start, sig.ts[j])
                    early_hours = min(hours, max(0.0, checkpoint - left) / 3600)
                    for length, is_early in ((early_hours, True), (hours - early_hours, False)):
                        if length <= 0:
                            continue
                        upper = power * length
                        col = int(self.group("ev_part", 1, 0, upper)[0])
                        items.append((col, j, upper))
                        slot_ev[self.si[s.station_id] * self.nt + j].append(col)
                        if is_early:
                            before.append(col)
                self.session_columns[s.session_id] = items
                self.before_checkpoint[s.session_id] = before
                self.eq([(c, 1) for c, _, _ in items], s.energy)
        eta = math.sqrt(settings.roundtrip_efficiency)
        for i in range(self.ns):
            for j in range(self.nt):
                terms = [(int(self.grid[i, j]), 1.0)]
                if self.has_pv:
                    terms.append((int(self.pv_use[i, j]), 1.0))
                if self.has_battery:
                    terms.extend([(int(self.discharge[i, j]), 1.0), (int(self.charge[i, j]), -1.0)])
                    self.eq([(int(self.soc[i, j + 1]), 1.0), (int(self.soc[i, j]), -1.0),
                             (int(self.charge[i, j]), -eta), (int(self.discharge[i, j]), 1 / eta)], 0.0)
                    if self.has_binary:
                        limit = settings.battery_kw * 0.5
                        self.ub([(int(self.charge[i, j]), 1), (int(self.binary[i, j]), -limit)], 0)
                        self.ub([(int(self.discharge[i, j]), 1), (int(self.binary[i, j]), limit)], limit)
                if self.smart:
                    terms.extend((c, -1.0) for c in slot_ev[i * self.nt + j])
                self.eq(terms, 0.0 if self.smart else self.fixed_load[i, j])
        self.nv = len(self.lower)
        self.aeq = coo_matrix((self.eq_vals, (self.eq_rows, self.eq_cols)), shape=(len(self.eq_rhs), self.nv)).tocsc()
        self.aub = coo_matrix((self.ub_vals, (self.ub_rows, self.ub_cols)), shape=(len(self.ub_rhs), self.nv)).tocsc()
        self.cost = np.zeros(self.nv)
        self.carbon = np.zeros(self.nv)
        self.throughput = np.zeros(self.nv)
        self.cost[self.grid.ravel()] = np.tile(sig.price / 1000.0, self.ns)
        self.carbon[self.grid.ravel()] = np.tile(sig.carbon / 1000.0, self.ns)
        if self.has_battery:
            self.cost[self.charge.ravel()] = settings.battery_wear
            self.cost[self.discharge.ravel()] = settings.battery_wear
            self.throughput[self.charge.ravel()] = 1
            self.throughput[self.discharge.ravel()] = 1

    def group(self, name, n, lo, hi):
        start = len(self.lower)
        self.lower.extend(np.broadcast_to(lo, (n,)).tolist())
        self.upper.extend(np.broadcast_to(hi, (n,)).tolist())
        ans = np.arange(start, start + n)
        self.groups.setdefault(name, []).extend(ans.tolist())
        return ans

    def eq(self, terms, rhs):
        r = len(self.eq_rhs)
        self.eq_rows.extend([r] * len(terms))
        self.eq_cols.extend(c for c, _ in terms)
        self.eq_vals.extend(v for _, v in terms)
        self.eq_rhs.append(rhs)

    def ub(self, terms, rhs):
        r = len(self.ub_rhs)
        self.ub_rows.extend([r] * len(terms))
        self.ub_cols.extend(c for c, _ in terms)
        self.ub_vals.extend(v for _, v in terms)
        self.ub_rhs.append(rhs)

    def solve(self, objective, budget=None):
        # Only a numerical tie-breaker, not an undocumented economic objective.
        c = (self.carbon.copy() if objective == "carbon" else self.cost.copy()) + 1e-8 * self.throughput
        aub, bub = self.aub, np.array(self.ub_rhs)
        if budget is not None:
            aub = vstack([aub, coo_matrix(self.carbon.reshape(1, -1))], format="csc")
            bub = np.append(bub, budget)
        started = time.perf_counter()
        if self.has_binary:
            integ = np.zeros(self.nv, dtype=int)
            integ[self.binary.ravel()] = 1
            constraints = [LinearConstraint(self.aeq, self.eq_rhs, self.eq_rhs)]
            if aub.shape[0]:
                constraints.append(LinearConstraint(aub, np.full(len(bub), -math.inf), bub))
            result = milp(c, integrality=integ, bounds=Bounds(self.lower, self.upper), constraints=constraints,
                          options={"time_limit": self.cfg.time_limit, "mip_rel_gap": self.cfg.mip_rel_gap})
            solver = "HiGHS_MILP"
        else:
            result = linprog(c, A_ub=aub if len(bub) else None, b_ub=bub if len(bub) else None,
                             A_eq=self.aeq, b_eq=self.eq_rhs, bounds=list(zip(self.lower, self.upper)),
                             method="highs", options={"time_limit": self.cfg.time_limit})
            solver = "HiGHS_LP"
        elapsed = time.perf_counter() - started
        status = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}.get(result.status, str(result.status))
        info = {"status": status, "optimal": bool(result.status == 0), "solver": solver,
                "solve_seconds": elapsed, "solver_message": str(result.message),
                "objective_value": float(result.fun) if result.fun is not None else None,
                "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None}
        if result.x is None:
            return info, None
        x = result.x
        info.update(self.check(x, budget))
        # Positive-price LP has a physically equivalent no-cycling optimum. If
        # numerical degeneracy nevertheless yields cycling, enforce it explicitly.
        if self.has_battery and not self.has_binary and info["max_simultaneous_kwh"] > TOL:
            retry = Model(self.sessions, self.sig, self.cap, self.configuration, self.cfg, force_mip=True,
                          checkpoints=self.checkpoints)
            # Keep extraction indices aligned: binary variables precede EV parts.
            self.__dict__.update(retry.__dict__)
            return retry.solve(objective, budget)
        if max(info[k] for k in ("max_energy_error_kwh", "max_balance_error_kwh", "max_bound_error_kwh", "max_carbon_budget_error_kg", "max_inequality_error_kwh", "max_integrality_error", "max_simultaneous_kwh")) > 1e-4:
            info["status"], info["optimal"] = "failed_postsolve_check", False
        return info, x

    def check(self, x, budget):
        delivered = [sum(x[c] for c, _, _ in self.session_columns[s.session_id]) if self.smart else s.energy for s in self.sessions]
        errors = [abs(e - s.energy) for e, s in zip(delivered, self.sessions)]
        balance = np.asarray(self.aeq @ x - np.array(self.eq_rhs))
        bounds = max(float(np.max(np.maximum(np.array(self.lower) - x, 0))), float(np.max(np.maximum(x - np.array(self.upper), 0))))
        carbon = float(self.carbon @ x)
        return {"max_energy_error_kwh": max(errors, default=0), "max_balance_error_kwh": float(np.max(np.abs(balance))),
                "max_bound_error_kwh": bounds,
                "max_inequality_error_kwh": max(0.0, float(np.max(self.aub @ x - np.array(self.ub_rhs)))) if len(self.ub_rhs) else 0.0,
                "max_integrality_error": float(np.max(np.abs(x[self.binary] - np.rint(x[self.binary])))) if self.has_binary else 0.0,
                "max_carbon_budget_error_kg": max(0.0, carbon - budget) if budget is not None else 0.0,
                "max_simultaneous_kwh": float(np.max(np.minimum(x[self.charge], x[self.discharge]))) if self.has_battery else 0.0}

    def extract(self, x, cap, objective):
        ev = np.zeros((self.ns, self.nt)) if self.smart else self.fixed_load.copy()
        delivery, allocation = [], []
        for s in self.sessions:
            if self.smart:
                items = [(j, x[c], upper) for c, j, upper in self.session_columns[s.session_id]]
                for j, e, upper in items:
                    ev[self.si[s.station_id], j] += e
            else:
                limits = {j: min(s.rated, self.cap) * h for j, h in intersections(s, self.sig)}
                items = [(j, e, limits[j]) for j, e in self.fixed_by_session[s.session_id]]
            delivered = sum(e for _, e, _ in items)
            base = {"power_cap_kw": cap, "configuration": self.configuration, "objective": objective}
            delivery.append({**base, "session_id": s.session_id, "station_id": s.station_id,
                             "connector_id": s.connector_id, "start_utc": iso(s.start), "end_proxy_utc": iso(s.end),
                             "energy_required_kwh": s.energy, "energy_delivered_kwh": delivered,
                             "connection_hours_proxy": (s.end - s.start) / 3600.0,
                             "power_upper_kw": min(s.rated, self.cap),
                             "window_utilization": s.energy / (min(s.rated, self.cap) * (s.end - s.start) / 3600.0)})
            allocation.extend({**base, "session_id": s.session_id, "station_id": s.station_id,
                               "timestamp_utc": iso(self.sig.ts[j]), "energy_kwh": e,
                               "energy_upper_kwh": upper} for j, e, upper in items)
        grid = x[self.grid]
        used = x[self.pv_use] if self.has_pv else np.zeros_like(ev)
        charge = x[self.charge] if self.has_battery else np.zeros_like(ev)
        discharge = x[self.discharge] if self.has_battery else np.zeros_like(ev)
        soc = x[self.soc] if self.has_battery else np.zeros((self.ns, self.nt + 1))
        rows = []
        for i, station in enumerate(self.stations):
            for j, ts in enumerate(self.sig.ts):
                rows.append({"power_cap_kw": cap, "configuration": self.configuration, "objective": objective,
                             "station_id": station, "timestamp_utc": iso(ts), "ev_kwh": ev[i, j], "grid_kwh": grid[i, j],
                             "pv_available_kwh": self.pv_available[i, j], "pv_used_kwh": used[i, j],
                             "pv_curtailed_kwh": self.pv_available[i, j] - used[i, j],
                             "battery_charge_kwh": charge[i, j], "battery_discharge_kwh": discharge[i, j],
                             "soc_start_kwh": soc[i, j], "soc_end_kwh": soc[i, j + 1],
                             "carbon_kg": grid[i, j] * self.sig.carbon[j] / 1000.0,
                             "energy_cost_gbp": grid[i, j] * self.sig.price[j] / 1000.0,
                             "battery_wear_gbp": (charge[i, j] + discharge[i, j]) * self.cfg.battery_wear,
                             "cost_gbp": grid[i, j] * self.sig.price[j] / 1000.0 + (charge[i, j] + discharge[i, j]) * self.cfg.battery_wear,
                             "carbon_g_per_kwh": self.sig.carbon[j], "price_gbp_per_mwh": self.sig.price[j]})
        summary = {"cost_gbp": float(self.cost @ x), "carbon_kg": float(self.carbon @ x), "delivered_kwh": float(ev.sum()),
                   "energy_cost_gbp": float(np.sum(grid * self.sig.price / 1000.0)),
                   "battery_wear_gbp": float((charge.sum() + discharge.sum()) * self.cfg.battery_wear),
                   "grid_import_kwh": float(grid.sum()), "pv_available_kwh": float(self.pv_available.sum()),
                   "pv_used_kwh": float(used.sum()), "pv_curtailed_kwh": float((self.pv_available - used).sum()),
                   "battery_charge_kwh": float(charge.sum()), "battery_discharge_kwh": float(discharge.sum()),
                   "initial_soc_kwh_total": float(soc[:, 0].sum()), "final_soc_kwh_total": float(soc[:, -1].sum()),
                   "n_variables": self.nv, "n_binary_variables": self.nst if self.has_binary else 0}
        return summary, rows, delivery, allocation


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def self_test():
    settings = Settings(time_limit=30)
    t = stamp("2025-01-01T23:30:00Z")
    sig = Signals(np.array([t, t + STEP]), np.array([500.0, 100.0]), np.array([-100.0, 200.0]), np.zeros(2))
    s = Session("cross_midnight", "station", "plug", t + 900, t + 2700, 1.0, 2.0)
    for config in ("fixed_grid", "smart_grid", "smart_pv", "smart_pv_battery"):
        m = Model([s], sig, 22, config, settings)
        info, x = m.solve("cost")
        assert info["optimal"], info
        assert info["max_energy_error_kwh"] < TOL
        assert info["max_balance_error_kwh"] < TOL
        assert info["max_simultaneous_kwh"] < TOL
        if m.smart:
            assert all(abs(x[c] - 0.5) < TOL for c, _, _ in m.session_columns[s.session_id]), "Partial-hour capacity incorrectly rounded"
        if m.has_battery:
            assert m.has_binary, "Negative-price battery requires mutual exclusion"
            assert np.max(np.abs(x[m.soc][:, 0] - x[m.soc][:, -1])) < TOL
    # No PV and a completely occupied fixed-energy window cannot meet a stricter budget.
    m = Model([s], sig, 22, "smart_grid", settings)
    info, x = m.solve("carbon", budget=0.25)
    assert info["status"] == "infeasible" and x is None
    # PV is not allowed to export and surplus can be curtailed, not forced into the battery.
    sig2 = Signals(sig.ts, np.ones(2) * 300, np.ones(2) * 100, np.ones(2))
    m = Model([s], sig2, 22, "fixed_pv", settings)
    info, x = m.solve("cost")
    assert info["optimal"] and np.max(x[m.grid]) < TOL
    assert (m.pv_available - x[m.pv_use]).min() >= -TOL
    print("PASS: exact partial-slot bounds; cross-midnight full energy; no unmet service; negative-price MILP exclusivity; cyclic SOC; infeasible carbon budget; PV curtailment/no export", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--sessions", type=Path, help="Optional alternative session CSV with identical schema")
    p.add_argument("--output-dir", type=Path, default=HERE / "results")
    p.add_argument("--power-caps", default="22", help="Comma separated kW; each cap gets its own documented eligible cohort")
    p.add_argument("--max-stations", type=int, default=10)
    p.add_argument("--station-ids", help="Comma separated station IDs; bypass top-N selection for paired comparisons")
    p.add_argument("--session-ids-file", type=Path, help="CSV containing session_id to restrict a paired common cohort")
    p.add_argument("--configurations", default="fixed_grid,fixed_pv,smart_grid,smart_pv,smart_pv_battery")
    p.add_argument("--budgets", default="0,10,20", help="Percent reductions relative to fixed_grid reference")
    p.add_argument("--skip-frontier", action="store_true", help="Run cost and carbon objectives only")
    p.add_argument("--start", help="Optional UTC horizon start, inclusive; must match a signal slot")
    p.add_argument("--days", type=float, help="Optional horizon duration; crossing sessions retained only if fully covered")
    p.add_argument("--pv-kwp", type=float, default=10)
    p.add_argument("--battery-kwh", type=float, default=20)
    p.add_argument("--battery-kw", type=float, default=10)
    p.add_argument("--roundtrip-efficiency", type=float, default=0.9)
    p.add_argument("--initial-soc-fraction", type=float, default=0.5)
    p.add_argument("--battery-wear", type=float, default=0.01, help="GBP per charge PLUS discharge throughput kWh, operational assumption, not measured")
    p.add_argument("--time-limit", type=float, default=120)
    p.add_argument("--mip-rel-gap", type=float, default=0.001)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    cfg = Settings(args.pv_kwp, args.battery_kwh, args.battery_kw, args.roundtrip_efficiency,
                   args.initial_soc_fraction, args.battery_wear, args.time_limit, args.mip_rel_gap)
    if not 0 < cfg.roundtrip_efficiency <= 1 or not 0 <= cfg.initial_soc_fraction <= 1 or cfg.battery_wear < 0:
        p.error("Invalid efficiency / SOC / wear settings")
    sessions, sig, read_rejected = read_inputs(args.data_dir, args.sessions)
    full_count = len(sessions)
    if args.session_ids_file:
        with args.session_ids_file.open(encoding="utf-8-sig", newline="") as f:
            selected_ids = {str(r["session_id"]) for r in csv.DictReader(f)}
        read_rejected.extend({"session_id": s.session_id, "reason": "outside_explicit_common_cohort"} for s in sessions if s.session_id not in selected_ids)
        sessions = [s for s in sessions if s.session_id in selected_ids]
    start = stamp(args.start) if args.start else sig.ts[0]
    finish = start + args.days * 86400 if args.days else sig.ts[-1] + STEP
    mask = (sig.ts >= start) & (sig.ts < finish)
    sig = Signals(*(a[mask] for a in (sig.ts, sig.carbon, sig.price, sig.pv)))
    if not len(sig.ts):
        raise ValueError("Empty selected horizon")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, timeseries, delivery, allocations, all_excluded = [], [], [], [], []
    cohorts = []
    allowed = {"fixed_grid", "fixed_pv", "smart_grid", "smart_pv", "smart_pv_battery"}
    configs = args.configurations.split(",")
    if any(c not in allowed for c in configs):
        p.error("Unknown configuration")
    for cap in map(float, args.power_caps.split(",")):
        cohort, excluded = select_sessions(sessions, sig, cap, args.max_stations,
                                           args.station_ids.split(",") if args.station_ids else None)
        all_excluded.extend({"power_cap_kw": cap, **r} for r in read_rejected + excluded)
        if not cohort:
            print(f"No eligible sessions at cap={cap}", flush=True)
            continue
        stations = sorted({s.station_id for s in cohort})
        reference, _ = early_reference(cohort, sig, cap, stations)
        base_carbon = float(np.sum(reference * sig.carbon / 1000.0))
        base_cost = float(np.sum(reference * sig.price / 1000.0))
        cohort_info = {"power_cap_kw": cap, "n_input_valid_sessions": full_count, "n_sessions": len(cohort),
                       "n_stations": len(stations), "n_excluded": len(excluded) + len(read_rejected),
                       "station_ids": stations, "delivered_kwh": sum(s.energy for s in cohort),
                       "baseline_carbon_kg": base_carbon, "baseline_cost_gbp": base_cost}
        cohorts.append(cohort_info)
        for configuration in configs:
            m = Model(cohort, sig, cap, configuration, cfg)
            tasks = [("cost", None, None), ("carbon", None, None)]
            if not args.skip_frontier:
                tasks += [(f"budget_{b:g}", b, base_carbon * (1 - b / 100)) for b in map(float, args.budgets.split(",")) if 0 <= b <= 100]
            for objective, target, budget in tasks:
                info, x = m.solve("carbon" if objective == "carbon" else "cost", budget)
                row = {"power_cap_kw": cap, "configuration": configuration, "objective": objective,
                       **info, "target_reduction_pct": target, "carbon_budget_kg": budget,
                       "baseline_carbon_kg": base_carbon, "baseline_cost_gbp": base_cost,
                       "n_sessions": len(cohort), "n_stations": len(stations), "n_excluded": len(excluded) + len(read_rejected),
                       "battery_wear_rate_gbp_per_kwh_throughput": cfg.battery_wear}
                if x is not None:
                    values, tsrows, dsrows, arows = m.extract(x, cap, objective)
                    row.update(values)
                    row["reduction_pct"] = (1 - values["carbon_kg"] / base_carbon) * 100 if base_carbon else 0
                    row["incremental_cost_vs_baseline_gbp"] = values["cost_gbp"] - base_cost
                    avoided = base_carbon - values["carbon_kg"]
                    row["additional_gbp_per_tonne_footprint_reduction"] = (values["cost_gbp"] - base_cost) * 1000 / avoided if avoided > TOL else None
                    # Include feasible incumbents, but preserve nonoptimal status on every row.
                    for dataset in (tsrows, dsrows, arows):
                        for rr in dataset:
                            rr.update({"status": info["status"], "optimal": info["optimal"]})
                    timeseries.extend(tsrows)
                    delivery.extend(dsrows)
                    allocations.extend(arows)
                summaries.append(row)
                print(f"cap={cap:g} {configuration} {objective}: {info['status']} "
                      f"cost={row.get('cost_gbp')} carbon={row.get('carbon_kg')} seconds={info['solve_seconds']:.2f}", flush=True)
                write_csv(args.output_dir / "summary.csv", summaries)
        write_csv(args.output_dir / "timeseries.csv", timeseries)
        write_csv(args.output_dir / "session_delivery.csv", delivery)
        write_csv(args.output_dir / "session_allocation.csv", allocations)
    write_csv(args.output_dir / "excluded_sessions.csv", all_excluded)
    manifest = {"created_utc": iso(datetime.now(timezone.utc).timestamp()), "scipy_version": scipy.__version__,
                "session_input": str((args.sessions or args.data_dir / "sessions.csv").resolve()),
                "signal_input": str((args.data_dir / "signals.csv").resolve()),
                "explicit_session_ids_file": str(args.session_ids_file.resolve()) if args.session_ids_file else None,
                "explicit_station_ids": args.station_ids.split(",") if args.station_ids else None,
                "settings": vars(cfg), "horizon_start_utc": iso(sig.ts[0]), "horizon_end_utc_exclusive": iso(sig.ts[-1] + STEP),
                "n_slots": len(sig.ts), "negative_price_slots": int((sig.price < 0).sum()),
                "cohorts": cohorts, "assumptions": [
                    "Ex-post perfect-information potential, not deployable online or day-ahead results.",
                    "Attributional purchased-electricity carbon footprint using supplied average carbon intensity, not system-wide causal avoided emissions.",
                    "Connection end is a proxy; the complete recorded energy is required within each retained full session.",
                    "Reference is immediate constant capped-power charging; not measured uncontrolled charging.",
                    "Rated power <=22kW is an AC eligibility proxy; each scenario uses min(rated power, scenario cap).",
                    "PV10kWp and BESS20kWh/10kW per selected station by default are hypothetical, not measured installed equipment.",
                    "PV may be curtailed. No grid export, no vehicle discharge. Fixed battery initial and final SOC are identical.",
                    "Battery wear is a stated GBP per total charge/discharge throughput operating-cost assumption; investment costs excluded.",
                    "Negative-price battery cost optimization uses mutually exclusive MILP charge/discharge; solver-limit incumbents are not called optima.",
                    "Different power-cap cohorts may differ; use an explicit common-cohort rerun for causal cap-to-cap comparison."]}
    (args.output_dir / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

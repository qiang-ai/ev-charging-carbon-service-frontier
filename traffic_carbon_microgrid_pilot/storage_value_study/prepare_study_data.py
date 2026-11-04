"""Prepare reproducible two-city, two-season public charging inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw"
HF = "https://huggingface.co/datasets/djordjebatic/GridCharge/raw/main/"
CITIES = {
    "edinburgh": ("City of Edinburgh", 55.9533, -3.1883),
    "glasgow": ("Glasgow City", 55.8642, -4.2518),
}
MONTHS = {"2023_01": "january-2023", "2023_07": "july-2023"}


def month_bounds(month):
    start = pd.Timestamp(f"{month.replace('_', '-')}-01", tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def source_urls():
    urls = {"gridcharge_README.md": HF + "README.md"}
    for city, (authority, lat, lon) in CITIES.items():
        urls[f"infrastructure_{city}.csv"] = (
            HF + f"carbon_mix/{quote(authority)}/charging_infrastructure.csv"
        )
        params = urlencode({
            "lat": lat, "lon": lon, "startyear": 2023, "endyear": 2023,
            "raddatabase": "PVGIS-ERA5", "pvcalculation": 1, "peakpower": 1,
            "loss": 14, "angle": 35, "aspect": 0, "outputformat": "json",
        })
        urls[f"pvgis_{city}_2023.json"] = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?" + params
    for month, gridcharge_name in MONTHS.items():
        urls[f"gridcharge_{month}.csv"] = HF + f"sessions/{gridcharge_name}.csv"
        start, end = month_bounds(month)
        left = start
        while left < end:
            right = min(left + pd.Timedelta(days=7), end)
            label = left.strftime("%Y%m%d")
            a, b = (stamp.strftime("%Y-%m-%dT%H:%MZ") for stamp in (left, right))
            urls[f"neso_{label}.json"] = f"https://api.carbonintensity.org.uk/intensity/{a}/{b}"
            params = urlencode({"from": a, "to": b, "dataProviders": "APXMIDP", "format": "json"})
            urls[f"elexon_{label}.json"] = (
                "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index?" + params
            )
            left = right
    return urls


def download(name, url):
    target = RAW / name
    if target.exists() and target.stat().st_size:
        return target.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(url, timeout=50) as response:
                payload = response.read()
            if not payload:
                raise ValueError("empty response")
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(payload)
            temp.replace(target)
            print(f"downloaded {name} ({len(payload)} bytes)", flush=True)
            return payload
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def source_manifest(offline):
    manifest_file = DATA / "raw_manifest.json"
    old = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    files, errors = {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {
            pool.submit((RAW / name).read_bytes) if offline else pool.submit(download, name, url): (name, url)
            for name, url in source_urls().items()
        }
        for future in as_completed(pending):
            name, url = pending[future]
            try:
                payload = future.result()
                digest = hashlib.sha256(payload).hexdigest()
                if name in old.get("files", {}) and digest != old["files"][name]["sha256"]:
                    raise ValueError(f"Raw snapshot checksum mismatch for {name}")
                files[name] = {"url": url, "bytes": len(payload), "sha256": digest}
            except Exception as exc:
                errors[name] = str(exc)
                print(f"FAILED {name}: {exc}", flush=True)
    if not errors or not manifest_file.exists():
        write_json(manifest_file, {"files": files, "errors": errors})
    if errors:
        raise RuntimeError(f"Unavailable or changed raw snapshots: {sorted(errors)}")
    return files


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def json_rows(prefix):
    for path in sorted(RAW.glob(f"{prefix}_*.json")):
        yield from json.loads(path.read_bytes())["data"]


def index_observations(frame, columns):
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    conflicts = frame.groupby("timestamp_utc")[columns].nunique(dropna=False).gt(1)
    if conflicts.any().any():
        raise ValueError(f"Conflicting duplicates in {columns}")
    return frame.drop_duplicates("timestamp_utc").set_index("timestamp_utc")


def market_and_carbon():
    carbon = pd.DataFrame([
        {"timestamp_utc": row["from"], "carbon_g_per_kwh": row["intensity"]["actual"]}
        for row in json_rows("neso")
    ])
    prices = pd.DataFrame([
        {"timestamp_utc": row["startTime"], "price_gbp_per_mwh": row["price"], "volume_mwh": row["volume"]}
        for row in json_rows("elexon") if row["dataProvider"] == "APXMIDP"
    ])
    return (
        index_observations(carbon, ["carbon_g_per_kwh"]),
        index_observations(prices, ["price_gbp_per_mwh", "volume_mwh"]),
    )


def pv_half_hourly(city):
    payload = json.loads((RAW / f"pvgis_{city}_2023.json").read_bytes())
    hourly = pd.DataFrame(payload["outputs"]["hourly"])
    times = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M", utc=True)
    if not times.dt.minute.eq(30).all() or len(times) != 8760:
        raise ValueError(f"Unexpected PVGIS ERA5 hourly coverage for {city}")
    first_half = pd.Series(pd.to_numeric(hourly["P"]).to_numpy() / 1000, index=times.dt.floor("h"))
    if not first_half.index.is_unique:
        raise ValueError(f"Duplicate PVGIS hour for {city}")
    second_half = first_half.copy()
    second_half.index += pd.Timedelta(minutes=30)
    # Each half-hour receives its containing hour's mean power. This preserves energy.
    return pd.concat([first_half, second_half]).rename("pv_kw_per_kwp")


def common_complete_window(month, carbon, prices, city_pv):
    start, end = month_bounds(month)
    index = pd.date_range(start, end, freq="30min", inclusive="left")
    aligned = pd.DataFrame(index=index).join(carbon).join(prices)
    for city, pv in city_pv.items():
        aligned[city + "_pv_kw_per_kwp"] = pv
    missing_times = {
        column: [timestamp.isoformat() for timestamp in aligned.index[aligned[column].isna()]]
        for column in aligned.columns
    }
    complete_days = np.isfinite(aligned.to_numpy(dtype=float)).all(axis=1).reshape(-1, 48).all(axis=1)
    blocks = []
    block_start = 0
    for day, complete in enumerate([*complete_days, False]):
        if not complete:
            blocks.append((block_start, day))
            block_start = day + 1
    first, last = max(blocks, key=lambda pair: pair[1] - pair[0])
    if last - first < 21:
        raise ValueError(f"Fewer than 21 consecutive complete days in {month}: {missing_times}")
    return start + pd.Timedelta(days=first), start + pd.Timedelta(days=last), {
        "requested_full_month_start_utc": start.isoformat(),
        "requested_full_month_end_exclusive_utc": end.isoformat(),
        "missing_signal_timestamps_in_full_month": missing_times,
        "complete_days_retained": last - first,
        "selection_rule": "Longest consecutive block of complete UTC days common to both cities, earliest tie. No missing observations filled.",
    }


def make_signals(start, end, carbon, prices, pv):
    expected = pd.date_range(start, end, freq="30min", inclusive="left", name="timestamp_utc")
    signals = pd.DataFrame(index=expected).join(carbon).join(prices).join(pv)
    missing = signals.isna().sum().to_dict()
    if not np.isfinite(signals.to_numpy(dtype=float)).all():
        raise ValueError(f"Incomplete signals for {start:%Y-%m}: {missing}")
    if (signals[["carbon_g_per_kwh", "pv_kw_per_kwp"]] < 0).any().any():
        raise ValueError("Negative carbon intensity or modeled PV power")
    info = {
        "rows": len(signals), "missing": missing,
        "carbon_g_per_kwh_range": [signals.carbon_g_per_kwh.min(), signals.carbon_g_per_kwh.max()],
        "price_gbp_per_mwh_range": [signals.price_gbp_per_mwh.min(), signals.price_gbp_per_mwh.max()],
        "pv_kwh_per_kwp": float(signals.pv_kw_per_kwp.sum() * .5),
        "market_periods_with_zero_reported_volume": int(signals.volume_mwh.eq(0).sum()),
    }
    info["carbon_g_per_kwh_range"] = [float(value) for value in info["carbon_g_per_kwh_range"]]
    return signals.drop(columns="volume_mwh").reset_index(), info


def normalize_device(series):
    # GridCharge uses an APT alias on some otherwise numeric device identifiers.
    return series.astype("string").str.strip().str.upper().str.replace(r"^APT(?=\d+$)", "", regex=True)


def normalize_address(address):
    return re.sub(r"[^\w]+", " ", address.casefold()).strip()


def infrastructure():
    cities = []
    for city, (authority, _, _) in CITIES.items():
        snapshot = pd.read_csv(RAW / f"infrastructure_{city}.csv", dtype={"CP ID": "string"})
        if not snapshot["Local Authority"].eq(authority).all():
            raise ValueError(f"Unexpected authority in {city} infrastructure snapshot")
        frame = snapshot.rename(columns={
            "CP ID": "infrastructure_device_id", "Connector": "port_id", "Address": "address",
            "Latitude": "latitude", "Longitude": "longitude", "Postcode": "postcode",
            "Nominal Power (kW)": "power_kw", "Connector Type": "connector_type",
            "Local Authority": "local_authority", "Region ID": "region_id",
        })
        frame["infrastructure_source_row"] = np.arange(1, len(frame) + 1)
        frame["city"] = city
        frame["device_id"] = normalize_device(frame.infrastructure_device_id)
        frame["port_id"] = pd.to_numeric(frame.port_id, errors="coerce").astype("Int64")
        frame["power_kw"] = pd.to_numeric(frame.power_kw, errors="coerce")
        frame["address_key"] = frame.address.fillna("").map(normalize_address)
        frame["site_pin_key"] = (
            frame.latitude.map("{:.6f}".format) + "," + frame.longitude.map("{:.6f}".format)
        )
        site_keys = city + "|" + frame.site_pin_key + "|" + frame.address_key
        frame["station_id"] = site_keys.map(lambda key: city + "_" + hashlib.sha256(key.encode()).hexdigest()[:12])
        frame["connector_id"] = frame.device_id + ":" + frame.port_id.astype("string")
        frame["mapping_status"] = "usable"
        bad_coords = ~frame.latitude.between(-90, 90) | ~frame.longitude.between(-180, 180)
        bad_key = frame.device_id.isna() | frame.device_id.eq("") | frame.port_id.isna() | frame.port_id.le(0)
        bad_power = ~np.isfinite(frame.power_kw) | frame.power_kw.le(0)
        frame.loc[bad_coords, "mapping_status"] = "invalid_coordinates"
        frame.loc[bad_key.fillna(True), "mapping_status"] = "invalid_device_or_port"
        frame.loc[bad_power, "mapping_status"] = "invalid_rated_power"
        cities.append(frame)

    mapping = pd.concat(cities, ignore_index=True)
    candidates = mapping.loc[mapping.mapping_status.eq("usable")]
    keys = ["device_id", "port_id"]
    conflicting = candidates.groupby(keys)[["station_id", "power_kw"]].nunique().gt(1).any(axis=1)
    conflict_keys = conflicting.index[conflicting]
    conflict_rows = pd.MultiIndex.from_frame(mapping[keys]).isin(conflict_keys)
    mapping.loc[conflict_rows, "mapping_status"] = "ambiguous_connector_mapping"
    valid = mapping.loc[mapping.mapping_status.eq("usable")]
    duplicates = valid.index[valid.duplicated(keys)]
    mapping.loc[duplicates, "mapping_status"] = "duplicate_mapping_entry"
    valid = mapping.loc[mapping.mapping_status.eq("usable")].copy()
    site_rows = valid.groupby("station_id", sort=True).agg(
        city=("city", "first"), local_authority=("local_authority", "first"),
        latitude=("latitude", "first"), longitude=("longitude", "first"),
        address=("address", "first"), postcode=("postcode", "first"),
        address_key=("address_key", "first"), site_pin_key=("site_pin_key", "first"),
        snapshot_devices=("device_id", "nunique"), snapshot_connectors=("connector_id", "nunique"),
    ).reset_index()
    site_rows["same_address_pin_count"] = site_rows.groupby(["city", "address_key"])["site_pin_key"].transform("nunique")
    site_rows.loc[site_rows.address_key.eq(""), "same_address_pin_count"] = pd.NA
    site_rows["same_pin_address_count"] = site_rows.groupby(["city", "site_pin_key"])["address_key"].transform("nunique")
    public_columns = [
        "city", "local_authority", "infrastructure_source_row", "infrastructure_device_id",
        "device_id", "port_id", "connector_id", "station_id", "power_kw", "connector_type",
        "latitude", "longitude", "address", "postcode", "region_id", "address_key", "site_pin_key", "mapping_status",
    ]
    return mapping[public_columns], valid, site_rows


def load_raw_sessions(month):
    source = pd.read_csv(RAW / f"gridcharge_{month}.csv", dtype={"CP ID": "string"})
    source = source.rename(columns={"CP ID": "source_device_id", "Connector": "port_id", "Consumed(kWh)": "energy_kwh"})
    source["source_session_row"] = np.arange(1, len(source) + 1)
    source["session_id"] = f"gc_{month}_" + source.source_session_row.astype(str)
    source["device_id"] = normalize_device(source.source_device_id)
    source["port_id"] = pd.to_numeric(source.port_id, errors="coerce").astype("Int64")
    source["energy_kwh"] = pd.to_numeric(source.energy_kwh, errors="coerce")
    local_start = pd.to_datetime(source.Start, format="ISO8601", errors="coerce")
    source["start_utc"] = local_start.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    duration = pd.to_timedelta(source.Duration, errors="coerce")
    source["duration_hours"] = duration.dt.total_seconds() / 3600
    source["end_proxy_utc"] = source.start_utc + duration
    details = {
        "all_scotland_raw_rows": len(source),
        "raw_naive_start_min": str(local_start.min()), "raw_naive_start_max": str(local_start.max()),
        "converted_all_scotland_start_min": str(source.start_utc.min()),
        "converted_all_scotland_start_max": str(source.start_utc.max()),
        "all_scotland_invalid_start_rows": int(source.start_utc.isna().sum()),
    }
    return source, details


def remove_overlaps(sessions):
    overlaps = []
    ordered = sessions.sort_values(["connector_id", "start_utc", "source_session_row"])
    for _, group in ordered.groupby("connector_id", sort=False):
        end = None
        for row in group.itertuples():
            if end is not None and row.start_utc < end:
                overlaps.append(row.Index)
            else:
                end = row.end_proxy_utc
    return ordered.drop(index=overlaps), len(overlaps)


def make_sessions(source, connector_map, city, start, end):
    device_keys = ["device_id", "port_id"]
    columns = device_keys + ["station_id", "connector_id", "power_kw", "connector_type", "address"]
    city_map = connector_map.loc[connector_map.city.eq(city), columns]
    sessions = source.merge(city_map, on=device_keys, how="inner", validate="many_to_one")
    mapped_rows = len(sessions)
    criteria = [
        ("invalid_time_or_duration", sessions.start_utc.notna() & sessions.duration_hours.notna()),
        ("duration_outside_0_to_24_hours", sessions.duration_hours.gt(0) & sessions.duration_hours.le(24)),
        ("nonpositive_or_nonfinite_energy", sessions.energy_kwh.gt(0) & np.isfinite(sessions.energy_kwh)),
        ("energy_exceeds_rated_power_times_duration", sessions.energy_kwh.le(sessions.power_kw * sessions.duration_hours + 1e-9)),
        ("session_not_fully_inside_utc_window", sessions.start_utc.ge(start) & sessions.end_proxy_utc.le(end)),
    ]
    retained = pd.Series(True, index=sessions.index)
    exclusions = {}
    for reason, condition in criteria:
        condition = condition.fillna(False)
        exclusions[reason] = int((retained & ~condition).sum())
        retained &= condition
    sessions = sessions.loc[retained].copy()
    duplicates = sessions.duplicated(["connector_id", "start_utc", "end_proxy_utc", "energy_kwh"])
    exclusions["exact_duplicate"] = int(duplicates.sum())
    sessions, exclusions["overlap_with_earlier_session_same_connector"] = remove_overlaps(sessions.loc[~duplicates])
    sessions["site_name"] = sessions.address
    session_columns = [
        "session_id", "station_id", "connector_id", "start_utc", "end_proxy_utc",
        "energy_kwh", "power_kw", "duration_hours", "device_id", "port_id",
        "source_device_id", "source_session_row", "connector_type", "site_name", "address",
    ]
    sessions = sessions[session_columns].sort_values(["start_utc", "station_id", "connector_id"])
    if sessions.empty or len(sessions) + sum(exclusions.values()) != mapped_rows:
        raise ValueError(f"Unexpected session accounting for {city} {start:%Y-%m}")
    summary = {
        "city_mapped_rows_before_cleaning": mapped_rows, "clean_sessions": len(sessions),
        "clean_sites": sessions.station_id.nunique(), "clean_devices": sessions.device_id.nunique(),
        "clean_connectors": sessions.connector_id.nunique(), "recorded_energy_kwh": float(sessions.energy_kwh.sum()),
        "sequential_exclusions": exclusions,
    }
    return sessions, summary


NOTES = {
    "zh": [
        "场址按同城、经纬度各保留6位小数和规范化地址相同归组。同一地址的不同坐标仍分为不同场址。该分组表示地理同址，实际并网点及供电容量未知。",
        "设备编号和接口编号分别保留，唯一接口键由两者组合。对纯数字编号移除APT前缀作连接，保留源编号和源行号。基础设施来自公开快照，历史额定功率和完整覆盖尚未单独核实。",
        "会话原始Start无时区。主结果按Europe/London转换UTC。1月源文件边界接近UTC月首月末，7月源边界接近当地时间月首01:00至次月01:00，支持BST解释。",
        "交付电量采用原始会话记录。Start加Duration为会话结束代理。真实拔枪时刻、提前申报离场和会话内功率曲线未公开。",
        "会话保留完整落在所选UTC研究区间内的记录。两城共用连续完整的日期范围，信号缺测不填补。清洗剔除重叠、重复以及与额定功率和时长不相容的记录，未按比例裁剪边界会话。",
        "电力碳强度为NESO的GB全国半小时实际平均信号，成本为Elexon APXMIDP批发指数情景。光伏是城市参考点1 kWp的PVGIS ERA5模型出力。两半小时使用同一小时平均功率，保持小时能量。",
    ],
    "en": [
        "A geographic site requires identical city, coordinates rounded to six decimal places, and normalized address. Separate pins remain separate even if they share an address. The actual electrical connection and site capacity are unknown.",
        "Device and port identifiers remain distinct. The connector key combines both. Numeric APT aliases are normalized, and original identifiers and source row numbers are retained. Historical ratings and complete coverage of the infrastructure snapshot are unverified.",
        "Raw session start times are timezone-naive and interpreted as Europe/London before UTC conversion. January bounds are near local midnight and July bounds near 01:00 BST, consistent with files spanning UTC calendar months.",
        "Delivered energy comes from raw session records. Start plus Duration is a session-end proxy. Physical unplug times, advance departure requests, and intra-session power traces are unavailable.",
        "Only complete sessions inside the selected UTC window are retained. Both cities use the same contiguous dates with complete signals. Overlaps, duplicates and records infeasible under snapshot power limits are excluded. Boundary sessions are never prorated.",
        "NESO GB half-hour actual average grid intensity and Elexon APXMIDP wholesale index are common to both cities. PVGIS ERA5 estimates 1 kWp PV output at each city reference point. The hourly mean is repeated within the hour to preserve energy.",
    ],
}


def prepare():
    carbon, prices = market_and_carbon()
    mapping, valid_map, sites = infrastructure()
    mapping.to_csv(DATA / "site_mapping.csv", index=False)
    sites.to_csv(DATA / "site_catalog.csv", index=False)
    city_pv = {city: pv_half_hourly(city) for city in CITIES}
    inputs, site_stats = {}, []
    for month in MONTHS:
        start, end, window_details = common_complete_window(month, carbon, prices, city_pv)
        source, source_details = load_raw_sessions(month)
        for city in CITIES:
            name = f"{city}_{month}"
            directory = DATA / name
            directory.mkdir(exist_ok=True)
            signals, signal_details = make_signals(start, end, carbon, prices, city_pv[city])
            sessions, session_details = make_sessions(source, valid_map, city, start, end)
            sessions.to_csv(directory / "sessions.csv", index=False)
            signals.to_csv(directory / "signals.csv", index=False)
            grouped = sessions.assign(utc_day=sessions.start_utc.dt.floor("D")).groupby("station_id")
            stats = grouped.agg(
                sessions=("session_id", "size"), energy_kwh=("energy_kwh", "sum"),
                active_utc_days=("utc_day", "nunique"), devices=("device_id", "nunique"),
                connectors=("connector_id", "nunique"), median_duration_hours=("duration_hours", "median"),
            ).reset_index()
            stats.insert(0, "month", month)
            stats.insert(0, "city", city)
            site_stats.append(stats)
            info = {
                "input_directory": name, "city": city, "month": month,
                "window_start_utc": start.isoformat(), "window_end_exclusive_utc": end.isoformat(),
                "source_sessions": source_details, "session_cleaning": session_details,
                "signals": signal_details, "window_selection": window_details, "notes": NOTES,
                "source_manifest": "../raw_manifest.json", "site_mapping": "../site_mapping.csv",
            }
            write_json(directory / "metadata.json", info)
            inputs[name] = {
                **session_details, "signal_rows": len(signals), "pv_kwh_per_kwp": signal_details["pv_kwh_per_kwp"],
                "window_start_utc": start.isoformat(), "window_end_exclusive_utc": end.isoformat(),
                "window_selection": window_details,
            }
            print(f"{name}: {len(sessions)} sessions at {session_details['clean_sites']} sites, {len(signals)} intervals", flush=True)
    pd.concat(site_stats, ignore_index=True).to_csv(DATA / "site_month_summary.csv", index=False)
    write_json(DATA / "metadata.json", {
        "inputs": inputs, "notes": NOTES,
        "mapping_status_counts": mapping.groupby(["city", "mapping_status"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "source_manifest": "raw_manifest.json",
        "reference_sources": [
            "https://huggingface.co/datasets/djordjebatic/GridCharge",
            "https://carbon-intensity.github.io/api-definitions/",
            "https://developer.data.elexon.co.uk/",
            "https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en",
        ],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Read and verify stored raw snapshots")
    parser.add_argument("--download-only", action="store_true", help="Download raw snapshots without deriving study tables")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    source_manifest(args.offline)
    if not args.download_only:
        prepare()


if __name__ == "__main__":
    main()

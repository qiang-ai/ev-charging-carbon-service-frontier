"""Build 2024 holdout inputs with the frozen 2023 site definitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urlencode

import pandas as pd
import numpy as np


HOLDOUT = Path(__file__).resolve().parent
REFERENCE = HOLDOUT.parent
DATA = HOLDOUT / "data"
RAW = DATA / "raw"
MONTHS = {"2024_01": "january-2024", "2024_07": "july-2024"}
HARMONIZED = HOLDOUT / "data_harmonized_2023"
GTI = "global_tilted_irradiance"
SOLAR_DOCS = "https://open-meteo.com/en/docs/historical-weather-api"
PV_NOTE = {
    "zh": "两年光伏同用Open-Meteo历史API的ERA5倾斜面辐照。城市参考点、倾角35°、正南，P(kW/kWp)=0.86×GTI(W/m²)/1000。整点辐照是前一小时均值，时间戳减1小时后复制到该小时的两个半小时。该简化模型统一采用固定系统效率。",
    "en": "Both years use ERA5 global tilted irradiance from the Open-Meteo Historical Weather API, at the city reference point with 35-degree tilt and due-south orientation. P(kW/kWp) = 0.86 × GTI(W/m²)/1000. Hour-end stamps are shifted back one hour. Each hour's mean is repeated in its two half-hours. The simplified PV model uses constant system efficiency.",
}
sys.path.insert(0, str(REFERENCE))
import prepare_study_data as study


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_reference_snapshots():
    reference_data = REFERENCE / "data"
    reference_manifest = json.loads((reference_data / "raw_manifest.json").read_text(encoding="utf-8"))
    names = ["gridcharge_README.md", *(f"infrastructure_{city}.csv" for city in study.CITIES)]
    frozen = {}
    for name in names:
        source = reference_data / "raw" / name
        destination = RAW / name
        expected = reference_manifest["files"][name]
        if sha256(source) != expected["sha256"]:
            raise ValueError(f"Frozen 2023 source snapshot does not match its manifest: {name}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256(destination) != expected["sha256"]:
            raise ValueError(f"Holdout copy does not match the frozen 2023 snapshot: {name}")
        frozen[name] = expected
    return frozen


def gti_url(city, month):
    _, lat, lon = study.CITIES[city]
    start, end = study.month_bounds(month)
    params = urlencode({
        "latitude": lat, "longitude": lon, "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"), "hourly": GTI, "tilt": 35,
        "azimuth": 0, "timezone": "GMT", "models": "era5",
    })
    return "https://archive-api.open-meteo.com/v1/archive?" + params


def pv_series(path, month):
    payload = json.loads(path.read_bytes())
    if payload["utc_offset_seconds"] != 0 or payload["hourly_units"][GTI] != "W/m²":
        raise ValueError(f"Unexpected Open-Meteo irradiance time offset or units in {path.name}")
    start, end = study.month_bounds(month)
    hour_start = pd.DatetimeIndex(pd.to_datetime(payload["hourly"]["time"], utc=True)) - pd.Timedelta(hours=1)
    irradiance = pd.Series(payload["hourly"][GTI], index=hour_start, dtype=float)
    if not irradiance.index.is_unique:
        raise ValueError(f"Duplicate Open-Meteo hour in {path.name}")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    irradiance = irradiance.reindex(expected)
    if (irradiance < 0).any():
        raise ValueError(f"Negative irradiance in {path.name}")
    first_half = irradiance * .86 / 1000
    second_half = first_half.copy()
    second_half.index += pd.Timedelta(minutes=30)
    return pd.concat([first_half, second_half]).rename("pv_kw_per_kwp")


def pv_half_hourly(city):
    return pd.concat([pv_series(RAW / f"era5_gti_{city}_{month}.json", month) for month in MONTHS])


def preserve_pvgis_probes():
    manifest = DATA / "raw_manifest.json"
    if not manifest.exists():
        return
    old = json.loads(manifest.read_text(encoding="utf-8"))
    probes = {name: entry for name, entry in old.get("files", {}).items() if name.startswith("pvgis_")}
    if probes:
        study.write_json(DATA / "unused_pvgis6_probes.json", {
            "files": probes,
            "reason": "The PVGIS 6 ERA5 legacy crystalline-silicon/Huld responses contained null daylight power values in the study windows. These snapshots are retained for provenance. The study uses the same Open-Meteo ERA5 irradiance conversion for both years.",
        })


def configure_study(frozen):
    study.DATA = DATA
    study.RAW = RAW
    study.MONTHS = MONTHS
    urls = study.source_urls()
    for city in study.CITIES:
        urls.pop(f"pvgis_{city}_2023.json")
        for month in MONTHS:
            urls[f"era5_gti_{city}_{month}.json"] = gti_url(city, month)
    urls.update({name: entry["url"] for name, entry in frozen.items()})
    study.source_urls = lambda: urls.copy()
    study.pv_half_hourly = pv_half_hourly
    study.NOTES = {
        **study.NOTES,
        "zh": [*study.NOTES["zh"][:-1], "跨年复现沿用2023年冻结的基础设施快照和场址映射规则。碳强度和市场指数沿用NESO全国半小时实际平均碳强度与Elexon APXMIDP批发指数。", PV_NOTE["zh"]],
        "en": [*study.NOTES["en"][:-1], "The holdout uses byte-identical frozen 2023 infrastructure snapshots and the same site grouping. The power-system inputs retain NESO GB half-hour actual average grid intensity and Elexon APXMIDP wholesale index.", PV_NOTE["en"]],
    }


def validate_reference_mapping():
    result = {}
    for name in ("site_mapping.csv", "site_catalog.csv"):
        reference_path = REFERENCE / "data" / name
        holdout_path = DATA / name
        pd.testing.assert_frame_equal(pd.read_csv(reference_path), pd.read_csv(holdout_path))
        result[name] = {
            "reference_path": f"../../data/{name}",
            "reference_sha256": sha256(reference_path),
            "holdout_sha256": sha256(holdout_path),
        }
    return result


def harmonize_2023(offline):
    months = ("2023_01", "2023_07")
    study.DATA = HARMONIZED
    study.RAW = HARMONIZED / "raw"
    study.RAW.mkdir(parents=True, exist_ok=True)
    urls = {f"era5_gti_{city}_{month}.json": gti_url(city, month) for city in study.CITIES for month in months}
    study.source_urls = lambda: urls.copy()
    study.source_manifest(offline)
    inputs = {}
    for city in study.CITIES:
        for month in months:
            name = f"{city}_{month}"
            reference = REFERENCE / "data" / name
            target = HARMONIZED / name
            target.mkdir(exist_ok=True)
            shutil.copyfile(reference / "sessions.csv", target / "sessions.csv")
            with (reference / "signals.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            timestamps = pd.to_datetime([row["timestamp_utc"] for row in rows], utc=True)
            raw_name = f"era5_gti_{city}_{month}.json"
            pv = pv_series(study.RAW / raw_name, month).reindex(timestamps)
            if not np.isfinite(pv.to_numpy()).all():
                raise ValueError(f"Incomplete harmonized PV for {name}")
            with (target / "signals.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fields, lineterminator="\n")
                writer.writeheader()
                for row, power in zip(rows, pv):
                    row["pv_kw_per_kwp"] = str(power)
                    writer.writerow(row)
            inputs[name] = {
                "reference_sessions_sha256": sha256(reference / "sessions.csv"),
                "sessions_sha256": sha256(target / "sessions.csv"),
                "reference_signals_sha256": sha256(reference / "signals.csv"),
                "signals_sha256": sha256(target / "signals.csv"),
                "source_irradiance_snapshot": raw_name,
                "signal_rows": len(rows), "pv_kwh_per_kwp": float(pv.sum() * .5),
            }
            print(f"harmonized {name}: {len(rows)} intervals with frozen 2023 sessions, price, and carbon", flush=True)
    study.write_json(HARMONIZED / "metadata.json", {
        "inputs": inputs, "pv_method": PV_NOTE, "source_documentation": SOLAR_DOCS,
        "source_manifest": "raw_manifest.json",
        "purpose": "PV-model harmonization sensitivity. Session files are byte-identical to frozen 2023 inputs. The original timestamps, price and carbon text fields are preserved verbatim.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use and verify the downloaded snapshots")
    parser.add_argument("--download-only", action="store_true", help="Download raw files without deriving tables")
    parser.add_argument("--harmonize-2023", action="store_true", help="Build only the same-PV-method 2023 inputs")
    args = parser.parse_args()
    if args.harmonize_2023:
        harmonize_2023(args.offline)
        return
    RAW.mkdir(parents=True, exist_ok=True)
    preserve_pvgis_probes()
    frozen = freeze_reference_snapshots()
    configure_study(frozen)
    study.source_manifest(args.offline)
    if args.download_only:
        return
    study.prepare()
    metadata = DATA / "metadata.json"
    info = json.loads(metadata.read_text(encoding="utf-8"))
    info["frozen_reference_mapping"] = validate_reference_mapping()
    info["frozen_reference_snapshots"] = frozen
    info["pv_method"] = PV_NOTE
    info["reference_sources"] = [*info["reference_sources"][:-1], SOLAR_DOCS]
    study.write_json(metadata, info)


if __name__ == "__main__":
    main()

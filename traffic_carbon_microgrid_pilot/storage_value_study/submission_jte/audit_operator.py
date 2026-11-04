"""Check the frozen panels against original CPS files and write aggregate JSON.

The script requires openpyxl and an IANA timezone database (tzdata on Windows).
Original operator files are downloaded into memory and verified before parsing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, time, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from openpyxl import load_workbook


BASE = "https://chargeplacescotland.org/wp-content/uploads/"
SOURCES = {
    "2023_01": {
        "sessions": ("2023/07/JAN-23-Public-Sessions.xlsx", "ecb731e22e6ede7540415741726d42f38b18b31c458352c6e8066f537482fc75"),
        "points": ("2023/07/monthy_cp_report-JAN-23.csv", "ebd72d8e5eb807c726e03f577805b50afba2399be451f8047653a1bf210901d5"),
    },
    "2023_07": {
        "sessions": ("2023/08/SESSIONS-JULY-23.xlsx", "b9a7486d2e8bd8d0497e1f1cfa976025989464605b0314914f662554b0060448"),
        "points": ("2023/08/monthy_cp_report-JULY.xlsx", "0372fa5a8575c8040c27d7965cefe81f9f4fc36234820addadc6d8bb14c7eaa6"),
    },
    "2024_01": {
        "sessions": ("2024/03/Sessions-JAN-24-CLEAN-1.xlsx", "6c59858833105d3b7567e28f675282e11494617526bdb4f6468418ed7c3d8374"),
        "points": ("2024/03/CP_report-JAN-24-CLEAN-1.xlsx", "4db0a132f82e8f422bb0279f175607b8442e534e8f498b9866c17b5acdf25f86"),
    },
    "2024_07": {
        "sessions": ("2024/08/CPS-JULY-Sessions.xlsx", "a52ff0dcf045109dd79ab5d5114d23356e33a3f9cfe62b903006120b41b93379"),
        "points": ("2024/08/CPS_monthy_cp_report-July24.csv", "b28b252b7c7c08ebb0856f05bd6310799202e39c52d7a6d09f02054c2c61b316"),
    },
}
LONDON = ZoneInfo("Europe/London")


def verified_source(entry):
    relative, expected_hash = entry
    url = BASE + relative
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                raw = response.read()
            break
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Operator snapshot differs from its frozen SHA-256: {url}")
    return raw, {"url": url, "sha256": digest, "bytes": len(raw)}


def normalize_device(value):
    return re.sub(r"^APT(?=\d+$)", "", str(value or "").strip().upper())


def local_datetime(value):
    if isinstance(value, datetime):
        return value.replace(microsecond=0)
    for pattern in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(str(value).strip(), pattern)
        except ValueError:
            continue
    return None


def duration_seconds(value):
    if isinstance(value, timedelta):
        return round(value.total_seconds())
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, (int, float)):
        return round(value * 86400)
    match = re.fullmatch(r"(?:(\d+) days? )?(\d+):(\d\d):(\d\d)(?:\.\d+)?", str(value))
    if match:
        days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    return None


def panel_keys(root):
    selected = defaultdict(set)
    inputs = {}
    for relative in ("results/case_sessions.csv", "cross_year_2024/results/case_sessions.csv"):
        path = root / relative
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                month = "_".join(row["session_id"].split("_")[1:3])
                if month not in SOURCES:
                    raise ValueError(f"Unexpected case month: {month}")
                device, port = row["connector_id"].split(":")
                local = datetime.fromtimestamp(float(row["start"]), timezone.utc).astimezone(LONDON).replace(tzinfo=None)
                seconds = round(float(row["end"]) - float(row["start"]))
                key = (device, int(port), local, seconds, round(float(row["energy"]), 5))
                if key in selected[month]:
                    raise ValueError(f"Duplicate transaction key in panel month: {month}")
                selected[month].add(key)
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(selected) != set(SOURCES):
        raise ValueError("One or more study months are absent from the panel files")
    return selected, inputs


def point_rows(raw, relative):
    if relative.endswith(".csv"):
        yield from csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        return
    workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True, keep_links=False)
    try:
        rows = iter(workbook.active.values)
        header = next(rows)
        for row in rows:
            yield dict(zip(header, row))
    finally:
        workbook.close()


def session_matches(raw, selected, devices):
    workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True, keep_links=False)
    matched, connector_types = set(), Counter()
    row_count = 0
    try:
        rows = iter(workbook.active.values)
        header = next(rows)
        indexes = {str(field).strip().casefold(): i for i, field in enumerate(header)}

        def column(*alternatives):
            return next(indexes[label] for label in alternatives if label in indexes)

        device_col = column("cp id", "cpid")
        port_col = column("connector", "connector id")
        start_col = column("start", "start time")
        duration_col = column("duration")
        energy_col = column("consum(kwh)", "consum")
        type_col = column("connector type")
        for row in rows:
            row_count += 1
            device = normalize_device(row[device_col])
            if device not in devices:
                continue
            try:
                key = (
                    device, int(row[port_col]), local_datetime(row[start_col]),
                    duration_seconds(row[duration_col]), round(float(row[energy_col]), 5),
                )
            except (ValueError, TypeError):
                continue
            if key in selected and key not in matched:
                matched.add(key)
                connector_types[str(row[type_col])] += 1
    finally:
        workbook.close()
    return len(matched), row_count, dict(connector_types), list(header)


def audit_month(month, selected):
    devices = {key[0] for key in selected}
    files = SOURCES[month]
    points, point_source = verified_source(files["points"])
    point_signatures = defaultdict(set)
    for row in point_rows(points, files["points"][0]):
        device = normalize_device(row["Display ID"])
        if device in devices:
            fields = ("Com-ID", "Local authority area", "Postcode")
            point_signatures[device].add(tuple(str(row.get(field) or "").strip() for field in fields))
    sessions, session_source = verified_source(files["sessions"])
    matches, rows, types, header = session_matches(sessions, selected, devices)
    result = {
        "panel_sessions": len(selected), "matched_sessions": matches,
        "operator_workbook_rows_excluding_header": rows, "original_session_columns": header,
        "matched_original_connector_labels": types,
        "panel_devices": len(devices), "devices_in_operator_point_report": len(point_signatures),
        "sources": {"sessions": session_source, "points": point_source},
    }
    print(f"{month}: {matches}/{len(selected)} sessions and {len(point_signatures)}/{len(devices)} devices verified", flush=True)
    return result, point_signatures


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=here.parent)
    parser.add_argument("--output", type=Path, default=here / "operator_audit.json")
    args = parser.parse_args()
    panels, inputs = panel_keys(args.study_root.resolve())
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {month: pool.submit(audit_month, month, keys) for month, keys in panels.items()}
        audited = {month: future.result() for month, future in pending.items()}
    months = {month: result for month, (result, _) in audited.items()}
    cross_year = {}
    for month in ("01", "07"):
        first = audited[f"2023_{month}"][1]
        second = audited[f"2024_{month}"][1]
        common = first.keys() & second.keys()
        cross_year[month] = {
            "shared_devices_in_operator_reports": len(common),
            "same_communication_id_authority_postcode": sum(first[device] == second[device] for device in common),
        }
    total = sum(item["panel_sessions"] for item in months.values())
    matched = sum(item["matched_sessions"] for item in months.values())
    passed = (
        matched == total
        and all(item["devices_in_operator_point_report"] == item["panel_devices"] for item in months.values())
        and all(item["shared_devices_in_operator_reports"] == item["same_communication_id_authority_postcode"] for item in cross_year.values())
    )
    report = {
        "pass": passed, "panel_sessions": total, "matched_sessions": matched,
        "months": months, "cross_year_same_month_device_identity": cross_year,
        "input_panel_sha256": inputs,
        "session_key_fields": ["device", "connector", "Europe/London local start", "duration (seconds)", "delivered energy (kWh, rounded to five decimals)"],
        "scope": "Public transaction fields and historical identifier continuity. The operator files do not separately identify unplug events, power histories or electrical topology.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{matched}/{total} total session matches. Aggregate audit written to {args.output.name}.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Recreate the session indexes omitted from the shareable archive."""
from dataclasses import asdict
import hashlib
import json
from shutil import copyfile

from run_service import HOLDOUT, PROTOCOL, load_cases, read_csv, sha256, write_csv


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for year, source, directory in ((2023, "data_harmonized_2023", "results_harmonized_2023"),
                                    (2024, "data", "results")):
        prepared = HOLDOUT / source
        output = HOLDOUT / directory
        expected = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for relative, digest in expected["input_sha256"].items():
            path = prepared / relative
            if sha256(path) != digest:
                lf = path.read_bytes().replace(b"\r\n", b"\n")
                variants = {hashlib.sha256(body).hexdigest() for body in (lf, lf.replace(b"\n", b"\r\n"))}
                if digest not in variants:
                    raise ValueError(f"Prepared input differs from the frozen panel: {path}")
        _, cases, _ = load_cases(prepared, protocol, year)
        if {case["case_id"] for case, *_ in cases} != {row["case_id"] for row in read_csv(output / "cases.csv")}:
            raise ValueError(f"The case cohort differs from the frozen {year} panel")
        write_csv(output / "case_sessions.csv",
                  ({"case_id": case["case_id"], **asdict(s)} for case, sessions, *_ in cases for s in sessions))
        if year == 2023:
            copyfile(output / "case_sessions.csv", HOLDOUT.parent / "results" / "case_sessions.csv")
        print(f"{year} {len(cases)} site-weeks; {sum(len(s) for _, s, *_ in cases)} sessions")


if __name__ == "__main__":
    main()

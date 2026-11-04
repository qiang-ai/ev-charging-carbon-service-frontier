"""Package the working manuscript and reproducible model without raw charging sessions."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
PROJECT = STUDY.parent
WORKSPACE = PROJECT.parent
FIGURE_NUMBERS = range(1, 11)


def paths():
    for name in ("requirements.txt", "optimize.py"):
        yield PROJECT / name
    for name in ("model.py", "prepare_study_data.py", "run_study.py", "RUNNING.txt"):
        yield STUDY / name
    for name in ("protocol.json", "prepare_holdout_data.py", "run_holdout.py", "analyze_holdout.py",
                 "audit_holdout.py", "RUNNING.txt"):
        yield STUDY / "cross_year_2024" / name
    for folder in (STUDY / "data", STUDY / "cross_year_2024" / "data",
                   STUDY / "cross_year_2024" / "data_harmonized_2023"):
        yield folder / "raw_manifest.json"
        for source in sorted((folder / "raw").iterdir()):
            if source.suffix == ".json" and not (folder.name == "data" and "_2024" in source.name
                                                  and source.name.startswith("pvgis_")):
                yield source
            elif source.name == "gridcharge_README.md" or source.name.startswith("infrastructure_"):
                yield source
    for relative in ("results", "cross_year_2024/results", "cross_year_2024/results_harmonized_2023"):
        for name in ("cases.csv", "outcomes.csv", "manifest.json"):
            yield STUDY / relative / name
    yield STUDY / "results_small_storage" / "outcomes.csv"
    for name in ("primary_comparison.csv", "matched_interaction_changes.csv", "week_ordinal_matches.csv",
                 "holdout_summary.json"):
        yield STUDY / "cross_year_2024" / "analysis_common_pv" / name
    for suffix in ("*.py", "*.txt", "*.json", "*.docx"):
        yield from sorted(p for p in ROOT.glob(suffix)
                          if not (p.name.startswith("literature_") and p.name != "literature_evidence.json"))
    for name in ("TEMPLATE_SOURCE.txt", "ascexmpl-new.tex", "ascelike-new.cls", "ascelike-new.bst",
                 "JTE_Part_A_Technical_Paper.tex", "JTE_Part_A_Technical_Paper.pdf"):
        yield ROOT / "asce" / name
    for number in FIGURE_NUMBERS:
        for suffix in ("tif", "svg", "png", "pdf"):
            yield ROOT / "figures" / f"Fig_{number}.{suffix}"
    yield ROOT / "figures" / "ALT_TEXT.txt"
    yield from sorted(p for p in (ROOT / "results").iterdir() if p.is_file())
    yield from sorted((ROOT / "results" / "primary_solutions").glob("*.npz"))
    for directory in ("controllers", "signal_diagnostic"):
        folder = ROOT / "results" / directory
        yield from sorted(p for p in folder.iterdir() if p.is_file())
        yield from sorted((folder / "traces").glob("*.npz"))
    for directory in ("online_forecast", "proxy_window_sensitivity", "m2_regional"):
        folder = ROOT / "results" / directory
        yield from sorted(p for p in folder.rglob("*") if p.is_file()
                          and p.name != "session_windows.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproducibility-only", action="store_true",
                        help="Create a separate archive without manuscript and peer-review correspondence")
    args = parser.parse_args()
    name = "JTE_Reproducibility_v9.zip" if args.reproducibility_only else "JTE_Part_A_Submission_Materials_v9.zip"
    output = WORKSPACE / name
    temporary = output.with_suffix(".part")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=7) as archive:
        hashes = []
        for path in paths():
            relative = path.relative_to(WORKSPACE)
            if args.reproducibility_only and (path.suffix == ".docx" or "asce" in relative.parts
                                              or path.name == "build_revision_documents.py"):
                continue
            archive.write(path, relative)
            with path.open("rb") as stream:
                hashes.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {relative.as_posix()}")
        archive.writestr("SHA256SUMS.txt", "\n".join(hashes) + "\n")
    temporary.replace(output)
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("An archive member failed its integrity check")
        print(f"{len(archive.namelist())} files; {output.stat().st_size:,} bytes; {output}")
if __name__ == "__main__":
    main()

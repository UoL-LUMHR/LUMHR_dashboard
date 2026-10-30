"""Split the February 2026 UK postcode-to-LSOA lookup into area files.

English records are assigned to their 2022 region using the small official ONS
LAD-to-region lookup stored beside the outputs. Other UK nations and Crown
Dependency pseudo-LAD rows are written to their own files.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "scripts/utils/source_data/Postcode_to_OA_(2021)_to_LSOA_to_MSOA_to_LAD_(February_2026)_Best_Fit_Lookup_in_the_UK/PCD_OA21_LSOA21_MSOA21_LAD_FEB26_UK_LU.csv"
DEFAULT_OUTPUT = ROOT / "datasets/Postcode_to_OA_(2021)_to_LSOA_to_MSOA_to_LAD_(February_2026)_Best_Fit_Lookup_in_the_UK"
REGION_LOOKUP = "LAD22_to_RGN22_England_December_2022.csv"
RECENT_ENGLISH_LAD_REGIONS = {
    # These six LAD codes were introduced in 2023, after the crosswalk vintage.
    "E06000063": "North West",  # Cumberland
    "E06000064": "North West",  # Westmorland and Furness
    "E06000065": "Yorkshire and The Humber",  # North Yorkshire
    "E06000066": "South West",  # Somerset
    "E08000038": "Yorkshire and The Humber",  # Barnsley
    "E08000039": "Yorkshire and The Humber",  # Sheffield
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _load_lad_regions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        regions = {
            row["LAD22CD"].strip(): row["RGN22NM"].strip()
            for row in csv.DictReader(source)
            if row.get("LAD22CD", "").strip() and row.get("RGN22NM", "").strip()
        }
    regions.update(RECENT_ENGLISH_LAD_REGIONS)
    return regions


def _area_key(lad_code: str, regions: dict[str, str]) -> str:
    code = lad_code.strip().upper()
    if code.startswith("E"):
        region = regions.get(code)
        return f"england_{_slug(region)}" if region else "england_unmapped_region"
    if code.startswith("S"):
        return "scotland"
    if code.startswith("W"):
        return "wales"
    if code.startswith("N"):
        return "northern_ireland"
    if code == "L99999999":
        return "channel_islands"
    if code == "M99999999":
        return "isle_of_man"
    if not code:
        return "unassigned_lad"
    return "other_unmapped_area"


def split_lookup(source_path: Path, output_dir: Path, region_lookup_path: Path) -> Counter[str]:
    regions = _load_lad_regions(region_lookup_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous_part in output_dir.glob("postcode_to_lsoa_2021_*.csv"):
        previous_part.unlink()
    counts: Counter[str] = Counter()
    handles: dict[str, tuple[object, csv.DictWriter]] = {}

    try:
        with source_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "ladcd" not in reader.fieldnames:
                raise ValueError("The input CSV must include a 'ladcd' column.")

            for row in reader:
                area = _area_key(row.get("ladcd", ""), regions)
                if area not in handles:
                    target = output_dir / f"postcode_to_lsoa_2021_{area}.csv"
                    handle = target.open("w", encoding="utf-8", newline="")
                    writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    handles[area] = (handle, writer)
                handles[area][1].writerow(row)
                counts[area] += 1
    finally:
        for handle, _writer in handles.values():
            handle.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--region-lookup",
        type=Path,
        help="ONS LAD-to-region CSV (defaults to the copy in the output directory).",
    )
    args = parser.parse_args()
    region_lookup = args.region_lookup or (args.output_dir / REGION_LOOKUP)
    counts = split_lookup(args.input, args.output_dir, region_lookup)
    for area, rows in sorted(counts.items()):
        output = args.output_dir / f"postcode_to_lsoa_2021_{area}.csv"
        print(f"{output.name}: {rows:,} rows, {output.stat().st_size / (1024 * 1024):.1f} MiB")
    print(f"Total rows: {sum(counts.values()):,}")


if __name__ == "__main__":
    main()

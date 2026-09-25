"""Filter Census 2021 housing-tenure and bedroom-occupancy CSVs to Lincolnshire."""

from __future__ import annotations

import csv
from pathlib import Path

from _lincolnshire_xlsx import lincolnshire_lsoa_codes, write_csv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

GEOJSON_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "lincolnshire_lsoa"
    / "lower-super-output-areas-2021-5RrVTw.geojson"
)
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "revekland"

DATASETS = (
    {
        "name": "TS054 tenure",
        "source": (
            BASE_DIR
            / "source_data"
            / "TS054_tenure"
            / "TS054-2021-4-filtered-2026-09-24T09_58_45Z.csv"
        ),
        "output": OUTPUT_DIR / "lincolnshire_ts054_tenure.csv",
        "expected_categories": 9,
    },
    {
        "name": "TS052 occupancy rating for bedrooms",
        "source": (
            BASE_DIR
            / "source_data"
            / "TS052_occupancy_rating_for_bedrooms"
            / "TS052-2021-5-filtered-2026-09-24T09_59_23Z.csv"
        ),
        "output": OUTPUT_DIR / "lincolnshire_ts052_occupancy_rating_for_bedrooms.csv",
        "expected_categories": 6,
    },
)

EXPECTED_LSOA_COUNT = 435


def filter_csv(source_path: Path, output_path: Path, lsoa_codes: set[str], expected_categories: int) -> int:
    """Write source rows whose LSOA code is in ``lsoa_codes``."""

    with source_path.open(encoding="utf-8", newline="") as source_file:
        reader = csv.reader(source_file)
        try:
            headers = next(reader)
        except StopIteration as error:
            raise ValueError(f"Source CSV is empty: {source_path}") from error

        if not headers or headers[0] != "Lower layer Super Output Areas Code":
            raise ValueError(
                f"Unexpected LSOA-code column in {source_path}: {headers[:1]}"
            )

        rows: list[list[str]] = []
        matched_codes: set[str] = set()
        category_codes: set[str] = set()

        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError(
                    f"Row has {len(row)} fields but expected {len(headers)} in {source_path}"
                )

            lsoa_code = row[0].strip()
            if lsoa_code not in lsoa_codes:
                continue

            rows.append(row)
            matched_codes.add(lsoa_code)
            category_codes.add(row[2].strip())

    if matched_codes != lsoa_codes:
        missing = sorted(lsoa_codes - matched_codes)
        raise ValueError(
            f"{source_path.name}: missing {len(missing)} Lincolnshire LSOAs: {missing[:5]}"
        )

    if len(category_codes) != expected_categories:
        raise ValueError(
            f"{source_path.name}: expected {expected_categories} categories, "
            f"found {len(category_codes)}"
        )

    expected_rows = len(lsoa_codes) * expected_categories
    if len(rows) != expected_rows:
        raise ValueError(
            f"{source_path.name}: expected {expected_rows:,} rows, found {len(rows):,}"
        )

    write_csv(output_path, headers, rows)
    return len(rows)


def main() -> None:
    lsoa_codes = lincolnshire_lsoa_codes(GEOJSON_PATH)
    if len(lsoa_codes) != EXPECTED_LSOA_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LSOA_COUNT} Lincolnshire LSOAs, found {len(lsoa_codes)}"
        )

    print(f"Loaded {len(lsoa_codes):,} Lincolnshire LSOA codes")
    for dataset in DATASETS:
        row_count = filter_csv(
            source_path=dataset["source"],
            output_path=dataset["output"],
            lsoa_codes=lsoa_codes,
            expected_categories=dataset["expected_categories"],
        )
        print(f"{dataset['name']}: wrote {row_count:,} rows to {dataset['output']}")


if __name__ == "__main__":
    main()

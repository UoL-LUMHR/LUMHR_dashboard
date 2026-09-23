"""Extract Lincolnshire LSOAs from the 2024 fuel-poverty source workbook."""

from pathlib import Path

from _lincolnshire_xlsx import (
    extract_lsoa_rows,
    lincolnshire_lsoa_codes,
    write_csv,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

GEOJSON_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "lincolnshire_lsoa"
    / "lower-super-output-areas-2021-5RrVTw.geojson"
)
SOURCE_WORKBOOK = (
    BASE_DIR
    / "source_data"
    / "fuel_poverty"
    / "2024"
    / "fuel-poverty-sub-regional-2026-2024-data-tables.xlsx"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "fuel_poverty"
    / "2024"
    / "lincolnshire_fuel_poverty_2024.csv"
)

EXPECTED_LSOA_COUNT = 435


def extract_fuel_poverty() -> None:
    lsoa_codes = lincolnshire_lsoa_codes(GEOJSON_PATH)
    if len(lsoa_codes) != EXPECTED_LSOA_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LSOA_COUNT} Lincolnshire LSOAs, found {len(lsoa_codes)}"
        )

    headers, rows, matched_codes = extract_lsoa_rows(
        workbook_path=SOURCE_WORKBOOK,
        sheet_name="Table 4",
        header_row_number=3,
        lsoa_column="A",
        lsoa_codes=lsoa_codes,
    )

    if matched_codes != lsoa_codes:
        missing = sorted(lsoa_codes - matched_codes)
        raise ValueError(
            f"Expected all {len(lsoa_codes)} LSOAs, but {len(missing)} were missing: "
            f"{missing[:5]}"
        )

    write_csv(OUTPUT_PATH, headers, rows)
    print(f"Wrote {len(rows):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_fuel_poverty()

"""Extract Lincolnshire LSOAs from the gas-network source workbook."""

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
    / "properties_not_connected_to_gas_network"
    / "LSOA_estimates_of_properties_not_connected_to_the_gas_network_2015-2024.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "properties_not_connected_to_gas_network"

YEARS = range(2015, 2025)
EXPECTED_LSOA_COUNT = 435

def convert_stored_percentages_to_percentage_points(
    headers: list[str], rows: list[list[str]]
) -> None:
    """Convert Excel percentage fractions such as 0.062 to 6.2 percentage points."""

    percentage_index = next(
        (
            index
            for index, header in enumerate(headers)
            if "Estimated percentage" in header
            and "properties not" in header
            and "gas grid" in header
        ),
        None,
    )
    if percentage_index is None:
        raise ValueError("Could not find the gas-grid percentage column")

    for row in rows:
        value = row[percentage_index].strip()
        if value:
            row[percentage_index] = format(float(value) * 100.0, ".15g")


def extract_properties() -> None:
    lsoa_codes = lincolnshire_lsoa_codes(GEOJSON_PATH)
    if len(lsoa_codes) != EXPECTED_LSOA_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LSOA_COUNT} Lincolnshire LSOAs, found {len(lsoa_codes)}"
        )

    print(f"Loaded {len(lsoa_codes):,} Lincolnshire LSOA codes from {GEOJSON_PATH}")

    for year in YEARS:
        headers, rows, matched_codes = extract_lsoa_rows(
            workbook_path=SOURCE_WORKBOOK,
            sheet_name=str(year),
            header_row_number=4,
            lsoa_column="E",
            lsoa_codes=lsoa_codes,
        )

        if matched_codes != lsoa_codes:
            missing = sorted(lsoa_codes - matched_codes)
            raise ValueError(
                f"{year}: expected all {len(lsoa_codes)} LSOAs, "
                f"but {len(missing)} were missing: {missing[:5]}"
            )

        convert_stored_percentages_to_percentage_points(headers, rows)

        output_path = (
            OUTPUT_DIR
            / f"lincolnshire_properties_not_connected_to_gas_network_{year}.csv"
        )
        write_csv(output_path, headers, rows)
        print(f"{year}: wrote {len(rows):,} rows to {output_path}")


if __name__ == "__main__":
    extract_properties()

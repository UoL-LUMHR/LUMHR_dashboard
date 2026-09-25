"""Load and derive non-SAMHI features used by the SAMHI models.

The source files are mostly 2021 Census snapshots.  Gas-grid coverage is the
exception: it is available annually and is kept wide by year so callers can
select the latest value available before a target year.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from openpyxl import load_workbook


HOUSEHOLD_FEATURE_COLUMNS = [
    "one_person_household_pct",
    "one_person_aged_66_plus_pct",
    "one_person_other_pct",
    "pensioner_couple_household_pct",
    "married_no_children_pct",
    "married_dependent_children_pct",
    "cohabiting_household_pct",
    "cohabiting_dependent_children_pct",
    "lone_parent_pct",
    "lone_parent_dependent_children_pct",
    "other_household_types_pct",
    "other_households_dependent_children_pct",
]

TENURE_FEATURE_COLUMNS = [
    "owner_occupied_pct",
    "shared_ownership_pct",
    "social_rented_pct",
    "private_rented_pct",
    "rent_free_pct",
]

OCCUPANCY_FEATURE_COLUMNS = [
    "underoccupied_2plus_bedrooms_pct",
    "underoccupied_1_bedroom_pct",
    "occupancy_balanced_pct",
    "overcrowded_1_bedroom_pct",
    "overcrowded_2plus_bedrooms_pct",
    "overcrowded_bedrooms_pct",
]

ADDED_FEATURE_COLUMNS = [
    "fuel_poverty_pct",
    "gas_grid_disconnection_pct",
    *TENURE_FEATURE_COLUMNS,
    *OCCUPANCY_FEATURE_COLUMNS,
    *HOUSEHOLD_FEATURE_COLUMNS,
]


def _clean_code(series: pd.Series) -> pd.Series:
    """Normalise LSOA codes from Census exports and ordinary CSVs."""
    return (
        series.astype(str)
        .str.split(":")
        .str[0]
        .str.strip()
    )


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _empty_frame(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=["LSOA21CD", *columns])


def load_household_features(path: Path) -> pd.DataFrame:
    """Return detailed TS003 household percentages for each 2021 LSOA."""
    if not path.exists():
        return _empty_frame(HOUSEHOLD_FEATURE_COLUMNS)

    try:
        if "lincolnshire" in path.name.lower():
            raw = pd.read_csv(path)
            code_col = "LSOA_Code"
            source_columns = {
                "one_person_household_pct": "One-person household_Pct",
                "one_person_aged_66_plus_pct": "One-person household: Aged 66 years and over_Pct",
                "one_person_other_pct": "One-person household: Other_Pct",
                "pensioner_couple_household_pct": "Single family household: All aged 66 years and over_Pct",
                "married_no_children_pct": "Single family household: Married or civil partnership couple: No children_Pct",
                "married_dependent_children_pct": "Single family household: Married or civil partnership couple: Dependent children_Pct",
                "cohabiting_household_pct": "Single family household: Cohabiting couple family_Pct",
                "cohabiting_dependent_children_pct": "Single family household: Cohabiting couple family: With dependent children_Pct",
                "lone_parent_pct": "Single family household: Lone parent family_Pct",
                "lone_parent_dependent_children_pct": "Single family household: Lone parent family: With dependent children_Pct",
                "other_household_types_pct": "Other household types_Pct",
                "other_households_dependent_children_pct": "Other household types: With dependent children_Pct",
            }
        else:
            # Nomis Census export: first seven rows are metadata.
            raw = pd.read_csv(path, skiprows=7)
            code_col = "2021 super output area - lower layer"
            source_columns = {
                "one_person_household_pct": "%.1",
                "one_person_aged_66_plus_pct": "%.2",
                "one_person_other_pct": "%.3",
                "pensioner_couple_household_pct": "%.5",
                "married_no_children_pct": "%.7",
                "married_dependent_children_pct": "%.8",
                "cohabiting_household_pct": "%.10",
                "cohabiting_dependent_children_pct": "%.12",
                "lone_parent_pct": "%.14",
                "lone_parent_dependent_children_pct": "%.15",
                "other_household_types_pct": "%.19",
                "other_households_dependent_children_pct": "%.20",
            }

        out = pd.DataFrame({"LSOA21CD": _clean_code(raw[code_col])})
        for destination, source in source_columns.items():
            out[destination] = _numeric(raw[source]) if source in raw.columns else np.nan
        return out.groupby("LSOA21CD", as_index=False)[HOUSEHOLD_FEATURE_COLUMNS].mean()
    except Exception:
        return _empty_frame(HOUSEHOLD_FEATURE_COLUMNS)


def _pivot_census_categories(
    path: Path,
    code_column: str,
    category_column: str,
    observation_column: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    raw["LSOA21CD"] = _clean_code(raw[code_column])
    raw["category_code"] = pd.to_numeric(raw[category_column], errors="coerce")
    raw["observation"] = _numeric(raw[observation_column]).fillna(0.0)
    return raw.pivot_table(
        index="LSOA21CD",
        columns="category_code",
        values="observation",
        aggfunc="sum",
        fill_value=0.0,
    )


def load_tenure_features(path: Path) -> pd.DataFrame:
    """Convert TS054 tenure counts into household percentages."""
    try:
        pivot = _pivot_census_categories(
            path,
            "Lower layer Super Output Areas Code",
            "Tenure of household (9 categories) Code",
            "Observation",
        )
        if pivot.empty:
            return _empty_frame(TENURE_FEATURE_COLUMNS)
        valid_total = pivot[[c for c in range(8) if c in pivot]].sum(axis=1)
        out = pd.DataFrame(index=pivot.index)
        out["owner_occupied_pct"] = pivot.get(0, 0) + pivot.get(1, 0)
        out["shared_ownership_pct"] = pivot.get(2, 0)
        out["social_rented_pct"] = pivot.get(3, 0) + pivot.get(4, 0)
        out["private_rented_pct"] = pivot.get(5, 0) + pivot.get(6, 0)
        out["rent_free_pct"] = pivot.get(7, 0)
        out = out.div(valid_total.replace(0, np.nan), axis=0) * 100.0
        return out.reset_index()[["LSOA21CD", *TENURE_FEATURE_COLUMNS]]
    except Exception:
        return _empty_frame(TENURE_FEATURE_COLUMNS)


def load_occupancy_features(path: Path) -> pd.DataFrame:
    """Convert TS052 bedroom occupancy counts into percentages."""
    try:
        pivot = _pivot_census_categories(
            path,
            "Lower layer Super Output Areas Code",
            "Occupancy rating for bedrooms (6 categories) Code",
            "Observation",
        )
        if pivot.empty:
            return _empty_frame(OCCUPANCY_FEATURE_COLUMNS)
        valid_total = pivot[[c for c in range(1, 6) if c in pivot]].sum(axis=1)
        out = pd.DataFrame(index=pivot.index)
        out["underoccupied_2plus_bedrooms_pct"] = pivot.get(1, 0)
        out["underoccupied_1_bedroom_pct"] = pivot.get(2, 0)
        out["occupancy_balanced_pct"] = pivot.get(3, 0)
        out["overcrowded_1_bedroom_pct"] = pivot.get(4, 0)
        out["overcrowded_2plus_bedrooms_pct"] = pivot.get(5, 0)
        out["overcrowded_bedrooms_pct"] = pivot.get(4, 0) + pivot.get(5, 0)
        out = out.div(valid_total.replace(0, np.nan), axis=0) * 100.0
        return out.reset_index()[["LSOA21CD", *OCCUPANCY_FEATURE_COLUMNS]]
    except Exception:
        return _empty_frame(OCCUPANCY_FEATURE_COLUMNS)


def _read_excel_columns(path: Path, sheet_name: str, header_row: int, columns) -> pd.DataFrame:
    """Read selected columns from a large workbook in read-only mode."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name]
        rows = sheet.iter_rows(values_only=True)
        headers = None
        indexes = None
        values = {column: [] for column in columns}
        for row_number, row in enumerate(rows):
            if row_number == header_row:
                headers = [str(value).strip() if value is not None else "" for value in row]
                indexes = {column: headers.index(column) for column in columns if column in headers}
                continue
            if row_number <= header_row or indexes is None:
                continue
            if not any(index < len(row) and row[index] is not None for index in indexes.values()):
                continue
            for column in columns:
                index = indexes.get(column)
                values[column].append(row[index] if index is not None and index < len(row) else None)
        return pd.DataFrame(values)
    finally:
        workbook.close()


def load_fuel_poverty_features(path: Path) -> pd.DataFrame:
    """Load LSOA fuel-poverty percentages from CSV or the national workbook."""
    if not path.exists():
        return _empty_frame(["fuel_poverty_pct"])
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            raw = _read_excel_columns(
                path,
                "Table 4",
                2,
                ["LSOA Code", "Proportion of households fuel poor (%)"],
            )
        else:
            raw = pd.read_csv(path)
        return pd.DataFrame({
            "LSOA21CD": _clean_code(raw["LSOA Code"]),
            "fuel_poverty_pct": _numeric(raw["Proportion of households fuel poor (%)"]),
        }).groupby("LSOA21CD", as_index=False).mean()
    except Exception:
        return _empty_frame(["fuel_poverty_pct"])


def load_gas_grid_features(datasets_dir: Path) -> pd.DataFrame:
    """Return one gas-grid-disconnection column per source year.

    The national workbook stores proportions (for example 0.0141), whereas
    the generated Lincolnshire CSVs store percentage points (1.41).
    """
    source_workbook = (
        datasets_dir.parent
        / "scripts"
        / "utils"
        / "source_data"
        / "properties_not_connected_to_gas_network"
        / "LSOA_estimates_of_properties_not_connected_to_the_gas_network_2015-2024.xlsx"
    )
    frames = []
    if source_workbook.exists():
        workbook = load_workbook(source_workbook, read_only=True, data_only=True)
        try:
            for year in range(2015, 2025):
                try:
                    sheet = workbook[str(year)]
                    rows = sheet.iter_rows(values_only=True)
                    headers = None
                    code_values = []
                    gas_values = []
                    for row_number, row in enumerate(rows):
                        if row_number == 3:
                            headers = [str(value).strip() if value is not None else "" for value in row]
                            code_index = headers.index("LSOA code")
                            gas_index = headers.index("Estimated percentage\nof properties not \non the gas grid")
                            continue
                        if row_number <= 3 or len(row) <= max(code_index, gas_index):
                            continue
                        code_values.append(row[code_index])
                        gas_values.append(row[gas_index])
                    values = _numeric(pd.Series(gas_values))
                    # Workbook values are proportions; convert to percentage points.
                    if values.dropna().median() <= 1.0:
                        values = values * 100.0
                    frames.append(pd.DataFrame({
                        "LSOA21CD": _clean_code(pd.Series(code_values)),
                        f"gas_grid_disconnection_pct_{year}": values,
                    }))
                except Exception:
                    continue
        finally:
            workbook.close()
    else:
        gas_dir = datasets_dir / "properties_not_connected_to_gas_network"
        for path in sorted(gas_dir.glob("lincolnshire_properties_not_connected_to_gas_network_*.csv")):
            try:
                year = path.stem.rsplit("_", 1)[-1]
                raw = pd.read_csv(path)
                frames.append(pd.DataFrame({
                    "LSOA21CD": _clean_code(raw["LSOA code"]),
                    f"gas_grid_disconnection_pct_{year}": _numeric(
                        raw["Estimated percentage\nof properties not \non the gas grid"]
                    ),
                }))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["LSOA21CD"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="LSOA21CD", how="outer")
    return out


def load_added_features(datasets_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all four requested feature groups in merge-ready form."""
    return {
        "fuel_poverty": load_fuel_poverty_features(
            datasets_dir.parent
            / "scripts"
            / "utils"
            / "source_data"
            / "fuel_poverty"
            / "2024"
            / "fuel-poverty-sub-regional-2026-2024-data-tables.xlsx"
            if (
                datasets_dir.parent
                / "scripts"
                / "utils"
                / "source_data"
                / "fuel_poverty"
                / "2024"
                / "fuel-poverty-sub-regional-2026-2024-data-tables.xlsx"
            ).exists()
            else datasets_dir / "fuel_poverty" / "2024" / "lincolnshire_fuel_poverty_2024.csv"
        ),
        "gas_grid": load_gas_grid_features(datasets_dir),
        "tenure": load_tenure_features(
            datasets_dir / "TS054_tenure" / "TS054-2021-4-filtered-2026-09-24T09_58_45Z.csv"
        ),
        "occupancy": load_occupancy_features(
            datasets_dir / "TS052_occupancy_rating_for_bedrooms" / "TS052-2021-5-filtered-2026-09-24T09_59_23Z.csv"
        ),
    }


def merge_added_features(master: pd.DataFrame, added: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge added features into a frame keyed by ``LSOA21CD``."""
    for key in ("fuel_poverty", "gas_grid", "tenure", "occupancy"):
        frame = added.get(key, pd.DataFrame())
        if not frame.empty:
            master = master.merge(frame, on="LSOA21CD", how="left")
    return master


def lagged_gas_feature(df: pd.DataFrame, target_year: int) -> np.ndarray:
    """Select gas-grid coverage from the latest year before target_year."""
    source_col = f"gas_grid_disconnection_pct_{target_year - 1}"
    if source_col in df.columns:
        return df[source_col].to_numpy()
    return np.full(len(df), np.nan)

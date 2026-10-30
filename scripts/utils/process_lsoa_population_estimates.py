"""Process ONS Small Area Population Estimates (Mid-2024 SAPE) for Lincolnshire LSOAs.

This script:
1. Loads sapelsoasyoa20222024.xlsx (Sheet: Mid-2024 LSOA 2021).
2. Filters to the 435 Lincolnshire 2021 LSOAs from the official boundary GeoJSON.
3. Aggregates single-year-of-age columns into demographic brackets (0-17, 18-64, 65+, 18+ adult).
4. Joins July 2026 GP registration counts to calculate registration rates, registration gap, and list inflation.
5. Saves clean CSV datasets to datasets/population_estimates/ and scripts/utils/.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def process_population_estimates(base_dir: Path) -> pd.DataFrame:
    # 1. Locate source population Excel file
    excel_candidates = list(base_dir.glob("**/sapelsoasyoa20222024.xlsx"))
    if not excel_candidates:
        raise FileNotFoundError("Could not find sapelsoasyoa20222024.xlsx in source directory.")
    excel_path = excel_candidates[0]
    print(f"Loading population estimates from: {excel_path}")

    # Read Mid-2024 sheet
    pop_df = pd.read_excel(excel_path, sheet_name="Mid-2024 LSOA 2021", skiprows=3, engine="calamine")

    # 2. Load Lincolnshire 2021 LSOA codes
    geojson_path = base_dir / "datasets" / "lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson"
    if not geojson_path.exists():
        geojson_path = base_dir.parent / "datasets" / "lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson"

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    lsoa_codes = [feat["properties"]["CODE"] for feat in gj["features"]]
    print(f"Target Lincolnshire LSOA count: {len(lsoa_codes)}")

    # 3. Filter to Lincolnshire LSOAs
    pop_lincs = pop_df[pop_df["LSOA 2021 Code"].isin(lsoa_codes)].copy()
    if len(pop_lincs) != len(lsoa_codes):
        print(f"Warning: matched {len(pop_lincs)} of {len(lsoa_codes)} LSOAs.")

    # 4. Identify age columns (F0..F90+, M0..M90+)
    f_cols = [c for c in pop_lincs.columns if c.startswith("F") and c[1:].isdigit()]
    m_cols = [c for c in pop_lincs.columns if c.startswith("M") and c[1:].isdigit()]

    f_0to17 = [c for c in f_cols if int(c[1:]) < 18]
    m_0to17 = [c for c in m_cols if int(c[1:]) < 18]

    f_18to64 = [c for c in f_cols if 18 <= int(c[1:]) < 65]
    m_18to64 = [c for c in m_cols if 18 <= int(c[1:]) < 65]

    f_65plus = [c for c in f_cols if int(c[1:]) >= 65]
    m_65plus = [c for c in m_cols if int(c[1:]) >= 65]

    f_18plus = [c for c in f_cols if int(c[1:]) >= 18]
    m_18plus = [c for c in m_cols if int(c[1:]) >= 18]

    pop_summary = pd.DataFrame(
        {
            "LSOA_CODE": pop_lincs["LSOA 2021 Code"].astype(str),
            "LSOA_NAME": pop_lincs["LSOA 2021 Name"].astype(str),
            "LAD_CODE": pop_lincs["LAD 2023 Code"].astype(str),
            "LAD_NAME": pop_lincs["LAD 2023 Name"].astype(str),
            "ONS_Pop_Total_2024": pd.to_numeric(pop_lincs["Total"], errors="coerce").fillna(0).astype(int),
            "ONS_Pop_0to17": pop_lincs[f_0to17 + m_0to17].sum(axis=1).astype(int),
            "ONS_Pop_18to64": pop_lincs[f_18to64 + m_18to64].sum(axis=1).astype(int),
            "ONS_Pop_65plus": pop_lincs[f_65plus + m_65plus].sum(axis=1).astype(int),
            "ONS_Pop_18plus": pop_lincs[f_18plus + m_18plus].sum(axis=1).astype(int),
        }
    )

    # Compute percentages
    pop_summary["Pct_18plus"] = (
        pop_summary["ONS_Pop_18plus"] / pop_summary["ONS_Pop_Total_2024"].replace(0, 1)
    ) * 100.0
    pop_summary["Pct_65plus"] = (
        pop_summary["ONS_Pop_65plus"] / pop_summary["ONS_Pop_Total_2024"].replace(0, 1)
    ) * 100.0
    pop_summary["Pct_0to17"] = (
        pop_summary["ONS_Pop_0to17"] / pop_summary["ONS_Pop_Total_2024"].replace(0, 1)
    ) * 100.0
    pop_summary["Pct_18to64"] = (
        pop_summary["ONS_Pop_18to64"] / pop_summary["ONS_Pop_Total_2024"].replace(0, 1)
    ) * 100.0

    # 5. Load GP patient registration data (July 2026)
    datasets_gp_dir = base_dir / "datasets" / "patients_registered_gp_practice" / "july_2026"
    if not datasets_gp_dir.exists():
        datasets_gp_dir = base_dir.parent / "datasets" / "patients_registered_gp_practice" / "july_2026"

    male_csv = datasets_gp_dir / "gp-reg-pat-prac-lsoa-male.csv"
    female_csv = datasets_gp_dir / "gp-reg-pat-prac-lsoa-female.csv"

    if male_csv.exists() and female_csv.exists():
        gp_m = pd.read_csv(male_csv)
        gp_f = pd.read_csv(female_csv)
        gp_all = pd.concat([gp_m, gp_f], ignore_index=True)
        gp_lsoa = gp_all.groupby("LSOA_CODE")["NUMBER_OF_PATIENTS"].sum().reset_index()
        gp_lsoa.rename(columns={"NUMBER_OF_PATIENTS": "GP_Registered_Patients"}, inplace=True)
    else:
        gp_lsoa = pd.DataFrame(columns=["LSOA_CODE", "GP_Registered_Patients"])

    merged = pop_summary.merge(gp_lsoa, on="LSOA_CODE", how="left")
    merged["GP_Registered_Patients"] = merged["GP_Registered_Patients"].fillna(0).astype(int)

    # 6. Compute registration rate and gap/inflation
    merged["GP_Registration_Rate_Pct"] = (
        merged["GP_Registered_Patients"] / merged["ONS_Pop_Total_2024"].replace(0, 1)
    ) * 100.0
    merged["Registration_Gap_Est"] = (
        merged["ONS_Pop_Total_2024"] - merged["GP_Registered_Patients"]
    ).clip(lower=0).astype(int)
    merged["List_Inflation_Est"] = (
        merged["GP_Registered_Patients"] - merged["ONS_Pop_Total_2024"]
    ).clip(lower=0).astype(int)

    # Ensure all 435 LSOAs are preserved
    ordered = pd.DataFrame({"LSOA_CODE": lsoa_codes}).merge(merged, on="LSOA_CODE", how="left")
    return ordered


def main():
    root = Path(__file__).resolve().parent.parent.parent
    output_df = process_population_estimates(root)

    print(f"\nProcessed {len(output_df)} LSOAs successfully.")
    print(f"Total ONS Population 2024: {output_df['ONS_Pop_Total_2024'].sum():,}")
    print(f"Total Adults 18+: {output_df['ONS_Pop_18plus'].sum():,} ({output_df['ONS_Pop_18plus'].sum()/output_df['ONS_Pop_Total_2024'].sum()*100:.1f}%)")
    print(f"Total Older People 65+: {output_df['ONS_Pop_65plus'].sum():,} ({output_df['ONS_Pop_65plus'].sum()/output_df['ONS_Pop_Total_2024'].sum()*100:.1f}%)")
    print(f"Total GP Registered Patients: {output_df['GP_Registered_Patients'].sum():,}")

    # Output paths
    out_dir_datasets = root / "datasets" / "population_estimates"
    out_dir_datasets.mkdir(parents=True, exist_ok=True)
    out_csv_datasets = out_dir_datasets / "lincolnshire_lsoa_population_estimates_2024.csv"
    output_df.to_csv(out_csv_datasets, index=False)
    print(f"Saved: {out_csv_datasets}")

    out_csv_utils = root / "scripts" / "utils" / "lincolnshire_lsoa_population_estimates_2024.csv"
    output_df.to_csv(out_csv_utils, index=False)
    print(f"Saved: {out_csv_utils}")


if __name__ == "__main__":
    main()


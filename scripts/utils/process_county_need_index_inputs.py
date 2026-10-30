"""Prepare current Need Index source files for Lincolnshire and neighbouring counties.

    Practice QOF and registration files include active ODS GP practices whose
    postcodes fall within each 2021 LSOA boundary and every practice with patients
    registered in an LSOA in that boundary. Registration mappings retain each
    selected practice's full patient list. May prescribing data can be sourced
    from adjacent area exports for cross-boundary practices. Parkfields Surgery
    (Y05733) uses January 2026 registrations for Derbyshire scopes.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd


SCOPES = {
    "lincolnshire": "lower-super-output-areas-2021-5RrVTw.geojson",
    "north_lincolnshire": "north_lincolnshire_lsoa_2021.geojson",
    "north_east_lincolnshire": "north_east_lincolnshire_lsoa_2021.geojson",
    "greater_lincolnshire": "greater_lincolnshire_lsoa_2021.geojson",
    "nottinghamshire": "nottinghamshire_lsoa_2021.geojson",
    "derbyshire": "derbyshire_lsoa_2021.geojson",
    "lincolnshire_nottinghamshire_derbyshire": "lincolnshire_nottinghamshire_derbyshire_lsoa_2021.geojson",
}
SCOPE_COMPONENTS = {
    "greater_lincolnshire": ["lincolnshire", "north_lincolnshire", "north_east_lincolnshire"],
    "lincolnshire_nottinghamshire_derbyshire": ["lincolnshire", "nottinghamshire", "derbyshire"],
}
PRESCRIBING_SOURCE_SCOPES = {
    "lincolnshire": ["north_lincolnshire", "north_east_lincolnshire", "lincolnshire"],
    "north_lincolnshire": ["lincolnshire", "north_east_lincolnshire", "north_lincolnshire"],
    "north_east_lincolnshire": ["lincolnshire", "north_lincolnshire", "north_east_lincolnshire"],
    "greater_lincolnshire": ["lincolnshire", "north_lincolnshire", "north_east_lincolnshire"],
    "nottinghamshire": ["bassetlaw", "nottinghamshire"],
    "derbyshire": [
        "lincolnshire", "north_lincolnshire", "north_east_lincolnshire",
        "bassetlaw", "nottinghamshire", "derbyshire",
    ],
    "lincolnshire_nottinghamshire_derbyshire": [
        "lincolnshire", "north_lincolnshire", "north_east_lincolnshire",
        "bassetlaw", "nottinghamshire", "derbyshire",
    ],
}
SPECIAL_ALLOCATION_PRACTICE_CODES = {"C84720"}  # NOTSPAR; retain marker, exclude from area aggregates.
QOF_WORKBOOK = Path(
    "scripts/utils/source_data/quality_outcomes_framework/2024-25/mental_health_neurology_group/"
    "qof-2425-prev-ach-pca-neu-prac.xlsx"
)
REGISTRATION_SOURCE = Path(
    "scripts/utils/source_data/patients_registered_gp_practice/england/2026/july/"
    "gp-reg-pat-prac-lsoa-all.csv"
)
PARKFIELDS_JANUARY_REGISTRATION_SOURCE = Path(
    "datasets/patients_registered_gp_practice/england/2026/january/"
    "gp-reg-pat-prac-lsoa-all.csv"
)
PARKFIELDS_PRACTICE_CODE = "Y05733"
ODS_GP_LOCATIONS = Path("datasets/gp_locations/GP_Locations_epraccur.csv")
POSTCODE_LSOA_LOOKUP_DIR = Path(
    "datasets/Postcode_to_OA_(2021)_to_LSOA_to_MSOA_to_LAD_"
    "(February_2026)_Best_Fit_Lookup_in_the_UK"
)
POPULATION_SOURCE = Path(
    "scripts/utils/source_data/lsoa_population_estimates/Mid-2022 revised (Nov 2025) to mid-2024/"
    "sapelsoasyoa20222024.xlsx"
)


def _boundary_codes(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        geojson = json.load(stream)
    codes = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        code = props.get("LSOA21CD") or props.get("CODE") or props.get("LSOA_CODE")
        if code:
            codes.append(str(code).strip())
    return codes


def _normalize_postcode(value: object) -> str:
    return "".join(str(value or "").upper().split())


def _load_ods_gp_practices(root: Path) -> pd.DataFrame:
    locations = pd.read_csv(
        root / ODS_GP_LOCATIONS,
        usecols=["OrganisationCode", "OrgName", "Postcode", "StatusName", "RoleID"],
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    ).fillna("")
    active_gp = locations["StatusName"].str.strip().str.upper().eq("ACTIVE") & locations[
        "RoleID"
    ].str.contains(r"(^|\|)RO76($|\|)", regex=True)
    locations = locations[active_gp].drop_duplicates("OrganisationCode", keep="last").copy()
    locations["postcode_key"] = locations["Postcode"].map(_normalize_postcode)
    wanted = set(locations.loc[locations["postcode_key"].ne(""), "postcode_key"])

    lookup_dir = root / POSTCODE_LSOA_LOOKUP_DIR
    lookup_parts = sorted(lookup_dir.glob("postcode_to_lsoa_2021_*.csv"))
    if not lookup_parts:
        raise FileNotFoundError(f"No split postcode-to-LSOA lookup CSVs found in {lookup_dir}")
    matches = []
    for lookup_path in lookup_parts:
        for chunk in pd.read_csv(
            lookup_path,
            usecols=["pcd7", "lsoa21cd", "dointr", "doterm"],
            dtype=str,
            chunksize=250_000,
            low_memory=False,
        ):
            chunk["postcode_key"] = chunk["pcd7"].map(_normalize_postcode)
            active_postcodes = chunk["doterm"].isna() | chunk["doterm"].astype(str).str.strip().eq("")
            selected = chunk[active_postcodes & chunk["postcode_key"].isin(wanted)]
            if not selected.empty:
                matches.append(selected[["postcode_key", "lsoa21cd", "dointr"]])

    postcode_lookup = (
        pd.concat(matches, ignore_index=True)
        if matches
        else pd.DataFrame(columns=["postcode_key", "lsoa21cd", "dointr"])
    )
    postcode_lookup["dointr"] = pd.to_numeric(postcode_lookup["dointr"], errors="coerce")
    postcode_lookup = postcode_lookup.sort_values("dointr").drop_duplicates("postcode_key", keep="last")
    postcode_to_lsoa = postcode_lookup.set_index("postcode_key")["lsoa21cd"].to_dict()
    locations["LSOA_CODE"] = locations["postcode_key"].map(postcode_to_lsoa).fillna("")
    return locations


def _load_parkfields_january_registration(root: Path) -> pd.DataFrame:
    """Load Parkfields' January 2026 practice-to-LSOA rows from the England file."""
    source = root / PARKFIELDS_JANUARY_REGISTRATION_SOURCE
    selected = []
    columns = [
        "PUBLICATION", "EXTRACT_DATE", "PRACTICE_CODE", "PRACTICE_NAME",
        "LSOA_CODE", "SEX", "NUMBER_OF_PATIENTS",
    ]
    for chunk in pd.read_csv(source, usecols=columns, dtype=str, chunksize=250_000, low_memory=False):
        chunk["PRACTICE_CODE"] = chunk["PRACTICE_CODE"].astype(str).str.strip()
        chunk["SEX"] = chunk["SEX"].astype(str).str.strip().str.upper()
        rows = chunk[
            chunk["PRACTICE_CODE"].eq(PARKFIELDS_PRACTICE_CODE)
            & chunk["SEX"].eq("ALL")
        ].copy()
        if not rows.empty:
            selected.append(rows)
    if not selected:
        raise ValueError(f"No January registration rows found for {PARKFIELDS_PRACTICE_CODE}")
    january = pd.concat(selected, ignore_index=True)
    january["LSOA_CODE"] = january["LSOA_CODE"].astype(str).str.strip()
    january["NUMBER_OF_PATIENTS"] = pd.to_numeric(january["NUMBER_OF_PATIENTS"], errors="coerce")
    january = january[
        january["LSOA_CODE"].str.fullmatch(r"E\d{8}")
        & january["NUMBER_OF_PATIENTS"].gt(0)
    ].copy()
    january["REGISTRATION_SOURCE_DATE"] = "2026-01-01"
    return january


def _extract_qof(root: Path, practices: set[str], scope: str) -> None:
    workbook = root / QOF_WORKBOOK
    dep = pd.read_excel(workbook, sheet_name="DEP", header=11)
    dep_out = pd.DataFrame(
        {
            "Practice Code": dep.iloc[:, 5],
            "Practice Name": dep.iloc[:, 6],
            "List size aged 18+ 2425": dep.iloc[:, 8],
            "Register 2425": dep.iloc[:, 9],
            "Prevalence (%) 2425": dep.iloc[:, 10],
            "Newly diagnosed 2425": dep.iloc[:, 24],
            "Overall PCA Rate (%) 2425": dep.iloc[:, 21],
            "DEP004 Patients receiving Intervention (%)": dep.iloc[:, 32],
        }
    )
    dep_out["Practice Code"] = dep_out["Practice Code"].astype(str).str.strip()
    dep_out = dep_out[dep_out["Practice Code"].isin(practices)].drop_duplicates("Practice Code")

    mh = pd.read_excel(workbook, sheet_name="MH", header=11)
    mh_out = pd.DataFrame(
        {
            "Practice Code": mh.iloc[:, 5],
            "Practice Name": mh.iloc[:, 6],
            "List size aged 18+ 2425": mh.iloc[:, 10],
            "Register 2425": mh.iloc[:, 11],
            "Prevalence (%) 2425": mh.iloc[:, 12],
            "MH002 Patients receiving Intervention (%)": mh.iloc[:, 35],
            "MH003 Patients receiving Intervention (%)": mh.iloc[:, 43],
            "MH006 Patients receiving Intervention (%)": mh.iloc[:, 51],
            "MH007 Patients receiving Intervention (%)": mh.iloc[:, 59],
            "MH011 Patients receiving Intervention (%)": mh.iloc[:, 67],
            "MH012 Patients receiving Intervention (%)": mh.iloc[:, 75],
            "MH021 Patients receiving Intervention (%)": mh.iloc[:, 83],
            "Overall PCA Rate (%) 2425": mh.iloc[:, 24],
        }
    )
    mh_out["Practice Code"] = mh_out["Practice Code"].astype(str).str.strip()
    mh_out = mh_out[mh_out["Practice Code"].isin(practices)].drop_duplicates("Practice Code")

    out_dir = root / "datasets" / "quality_outcomes_framework"
    out_dir.mkdir(parents=True, exist_ok=True)
    dep_out.to_csv(out_dir / f"qof_depression_2425_{scope}.csv", index=False, na_rep="")
    mh_out.to_csv(out_dir / f"qof_mental_health_2425_{scope}.csv", index=False, na_rep="")
    print(f"{scope}: QOF depression {len(dep_out):,}; SMI {len(mh_out):,} practices")


def _extract_prescribing(root: Path, scope: str, practices: set[str]) -> set[str]:
    source_scopes = PRESCRIBING_SOURCE_SCOPES.get(scope, [scope])
    normalized = []
    for source_scope in source_scopes:
        source = root / "datasets" / "gp_prescribing_data" / f"items_for_antidepressant_drugs_per_gp_{source_scope}_may_2026.csv"
        raw = pd.read_csv(source, low_memory=False)
        dates = pd.to_datetime(raw["date"], errors="coerce", format="%Y-%m-%d")
        fallback = dates.isna()
        dates.loc[fallback] = pd.to_datetime(raw.loc[fallback, "date"], errors="coerce", format="%d/%m/%Y")
        raw = raw.loc[dates.dt.year.eq(2026) & dates.dt.month.eq(5)].copy()
        items_col = "items" if "items" in raw.columns else "y_items"
        cost_col = "actual_cost" if "actual_cost" in raw.columns else "y_actual_cost"
        normalized.append(
            pd.DataFrame(
                {
                    "date": dates.loc[raw.index].dt.strftime("%d/%m/%Y"),
                    "id": raw["id"],
                    "name": raw["name"],
                    "items": pd.to_numeric(raw[items_col], errors="coerce"),
                    "actual_cost": pd.to_numeric(raw[cost_col], errors="coerce"),
                }
            )
        )
    output = pd.concat(normalized, ignore_index=True).drop_duplicates("id", keep="last")
    output["id"] = output["id"].astype(str).str.strip()
    output = output[output["id"].isin(practices)].copy()
    out = root / "datasets" / "gp_prescribing_data"
    output.to_csv(out / f"need_index_antidepressant_items_per_gp_{scope}_may_2026.csv", index=False)
    return set(output["id"].astype(str).str.strip())


def _extract_scope(
    root: Path,
    scope: str,
    registration: pd.DataFrame,
    population_raw: pd.DataFrame,
    ods_gp_practices: pd.DataFrame,
    parkfields_january: pd.DataFrame,
) -> None:
    lsoa_codes = _boundary_codes(root / "datasets" / "lsoa" / SCOPES[scope])
    ods_practices = set(
        ods_gp_practices.loc[ods_gp_practices["LSOA_CODE"].isin(lsoa_codes), "OrganisationCode"]
        .astype(str)
        .str.strip()
    )
    registration_practices = set(
        registration.loc[registration["LSOA_CODE"].isin(lsoa_codes), "PRACTICE_CODE"]
        .astype(str)
        .str.strip()
    )
    practices = ods_practices | registration_practices
    includes_derbyshire = "derbyshire" in SCOPE_COMPONENTS.get(scope, [scope])
    if includes_derbyshire:
        practices.add(PARKFIELDS_PRACTICE_CODE)
    prescribing_practices = _extract_prescribing(root, scope, practices)
    practices |= prescribing_practices
    scope_registration = registration[registration["PRACTICE_CODE"].isin(practices)].copy()
    county_registration = registration[
        registration["LSOA_CODE"].isin(lsoa_codes)
        & ~registration["PRACTICE_CODE"].isin(SPECIAL_ALLOCATION_PRACTICE_CODES)
    ].copy()
    if includes_derbyshire:
        july_parkfields = scope_registration[
            scope_registration["PRACTICE_CODE"].eq(PARKFIELDS_PRACTICE_CODE)
        ][["LSOA_CODE", "NUMBER_OF_PATIENTS"]].rename(
            columns={"NUMBER_OF_PATIENTS": "JULY_2026_NUMBER_OF_PATIENTS"}
        )
        january_rows = parkfields_january.merge(july_parkfields, on="LSOA_CODE", how="left")
        january_rows["JULY_2026_NUMBER_OF_PATIENTS"] = january_rows[
            "JULY_2026_NUMBER_OF_PATIENTS"
        ].fillna(0)
        scope_registration = scope_registration[
            ~scope_registration["PRACTICE_CODE"].eq(PARKFIELDS_PRACTICE_CODE)
        ]
        scope_registration = pd.concat([scope_registration, january_rows], ignore_index=True)

        county_registration = county_registration[
            ~county_registration["PRACTICE_CODE"].eq(PARKFIELDS_PRACTICE_CODE)
        ]
        county_january = january_rows[january_rows["LSOA_CODE"].isin(lsoa_codes)]
        county_registration = pd.concat([county_registration, county_january], ignore_index=True)

    map_dir = root / "datasets" / "patients_registered_gp_practice" / f"{scope}_july_2026"
    map_dir.mkdir(parents=True, exist_ok=True)
    scope_registration.to_csv(map_dir / "gp-reg-pat-prac-lsoa-all.csv", index=False)

    _extract_qof(root, practices, scope)

    pop = population_raw[population_raw["LSOA 2021 Code"].astype(str).isin(lsoa_codes)].copy()
    f_cols = [col for col in pop.columns if str(col).startswith("F") and str(col)[1:].isdigit()]
    m_cols = [col for col in pop.columns if str(col).startswith("M") and str(col)[1:].isdigit()]
    f_0_17 = [col for col in f_cols if int(col[1:]) < 18]
    m_0_17 = [col for col in m_cols if int(col[1:]) < 18]
    f_18_64 = [col for col in f_cols if 18 <= int(col[1:]) < 65]
    m_18_64 = [col for col in m_cols if 18 <= int(col[1:]) < 65]
    f_65 = [col for col in f_cols if int(col[1:]) >= 65]
    m_65 = [col for col in m_cols if int(col[1:]) >= 65]
    f_18 = [col for col in f_cols if int(col[1:]) >= 18]
    m_18 = [col for col in m_cols if int(col[1:]) >= 18]
    pop_summary = pd.DataFrame(
        {
            "LSOA_CODE": pop["LSOA 2021 Code"].astype(str),
            "LSOA_NAME": pop["LSOA 2021 Name"].astype(str),
            "LAD_CODE": pop["LAD 2023 Code"].astype(str),
            "LAD_NAME": pop["LAD 2023 Name"].astype(str),
            "ONS_Pop_Total_2024": pd.to_numeric(pop["Total"], errors="coerce").fillna(0).astype(int),
            "ONS_Pop_0to17": pop[f_0_17 + m_0_17].sum(axis=1).astype(int),
            "ONS_Pop_18to64": pop[f_18_64 + m_18_64].sum(axis=1).astype(int),
            "ONS_Pop_65plus": pop[f_65 + m_65].sum(axis=1).astype(int),
            "ONS_Pop_18plus": pop[f_18 + m_18].sum(axis=1).astype(int),
        }
    )
    gp_counts = county_registration.groupby("LSOA_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()
    gp_counts = gp_counts.rename(columns={"NUMBER_OF_PATIENTS": "GP_Registered_Patients"})
    pop_summary = pop_summary.merge(gp_counts, on="LSOA_CODE", how="left")
    pop_summary["GP_Registered_Patients"] = pop_summary["GP_Registered_Patients"].fillna(0).astype(int)
    denominator = pop_summary["ONS_Pop_Total_2024"].replace(0, 1)
    for age, column in [("18plus", "ONS_Pop_18plus"), ("65plus", "ONS_Pop_65plus"), ("0to17", "ONS_Pop_0to17"), ("18to64", "ONS_Pop_18to64")]:
        pop_summary[f"Pct_{age}"] = pop_summary[column] / denominator * 100.0
    pop_summary["GP_Registration_Rate_Pct"] = pop_summary["GP_Registered_Patients"] / denominator * 100.0
    pop_summary["Registration_Gap_Est"] = (pop_summary["ONS_Pop_Total_2024"] - pop_summary["GP_Registered_Patients"]).clip(lower=0).astype(int)
    pop_summary["List_Inflation_Est"] = (pop_summary["GP_Registered_Patients"] - pop_summary["ONS_Pop_Total_2024"]).clip(lower=0).astype(int)
    pop_dir = root / "datasets" / "population_estimates"
    pop_dir.mkdir(parents=True, exist_ok=True)
    pop_summary.to_csv(pop_dir / f"{scope}_lsoa_population_estimates_2024.csv", index=False)
    print(f"{scope}: {len(lsoa_codes):,} LSOAs; {len(practices):,} practices; population matched {len(pop_summary):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=sorted(SCOPES),
        help="Regenerate only these area scopes (defaults to all scopes).",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    registration = pd.read_csv(root / REGISTRATION_SOURCE, low_memory=False)
    registration = registration[registration["SEX"].astype(str).str.upper().eq("ALL")].copy()
    registration["PRACTICE_CODE"] = registration["PRACTICE_CODE"].astype(str).str.strip()
    registration["LSOA_CODE"] = registration["LSOA_CODE"].astype(str).str.strip()
    registration["NUMBER_OF_PATIENTS"] = pd.to_numeric(registration["NUMBER_OF_PATIENTS"], errors="coerce")
    registration["REGISTRATION_SOURCE_DATE"] = "2026-07-01"
    registration = registration[
        registration["NUMBER_OF_PATIENTS"].gt(0)
        & registration["LSOA_CODE"].str.fullmatch(r"E\d{8}")
    ]
    ods_gp_practices = _load_ods_gp_practices(root)
    parkfields_january = _load_parkfields_january_registration(root)
    population_raw = pd.read_excel(root / POPULATION_SOURCE, sheet_name="Mid-2024 LSOA 2021", skiprows=3, engine="openpyxl")
    for scope in args.scopes or SCOPES:
        _extract_scope(root, scope, registration, population_raw, ods_gp_practices, parkfields_january)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd

from .common import find_column, normalize_code, parse_numeric


def build_2011_to_2021_lookup(lookup_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each 2011 LSOA code to the 2021 LSOA code(s) it corresponds to (handles splits/merges)."""
    code_2011_col = find_column(lookup_df, ["LSOA11CD"])
    code_2021_col = find_column(lookup_df, ["LSOA21CD"])

    mapping: dict[str, list[str]] = {}
    for code_2011, code_2021 in zip(
        lookup_df[code_2011_col].map(normalize_code), lookup_df[code_2021_col].map(normalize_code)
    ):
        if not code_2011 or not code_2021:
            continue
        mapping.setdefault(code_2011, []).append(code_2021)
    return mapping


def _remap_2011_df_to_2021(df: pd.DataFrame, lookup_map: dict[str, list[str]], value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        code_2011 = row["LSOA_CODE"]
        targets = lookup_map.get(code_2011, [code_2011])
        for target in targets:
            new_row = {"LSOA_CODE": target}
            for col in value_cols:
                new_row[col] = row.get(col)
            rows.append(new_row)

    remapped = pd.DataFrame(rows)
    # Splits duplicate the 2011 value across new 2021 areas; merges are averaged across their 2011 sources.
    return remapped.groupby("LSOA_CODE", as_index=False)[value_cols].mean()


def prepare_travel_times(
    gp_raw: pd.DataFrame,
    hospital_raw: pd.DataFrame,
    lookup_map: dict[str, list[str]],
) -> pd.DataFrame:
    gp_code_col = find_column(gp_raw, ["LSOA_code", "LSOA11CD", "lsoa11"])
    gp_pt_col = find_column(gp_raw, ["GPPTt", "gp_pt_time"])
    gp_car_col = find_column(gp_raw, ["GPCart", "gp_car_time"])

    gp_df = pd.DataFrame(
        {
            "LSOA_CODE": gp_raw[gp_code_col].map(normalize_code),
            "GP_PT_Time": parse_numeric(gp_raw[gp_pt_col]),
            "GP_Car_Time": parse_numeric(gp_raw[gp_car_col]),
        }
    )
    gp_df = gp_df[gp_df["LSOA_CODE"].ne("")].copy()
    gp_2021 = _remap_2011_df_to_2021(gp_df, lookup_map, ["GP_PT_Time", "GP_Car_Time"])

    hosp_code_col = find_column(hospital_raw, ["LSOA_code", "LSOA11CD", "lsoa11"])
    hosp_pt_col = find_column(hospital_raw, ["HospPTt", "hosp_pt_time"])
    hosp_car_col = find_column(hospital_raw, ["HospCart", "hosp_car_time"])

    hosp_df = pd.DataFrame(
        {
            "LSOA_CODE": hospital_raw[hosp_code_col].map(normalize_code),
            "Hosp_PT_Time": parse_numeric(hospital_raw[hosp_pt_col]),
            "Hosp_Car_Time": parse_numeric(hospital_raw[hosp_car_col]),
        }
    )
    hosp_df = hosp_df[hosp_df["LSOA_CODE"].ne("")].copy()
    hosp_2021 = _remap_2011_df_to_2021(hosp_df, lookup_map, ["Hosp_PT_Time", "Hosp_Car_Time"])

    return gp_2021.merge(hosp_2021, on="LSOA_CODE", how="outer")


# Ordered from most-connected (0) to most-isolated (5); higher = more isolated.
ISOLATION_SCALE = {
    "Urban: Nearer to a major town or city": 0,
    "Urban: Further from a major town or city": 1,
    "Larger rural: Nearer to a major town or city": 2,
    "Larger rural: Further from a major town or city": 3,
    "Smaller rural: Nearer to a major town or city": 4,
    "Smaller rural: Further from a major town or city": 5,
}


def prepare_rural_urban(df: pd.DataFrame) -> pd.DataFrame:
    code_col = find_column(df, ["LSOA21CD", "LSOA_CODE"])
    name_col = find_column(df, ["RUC21NM"])
    flag_col = find_column(df, ["Urban_rural_flag", "Flag", "UrbanRuralFlag", "Urban_Rural_Flag"], required=False)

    ruc_series = df[name_col].astype(str).str.strip()
    if flag_col and flag_col in df.columns:
        flag_series = df[flag_col].astype(str).str.strip().str.capitalize()
    else:
        flag_series = ruc_series.map(lambda x: "Rural" if "rural" in str(x).lower() else "Urban")

    out = pd.DataFrame(
        {
            "LSOA_CODE": df[code_col].map(normalize_code),
            "RUC21NM": ruc_series,
            "Urban_rural_flag": flag_series,
        }
    )
    out = out[out["LSOA_CODE"].ne("")].copy()
    out["Isolation_Scale"] = out["RUC21NM"].map(ISOLATION_SCALE)

    min_scale = min(ISOLATION_SCALE.values())
    max_scale = max(ISOLATION_SCALE.values())
    out["Isolation_Normalized"] = (out["Isolation_Scale"] - min_scale) / (max_scale - min_scale)
    out["Rural_Access"] = 1 - out["Isolation_Normalized"]
    return out


def prepare_car_availability(df: pd.DataFrame) -> pd.DataFrame:
    code_col = find_column(df, ["Lower layer Super Output Areas Code", "LSOA21CD", "LSOA_CODE"])
    cat_code_col = find_column(df, ["Car or van availability (5 categories) Code", "Category Code"])
    obs_col = find_column(df, ["Observation", "OBSERVATION", "Count"])

    work = pd.DataFrame(
        {
            "LSOA_CODE": df[code_col].map(normalize_code),
            "Cat_Code": pd.to_numeric(df[cat_code_col], errors="coerce"),
            "Obs": parse_numeric(df[obs_col]),
        }
    )
    work = work[work["LSOA_CODE"].ne("") & work["Cat_Code"].isin([0, 1, 2, 3])].copy()

    pivoted = work.pivot_table(
        index="LSOA_CODE",
        columns="Cat_Code",
        values="Obs",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    pivoted.columns = ["LSOA_CODE", "Cars_0", "Cars_1", "Cars_2", "Cars_3"]

    total_households = pivoted["Cars_0"] + pivoted["Cars_1"] + pivoted["Cars_2"] + pivoted["Cars_3"]
    total_valid = total_households.replace(0, float("nan"))

    pivoted["Total_Households"] = total_households
    pivoted["No_Cars_Pct"] = (pivoted["Cars_0"] / total_valid) * 100.0
    pivoted["One_Car_Pct"] = (pivoted["Cars_1"] / total_valid) * 100.0
    pivoted["Two_Cars_Pct"] = (pivoted["Cars_2"] / total_valid) * 100.0
    pivoted["Three_Plus_Cars_Pct"] = (pivoted["Cars_3"] / total_valid) * 100.0
    pivoted["Car_Access_Pct"] = ((pivoted["Cars_1"] + pivoted["Cars_2"] + pivoted["Cars_3"]) / total_valid) * 100.0

    car_score = (
        0.0 * pivoted["Cars_0"]
        + 1.0 * pivoted["Cars_1"]
        + 2.0 * pivoted["Cars_2"]
        + 3.0 * pivoted["Cars_3"]
    ) / total_valid
    pivoted["Car_Availability_Score"] = car_score

    min_score = float(car_score.min())
    max_score = float(car_score.max())
    span = (max_score - min_score) if (max_score - min_score) > 0 else 1.0
    pivoted["Car_Access"] = (car_score - min_score) / span

    return pivoted[
        [
            "LSOA_CODE",
            "Total_Households",
            "No_Cars_Pct",
            "One_Car_Pct",
            "Two_Cars_Pct",
            "Three_Plus_Cars_Pct",
            "Car_Access_Pct",
            "Car_Availability_Score",
            "Car_Access",
        ]
    ]


def prepare_digital_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    code_col = "LSOA_Code" if "LSOA_Code" in work.columns else "LSOA21CD"
    work["LSOA_CODE"] = work[code_col].astype(str).str.strip()

    keep_cols = [
        "LSOA_CODE",
        "DERI_Score",
        "Digital_Access",
        "Demography_Score",
        "Deprivation_Score",
        "Broadband_Score",
        "Avg_Download_Speed_Mbps",
        "No_Superfast_Broadband_Pct",
        "Slow_Connections_Pct",
        "Aged_65_Plus_Pct",
        "Day_To_Day_Limited_Pct",
        "No_Qualifications_Pct",
        "Pension_Credit_Rate",
        "Unemployment_Rate",
        "Social_Grade_DE_Pct",
        "IMD_2019_Score",
    ]
    present = [c for c in keep_cols if c in work.columns]
    return work[present].drop_duplicates(subset=["LSOA_CODE"])


def prepare_population_estimates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["LSOA_CODE"] = work["LSOA_CODE"].astype(str).str.strip()

    keep_cols = [
        "LSOA_CODE",
        "ONS_Pop_Total_2024",
        "ONS_Pop_0to17",
        "ONS_Pop_18to64",
        "ONS_Pop_65plus",
        "ONS_Pop_18plus",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "Pct_18to64",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    present = [c for c in keep_cols if c in work.columns]
    return work[present].drop_duplicates(subset=["LSOA_CODE"])


def prepare_imd_2025(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    code_col = [c for c in work.columns if "LSOA" in str(c).upper() and "CODE" in str(c).upper()][0]
    rank_col = [c for c in work.columns if "RANK" in str(c).upper()][0]
    decile_col = [c for c in work.columns if "DECILE" in str(c).upper()][0]

    out = pd.DataFrame(
        {
            "LSOA_CODE": work[code_col].astype(str).str.strip(),
            "IMD_2025_Rank": pd.to_numeric(work[rank_col], errors="coerce"),
            "IMD_2025_Decile": pd.to_numeric(work[decile_col], errors="coerce"),
        }
    )
    return out.drop_duplicates(subset=["LSOA_CODE"])


def prepare_lsoac_classification(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    code_col = [c for c in work.columns if "LSOA" in str(c).upper() and "CODE" in str(c).upper()][0]

    out = pd.DataFrame(
        {
            "LSOA_CODE": work[code_col].astype(str).str.strip(),
            "Supergroup_Code": work.get("Supergroup_Code", "").astype(str).str.strip(),
            "Supergroup_Name": work.get("Supergroup_Name", "").astype(str).str.strip(),
            "Group_Code": work.get("Group_Code", "").astype(str).str.strip(),
            "Group_Name": work.get("Group_Name", "").astype(str).str.strip(),
            "Subgroup_Code": work.get("Subgroup_Code", "").astype(str).str.strip(),
            "Subgroup_Name": work.get("Subgroup_Name", "").astype(str).str.strip(),
        }
    )
    return out.drop_duplicates(subset=["LSOA_CODE"])


def prepare_ts003_household_composition(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    code_col = [c for c in work.columns if "LSOA" in str(c).upper() and "CODE" in str(c).upper()][0]

    total_hh = pd.to_numeric(work.get("Total_All_Households", 0), errors="coerce").fillna(0)
    single_pensioner_count = pd.to_numeric(work.get("One-person household: Aged 66 years and over_Count", 0), errors="coerce").fillna(0)
    single_pensioner_pct = pd.to_numeric(work.get("One-person household: Aged 66 years and over_Pct", 0), errors="coerce").fillna(0)
    one_person_other_count = pd.to_numeric(work.get("One-person household: Other_Count", 0), errors="coerce").fillna(0)
    one_person_other_pct = pd.to_numeric(work.get("One-person household: Other_Pct", 0), errors="coerce").fillna(0)
    pensioner_couple_count = pd.to_numeric(work.get("Single family household: All aged 66 years and over_Count", 0), errors="coerce").fillna(0)
    pensioner_couple_pct = pd.to_numeric(work.get("Single family household: All aged 66 years and over_Pct", 0), errors="coerce").fillna(0)
    lone_parent_dep_count = pd.to_numeric(work.get("Single family household: Lone parent family: With dependent children_Count", 0), errors="coerce").fillna(0)
    lone_parent_dep_pct = pd.to_numeric(work.get("Single family household: Lone parent family: With dependent children_Pct", 0), errors="coerce").fillna(0)
    lone_parent_total_count = pd.to_numeric(work.get("Single family household: Lone parent family_Count", 0), errors="coerce").fillna(0)
    lone_parent_total_pct = pd.to_numeric(work.get("Single family household: Lone parent family_Pct", 0), errors="coerce").fillna(0)

    one_person_total_count = pd.to_numeric(work.get("One-person household_Count", 0), errors="coerce").fillna(0)
    non_couple_count = one_person_total_count + lone_parent_total_count
    valid_total = total_hh.replace(0, float("nan"))
    non_couple_pct = (non_couple_count / valid_total) * 100.0

    # Household vulnerability score combining solitary elderly, lone parents, and pensioner couples
    hh_vuln_score = (
        (single_pensioner_pct * 0.45)
        + (lone_parent_dep_pct * 0.30)
        + (pensioner_couple_pct * 0.25)
    )

    out = pd.DataFrame(
        {
            "LSOA_CODE": work[code_col].astype(str).str.strip(),
            "Total_Households_2021": total_hh,
            "Single_Pensioner_HH_Count": single_pensioner_count,
            "Single_Pensioner_HH_Pct": single_pensioner_pct,
            "One_Person_Other_HH_Count": one_person_other_count,
            "One_Person_Other_HH_Pct": one_person_other_pct,
            "Pensioner_Couple_HH_Count": pensioner_couple_count,
            "Pensioner_Couple_HH_Pct": pensioner_couple_pct,
            "Lone_Parent_Dep_Children_HH_Count": lone_parent_dep_count,
            "Lone_Parent_Dep_Children_HH_Pct": lone_parent_dep_pct,
            "Lone_Parent_Total_HH_Count": lone_parent_total_count,
            "Lone_Parent_Total_HH_Pct": lone_parent_total_pct,
            "Non_Couple_HH_Count": non_couple_count,
            "Non_Couple_HH_Pct": non_couple_pct.fillna(0.0),
            "Household_Vulnerability_Score": hh_vuln_score,
        }
    )
    return out.drop_duplicates(subset=["LSOA_CODE"])






from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    clean_text,
    find_column,
    find_prefix_column,
    normalize_code,
    optional_numeric,
    parse_numeric,
    safe_divide,
)


def prepare_depression(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["Practice Code", "PRACTICE_CODE", "Practice_Code"])
    name_col = find_column(df, ["Practice Name", "PRACTICE_NAME", "Practice_Name"], required=False)
    reg_col = find_column(df, ["Register 2425", "Register"])
    list_col = find_column(df, ["List size aged 18+ 2425", "List Size", "List size"])
    prev_col = find_column(df, ["Prevalence (%) 2425", "Prevalence"])
    pca_col = find_column(df, ["Overall PCA Rate (%) 2425", "Overall PCA Rate (%)"], required=False)
    dep004_col = find_prefix_column(df, "DEP004", required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_Dep": df[name_col].map(clean_text) if name_col else "",
            "Dep_Register": parse_numeric(df[reg_col]),
            "Dep_List_Size": parse_numeric(df[list_col]),
            "Dep_Prevalence_Pct": parse_numeric(df[prev_col]),
            "Dep_Exception_Rate_Pct": optional_numeric(df, pca_col),
            "DEP004_Pct": optional_numeric(df, dep004_col),
        }
    )

    out = out[out["PRACTICE_CODE"].ne("")].copy()
    calc_prev = safe_divide(out["Dep_Register"], out["Dep_List_Size"]) * 100
    out["Dep_Prevalence_Pct"] = out["Dep_Prevalence_Pct"].fillna(calc_prev)
    return out


def prepare_smi(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["Practice Code", "PRACTICE_CODE", "Practice_Code"])
    name_col = find_column(df, ["Practice Name", "PRACTICE_NAME", "Practice_Name"], required=False)
    reg_col = find_column(df, ["Register 2425", "Register"])
    list_col = find_column(df, ["List size aged 18+ 2425", "List Size", "List size"])
    prev_col = find_column(df, ["Prevalence (%) 2425", "Prevalence"])

    mh002_col = find_prefix_column(df, "MH002", required=False)
    mh003_col = find_prefix_column(df, "MH003", required=False)
    mh006_col = find_prefix_column(df, "MH006", required=False)
    mh007_col = find_prefix_column(df, "MH007", required=False)
    mh011_col = find_prefix_column(df, "MH011", required=False)
    mh012_col = find_prefix_column(df, "MH012", required=False)
    mh021_col = find_prefix_column(df, "MH021", required=False)
    pca_col = find_column(df, ["Overall PCA Rate (%) 2425", "Overall PCA Rate (%)"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_SMI": df[name_col].map(clean_text) if name_col else "",
            "SMI_Register": parse_numeric(df[reg_col]),
            "SMI_List_Size": parse_numeric(df[list_col]),
            "SMI_Prevalence_Pct": parse_numeric(df[prev_col]),
            "MH002_Pct": optional_numeric(df, mh002_col),
            "MH003_Pct": optional_numeric(df, mh003_col),
            "MH006_Pct": optional_numeric(df, mh006_col),
            "MH007_Pct": optional_numeric(df, mh007_col),
            "MH011_Pct": optional_numeric(df, mh011_col),
            "MH012_Pct": optional_numeric(df, mh012_col),
            "MH021_Pct": optional_numeric(df, mh021_col),
            "SMI_Exception_Rate_Pct": optional_numeric(df, pca_col),
        }
    )

    out = out[out["PRACTICE_CODE"].ne("")].copy()
    calc_prev = safe_divide(out["SMI_Register"], out["SMI_List_Size"]) * 100
    out["SMI_Prevalence_Pct"] = out["SMI_Prevalence_Pct"].fillna(calc_prev)
    out["Physical_Health_Review_Avg_Pct"] = out[
        ["MH003_Pct", "MH006_Pct", "MH007_Pct", "MH011_Pct", "MH012_Pct"]
    ].mean(axis=1, skipna=True)
    return out


def prepare_prescribing(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["id", "PRACTICE_CODE", "Practice Code"])
    items_col = find_column(df, ["items", "ITEMS", "Number of Items"])
    name_col = find_column(df, ["name", "Practice Name", "PRACTICE_NAME"], required=False)
    cost_col = find_column(df, ["actual_cost", "ACTUAL_COST", "cost"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_Rx": df[name_col].map(clean_text) if name_col else "",
            "Antidepressant_Items": parse_numeric(df[items_col]),
            "Antidepressant_Actual_Cost": optional_numeric(df, cost_col),
        }
    )
    out = out[out["PRACTICE_CODE"].ne("")].copy()
    out = out.groupby("PRACTICE_CODE", as_index=False).agg(
        {
            "Practice_Name_Rx": "first",
            "Antidepressant_Items": "sum",
            "Antidepressant_Actual_Cost": "sum",
        }
    )
    return out


def prepare_mapping(df: pd.DataFrame, sex_label: str | None = None) -> pd.DataFrame:
    practice_col = find_column(df, ["PRACTICE_CODE", "Practice Code"])
    lsoa_col = find_column(df, ["LSOA_CODE", "LSOA"])
    patients_col = find_column(df, ["NUMBER_OF_PATIENTS", "Number of Patients"])
    sex_col = find_column(df, ["SEX"], required=False)

    if sex_col:
        sex_series = df[sex_col].astype(str).str.strip().str.upper().replace({"M": "MALE", "F": "FEMALE"})
    elif sex_label:
        sex_series = pd.Series(str(sex_label).upper(), index=df.index)
    else:
        sex_series = pd.Series("ALL", index=df.index)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "LSOA_CODE": df[lsoa_col].map(normalize_code),
            "NUMBER_OF_PATIENTS": parse_numeric(df[patients_col]),
            "SEX": sex_series,
        }
    )

    out = out[out["NUMBER_OF_PATIENTS"].notna()]
    if sex_col and sex_label is None:
        out = out[out["SEX"] == "ALL"]

    out = out[(out["PRACTICE_CODE"].ne("")) & (out["LSOA_CODE"].ne(""))]
    out = out[out["NUMBER_OF_PATIENTS"] > 0]
    return out


def prepare_mapping_from_sex_split(male_df: pd.DataFrame, female_df: pd.DataFrame) -> pd.DataFrame:
    male = prepare_mapping(male_df, sex_label="MALE")
    female = prepare_mapping(female_df, sex_label="FEMALE")

    combined = pd.concat([male, female], ignore_index=True)
    combined = combined[combined["SEX"].isin(["MALE", "FEMALE"])].copy()

    split = (
        combined.pivot_table(
            index=["PRACTICE_CODE", "LSOA_CODE"],
            columns="SEX",
            values="NUMBER_OF_PATIENTS",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )

    if "MALE" not in split.columns:
        split["MALE"] = 0
    if "FEMALE" not in split.columns:
        split["FEMALE"] = 0

    split = split.rename(columns={"MALE": "NUMBER_OF_PATIENTS_MALE", "FEMALE": "NUMBER_OF_PATIENTS_FEMALE"})
    split["NUMBER_OF_PATIENTS"] = split["NUMBER_OF_PATIENTS_MALE"] + split["NUMBER_OF_PATIENTS_FEMALE"]
    split = split[split["NUMBER_OF_PATIENTS"] > 0].copy()

    gp_totals = split.groupby("PRACTICE_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()
    gp_totals = gp_totals.rename(columns={"NUMBER_OF_PATIENTS": "GP_Total_Mapped_Patients"})
    split = split.merge(gp_totals, on="PRACTICE_CODE", how="left")
    split["Weight"] = safe_divide(split["NUMBER_OF_PATIENTS"], split["GP_Total_Mapped_Patients"])
    return split


def build_gp_master(dep_df: pd.DataFrame, smi_df: pd.DataFrame, prescribing_df: pd.DataFrame) -> pd.DataFrame:
    gp = dep_df.merge(smi_df, on="PRACTICE_CODE", how="outer")
    gp = gp.merge(prescribing_df, on="PRACTICE_CODE", how="outer")

    gp["Practice_Name"] = gp["Practice_Name_Dep"].where(gp["Practice_Name_Dep"].astype(bool), gp["Practice_Name_SMI"])
    gp["Practice_Name"] = gp["Practice_Name"].where(gp["Practice_Name"].astype(bool), gp["Practice_Name_Rx"])
    gp["Practice_Name"] = gp["Practice_Name"].fillna("")

    gp["Exception_Rate_Pct"] = gp["SMI_Exception_Rate_Pct"].where(
        gp["SMI_Exception_Rate_Pct"].notna(), gp["Dep_Exception_Rate_Pct"]
    )
    return gp


def allocate_registers_to_lsoa(mapping_df: pd.DataFrame, gp_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    alloc = mapping_df.merge(
        gp_df[["PRACTICE_CODE", "Dep_Register", "SMI_Register", "Antidepressant_Items"]],
        on="PRACTICE_CODE",
        how="left",
    )

    missing_dep_by_practice = alloc["Dep_Register"].isna().groupby(alloc["PRACTICE_CODE"]).first()
    missing_smi_by_practice = alloc["SMI_Register"].isna().groupby(alloc["PRACTICE_CODE"]).first()
    missing_rx_by_practice = alloc["Antidepressant_Items"].isna().groupby(alloc["PRACTICE_CODE"]).first()
    missing_any_by_practice = missing_dep_by_practice | missing_smi_by_practice

    mismatch_summary = {
        "mapped_practices_without_any_qof": int(missing_any_by_practice.sum()),
        "mapped_practices_without_dep_qof": int(missing_dep_by_practice.sum()),
        "mapped_practices_without_smi_qof": int(missing_smi_by_practice.sum()),
        "mapped_practices_without_prescribing": int(missing_rx_by_practice.sum()),
        "allocation_rows_missing_any_qof": int((alloc["Dep_Register"].isna() | alloc["SMI_Register"].isna()).sum()),
        "allocation_rows_missing_prescribing": int(alloc["Antidepressant_Items"].isna().sum()),
    }

    alloc["Allocated_Depression"] = alloc["Dep_Register"] * alloc["Weight"].fillna(0)
    alloc["Allocated_SMI"] = alloc["SMI_Register"] * alloc["Weight"].fillna(0)
    alloc["Allocated_Antidepressant_Items"] = alloc["Antidepressant_Items"] * alloc["Weight"].fillna(0)
    alloc["Antidepressant_Covered_Patients"] = alloc["NUMBER_OF_PATIENTS"].where(
        alloc["Antidepressant_Items"].notna(), 0
    )

    alloc["Depression_Covered_Patients"] = alloc["NUMBER_OF_PATIENTS"].where(
        alloc["Dep_Register"].notna(), 0
    )
    alloc["SMI_Covered_Patients"] = alloc["NUMBER_OF_PATIENTS"].where(
        alloc["SMI_Register"].notna(), 0
    )
    lsoa_dep = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_Depression"].sum(min_count=1)
    lsoa_smi = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_SMI"].sum(min_count=1)
    lsoa_rx = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_Antidepressant_Items"].sum(min_count=1)
    lsoa_dep_covered = alloc.groupby("LSOA_CODE", as_index=False)["Depression_Covered_Patients"].sum()
    lsoa_smi_covered = alloc.groupby("LSOA_CODE", as_index=False)["SMI_Covered_Patients"].sum()
    lsoa_rx_covered = alloc.groupby("LSOA_CODE", as_index=False)["Antidepressant_Covered_Patients"].sum()
    lsoa_list = alloc.groupby("LSOA_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()

    lsoa = lsoa_list.merge(lsoa_dep, on="LSOA_CODE", how="left").merge(lsoa_smi, on="LSOA_CODE", how="left").merge(
        lsoa_rx, on="LSOA_CODE", how="left"
    ).merge(lsoa_dep_covered, on="LSOA_CODE", how="left").merge(
        lsoa_smi_covered, on="LSOA_CODE", how="left"
    ).merge(lsoa_rx_covered, on="LSOA_CODE", how="left")
    lsoa = lsoa.rename(columns={"NUMBER_OF_PATIENTS": "LSOA_Total_List"})

    lsoa["Depression_Prevalence"] = safe_divide(
        lsoa["Allocated_Depression"], lsoa["Depression_Covered_Patients"]
    )
    lsoa["SMI_Prevalence"] = safe_divide(lsoa["Allocated_SMI"], lsoa["SMI_Covered_Patients"])
    lsoa["Depression_Coverage_Pct"] = safe_divide(
        lsoa["Depression_Covered_Patients"], lsoa["LSOA_Total_List"]
    ) * 100.0
    lsoa["SMI_Coverage_Pct"] = safe_divide(
        lsoa["SMI_Covered_Patients"], lsoa["LSOA_Total_List"]
    ) * 100.0
    lsoa["Antidepressant_Items_Per_Patient"] = safe_divide(
        lsoa["Allocated_Antidepressant_Items"], lsoa["Antidepressant_Covered_Patients"]
    )
    lsoa["Antidepressant_Coverage_Pct"] = safe_divide(
        lsoa["Antidepressant_Covered_Patients"], lsoa["LSOA_Total_List"]
    ) * 100.0
    return lsoa, mismatch_summary


ACCESS_RATE_COLUMNS = {
    "MH002_Pct": "MH002_Access_Pct",
    "MH021_Pct": "MH021_Access_Pct",
    "SMI_Exception_Rate_Pct": "MH_PCA_Access_Pct",
    "Dep_Exception_Rate_Pct": "Dep_PCA_Access_Pct",
    "DEP004_Pct": "DEP004_Access_Pct",
}


def allocate_access_rates_to_lsoa(mapping_df: pd.DataFrame, gp_df: pd.DataFrame) -> pd.DataFrame:
    """Patient-weighted average of GP-level QOF intervention/PCA rates, rolled up to LSOA."""
    rate_source_cols = [c for c in ACCESS_RATE_COLUMNS if c in gp_df.columns]
    if not rate_source_cols:
        return pd.DataFrame(columns=["LSOA_CODE"])

    alloc = mapping_df.merge(
        gp_df[["PRACTICE_CODE", *rate_source_cols]],
        on="PRACTICE_CODE",
        how="left",
    )
    lsoa_codes = alloc["LSOA_CODE"]
    patient_counts = alloc["NUMBER_OF_PATIENTS"]

    # Compute the patient-weighted mean for each LSOA/rate in a few pandas
    # groupby operations. Iterating through every LSOA and rate in Python made
    # the first load of an area take over a minute for the full registration file.
    weighted_rates = alloc[rate_source_cols].multiply(patient_counts, axis=0)
    numerators = weighted_rates.groupby(lsoa_codes, sort=False).sum(min_count=1)
    denominators = pd.DataFrame(
        {
            column: patient_counts.where(alloc[column].notna())
            for column in rate_source_cols
        },
        index=alloc.index,
    ).groupby(lsoa_codes, sort=False).sum(min_count=1)

    result = numerators.div(denominators.where(denominators.gt(0)))
    result = result.rename(columns=ACCESS_RATE_COLUMNS)
    result.index.name = "LSOA_CODE"
    return result.reset_index()


def prepare_gp_locations(df: pd.DataFrame) -> pd.DataFrame:
    code_col = find_column(df, ["Code", "PRACTICE_CODE", "Practice Code", "OrganisationCode"])
    name_col = find_column(df, ["Name", "Practice Name", "PRACTICE_NAME", "OrgName"], required=False)
    lsoa_col = find_column(df, ["LSOA", "LSOA_CODE", "lsoa21cd"], required=False)

    lat_col = find_column(df, ["Latitude", "LAT", "lat"], required=False)
    lon_col = find_column(df, ["Longitude", "LON", "lng", "Long"], required=False)

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[code_col].map(normalize_code),
            "GP_Name_File": df[name_col].map(clean_text) if name_col else "",
            "GP_LSOA": df[lsoa_col].map(normalize_code) if lsoa_col else "",
            "GP_Lat": parse_numeric(df[lat_col]) if lat_col else np.nan,
            "GP_Lon": parse_numeric(df[lon_col]) if lon_col else np.nan,
        }
    )
    out = out[out["PRACTICE_CODE"].ne("")].copy()

    valid_lat = out["GP_Lat"].between(49.0, 61.0)
    valid_lon = out["GP_Lon"].between(-8.5, 2.5)
    out.loc[~(valid_lat & valid_lon), ["GP_Lat", "GP_Lon"]] = np.nan
    return out


def build_gp_marker_df(
    gp_locations: pd.DataFrame,
    gp_master: pd.DataFrame,
    mapping_df: pd.DataFrame,
    lsoa_centroids: pd.DataFrame,
    in_area_lsoa_codes: set[str],
) -> pd.DataFrame:
    dominant_lsoa = (
        mapping_df.sort_values("NUMBER_OF_PATIENTS", ascending=False)
        .drop_duplicates(subset=["PRACTICE_CODE"])[["PRACTICE_CODE", "LSOA_CODE"]]
        .rename(columns={"LSOA_CODE": "Dominant_LSOA"})
    )

    dominant_lsoa_in_area = (
        mapping_df[mapping_df["LSOA_CODE"].isin(in_area_lsoa_codes)]
        .sort_values("NUMBER_OF_PATIENTS", ascending=False)
        .drop_duplicates(subset=["PRACTICE_CODE"])[["PRACTICE_CODE", "LSOA_CODE"]]
        .rename(columns={"LSOA_CODE": "Dominant_LSOA_InArea"})
    )

    gp_registered_totals = (
        mapping_df.groupby("PRACTICE_CODE", as_index=False)["NUMBER_OF_PATIENTS"]
        .sum()
        .rename(columns={"NUMBER_OF_PATIENTS": "NUMBER_OF_PATIENTS"})
    )

    gp = gp_locations.merge(gp_master, on="PRACTICE_CODE", how="outer")
    gp = gp.merge(dominant_lsoa, on="PRACTICE_CODE", how="left")
    gp = gp.merge(dominant_lsoa_in_area, on="PRACTICE_CODE", how="left")
    gp = gp.merge(gp_registered_totals, on="PRACTICE_CODE", how="left")

    gp["Effective_LSOA"] = gp["GP_LSOA"].where(gp["GP_LSOA"].astype(str).str.len() > 0, gp["Dominant_LSOA"])
    gp["Effective_LSOA"] = gp["Effective_LSOA"].where(
        gp["Effective_LSOA"].isin(in_area_lsoa_codes), gp["Dominant_LSOA_InArea"]
    )

    gp = gp.merge(
        lsoa_centroids.rename(columns={"LSOA_CODE": "Effective_LSOA", "lat": "LSOA_Lat", "lon": "LSOA_Lon"}),
        on="Effective_LSOA",
        how="left",
    )

    gp["Lat"] = gp["GP_Lat"].where(gp["GP_Lat"].notna(), gp["LSOA_Lat"])
    gp["Lon"] = gp["GP_Lon"].where(gp["GP_Lon"].notna(), gp["LSOA_Lon"])
    gp["Practice_Name"] = gp["Practice_Name"].where(gp["Practice_Name"].astype(str).str.len() > 0, gp["GP_Name_File"])

    return gp

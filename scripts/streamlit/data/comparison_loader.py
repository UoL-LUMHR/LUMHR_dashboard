from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.common import (
    clean_text,
    find_column,
    find_prefix_column,
    normalize_code,
    optional_numeric,
    parse_numeric,
    resolve_base_dir,
    safe_divide,
)
from analysis.samhi import join_samhi, prepare_samhi


def get_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "depression": base_dir / "datasets" / "quality_outcomes_framework" / "qof_depression_2425_lincolnshire.csv",
        "smi": base_dir / "datasets" / "quality_outcomes_framework" / "qof_mental_health_2425_lincolnshire.csv",
        "prescribing": base_dir
        / "datasets"
        / "gp_prescribing_data"
        / "items_for_antidepressant_drugs_per_gp_lincolnshire_may_2026.csv",
        "mapping_male": base_dir
        / "datasets"
        / "patients_registered_gp_practice"
        / "july_2026"
        / "gp-reg-pat-prac-lsoa-male.csv",
        "mapping_female": base_dir
        / "datasets"
        / "patients_registered_gp_practice"
        / "july_2026"
        / "gp-reg-pat-prac-lsoa-female.csv",
        "samhi": base_dir / "datasets" / "samhi" / "samhi_lincolnshire_2021_lsoa.csv",
    }


def prepare_depression(df: pd.DataFrame) -> pd.DataFrame:
    practice_col = find_column(df, ["Practice Code", "PRACTICE_CODE", "Practice_Code"])
    name_col = find_column(df, ["Practice Name", "PRACTICE_NAME", "Practice_Name"], required=False)
    reg_col = find_column(df, ["Register 2425", "Register"])
    list_col = find_column(df, ["List size aged 18+ 2425", "List Size", "List size"])
    prev_col = find_column(df, ["Prevalence (%) 2425", "Prevalence"])

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "Practice_Name_Dep": df[name_col].map(clean_text) if name_col else "",
            "Dep_Register": parse_numeric(df[reg_col]),
            "Dep_List_Size": parse_numeric(df[list_col]),
            "Dep_Prevalence_Pct": parse_numeric(df[prev_col]),
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
        }
    )

    out = out[out["PRACTICE_CODE"].ne("")].copy()
    calc_prev = safe_divide(out["SMI_Register"], out["SMI_List_Size"]) * 100
    out["SMI_Prevalence_Pct"] = out["SMI_Prevalence_Pct"].fillna(calc_prev)
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


def prepare_mapping(df: pd.DataFrame, sex_label: str) -> pd.DataFrame:
    practice_col = find_column(df, ["PRACTICE_CODE", "Practice Code"])
    lsoa_col = find_column(df, ["LSOA_CODE", "LSOA"])
    patients_col = find_column(df, ["NUMBER_OF_PATIENTS", "Number of Patients"])

    out = pd.DataFrame(
        {
            "PRACTICE_CODE": df[practice_col].map(normalize_code),
            "LSOA_CODE": df[lsoa_col].map(normalize_code),
            "NUMBER_OF_PATIENTS": parse_numeric(df[patients_col]),
            "SEX": sex_label,
        }
    )

    out = out[out["NUMBER_OF_PATIENTS"].notna()]
    out = out[(out["PRACTICE_CODE"].ne("")) & (out["LSOA_CODE"].ne(""))]
    out = out[out["NUMBER_OF_PATIENTS"] > 0]
    return out


def prepare_mapping_from_sex_split(male_df: pd.DataFrame, female_df: pd.DataFrame) -> pd.DataFrame:
    male = prepare_mapping(male_df, sex_label="MALE")
    female = prepare_mapping(female_df, sex_label="FEMALE")

    combined = pd.concat([male, female], ignore_index=True)
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
    return gp


def allocate_registers_to_lsoa(mapping_df: pd.DataFrame, gp_df: pd.DataFrame) -> pd.DataFrame:
    alloc = mapping_df.merge(
        gp_df[["PRACTICE_CODE", "Dep_Register", "SMI_Register", "Antidepressant_Items"]],
        on="PRACTICE_CODE",
        how="left",
    )

    alloc["Allocated_Depression"] = alloc["Dep_Register"].fillna(0) * alloc["Weight"].fillna(0)
    alloc["Allocated_SMI"] = alloc["SMI_Register"].fillna(0) * alloc["Weight"].fillna(0)
    alloc["Allocated_Antidepressant_Items"] = alloc["Antidepressant_Items"].fillna(0) * alloc["Weight"].fillna(0)

    lsoa_dep = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_Depression"].sum()
    lsoa_smi = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_SMI"].sum()
    lsoa_rx = alloc.groupby("LSOA_CODE", as_index=False)["Allocated_Antidepressant_Items"].sum()
    lsoa_list = alloc.groupby("LSOA_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()

    lsoa = lsoa_list.merge(lsoa_dep, on="LSOA_CODE", how="left").merge(lsoa_smi, on="LSOA_CODE", how="left").merge(
        lsoa_rx, on="LSOA_CODE", how="left"
    )
    lsoa = lsoa.rename(columns={"NUMBER_OF_PATIENTS": "LSOA_Total_List"})

    lsoa["Depression_Prevalence"] = safe_divide(lsoa["Allocated_Depression"], lsoa["LSOA_Total_List"])
    lsoa["SMI_Prevalence"] = safe_divide(lsoa["Allocated_SMI"], lsoa["LSOA_Total_List"])
    lsoa["Antidepressant_Items_Per_Patient"] = safe_divide(
        lsoa["Allocated_Antidepressant_Items"], lsoa["LSOA_Total_List"]
    )
    return lsoa


@st.cache_data(show_spinner=False)
def get_comparison_bundle_cached(base_dir_str: str) -> dict[str, object]:
    base_dir = Path(base_dir_str)
    paths = get_paths(base_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found for {name}: {path}")

    dep_df = prepare_depression(pd.read_csv(paths["depression"]))
    smi_df = prepare_smi(pd.read_csv(paths["smi"]))
    prescribing_df = prepare_prescribing(pd.read_csv(paths["prescribing"]))
    mapping_df = prepare_mapping_from_sex_split(pd.read_csv(paths["mapping_male"]), pd.read_csv(paths["mapping_female"]))
    samhi_df = prepare_samhi(pd.read_csv(paths["samhi"]))

    gp_master = build_gp_master(dep_df, smi_df, prescribing_df)
    lsoa_metrics = allocate_registers_to_lsoa(mapping_df, gp_master)
    lsoa_metrics = join_samhi(lsoa_metrics, samhi_df)
    return {"lsoa_metrics": lsoa_metrics}

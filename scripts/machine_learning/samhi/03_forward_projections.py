"""
03_forward_projections.py
-------------------------
Phase 3: True Multi-Year Forward Projections (2023, 2024, 2025) for SAMHI.

Incorporates:
  - Complete 2014-2022 panel training pool (295k+ national observations)
  - IMD 2025 Deprivation Rank & Decile (National England)
  - Hospital travel times: public transit & car (National England)
  - GP travel times: public transit & car (National England)
  - Rural/Urban isolation scale & isolation normalized (National England)
  - Digital infrastructure (broadband download speeds) & car availability
  - Demographics (chronic disability %, elderly 65+ %, social grade DE %, single occupancy %)
  - Spatial neighbor spillovers (Queen contiguity) & District-level clusters

Exports:
  - results/forward_projections_2023_2025_lincolnshire.csv
  - results/forward_projections_2023_2025_national.csv
  - results/forward_projections_summary.csv
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from interpret.glassbox import ExplainableBoostingRegressor

# Reuse the dashboard's tested practice-to-LSOA QOF allocation functions.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flask"))
from core.allocation import (  # noqa: E402
    allocate_access_rates_to_lsoa,
    prepare_depression,
    prepare_mapping,
    prepare_smi,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SAMHI_ForwardProjections")

LINCOLNSHIRE_LADS = {
    "Boston",
    "East Lindsey",
    "Lincoln",
    "North Kesteven",
    "South Holland",
    "South Kesteven",
    "West Lindsey"
}

ISOLATION_SCALE_MAP = {
    "Urban major conurbation": 1,
    "Urban minor conurbation": 2,
    "Urban city and town": 3,
    "Urban with significant rural": 4,
    "Rural town and fringe": 5,
    "Rural village and dispersed": 6,
}


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parents[3]


def load_all_datasets(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load national SAMHI and multimodal auxiliary datasets including hospital, GP, and IMD 2025 data."""
    datasets_dir = project_root / "datasets"

    samhi_path = datasets_dir / "samhi" / "samhi_21_01_v5.00_2011_2022_LSOA.csv"
    lookup_path = (
        datasets_dir
        / "lincolnshire_lsoa"
        / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    deri_path = datasets_dir / "digital_exclusion_risk_index" / "DERI dataset_v1.6.csv"
    ru_path = (
        datasets_dir
        / "rural_urban_classification_2021_lsoa"
        / "Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv"
    )
    ts003_path = datasets_dir / "TS003_household_composition" / "TS003 - Household composition.csv"
    if not ts003_path.exists():
        ts003_path = datasets_dir / "TS003_household_composition" / "lincolnshire_ts003_household_composition.csv"

    car_path = datasets_dir / "car_or_van_availability" / "TS045-2021-4-filtered-2026-08-24T10_54_11Z.csv"
    geojson_path = datasets_dir / "lincolnshire_lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson"

    # National additions
    imd25_path = datasets_dir / "indices_of_deprivation_imd" / "2025" / "england_imd_2025_lsoa.csv"
    hosp_path = datasets_dir / "journey_time_statistics" / "england" / "clean_hospitals_2019_england.csv"
    gp_path = datasets_dir / "journey_time_statistics" / "england" / "clean_gps_2019_england.csv"

    qof_dep_path = datasets_dir / "quality_outcomes_framework" / "qof_depression_2425_lincolnshire.csv"
    qof_smi_path = datasets_dir / "quality_outcomes_framework" / "qof_mental_health_2425_lincolnshire.csv"
    qof_mapping_path = datasets_dir / "patients_registered_gp_practice" / "july_2026" / "gp-reg-pat-prac-lsoa-all.csv"


    logger.info("Loading SAMHI and auxiliary datasets...")
    samhi_df = pd.read_csv(samhi_path)
    samhi_df["lsoa11"] = samhi_df["lsoa11"].astype(str).str.strip()

    lookup_df = pd.read_csv(lookup_path)
    lookup_df["LSOA11CD"] = lookup_df["LSOA11CD"].astype(str).str.strip()
    lookup_df["LSOA21CD"] = lookup_df["LSOA21CD"].astype(str).str.strip()
    lookup_df["LSOA21NM"] = lookup_df["LSOA21NM"].astype(str).str.strip()
    lookup_df["LAD22NM"] = lookup_df["LAD22NM"].astype(str).str.strip()

    deri_df = pd.read_csv(deri_path)
    deri_df["LSOA code"] = deri_df["LSOA code"].astype(str).str.strip()

    ru_df = pd.read_csv(ru_path)
    ru_df["LSOA21CD"] = ru_df["LSOA21CD"].astype(str).str.strip()
    ru_df["is_rural"] = (ru_df["Urban_rural_flag"].astype(str).str.strip() == "Rural").astype(int)
    ru_df["isolation_scale"] = ru_df["RUC21NM"].map(ISOLATION_SCALE_MAP).fillna(3).astype(float)
    ru_df["isolation_normalized"] = (ru_df["isolation_scale"] - 1.0) / 5.0

    ts003_df = pd.DataFrame()
    if ts003_path.exists():
        try:
            if "lincolnshire" in str(ts003_path).lower():
                ts003_raw = pd.read_csv(ts003_path)
                ts003_df = ts003_raw[["LSOA_Code", "One-person household_Pct", "Single family household: Lone parent family_Pct"]].rename(columns={
                    "LSOA_Code": "LSOA21CD",
                    "One-person household_Pct": "one_person_household_pct",
                    "Single family household: Lone parent family_Pct": "lone_parent_pct"
                })
            else:
                ts003_raw = pd.read_csv(ts003_path, skiprows=7)
                code_s = ts003_raw["2021 super output area - lower layer"].str.split(":").str[0].str.strip()
                one_p = pd.to_numeric(ts003_raw["%.1"], errors="coerce")
                lone_p = pd.to_numeric(ts003_raw.get("%.10", ts003_raw["%.1"]), errors="coerce")
                ts003_df = pd.DataFrame({"LSOA21CD": code_s, "one_person_household_pct": one_p, "lone_parent_pct": lone_p})
        except Exception as e:
            logger.warning(f"Could not load TS003: {e}")
            ts003_df = pd.DataFrame()

    if not ts003_df.empty:
        ts003_df["LSOA21CD"] = ts003_df["LSOA21CD"].astype(str).str.strip()
        ts003_df = ts003_df.groupby("LSOA21CD", as_index=False)[
            ["one_person_household_pct", "lone_parent_pct"]
        ].mean()

    car_df = pd.read_csv(car_path) if car_path.exists() else pd.DataFrame()

    # IMD 2025
    imd25_df = pd.read_csv(imd25_path) if imd25_path.exists() else pd.DataFrame()
    if not imd25_df.empty:
        col_map = {
            "LSOA code (2021)": "LSOA21CD",
            "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": "imd_2025_rank",
            "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA": "imd_2025_decile"
        }
        imd25_df = imd25_df.rename(columns=col_map)
        imd25_df["LSOA21CD"] = imd25_df["LSOA21CD"].astype(str).str.strip()
        imd25_df["imd_2025_rank"] = pd.to_numeric(imd25_df["imd_2025_rank"], errors="coerce")
        imd25_df["imd_2025_decile"] = pd.to_numeric(imd25_df["imd_2025_decile"], errors="coerce")

    # National Travel Times
    hosp_df = pd.read_csv(hosp_path) if hosp_path.exists() else pd.DataFrame()
    if not hosp_df.empty:
        hosp_df["lsoa11"] = hosp_df["lsoa11"].astype(str).str.strip()
        hosp_df["hosp_pt_time"] = pd.to_numeric(hosp_df["hosp_pt_time"], errors="coerce")
        hosp_df["hosp_car_time"] = pd.to_numeric(hosp_df["hosp_car_time"], errors="coerce")

    gp_df = pd.read_csv(gp_path) if gp_path.exists() else pd.DataFrame()
    if not gp_df.empty:
        gp_df["lsoa11"] = gp_df["lsoa11"].astype(str).str.strip()
        gp_df["gp_pt_time"] = pd.to_numeric(gp_df["gp_pt_time"], errors="coerce")
        gp_df["gp_car_time"] = pd.to_numeric(gp_df["gp_car_time"], errors="coerce")

    # Use the same 2024-25 QOF allocation as the multimodal benchmark.
    qof_access_df = pd.DataFrame()
    if qof_dep_path.exists() and qof_smi_path.exists() and qof_mapping_path.exists():
        try:
            dep_qof = prepare_depression(pd.read_csv(qof_dep_path))
            smi_qof = prepare_smi(pd.read_csv(qof_smi_path))
            qof_gp = dep_qof.merge(smi_qof, on="PRACTICE_CODE", how="outer")
            mapping = prepare_mapping(pd.read_csv(qof_mapping_path))
            qof_access_df = allocate_access_rates_to_lsoa(mapping, qof_gp).rename(columns={
                "MH002_Access_Pct": "qof_mh002_pct",
                "MH021_Access_Pct": "qof_mh021_pct",
                "SMI_Exception_Rate_Pct": "qof_mh_pca_pct",
                "Dep_Exception_Rate_Pct": "qof_dep_pca_pct",
                "DEP004_Pct": "qof_dep004_pct",
            })
            qof_cols = ["qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"]
            for col in qof_cols:
                if col not in qof_access_df:
                    qof_access_df[col] = np.nan
            # A mapped LSOA can receive multiple practice rows; aggregate before merging.
            qof_access_df = qof_access_df.groupby("LSOA_CODE", as_index=False)[qof_cols].mean()
            logger.info("Loaded QOF 2024-25 rates for %s unique LSOAs", len(qof_access_df))
        except Exception as exc:
            logger.warning("Could not load QOF projection features: %s", exc)


    return {
        "samhi": samhi_df,
        "lookup": lookup_df,
        "deri": deri_df,
        "rural_urban": ru_df,
        "ts003": ts003_df,
        "car": car_df,
        "imd25": imd25_df,
        "hosp": hosp_df,
        "gp": gp_df,
        "qof_access": qof_access_df,
        "geojson_path": geojson_path
    }


def compute_spatial_neighbors(geojson_path: Path) -> Dict[str, List[str]]:
    """Compute Queen spatial adjacency contiguity graph from GeoJSON."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(geojson_path)
        code_col = "CODE" if "CODE" in gdf.columns else "LSOA21CD"
        neighbors = {}
        for _, row in gdf.iterrows():
            code = row[code_col]
            geom = row.geometry
            touching = gdf[gdf.geometry.touches(geom)][code_col].tolist()
            neighbors[code] = touching
        return neighbors
    except Exception as e:
        logger.warning(f"Could not compute spatial adjacency: {e}")
        return {}


def build_master_frame(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build unified cross-sectional dataset incorporating the new national features."""
    samhi_df = data["samhi"]
    lookup_df = data["lookup"]
    deri_df = data["deri"]
    ru_df = data["rural_urban"]
    ts003_df = data["ts003"]
    car_df = data["car"]
    imd25_df = data["imd25"]
    hosp_df = data["hosp"]
    qof_access_df = data["qof_access"]
    gp_df = data["gp"]

    primary_lookup = lookup_df.sort_values("ObjectId").drop_duplicates(subset=["LSOA11CD"]).copy()

    master = samhi_df.merge(
        primary_lookup[["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22CD", "LAD22NM"]],
        left_on="lsoa11",
        right_on="LSOA11CD",
        how="left"
    )
    master["is_lincolnshire"] = master["LAD22NM"].isin(LINCOLNSHIRE_LADS).astype(int)

    # DERI features
    deri_cols = {
        "Index of Multiple Deprivation 2019 score - England base": "imd_2019_score",
        "Unemployment rate": "unemployment_rate",
        "Percentage of population aged 65 and over": "pct_aged_65_plus",
        "Percentage of residents whose day-to-day activities are limited": "pct_disability_limited",
        "Percentage of residents aged 16+ with no qualifications": "pct_no_qualifications",
        "Percentage of population in social grade DE": "pct_social_grade_de",
        "Guaranteed pension credit (rate per 1,000 aged 65+)": "pension_credit_rate",
        "Average download speed (Mbit/s)": "avg_download_speed",
        "Region name": "region_name"
    }
    deri_sub = deri_df[["LSOA code"] + list(deri_cols.keys())].rename(columns=deri_cols).copy()
    for col in deri_cols.values():
        if col != "region_name":
            deri_sub[col] = pd.to_numeric(deri_sub[col], errors="coerce")

    master = master.merge(deri_sub, left_on="lsoa11", right_on="LSOA code", how="left")
    master.drop(columns=["LSOA code"], inplace=True)

    # Rural-Urban & Isolation Scale
    ru_sub = ru_df[["LSOA21CD", "is_rural", "isolation_scale", "isolation_normalized"]].drop_duplicates(subset=["LSOA21CD"])
    master = master.merge(ru_sub, on="LSOA21CD", how="left")
    master["is_rural"] = master["is_rural"].fillna(0).astype(int)
    master["isolation_scale"] = master["isolation_scale"].fillna(3.0).astype(float)
    master["isolation_normalized"] = master["isolation_normalized"].fillna(0.4).astype(float)

    # IMD 2025
    if not imd25_df.empty:
        imd25_sub = imd25_df[["LSOA21CD", "imd_2025_rank", "imd_2025_decile"]].drop_duplicates(subset=["LSOA21CD"])
        master = master.merge(imd25_sub, on="LSOA21CD", how="left")
    else:
        master["imd_2025_rank"] = np.nan
        master["imd_2025_decile"] = np.nan

    # Hospital & GP Travel Times (National)
    if not hosp_df.empty:
        master = master.merge(hosp_df[["lsoa11", "hosp_pt_time", "hosp_car_time"]], on="lsoa11", how="left")
    else:
        master["hosp_pt_time"] = np.nan
        master["hosp_car_time"] = np.nan

    if not gp_df.empty:
        master = master.merge(gp_df[["lsoa11", "gp_pt_time", "gp_car_time"]], on="lsoa11", how="left")
    else:
        master["gp_pt_time"] = np.nan
        master["gp_car_time"] = np.nan

    # Household structure
    if not ts003_df.empty:
        code_c = "LSOA_Code" if "LSOA_Code" in ts003_df.columns else "Lower layer Super Output Areas Code"
        ts003_sub = ts003_df.copy()
        if {"LSOA21CD", "one_person_household_pct", "lone_parent_pct"}.issubset(ts003_sub.columns):
            ts003_sub = ts003_sub[["LSOA21CD", "one_person_household_pct", "lone_parent_pct"]].copy()
        elif "One-person household_Pct" in ts003_sub.columns:
            ts003_sub = ts003_sub[[code_c, "One-person household_Pct", "Single family household: Lone parent family_Pct"]].rename(columns={
                code_c: "LSOA21CD",
                "One-person household_Pct": "one_person_household_pct",
                "Single family household: Lone parent family_Pct": "lone_parent_pct"
            })
        else:
            ts003_sub = pd.DataFrame({"LSOA21CD": master["LSOA21CD"], "one_person_household_pct": np.nan, "lone_parent_pct": np.nan})
        master = master.merge(ts003_sub, on="LSOA21CD", how="left")
    else:
        master["one_person_household_pct"] = np.nan
        master["lone_parent_pct"] = np.nan

    # Car availability
    if not car_df.empty:
        try:
            car_df["code"] = car_df["Lower layer Super Output Areas Code"].astype(str).str.strip()
            car_piv = car_df.pivot(index="code", columns="Car or van availability (5 categories) Code", values="Observation").fillna(0)
            total_hh = car_piv[[0, 1, 2, 3]].sum(axis=1)
            no_car = (car_piv[0] / total_hh * 100).rename("no_car_pct").reset_index()
            master = master.merge(no_car, left_on="LSOA21CD", right_on="code", how="left").drop(columns=["code"])
        except Exception:
            master["no_car_pct"] = np.nan
    else:
        master["no_car_pct"] = np.nan

    # QOF measures used by the benchmark, allocated to LSOAs.
    qof_cols = ["qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"]
    if not qof_access_df.empty:
        master = master.merge(qof_access_df[["LSOA_CODE", *qof_cols]], left_on="LSOA21CD", right_on="LSOA_CODE", how="left")
        master.drop(columns=["LSOA_CODE"], inplace=True)
    else:
        for col in qof_cols:
            master[col] = np.nan

    # Median impute all numeric features
    all_num_covariates = [
        "imd_2019_score", "imd_2025_rank", "imd_2025_decile",
        "unemployment_rate", "pct_aged_65_plus", "pct_disability_limited",
        "pct_no_qualifications", "pct_social_grade_de", "pension_credit_rate", "avg_download_speed",
        "one_person_household_pct", "lone_parent_pct",
        "gp_pt_time", "gp_car_time", "hosp_pt_time", "hosp_car_time",
        "no_car_pct", "isolation_scale", "isolation_normalized",
        "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct",
    ]
    for col in all_num_covariates:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")
            med = master[col].median()
            master[col] = master[col].fillna(med if not pd.isna(med) else 0.0)

    return master


def create_training_panel(
    df: pd.DataFrame,
    spatial_neighbors: Dict[str, List[str]],
    start_year: int = 2014,
    end_year: int = 2022
) -> pd.DataFrame:
    """Build panel dataset of all historical transitions up to 2022 with the expanded feature set."""
    records = []
    lsoa21_to_idx = {code: idx for idx, code in enumerate(df["LSOA21CD"].values)}

    for t in range(start_year, end_year + 1):
        target_col = f"samhi_index.{t}"
        target_dec_col = f"samhi_dec.{t}"
        lag1_col = f"samhi_index.{t-1}"
        lag2_col = f"samhi_index.{t-2}"
        lag3_col = f"samhi_index.{t-3}"
        dec1_col = f"samhi_dec.{t-1}"

        if target_col not in df.columns or lag3_col not in df.columns:
            continue

        y_true = df[target_col].values
        d_true = df[target_dec_col].values if target_dec_col in df.columns else np.zeros(len(df))

        l1 = df[lag1_col].values
        l2 = df[lag2_col].values
        l3 = df[lag3_col].values
        d1 = df[dec1_col].values if dec1_col in df.columns else np.zeros(len(df))

        delta1 = l1 - l2
        delta2 = l2 - l3
        accel = delta1 - delta2

        stacked = np.column_stack([l1, l2, l3])
        roll_mean = np.mean(stacked, axis=1)
        roll_std = np.std(stacked, axis=1)
        roll_min = np.min(stacked, axis=1)
        roll_max = np.max(stacked, axis=1)

        df_temp = pd.DataFrame({"LAD22NM": df["LAD22NM"], "l1": l1})
        lad_means = df_temp.groupby("LAD22NM")["l1"].transform("mean").values

        spatial_lag_1 = np.zeros(len(df))
        spatial_delta_1 = np.zeros(len(df))

        if spatial_neighbors:
            for i, code in enumerate(df["LSOA21CD"].values):
                nbr_codes = spatial_neighbors.get(code, [])
                if nbr_codes:
                    nbr_indices = [lsoa21_to_idx[c] for c in nbr_codes if c in lsoa21_to_idx]
                    if nbr_indices:
                        spatial_lag_1[i] = np.mean(l1[nbr_indices])
                        spatial_delta_1[i] = np.mean(delta1[nbr_indices])
                    else:
                        spatial_lag_1[i] = l1[i]
                        spatial_delta_1[i] = delta1[i]
                else:
                    spatial_lag_1[i] = l1[i]
                    spatial_delta_1[i] = delta1[i]
        else:
            spatial_lag_1 = lad_means
            spatial_delta_1 = delta1

        year_records = pd.DataFrame({
            "lsoa11": df["lsoa11"],
            "LSOA21CD": df["LSOA21CD"],
            "LSOA21NM": df.get("LSOA21NM", ""),
            "LAD22NM": df["LAD22NM"],
            "is_lincolnshire": df["is_lincolnshire"],
            "year": t,
            "target": y_true,
            "target_decile": d_true,
            # Features
            "lag_1": l1,
            "lag_2": l2,
            "lag_3": l3,
            "delta_1": delta1,
            "delta_2": delta2,
            "acceleration": accel,
            "rolling_mean_3yr": roll_mean,
            "rolling_std_3yr": roll_std,
            "rolling_min_3yr": roll_min,
            "rolling_max_3yr": roll_max,
            "decile_lag_1": d1,
            "lad_mean_samhi_lag_1": lad_means,
            "spatial_lag_1": spatial_lag_1,
            "spatial_delta_1": spatial_delta_1,
            "imd_2019_score": df["imd_2019_score"],
            "imd_2025_decile": df["imd_2025_decile"],
            "unemployment_rate": df["unemployment_rate"],
            "pct_social_grade_de": df["pct_social_grade_de"],
            "pension_credit_rate": df["pension_credit_rate"],
            "pct_aged_65_plus": df["pct_aged_65_plus"],
            "pct_disability_limited": df["pct_disability_limited"],
            "pct_no_qualifications": df["pct_no_qualifications"],
            "one_person_household_pct": df["one_person_household_pct"],
            "lone_parent_pct": df["lone_parent_pct"],
            "is_rural": df["is_rural"],
            "isolation_scale": df["isolation_scale"],
            "isolation_normalized": df["isolation_normalized"],
            "avg_download_speed": df["avg_download_speed"],
            "gp_pt_time": df["gp_pt_time"],
            "gp_car_time": df["gp_car_time"],
            "hosp_pt_time": df["hosp_pt_time"],
            "hosp_car_time": df["hosp_car_time"],
            "no_car_pct": df["no_car_pct"],
            "qof_mh002_pct": df["qof_mh002_pct"],
            "qof_mh021_pct": df["qof_mh021_pct"],
            "qof_mh_pca_pct": df["qof_mh_pca_pct"],
            "qof_dep_pca_pct": df["qof_dep_pca_pct"],
            "qof_dep004_pct": df["qof_dep004_pct"]
        })
        records.append(year_records)

    return pd.concat(records, ignore_index=True)


MODEL_PREDICTION_SLUGS = {
    "Multimodal Ridge": "multimodal_ridge",
    "Multimodal ElasticNet": "multimodal_elasticnet",
    "Multimodal Random Forest": "multimodal_random_forest",
    "Multimodal Extra-Trees": "multimodal_extra-trees",
    "Multimodal LightGBM": "multimodal_lightgbm",
    "Multimodal XGBoost": "multimodal_xgboost",
    "Multimodal CatBoost": "multimodal_catboost",
    "Explainable Boosting Machine (EBM)": "explainable_boosting_machine_(ebm)",
    "Stacking Ensemble (Super Learner)": "stacking_ensemble_(super_learner)",
    "AR Baseline (Persistence)": "ar_baseline_persistence",
    "AR Baseline (Momentum Drift)": "ar_baseline_momentum_drift",
}


def model_slug(model_name: str) -> str:
    """Return the prediction-column suffix used by the benchmark and Flask page."""
    return MODEL_PREDICTION_SLUGS[model_name]


def train_models_for_projection(
    train_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[Dict[str, any], Dict[str, float]]:
    """Fit the benchmark model set on all historical data up to 2022."""
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    models = {
        "Multimodal Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=25.0, random_state=42))
        ]),
        "Multimodal ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("elastic", ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=42, max_iter=2000))
        ]),
        "Multimodal Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "Multimodal Extra-Trees": ExtraTreesRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "Multimodal LightGBM": lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.03, max_depth=5, num_leaves=20,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        ),
        "Multimodal XGBoost": xgb.XGBRegressor(
            n_estimators=250, learning_rate=0.03, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, tree_method="hist"
        ),
        "Multimodal CatBoost": cb.CatBoostRegressor(
            iterations=250, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
            random_seed=42, verbose=0, thread_count=-1
        ),
        "Explainable Boosting Machine (EBM)": ExplainableBoostingRegressor(
            interactions=0, max_bins=64, random_state=42, n_jobs=-1
        ),
        "Stacking Ensemble (Super Learner)": StackingRegressor(
            estimators=[
                ("lgb", lgb.LGBMRegressor(n_estimators=120, max_depth=5, learning_rate=0.04, verbosity=-1, random_state=42)),
                ("xgb", xgb.XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.04, random_state=42, n_jobs=-1, tree_method="hist")),
                ("cb", cb.CatBoostRegressor(iterations=120, depth=5, learning_rate=0.04, verbose=0, random_seed=42, thread_count=-1)),
                ("ridge", Ridge(alpha=25.0, random_state=42))
            ],
            final_estimator=Ridge(alpha=10.0, random_state=42),
            cv=3,
            n_jobs=-1
        )
    }

    trained = {}
    for name, model in models.items():
        logger.info(f"Fitting {name} on {len(train_df):,} historical observations (2014-2022)...")
        model.fit(X_train, y_train)
        trained[name] = model

    model_rmse = {}
    for name, model in trained.items():
        y_pred = model.predict(X_train)
        model_rmse[name] = float(np.sqrt(mean_squared_error(y_train, y_pred)))
        logger.info(f"Full-history {name} RMSE: {model_rmse[name]:.4f}")

    # Baseline residual scales are retained so every projected model can have
    # a comparable model-based uncertainty band in the map inspector.
    persistence = train_df["lag_1"].values
    momentum = persistence + train_df["delta_1"].values
    model_rmse["AR Baseline (Persistence)"] = float(np.sqrt(mean_squared_error(y_train, persistence)))
    model_rmse["AR Baseline (Momentum Drift)"] = float(np.sqrt(mean_squared_error(y_train, momentum)))

    return trained, model_rmse


def run_recursive_forward_projections(
    master_df: pd.DataFrame,
    spatial_neighbors: Dict[str, List[str]],
    trained_models: Dict[str, any],
    feature_cols: List[str],
    model_rmse: Dict[str, float],
    projection_years: List[int] = [2023, 2024, 2025]
) -> pd.DataFrame:
    """Recursively project each model from its own previous predictions."""
    model_names = (
        "Multimodal Ridge",
        "Multimodal ElasticNet",
        "Multimodal Random Forest",
        "Multimodal Extra-Trees",
        "Multimodal LightGBM",
        "Multimodal XGBoost",
        "Multimodal CatBoost",
        "Explainable Boosting Machine (EBM)",
        "Stacking Ensemble (Super Learner)",
        "AR Baseline (Persistence)",
        "AR Baseline (Momentum Drift)",
    )
    primary_model = "Multimodal ElasticNet"
    lsoa21_to_idx = {code: idx for idx, code in enumerate(master_df["LSOA21CD"].values)}
    n_areas = len(master_df)

    simulated_series = {
        name: {
            2020: master_df["samhi_index.2020"].values.copy(),
            2021: master_df["samhi_index.2021"].values.copy(),
            2022: master_df["samhi_index.2022"].values.copy(),
        }
        for name in model_names
    }

    baseline_2022 = master_df["samhi_index.2022"].values.copy()
    projection_records = []

    for t in projection_years:
        logger.info(f"Generating recursive forward projections for year {t}...")
        step_predictions = {}
        step_lag1 = {}

        for name in model_names:
            series = simulated_series[name]
            l1 = series[t - 1]
            l2 = series[t - 2]
            l3 = series[t - 3]
            delta1 = l1 - l2
            delta2 = l2 - l3
            accel = delta1 - delta2

            stacked = np.column_stack([l1, l2, l3])
            roll_mean = np.mean(stacked, axis=1)
            roll_std = np.std(stacked, axis=1)
            roll_min = np.min(stacked, axis=1)
            roll_max = np.max(stacked, axis=1)

            df_lad = pd.DataFrame({"LAD22NM": master_df["LAD22NM"], "l1": l1})
            lad_means = df_lad.groupby("LAD22NM")["l1"].transform("mean").values

            spatial_lag = np.zeros(n_areas)
            spatial_delta = np.zeros(n_areas)
            if spatial_neighbors:
                for i, code in enumerate(master_df["LSOA21CD"].values):
                    nbr_codes = spatial_neighbors.get(code, [])
                    nbr_indices = [lsoa21_to_idx[c] for c in nbr_codes if c in lsoa21_to_idx]
                    if nbr_indices:
                        spatial_lag[i] = np.mean(l1[nbr_indices])
                        spatial_delta[i] = np.mean(delta1[nbr_indices])
                    else:
                        spatial_lag[i] = l1[i]
                        spatial_delta[i] = delta1[i]
            else:
                spatial_lag = lad_means
                spatial_delta = delta1

            try:
                d1 = pd.qcut(l1, q=10, labels=False, duplicates="drop") + 1
            except Exception:
                d1 = np.ones(n_areas) * 5

            feat_df = pd.DataFrame({
                "lag_1": l1,
                "lag_2": l2,
                "lag_3": l3,
                "delta_1": delta1,
                "delta_2": delta2,
                "acceleration": accel,
                "rolling_mean_3yr": roll_mean,
                "rolling_std_3yr": roll_std,
                "rolling_min_3yr": roll_min,
                "rolling_max_3yr": roll_max,
                "decile_lag_1": d1,
                "lad_mean_samhi_lag_1": lad_means,
                "spatial_lag_1": spatial_lag,
                "spatial_delta_1": spatial_delta,
                "imd_2019_score": master_df["imd_2019_score"],
                "imd_2025_decile": master_df["imd_2025_decile"],
                "unemployment_rate": master_df["unemployment_rate"],
                "pct_social_grade_de": master_df["pct_social_grade_de"],
                "pension_credit_rate": master_df["pension_credit_rate"],
                "pct_aged_65_plus": master_df["pct_aged_65_plus"],
                "pct_disability_limited": master_df["pct_disability_limited"],
                "pct_no_qualifications": master_df["pct_no_qualifications"],
                "one_person_household_pct": master_df["one_person_household_pct"],
                "lone_parent_pct": master_df["lone_parent_pct"],
                "is_rural": master_df["is_rural"],
                "isolation_scale": master_df["isolation_scale"],
                "isolation_normalized": master_df["isolation_normalized"],
                "avg_download_speed": master_df["avg_download_speed"],
                "gp_pt_time": master_df["gp_pt_time"],
                "gp_car_time": master_df["gp_car_time"],
                "hosp_pt_time": master_df["hosp_pt_time"],
                "hosp_car_time": master_df["hosp_car_time"],
                "no_car_pct": master_df["no_car_pct"],
                "qof_mh002_pct": master_df["qof_mh002_pct"],
                "qof_mh021_pct": master_df["qof_mh021_pct"],
                "qof_mh_pca_pct": master_df["qof_mh_pca_pct"],
                "qof_dep_pca_pct": master_df["qof_dep_pca_pct"],
                "qof_dep004_pct": master_df["qof_dep004_pct"],
            })

            if name == "AR Baseline (Persistence)":
                pred = l1
            elif name == "AR Baseline (Momentum Drift)":
                pred = l1 + delta1
            else:
                pred = trained_models[name].predict(feat_df[feature_cols].values)
            simulated_series[name][t] = pred
            step_predictions[name] = pred
            step_lag1[name] = l1

        primary_pred = step_predictions[primary_model]
        primary_lag1 = step_lag1[primary_model]
        horizon = t - 2022
        step_records = pd.DataFrame({
            "lsoa11": master_df["lsoa11"],
            "LSOA21CD": master_df["LSOA21CD"],
            "LSOA21NM": master_df["LSOA21NM"],
            "LAD22NM": master_df["LAD22NM"],
            "is_lincolnshire": master_df["is_lincolnshire"],
            "year": t,
            "baseline_2022": baseline_2022,
            "projection_model": primary_model,
            "horizon_years": horizon,
        })

        # Store prediction, change, decile, and uncertainty columns for every
        # model. Legacy primary-model columns are retained below for the API
        # and existing downloaded results.
        for name, pred in step_predictions.items():
            slug = model_slug(name)
            try:
                decile = pd.qcut(pred, q=10, labels=False, duplicates="drop") + 1
            except Exception:
                decile = np.ones(n_areas) * 5
            ci_bound = 1.96 * model_rmse[name] * np.sqrt(horizon)
            step_records[f"pred_{slug}"] = pred
            step_records[f"projected_change_{slug}_vs_2022"] = pred - baseline_2022
            step_records[f"projected_annual_change_{slug}"] = pred - step_lag1[name]
            step_records[f"projected_decile_{slug}"] = decile
            step_records[f"ci_lower_95_{slug}"] = pred - ci_bound
            step_records[f"ci_upper_95_{slug}"] = pred + ci_bound

        primary_slug = model_slug(primary_model)
        step_records["projected_change_vs_2022"] = step_records[f"projected_change_{primary_slug}_vs_2022"]
        step_records["projected_annual_change"] = step_records[f"projected_annual_change_{primary_slug}"]
        step_records["projected_decile"] = step_records[f"projected_decile_{primary_slug}"]
        step_records["ci_lower_95"] = step_records[f"ci_lower_95_{primary_slug}"]
        step_records["ci_upper_95"] = step_records[f"ci_upper_95_{primary_slug}"]
        projection_records.append(step_records)

    return pd.concat(projection_records, ignore_index=True)

def expand_projections_to_all_2021_lsoas(
    proj_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    scope: str = "lincolnshire"
) -> pd.DataFrame:
    lookup_21 = lookup_df.drop_duplicates(subset=["LSOA21CD"]).copy()
    if scope == "lincolnshire":
        lookup_21 = lookup_21[lookup_21["LAD22NM"].isin(LINCOLNSHIRE_LADS)]

    cols_to_keep = [c for c in proj_df.columns if c not in ["LSOA21CD", "LSOA21NM", "LAD22NM", "ObjectId"]]

    expanded = lookup_21[["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22NM"]].merge(
        proj_df[cols_to_keep],
        left_on="LSOA11CD",
        right_on="lsoa11",
        how="inner"
    )

    duplicate_keys = int(expanded.duplicated(["LSOA21CD", "year"]).sum())
    if duplicate_keys:
        original_columns = list(expanded.columns)
        value_columns = [c for c in original_columns if c not in ["LSOA21CD", "year"]]
        aggregations = {}
        for column in value_columns:
            aggregations[column] = "mean" if pd.api.types.is_numeric_dtype(expanded[column]) else "first"
        expanded = expanded.groupby(["LSOA21CD", "year"], as_index=False, sort=False).agg(aggregations)
        expanded = expanded[[c for c in original_columns if c in expanded.columns]]
        logger.warning("Collapsed %s duplicate LSOA21-year projection rows using mean/first aggregation", duplicate_keys)

    logger.info(f"[{scope.upper()}] Expanded {len(proj_df):,} raw projections to {len(expanded):,} 2021 LSOA boundaries ({expanded['LSOA21CD'].nunique()} unique areas)")
    return expanded


def generate_summary_kpis(proj_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    summaries = []
    for yr in [2023, 2024, 2025]:
        sub = proj_df[proj_df["year"] == yr]
        mean_score = sub["pred_multimodal_elasticnet"].mean()
        mean_change = sub["projected_change_vs_2022"].mean()
        pct_worsening = (sub["projected_change_vs_2022"] > 0).mean() * 100.0
        pct_improving = (sub["projected_change_vs_2022"] < 0).mean() * 100.0
        num_high_risk = (sub["projected_decile"] >= 9).sum()

        summaries.append({
            "Scope": scope,
            "Year": yr,
            "Mean_Projected_SAMHI": mean_score,
            "Mean_Change_vs_2022": mean_change,
            "Pct_Areas_Worsening": pct_worsening,
            "Pct_Areas_Improving": pct_improving,
            "Total_LSOAs": len(sub),
            "High_Risk_Decile_9_10_Count": num_high_risk
        })
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description="SAMHI Forward Projections (2023-2025)")
    parser.add_argument("--scope", choices=["lincolnshire", "national", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default="scripts/machine_learning/samhi/results")
    args = parser.parse_args()

    project_root = get_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_datasets(project_root)
    spatial_neighbors = compute_spatial_neighbors(data["geojson_path"])
    master_df = build_master_frame(data)

    panel_df = create_training_panel(master_df, spatial_neighbors, start_year=2014, end_year=2022)

    feature_cols = [
        "lag_1", "lag_2", "lag_3",
        "delta_1", "delta_2", "acceleration",
        "rolling_mean_3yr", "rolling_std_3yr", "rolling_min_3yr", "rolling_max_3yr",
        "decile_lag_1",
        "lad_mean_samhi_lag_1", "spatial_lag_1", "spatial_delta_1",
        "imd_2019_score", "imd_2025_decile", "unemployment_rate", "pct_social_grade_de", "pension_credit_rate",
        "pct_aged_65_plus", "pct_disability_limited", "pct_no_qualifications",
        "one_person_household_pct", "lone_parent_pct",
        "is_rural", "isolation_scale", "isolation_normalized",
        "avg_download_speed", "gp_pt_time", "gp_car_time", "hosp_pt_time", "hosp_car_time", "no_car_pct",
    ]

    scopes = ["lincolnshire", "national"] if args.scope == "both" else [args.scope]
    all_summaries = []

    for sc in scopes:
        logger.info(f"\n=======================================================")
        logger.info(f" RUNNING FORWARD PROJECTIONS: {sc.upper()} (2023 - 2025)")
        logger.info(f"=======================================================")

        if sc == "lincolnshire":
            train_pool = panel_df[panel_df["is_lincolnshire"] == 1].copy()
            scope_master = master_df[master_df["is_lincolnshire"] == 1].copy().reset_index(drop=True)
            sc_neighbors = spatial_neighbors
        else:
            train_pool = panel_df.copy()
            scope_master = master_df.copy().reset_index(drop=True)
            sc_neighbors = {}

        trained_models, model_rmse = train_models_for_projection(train_pool, feature_cols)

        proj_raw = run_recursive_forward_projections(
            scope_master,
            spatial_neighbors=sc_neighbors,
            trained_models=trained_models,
            feature_cols=feature_cols,
            model_rmse=model_rmse,
            projection_years=[2023, 2024, 2025]
        )

        proj_expanded = expand_projections_to_all_2021_lsoas(proj_raw, data["lookup"], scope=sc)

        out_csv = output_dir / f"forward_projections_2023_2025_{sc}.csv"
        proj_expanded.to_csv(out_csv, index=False)
        logger.info(f"Saved forward projections to {out_csv}")

        summary_kpis = generate_summary_kpis(proj_expanded, sc)
        all_summaries.append(summary_kpis)
        print(f"\nForward Projection Trend Summary ({sc.upper()}):")
        print(summary_kpis.to_string(index=False))

    if all_summaries:
        comb_summary = pd.concat(all_summaries, ignore_index=True)
        comb_summary_path = output_dir / "forward_projections_summary.csv"
        comb_summary.to_csv(comb_summary_path, index=False)
        logger.info(f"Saved combined projection summary to {comb_summary_path}")

    logger.info("Phase 3 Forward Projections with new national features completed successfully!")


if __name__ == "__main__":
    main()

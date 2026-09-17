"""
02_multimodal_features.py
-------------------------
Phase 2: Multimodal & Spatial Machine Learning Pipeline for SAMHI Forecasting.

Integrates:
  1. Temporal autoregressive signals (lags, momentum, acceleration, rolling 3yr stats)
  2. Spatial neighbor contiguity spillovers (spatial_lag_1, spatial_delta_1 from Queen adjacency)
  3. Local Authority District (LAD) fixed effects & clusters (lad_mean_samhi_lag_1)
  4. Socioeconomic deprivation: IMD 2019 score, IMD 2025 Decile & Rank, unemployment, social grade DE, pension credit
  5. Demographic vulnerability: Age 65+ %, chronic disability/illness %, no qualifications %
  6. Household structure: TS003 single occupancy %, lone parent %
  7. Healthcare & secondary hospital access: GP travel times (PT & Car), Hospital travel times (PT & Car)
  8. Geographic isolation: Rural/urban flag, Isolation Scale & Isolation Normalized
  9. Digital & transport infrastructure: Broadband download speeds, vehicle non-ownership %

Evaluates:
  - Naive Persistence Baseline (y_{t-1})
  - Naive Momentum Drift (y_{t-1} + Delta y)
  - Multimodal Ridge Regression
  - Multimodal Random Forest
  - Multimodal LightGBM
  - TreeSHAP and LinearSHAP feature attribution for LightGBM and ElasticNet

Exports results to scripts/machine_learning/samhi/results/
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from interpret.glassbox import ExplainableBoostingRegressor
import shap

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
logger = logging.getLogger("SAMHI_Multimodal")

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


# File stems for model-specific feature explanations exposed by the Flask page.
SHAP_FILE_STEMS = {
    "Multimodal Ridge": "ridge",
    "Multimodal ElasticNet": "elasticnet",
    "Multimodal Random Forest": "random_forest",
    "Multimodal Extra-Trees": "extra-trees",
    "Multimodal LightGBM": "lightgbm",
    "Multimodal XGBoost": "xgboost",
    "Multimodal CatBoost": "catboost",
    "Explainable Boosting Machine (EBM)": "explainable_boosting_machine_(ebm)",
    "Stacking Ensemble (Super Learner)": "stacking_ensemble_(super_learner)",
}


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parents[3]


def load_all_datasets(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load national SAMHI and multimodal auxiliary datasets including national hospital, GP, and IMD 2025 data."""
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

    # QOF 2024-25 achievement/PCA rates, allocated to LSOAs using the same
    # patient-registration weights as the dashboard maps.
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
            logger.warning("Could not load QOF model features: %s", exc)

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
    gp_df = data["gp"]
    qof_access_df = data["qof_access"]

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

    # QOF measures used by the Access maps, rolled up to LSOA.
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
        "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"
    ]
    for col in all_num_covariates:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")
            med = master[col].median()
            master[col] = master[col].fillna(med if not pd.isna(med) else 0.0)

    return master


def create_panel_dataset(
    df: pd.DataFrame,
    spatial_neighbors: Dict[str, List[str]],
    start_year: int = 2014,
    end_year: int = 2022
) -> pd.DataFrame:
    """Construct longitudinal panel dataset with sliding lag windows, momentum, and spatial spillovers."""
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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_lag1: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R2, Directional Accuracy, and Decile Accuracy."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    actual_change = y_true - y_lag1
    pred_change = y_pred - y_lag1
    correct_dir = ((actual_change * pred_change) > 0) | ((actual_change == 0) & (pred_change == 0))
    dir_acc = float(np.mean(correct_dir)) * 100.0

    try:
        d_true = pd.qcut(y_true, q=10, labels=False, duplicates="drop") + 1
        d_pred = pd.qcut(y_pred, q=10, labels=False, duplicates="drop") + 1
        dec_exact = float(np.mean(d_true == d_pred)) * 100.0
        dec_within1 = float(np.mean(np.abs(d_true - d_pred) <= 1)) * 100.0
    except Exception:
        dec_exact = 0.0
        dec_within1 = 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Dir_Acc_%": dir_acc,
        "Decile_Exact_%": dec_exact,
        "Decile_Within_1_%": dec_within1
    }


def compute_shap_importance(
    model: any,
    X_sample: np.ndarray,
    feature_names: List[str]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute TreeSHAP attribution and aggregate by structural domain."""
    logger.info("Computing TreeSHAP feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = np.sum(mean_abs_shap) if np.sum(mean_abs_shap) > 0 else 1.0
    importance_pct = (mean_abs_shap / total_shap) * 100.0

    domain_mapping = {
        "lag_1": "1_Temporal_History",
        "lag_2": "1_Temporal_History",
        "lag_3": "1_Temporal_History",
        "delta_1": "1_Temporal_History",
        "delta_2": "1_Temporal_History",
        "acceleration": "1_Temporal_History",
        "rolling_mean_3yr": "1_Temporal_History",
        "rolling_std_3yr": "1_Temporal_History",
        "rolling_min_3yr": "1_Temporal_History",
        "rolling_max_3yr": "1_Temporal_History",
        "decile_lag_1": "1_Temporal_History",
        "lad_mean_samhi_lag_1": "2_Geography_Accessibility",
        "spatial_lag_1": "2_Geography_Accessibility",
        "spatial_delta_1": "2_Geography_Accessibility",
        "is_rural": "2_Geography_Accessibility",
        "isolation_scale": "2_Geography_Accessibility",
        "isolation_normalized": "2_Geography_Accessibility",
        "gp_pt_time": "2_Geography_Accessibility",
        "gp_car_time": "2_Geography_Accessibility",
        "hosp_pt_time": "2_Geography_Accessibility",
        "hosp_car_time": "2_Geography_Accessibility",
        "no_car_pct": "2_Geography_Accessibility",
        "avg_download_speed": "2_Geography_Accessibility",
        "imd_2019_score": "3_Deprivation_Economics",
        "imd_2025_decile": "3_Deprivation_Economics",
        "unemployment_rate": "3_Deprivation_Economics",
        "pct_social_grade_de": "3_Deprivation_Economics",
        "pension_credit_rate": "3_Deprivation_Economics",
        "pct_aged_65_plus": "4_Demographics_Vulnerability",
        "pct_disability_limited": "4_Demographics_Vulnerability",
        "pct_no_qualifications": "4_Demographics_Vulnerability",
        "one_person_household_pct": "4_Demographics_Vulnerability",
        "lone_parent_pct": "4_Demographics_Vulnerability"
    }

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "importance_pct": importance_pct,
        "domain": [domain_mapping.get(f, "5_Other") for f in feature_names]
    }).sort_values(by="importance_pct", ascending=False).reset_index(drop=True)

    domain_summary = df_shap.groupby("domain")["importance_pct"].sum().to_dict()
    return df_shap, domain_summary



def compute_linear_shap_importance(
    model: Pipeline,
    X_sample: np.ndarray,
    feature_names: List[str],
    estimator_step: str = "elastic",
    model_label: str = "linear model",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute LinearSHAP attribution for a scaled linear pipeline."""
    logger.info("Computing LinearSHAP feature importance for %s...", model_label)
    scaler = model.named_steps["scaler"]
    estimator = model.named_steps[estimator_step]
    X_scaled = scaler.transform(X_sample)

    try:
        masker = shap.maskers.Independent(X_scaled, max_samples=len(X_scaled))
        explainer = shap.LinearExplainer(estimator, masker)
        shap_values = explainer.shap_values(X_scaled)
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as exc:
        logger.warning("LinearSHAP failed; using exact linear contributions instead: %s", exc)
        shap_values = (X_scaled - X_scaled.mean(axis=0)) * estimator.coef_

    shap_values = np.asarray(shap_values)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = np.sum(mean_abs_shap) if np.sum(mean_abs_shap) > 0 else 1.0
    importance_pct = (mean_abs_shap / total_shap) * 100.0

    domain_mapping = {
        "lag_1": "1_Temporal_History",
        "lag_2": "1_Temporal_History",
        "lag_3": "1_Temporal_History",
        "delta_1": "1_Temporal_History",
        "delta_2": "1_Temporal_History",
        "acceleration": "1_Temporal_History",
        "rolling_mean_3yr": "1_Temporal_History",
        "rolling_std_3yr": "1_Temporal_History",
        "rolling_min_3yr": "1_Temporal_History",
        "rolling_max_3yr": "1_Temporal_History",
        "decile_lag_1": "1_Temporal_History",
        "lad_mean_samhi_lag_1": "2_Geography_Accessibility",
        "spatial_lag_1": "2_Geography_Accessibility",
        "spatial_delta_1": "2_Geography_Accessibility",
        "is_rural": "2_Geography_Accessibility",
        "isolation_scale": "2_Geography_Accessibility",
        "isolation_normalized": "2_Geography_Accessibility",
        "gp_pt_time": "2_Geography_Accessibility",
        "gp_car_time": "2_Geography_Accessibility",
        "hosp_pt_time": "2_Geography_Accessibility",
        "hosp_car_time": "2_Geography_Accessibility",
        "no_car_pct": "2_Geography_Accessibility",
        "avg_download_speed": "2_Geography_Accessibility",
        "imd_2019_score": "3_Deprivation_Economics",
        "imd_2025_decile": "3_Deprivation_Economics",
        "unemployment_rate": "3_Deprivation_Economics",
        "pct_social_grade_de": "3_Deprivation_Economics",
        "pension_credit_rate": "3_Deprivation_Economics",
        "pct_aged_65_plus": "4_Demographics_Vulnerability",
        "pct_disability_limited": "4_Demographics_Vulnerability",
        "pct_no_qualifications": "4_Demographics_Vulnerability",
        "one_person_household_pct": "4_Demographics_Vulnerability",
        "lone_parent_pct": "4_Demographics_Vulnerability",
        "qof_mh002_pct": "5_Healthcare_Service",
        "qof_mh021_pct": "5_Healthcare_Service",
        "qof_mh_pca_pct": "5_Healthcare_Service",
        "qof_dep_pca_pct": "5_Healthcare_Service",
        "qof_dep004_pct": "5_Healthcare_Service",
    }

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "importance_pct": importance_pct,
        "domain": [domain_mapping.get(f, "6_Other") for f in feature_names]
    }).sort_values(by="importance_pct", ascending=False).reset_index(drop=True)
    domain_summary = df_shap.groupby("domain")["importance_pct"].sum().to_dict()
    return df_shap, domain_summary

def compute_permutation_importance(
    model: any,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_names: List[str],
    model_label: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute model-specific permutation importance for non-tree models."""
    logger.info("Computing permutation feature importance for %s...", model_label)
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    mean_abs_importance = np.abs(result.importances_mean)
    total_importance = np.sum(mean_abs_importance) if np.sum(mean_abs_importance) > 0 else 1.0
    importance_pct = (mean_abs_importance / total_importance) * 100.0
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_importance,
        "importance_pct": importance_pct,
        "domain": ["Model-specific permutation importance"] * len(feature_names),
    }).sort_values(by="importance_pct", ascending=False).reset_index(drop=True)
    domain_summary = df_importance.groupby("domain")["importance_pct"].sum().to_dict()
    return df_importance, domain_summary


def run_scope_benchmark(
    panel_df: pd.DataFrame,
    scope: str,
    feature_cols: List[str],
    lookup_df: pd.DataFrame,
    output_dir: Path,
    experiment_set: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute one temporal benchmark with TreeSHAP and LinearSHAP."""
    if experiment_set == 1:
        train_years, val_year, test_years = (2014, 2018), 2019, (2020, 2022)
    elif experiment_set == 2:
        train_years, val_year, test_years = (2014, 2016), 2017, (2018, 2019)
    else:
        raise ValueError("experiment_set must be 1 or 2")
    logger.info(f"\n=======================================================")
    logger.info(f" EXECUTING MULTIMODAL BENCHMARK: {scope.upper()}")
    logger.info(f"=======================================================")

    if scope == "lincolnshire":
        df = panel_df[panel_df["is_lincolnshire"] == 1].copy()
    else:
        df = panel_df.copy()

    train_df = df[df["year"].between(*train_years)]
    val_df = df[df["year"] == val_year]
    test_df = df[df["year"].between(*test_years)].copy()

    logger.info(f"Splits for {scope}: Train={len(train_df):,} rows ({train_years[0]}-{train_years[1]}), Val={len(val_df):,} rows ({val_year}), Test={len(test_df):,} rows ({test_years[0]}-{test_years[1]})")

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values
    y_val_lag1 = val_df["lag_1"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values
    y_test_lag1 = test_df["lag_1"].values

    # Full Machine Learning Model Tournament Lineup
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
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),
        "Multimodal Extra-Trees": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),
        "Multimodal LightGBM": lgb.LGBMRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=20,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        ),
        "Multimodal XGBoost": xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ),
        "Multimodal CatBoost": cb.CatBoostRegressor(
            iterations=250,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            thread_count=-1
        ),
        "Explainable Boosting Machine (EBM)": ExplainableBoostingRegressor(
            interactions=5 if scope == "lincolnshire" else 0,
            max_bins=128 if scope == "lincolnshire" else 64,
            random_state=42,
            n_jobs=-1
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

    metrics_records = []

    # 1. Naive Persistence Baseline
    val_label = f"Validation ({val_year})"
    test_label = f"Test ({test_years[0]}-{test_years[1]} Aggregate)"
    metrics_records.append({"Scope": scope, "Split": val_label, "Model": "AR Baseline (Persistence)", **evaluate_predictions(y_val, y_val_lag1, y_val_lag1)})
    metrics_records.append({"Scope": scope, "Split": test_label, "Model": "AR Baseline (Persistence)", **evaluate_predictions(y_test, y_test_lag1, y_test_lag1)})
    for yr in range(test_years[0], test_years[1] + 1):
        sub_yr = test_df[test_df["year"] == yr]
        metrics_records.append({"Scope": scope, "Split": f"Test ({yr})", "Model": "AR Baseline (Persistence)", **evaluate_predictions(sub_yr["target"].values, sub_yr["lag_1"].values, sub_yr["lag_1"].values)})

    # 2. Naive Momentum Drift Baseline
    y_val_drift = val_df["lag_1"].values + val_df["delta_1"].values
    y_test_drift = test_df["lag_1"].values + test_df["delta_1"].values
    metrics_records.append({"Scope": scope, "Split": val_label, "Model": "AR Baseline (Momentum Drift)", **evaluate_predictions(y_val, y_val_drift, y_val_lag1)})
    metrics_records.append({"Scope": scope, "Split": test_label, "Model": "AR Baseline (Momentum Drift)", **evaluate_predictions(y_test, y_test_drift, y_test_lag1)})
    for yr in range(test_years[0], test_years[1] + 1):
        sub_yr = test_df[test_df["year"] == yr]
        sub_drift = sub_yr["lag_1"].values + sub_yr["delta_1"].values
        metrics_records.append({"Scope": scope, "Split": f"Test ({yr})", "Model": "AR Baseline (Momentum Drift)", **evaluate_predictions(sub_yr["target"].values, sub_drift, sub_yr["lag_1"].values)})

    test_preds_df = test_df[["lsoa11", "LSOA21CD", "LSOA21NM", "LAD22NM", "is_lincolnshire", "year", "target", "lag_1", "delta_1", "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"]].rename(columns={"target": "actual_samhi"}).copy()
    test_preds_df["pred_ar_baseline_persistence"] = test_preds_df["lag_1"]
    test_preds_df["err_ar_baseline_persistence"] = test_preds_df["pred_ar_baseline_persistence"] - test_preds_df["actual_samhi"]
    test_preds_df["pred_ar_baseline_momentum_drift"] = test_preds_df["lag_1"] + test_preds_df["delta_1"]
    test_preds_df["err_ar_baseline_momentum_drift"] = test_preds_df["pred_ar_baseline_momentum_drift"] - test_preds_df["actual_samhi"]

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name} on {scope}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Val
        val_pred = model.predict(X_val)
        metrics_records.append({"Scope": scope, "Split": val_label, "Model": name, **evaluate_predictions(y_val, val_pred, y_val_lag1)})

        # Test
        test_pred = model.predict(X_test)
        metrics_records.append({"Scope": scope, "Split": test_label, "Model": name, **evaluate_predictions(y_test, test_pred, y_test_lag1)})

        col_slug = name.lower().replace(" ", "_")
        test_preds_df[f"pred_{col_slug}"] = test_pred
        test_preds_df[f"err_{col_slug}"] = test_pred - y_test

        for yr in range(test_years[0], test_years[1] + 1):
            sub_yr = test_df[test_df["year"] == yr]
            sub_pred = model.predict(sub_yr[feature_cols].values)
            metrics_records.append({"Scope": scope, "Split": f"Test ({yr})", "Model": name, **evaluate_predictions(sub_yr["target"].values, sub_pred, sub_yr["lag_1"].values)})

    metrics_df = pd.DataFrame(metrics_records)

    # Generate a model-specific explanation for every fitted model.
    shap_sample = X_test[:1000] if len(X_test) > 1000 else X_test
    y_shap_sample = y_test[:len(shap_sample)]
    model_explanations = {}
    tree_model_names = [
        "Multimodal Random Forest",
        "Multimodal Extra-Trees",
        "Multimodal LightGBM",
        "Multimodal XGBoost",
        "Multimodal CatBoost",
    ]
    for model_name in tree_model_names:
        try:
            model_explanations[model_name] = compute_shap_importance(
                trained_models[model_name], shap_sample, feature_cols
            )
        except Exception as exc:
            logger.warning("TreeSHAP failed for %s: %s", model_name, exc)

    model_explanations["Multimodal Ridge"] = compute_linear_shap_importance(
        trained_models["Multimodal Ridge"], shap_sample, feature_cols,
        estimator_step="ridge", model_label="Ridge"
    )
    model_explanations["Multimodal ElasticNet"] = compute_linear_shap_importance(
        trained_models["Multimodal ElasticNet"], shap_sample, feature_cols,
        estimator_step="elastic", model_label="ElasticNet"
    )
    for model_name in ["Explainable Boosting Machine (EBM)", "Stacking Ensemble (Super Learner)"]:
        model_explanations[model_name] = compute_permutation_importance(
            trained_models[model_name], shap_sample, y_shap_sample, feature_cols, model_name
        )

    df_shap, lgb_domain_summary = model_explanations["Multimodal LightGBM"]
    df_shap_elasticnet, elasticnet_domain_summary = model_explanations["Multimodal ElasticNet"]

    logger.info(f"\nLightGBM TreeSHAP Importance by Domain ({scope.upper()}):")
    for dom, pct in sorted(lgb_domain_summary.items()):
        print(f"  {dom}: {pct:.2f}%")
    logger.info(f"\nElasticNet LinearSHAP Importance by Domain ({scope.upper()}):")
    for dom, pct in sorted(elasticnet_domain_summary.items()):
        print(f"  {dom}: {pct:.2f}%")

    # Expand predictions to all 2021 LSOA polygons
    lookup_21 = lookup_df.drop_duplicates(subset=["LSOA21CD"]).copy()
    if scope == "lincolnshire":
        lookup_21 = lookup_21[lookup_21["LAD22NM"].isin(LINCOLNSHIRE_LADS)]
    
    cols_to_merge = [c for c in test_preds_df.columns if c not in ["LSOA21CD", "LSOA21NM", "LAD22NM", "ObjectId"]]
    expanded_preds = lookup_21[["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22NM"]].merge(
        test_preds_df[cols_to_merge],
        left_on="LSOA11CD",
        right_on="lsoa11",
        how="inner"
    )

    # Export
    suffix = "2020_2022" if experiment_set == 1 else "pre_covid_2018_2019"
    metrics_df.to_csv(output_dir / f"multimodal_metrics_{suffix}_{scope}.csv", index=False)
    expanded_preds.to_csv(output_dir / f"multimodal_predictions_{suffix}_{scope}.csv", index=False)
    # Keep the legacy LightGBM filename and write explicit model-specific files.
    df_shap.to_csv(output_dir / f"shap_feature_importance_{suffix}_{scope}.csv", index=False)
    for model_name, (df_model_explanation, _) in model_explanations.items():
        stem = SHAP_FILE_STEMS[model_name]
        df_model_explanation.to_csv(
            output_dir / f"shap_feature_importance_{stem}_{suffix}_{scope}.csv", index=False
        )
    df_shap_elasticnet.to_csv(output_dir / f"shap_feature_importance_elasticnet_{suffix}_{scope}.csv", index=False)

    return metrics_df, expanded_preds, df_shap


def main():
    parser = argparse.ArgumentParser(description="SAMHI Multimodal & Spatial Machine Learning Pipeline")
    parser.add_argument("--scope", choices=["lincolnshire", "national", "both"], default="both")
    parser.add_argument("--experiment-set", type=int, choices=[1, 2], default=1,
                        help="1=COVID-era test (2020-2022), 2=pre-COVID test (2018-2019)")
    parser.add_argument("--output-dir", type=str, default="scripts/machine_learning/samhi/results")
    args = parser.parse_args()

    project_root = get_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_datasets(project_root)
    spatial_neighbors = compute_spatial_neighbors(data["geojson_path"])
    master_df = build_master_frame(data)

    panel_df = create_panel_dataset(master_df, spatial_neighbors, start_year=2014, end_year=2022)

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
        "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"
    ]

    scopes = ["lincolnshire", "national"] if args.scope == "both" else [args.scope]
    all_metrics = []

    for sc in scopes:
        m_df, _, _ = run_scope_benchmark(panel_df, sc, feature_cols, data["lookup"], output_dir, args.experiment_set)
        all_metrics.append(m_df)

    if len(all_metrics) > 1:
        comp_df = pd.concat(all_metrics, ignore_index=True)
        suffix = "2020_2022" if args.experiment_set == 1 else "pre_covid_2018_2019"
        comparison_path = output_dir / f"multimodal_metrics_comparison_{suffix}.csv"
        comp_df.to_csv(comparison_path, index=False)
        logger.info(f"Combined metrics saved to {comparison_path}")

    logger.info("Phase 2 Multimodal Pipeline with new national features completed successfully!")


if __name__ == "__main__":
    main()

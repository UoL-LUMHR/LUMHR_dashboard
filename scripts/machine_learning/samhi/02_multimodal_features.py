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

from _shared import project_root as get_project_root

# Reuse the dashboard's tested practice-to-LSOA QOF allocation functions.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flask"))
from core.allocation import (  # noqa: E402
    allocate_access_rates_to_lsoa,
    prepare_depression,
    prepare_mapping,
    prepare_smi,
)
from external_features import (  # noqa: E402
    ADDED_FEATURE_COLUMNS,
    HOUSEHOLD_FEATURE_COLUMNS,
    load_added_features,
    load_household_features,
    lagged_gas_feature,
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

POPULATION_SOURCE = (
    "scripts/utils/source_data/lsoa_population_estimates/"
    "Mid-2022 revised (Nov 2025) to mid-2024/sapelsoasyoa20222024.xlsx"
)

# 2021 RUC codes classified as nearer to a major town/city:
# Urban (UN1), larger rural (RLN1), and smaller rural (RSN1).
RUC_NEARER_CODES = {"UN1", "RLN1", "RSN1"}

SAMHI_HISTORY_FEATURES = {
    "lag_1", "lag_2", "lag_3",
    "delta_1", "delta_2", "acceleration",
    "rolling_mean_3yr", "rolling_std_3yr", "rolling_min_3yr", "rolling_max_3yr",
    "decile_lag_1",
    "lad_mean_samhi_lag_1", "spatial_lag_1", "spatial_delta_1",
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


def load_all_datasets(project_root: Path) -> Dict[str, pd.DataFrame]:
    """Load national SAMHI and multimodal auxiliary datasets including national hospital, GP, and IMD 2025 data."""
    datasets_dir = project_root / "datasets"

    samhi_path = datasets_dir / "samhi" / "samhi_21_01_v5.00_2011_2022_LSOA.csv"
    lookup_path = (
        datasets_dir
        / "lsoa"
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

    car_path = datasets_dir / "car_or_van_availability" / "TS045-2021-4-filtered-2026-11-02T14_52_56Z.csv"
    geojson_path = datasets_dir / "lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson"

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
    ru_codes = ru_df["RUC21CD"].astype(str).str.strip().str.upper()
    ru_df["ruc_nearer_population_share"] = ru_codes.isin(RUC_NEARER_CODES).astype(float)
    known_ruc_codes = RUC_NEARER_CODES | {"UF1", "RLF1", "RSF1"}
    unknown_ruc_codes = set(ru_codes.dropna().unique()) - known_ruc_codes
    if unknown_ruc_codes:
        logger.warning("Unmapped 2021 RUC codes: %s", sorted(unknown_ruc_codes))

    ts003_df = load_household_features(ts003_path, include_weight=True)
    added_features = load_added_features(datasets_dir)

    population_path = project_root / POPULATION_SOURCE
    if not population_path.exists():
        raise FileNotFoundError(f"Required 2021-LSOA population weights not found: {population_path}")
    population_weights = pd.read_excel(
        population_path,
        sheet_name="Mid-2024 LSOA 2021",
        skiprows=3,
        usecols=["LSOA 2021 Code", "Total"],
        engine="openpyxl",
    ).rename(columns={"LSOA 2021 Code": "LSOA21CD", "Total": "__population_weight"})
    population_weights["LSOA21CD"] = population_weights["LSOA21CD"].astype(str).str.strip()
    population_weights["__population_weight"] = pd.to_numeric(
        population_weights["__population_weight"], errors="coerce"
    )

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
            patient_weights = mapping.groupby("LSOA_CODE", as_index=False)["NUMBER_OF_PATIENTS"].sum()
            patient_weights = patient_weights.rename(columns={"NUMBER_OF_PATIENTS": "__patient_weight"})
            qof_access_df = qof_access_df.merge(patient_weights, on="LSOA_CODE", how="left")
            logger.info("Loaded QOF 2024-25 rates for %s unique LSOAs", len(qof_access_df))
        except Exception as exc:
            logger.warning("Could not load QOF model features: %s", exc)

    return {
        "samhi": samhi_df,
        "lookup": lookup_df,
        "deri": deri_df,
        "rural_urban": ru_df,
        "ts003": ts003_df,
        "population_weights": population_weights,
        "car": car_df,
        "imd25": imd25_df,
        "hosp": hosp_df,
        "gp": gp_df,
        "qof_access": qof_access_df,
        "added_features": added_features,
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


def aggregate_2021_features_to_2011(
    frame: pd.DataFrame,
    lookup_df: pd.DataFrame,
    feature_columns: List[str],
    weight_col: str,
    binary_majority_columns: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """Roll LSOA21 feature rates up to their 2011 parent using explicit weights."""
    output_columns = ["LSOA11CD", *feature_columns]
    if frame.empty or not feature_columns:
        return pd.DataFrame(columns=output_columns)

    mapping = lookup_df[["LSOA11CD", "LSOA21CD"]].drop_duplicates()
    source = frame[["LSOA21CD", *feature_columns, weight_col]].copy()
    source = source.merge(mapping, on="LSOA21CD", how="left")
    source[weight_col] = pd.to_numeric(source[weight_col], errors="coerce")
    result = pd.DataFrame({"LSOA11CD": source["LSOA11CD"].dropna().drop_duplicates()})

    for feature in feature_columns:
        values = pd.to_numeric(source[feature], errors="coerce")
        weights = source[weight_col]
        valid = values.notna() & weights.gt(0) & source["LSOA11CD"].notna()
        numerator = (values[valid] * weights[valid]).groupby(source.loc[valid, "LSOA11CD"]).sum()
        denominator = weights[valid].groupby(source.loc[valid, "LSOA11CD"]).sum()
        aggregated = numerator.div(denominator)
        if feature in binary_majority_columns:
            aggregated = (aggregated >= 0.5).astype(float)
        result = result.merge(
            aggregated.rename(feature).reset_index().rename(columns={"index": "LSOA11CD"}),
            on="LSOA11CD",
            how="left",
        )

    return result[output_columns]


def aggregate_spatial_neighbors_to_2011(
    neighbors_2021: Dict[str, List[str]], lookup_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Collapse child-polygon adjacency into adjacency between 2011 parent areas."""
    pairs = lookup_df[["LSOA11CD", "LSOA21CD"]].drop_duplicates()
    parents_per_child = pairs.groupby("LSOA21CD")["LSOA11CD"].nunique()
    unambiguous_children = set(parents_per_child[parents_per_child.eq(1)].index)
    child_to_parent = (
        pairs[pairs["LSOA21CD"].isin(unambiguous_children)]
        .drop_duplicates("LSOA21CD")
        .set_index("LSOA21CD")["LSOA11CD"]
        .to_dict()
    )

    parent_neighbors: Dict[str, set[str]] = {}
    for child, child_neighbors in neighbors_2021.items():
        parent = child_to_parent.get(child)
        if parent is None:
            continue
        parent_neighbors.setdefault(parent, set())
        for neighbor_child in child_neighbors:
            neighbor_parent = child_to_parent.get(neighbor_child)
            if neighbor_parent is None or neighbor_parent == parent:
                continue
            parent_neighbors[parent].add(neighbor_parent)
            parent_neighbors.setdefault(neighbor_parent, set()).add(parent)

    return {parent: sorted(neighbor_codes) for parent, neighbor_codes in parent_neighbors.items()}


def first_child_spatial_neighbors_to_2011(
    neighbors_2021: Dict[str, List[str]], lookup_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Reproduce the former graph that used only each 2011 area's first child."""
    selected = lookup_df.sort_values("ObjectId").drop_duplicates("LSOA11CD")
    selected_children = selected.set_index("LSOA21CD")["LSOA11CD"].to_dict()
    parent_neighbors: Dict[str, set[str]] = {}
    for child, parent in selected_children.items():
        parent_neighbors.setdefault(parent, set())
        for neighbor_child in neighbors_2021.get(child, []):
            neighbor_parent = selected_children.get(neighbor_child)
            if neighbor_parent is not None and neighbor_parent != parent:
                parent_neighbors[parent].add(neighbor_parent)
                parent_neighbors.setdefault(neighbor_parent, set()).add(parent)
    return {parent: sorted(neighbor_codes) for parent, neighbor_codes in parent_neighbors.items()}


def build_master_frame(data: Dict[str, pd.DataFrame], boundary_mode: str = "aggregate_2011") -> pd.DataFrame:
    """Build a 2011-LSOA frame, aggregating 2021-geography inputs to parent areas."""
    samhi_df = data["samhi"]
    lookup_df = data["lookup"]
    deri_df = data["deri"]
    ru_df = data["rural_urban"]
    ts003_df = data["ts003"]
    population_weights = data["population_weights"]
    car_df = data["car"]
    imd25_df = data["imd25"]
    hosp_df = data["hosp"]
    gp_df = data["gp"]
    qof_access_df = data["qof_access"]
    added_features = data["added_features"]

    # Model units and outcomes are LSOA11. Do not choose one arbitrary LSOA21
    # child as the geography for the parent. Use the lookup only for stable LAD
    # labels here; all LSOA21-only feature tables are aggregated below.
    parent_lookup = lookup_df.sort_values("ObjectId").drop_duplicates(subset=["LSOA11CD"])
    first_child_lookup = parent_lookup[["LSOA11CD", "LSOA21CD"]]

    master = samhi_df.merge(
        parent_lookup[["LSOA11CD", "LAD22CD", "LAD22NM"]],
        left_on="lsoa11",
        right_on="LSOA11CD",
        how="left"
    )
    master.drop(columns=["LSOA11CD"], inplace=True)
    master["is_lincolnshire"] = master["LAD22NM"].isin(LINCOLNSHIRE_LADS).astype(int)

    def merge_weighted_2021_features(
        frame: pd.DataFrame,
        feature_columns: List[str],
        weight_col: str,
        binary_majority_columns: Tuple[str, ...] = (),
    ) -> None:
        nonlocal master
        if frame.empty:
            for column in feature_columns:
                master[column] = np.nan
            return
        if boundary_mode == "first_child":
            selected = first_child_lookup.merge(
                frame[["LSOA21CD", *feature_columns]].drop_duplicates("LSOA21CD"),
                on="LSOA21CD",
                how="left",
            )
            master = master.merge(
                selected.rename(columns={"LSOA11CD": "lsoa11"}).drop(columns=["LSOA21CD"]),
                on="lsoa11",
                how="left",
            )
            return
        aggregated = aggregate_2021_features_to_2011(
            frame,
            lookup_df,
            feature_columns,
            weight_col,
            binary_majority_columns,
        )
        master = master.merge(
            aggregated.rename(columns={"LSOA11CD": "lsoa11"}),
            on="lsoa11",
            how="left",
        )

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

    # 2021 RUC combines settlement type with nearer/further relative access.
    # For a split parent, retain the population share in nearer children and
    # assign rural/urban by the population-weighted majority.
    ru_sub = ru_df[["LSOA21CD", "is_rural", "ruc_nearer_population_share"]].drop_duplicates(subset=["LSOA21CD"])
    ru_sub = ru_sub.merge(population_weights, on="LSOA21CD", how="left")
    merge_weighted_2021_features(
        ru_sub,
        ["is_rural", "ruc_nearer_population_share"],
        "__population_weight",
        binary_majority_columns=("is_rural",),
    )
    master["is_rural"] = master["is_rural"].fillna(0).astype(int)

    # IMD is an area rank/decile, not an additive quantity. For a split parent,
    # represent the children with a population-weighted mean of their supplied
    # rank/decile; this is a descriptive area summary, not a recalculated IMD.
    if not imd25_df.empty:
        imd25_sub = imd25_df[["LSOA21CD", "imd_2025_rank", "imd_2025_decile"]].drop_duplicates(subset=["LSOA21CD"])
        imd25_sub = imd25_sub.merge(population_weights, on="LSOA21CD", how="left")
        merge_weighted_2021_features(
            imd25_sub,
            ["imd_2025_rank", "imd_2025_decile"],
            "__population_weight",
        )
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

    # Census household rates use each child's all-households count as weight.
    if not ts003_df.empty:
        merge_weighted_2021_features(
            ts003_df,
            HOUSEHOLD_FEATURE_COLUMNS,
            "__household_weight",
        )
    else:
        for col in HOUSEHOLD_FEATURE_COLUMNS:
            master[col] = np.nan

    household_weights = ts003_df[["LSOA21CD", "__household_weight"]].copy() if "__household_weight" in ts003_df else pd.DataFrame()
    if household_weights.empty:
        logger.warning("TS003 household counts unavailable; population weights will be used for household-rate inputs")
        household_weights = population_weights.rename(columns={"__population_weight": "__household_weight"})

    # Car availability
    if not car_df.empty:
        try:
            car_df["code"] = car_df["Lower layer Super Output Areas Code"].astype(str).str.strip()
            car_piv = car_df.pivot(index="code", columns="Car or van availability (5 categories) Code", values="Observation").fillna(0)
            total_hh = car_piv[[0, 1, 2, 3]].sum(axis=1)
            no_car = pd.DataFrame({
                "LSOA21CD": car_piv.index,
                "no_car_pct": (car_piv[0] / total_hh * 100).values,
                "__household_weight": total_hh.values,
            })
            merge_weighted_2021_features(no_car, ["no_car_pct"], "__household_weight")
        except Exception:
            master["no_car_pct"] = np.nan
    else:
        master["no_car_pct"] = np.nan

    # QOF rates are patient-weighted within each child LSOA, then rolled up
    # using the registered-patient count as the parent-area denominator.
    qof_cols = ["qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"]
    if not qof_access_df.empty:
        qof_sub = qof_access_df.rename(columns={"LSOA_CODE": "LSOA21CD"})
        merge_weighted_2021_features(qof_sub, qof_cols, "__patient_weight")
    else:
        for col in qof_cols:
            master[col] = np.nan

    # Fuel poverty, gas-grid, tenure and occupancy inputs are household rates.
    # Use TS003 household counts as denominator weights where source counts are
    # not carried with the rate table.
    for frame in added_features.values():
        if frame.empty:
            continue
        feature_columns = [column for column in frame.columns if column != "LSOA21CD"]
        weighted_frame = frame.merge(household_weights, on="LSOA21CD", how="left")
        merge_weighted_2021_features(weighted_frame, feature_columns, "__household_weight")

    # Median impute all numeric features
    all_num_covariates = [
        "imd_2019_score", "imd_2025_rank", "imd_2025_decile",
        "unemployment_rate", "pct_aged_65_plus", "pct_disability_limited",
        "pct_no_qualifications", "pct_social_grade_de", "pension_credit_rate", "avg_download_speed",
        "one_person_household_pct", "lone_parent_pct",
        "gp_pt_time", "gp_car_time", "hosp_pt_time", "hosp_car_time",
        "no_car_pct", "ruc_nearer_population_share",
        "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct",
        *[c for c in ADDED_FEATURE_COLUMNS if c != "gas_grid_disconnection_pct"]
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
    lsoa11_to_idx = {code: idx for idx, code in enumerate(df["lsoa11"].values)}

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
            for i, code in enumerate(df["lsoa11"].values):
                nbr_codes = spatial_neighbors.get(code, [])
                if nbr_codes:
                    nbr_indices = [lsoa11_to_idx[c] for c in nbr_codes if c in lsoa11_to_idx]
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
            "ruc_nearer_population_share": df["ruc_nearer_population_share"],
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
            "qof_dep004_pct": df["qof_dep004_pct"],
            "gas_grid_disconnection_pct": lagged_gas_feature(df, t),
        })
        for feature in ADDED_FEATURE_COLUMNS:
            if feature != "gas_grid_disconnection_pct":
                year_records[feature] = df[feature].to_numpy()
        records.append(year_records)

    panel_df = pd.concat(records, ignore_index=True)
    if "gas_grid_disconnection_pct" in panel_df:
        gas_median = panel_df["gas_grid_disconnection_pct"].median()
        panel_df["gas_grid_disconnection_pct"] = panel_df["gas_grid_disconnection_pct"].fillna(
            gas_median if not pd.isna(gas_median) else 0.0
        )
    return panel_df


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
        "ruc_nearer_population_share": "2_Geography_Accessibility",
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

    for feature in ADDED_FEATURE_COLUMNS:
        if feature in {"fuel_poverty_pct", "gas_grid_disconnection_pct"}:
            domain_mapping[feature] = "3_Deprivation_Economics"
        elif feature in {"one_person_household_pct", "one_person_aged_66_plus_pct", "one_person_other_pct",
                         "pensioner_couple_household_pct", "married_no_children_pct",
                         "married_dependent_children_pct", "cohabiting_household_pct",
                         "cohabiting_dependent_children_pct", "lone_parent_pct",
                         "lone_parent_dependent_children_pct", "other_household_types_pct",
                         "other_households_dependent_children_pct"}:
            domain_mapping[feature] = "4_Demographics_Vulnerability"
        else:
            domain_mapping[feature] = "6_Housing"

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
        "ruc_nearer_population_share": "2_Geography_Accessibility",
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

    for feature in ADDED_FEATURE_COLUMNS:
        if feature in {"fuel_poverty_pct", "gas_grid_disconnection_pct"}:
            domain_mapping[feature] = "3_Deprivation_Economics"
        elif feature in {"one_person_household_pct", "one_person_aged_66_plus_pct", "one_person_other_pct",
                         "pensioner_couple_household_pct", "married_no_children_pct",
                         "married_dependent_children_pct", "cohabiting_household_pct",
                         "cohabiting_dependent_children_pct", "lone_parent_pct",
                         "lone_parent_dependent_children_pct", "other_household_types_pct",
                         "other_households_dependent_children_pct"}:
            domain_mapping[feature] = "4_Demographics_Vulnerability"
        else:
            domain_mapping[feature] = "6_Housing"

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
    experiment_set: int = 1,
    history_mode: str = "with_history",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute one temporal benchmark with TreeSHAP and LinearSHAP."""
    if experiment_set == 1:
        train_years, val_year, test_years = (2014, 2018), 2019, (2020, 2022)
    elif experiment_set == 2:
        train_years, val_year, test_years = (2014, 2016), 2017, (2018, 2019)
    else:
        raise ValueError("experiment_set must be 1 or 2")
    logger.info(f"\n=======================================================")
    logger.info(f" EXECUTING MULTIMODAL BENCHMARK: {scope.upper()} ({history_mode})")
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

    test_preds_df = test_df[["lsoa11", "LAD22NM", "is_lincolnshire", "year", "target", "lag_1", "delta_1", "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct"]].rename(columns={"target": "actual_samhi"}).copy()
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
    metrics_df["History_Mode"] = history_mode

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
    prediction_columns = [column for column in test_preds_df.columns if column != "LAD22NM"]
    expanded_preds = lookup_21[["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22NM"]].merge(
        test_preds_df[prediction_columns],
        left_on="LSOA11CD",
        right_on="lsoa11",
        how="inner"
    )

    # Export
    suffix = "2020_2022" if experiment_set == 1 else "pre_covid_2018_2019"
    file_suffix = suffix if history_mode == "with_history" else f"{suffix}_without_history"
    metrics_df.to_csv(output_dir / f"multimodal_metrics_{file_suffix}_{scope}.csv", index=False)
    expanded_preds.to_csv(output_dir / f"multimodal_predictions_{file_suffix}_{scope}.csv", index=False)
    # Keep the legacy LightGBM filename and write explicit model-specific files.
    df_shap.to_csv(output_dir / f"shap_feature_importance_{file_suffix}_{scope}.csv", index=False)
    for model_name, (df_model_explanation, _) in model_explanations.items():
        stem = SHAP_FILE_STEMS[model_name]
        df_model_explanation.to_csv(
            output_dir / f"shap_feature_importance_{stem}_{file_suffix}_{scope}.csv", index=False
        )
    df_shap_elasticnet.to_csv(output_dir / f"shap_feature_importance_elasticnet_{file_suffix}_{scope}.csv", index=False)

    return metrics_df, expanded_preds, df_shap


def main():
    parser = argparse.ArgumentParser(description="SAMHI Multimodal & Spatial Machine Learning Pipeline")
    parser.add_argument("--scope", choices=["lincolnshire", "national", "both"], default="both")
    parser.add_argument("--experiment-set", type=int, choices=[1, 2], default=1,
                        help="1=COVID-era test (2020-2022), 2=pre-COVID test (2018-2019)")
    parser.add_argument("--output-dir", type=str, default="scripts/machine_learning/samhi/results")
    parser.add_argument("--history", choices=["with", "without", "both"], default="with",
                        help="Use previous-SAMHI predictors, exclude them, or run both experiments")
    parser.add_argument(
        "--boundary-mode",
        choices=["aggregate_2011", "first_child"],
        default="aggregate_2011",
        help="Aggregate 2021 child features to 2011 parents (default), or reproduce the former first-child mapping for comparison",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_datasets(project_root)
    spatial_neighbors_2021 = compute_spatial_neighbors(data["geojson_path"])
    if args.boundary_mode == "first_child":
        spatial_neighbors = first_child_spatial_neighbors_to_2011(spatial_neighbors_2021, data["lookup"])
    else:
        spatial_neighbors = aggregate_spatial_neighbors_to_2011(spatial_neighbors_2021, data["lookup"])
    logger.info("Built a 2011-LSOA spatial graph from %s 2021-LSOA neighbour sets", len(spatial_neighbors_2021))
    master_df = build_master_frame(data, boundary_mode=args.boundary_mode)

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
        "is_rural", "ruc_nearer_population_share",
        "avg_download_speed", "gp_pt_time", "gp_car_time", "hosp_pt_time", "hosp_car_time", "no_car_pct",
        "qof_mh002_pct", "qof_mh021_pct", "qof_mh_pca_pct", "qof_dep_pca_pct", "qof_dep004_pct",
        *ADDED_FEATURE_COLUMNS,
    ]
    # The detailed household family also contains two fields named above;
    # train and explain each distinct input only once.
    feature_cols = list(dict.fromkeys(feature_cols))
    external_only_cols = [c for c in feature_cols if c not in SAMHI_HISTORY_FEATURES]

    scopes = ["lincolnshire", "national"] if args.scope == "both" else [args.scope]
    history_modes = ["with_history", "without_history"] if args.history == "both" else [
        "with_history" if args.history == "with" else "without_history"
    ]
    all_metrics = []

    for sc in scopes:
        for history_mode in history_modes:
            selected_cols = feature_cols if history_mode == "with_history" else external_only_cols
            m_df, _, _ = run_scope_benchmark(
                panel_df, sc, selected_cols, data["lookup"], output_dir,
                args.experiment_set, history_mode=history_mode
            )
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

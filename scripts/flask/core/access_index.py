from __future__ import annotations

import pandas as pd

from .common import minmax_scale, normalize_weight_values


WEIGHT_KEYS = [
    "mh002_weight",
    "mh021_weight",
    "mh_pca_weight",
    "dep_pca_weight",
    "dep004_weight",
    "gp_pt_weight",
    "gp_car_weight",
    "hosp_pt_weight",
    "hosp_car_weight",
    "rural_weight",
    "car_weight",
    "digital_weight",
]

# PCA (exception) rates reduce effective access, so they are inverted after normalization.
INVERTED_COLUMNS = {"MH_PCA_Normalized", "Dep_PCA_Normalized"}


def normalize_weights(weight_values: dict[str, float]) -> dict[str, float]:
    return normalize_weight_values(weight_values, WEIGHT_KEYS)


def apply_access_index(
    lsoa_df: pd.DataFrame,
    mh002_weight: float,
    mh021_weight: float,
    mh_pca_weight: float,
    dep_pca_weight: float,
    dep004_weight: float,
    gp_pt_weight: float,
    gp_car_weight: float,
    hosp_pt_weight: float,
    hosp_car_weight: float,
    rural_weight: float,
    car_weight: float,
    digital_weight: float,
) -> pd.DataFrame:
    normalized_weights = normalize_weights(
        {
            "mh002_weight": mh002_weight,
            "mh021_weight": mh021_weight,
            "mh_pca_weight": mh_pca_weight,
            "dep_pca_weight": dep_pca_weight,
            "dep004_weight": dep004_weight,
            "gp_pt_weight": gp_pt_weight,
            "gp_car_weight": gp_car_weight,
            "hosp_pt_weight": hosp_pt_weight,
            "hosp_car_weight": hosp_car_weight,
            "rural_weight": rural_weight,
            "car_weight": car_weight,
            "digital_weight": digital_weight,
        }
    )

    out = lsoa_df.copy()
    out["MH002_Normalized"] = minmax_scale(out["MH002_Access_Pct"])
    out["MH021_Normalized"] = minmax_scale(out["MH021_Access_Pct"])
    out["MH_PCA_Normalized"] = 1.0 - minmax_scale(out["MH_PCA_Access_Pct"])
    out["Dep_PCA_Normalized"] = 1.0 - minmax_scale(out["Dep_PCA_Access_Pct"])
    out["DEP004_Normalized"] = minmax_scale(out["DEP004_Access_Pct"])
    # Lower travel time is better access, so these are inverted after normalization.
    out["GP_PT_Normalized"] = 1.0 - minmax_scale(out["GP_PT_Time"])
    out["GP_Car_Normalized"] = 1.0 - minmax_scale(out["GP_Car_Time"])
    out["Hosp_PT_Normalized"] = 1.0 - minmax_scale(out["Hosp_PT_Time"])
    out["Hosp_Car_Normalized"] = 1.0 - minmax_scale(out["Hosp_Car_Time"])
    out["Rural_Access_Normalized"] = pd.to_numeric(out["Rural_Access"], errors="coerce")
    out["Car_Access_Normalized"] = pd.to_numeric(out["Car_Access"], errors="coerce")
    out["Digital_Access_Normalized"] = pd.to_numeric(out["Digital_Access"], errors="coerce")

    out["Access_Index"] = (
        out["MH002_Normalized"].fillna(0) * normalized_weights["mh002_weight"]
        + out["MH021_Normalized"].fillna(0) * normalized_weights["mh021_weight"]
        + out["MH_PCA_Normalized"].fillna(0) * normalized_weights["mh_pca_weight"]
        + out["Dep_PCA_Normalized"].fillna(0) * normalized_weights["dep_pca_weight"]
        + out["DEP004_Normalized"].fillna(0) * normalized_weights["dep004_weight"]
        + out["GP_PT_Normalized"].fillna(0) * normalized_weights["gp_pt_weight"]
        + out["GP_Car_Normalized"].fillna(0) * normalized_weights["gp_car_weight"]
        + out["Hosp_PT_Normalized"].fillna(0) * normalized_weights["hosp_pt_weight"]
        + out["Hosp_Car_Normalized"].fillna(0) * normalized_weights["hosp_car_weight"]
        + out["Rural_Access_Normalized"].fillna(0) * normalized_weights["rural_weight"]
        + out["Car_Access_Normalized"].fillna(0) * normalized_weights["car_weight"]
        + out["Digital_Access_Normalized"].fillna(0) * normalized_weights["digital_weight"]
    )
    return out

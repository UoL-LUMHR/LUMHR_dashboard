"""Preparation helpers for Census 2021 housing indicators."""

from __future__ import annotations

import pandas as pd

from .common import find_column, normalize_code, parse_numeric


def prepare_housing_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate broad tenure percentages for each 2021 LSOA.

    The owner-occupied measure includes outright ownership, ownership with a
    mortgage or loan, and shared ownership. Percentages use all applicable
    tenure observations and exclude the Census ``Does not apply`` category.
    """

    code_col = find_column(df, ["Lower layer Super Output Areas Code", "LSOA21CD", "LSOA_CODE"])
    category_col = find_column(
        df,
        ["Tenure of household (9 categories) Code", "Category Code"],
    )
    observation_col = find_column(df, ["Observation", "OBSERVATION", "Count"])

    work = pd.DataFrame(
        {
            "LSOA_CODE": df[code_col].map(normalize_code),
            "Category_Code": pd.to_numeric(df[category_col], errors="coerce"),
            "Observation": parse_numeric(df[observation_col]),
        }
    )
    work = work[work["LSOA_CODE"].ne("") & work["Category_Code"].ge(0)].copy()

    def category_total(category_codes: list[int]) -> pd.Series:
        return (
            work[work["Category_Code"].isin(category_codes)]
            .groupby("LSOA_CODE")["Observation"]
            .sum()
        )

    totals = work.groupby("LSOA_CODE")["Observation"].sum().rename("Tenure_Households")
    owner = category_total([0, 1, 2]).rename("Owner_Occupied_HH_Count")
    social = category_total([3, 4]).rename("Social_Rented_HH_Count")
    private = category_total([5, 6]).rename("Private_Rented_HH_Count")
    rent_free = category_total([7]).rename("Rent_Free_HH_Count")

    out = pd.concat([totals, owner, social, private, rent_free], axis=1).fillna(0.0)
    denominator = out["Tenure_Households"].replace(0.0, float("nan"))
    out["Owner_Occupied_Pct"] = out["Owner_Occupied_HH_Count"] / denominator * 100.0
    out["Social_Rented_Pct"] = out["Social_Rented_HH_Count"] / denominator * 100.0
    out["Private_Rented_Pct"] = out["Private_Rented_HH_Count"] / denominator * 100.0
    out["Rent_Free_Pct"] = out["Rent_Free_HH_Count"] / denominator * 100.0

    return out.reset_index()


def prepare_bedroom_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the percentage of households with fewer bedrooms than required."""

    code_col = find_column(df, ["Lower layer Super Output Areas Code", "LSOA21CD", "LSOA_CODE"])
    category_col = find_column(
        df,
        ["Occupancy rating for bedrooms (6 categories) Code", "Category Code"],
    )
    observation_col = find_column(df, ["Observation", "OBSERVATION", "Count"])

    work = pd.DataFrame(
        {
            "LSOA_CODE": df[code_col].map(normalize_code),
            "Category_Code": pd.to_numeric(df[category_col], errors="coerce"),
            "Observation": parse_numeric(df[observation_col]),
        }
    )
    work = work[work["LSOA_CODE"].ne("") & work["Category_Code"].ge(1)].copy()

    totals = work.groupby("LSOA_CODE")["Observation"].sum().rename("Occupancy_Rated_Households")
    overcrowded = (
        work[work["Category_Code"].isin([4, 5])]
        .groupby("LSOA_CODE")["Observation"]
        .sum()
        .rename("Overcrowded_HH_Count")
    )

    out = pd.concat([totals, overcrowded], axis=1).fillna(0.0)
    denominator = out["Occupancy_Rated_Households"].replace(0.0, float("nan"))
    out["Overcrowded_HH_Pct"] = out["Overcrowded_HH_Count"] / denominator * 100.0
    return out.reset_index()

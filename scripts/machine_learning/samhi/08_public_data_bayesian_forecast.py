"""Bayesian forecasting experiment using only public SAMHI components.

The hospital component is not available in this public-data workflow.  This
script therefore forecasts the published SAMHI index using one-year-lagged
public components and reports the result as a calibrated public-data model,
not as an official reconstruction of SAMHI v5.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SAMHI_Public_Bayesian")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normal_crps(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-6)
    z = (np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def load_panel(root: Path) -> pd.DataFrame:
    path = root / "scripts" / "machine_learning" / "samhi" / "results" / "component_reconstruction" / "three_component_panel.csv"
    panel = pd.read_csv(path, low_memory=False)
    panel["lsoa11"] = panel["lsoa11"].astype(str).str.strip()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype(int)
    panel = panel.sort_values(["lsoa11", "year"]).reset_index(drop=True)
    lag_columns = ["samhi_index", "antidep_rate", "qof_dep_pct", "dla_pip"]
    for column in lag_columns:
        panel[f"{column}_lag1"] = panel.groupby("lsoa11")[column].shift(1)
    panel["year_centered"] = panel["year"] - 2011
    return panel


def dwp_calibration(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate overlap calibration from DWP counts to PLDR claimant counts."""
    rows = []
    overlap = panel[panel["year"].between(2019, 2022)].dropna(subset=["dla_pip", "dwp_dla_pip_count"])
    for year, group in overlap.groupby("year"):
        x = group["dwp_dla_pip_count"].to_numpy(dtype=float)
        y = group["dla_pip"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1) if len(group) > 1 else (np.nan, np.nan)
        rows.append({
            "year": int(year), "n": len(group), "dwp_to_pldr_slope": slope,
            "dwp_to_pldr_intercept": intercept, "pearson_r": group["dwp_dla_pip_count"].corr(group["dla_pip"]),
            "mae_raw_count": (group["dwp_dla_pip_count"] - group["dla_pip"]).abs().mean(),
        })
    if len(overlap) > 1:
        x = overlap["dwp_dla_pip_count"].to_numpy(dtype=float)
        y = overlap["dla_pip"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "year": "pooled", "n": len(overlap), "dwp_to_pldr_slope": slope,
            "dwp_to_pldr_intercept": intercept, "pearson_r": overlap["dwp_dla_pip_count"].corr(overlap["dla_pip"]),
            "mae_raw_count": (overlap["dwp_dla_pip_count"] - overlap["dla_pip"]).abs().mean(),
        })
    return pd.DataFrame(rows)


def fit_and_score(panel: pd.DataFrame, target_year: int, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel[(panel["year"] < target_year) & panel["samhi_index"].notna()].copy()
    test = panel[(panel["year"] == target_year) & panel["samhi_index"].notna()].copy()
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()
    preprocessor = ColumnTransformer([( "numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), features)], remainder="drop")
    model = Pipeline([( "features", preprocessor), ("bayesian_ridge", BayesianRidge(compute_score=True))])
    model.fit(train[features], train["samhi_index"])
    bayes_pred, bayes_sd = model.predict(test[features], return_std=True)
    persistence = test["samhi_index_lag1"].to_numpy(dtype=float)
    prediction_rows = []
    for idx, row in test.reset_index(drop=True).iterrows():
        prediction_rows.append({
            "lsoa11": row["lsoa11"], "year": target_year, "actual_samhi": row["samhi_index"],
            "model": "BayesianRidge_public_lagged", "prediction": bayes_pred[idx],
            "interval_low": bayes_pred[idx] - 1.96 * bayes_sd[idx],
            "interval_high": bayes_pred[idx] + 1.96 * bayes_sd[idx], "predictive_sd": bayes_sd[idx],
            "direction_correct": (bayes_pred[idx] - row["samhi_index_lag1"]) * (row["samhi_index"] - row["samhi_index_lag1"]) >= 0 if pd.notna(row["samhi_index_lag1"]) else np.nan,
        })
        if pd.notna(persistence[idx]):
            prediction_rows.append({
                "lsoa11": row["lsoa11"], "year": target_year, "actual_samhi": row["samhi_index"],
                "model": "Persistence", "prediction": persistence[idx],
                "interval_low": np.nan, "interval_high": np.nan, "predictive_sd": np.nan,
                "direction_correct": np.nan,
            })
    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for model_name, group in predictions.groupby("model"):
        y = group["actual_samhi"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        sd = group["predictive_sd"].to_numpy(dtype=float)
        valid_interval = np.isfinite(sd)
        metric_rows.append({
            "year": target_year, "model": model_name, "n": len(group),
            "rmse": np.sqrt(mean_squared_error(y, pred)), "mae": mean_absolute_error(y, pred),
            "directional_accuracy": pd.to_numeric(group["direction_correct"], errors="coerce").mean(),
            "interval_coverage_95": np.mean((y[valid_interval] >= group.loc[valid_interval, "interval_low"]) & (y[valid_interval] <= group.loc[valid_interval, "interval_high"])) if valid_interval.any() else np.nan,
            "interval_width_95": np.mean(group.loc[valid_interval, "interval_high"] - group.loc[valid_interval, "interval_low"]) if valid_interval.any() else np.nan,
            "crps": normal_crps(y[valid_interval], pred[valid_interval], sd[valid_interval]) if valid_interval.any() else np.nan,
        })
    return predictions, pd.DataFrame(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin Bayesian forecast from public SAMHI components")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_bayesian")
    args = parser.parse_args()
    root = project_root()
    panel = load_panel(root)
    calibration = dwp_calibration(panel)
    pooled = calibration[calibration["year"].astype(str).eq("pooled")]
    panel["dla_pip_calibrated"] = panel["dla_pip"]
    if not pooled.empty:
        slope = float(pooled["dwp_to_pldr_slope"].iloc[0])
        intercept = float(pooled["dwp_to_pldr_intercept"].iloc[0])
        dwp_mask = panel["dla_pip_source"].eq("DWP_August")
        panel.loc[dwp_mask, "dla_pip_calibrated"] = slope * panel.loc[dwp_mask, "dwp_dla_pip_count"] + intercept
    panel["dla_pip_calibrated_lag1"] = panel.groupby("lsoa11")["dla_pip_calibrated"].shift(1)
    features = ["samhi_index_lag1", "antidep_rate_lag1", "qof_dep_pct_lag1", "dla_pip_calibrated_lag1", "year_centered"]
    predictions, metrics = [], []
    for year in range(2017, 2023):
        pred, score = fit_and_score(panel, year, features)
        if not pred.empty:
            predictions.append(pred)
            metrics.append(score)
            LOGGER.info("Scored public Bayesian model for %s", year)
    train = panel[(panel["year"] <= 2022) & panel["samhi_index"].notna()].copy()
    future = panel[panel["year"].between(2023, 2025)].copy()
    preprocessor = ColumnTransformer([( "numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), features)], remainder="drop")
    future_model = Pipeline([( "features", preprocessor), ("bayesian_ridge", BayesianRidge(compute_score=True))])
    future_model.fit(train[features], train["samhi_index"])
    # Forecast recursively: after 2023, use the previous public-data forecast
    # as the SAMHI-history lag rather than silently imputing the unavailable
    # post-2022 published SAMHI target.
    previous_prediction = panel[panel["year"] == 2022].groupby("lsoa11")["samhi_index"].first()
    future_rows = []
    for year in range(2023, 2026):
        future_year = future[future["year"] == year].copy()
        future_year["samhi_index_lag1"] = future_year["lsoa11"].map(previous_prediction)
        future_pred, future_sd = future_model.predict(future_year[features], return_std=True)
        future_year["model"] = "BayesianRidge_public_lagged_recursive"
        future_year["prediction"] = future_pred
        future_year["interval_low"] = future_pred - 1.96 * future_sd
        future_year["interval_high"] = future_pred + 1.96 * future_sd
        future_year["predictive_sd"] = future_sd
        future_rows.append(future_year[["lsoa11", "year", "qof_dep_source", "dla_pip_source", "model", "prediction", "interval_low", "interval_high", "predictive_sd"]])
        previous_prediction = future_year.groupby("lsoa11")["prediction"].first()
    future_output = pd.concat(future_rows, ignore_index=True)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "rolling_origin_predictions.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(output_dir / "rolling_origin_metrics.csv", index=False)
    future_output.to_csv(output_dir / "future_predictions_2023_2025.csv", index=False)
    calibration.to_csv(output_dir / "dwp_pldr_calibration.csv", index=False)
    LOGGER.info("Wrote public Bayesian outputs to %s", output_dir)


if __name__ == "__main__":
    main()

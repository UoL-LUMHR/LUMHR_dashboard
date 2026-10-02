"""Lincolnshire spatial Bayesian pilot for the public SAMHI component panel."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SAMHI_Public_Spatial_Bayesian")


def root_path() -> Path:
    return Path(__file__).resolve().parents[3]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crps_normal(actual: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    from scipy.stats import norm

    sd = np.maximum(np.asarray(sd, dtype=float), 1e-6)
    z = (np.asarray(actual) - np.asarray(mean)) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def load_panel(root: Path):
    bayesian = import_module(root / "scripts/machine_learning/samhi/08_public_data_bayesian_forecast.py", "public_bayesian_spatial")
    comparison = import_module(root / "scripts/machine_learning/samhi/09_public_data_model_comparison.py", "public_model_comparison_spatial")
    panel = comparison.prepare_panel(root, bayesian)
    lookup_path = root / "datasets/lincolnshire_lsoa/lsoa_2011_to_2021_lookup/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    lookup = pd.read_csv(lookup_path, usecols=["LSOA11CD", "LSOA21CD", "LAD22NM"])
    lookup = lookup.rename(columns={"LSOA11CD": "lsoa11", "LSOA21CD": "lsoa21", "LAD22NM": "lad"})
    for column in ["lsoa11", "lsoa21"]:
        lookup[column] = lookup[column].astype(str).str.split(":").str[0].str.strip()
    panel = panel.merge(lookup.drop_duplicates("lsoa11"), on="lsoa11", how="inner")
    lads = {"Boston", "East Lindsey", "Lincoln", "North Kesteven", "South Holland", "South Kesteven", "West Lindsey"}
    panel = panel[panel["lad"].isin(lads)].copy()
    panel["target"] = panel["samhi_index"]
    panel["lag_1"] = panel["samhi_index_lag1"]
    panel["year_centered"] = panel["year"] - 2011
    return panel, bayesian, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-data spatial Bayesian pilot for Lincolnshire")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_spatial_bayesian")
    args = parser.parse_args()
    root = root_path()
    panel, bayesian, comparison = load_panel(root)
    features = ["samhi_index_lag1", "antidep_rate_lag1", "qof_dep_pct_lag1", "dla_pip_calibrated_lag1"]
    historical = import_module(root / "scripts/machine_learning/samhi/06_historical_bayesian_experiment.py", "historical_spatial_helpers")
    adjacency = historical.load_spatial_adjacency(root, panel["lsoa21"].dropna().unique())
    if not adjacency:
        raise SystemExit("No Lincolnshire adjacency graph could be built")
    metrics, predictions, baseline_metrics = [], [], []
    for target_year in range(2017, 2023):
        train = panel[(panel["year"] < target_year) & panel["target"].notna() & panel["lag_1"].notna()].copy()
        test = panel[(panel["year"] == target_year) & panel["target"].notna() & panel["lag_1"].notna()].copy()
        if train.empty or test.empty:
            continue
        pred, sd = historical.fit_spatial_bayesian(train, test, features, adjacency)
        actual = test["target"].to_numpy(float)
        lag = test["lag_1"].to_numpy(float)
        metrics.append({
            "year": target_year, "model": "PublicSpatialCAR", "n": len(test),
            "rmse": np.sqrt(mean_squared_error(actual, pred)), "mae": mean_absolute_error(actual, pred),
            "directional_accuracy": np.mean(np.sign(actual - lag) == np.sign(pred - lag)),
            "interval_coverage_95": np.mean((actual >= pred - 1.96 * sd) & (actual <= pred + 1.96 * sd)),
            "interval_width_95": np.mean(3.92 * sd), "crps": crps_normal(actual, pred, sd),
        })
        baseline = comparison.models()["BayesianRidge"]
        baseline.fit(train[features], train["target"])
        baseline_pred, baseline_sd = baseline.predict(test[features], return_std=True)
        baseline_metrics.append({
            "year": target_year, "model": "BayesianRidge_Lincolnshire", "n": len(test),
            "rmse": np.sqrt(mean_squared_error(actual, baseline_pred)), "mae": mean_absolute_error(actual, baseline_pred),
            "directional_accuracy": np.mean(np.sign(actual - lag) == np.sign(baseline_pred - lag)),
            "interval_coverage_95": np.mean((actual >= baseline_pred - 1.96 * baseline_sd) & (actual <= baseline_pred + 1.96 * baseline_sd)),
            "interval_width_95": np.mean(3.92 * baseline_sd), "crps": crps_normal(actual, baseline_pred, baseline_sd),
        })
        predictions.append(pd.DataFrame({
            "lsoa11": test["lsoa11"].to_numpy(), "lsoa21": test["lsoa21"].to_numpy(),
            "year": target_year, "actual_samhi": actual, "prediction": pred,
            "predictive_sd": sd, "interval_low": pred - 1.96 * sd, "interval_high": pred + 1.96 * sd,
        }))
        LOGGER.info("Scored spatial Bayesian model for %s (%s LSOAs)", target_year, len(test))
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / "rolling_origin_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "rolling_origin_predictions.csv", index=False)
    pd.DataFrame([{"lsoa21_count": len(adjacency), "edge_count": sum(map(len, adjacency.values())) // 2}]).to_csv(output / "adjacency_summary.csv", index=False)
    pd.concat([pd.DataFrame(metrics), pd.DataFrame(baseline_metrics)], ignore_index=True).to_csv(output / "spatial_vs_bayesian_metrics.csv", index=False)
    LOGGER.info("Wrote spatial Bayesian outputs to %s", output)


if __name__ == "__main__":
    main()

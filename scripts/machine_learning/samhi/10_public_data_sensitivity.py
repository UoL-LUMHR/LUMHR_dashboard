"""Sensitivity tests for the public-data SAMHI forecast."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def root_path() -> Path:
    return Path(__file__).resolve().parents[3]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root: Path):
    bayesian = import_module(root / "scripts/machine_learning/samhi/08_public_data_bayesian_forecast.py", "public_bayesian_sensitivity")
    comparison = import_module(root / "scripts/machine_learning/samhi/09_public_data_model_comparison.py", "public_comparison_sensitivity")
    return comparison.prepare_panel(root, bayesian), bayesian


def scenario_panel(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    out = panel.copy()
    if name == "raw_dwp":
        out["dla_pip_calibrated"] = out["dla_pip"]
    elif name == "qof_carry_forward":
        source = out[out["year"].eq(2023)][["lsoa11", "qof_dep_pct"]].rename(columns={"qof_dep_pct": "qof_2023"})
        out = out.merge(source, on="lsoa11", how="left")
        out.loc[out["year"].eq(2024) & out["qof_dep_pct"].isna(), "qof_dep_pct"] = out.loc[out["year"].eq(2024) & out["qof_dep_pct"].isna(), "qof_2023"]
        out = out.drop(columns=["qof_2023"])
    out["dla_pip_calibrated_lag1"] = out.groupby("lsoa11")["dla_pip_calibrated"].shift(1)
    out["qof_dep_pct_lag1"] = out.groupby("lsoa11")["qof_dep_pct"].shift(1)
    return out


def fit_model(name: str):
    prep = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    if name == "BayesianRidge":
        return Pipeline([("prep", prep), ("model", BayesianRidge(compute_score=True))])
    return Pipeline([("prep", prep), ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000, random_state=42))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity tests for public-data SAMHI models")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_sensitivity")
    args = parser.parse_args()
    root = root_path()
    panel, _ = prepare(root)
    scenarios = ["base_calibrated", "raw_dwp", "qof_carry_forward", "no_samhi_history"]
    models = ["BayesianRidge", "ElasticNet"]
    metrics, future = [], []
    for scenario in scenarios:
        current = scenario_panel(panel, scenario)
        features = ["antidep_rate_lag1", "qof_dep_pct_lag1", "dla_pip_calibrated_lag1", "year_centered"]
        if scenario != "no_samhi_history":
            features = ["samhi_index_lag1"] + features
        for target_year in range(2017, 2023):
            train = current[(current["year"] < target_year) & current["samhi_index"].notna()]
            test = current[(current["year"] == target_year) & current["samhi_index"].notna()].copy()
            for model_name in models:
                model = fit_model(model_name).fit(train[features], train["samhi_index"])
                if model_name == "BayesianRidge":
                    pred, sd = model.predict(test[features], return_std=True)
                else:
                    pred, sd = model.predict(test[features]), np.full(len(test), np.nan)
                metrics.append({
                    "scenario": scenario, "model": model_name, "year": target_year, "n": len(test),
                    "rmse": np.sqrt(mean_squared_error(test["samhi_index"], pred)),
                    "mae": mean_absolute_error(test["samhi_index"], pred),
                    "interval_coverage_95": np.mean((test["samhi_index"].to_numpy() >= pred - 1.96 * sd) & (test["samhi_index"].to_numpy() <= pred + 1.96 * sd)) if np.isfinite(sd).any() else np.nan,
                })
        train = current[(current["year"] <= 2022) & current["samhi_index"].notna()]
        initial_previous = current[current["year"].eq(2022)].groupby("lsoa11")["samhi_index"].first()
        fitted = {model_name: fit_model(model_name).fit(train[features], train["samhi_index"]) for model_name in models}
        previous_by_model = {model_name: initial_previous.copy() for model_name in models}
        for year in range(2023, 2026):
            block = current[current["year"].eq(year)].copy()
            for model_name, model in fitted.items():
                block["samhi_index_lag1"] = block["lsoa11"].map(previous_by_model[model_name])
                if model_name == "BayesianRidge":
                    pred, sd = model.predict(block[features], return_std=True)
                else:
                    pred, sd = model.predict(block[features]), np.full(len(block), np.nan)
                future.append({"scenario": scenario, "model": model_name, "year": year, "mean_prediction": np.mean(pred), "median_prediction": np.median(pred), "mean_sd": np.nanmean(sd) if np.isfinite(sd).any() else np.nan})
                previous_by_model[model_name] = pd.Series(pred, index=block["lsoa11"]).groupby(level=0).first()
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / "rolling_origin_metrics.csv", index=False)
    pd.DataFrame(future).to_csv(output / "future_scenario_summary.csv", index=False)


if __name__ == "__main__":
    main()

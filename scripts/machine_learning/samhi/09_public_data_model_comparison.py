"""Compare machine-learning and Bayesian SAMHI forecasts on the public panel.

All models use the same one-year-lagged public features and rolling-origin
splits.  The hospital component is not used or imputed.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SAMHI_Public_Model_Comparison")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_helpers(root: Path):
    path = root / "scripts" / "machine_learning" / "samhi" / "08_public_data_bayesian_forecast.py"
    spec = importlib.util.spec_from_file_location("public_bayesian", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_panel(root: Path, helpers) -> pd.DataFrame:
    panel = helpers.load_panel(root)
    calibration = helpers.dwp_calibration(panel)
    pooled = calibration[calibration["year"].astype(str).eq("pooled")]
    panel["dla_pip_calibrated"] = panel["dla_pip"]
    if not pooled.empty:
        slope = float(pooled["dwp_to_pldr_slope"].iloc[0])
        intercept = float(pooled["dwp_to_pldr_intercept"].iloc[0])
        mask = panel["dla_pip_source"].eq("DWP_August")
        panel.loc[mask, "dla_pip_calibrated"] = slope * panel.loc[mask, "dwp_dla_pip_count"] + intercept
    panel["dla_pip_calibrated_lag1"] = panel.groupby("lsoa11")["dla_pip_calibrated"].shift(1)
    return panel


def models() -> dict[str, object]:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return {
        "BayesianRidge": Pipeline([( "prep", numeric), ("model", BayesianRidge(compute_score=True))]),
        "ElasticNet": Pipeline([( "prep", numeric), ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000, random_state=42))]),
        "RandomForest": Pipeline([( "prep", SimpleImputer(strategy="median", add_indicator=True)), ("model", RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=4, n_jobs=-1, random_state=42))]),
        "ExtraTrees": Pipeline([( "prep", SimpleImputer(strategy="median", add_indicator=True)), ("model", ExtraTreesRegressor(n_estimators=100, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=42))]),
    }


def score_year(panel: pd.DataFrame, target_year: int, features: list[str]) -> tuple[list[pd.DataFrame], list[dict]]:
    train = panel[(panel["year"] < target_year) & panel["samhi_index"].notna()]
    test = panel[(panel["year"] == target_year) & panel["samhi_index"].notna()].copy()
    if train.empty or test.empty:
        return [], []
    outputs, metrics = [], []
    for name, model in models().items():
        model.fit(train[features], train["samhi_index"])
        if name == "BayesianRidge":
            prediction, predictive_sd = model.predict(test[features], return_std=True)
        else:
            prediction = model.predict(test[features])
            predictive_sd = np.full(len(test), np.nan)
        direction = (prediction - test["samhi_index_lag1"].to_numpy()) * (test["samhi_index"].to_numpy() - test["samhi_index_lag1"].to_numpy()) >= 0
        out = pd.DataFrame({
            "lsoa11": test["lsoa11"].to_numpy(), "year": target_year, "model": name,
            "actual_samhi": test["samhi_index"].to_numpy(), "prediction": prediction,
            "predictive_sd": predictive_sd, "interval_low": prediction - 1.96 * predictive_sd,
            "interval_high": prediction + 1.96 * predictive_sd, "direction_correct": direction,
        })
        outputs.append(out)
        interval = np.isfinite(predictive_sd)
        metrics.append({
            "year": target_year, "model": name, "n": len(out),
            "rmse": np.sqrt(mean_squared_error(out["actual_samhi"], out["prediction"])),
            "mae": mean_absolute_error(out["actual_samhi"], out["prediction"]),
            "directional_accuracy": pd.Series(direction).mean(),
            "interval_coverage_95": np.mean((out.loc[interval, "actual_samhi"] >= out.loc[interval, "interval_low"]) & (out.loc[interval, "actual_samhi"] <= out.loc[interval, "interval_high"])) if interval.any() else np.nan,
            "interval_width_95": np.mean(out.loc[interval, "interval_high"] - out.loc[interval, "interval_low"]) if interval.any() else np.nan,
        })
    return outputs, metrics


def future_predictions(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    train = panel[(panel["year"] <= 2022) & panel["samhi_index"].notna()]
    future = panel[panel["year"].between(2023, 2025)]
    initial_previous = panel[panel["year"] == 2022].groupby("lsoa11")["samhi_index"].first()
    rows = []
    fitted = {}
    for name, model in models().items():
        model.fit(train[features], train["samhi_index"])
        fitted[name] = model
    previous_by_model = {name: initial_previous.copy() for name in fitted}
    for year in range(2023, 2026):
        for name, model in fitted.items():
            block = future[future["year"] == year].copy()
            block["samhi_index_lag1"] = block["lsoa11"].map(previous_by_model[name])
            if name == "BayesianRidge":
                prediction, sd = model.predict(block[features], return_std=True)
            else:
                prediction = model.predict(block[features])
                sd = np.full(len(block), np.nan)
            rows.append(pd.DataFrame({
                "lsoa11": block["lsoa11"].to_numpy(), "year": year, "model": name,
                "prediction": prediction, "predictive_sd": sd,
                "interval_low": prediction - 1.96 * sd, "interval_high": prediction + 1.96 * sd,
            }))
            previous_by_model[name] = pd.Series(prediction, index=block["lsoa11"]).groupby(level=0).first()
        LOGGER.info("Generated public-data predictions for %s", year)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare public-data machine-learning and Bayesian SAMHI forecasts")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_model_comparison")
    args = parser.parse_args()
    root = project_root()
    helpers = load_helpers(root)
    panel = prepare_panel(root, helpers)
    features = ["samhi_index_lag1", "antidep_rate_lag1", "qof_dep_pct_lag1", "dla_pip_calibrated_lag1", "year_centered"]
    predictions, metrics = [], []
    for year in range(2017, 2023):
        pred, score = score_year(panel, year, features)
        predictions.extend(pred)
        metrics.extend(score)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "rolling_origin_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "rolling_origin_metrics.csv", index=False)
    future_predictions(panel, features).to_csv(output_dir / "future_predictions_2023_2025.csv", index=False)
    helpers.dwp_calibration(panel).to_csv(output_dir / "dwp_pldr_calibration.csv", index=False)
    LOGGER.info("Wrote model comparison outputs to %s", output_dir)


if __name__ == "__main__":
    main()

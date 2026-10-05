"""Compare the earlier ML model families on published SAMHI and Public SAMHI-3.

This is a controlled model tournament: every estimator sees the same lagged
features within a target specification.  The multimodal feature blocks used
by the older experiments are intentionally not added here, so differences are
attributable to the target and estimator rather than a changing feature set.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import catboost as cb
import lightgbm as lgb
import xgboost as xgb


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_panel(root: Path) -> pd.DataFrame:
    bayesian = import_module(root / "scripts/machine_learning/samhi/08_public_data_bayesian_forecast.py", "public_tournament_bayesian")
    comparison = import_module(root / "scripts/machine_learning/samhi/09_public_data_model_comparison.py", "public_tournament_comparison")
    panel = comparison.prepare_panel(root, bayesian).sort_values(["lsoa11", "year"]).reset_index(drop=True)
    samhi3 = import_module(root / "scripts/machine_learning/samhi/14_public_samhi3_index.py", "public_tournament_samhi3")
    panel, _ = samhi3.build_index(panel)
    return panel


def make_models() -> dict[str, object]:
    linear_prep = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    tree_prep = SimpleImputer(strategy="median", add_indicator=True)
    return {
        "BayesianRidge": Pipeline([( "prep", linear_prep), ("model", BayesianRidge(compute_score=True))]),
        "ElasticNet": Pipeline([( "prep", linear_prep), ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000, random_state=42))]),
        "Ridge": Pipeline([( "prep", linear_prep), ("model", Ridge(alpha=25.0))]),
        "RandomForest": Pipeline([( "prep", tree_prep), ("model", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42))]),
        "ExtraTrees": Pipeline([( "prep", tree_prep), ("model", ExtraTreesRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=42))]),
        "LightGBM": Pipeline([( "prep", tree_prep), ("model", lgb.LGBMRegressor(n_estimators=180, learning_rate=0.03, max_depth=5, num_leaves=20, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1))]),
        "XGBoost": Pipeline([( "prep", tree_prep), ("model", xgb.XGBRegressor(n_estimators=180, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist"))]),
        "CatBoost": Pipeline([( "prep", tree_prep), ("model", cb.CatBoostRegressor(iterations=180, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=42, verbose=0, thread_count=-1))]),
    }


def target_specs() -> dict[str, tuple[str, list[str]]]:
    zlags = [f"public_samhi3_z_{label}_lag1" for label in ["antidepressant", "qof", "welfare"]]
    return {
        "PublishedSAMHI": (
            "samhi_index",
            ["samhi_index_lag1", "antidep_rate_lag1", "qof_dep_pct_lag1", "dla_pip_calibrated_lag1", "year_centered"],
        ),
        "PublicSAMHI3_HistoryPlusComponents": (
            "public_samhi3_index",
            ["public_samhi3_index_lag1", *zlags, "year_centered"],
        ),
        "PublicSAMHI3_ComponentsOnly": (
            "public_samhi3_index",
            [*zlags, "year_centered"],
        ),
    }


def metric_row(target_name: str, year: int, model_name: str, actual: np.ndarray, prediction: np.ndarray, lag: np.ndarray, sd: np.ndarray | None) -> dict:
    row = {
        "target": target_name, "year": year, "model": model_name, "n": len(actual),
        "rmse": np.sqrt(mean_squared_error(actual, prediction)),
        "mae": mean_absolute_error(actual, prediction),
        "directional_accuracy": np.nan if model_name == "Persistence" else np.mean((prediction - lag) * (actual - lag) >= 0),
        "interval_coverage_95": np.nan, "interval_width_95": np.nan,
    }
    if sd is not None:
        row["interval_coverage_95"] = np.mean((actual >= prediction - 1.96 * sd) & (actual <= prediction + 1.96 * sd))
        row["interval_width_95"] = np.mean(3.92 * sd)
    return row


def score_target(panel: pd.DataFrame, target_name: str, target: str, features: list[str], years: range) -> list[dict]:
    rows = []
    for year in years:
        train = panel[(panel["year"] < year) & panel[target].notna()]
        test = panel[(panel["year"] == year) & panel[target].notna()].copy()
        if train.empty or test.empty:
            continue
        actual = test[target].to_numpy(float)
        lag_column = "samhi_index_lag1" if target == "samhi_index" else "public_samhi3_index_lag1"
        lag = test[lag_column].to_numpy(float)
        valid = np.isfinite(lag)
        if valid.any():
            rows.append(metric_row(target_name, year, "Persistence", actual[valid], lag[valid], lag[valid], None))
        for name, fitted in make_models().items():
            fitted.fit(train[features], train[target])
            if name == "BayesianRidge":
                prediction, sd = fitted.predict(test[features], return_std=True)
            else:
                prediction, sd = fitted.predict(test[features]), None
            rows.append(metric_row(target_name, year, name, actual, prediction, lag, sd))
    return rows


def future_summary(panel: pd.DataFrame, target_name: str, target: str, features: list[str]) -> list[dict]:
    train = panel[(panel["year"] <= 2022) & panel[target].notna()]
    future = panel[panel["year"].between(2023, 2025)].copy()
    lag_column = "samhi_index_lag1" if target == "samhi_index" else "public_samhi3_index_lag1"
    initial = panel[panel["year"].eq(2022)].groupby("lsoa11")[target].first()
    rows = []
    for name, fitted in make_models().items():
        fitted.fit(train[features], train[target])
        previous = initial.copy()
        for year in range(2023, 2026):
            block = future[future["year"].eq(year)].copy()
            block[lag_column] = block["lsoa11"].map(previous)
            if name == "BayesianRidge":
                prediction, sd = fitted.predict(block[features], return_std=True)
            else:
                prediction, sd = fitted.predict(block[features]), np.full(len(block), np.nan)
            rows.append({
                "target": target_name, "year": year, "model": name,
                "mean_prediction": float(np.nanmean(prediction)), "median_prediction": float(np.nanmedian(prediction)),
                "mean_sd": float(np.nanmean(sd)) if np.isfinite(sd).any() else np.nan,
            })
            previous = pd.Series(prediction, index=block["lsoa11"]).groupby(level=0).first()
    for year in range(2023, 2026):
        rows.append({
            "target": target_name, "year": year, "model": "Persistence",
            "mean_prediction": float(initial.mean()), "median_prediction": float(initial.median()), "mean_sd": np.nan,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tournament of ML models for published SAMHI and Public SAMHI-3")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_target_tournament")
    args = parser.parse_args()
    root = project_root()
    panel = load_panel(root)
    metrics, future = [], []
    for target_name, (target, features) in target_specs().items():
        metrics.extend(score_target(panel, target_name, target, features, range(2017, 2023)))
        future.extend(future_summary(panel, target_name, target, features))
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output / "rolling_origin_metrics.csv", index=False)
    metric_frame.groupby(["target", "model"], as_index=False).agg(
        years=("year", "count"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
        mean_interval_coverage_95=("interval_coverage_95", "mean"),
        mean_interval_width_95=("interval_width_95", "mean"),
    ).to_csv(output / "model_summary.csv", index=False)
    pd.DataFrame(future).to_csv(output / "future_summary.csv", index=False)


if __name__ == "__main__":
    main()

"""Indicator-removal and missingness stress tests for public SAMHI forecasts.

This experiment keeps the published SAMHI target unchanged and removes or
masks only the public predictor components.  The hospital component is not
available in the public data and is never imputed.  National rolling-origin
metrics are reported for Bayesian Ridge, ElasticNet and persistence; the same
scenarios are also evaluated on the Lincolnshire spatial CAR-style pilot.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SAMHI_Public_Missingness")

INDICATOR_COLUMNS = {
    "qof": "qof_dep_pct",
    "dwp": "dla_pip_calibrated",
    "antidepressant": "antidep_rate",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_panel(root: Path) -> tuple[pd.DataFrame, object, object]:
    bayesian = import_module(
        root / "scripts/machine_learning/samhi/08_public_data_bayesian_forecast.py",
        "public_bayesian_missingness",
    )
    comparison = import_module(
        root / "scripts/machine_learning/samhi/09_public_data_model_comparison.py",
        "public_comparison_missingness",
    )
    panel = comparison.prepare_panel(root, bayesian).sort_values(["lsoa11", "year"]).reset_index(drop=True)
    return panel, bayesian, comparison


def recalculate_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Recalculate lags after a component has been removed or masked."""
    out = panel.sort_values(["lsoa11", "year"]).copy()
    for column in ["samhi_index", "antidep_rate", "qof_dep_pct", "dla_pip_calibrated"]:
        if column in out:
            out[f"{column}_lag1"] = out.groupby("lsoa11")[column].shift(1)
    out["year_centered"] = out["year"] - 2011
    return out


def observed_future_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the observed component missingness rates for 2023--2025."""
    rows = []
    future = panel[panel["year"].isin([2023, 2024, 2025])]
    for year, group in future.groupby("year"):
        for indicator, column in INDICATOR_COLUMNS.items():
            rows.append({
                "year": int(year),
                "indicator": indicator,
                "source_column": column,
                "n": len(group),
                "missing_n": int(group[column].isna().sum()),
                "missing_rate": float(group[column].isna().mean()),
            })
    result = pd.DataFrame(rows)
    aggregate = result.groupby(["indicator", "source_column"], as_index=False).agg(
        n=("n", "sum"), missing_n=("missing_n", "sum")
    )
    aggregate["year"] = "2023_2025_pooled"
    aggregate["missing_rate"] = aggregate["missing_n"] / aggregate["n"]
    return pd.concat([result, aggregate[result.columns]], ignore_index=True)


def apply_scenario(panel: pd.DataFrame, name: str, missing_rates: dict[str, float], seed: int = 20261005) -> pd.DataFrame:
    """Apply a removal or future-like missingness scenario."""
    out = panel.copy()
    removals = {
        "remove_qof": {"qof"},
        "remove_dwp": {"dwp"},
        "remove_antidepressant": {"antidepressant"},
        "remove_qof_dwp": {"qof", "dwp"},
        "remove_qof_antidepressant": {"qof", "antidepressant"},
        "remove_dwp_antidepressant": {"dwp", "antidepressant"},
    }
    if name in removals:
        for indicator in removals[name]:
            out[INDICATOR_COLUMNS[indicator]] = np.nan
    elif name == "missingness_2023_2025":
        # The rates are calculated from the observed 2023--2025 panel.  A
        # seeded mask makes the stress test reproducible while preserving the
        # same expected missingness for every component.
        rng = np.random.default_rng(seed)
        for indicator, column in INDICATOR_COLUMNS.items():
            rate = float(missing_rates.get(indicator, 0.0))
            mask = rng.random(len(out)) < rate
            out.loc[mask, column] = np.nan
    elif name not in {"all_indicators", "components_only"}:
        raise ValueError(f"Unknown scenario: {name}")
    return recalculate_lags(out)


def model(name: str):
    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    if name == "BayesianRidge":
        estimator = BayesianRidge(compute_score=True)
    elif name == "ElasticNet":
        estimator = ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000, random_state=42)
    else:
        raise ValueError(name)
    return Pipeline([("prep", prep), ("model", estimator)])


def feature_columns(scenario: str) -> list[str]:
    removed = {
        "remove_qof": {"qof"},
        "remove_dwp": {"dwp"},
        "remove_antidepressant": {"antidepressant"},
        "remove_qof_dwp": {"qof", "dwp"},
        "remove_qof_antidepressant": {"qof", "antidepressant"},
        "remove_dwp_antidepressant": {"dwp", "antidepressant"},
    }.get(scenario, set())
    # The components-only scenario deliberately excludes historical SAMHI.
    # All other scenarios retain the lagged published target as an autoregressive
    # benchmark, matching the main public-data forecast specification.
    features = [] if scenario == "components_only" else ["samhi_index_lag1"]
    for indicator in ["antidepressant", "qof", "dwp"]:
        if indicator not in removed:
            features.append(f"{INDICATOR_COLUMNS[indicator]}_lag1")
    features.append("year_centered")
    return features


def metric_row(scenario: str, scope: str, year: int, model_name: str, actual: np.ndarray,
               prediction: np.ndarray, lag: np.ndarray, sd: np.ndarray | None) -> dict:
    valid_direction = np.isfinite(lag)
    # Persistence predicts no movement, so directional accuracy is undefined
    # rather than artificially scoring every unchanged prediction as correct.
    if model_name == "Persistence":
        direction = np.nan
    else:
        direction = np.mean(((prediction[valid_direction] - lag[valid_direction]) *
                             (actual[valid_direction] - lag[valid_direction])) >= 0) if valid_direction.any() else np.nan
    row = {
        "scenario": scenario, "scope": scope, "year": year, "model": model_name, "n": len(actual),
        "rmse": np.sqrt(mean_squared_error(actual, prediction)),
        "mae": mean_absolute_error(actual, prediction),
        "directional_accuracy": direction,
        "interval_coverage_95": np.nan,
        "interval_width_95": np.nan,
    }
    if sd is not None and np.isfinite(sd).any():
        valid = np.isfinite(sd)
        row["interval_coverage_95"] = np.mean(
            (actual[valid] >= prediction[valid] - 1.96 * sd[valid]) &
            (actual[valid] <= prediction[valid] + 1.96 * sd[valid])
        )
        row["interval_width_95"] = np.mean(3.92 * sd[valid])
    return row


def score_national(panel: pd.DataFrame, scenario: str, features: list[str]) -> list[dict]:
    rows = []
    for target_year in range(2017, 2023):
        train = panel[(panel["year"] < target_year) & panel["samhi_index"].notna()]
        test = panel[(panel["year"] == target_year) & panel["samhi_index"].notna()].copy()
        if train.empty or test.empty:
            continue
        actual = test["samhi_index"].to_numpy(float)
        lag = test["samhi_index_lag1"].to_numpy(float)
        rows.append(metric_row(scenario, "national", target_year, "Persistence", actual, lag.copy(), lag, None))
        for model_name in ["BayesianRidge", "ElasticNet"]:
            fitted = model(model_name).fit(train[features], train["samhi_index"])
            if model_name == "BayesianRidge":
                prediction, sd = fitted.predict(test[features], return_std=True)
            else:
                prediction, sd = fitted.predict(test[features]), None
            rows.append(metric_row(scenario, "national", target_year, model_name, actual, prediction, lag, sd))
    return rows


def score_spatial(panel: pd.DataFrame, scenario: str, features: list[str], spatial_helpers, adjacency: dict[str, list[str]]) -> list[dict]:
    rows = []
    for target_year in range(2017, 2023):
        train = panel[(panel["year"] < target_year) & panel["target"].notna() & panel["lag_1"].notna()].copy()
        test = panel[(panel["year"] == target_year) & panel["target"].notna() & panel["lag_1"].notna()].copy()
        if train.empty or test.empty:
            continue
        actual = test["target"].to_numpy(float)
        lag = test["lag_1"].to_numpy(float)
        rows.append(metric_row(scenario, "lincolnshire", target_year, "Persistence", actual, lag.copy(), lag, None))
        for model_name in ["BayesianRidge", "ElasticNet"]:
            fitted = model(model_name).fit(train[features], train["target"])
            if model_name == "BayesianRidge":
                prediction, sd = fitted.predict(test[features], return_std=True)
            else:
                prediction, sd = fitted.predict(test[features]), None
            rows.append(metric_row(scenario, "lincolnshire", target_year, model_name, actual, prediction, lag, sd))
        spatial_prediction, spatial_sd = spatial_helpers.fit_spatial_bayesian(train, test, features, adjacency)
        rows.append(metric_row(scenario, "lincolnshire", target_year, "PublicSpatialCAR", actual, spatial_prediction, lag, spatial_sd))
    return rows


def future_summary(panel: pd.DataFrame, scenario: str, features: list[str]) -> list[dict]:
    train = panel[(panel["year"] <= 2022) & panel["samhi_index"].notna()]
    future = panel[panel["year"].between(2023, 2025)].copy()
    previous_initial = panel[panel["year"].eq(2022)].groupby("lsoa11")["samhi_index"].first()
    fitted = {name: model(name).fit(train[features], train["samhi_index"]) for name in ["BayesianRidge", "ElasticNet"]}
    previous = {name: previous_initial.copy() for name in fitted}
    rows = []
    for year in range(2023, 2026):
        block = future[future["year"].eq(year)].copy()
        for name, fitted_model in fitted.items():
            block["samhi_index_lag1"] = block["lsoa11"].map(previous[name])
            if name == "BayesianRidge":
                prediction, sd = fitted_model.predict(block[features], return_std=True)
            else:
                prediction, sd = fitted_model.predict(block[features]), np.full(len(block), np.nan)
            rows.append({
                "scenario": scenario, "model": name, "year": year,
                "mean_prediction": float(np.nanmean(prediction)),
                "median_prediction": float(np.nanmedian(prediction)),
                "mean_sd": float(np.nanmean(sd)) if np.isfinite(sd).any() else np.nan,
            })
            previous[name] = pd.Series(prediction, index=block["lsoa11"]).groupby(level=0).first()
        persistence = block["lsoa11"].map(previous_initial).dropna()
        rows.append({
            "scenario": scenario, "model": "Persistence", "year": year,
            "mean_prediction": float(persistence.mean()), "median_prediction": float(persistence.median()), "mean_sd": np.nan,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Public SAMHI indicator-removal and missingness experiments")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_missingness")
    args = parser.parse_args()
    root = project_root()
    panel, _, _ = prepare_panel(root)
    observed = observed_future_missingness(panel)
    pooled = observed[observed["year"].eq("2023_2025_pooled")]
    missing_rates = pooled.set_index("indicator")["missing_rate"].to_dict()
    scenarios = [
        "all_indicators",
        "components_only",
        "remove_qof",
        "remove_dwp",
        "remove_antidepressant",
        "remove_qof_dwp",
        "remove_qof_antidepressant",
        "remove_dwp_antidepressant",
        "missingness_2023_2025",
    ]
    spatial_module = import_module(
        root / "scripts/machine_learning/samhi/11_public_spatial_bayesian.py",
        "public_spatial_missingness",
    )
    spatial_helpers = import_module(
        root / "scripts/machine_learning/samhi/06_historical_bayesian_experiment.py",
        "historical_spatial_missingness",
    )
    spatial_panel, _, _ = spatial_module.load_panel(root)
    spatial_panel["target"] = spatial_panel["samhi_index"]
    spatial_panel["lag_1"] = spatial_panel["samhi_index_lag1"]
    adjacency = spatial_helpers.load_spatial_adjacency(root, spatial_panel["lsoa21"].dropna().unique())
    if not adjacency:
        raise SystemExit("No Lincolnshire adjacency graph could be built")

    metric_rows, future_rows = [], []
    for scenario in scenarios:
        current = apply_scenario(panel, scenario, missing_rates)
        features = feature_columns(scenario)
        metric_rows.extend(score_national(current, scenario, features))
        future_rows.extend(future_summary(current, scenario, features))
        spatial_current = apply_scenario(spatial_panel, scenario, missing_rates)
        spatial_current["target"] = spatial_current["samhi_index"]
        spatial_current["lag_1"] = spatial_current["samhi_index_lag1"]
        metric_rows.extend(score_spatial(spatial_current, scenario, features, spatial_helpers, adjacency))
        LOGGER.info("Completed missingness scenario %s", scenario)

    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "rolling_origin_metrics.csv", index=False)
    metrics.groupby(["scenario", "scope", "model"], as_index=False).agg(
        years=("year", "count"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"),
        mean_directional_accuracy=("directional_accuracy", "mean"),
        mean_interval_coverage_95=("interval_coverage_95", "mean"),
        mean_interval_width_95=("interval_width_95", "mean"),
    ).to_csv(output / "scenario_summary.csv", index=False)
    pd.DataFrame(future_rows).to_csv(output / "future_scenario_summary.csv", index=False)
    observed.to_csv(output / "observed_2023_2025_missingness.csv", index=False)
    pd.DataFrame([{
        "scenario": "missingness_2023_2025", "indicator": indicator,
        "applied_missing_rate": rate, "seed": 20261005,
    } for indicator, rate in missing_rates.items()]).to_csv(output / "applied_missingness.csv", index=False)
    pd.DataFrame([{
        "lsoa21_count": len(adjacency), "edge_count": sum(map(len, adjacency.values())) // 2,
    }]).to_csv(output / "adjacency_summary.csv", index=False)
    LOGGER.info("Wrote missingness experiment outputs to %s", output)


if __name__ == "__main__":
    main()

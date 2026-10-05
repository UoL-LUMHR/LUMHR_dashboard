"""Build and forecast a public-data three-component SAMHI-like index.

The index is deliberately separate from the published SAMHI score.  It uses
only antidepressant prescribing, QOF depression and DLA/PIP, with stable
standardisation fitted on 2011--2022 data.  The forecasts are evaluated against
the public three-component target, and the resulting index is compared with
published SAMHI where both are available.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COMPONENTS = {
    "antidepressant": "antidep_rate",
    "qof": "qof_dep_pct",
    "welfare": "dla_pip_calibrated",
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


def load_panel(root: Path) -> pd.DataFrame:
    bayesian = import_module(root / "scripts/machine_learning/samhi/08_public_data_bayesian_forecast.py", "public_samhi3_bayesian")
    comparison = import_module(root / "scripts/machine_learning/samhi/09_public_data_model_comparison.py", "public_samhi3_comparison")
    return comparison.prepare_panel(root, bayesian).sort_values(["lsoa11", "year"]).reset_index(drop=True)


def build_index(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stable equal-weight z-score index using 2011--2022 references."""
    out = panel.copy()
    reference = []
    historical = out[out["year"].between(2011, 2022)]
    for label, column in COMPONENTS.items():
        values = pd.to_numeric(historical[column], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        zcolumn = f"public_samhi3_z_{label}"
        out[zcolumn] = (pd.to_numeric(out[column], errors="coerce") - mean) / std
        reference.append({"component": label, "source_column": column, "reference_years": "2011-2022", "mean": mean, "std": std})
    zcolumns = [f"public_samhi3_z_{label}" for label in COMPONENTS]
    out["public_samhi3_n_components"] = out[zcolumns].notna().sum(axis=1)
    out["public_samhi3_index"] = out[zcolumns].mean(axis=1, skipna=False)
    out["public_samhi3_index_2plus"] = out[zcolumns].mean(axis=1, skipna=True).where(out["public_samhi3_n_components"] >= 2)
    out = out.sort_values(["lsoa11", "year"]).reset_index(drop=True)
    out["public_samhi3_index_lag1"] = out.groupby("lsoa11")["public_samhi3_index"].shift(1)
    for column in zcolumns:
        out[f"{column}_lag1"] = out.groupby("lsoa11")[column].shift(1)
    return out, pd.DataFrame(reference)


def normal_crps(actual: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-6)
    z = (np.asarray(actual, dtype=float) - np.asarray(mean, dtype=float)) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def make_model(name: str):
    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    estimator = BayesianRidge(compute_score=True) if name == "BayesianRidge" else ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000, random_state=42)
    return Pipeline([("prep", prep), ("model", estimator)])


def score_model(panel: pd.DataFrame, target_year: int, model_name: str, features: list[str]) -> tuple[dict, pd.DataFrame]:
    train = panel[(panel["year"] < target_year) & panel["public_samhi3_index"].notna()]
    test = panel[(panel["year"] == target_year) & panel["public_samhi3_index"].notna() & panel["public_samhi3_index_lag1"].notna()].copy()
    if train.empty or test.empty:
        return {}, pd.DataFrame()
    actual = test["public_samhi3_index"].to_numpy(float)
    lag = test["public_samhi3_index_lag1"].to_numpy(float)
    if model_name == "Persistence":
        prediction, sd = lag, None
    else:
        estimator_name = "BayesianRidge" if model_name.endswith("BayesianRidge") else "ElasticNet"
        fitted = make_model(estimator_name).fit(train[features], train["public_samhi3_index"])
        if estimator_name == "BayesianRidge":
            prediction, sd = fitted.predict(test[features], return_std=True)
        else:
            prediction, sd = fitted.predict(test[features]), None
    row = {
        "year": target_year, "model": model_name, "n": len(test),
        "rmse": np.sqrt(mean_squared_error(actual, prediction)),
        "mae": mean_absolute_error(actual, prediction),
        "directional_accuracy": np.nan if model_name == "Persistence" else np.mean((prediction - lag) * (actual - lag) >= 0),
        "interval_coverage_95": np.nan, "interval_width_95": np.nan, "crps": np.nan,
    }
    if sd is not None:
        row["interval_coverage_95"] = np.mean((actual >= prediction - 1.96 * sd) & (actual <= prediction + 1.96 * sd))
        row["interval_width_95"] = np.mean(3.92 * sd)
        row["crps"] = normal_crps(actual, prediction, sd)
    predictions = pd.DataFrame({
        "lsoa11": test["lsoa11"].to_numpy(), "year": target_year,
        "model": model_name, "actual_public_samhi3": actual,
        "prediction": prediction, "predictive_sd": sd if sd is not None else np.nan,
        "interval_low": prediction - 1.96 * sd if sd is not None else np.nan,
        "interval_high": prediction + 1.96 * sd if sd is not None else np.nan,
    })
    return row, predictions


def future_predictions(panel: pd.DataFrame, features_by_model: dict[str, list[str]]) -> pd.DataFrame:
    train = panel[(panel["year"] <= 2022) & panel["public_samhi3_index"].notna()]
    future = panel[panel["year"].between(2023, 2025)].copy()
    initial = panel[panel["year"].eq(2022)].groupby("lsoa11")["public_samhi3_index"].first()
    fitted = {
        name: make_model("BayesianRidge" if name.endswith("BayesianRidge") else "ElasticNet").fit(train[features], train["public_samhi3_index"])
        for name, features in features_by_model.items()
    }
    previous = {name: initial.copy() for name in fitted}
    rows = []
    for year in range(2023, 2026):
        block = future[future["year"].eq(year)].copy()
        for name, fitted_model in fitted.items():
            block["public_samhi3_index_lag1"] = block["lsoa11"].map(previous[name])
            if name.endswith("BayesianRidge"):
                prediction, sd = fitted_model.predict(block[features_by_model[name]], return_std=True)
            else:
                prediction, sd = fitted_model.predict(block[features_by_model[name]]), np.full(len(block), np.nan)
            rows.append(pd.DataFrame({
                "lsoa11": block["lsoa11"], "year": year, "model": name,
                "prediction": prediction, "predictive_sd": sd,
                "interval_low": prediction - 1.96 * sd, "interval_high": prediction + 1.96 * sd,
            }))
            previous[name] = pd.Series(prediction, index=block["lsoa11"]).groupby(level=0).first()
        persistence = block["lsoa11"].map(initial).to_numpy(float)
        rows.append(pd.DataFrame({
            "lsoa11": block["lsoa11"], "year": year, "model": "Persistence",
            "prediction": persistence, "predictive_sd": np.nan,
            "interval_low": np.nan, "interval_high": np.nan,
        }))
    return pd.concat(rows, ignore_index=True)


def compare_published(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in panel.groupby("year"):
        valid = group[["public_samhi3_index", "samhi_index"]].dropna()
        if len(valid) < 2:
            continue
        rows.append({
            "year": int(year), "n": len(valid),
            "pearson_r": valid["public_samhi3_index"].corr(valid["samhi_index"]),
            "spearman_r": valid["public_samhi3_index"].corr(valid["samhi_index"], method="spearman"),
            "public_samhi3_mean": valid["public_samhi3_index"].mean(),
            "published_samhi_mean": valid["samhi_index"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and forecast the public three-component SAMHI-like index")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_samhi3")
    args = parser.parse_args()
    root = project_root()
    panel, reference = build_index(load_panel(root))
    zlags = [f"public_samhi3_z_{label}_lag1" for label in COMPONENTS]
    components_only = zlags
    history_plus_components = ["public_samhi3_index_lag1", *zlags]
    features_by_model = {
        "ComponentsOnly_BayesianRidge": components_only,
        "ComponentsOnly_ElasticNet": components_only,
        "IndexHistoryPlusComponents_BayesianRidge": history_plus_components,
        "IndexHistoryPlusComponents_ElasticNet": history_plus_components,
    }
    metrics, predictions = [], []
    for year in range(2017, 2023):
        for model_name, features in [("Persistence", []), *features_by_model.items()]:
            row, output = score_model(panel, year, model_name, features)
            if row:
                metrics.append(row)
                predictions.append(output)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_columns = [
        "lsoa11", "year", "antidep_rate", "qof_dep_pct", "dla_pip_calibrated",
        "dla_pip_source", "qof_dep_source", "public_samhi3_n_components",
        "public_samhi3_index", "public_samhi3_index_2plus", "samhi_index",
        *[f"public_samhi3_z_{label}" for label in COMPONENTS],
    ]
    panel[index_columns].to_csv(output_dir / "public_samhi3_index.csv", index=False)
    reference.to_csv(output_dir / "standardisation_reference.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "rolling_origin_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "rolling_origin_predictions.csv", index=False)
    future_predictions(panel, features_by_model).to_csv(output_dir / "future_predictions_2023_2025.csv", index=False)
    compare_published(panel).to_csv(output_dir / "published_samhi_comparison.csv", index=False)


if __name__ == "__main__":
    main()

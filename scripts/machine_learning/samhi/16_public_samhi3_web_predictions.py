"""Create per-LSOA Public SAMHI-3 data for the dashboard map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from importlib.util import module_from_spec, spec_from_file_location


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dashboard-ready Public SAMHI-3 predictions")
    parser.add_argument("--output-dir", default="datasets/public_samhi3")
    args = parser.parse_args()
    root = project_root()
    tournament = load_module(root / "scripts/machine_learning/samhi/15_public_target_model_tournament.py", "public_web_tournament")
    panel = tournament.load_panel(root).sort_values(["lsoa11", "year"]).reset_index(drop=True)
    target = "public_samhi3_index"
    features = [
        "public_samhi3_index_lag1",
        "public_samhi3_z_antidepressant_lag1",
        "public_samhi3_z_qof_lag1",
        "public_samhi3_z_welfare_lag1",
        "year_centered",
    ]
    # Keep the dashboard selector aligned with the full validated tournament.
    # Bayesian Ridge is the only model with native predictive standard errors.
    model_names = list(tournament.make_models().keys())
    model_frames = []

    # Rolling-origin predictions for observed proxy years.
    for year in range(2017, 2023):
        train = panel[(panel["year"] < year) & panel[target].notna()]
        test = panel[(panel["year"] == year) & panel[target].notna() & panel["public_samhi3_index_lag1"].notna()].copy()
        for name in model_names:
            fitted = tournament.make_models()[name].fit(train[features], train[target])
            if name == "BayesianRidge":
                prediction, sd = fitted.predict(test[features], return_std=True)
            else:
                prediction, sd = fitted.predict(test[features]), np.full(len(test), np.nan)
            model_frames.append(pd.DataFrame({
                "lsoa11": test["lsoa11"].to_numpy(), "year": year,
                f"pred_{name.lower()}": prediction,
                f"ci_lower_{name.lower()}": prediction - 1.96 * sd,
                f"ci_upper_{name.lower()}": prediction + 1.96 * sd,
            }))

    # Recursive 2023--2025 forecasts. Components are available as lagged
    # public observations, while the Public SAMHI-3 target lag is recursive.
    train = panel[(panel["year"] <= 2022) & panel[target].notna()]
    future = panel[panel["year"].between(2023, 2025)].copy()
    fitted_models = {name: tournament.make_models()[name].fit(train[features], train[target]) for name in model_names}
    initial = panel[panel["year"].eq(2022)].groupby("lsoa11")[target].first()
    previous = {name: initial.copy() for name in model_names}

    # For 2026--2027 there are no component observations. Hold each LSOA's
    # latest available standardised component value (up to 2025) constant,
    # while continuing each model's index forecast recursively. These are
    # longer-range projections, not observed proxy values or validated errors.
    latest = panel[panel["year"].le(2025)].sort_values(["lsoa11", "year"])
    latest_rows = latest.groupby("lsoa11", as_index=False).tail(1).copy()
    component_lags = {}
    for label in ("antidepressant", "qof", "welfare"):
        zcolumn = f"public_samhi3_z_{label}"
        component_lags[label] = latest.groupby("lsoa11")[zcolumn].last()
    extended_future = {}
    for year in (2026, 2027):
        block = latest_rows.copy()
        block["year"] = year
        block["year_centered"] = year - 2011
        for label in ("antidepressant", "qof", "welfare"):
            zcolumn = f"public_samhi3_z_{label}"
            block[f"{zcolumn}_lag1"] = block["lsoa11"].map(component_lags[label])
        extended_future[year] = block

    for year in range(2023, 2028):
        block = future[future["year"].eq(year)].copy() if year <= 2025 else extended_future[year]
        for name, fitted in fitted_models.items():
            block["public_samhi3_index_lag1"] = block["lsoa11"].map(previous[name])
            if name == "BayesianRidge":
                prediction, sd = fitted.predict(block[features], return_std=True)
            else:
                prediction, sd = fitted.predict(block[features]), np.full(len(block), np.nan)
            model_frames.append(pd.DataFrame({
                "lsoa11": block["lsoa11"].to_numpy(), "year": year,
                f"pred_{name.lower()}": prediction,
                f"ci_lower_{name.lower()}": prediction - 1.96 * sd,
                f"ci_upper_{name.lower()}": prediction + 1.96 * sd,
            }))
            previous[name] = pd.Series(prediction, index=block["lsoa11"]).groupby(level=0).first()

    output = panel[[
        "lsoa11", "year", "public_samhi3_index", "public_samhi3_index_2plus",
        "public_samhi3_n_components", "samhi_index",
    ]].copy()
    output = output.rename(columns={
        "public_samhi3_index": "public_samhi3_proxy",
        "public_samhi3_index_2plus": "public_samhi3_proxy_2plus",
    })
    output["public_samhi3_proxy_lag1"] = output.groupby("lsoa11")["public_samhi3_proxy"].shift(1)
    prediction_frame = pd.concat(model_frames, ignore_index=True).groupby(
        ["lsoa11", "year"], as_index=False
    ).first()
    output = output.merge(prediction_frame, on=["lsoa11", "year"], how="left")
    output["proxy_change"] = output["public_samhi3_proxy"] - output["public_samhi3_proxy_lag1"]
    projection_rows = []
    latest_codes = latest_rows["lsoa11"].drop_duplicates()
    for year in (2026, 2027):
        projection = pd.DataFrame({"lsoa11": latest_codes, "year": year})
        projection_rows.append(projection)
    projections = pd.concat(projection_rows, ignore_index=True)
    projections = projections.merge(prediction_frame, on=["lsoa11", "year"], how="left")
    projections["forecast_assumption"] = "Latest available components held constant from 2025; index forecast recursive"
    output = pd.concat([output, projections], ignore_index=True, sort=False)
    output = output.drop_duplicates(["lsoa11", "year"], keep="last")
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "public_samhi3_web_predictions.csv.gz", index=False, compression="gzip")
    print(f"Wrote {len(output):,} Public SAMHI-3 dashboard rows to {output_dir}")


if __name__ == "__main__":
    main()

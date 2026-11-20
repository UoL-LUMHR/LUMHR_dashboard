"""Evaluate the public SAMHI-3 proxy with origin-specific standardisation.

Each rolling origin fits component means and standard deviations using only
earlier years. Models are trained nationally, then scored nationally and on
the Lincolnshire rural/urban subsets using the same one-year-ahead LSOAs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from _shared import import_module, project_root


def load_helpers(root: Path):
    samhi3 = import_module(
        root / "scripts/machine_learning/samhi/14_public_samhi3_index.py",
        "origin_eval_samhi3_index",
    )
    tournament = import_module(
        root / "scripts/machine_learning/samhi/15_public_target_model_tournament.py",
        "origin_eval_model_tournament",
    )
    return samhi3, tournament


def load_rurality(root: Path) -> tuple[dict[str, str], set[str]]:
    lookup_path = root / "datasets/lsoa/lsoa_2011_to_2021_lookup/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    ruc_path = root / "datasets/rural_urban_classification_2021_lsoa/Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv"
    lincolnshire_path = root / "datasets/samhi_forecast/multimodal_predictions_2020_2022_lincolnshire.csv"
    lookup = pd.read_csv(lookup_path, dtype={"LSOA11CD": str, "LSOA21CD": str}, usecols=["LSOA11CD", "LSOA21CD"])
    ruc = pd.read_csv(ruc_path, dtype={"LSOA21CD": str}, usecols=["LSOA21CD", "Urban_rural_flag"])
    joined = lookup.drop_duplicates().merge(ruc, on="LSOA21CD", how="left", validate="many_to_one")
    classes = joined.groupby("LSOA11CD")["Urban_rural_flag"].agg(
        lambda values: values.iloc[0] if values.nunique() == 1 else "Mixed"
    )
    lincolnshire = set(pd.read_csv(lincolnshire_path, dtype={"lsoa11": str}, usecols=["lsoa11"]).lsoa11)
    return classes.to_dict(), lincolnshire


def make_origin_panel(raw: pd.DataFrame, components: dict[str, str], origin: int):
    """Rebuild the proxy using statistics available before this test year."""
    panel = raw[raw["year"].between(2011, origin)].copy()
    training_period = panel[panel["year"] < origin]
    references = []
    z_columns = []
    for label, source in components.items():
        values = pd.to_numeric(training_period[source], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        z_column = f"public_samhi3_z_{label}"
        panel[z_column] = (pd.to_numeric(panel[source], errors="coerce") - mean) / std
        panel[f"{z_column}_lag1"] = panel.groupby("lsoa11")[z_column].shift(1)
        z_columns.append(z_column)
        references.append({
            "origin_year": origin,
            "component": label,
            "source_column": source,
            "standardisation_years": f"2011-{origin - 1}",
            "n_observations": int(values.notna().sum()),
            "mean": mean,
            "std": std,
        })
    panel["public_samhi3_index"] = panel[z_columns].mean(axis=1, skipna=False)
    panel["public_samhi3_index_lag1"] = panel.groupby("lsoa11")["public_samhi3_index"].shift(1)
    panel["year_centered"] = panel["year"] - 2011
    return panel, z_columns, references


def metric_row(actual, prediction, persistence, model_name: str, *, feature_set: str,
    origin: int, scope: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, prediction)))
    mae = float(mean_absolute_error(actual, prediction))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, persistence)))
    baseline_mae = float(mean_absolute_error(actual, persistence))
    directional = np.nan
    if model_name != "Persistence":
        directional = float(np.mean((prediction - persistence) * (actual - persistence) >= 0))
    return {
        "origin_year": origin,
        "held_out_year": origin,
        "scope": scope,
        "feature_set": feature_set,
        "model": model_name,
        "n_lsoas": int(len(actual)),
        "rmse": rmse,
        "mae": mae,
        "persistence_rmse": baseline_rmse,
        "persistence_mae": baseline_mae,
        "rmse_skill_vs_persistence_pct": 100 * (baseline_rmse - rmse) / baseline_rmse if baseline_rmse else np.nan,
        "mae_skill_vs_persistence_pct": 100 * (baseline_mae - mae) / baseline_mae if baseline_mae else np.nan,
        "directional_accuracy": directional,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_samhi3_origin_eval")
    args = parser.parse_args()
    root = project_root()
    samhi3, tournament = load_helpers(root)
    panel = samhi3.load_panel(root)
    panel = panel[panel["year"].between(2011, 2022)].sort_values(["lsoa11", "year"]).reset_index(drop=True)
    rurality, lincolnshire_codes = load_rurality(root)
    panel["rurality"] = panel["lsoa11"].map(rurality)
    components = samhi3.COMPONENTS

    zlags_by_origin = {}
    results, lincolnshire_predictions, standardisation_rows = [], [], []
    for origin in range(2017, 2023):
        fold_panel, z_columns, reference_rows = make_origin_panel(panel, components, origin)
        standardisation_rows.extend(reference_rows)
        train = fold_panel[(fold_panel["year"] < origin) & fold_panel["public_samhi3_index"].notna()].copy()
        test = fold_panel[(fold_panel["year"] == origin) & fold_panel["public_samhi3_index"].notna()].copy()
        zlags = [f"{column}_lag1" for column in z_columns]
        test_features = ["public_samhi3_index_lag1", *zlags]
        test = test.dropna(subset=test_features).copy()
        actual = test["public_samhi3_index"].to_numpy(dtype=float)
        persistence = test["public_samhi3_index_lag1"].to_numpy(dtype=float)

        feature_sets = {
            "Components only": [*zlags, "year_centered"],
            "Index history + components": ["public_samhi3_index_lag1", *zlags, "year_centered"],
        }
        scopes = {
            "England": np.ones(len(test), dtype=bool),
            "Lincolnshire": test["lsoa11"].isin(lincolnshire_codes).to_numpy(),
            "Lincolnshire rural": (test["lsoa11"].isin(lincolnshire_codes) & test["rurality"].eq("Rural")).to_numpy(),
            "Lincolnshire urban": (test["lsoa11"].isin(lincolnshire_codes) & test["rurality"].eq("Urban")).to_numpy(),
        }

        for feature_set, features in feature_sets.items():
            fitted_models = tournament.make_models()
            for scope, mask in scopes.items():
                if not mask.any():
                    continue
                scope_rows = test.loc[mask]
                scope_actual = actual[mask]
                scope_persistence = persistence[mask]
                results.append(metric_row(scope_actual, scope_persistence, scope_persistence, "Persistence",
                    feature_set=feature_set, origin=origin, scope=scope))

            for model_name, model in fitted_models.items():
                model.fit(train[features], train["public_samhi3_index"])
                prediction = model.predict(test[features])
                for scope, mask in scopes.items():
                    if not mask.any():
                        continue
                    results.append(metric_row(actual[mask], prediction[mask], persistence[mask], model_name,
                        feature_set=feature_set, origin=origin, scope=scope))
                county_mask = test["lsoa11"].isin(lincolnshire_codes).to_numpy()
                if county_mask.any():
                    county = test.loc[county_mask, ["lsoa11", "rurality"]].copy()
                    county["origin_year"] = origin
                    county["feature_set"] = feature_set
                    county["model"] = model_name
                    county["actual_public_samhi3"] = actual[county_mask]
                    county["prediction"] = prediction[county_mask]
                    county["persistence"] = persistence[county_mask]
                    lincolnshire_predictions.append(county)

        print(f"Completed origin {origin}: train={len(train):,}, test={len(test):,}, Lincolnshire={test.lsoa11.isin(lincolnshire_codes).sum():,}")

    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "rolling_origin_metrics.csv", index=False)
    pd.DataFrame(standardisation_rows).to_csv(output / "standardisation_reference.csv", index=False)
    pd.concat(lincolnshire_predictions, ignore_index=True).to_csv(output / "lincolnshire_predictions.csv", index=False)

    summaries = []
    for period, subset in [("2017-2022", metrics), ("2020-2022", metrics[metrics.held_out_year >= 2020])]:
        summary = subset.groupby(["scope", "feature_set", "model"], as_index=False).agg(
            folds=("held_out_year", "nunique"),
            mean_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "mean"),
            median_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "median"),
            folds_better_than_persistence=("rmse_skill_vs_persistence_pct", lambda values: int((values > 0).sum())),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_n_lsoas=("n_lsoas", "mean"),
        )
        summary.insert(0, "period", period)
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(output / "skill_summary.csv", index=False)

    counts = pd.DataFrame([
        {"scope": "Lincolnshire", "n_lsoas": len(lincolnshire_codes)},
        {"scope": "Lincolnshire rural", "n_lsoas": sum(rurality.get(code) == "Rural" for code in lincolnshire_codes)},
        {"scope": "Lincolnshire urban", "n_lsoas": sum(rurality.get(code) == "Urban" for code in lincolnshire_codes)},
        {"scope": "Lincolnshire mixed/unclassified", "n_lsoas": sum(rurality.get(code) not in {"Rural", "Urban"} for code in lincolnshire_codes)},
    ])
    counts.to_csv(output / "lincolnshire_rurality_counts.csv", index=False)
    (output / "README.md").write_text(
        "# Origin-specific Public SAMHI-3 evaluation\n\n"
        "The target is the equal-weight mean of three component z-scores: antidepressant prescribing, QOF depression and DLA/PIP. "
        "The hospital component is unavailable, so this is a public-data proxy, not official SAMHI.\n\n"
        "For each one-year-ahead origin (2017–2022), component means and standard deviations are fitted using only 2011 through the preceding year. "
        "The index and its lagged component features are rebuilt using that origin-specific reference. Models train on all available England LSOAs before the origin; "
        "metrics are reported for England, Lincolnshire, and rural/urban Lincolnshire subsets. Skill is expressed as percentage RMSE reduction against persistence, "
        "so values remain comparable when each fold has its own index scale.\n\n"
        "The rural/urban split uses the 2021 ONS classification transferred to 2011 LSOAs through the exact-fit lookup. Mixed/unclassified Lincolnshire areas are excluded from the rural/urban metrics.\n"
    )
    print(f"Saved origin-specific SAMHI-3 results to {output}")


if __name__ == "__main__":
    main()

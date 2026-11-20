"""Evaluate Public SAMHI-3 forecasts against observed proxy values in 2023–25.

This extends the origin-specific one-year-ahead evaluation beyond the years
available in the initial analysis. Each year is a new forecast origin: models
are trained nationally on earlier years, and component standardisation uses
only data available before the held-out year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _shared import import_module, project_root


def main() -> None:
    root = project_root()
    origin_eval = import_module(
        root / "scripts/machine_learning/samhi/18_public_samhi3_origin_evaluation.py",
        "samhi3_post2022_origin_eval_helpers",
    )
    samhi3, tournament = origin_eval.load_helpers(root)
    raw = samhi3.load_panel(root)
    raw = raw[raw["year"].between(2011, 2025)].sort_values(["lsoa11", "year"]).reset_index(drop=True)
    if not set((2023, 2024, 2025)).issubset(set(raw["year"].unique())):
        raise ValueError("Observed Public SAMHI-3 component data are not available for every year 2023–25")

    rurality, lincolnshire_codes = origin_eval.load_rurality(root)
    raw["rurality"] = raw["lsoa11"].map(rurality)
    results: list[dict] = []
    lincolnshire_predictions: list[pd.DataFrame] = []
    standardisation_rows: list[dict] = []
    feature_set = "Index history + components"

    for origin in range(2023, 2026):
        panel, z_columns, references = origin_eval.make_origin_panel(
            raw, samhi3.COMPONENTS, origin
        )
        standardisation_rows.extend(references)
        target = "public_samhi3_index"
        train = panel[(panel["year"] < origin) & panel[target].notna()].copy()
        test = panel[(panel["year"] == origin) & panel[target].notna()].copy()
        z_lags = [f"{column}_lag1" for column in z_columns]
        features = ["public_samhi3_index_lag1", *z_lags, "year_centered"]
        test = test.dropna(subset=features).copy()
        if train.empty or test.empty:
            raise ValueError(f"No complete training or test data for forecast year {origin}")

        actual = test[target].to_numpy(dtype=float)
        persistence = test["public_samhi3_index_lag1"].to_numpy(dtype=float)
        scopes = {
            "England": np.ones(len(test), dtype=bool),
            "Lincolnshire": test["lsoa11"].isin(lincolnshire_codes).to_numpy(),
            "Lincolnshire rural": (
                test["lsoa11"].isin(lincolnshire_codes) & test["rurality"].eq("Rural")
            ).to_numpy(),
            "Lincolnshire urban": (
                test["lsoa11"].isin(lincolnshire_codes) & test["rurality"].eq("Urban")
            ).to_numpy(),
        }

        for scope, mask in scopes.items():
            if mask.any():
                results.append(origin_eval.metric_row(
                    actual[mask], persistence[mask], persistence[mask], "Persistence",
                    feature_set=feature_set, origin=origin, scope=scope,
                ))

        county_mask = test["lsoa11"].isin(lincolnshire_codes).to_numpy()
        for model_name, model in tournament.make_models().items():
            model.fit(train[features], train[target])
            prediction = np.asarray(model.predict(test[features]), dtype=float)
            for scope, mask in scopes.items():
                if mask.any():
                    results.append(origin_eval.metric_row(
                        actual[mask], prediction[mask], persistence[mask], model_name,
                        feature_set=feature_set, origin=origin, scope=scope,
                    ))
            if county_mask.any():
                county = test.loc[county_mask, ["lsoa11", "rurality"]].copy()
                county["origin_year"] = origin
                county["held_out_year"] = origin
                county["model"] = model_name
                county["actual_public_samhi3"] = actual[county_mask]
                county["prediction"] = prediction[county_mask]
                county["persistence"] = persistence[county_mask]
                county["absolute_error"] = np.abs(actual[county_mask] - prediction[county_mask])
                lincolnshire_predictions.append(county)

        print(
            f"Completed held-out year {origin}: train={len(train):,}, "
            f"test={len(test):,}, Lincolnshire={int(county_mask.sum()):,}"
        )

    output = root / "scripts/machine_learning/samhi/results/public_samhi3_post2022_eval"
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(results)
    metrics.to_csv(output / "rolling_origin_metrics.csv", index=False)
    pd.concat(lincolnshire_predictions, ignore_index=True).to_csv(
        output / "lincolnshire_predictions.csv", index=False
    )
    pd.DataFrame(standardisation_rows).to_csv(
        output / "standardisation_reference.csv", index=False
    )

    summary = metrics.groupby(["scope", "model"], as_index=False).agg(
        years=("held_out_year", "nunique"),
        mean_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "mean"),
        median_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "median"),
        years_better_than_persistence=(
            "rmse_skill_vs_persistence_pct", lambda values: int((values > 0).sum())
        ),
        mean_directional_accuracy=("directional_accuracy", "mean"),
        mean_n_lsoas=("n_lsoas", "mean"),
    )
    summary.to_csv(output / "skill_summary.csv", index=False)
    yearly = metrics[metrics["model"].ne("Persistence")].groupby(
        ["held_out_year", "scope"], as_index=False
    ).agg(
        model_families=("model", "nunique"),
        mean_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "mean"),
        min_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "min"),
        max_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "max"),
        models_better_than_persistence=(
            "rmse_skill_vs_persistence_pct", lambda values: int((values > 0).sum())
        ),
        mean_directional_accuracy=("directional_accuracy", "mean"),
        mean_n_lsoas=("n_lsoas", "mean"),
    )
    yearly.to_csv(output / "year_scope_summary.csv", index=False)
    (output / "README.md").write_text(
        "# Post-2022 Public SAMHI-3 evaluation\n\n"
        "This is a rolling one-year-ahead evaluation of the three-component Public SAMHI-3 proxy for 2023–2025. "
        "For each held-out year, models are trained on all earlier years and refitted at the next origin. "
        "The target is the equal-weight mean of antidepressant, QOF depression and DLA/PIP component z-scores; "
        "the hospital component is unavailable, so the target is a proxy rather than official SAMHI.\n\n"
        "Component means and standard deviations are calculated from 2011 through the year before each forecast. "
        "The model feature set matches Figure 2: lagged proxy index, lagged standardised components and year. "
        "Models are trained nationally; metrics are reported for England, Lincolnshire, and Lincolnshire rural/urban "
        "subsets. RMSE skill is the percentage reduction relative to persistence.\n\n"
        "This extends the one-year-ahead rolling-origin evaluation beyond 2022; it is not a fixed-origin recursive "
        "forecast from 2022. The rural/urban split uses the 2021 ONS classification transferred to 2011 LSOAs "
        "through the exact-fit lookup; mixed/unclassified Lincolnshire areas are excluded from those two subsets.\n\n"
        "Source continuity is a limitation: PLDR DLA/PIP data end in 2022 and calibrated DWP August DLA/PIP values "
        "are used from 2023. The 2024 QOF depression value is carried forward from the previous public vintage. "
        "These source transitions may contribute to year-to-year changes in forecast performance and should be "
        "considered before interpreting the results as evidence of stable future performance.\n"
    )
    print(f"Saved post-2022 Public SAMHI-3 evaluation to {output}")


if __name__ == "__main__":
    main()

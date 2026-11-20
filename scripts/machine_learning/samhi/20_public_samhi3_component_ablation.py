"""Measure the predictive value of SAMHI-3 lagged component inputs.

The outcome remains the complete three-component proxy in every ablation.
Only lagged predictor columns are removed, preserving the same LSOA sample,
origin-specific standardisation, models and folds as Figure 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _shared import import_module, project_root


def main() -> None:
    root = project_root()
    origin_eval = import_module(
        root / "scripts/machine_learning/samhi/18_public_samhi3_origin_evaluation.py",
        "samhi3_ablation_origin_eval_helpers",
    )
    samhi3, tournament = origin_eval.load_helpers(root)
    raw = samhi3.load_panel(root)
    raw = raw.sort_values(["lsoa11", "year"]).reset_index(drop=True)
    rurality, lincolnshire_codes = origin_eval.load_rurality(root)
    raw["rurality"] = raw["lsoa11"].map(rurality)

    # Check proxy alignment on the overlapping published-SAMHI years using
    # the same fixed 2011–2022 component reference as the public proxy index.
    fixed_index, reference = samhi3.build_index(raw)
    overlap = fixed_index[
        fixed_index["year"].between(2011, 2022)
        & fixed_index["public_samhi3_index"].notna()
        & fixed_index["samhi_index"].notna()
    ]
    correlations = []
    for year, group in overlap.groupby("year"):
        correlations.append({
            "year": int(year),
            "n_lsoas": int(len(group)),
            "pearson_r": group["public_samhi3_index"].corr(group["samhi_index"]),
            "spearman_rho": group["public_samhi3_index"].corr(group["samhi_index"], method="spearman"),
        })
    correlation_frame = pd.DataFrame(correlations)
    correlation_summary = pd.DataFrame([{
        "period": "2011-2022",
        "years": int(correlation_frame["year"].nunique()),
        "mean_annual_pearson_r": correlation_frame["pearson_r"].mean(),
        "mean_annual_spearman_rho": correlation_frame["spearman_rho"].mean(),
        "min_annual_pearson_r": correlation_frame["pearson_r"].min(),
        "max_annual_pearson_r": correlation_frame["pearson_r"].max(),
    }])

    result_rows: list[dict] = []
    lincolnshire_rows: list[pd.DataFrame] = []
    full_label = "History + all 3 components"
    output = root / "scripts/machine_learning/samhi/results/public_samhi3_component_ablation"
    output.mkdir(parents=True, exist_ok=True)
    correlation_frame.to_csv(output / "proxy_validity_by_year.csv", index=False)
    correlation_summary.to_csv(output / "proxy_validity_summary.csv", index=False)

    for origin in range(2017, 2023):
        panel, z_columns, _ = origin_eval.make_origin_panel(raw, samhi3.COMPONENTS, origin)
        target = "public_samhi3_index"
        train = panel[(panel["year"] < origin) & panel[target].notna()].copy()
        test = panel[(panel["year"] == origin) & panel[target].notna()].copy()
        z_lags = [f"{column}_lag1" for column in z_columns]
        full_features = ["public_samhi3_index_lag1", *z_lags, "year_centered"]
        # Use a matched complete-case test sample for every feature ablation.
        test = test.dropna(subset=full_features).copy()
        if train.empty or test.empty:
            raise ValueError(f"No complete training or test data for origin {origin}")

        feature_sets = {
            full_label: full_features,
            "Index history only": ["public_samhi3_index_lag1", "year_centered"],
            "Without antidepressant lag": [
                "public_samhi3_index_lag1", *z_lags[1:], "year_centered"
            ],
            "Without QOF lag": [
                "public_samhi3_index_lag1", z_lags[0], z_lags[2], "year_centered"
            ],
            "Without DLA/PIP lag": [
                "public_samhi3_index_lag1", *z_lags[:2], "year_centered"
            ],
        }
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
        county_mask = test["lsoa11"].isin(lincolnshire_codes).to_numpy()

        for model_name, model in tournament.make_models().items():
            for feature_label, features in feature_sets.items():
                model.fit(train[features], train[target])
                prediction = np.asarray(model.predict(test[features]), dtype=float)
                for scope, mask in scopes.items():
                    if mask.any():
                        result_rows.append(origin_eval.metric_row(
                            actual[mask], prediction[mask], persistence[mask], model_name,
                            feature_set=feature_label, origin=origin, scope=scope,
                        ))
                if county_mask.any():
                    county = test.loc[county_mask, ["lsoa11", "rurality"]].copy()
                    county["held_out_year"] = origin
                    county["model"] = model_name
                    county["feature_set"] = feature_label
                    county["actual_public_samhi3"] = actual[county_mask]
                    county["prediction"] = prediction[county_mask]
                    county["persistence"] = persistence[county_mask]
                    lincolnshire_rows.append(county)

        print(
            f"Completed ablation origin {origin}: train={len(train):,}, "
            f"matched test={len(test):,}, Lincolnshire={int(county_mask.sum()):,}"
        )

    metrics = pd.DataFrame(result_rows)
    metrics.to_csv(output / "rolling_origin_metrics.csv", index=False)
    pd.concat(lincolnshire_rows, ignore_index=True).to_csv(
        output / "lincolnshire_predictions.csv", index=False
    )
    summary = metrics.groupby(["scope", "feature_set"], as_index=False).agg(
        model_families=("model", "nunique"),
        model_fold_cases=("model", "size"),
        mean_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "mean"),
        median_rmse_skill_vs_persistence_pct=("rmse_skill_vs_persistence_pct", "median"),
        model_fold_cases_better_than_persistence=(
            "rmse_skill_vs_persistence_pct", lambda values: int((values > 0).sum())
        ),
        mean_directional_accuracy=("directional_accuracy", "mean"),
    )
    summary.to_csv(output / "feature_set_summary.csv", index=False)

    rmse_wide = metrics.pivot_table(
        index=["held_out_year", "scope", "model"],
        columns="feature_set", values="rmse", aggfunc="first",
    )
    delta_rows = []
    for feature_label in feature_sets:
        change = 100 * (rmse_wide[feature_label] - rmse_wide[full_label]) / rmse_wide[full_label]
        for (year, scope, model), value in change.items():
            delta_rows.append({
                "held_out_year": year,
                "scope": scope,
                "model": model,
                "feature_set": feature_label,
                "rmse_change_vs_full_pct": value,
            })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(output / "rmse_change_vs_full.csv", index=False)
    delta_summary = delta.groupby(["scope", "feature_set"], as_index=False).agg(
        model_fold_cases=("rmse_change_vs_full_pct", "size"),
        mean_rmse_change_vs_full_pct=("rmse_change_vs_full_pct", "mean"),
        median_rmse_change_vs_full_pct=("rmse_change_vs_full_pct", "median"),
        cases_full_model_better=("rmse_change_vs_full_pct", lambda values: int((values > 0).sum())),
    )
    delta_summary.to_csv(output / "rmse_change_summary.csv", index=False)

    reference.to_csv(output / "fixed_reference_2011_2022.csv", index=False)
    (output / "README.md").write_text(
        "# Public SAMHI-3 proxy validity and component ablation\n\n"
        "Proxy validity is measured as the annual cross-sectional Pearson and Spearman correlation between the equal-weight "
        "three-component proxy and published SAMHI for 2011–2022. This indicates alignment, not equivalence.\n\n"
        "The ablation uses the same six origin-specific folds (2017–2022), national training cohort, rurality groups, "
        "models and test LSOAs as Figure 2. Component standardisation at each origin uses only prior years. The target "
        "remains the complete three-component proxy in every run; only lagged predictor columns are removed. This tests "
        "predictive information in the lagged component features, not the effect of removing a component from the proxy target.\n\n"
        "The full feature set is lagged proxy history plus all three lagged components and year. Ablations compare it with "
        "index history alone and versions leaving out one lagged component. RMSE skill is the reduction relative to "
        "persistence. `rmse_change_summary.csv` reports the percentage RMSE change from the full feature set; positive values "
        "mean the ablated set had higher RMSE. All comparisons use a matched complete-case test sample within each origin.\n"
    )
    print(f"Saved Public SAMHI-3 proxy and ablation results to {output}")


if __name__ == "__main__":
    main()

"""
01_baseline_autoregressive.py
-----------------------------
Autoregressive machine learning baseline models for Small Area Mental Health Index (SAMHI) forecasting.

Evaluates:
  1. Naive Persistence Baseline (y_t = y_{t-1})
  2. Naive Momentum / Drift Baseline (y_t = y_{t-1} + (y_{t-1} - y_{t-2}))
  3. Ridge Regression (L2 Regularized Linear AR)
  4. Random Forest Regressor
  5. LightGBM Regressor

Evaluation Split:
  - Train: 2014-2018
  - Validation: 2019 (Pre-COVID baseline check)
  - Test: 2020-2022 (COVID and immediate post-COVID shock)

Supports:
  - --scope {lincolnshire, national, both}
  - --target {level, change}
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SAMHI_Baseline")

# Lincolnshire Local Authority Districts
LINCOLNSHIRE_LADS = {
    "Boston",
    "East Lindsey",
    "Lincoln",
    "North Kesteven",
    "South Holland",
    "South Kesteven",
    "West Lindsey"
}


def get_project_root() -> Path:
    """Resolve the project root directory assuming this script is in scripts/machine_learning/samhi."""
    current = Path(__file__).resolve()
    # Go up 3 levels: samhi -> machine_learning -> scripts -> LUMHR_dashboard
    return current.parents[3]


def load_raw_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load national SAMHI data and the 2011 to 2021 LSOA lookup table."""
    samhi_path = project_root / "datasets" / "samhi" / "samhi_21_01_v5.00_2011_2022_LSOA.csv"
    lookup_path = (
        project_root
        / "datasets"
        / "lincolnshire_lsoa"
        / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )

    if not samhi_path.exists():
        raise FileNotFoundError(f"SAMHI dataset not found at {samhi_path}")
    if not lookup_path.exists():
        raise FileNotFoundError(f"LSOA lookup table not found at {lookup_path}")

    logger.info(f"Loading SAMHI dataset from {samhi_path.name}...")
    samhi_df = pd.read_csv(samhi_path)
    samhi_df["lsoa11"] = samhi_df["lsoa11"].astype(str).str.strip()

    logger.info(f"Loading LSOA lookup table from {lookup_path.name}...")
    lookup_df = pd.read_csv(lookup_path)
    lookup_df["LSOA11CD"] = lookup_df["LSOA11CD"].astype(str).str.strip()
    lookup_df["LSOA21CD"] = lookup_df["LSOA21CD"].astype(str).str.strip()
    lookup_df["LAD22NM"] = lookup_df["LAD22NM"].astype(str).str.strip()

    return samhi_df, lookup_df


def build_merged_master_dataset(samhi_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SAMHI with lookup table.
    Ensures every 2011 LSOA has its corresponding LAD and Lincolnshire flag.
    """
    primary_lookup = lookup_df.sort_values("ObjectId").drop_duplicates(subset=["LSOA11CD"]).copy()

    merged = samhi_df.merge(
        primary_lookup[["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22CD", "LAD22NM"]],
        left_on="lsoa11",
        right_on="LSOA11CD",
        how="left"
    )

    # Flag Lincolnshire LSOAs
    merged["is_lincolnshire"] = merged["LAD22NM"].isin(LINCOLNSHIRE_LADS).astype(int)
    num_lincs = merged["is_lincolnshire"].sum()
    logger.info(f"Merged master dataset created: {len(merged):,} LSOAs total ({num_lincs} Lincolnshire 2011 LSOAs)")

    return merged


def create_panel_feature_dataset(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2022
) -> pd.DataFrame:
    """
    Transform wide annual SAMHI columns into a long panel dataset with autoregressive features.
    """
    records = []

    for t in range(start_year, end_year + 1):
        target_col = f"samhi_index.{t}"
        target_dec_col = f"samhi_dec.{t}"
        lag1_col = f"samhi_index.{t-1}"
        lag2_col = f"samhi_index.{t-2}"
        lag3_col = f"samhi_index.{t-3}"
        dec1_col = f"samhi_dec.{t-1}"

        if target_col not in df.columns or lag3_col not in df.columns:
            continue

        y_true = df[target_col].values
        d_true = df[target_dec_col].values if target_dec_col in df.columns else np.zeros(len(df))

        l1 = df[lag1_col].values
        l2 = df[lag2_col].values
        l3 = df[lag3_col].values
        d1 = df[dec1_col].values if dec1_col in df.columns else np.zeros(len(df))

        delta1 = l1 - l2
        delta2 = l2 - l3
        accel = delta1 - delta2

        # Rolling 3-year window across lags
        stacked_lags = np.column_stack([l1, l2, l3])
        roll_mean = np.mean(stacked_lags, axis=1)
        roll_std = np.std(stacked_lags, axis=1)
        roll_min = np.min(stacked_lags, axis=1)
        roll_max = np.max(stacked_lags, axis=1)

        year_records = pd.DataFrame({
            "lsoa11": df["lsoa11"],
            "LSOA21CD": df.get("LSOA21CD", df["lsoa11"]),
            "LSOA21NM": df.get("LSOA21NM", ""),
            "LAD22NM": df.get("LAD22NM", ""),
            "is_lincolnshire": df["is_lincolnshire"],
            "year": t,
            "target": y_true,
            "target_decile": d_true,
            "target_change": y_true - l1,
            # Features
            "lag_1": l1,
            "lag_2": l2,
            "lag_3": l3,
            "delta_1": delta1,
            "delta_2": delta2,
            "acceleration": accel,
            "rolling_mean_3yr": roll_mean,
            "rolling_std_3yr": roll_std,
            "rolling_min_3yr": roll_min,
            "rolling_max_3yr": roll_max,
            "decile_lag_1": d1
        })
        records.append(year_records)

    panel_df = pd.concat(records, ignore_index=True)
    logger.info(f"Panel feature dataset built: {len(panel_df):,} rows spanning years {start_year} to {end_year}")
    return panel_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, lag_1: np.ndarray, d_true: np.ndarray = None) -> Dict[str, float]:
    """Calculate RMSE, MAE, R2, Directional Accuracy of Change, and Decile Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: did the model predict the correct direction of change compared to lag_1?
    actual_change = y_true - lag_1
    pred_change = y_pred - lag_1
    dir_correct = (np.sign(pred_change) == np.sign(actual_change))
    dir_correct = dir_correct | ((pred_change == 0) & (actual_change == 0))
    dir_acc = np.mean(dir_correct) * 100.0

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Dir_Acc_%": float(dir_acc)
    }

    # If decile true is provided, estimate predicted decile by empirical rank / quantiles
    if d_true is not None and len(d_true) > 0 and not np.all(d_true == 0):
        try:
            pred_dec = pd.qcut(y_pred, q=10, labels=False, duplicates="drop") + 1
            dec_acc = np.mean(pred_dec == d_true) * 100.0
            dec_within1 = np.mean(np.abs(pred_dec - d_true) <= 1) * 100.0
            metrics["Dec_Exact_%"] = float(dec_acc)
            metrics["Dec_Within1_%"] = float(dec_within1)
        except Exception:
            metrics["Dec_Exact_%"] = 0.0
            metrics["Dec_Within1_%"] = 0.0

    return metrics


def train_and_evaluate(
    panel_df: pd.DataFrame,
    scope: str = "lincolnshire",
    target_type: str = "level",
    output_dir: Path = None,
    experiment_set: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate models.
    
    Split:
      - Train: 2014-2018
      - Validation: 2019
      - Test: 2020, 2021, 2022
    """
    feature_cols = [
        "lag_1", "lag_2", "lag_3",
        "delta_1", "delta_2", "acceleration",
        "rolling_mean_3yr", "rolling_std_3yr", "rolling_min_3yr", "rolling_max_3yr",
        "decile_lag_1"
    ]

    # Filter training data based on scope
    if scope == "lincolnshire":
        data_train_pool = panel_df[panel_df["is_lincolnshire"] == 1]
    else:
        # national scope: train on national data
        data_train_pool = panel_df

    if experiment_set == 1:
        train_years, val_year, test_years = (2014, 2018), 2019, (2020, 2022)
    elif experiment_set == 2:
        train_years, val_year, test_years = (2014, 2016), 2017, (2018, 2019)
    else:
        raise ValueError("experiment_set must be 1 or 2")

    train_mask = data_train_pool["year"].between(*train_years)
    val_mask = data_train_pool["year"] == val_year
    test_mask = data_train_pool["year"].between(*test_years)

    df_train = data_train_pool[train_mask].copy()
    df_val = data_train_pool[val_mask].copy()
    df_test = data_train_pool[test_mask].copy()

    logger.info(f"[{scope.upper()}] Dataset sizes - Train ({train_years[0]}-{train_years[1]}): {len(df_train):,}, Val ({val_year}): {len(df_val):,}, Test ({test_years[0]}-{test_years[1]}): {len(df_test):,}")

    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values if target_type == "level" else df_train["target_change"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["target"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["target"].values

    # Models dictionary
    models = {
        "Naive Persistence (y_{t-1})": None,
        "Naive Momentum (y_{t-1} + delta)": None,
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=42))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=20,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
    }

    # Fit trained ML models
    trained_models = {}
    feature_importances = {}

    for name, model in models.items():
        if model is not None:
            logger.info(f"Fitting {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

            if name == "LightGBM":
                feature_importances["LightGBM"] = model.feature_importances_
            elif name == "Random Forest":
                feature_importances["Random Forest"] = model.feature_importances_
            elif name == "Ridge Regression":
                feature_importances["Ridge Regression"] = np.abs(model.named_steps["ridge"].coef_)

    # Collect predictions
    val_preds = {}
    test_preds = {}

    for name in models:
        # Predict on validation (2019)
        if name == "Naive Persistence (y_{t-1})":
            val_p = df_val["lag_1"].values
            test_p = df_test["lag_1"].values
        elif name == "Naive Momentum (y_{t-1} + delta)":
            val_p = df_val["lag_1"].values + df_val["delta_1"].values
            test_p = df_test["lag_1"].values + df_test["delta_1"].values
        else:
            model = trained_models[name]
            if target_type == "level":
                val_p = model.predict(X_val)
                test_p = model.predict(X_test)
            else:
                val_p = df_val["lag_1"].values + model.predict(X_val)
                test_p = df_test["lag_1"].values + model.predict(X_test)

        val_preds[name] = val_p
        test_preds[name] = test_p

    # Compute Evaluation Metrics Table
    metrics_list = []

    val_label = f"Validation ({val_year})"
    test_label = f"Test ({test_years[0]}-{test_years[1]} Aggregate)"

    # 1. Validation Set
    for name in models:
        m = compute_metrics(
            y_val,
            val_preds[name],
            df_val["lag_1"].values,
            df_val["target_decile"].values
        )
        m.update({
            "Scope": scope,
            "Split": val_label,
            "Model": name,
            "N_Obs": len(df_val)
        })
        metrics_list.append(m)

    # 2. Test Set Aggregate (2020-2022 Shock)
    for name in models:
        m = compute_metrics(
            y_test,
            test_preds[name],
            df_test["lag_1"].values,
            df_test["target_decile"].values
        )
        m.update({
            "Scope": scope,
            "Split": test_label,
            "Model": name,
            "N_Obs": len(df_test)
        })
        metrics_list.append(m)

    # 3. Test Set Breakdown by Individual Year
    for yr in range(test_years[0], test_years[1] + 1):
        yr_idx = (df_test["year"] == yr).values
        df_yr = df_test[yr_idx]
        y_yr = y_test[yr_idx]

        for name in models:
            m = compute_metrics(
                y_yr,
                test_preds[name][yr_idx],
                df_yr["lag_1"].values,
                df_yr["target_decile"].values
            )
            m.update({
                "Scope": scope,
                "Split": f"Test ({yr})",
                "Model": name,
                "N_Obs": len(df_yr)
            })
            metrics_list.append(m)

    # 4. If National scope, also evaluate how the national model performs on Lincolnshire specifically!
    if scope == "national":
        lincs_test_idx = (df_test["is_lincolnshire"] == 1).values
        if lincs_test_idx.sum() > 0:
            df_test_lincs = df_test[lincs_test_idx]
            y_test_lincs = y_test[lincs_test_idx]

            for name in models:
                m = compute_metrics(
                    y_test_lincs,
                    test_preds[name][lincs_test_idx],
                    df_test_lincs["lag_1"].values,
                    df_test_lincs["target_decile"].values
                )
                m.update({
                    "Scope": "national_on_lincolnshire",
                    "Split": f"Test ({test_years[0]}-{test_years[1]} Lincs Subset)",
                    "Model": name,
                    "N_Obs": len(df_test_lincs)
                })
                metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    cols_order = ["Scope", "Split", "Model", "N_Obs", "RMSE", "MAE", "R2", "Dir_Acc_%", "Dec_Exact_%", "Dec_Within1_%"]
    cols_order = [c for c in cols_order if c in metrics_df.columns]
    metrics_df = metrics_df[cols_order]

    # Build predictions dataframe for export
    pred_export = df_test[[
        "lsoa11", "LSOA21CD", "LSOA21NM", "LAD22NM", "is_lincolnshire", "year", "target", "lag_1", "delta_1"
    ]].copy()
    pred_export.rename(columns={"target": "actual_samhi"}, inplace=True)

    for name in models:
        key = name.split()[0].lower()
        pred_export[f"pred_{key}"] = test_preds[name]
        pred_export[f"err_{key}"] = pred_export[f"pred_{key}"] - pred_export["actual_samhi"]

    # Build feature importance dataframe
    feat_imp_df = pd.DataFrame({"feature": feature_cols})
    for m_name, imp in feature_importances.items():
        feat_imp_df[m_name] = (imp / np.sum(imp)) * 100.0
    feat_imp_df = feat_imp_df.sort_values(by=list(feature_importances.keys())[-1], ascending=False)

    return metrics_df, pred_export, feat_imp_df


def print_formatted_summary(metrics_df: pd.DataFrame, scope: str):
    """Print readable benchmark tables."""
    print("\n" + "=" * 95)
    print(f" SAMHI MACHINE LEARNING BASELINE BENCHMARK ({scope.upper()})")
    print("=" * 95)

    splits = metrics_df["Split"].unique()
    for s in splits:
        sub = metrics_df[metrics_df["Split"] == s]
        print(f"\n--- {s} (N = {sub['N_Obs'].iloc[0]:,}) ---")
        display_cols = ["Model", "RMSE", "MAE", "R2", "Dir_Acc_%"]
        if "Dec_Exact_%" in sub.columns:
            display_cols.extend(["Dec_Exact_%", "Dec_Within1_%"])
        print(sub[display_cols].to_string(index=False))

    print("\n" + "=" * 95)


def main():
    parser = argparse.ArgumentParser(description="SAMHI Autoregressive Machine Learning Baselines")
    parser.add_argument("--experiment-set", type=int, choices=[1, 2], default=1,
                        help="1=COVID-era test (2020-2022), 2=pre-COVID test (2018-2019)")
    parser.add_argument(
        "--scope",
        choices=["lincolnshire", "national", "both"],
        default="both",
        help="Geographic training scope: 'lincolnshire', 'national', or 'both'"
    )
    parser.add_argument(
        "--target",
        choices=["level", "change"],
        default="level",
        help="Predict raw SAMHI level or annual change"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/machine_learning/samhi/results",
        help="Directory to save metric tables and predictions"
    )
    args = parser.parse_args()

    project_root = get_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    samhi_df, lookup_df = load_raw_data(project_root)

    # 2. Master merge
    master_df = build_merged_master_dataset(samhi_df, lookup_df)

    # 3. Create panel dataset
    panel_df = create_panel_feature_dataset(master_df, start_year=2014, end_year=2022)

    scopes_to_run = ["lincolnshire", "national"] if args.scope == "both" else [args.scope]

    all_metrics = []

    for sc in scopes_to_run:
        logger.info(f"\n>>> Running Benchmark for Scope: {sc.upper()} <<<")
        metrics_df, pred_export, feat_imp = train_and_evaluate(
            panel_df,
            scope=sc,
            target_type=args.target,
            output_dir=output_dir,
            experiment_set=args.experiment_set
        )
        print_formatted_summary(metrics_df, sc)
        all_metrics.append(metrics_df)

        # Save files
        suffix = "2020_2022" if args.experiment_set == 1 else "pre_covid_2018_2019"
        metrics_path = output_dir / f"baseline_metrics_{suffix}_{sc}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")

        preds_path = output_dir / f"baseline_predictions_{suffix}_{sc}.csv"
        pred_export.to_csv(preds_path, index=False)
        logger.info(f"Saved test predictions to {preds_path}")

        feat_path = output_dir / f"feature_importance_{suffix}_{sc}.csv"
        feat_imp.to_csv(feat_path, index=False)
        logger.info(f"Saved feature importances to {feat_path}")
        print("\nFeature Importances (%):")
        print(feat_imp.to_string(index=False))

    if len(all_metrics) > 1:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        comparison_path = output_dir / f"baseline_metrics_comparison_{suffix}.csv"
        combined_metrics.to_csv(comparison_path, index=False)
        logger.info(f"Saved combined metrics comparison to {comparison_path}")

    logger.info("Baseline autoregressive experiments completed successfully!")


if __name__ == "__main__":
    main()


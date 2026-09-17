"""Create explanatory charts from the SAMHI machine-learning result CSVs.

The script does not retrain any models. It reads the CSV files produced by
01_baseline_autoregressive.py, 02_multimodal_features.py, and
03_forward_projections.py and writes publication-ready PNG files.

Examples
--------
python 04_plot_results.py --scope both
python 04_plot_results.py --scope lincolnshire --output-dir results/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


COLOURS = {
    "actual": "#1f2937",
    "baseline": "#9ca3af",
    "model": "#2563eb",
    "secondary": "#0f766e",
    "positive": "#15803d",
    "negative": "#b91c1c",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def clean_model_name(name: str) -> str:
    """Turn CSV model names into compact labels suitable for chart axes."""
    name = re.sub(r"\s*\([^)]*\)", "", str(name))
    return name.replace("Multimodal ", "").replace("AR Baseline ", "")


def save(fig: plt.Figure, output: Path, title: str) -> None:
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.05, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(results: Path, output: Path, scope: str) -> None:
    path = results / "multimodal_metrics_comparison_2020_2022.csv"
    if not path.exists():
        path = results / "multimodal_metrics_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["Scope"].eq(scope)].copy()
    if df.empty:
        return
    # Prefer the stress-test rows when available; otherwise use all available rows.
    test = df[df["Split"].astype(str).str.contains("Test|2020", case=False, regex=True)]
    if not test.empty:
        df = test
    df["label"] = df["Model"].map(clean_model_name)
    df = df.sort_values("RMSE", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(df) * 0.38)))
    y = range(len(df))
    axes[0].barh(list(y), df["RMSE"], color=COLOURS["model"])
    axes[0].set_yticks(list(y), df["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (lower is better)")
    axes[0].grid(axis="x", alpha=.25)
    axes[1].barh(list(y), df["R2"], color=COLOURS["secondary"])
    axes[1].set_yticks(list(y), df["label"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("R² (higher is better)")
    axes[1].grid(axis="x", alpha=.25)
    split = df["Split"].iloc[0]
    save(fig, output / f"01_model_performance_{scope}.png", f"SAMHI model performance — {scope.title()} ({split})")


def plot_predictions(results: Path, output: Path, scope: str) -> None:
    path = results / f"multimodal_predictions_2020_2022_{scope}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if df.empty or not pred_cols:
        return
    # Aggregate first: this keeps the national chart readable while retaining every LSOA.
    actual = df.groupby("year", as_index=False)["actual_samhi"].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual["year"], actual["actual_samhi"], marker="o", linewidth=3,
            color=COLOURS["actual"], label="Observed SAMHI")
    for col in pred_cols:
        label = clean_model_name(col.removeprefix("pred_"))
        colour = COLOURS["baseline"] if "baseline" in col else COLOURS["model"]
        pred = df.groupby("year", as_index=False)[col].mean()
        ax.plot(pred["year"], pred[col], marker="o", alpha=.75, label=label, color=colour)
    ax.axvspan(2020, 2022, color="#f59e0b", alpha=.08, label="COVID-era test period")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean SAMHI")
    ax.set_xticks(sorted(df["year"].unique()))
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=8)
    save(fig, output / f"02_observed_vs_predicted_{scope}.png", f"Observed and predicted mean SAMHI — {scope.title()}")


def plot_feature_importance(results: Path, output: Path, scope: str) -> None:
    candidates = [
        results / f"shap_feature_importance_elasticnet_2020_2022_{scope}.csv",
        results / f"shap_feature_importance_2020_2022_{scope}.csv",
        results / f"shap_feature_importance_elasticnet_pre_covid_2018_2019_{scope}.csv",
        results / f"shap_feature_importance_pre_covid_2018_2019_{scope}.csv",
        results / f"shap_feature_importance_{scope}.csv",
        results / f"feature_importance_{scope}.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return
    df = pd.read_csv(path)
    value = "importance_pct" if "importance_pct" in df.columns else next(c for c in df.columns if c != "feature")
    df[value] = pd.to_numeric(df[value], errors="coerce")
    df = df.dropna(subset=[value]).sort_values(value).tail(12)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"], df[value], color=COLOURS["secondary"])
    ax.set_xlabel("Mean absolute SHAP contribution (%)" if value == "importance_pct" else "Relative importance")
    ax.grid(axis="x", alpha=.25)
    save(fig, output / f"03_feature_importance_{scope}.png", f"Most influential features — {scope.title()}")


def plot_qof_features(results: Path, output: Path, scope: str) -> None:
    """Plot the contribution and distribution of the five QOF map measures."""
    shap_candidates = [
        results / f"shap_feature_importance_elasticnet_2020_2022_{scope}.csv",
        results / f"shap_feature_importance_2020_2022_{scope}.csv",
    ]
    shap_path = next((candidate for candidate in shap_candidates if candidate.exists()), None)
    pred_path = results / f"multimodal_predictions_2020_2022_{scope}.csv"
    if shap_path is None or not pred_path.exists():
        return
    labels = {
        "qof_mh002_pct": "MH002 — SMI care plan",
        "qof_mh021_pct": "MH021 — SMI health check",
        "qof_mh_pca_pct": "Mental Health PCA — SMI exceptions",
        "qof_dep_pca_pct": "Depression PCA — exceptions",
        "qof_dep004_pct": "DEP004 — depression review",
    }
    shap = pd.read_csv(shap_path)
    shap = shap[shap["feature"].isin(labels)].copy()
    shap["label"] = shap["feature"].map(labels)
    shap = shap.sort_values("importance_pct")
    pred = pd.read_csv(pred_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(shap["label"], shap["importance_pct"], color=COLOURS["secondary"])
    axes[0].set_xlabel("Mean absolute SHAP contribution (%)")
    axes[0].grid(axis="x", alpha=.25)
    values = pred[list(labels)].apply(pd.to_numeric, errors="coerce").melt(var_name="feature", value_name="rate").dropna()
    values["label"] = values["feature"].map(labels)
    values.boxplot(column="rate", by="label", ax=axes[1], grid=False, vert=False)
    axes[1].set_title("")
    axes[1].set_xlabel("Patient-weighted QOF rate (%)")
    axes[1].set_ylabel("")
    fig.suptitle("")
    save(fig, output / f"05_qof_feature_contribution_{scope}.png", f"QOF measures used by the SAMHI model — {scope.title()}")


def plot_projections(results: Path, output: Path, scope: str) -> None:
    path = results / "forward_projections_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["Scope"].eq(scope)].copy()
    if df.empty:
        return
    df["Year"] = df["Year"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(df["Year"], df["Mean_Projected_SAMHI"], marker="o", color=COLOURS["model"], linewidth=2.5)
    axes[0].axhline(0, color="#6b7280", linewidth=.8)
    axes[0].set_ylabel("Mean projected SAMHI")
    axes[0].set_xlabel("Projection year")
    axes[0].grid(alpha=.25)
    axes[1].plot(df["Year"], df["Pct_Areas_Worsening"], marker="o", color=COLOURS["negative"], label="Worsening")
    axes[1].plot(df["Year"], df["Pct_Areas_Improving"], marker="o", color=COLOURS["positive"], label="Improving")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Share of areas (%)")
    axes[1].set_xlabel("Projection year")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    save(fig, output / f"04_forward_projection_summary_{scope}.png", f"Forward SAMHI projections — {scope.title()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("lincolnshire", "national", "both"), default="both")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing result CSVs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG charts")
    args = parser.parse_args()
    results = args.results_dir or Path(__file__).resolve().parent / "results"
    output = args.output_dir or results / "plots"
    output.mkdir(parents=True, exist_ok=True)
    scopes = ("lincolnshire", "national") if args.scope == "both" else (args.scope,)
    for scope in scopes:
        plot_metrics(results, output, scope)
        plot_predictions(results, output, scope)
        plot_feature_importance(results, output, scope)
        plot_qof_features(results, output, scope)
        plot_projections(results, output, scope)
        print(f"Created charts for {scope} in {output}")


if __name__ == "__main__":
    main()

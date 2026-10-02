"""Plot the refreshed public-data SAMHI reconstruction and forecasts.

The script only reads CSV outputs from scripts 07--11; it does not retrain
models.  Charts are written to ``results/public_plots`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLOURS = {
    "BayesianRidge": "#2563eb",
    "BayesianRidge_public_lagged": "#2563eb",
    "ElasticNet": "#0f766e",
    "RandomForest": "#d97706",
    "ExtraTrees": "#7c3aed",
    "Persistence": "#6b7280",
    "PublicSpatialCAR": "#15803d",
    "BayesianRidge_Lincolnshire": "#2563eb",
}


def save(fig: plt.Figure, path: Path, title: str) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def model_rmse(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "public_model_comparison/rolling_origin_metrics.csv")
    summary = df.groupby("model", as_index=False)["rmse"].mean().sort_values("rmse")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(summary["model"], summary["rmse"], color=[COLOURS.get(x, "#2563eb") for x in summary["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean rolling-origin RMSE (lower is better)")
    ax.grid(axis="x", alpha=.25)
    for index, value in enumerate(summary["rmse"]):
        ax.text(value + .002, index, f"{value:.3f}", va="center", fontsize=9)
    save(fig, output / "01_public_model_ranking.png", "Public-data model comparison — 2017–2022")


def rolling_error(results: Path, output: Path) -> None:
    comparison = pd.read_csv(results / "public_model_comparison/rolling_origin_metrics.csv")
    persistence = pd.read_csv(results / "public_bayesian/rolling_origin_metrics.csv")
    persistence = persistence[persistence["model"].eq("Persistence")]
    df = pd.concat([comparison, persistence], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, group in df.groupby("model"):
        group = group.sort_values("year")
        ax.plot(group["year"], group["rmse"], marker="o", linewidth=2, label=model, color=COLOURS.get(model, "#374151"))
    ax.set_xlabel("Held-out year")
    ax.set_ylabel("RMSE")
    ax.set_xticks(sorted(df["year"].unique()))
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=8)
    save(fig, output / "02_public_rolling_origin_rmse.png", "Public-data rolling-origin error")


def forecast_trajectories(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "public_model_comparison/future_predictions_2023_2025.csv")
    means = df.groupby(["year", "model"], as_index=False)["prediction"].mean()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, group in means.groupby("model"):
        group = group.sort_values("year")
        ax.plot(group["year"], group["prediction"], marker="o", linewidth=2, label=model, color=COLOURS.get(model, "#374151"))
    bayesian = pd.read_csv(results / "public_bayesian/future_predictions_2023_2025.csv")
    interval = bayesian.groupby("year", as_index=False)[["interval_low", "interval_high"]].mean()
    ax.fill_between(interval["year"], interval["interval_low"], interval["interval_high"], color="#2563eb", alpha=.12, label="Bayesian mean 95% interval")
    ax.axhline(0, color="#6b7280", linewidth=.8)
    ax.set_xlabel("Forecast year")
    ax.set_ylabel("Mean predicted SAMHI scale")
    ax.set_xticks([2023, 2024, 2025])
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=8)
    save(fig, output / "03_public_forecast_trajectories.png", "Public-data recursive forecasts")


def component_correlations(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "component_reconstruction/component_vs_published_samhi_correlations.csv")
    df = df[df["component"].isin(["antidep_rate", "qof_dep_pct", "dla_pip", "three_component_mean_z"])]
    labels = {
        "antidep_rate": "Antidepressants",
        "qof_dep_pct": "QOF depression",
        "dla_pip": "DLA/PIP",
        "three_component_mean_z": "Three-component proxy",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for column, ax, title in [("pearson_r", axes[0], "Pearson"), ("spearman_r", axes[1], "Spearman")]:
        for component, group in df.groupby("component"):
            group = group.sort_values("year")
            ax.plot(group["year"], group[column], marker="o", linewidth=1.8, label=labels[component])
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Correlation with published SAMHI")
        ax.set_ylim(0.35, 0.95)
        ax.grid(alpha=.25)
    axes[1].legend(fontsize=8)
    save(fig, output / "04_component_correlations.png", "Observable-component agreement with published SAMHI")


def coverage(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "component_reconstruction/component_coverage.csv")
    components = ["antidep_rate", "qof_dep_pct", "dla_pip"]
    df = df[df["component"].isin(components)]
    labels = {"antidep_rate": "Antidepressants", "qof_dep_pct": "QOF depression", "dla_pip": "DLA/PIP"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for component, group in df.groupby("component"):
        group = group.sort_values("year")
        ax.plot(group["year"], group["coverage_pct"], marker="o", linewidth=2, label=labels[component])
    ax.set_xlabel("Year")
    ax.set_ylabel("LSOA coverage (%)")
    ax.set_ylim(0, 101)
    ax.grid(alpha=.25)
    ax.legend()
    save(fig, output / "05_component_coverage.png", "Coverage of observable public components")


def spatial_comparison(results: Path, output: Path) -> None:
    df = pd.read_csv(results / "public_spatial_bayesian/spatial_vs_bayesian_metrics.csv")
    summary = df.groupby("model", as_index=False)[["rmse", "interval_coverage_95"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colours = [COLOURS.get(model, "#374151") for model in summary["model"]]
    axes[0].bar(summary["model"], summary["rmse"], color=colours)
    axes[0].set_ylabel("Mean RMSE")
    axes[0].set_title("Error")
    axes[1].bar(summary["model"], summary["interval_coverage_95"], color=colours)
    axes[1].set_ylabel("Mean 95% interval coverage")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Calibration")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.25)
    save(fig, output / "06_spatial_bayesian_comparison.png", "Lincolnshire spatial Bayesian pilot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir / "public_plots"
    output.mkdir(parents=True, exist_ok=True)
    model_rmse(args.results_dir, output)
    rolling_error(args.results_dir, output)
    forecast_trajectories(args.results_dir, output)
    component_correlations(args.results_dir, output)
    coverage(args.results_dir, output)
    spatial_comparison(args.results_dir, output)
    print(f"Created public-data charts in {output}")


if __name__ == "__main__":
    main()

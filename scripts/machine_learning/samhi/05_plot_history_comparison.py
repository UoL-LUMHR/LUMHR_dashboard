"""Create plots and summary tables for the with/without SAMHI-history runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("multimodal_metrics_comparison*.csv")):
        period = "2020-2022" if "2020_2022" in path.name else "2018-2019"
        frame = pd.read_csv(path)
        frame["Period"] = period
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No multimodal_metrics_comparison*.csv files found")
    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["Split"].str.contains("Aggregate")].copy()


def plot_metric_comparison(aggregate: pd.DataFrame, plots_dir: Path) -> None:
    for period in sorted(aggregate["Period"].unique()):
        for scope in ["lincolnshire", "national"]:
            subset = aggregate[(aggregate["Period"] == period) & (aggregate["Scope"] == scope)]
            if subset.empty:
                continue
            pivot = subset.pivot(index="Model", columns="History_Mode", values="RMSE")
            pivot = pivot.sort_values("with_history")
            ax = pivot.plot(kind="bar", figsize=(12, 6), color=["#4472c4", "#ed7d31"])
            ax.set_title(f"Test RMSE: {scope.title()} ({period})")
            ax.set_xlabel("")
            ax.set_ylabel("RMSE (lower is better)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(title="Predictors")
            plt.tight_layout()
            plt.savefig(plots_dir / f"rmse_{scope}_{period}.png", dpi=180)
            plt.close()


def plot_history_delta(aggregate: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    rows = []
    for (period, scope, model), group in aggregate.groupby(["Period", "Scope", "Model"]):
        values = group.set_index("History_Mode")
        if not {"with_history", "without_history"}.issubset(values.index):
            continue
        rows.append(
            {
                "Period": period,
                "Scope": scope,
                "Model": model,
                "RMSE_with_history": values.loc["with_history", "RMSE"],
                "RMSE_without_history": values.loc["without_history", "RMSE"],
                "RMSE_change_without_minus_with": values.loc["without_history", "RMSE"]
                - values.loc["with_history", "RMSE"],
                "R2_with_history": values.loc["with_history", "R2"],
                "R2_without_history": values.loc["without_history", "R2"],
                "R2_change_without_minus_with": values.loc["without_history", "R2"]
                - values.loc["with_history", "R2"],
            }
        )
    delta = pd.DataFrame(rows)
    delta.to_csv(plots_dir.parent / "history_effect_by_model.csv", index=False)

    for period in sorted(delta["Period"].unique()):
        for scope in ["lincolnshire", "national"]:
            subset = delta[(delta["Period"] == period) & (delta["Scope"] == scope)]
            if subset.empty:
                continue
            subset = subset.sort_values("RMSE_change_without_minus_with")
            colors = ["#70ad47" if value > 0 else "#c00000" for value in subset["RMSE_change_without_minus_with"]]
            ax = subset.plot(
                x="Model",
                y="RMSE_change_without_minus_with",
                kind="bar",
                legend=False,
                figsize=(12, 6),
                color=colors,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"Effect of removing previous SAMHI predictors: {scope.title()} ({period})")
            ax.set_xlabel("")
            ax.set_ylabel("RMSE without history − RMSE with history")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(plots_dir / f"history_rmse_delta_{scope}_{period}.png", dpi=180)
            plt.close()
    return delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.results_dir)
    aggregate = aggregate_metrics(metrics)
    aggregate.to_csv(args.results_dir / "aggregate_test_metrics.csv", index=False)
    plot_metric_comparison(aggregate, plots_dir)
    plot_history_delta(aggregate, plots_dir)
    print(f"Wrote plots and summary tables to {args.results_dir}")


if __name__ == "__main__":
    main()

"""Export global SHAP feature impacts for the Public SAMHI-3 models."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import shap


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FEATURE_INFO = {
    "public_samhi3_index_lag1": (
        "Previous Year Public SAMHI-3 Proxy (t-1)",
        "Temporal History",
        "Most recently recorded proxy level used as the baseline.",
    ),
    "public_samhi3_z_antidepressant_lag1": (
        "Antidepressant Prescribing (t-1)",
        "Routine Indicators",
        "Lagged standardised antidepressant prescribing component.",
    ),
    "public_samhi3_z_qof_lag1": (
        "QOF Depression (t-1)",
        "Routine Indicators",
        "Lagged standardised QOF depression component.",
    ),
    "public_samhi3_z_welfare_lag1": (
        "DLA/PIP Mental-Health Claims (t-1)",
        "Routine Indicators",
        "Lagged standardised DLA/PIP mental-health and learning-difficulty component.",
    ),
    "year_centered": (
        "Calendar Year Trend",
        "Time Trend",
        "Linear time trend included to capture gradual change.",
    ),
}


def shap_array(explainer, matrix: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[0]
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Public SAMHI-3 SHAP feature impacts")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/public_samhi3_shap")
    parser.add_argument("--sample-size", type=int, default=1500)
    args = parser.parse_args()

    root = project_root()
    tournament = import_module(root / "scripts/machine_learning/samhi/15_public_target_model_tournament.py", "public_shap_tournament")
    panel = tournament.load_panel(root)
    features = [
        "public_samhi3_index_lag1",
        "public_samhi3_z_antidepressant_lag1",
        "public_samhi3_z_qof_lag1",
        "public_samhi3_z_welfare_lag1",
        "year_centered",
    ]
    train = panel[(panel["year"] <= 2022) & panel["public_samhi3_index"].notna()].copy()
    sample = train.sample(min(args.sample_size, len(train)), random_state=42)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for model_name, fitted in tournament.make_models().items():
        fitted.fit(train[features], train["public_samhi3_index"])
        prep = fitted.named_steps["prep"]
        estimator = fitted.named_steps["model"]
        matrix = prep.transform(sample[features])
        transformed_names = list(prep.get_feature_names_out(features))
        if model_name in {"BayesianRidge", "ElasticNet", "Ridge"}:
            explainer = shap.LinearExplainer(estimator, matrix)
            method = "LinearSHAP"
        else:
            explainer = shap.TreeExplainer(estimator)
            method = "TreeSHAP"
        values = shap_array(explainer, matrix)
        if values.ndim != 2 or values.shape[1] != len(transformed_names):
            raise ValueError(f"Unexpected SHAP shape for {model_name}: {values.shape}")
        raw_importance = np.abs(values).mean(axis=0)

        grouped: dict[str, float] = {}
        for name, importance in zip(transformed_names, raw_importance):
            base = str(name).replace("missingindicator_", "")
            grouped[base] = grouped.get(base, 0.0) + float(importance)
        total = sum(grouped.values()) or 1.0
        rows = []
        for feature, importance in grouped.items():
            label, domain, description = FEATURE_INFO.get(
                feature,
                (feature.replace("_", " ").title(), "Model Features", "Model input feature."),
            )
            rows.append({
                "model": model_name,
                "method": method,
                "feature": feature,
                "label": label,
                "domain": domain,
                "description": description,
                "importance_abs": importance,
                "importance_pct": 100.0 * importance / total,
            })
        frame = pd.DataFrame(rows).sort_values("importance_pct", ascending=False)
        frame.to_csv(output_dir / f"public_samhi3_shap_{model_name.lower()}.csv", index=False)
        all_rows.extend(frame.to_dict(orient="records"))

    pd.DataFrame(all_rows).to_csv(output_dir / "public_samhi3_shap_summary.csv", index=False)
    print(f"Wrote Public SAMHI-3 SHAP impacts for {len(tournament.make_models())} models to {output_dir}")


if __name__ == "__main__":
    main()

"""Data-cleaning helpers shared by the Flask and Streamlit applications."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def resolve_base_dir(script_file: Path) -> Path:
    """Find the repository directory containing the datasets folder."""
    script_dir = script_file.resolve().parent
    candidates = [script_dir, *script_dir.parents]
    for candidate in candidates:
        if (candidate / "datasets").exists():
            return candidate
    searched = "\n".join(str(candidate / "datasets") for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate the datasets directory. Searched:\n"
        f"{searched}"
    )


def normalize_key(text: str) -> str:
    if text is None:
        return ""
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def normalize_postcode(value: object) -> str:
    return "".join(str(value or "").upper().split())


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def optional_numeric(df: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name:
        return parse_numeric(df[column_name])
    return pd.Series(np.nan, index=df.index)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def minmax_scale(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    min_v = valid.min()
    max_v = valid.max()
    if np.isclose(max_v, min_v):
        return pd.Series(0.5, index=series.index)
    return (series - min_v) / (max_v - min_v)


def normalize_weight_values(
    weight_values: dict[str, float],
    weight_keys: list[str],
) -> dict[str, float]:
    """Clamp and normalize a named group of weights without overflowing.

    A group with no positive weight falls back to an equal distribution. Very
    large finite values are scaled before summing so callers can safely accept
    the full range of valid floating-point request values.
    """
    values = np.array(
        [float(weight_values.get(key, 0.0)) for key in weight_keys],
        dtype=float,
    )
    values = np.clip(values, a_min=0.0, a_max=None)
    if values.max() > np.finfo(float).max / len(values):
        values = values / values.max()

    total = float(values.sum())
    if np.isclose(total, 0.0):
        equal_weight = 1.0 / len(weight_keys)
        return {key: equal_weight for key in weight_keys}

    return {
        key: float(value)
        for key, value in zip(weight_keys, values / total)
    }


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    columns = list(df.columns)
    normalized_map = {normalize_key(c): c for c in columns}

    for candidate in candidates:
        if candidate in columns:
            return candidate
        normalized_candidate = normalize_key(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    for candidate in candidates:
        normalized_candidate = normalize_key(candidate)
        partial_matches = [
            column
            for column in columns
            if normalized_candidate in normalize_key(column)
        ]
        if partial_matches:
            return partial_matches[0]

    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}")
    return None


def find_prefix_column(df: pd.DataFrame, prefix: str, required: bool = True) -> str | None:
    prefix_norm = normalize_key(prefix)
    for column in df.columns:
        if normalize_key(column).startswith(prefix_norm):
            return column
    if required:
        raise ValueError(f"Missing required column with prefix: {prefix}")
    return None

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import minmax_scale


WEIGHT_KEYS = ["dep_weight", "smi_weight", "prescribing_weight", "samhi_weight", "dla_weight", "pip_weight"]


def normalize_weights(weight_values: dict[str, float]) -> dict[str, float]:
    values = np.array([float(weight_values.get(k, 0.0)) for k in WEIGHT_KEYS], dtype=float)
    values = np.clip(values, a_min=0.0, a_max=None)
    total = float(values.sum())
    if np.isclose(total, 0.0):
        equal = 1.0 / len(WEIGHT_KEYS)
        return {k: equal for k in WEIGHT_KEYS}
    values = values / total
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, values)}


def apply_need_index(
    lsoa_df: pd.DataFrame,
    dep_weight: float,
    smi_weight: float,
    prescribing_weight: float,
    samhi_weight: float,
    samhi_column: str,
    dla_weight: float = 0.0,
    pip_weight: float = 0.0,
) -> pd.DataFrame:
    normalized_weights = normalize_weights(
        {
            "dep_weight": dep_weight,
            "smi_weight": smi_weight,
            "prescribing_weight": prescribing_weight,
            "samhi_weight": samhi_weight,
            "dla_weight": dla_weight,
            "pip_weight": pip_weight,
        }
    )

    out = lsoa_df.copy()
    out["SAMHI_Selected"] = pd.to_numeric(out.get(samhi_column), errors="coerce")
    out["Depression_Normalized"] = minmax_scale(out["Depression_Prevalence"])
    out["SMI_Normalized"] = minmax_scale(out["SMI_Prevalence"])
    out["Prescribing_Normalized"] = minmax_scale(out["Antidepressant_Items_Per_Patient"])
    out["SAMHI_Normalized"] = minmax_scale(out["SAMHI_Selected"])
    out["DLA_Normalized"] = minmax_scale(out.get("dwp_dla_rate_pct", pd.Series(np.nan, index=out.index)))
    out["PIP_Normalized"] = minmax_scale(out.get("dwp_pip_rate_pct", pd.Series(np.nan, index=out.index)))

    components = [
        ("Depression_Normalized", "dep_weight", out["Depression_Prevalence"].notna()),
        ("SMI_Normalized", "smi_weight", out["SMI_Prevalence"].notna()),
        ("Prescribing_Normalized", "prescribing_weight", out["Antidepressant_Items_Per_Patient"].notna()),
        ("SAMHI_Normalized", "samhi_weight", out["SAMHI_Selected"].notna()),
        ("DLA_Normalized", "dla_weight", pd.to_numeric(out.get("dwp_dla_rate_pct", pd.Series(np.nan, index=out.index)), errors="coerce").notna()),
        ("PIP_Normalized", "pip_weight", pd.to_numeric(out.get("dwp_pip_rate_pct", pd.Series(np.nan, index=out.index)), errors="coerce").notna()),
    ]
    weighted_sum = pd.Series(0.0, index=out.index)
    available_weight = pd.Series(0.0, index=out.index)
    for value_column, weight_key, available in components:
        weighted_sum = weighted_sum + out[value_column].fillna(0.0) * normalized_weights[weight_key] * available.astype(float)
        available_weight = available_weight + normalized_weights[weight_key] * available.astype(float)
    out["Need_Index"] = weighted_sum / available_weight.replace(0.0, np.nan)
    return out

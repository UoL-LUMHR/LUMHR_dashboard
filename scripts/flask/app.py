from __future__ import annotations

import csv
import importlib.util
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

from core.allocation import (
    allocate_access_rates_to_lsoa,
    allocate_registers_to_lsoa,
    build_gp_master,
    build_gp_marker_df,
    prepare_depression,
    prepare_prescribing,
    prepare_smi,
)
from core.need_index import apply_need_index
from core.access_index import apply_access_index
from core.rural_risk_index import apply_rural_risk_index
from core.housing import prepare_bedroom_occupancy, prepare_housing_tenure
from core.travel import (
    build_2011_to_2021_lookup,
    prepare_car_availability,
    prepare_digital_exclusion,
    prepare_imd_2025,
    prepare_lsoac_classification,
    prepare_rural_urban,
    prepare_travel_times,
    prepare_ts003_household_composition,
)
from core.common import find_column, normalize_code, parse_numeric
from core.samhi import SAMHI_YEARS, get_samhi_columns
from data_loader import get_prepared_bundle_cached, resolve_base_dir

app = Flask(__name__, template_folder="templates")

BASE_DIR = resolve_base_dir(Path(__file__))
DATASETS_DIR = BASE_DIR / "datasets"
GEOJSON_REL_PATH = "lsoa/lower-super-output-areas-2021-5RrVTw.geojson"
ENGLAND_GEOJSON_REL_PATH = "lsoa/england_lsoa_2021.geojson"
PUBLIC_SAMHI3_GEOJSON_PATHS = {
    "lincolnshire": GEOJSON_REL_PATH,
    "north_lincolnshire": "lsoa/north_lincolnshire_lsoa_2021.geojson",
    "north_east_lincolnshire": "lsoa/north_east_lincolnshire_lsoa_2021.geojson",
    "greater_lincolnshire": "lsoa/greater_lincolnshire_lsoa_2021.geojson",
    "nottinghamshire": "lsoa/nottinghamshire_lsoa_2021.geojson",
    "derbyshire": "lsoa/derbyshire_lsoa_2021.geojson",
    "lincolnshire_nottinghamshire_derbyshire": "lsoa/lincolnshire_nottinghamshire_derbyshire_lsoa_2021.geojson",
    "national": ENGLAND_GEOJSON_REL_PATH,
}
NEED_INDEX_GEOJSON_PATHS = {
    "lincolnshire": GEOJSON_REL_PATH,
    "north_lincolnshire": "lsoa/north_lincolnshire_lsoa_2021.geojson",
    "north_east_lincolnshire": "lsoa/north_east_lincolnshire_lsoa_2021.geojson",
    "greater_lincolnshire": "lsoa/greater_lincolnshire_lsoa_2021.geojson",
    "nottinghamshire": "lsoa/nottinghamshire_lsoa_2021.geojson",
    "derbyshire": "lsoa/derbyshire_lsoa_2021.geojson",
    "lincolnshire_nottinghamshire_derbyshire": "lsoa/lincolnshire_nottinghamshire_derbyshire_lsoa_2021.geojson",
}
AREA_COMPONENT_SCOPES = {
    "lincolnshire": ["lincolnshire"],
    "north_lincolnshire": ["north_lincolnshire"],
    "north_east_lincolnshire": ["north_east_lincolnshire"],
    "greater_lincolnshire": ["lincolnshire", "north_lincolnshire", "north_east_lincolnshire"],
    "nottinghamshire": ["nottinghamshire"],
    "derbyshire": ["derbyshire"],
    "lincolnshire_nottinghamshire_derbyshire": ["lincolnshire", "nottinghamshire", "derbyshire"],
}
SPECIAL_ALLOCATION_PRACTICE_CODES = {"C84720"}  # NOTSPAR remains a marker, not an area-level contributor.
SAMHI_FORECAST_DATA_DIR = DATASETS_DIR / "samhi_forecast"
PUBLIC_SAMHI3_DATA_DIR = DATASETS_DIR / "public_samhi3"


def _dashboard_data_path(directory: Path, filename: str) -> Path:
    """Return a tracked dashboard data file, allowing GitHub-safe gzip storage."""
    path = directory / filename
    compressed_path = Path(f"{path}.gz")
    if path.exists():
        return path
    if compressed_path.exists():
        return compressed_path
    return path

# Load once at startup so API requests only do lightweight transforms.
BUNDLE = get_prepared_bundle_cached(str(BASE_DIR))
LSOA_METRICS = BUNDLE["lsoa_metrics"].copy()
GP_MARKER_DF = BUNDLE["gp_marker_df"].copy()

# Public DWP snapshots are read from the copied raw exports in datasets/.  The
# reconstructed SAMHI panel supplies denominator/vintage metadata; the raw
# files are also needed to retain split LSOA21 rows and partial disclosures.
# These are deliberately exposed as separate map layers: they are not official
# SAMHI components, but are available as separately weighted inputs to the
# current-data Need, Rural Risk, and Access Gap calculations.
DWP_AVAILABLE_YEARS = (2023, 2024, 2025)


def _read_public_dwp_august_counts(base_dir: Path) -> pd.DataFrame:
    """Read 2023–2025 DLA/PIP August counts using LSOA21 names.

    The reconstructed panel stores the older LSOA11 key.  Reading the public
    Stat-Xplore exports here lets us retain both children when an LSOA11 was
    split during the 2021 geography revision.
    """
    boundary_path = base_dir / "datasets" / "lsoa" / "england_lsoa_2021.geojson"
    filenames = {
        "dwp_dla_count": "DLA_claimants_mental_health_learning_difficulties_from_may_2018_lsoa.csv",
        "dwp_pip_count": "PIP_cases_with_entitlement_mental_health_learning_difficulties_from_2019_lsoa.csv",
    }
    source_dir = base_dir / "datasets" / "DWP_DLA_PIP_data"
    if not all((source_dir / filename).exists() for filename in filenames.values()) or not boundary_path.exists():
        return pd.DataFrame(columns=["LSOA_CODE", "year", "dwp_dla_count", "dwp_pip_count"])
    files = {name: source_dir / filename for name, filename in filenames.items()}

    with boundary_path.open(encoding="utf-8") as handle:
        geojson = json.load(handle)
    name_to_code = {
        str(feature.get("properties", {}).get("LSOA21NM", "")).strip(): str(feature.get("properties", {}).get("LSOA21CD", "")).strip()
        for feature in geojson.get("features", [])
        if str(feature.get("properties", {}).get("LSOA21CD", "")).strip()
        and str(feature.get("properties", {}).get("LSOA21NM", "")).strip()
    }

    def read_component(path: Path, value_name: str) -> pd.DataFrame:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        header_idx = next((idx for idx, row in enumerate(rows) if row and row[0] in {"Month", "Quarter"}), None)
        if header_idx is None:
            return pd.DataFrame(columns=["LSOA_CODE", "year", value_name])
        header = rows[header_idx]
        period_columns = []
        for idx, label in enumerate(header[1:], start=1):
            label = str(label).strip()
            if not label.startswith("Aug-"):
                continue
            try:
                year = int(pd.to_datetime(label, format="%b-%y").year)
            except (TypeError, ValueError):
                continue
            if year in DWP_AVAILABLE_YEARS:
                period_columns.append((idx, year))
        records = []
        for row in rows[header_idx + 1:]:
            if not row:
                continue
            code = name_to_code.get(str(row[0]).strip())
            if not code:
                continue
            for idx, year in period_columns:
                value = row[idx] if idx < len(row) else ""
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric):
                    records.append((code, year, float(numeric)))
        return pd.DataFrame(records, columns=["LSOA_CODE", "year", value_name])

    dla = read_component(files["dwp_dla_count"], "dwp_dla_count")
    pip = read_component(files["dwp_pip_count"], "dwp_pip_count")
    if dla.empty and pip.empty:
        return pd.DataFrame(columns=["LSOA_CODE", "year", "dwp_dla_count", "dwp_pip_count"])
    return dla.merge(pip, on=["LSOA_CODE", "year"], how="outer")


@lru_cache(maxsize=1)
def _load_public_dwp_pip_suppression(base_dir_str: str) -> pd.DataFrame:
    """Identify DLA/PIP counts explicitly suppressed in DWP's August exports."""
    base_dir = Path(base_dir_str)
    boundary_path = base_dir / "datasets" / "lsoa" / "england_lsoa_2021.geojson"
    source_files = {
        "dwp_dla_source_status": "DLA_claimants_mental_health_learning_difficulties_from_may_2018_lsoa.csv",
        "dwp_pip_source_status": "PIP_cases_with_entitlement_mental_health_learning_difficulties_from_2019_lsoa.csv",
    }
    output_columns = ["LSOA_CODE", "year", *source_files]
    source_dir = base_dir / "datasets" / "DWP_DLA_PIP_data"
    if not boundary_path.exists() or not all((source_dir / name).exists() for name in source_files.values()):
        return pd.DataFrame(columns=output_columns)

    with boundary_path.open(encoding="utf-8") as handle:
        geojson = json.load(handle)
    name_to_code = {
        str(feature.get("properties", {}).get("LSOA21NM", "")).strip(): str(
            feature.get("properties", {}).get("LSOA21CD", "")
        ).strip()
        for feature in geojson.get("features", [])
        if str(feature.get("properties", {}).get("LSOA21NM", "")).strip()
        and str(feature.get("properties", {}).get("LSOA21CD", "")).strip()
    }

    suppressed = []
    for status_column, filename in source_files.items():
        with (source_dir / filename).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        header_idx = next(
            (idx for idx, row in enumerate(rows) if row and row[0] in {"Month", "Quarter"}),
            None,
        )
        if header_idx is None:
            continue
        periods = {}
        for idx, label in enumerate(rows[header_idx][1:], start=1):
            label = str(label).strip()
            if not label.startswith("Aug-"):
                continue
            try:
                year = int(pd.to_datetime(label, format="%b-%y").year)
            except (TypeError, ValueError):
                continue
            if year in DWP_AVAILABLE_YEARS:
                periods[idx] = year

        for row in rows[header_idx + 1:]:
            if not row:
                continue
            code = name_to_code.get(str(row[0]).strip())
            if not code:
                continue
            for idx, year in periods.items():
                if idx < len(row) and row[idx].strip() == "..":
                    suppressed.append((code, year, status_column, "Suppressed in DWP source"))

    if not suppressed:
        return pd.DataFrame(columns=output_columns)
    statuses = pd.DataFrame(
        suppressed,
        columns=["LSOA_CODE", "year", "status_column", "status"],
    ).drop_duplicates(["LSOA_CODE", "year", "status_column"])
    statuses = statuses.pivot(
        index=["LSOA_CODE", "year"],
        columns="status_column",
        values="status",
    ).reset_index()
    statuses.columns.name = None
    for column in source_files:
        if column not in statuses:
            statuses[column] = pd.NA
    return statuses[output_columns]


@lru_cache(maxsize=1)
def _load_public_dwp_panel(base_dir: str) -> pd.DataFrame:
    """Load public DWP DLA/PIP rates and convert LSOA11 to LSOA21 codes.

    The source panel is intentionally loaded lazily on the first map request;
    importing the dashboard therefore does not pay the cost of reading the
    100+ MB national panel when the DWP overlay is not used.
    """
    panel_path = _dashboard_data_path(Path(base_dir) / "datasets" / "samhi_forecast", "three_component_panel.csv")
    lookup_path = (
        Path(base_dir)
        / "datasets"
        / "lsoa"
        / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    cols = [
        "lsoa11",
        "year",
        "dwp_dla_pip_count",
        "dwp_dla_pip_rate_pct",
        "dwp_dla_count",
        "dwp_pip_count",
        "dwp_dla_rate_pct",
        "dwp_pip_rate_pct",
        "dwp_population_source_year",
        "dla_pip_source",
    ]
    if not panel_path.exists() or not lookup_path.exists():
        return pd.DataFrame(columns=["LSOA_CODE", *cols[1:], "dwp_dla_rate_pct", "dwp_pip_rate_pct"])

    panel = pd.read_csv(panel_path, usecols=cols, low_memory=False)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel[panel["year"].isin(DWP_AVAILABLE_YEARS)].copy()
    panel["lsoa11"] = panel["lsoa11"].astype(str).str.strip()
    parent_denominators = panel[["lsoa11", "year", "dwp_dla_pip_count", "dwp_dla_pip_rate_pct"]].copy()
    parent_denominators["parent_denominator"] = (
        pd.to_numeric(parent_denominators["dwp_dla_pip_count"], errors="coerce")
        / pd.to_numeric(parent_denominators["dwp_dla_pip_rate_pct"], errors="coerce").replace(0.0, pd.NA)
        * 100.0
    )

    lookup = pd.read_csv(
        lookup_path,
        usecols=["LSOA11CD", "LSOA21CD"],
        dtype=str,
    ).rename(columns={"LSOA11CD": "lsoa11", "LSOA21CD": "LSOA_CODE"})
    lookup["lsoa11"] = lookup["lsoa11"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    lookup["LSOA_CODE"] = lookup["LSOA_CODE"].astype(str).str.strip()

    # Resolve one-to-many 2011-to-2021 splits by matching each panel count pair
    # to the raw DWP LSOA21 exports. A simple ``drop_duplicates(lsoa11)``
    # incorrectly loses the second child (for example Lincoln 007F).
    panel = panel.reset_index(drop=True)
    panel["_panel_row"] = panel.index
    panel_for_match = panel.rename(
        columns={"dwp_dla_count": "panel_dla_count", "dwp_pip_count": "panel_pip_count"}
    )
    raw_counts = _read_public_dwp_august_counts(Path(base_dir))
    fallback_population = _load_dwp_fallback_population(base_dir)
    if not raw_counts.empty:
        candidates = panel_for_match.merge(lookup, on="lsoa11", how="inner")
        matched = candidates.merge(raw_counts, on=["LSOA_CODE", "year"], how="inner")
        matched = matched[
            pd.to_numeric(matched["panel_dla_count"], errors="coerce").eq(
                pd.to_numeric(matched["dwp_dla_count"], errors="coerce")
            )
            & pd.to_numeric(matched["panel_pip_count"], errors="coerce").eq(
                pd.to_numeric(matched["dwp_pip_count"], errors="coerce")
            )
        ]
        resolved = matched[["_panel_row", "LSOA_CODE"]].drop_duplicates("_panel_row")
    else:
        resolved = pd.DataFrame(columns=["_panel_row", "LSOA_CODE"])

    panel = panel.merge(resolved.rename(columns={"LSOA_CODE": "_resolved_code"}), on="_panel_row", how="left")
    single_lookup = (
        lookup.groupby("lsoa11", as_index=False)["LSOA_CODE"]
        .agg(lambda values: values.iloc[0] if values.nunique() == 1 else pd.NA)
        .rename(columns={"LSOA_CODE": "_fallback_code"})
    )
    panel = panel.merge(single_lookup, on="lsoa11", how="left")
    panel["LSOA_CODE"] = panel["_resolved_code"].combine_first(panel["_fallback_code"])
    panel["LSOA_CODE"] = panel["LSOA_CODE"].replace({"nan": pd.NA, "": pd.NA})
    panel = panel.dropna(subset=["LSOA_CODE"])
    # Some 2021 LSOAs merge two 2011 LSOAs. When a panel row matches the
    # published 2021 DWP counts, prefer that resolved row over the other
    # parent's fallback mapping instead of arbitrarily keeping the last row.
    panel["_resolved_priority"] = panel["_resolved_code"].notna().astype("int8")
    panel["_component_rate_coverage"] = panel[["dwp_dla_rate_pct", "dwp_pip_rate_pct"]].notna().sum(axis=1)
    panel = panel.sort_values(
        ["year", "LSOA_CODE", "_resolved_priority", "_component_rate_coverage"],
        kind="stable",
    ).drop_duplicates(subset=["year", "LSOA_CODE"], keep="last")
    # The panel publishes the combined DLA+PIP rate and separate claimant
    # counts. Derive component rates against the same denominator so the two
    # indicators remain directly comparable with the combined layer.
    combined_count = pd.to_numeric(panel["dwp_dla_pip_count"], errors="coerce")
    combined_rate = pd.to_numeric(panel["dwp_dla_pip_rate_pct"], errors="coerce")
    denominator = (combined_count / combined_rate.replace(0.0, pd.NA)) * 100.0
    # Keep the component rates published in the panel. Re-deriving both from
    # the combined DLA+PIP denominator erased an available DLA rate whenever
    # the PIP count (and therefore the combined rate) was disclosure-suppressed.
    panel["dwp_dla_rate_pct"] = pd.to_numeric(panel["dwp_dla_rate_pct"], errors="coerce").combine_first(
        pd.to_numeric(panel["dwp_dla_count"], errors="coerce") / denominator * 100.0
    )
    panel["dwp_pip_rate_pct"] = pd.to_numeric(panel["dwp_pip_rate_pct"], errors="coerce").combine_first(
        pd.to_numeric(panel["dwp_pip_count"], errors="coerce") / denominator * 100.0
    )
    output = panel[["LSOA_CODE", "year", "dwp_dla_pip_count", "dwp_dla_pip_rate_pct", "dwp_dla_count", "dwp_pip_count", "dwp_dla_rate_pct", "dwp_pip_rate_pct", "dwp_population_source_year", "dla_pip_source"]].copy()

    # The map uses 2021 LSOA geometries, while the reconstructed panel is now
    # aggregated to 2011 SAMHI geographies. For one-to-many 2011→2021 splits,
    # remove any parent-derived rows and rebuild each child from the raw DWP
    # count, allocating the parent's panel denominator by each child's current
    # working-age population share. This prevents a split child from being
    # dropped or duplicated after the lookup join.
    if not raw_counts.empty:
        split_lsoa11 = (
            lookup.groupby("lsoa11")["LSOA_CODE"]
            .nunique()
            .loc[lambda values: values.gt(1)]
            .index
        )
        split_lookup = lookup[lookup["lsoa11"].isin(split_lsoa11)].copy()
        allocated = pd.DataFrame()
        if not split_lookup.empty:
            output = output[~output["LSOA_CODE"].isin(split_lookup["LSOA_CODE"])]
            allocated = raw_counts.merge(split_lookup, on="LSOA_CODE", how="inner")
            allocated = allocated.merge(
                parent_denominators[["lsoa11", "year", "parent_denominator"]],
                on=["lsoa11", "year"],
                how="left",
            )
            child_population = fallback_population
            allocated = allocated.merge(child_population, on="LSOA_CODE", how="left")
            allocated["parent_child_population"] = allocated.groupby(["lsoa11", "year"])["ONS_Pop_18to64"].transform("sum")
            complete_split = (
                allocated["parent_denominator"].notna()
                & allocated["dwp_dla_count"].notna()
                & allocated["dwp_pip_count"].notna()
            )
            allocated["dwp_denominator"] = (
                allocated["parent_denominator"].where(complete_split)
                * allocated["ONS_Pop_18to64"]
                / allocated["parent_child_population"].replace(0.0, pd.NA)
            )
            allocated["dwp_denominator"] = allocated["dwp_denominator"].fillna(
                pd.to_numeric(allocated["ONS_Pop_18to64"], errors="coerce")
            )
            allocated["dwp_dla_pip_count"] = allocated[["dwp_dla_count", "dwp_pip_count"]].sum(axis=1, min_count=1)
            denominator = allocated["dwp_denominator"].replace(0.0, pd.NA)
            allocated["dwp_dla_pip_rate_pct"] = allocated["dwp_dla_pip_count"] / denominator * 100.0
            allocated["dwp_dla_rate_pct"] = allocated["dwp_dla_count"] / denominator * 100.0
            allocated["dwp_pip_rate_pct"] = allocated["dwp_pip_count"] / denominator * 100.0
            allocated["dwp_population_source_year"] = 2024.0
            allocated["dla_pip_source"] = complete_split.map(
                {True: "DWP_August_split_parent_denominator", False: "DWP_August_split_18to64_fallback"}
            )

        existing_keys = output[["LSOA_CODE", "year"]].assign(_present=True)
        partial = raw_counts.merge(existing_keys, on=["LSOA_CODE", "year"], how="left")
        partial = partial[partial["_present"].isna()].drop(columns=["_present"], errors="ignore")
        if not allocated.empty:
            allocated_keys = allocated[["LSOA_CODE", "year"]].assign(_allocated=True)
            partial = partial.merge(allocated_keys, on=["LSOA_CODE", "year"], how="left")
            partial = partial[partial["_allocated"].isna()].drop(columns=["_allocated"], errors="ignore")
        partial = partial.merge(fallback_population, on="LSOA_CODE", how="left")
        partial["dwp_population_source_year"] = 2024.0
        partial["dwp_dla_pip_count"] = partial[["dwp_dla_count", "dwp_pip_count"]].sum(axis=1, min_count=1)
        denominator = pd.to_numeric(partial["ONS_Pop_18to64"], errors="coerce")
        partial["dwp_dla_pip_rate_pct"] = partial["dwp_dla_pip_count"] / denominator.replace(0.0, pd.NA) * 100.0
        partial["dwp_dla_rate_pct"] = partial["dwp_dla_count"] / denominator.replace(0.0, pd.NA) * 100.0
        partial["dwp_pip_rate_pct"] = partial["dwp_pip_count"] / denominator.replace(0.0, pd.NA) * 100.0
        partial["dla_pip_source"] = "DWP_August_partial_18to64_fallback"
        append_columns = ["LSOA_CODE", "year", "dwp_dla_pip_count", "dwp_dla_pip_rate_pct", "dwp_dla_count", "dwp_pip_count", "dwp_dla_rate_pct", "dwp_pip_rate_pct", "dwp_population_source_year", "dla_pip_source"]
        frames = [output]
        if not allocated.empty:
            frames.append(allocated[append_columns])
        frames.append(partial[append_columns])
        output = pd.concat(frames, ignore_index=True)
    return output.drop_duplicates(["LSOA_CODE", "year"], keep="last")


def _attach_public_dwp_layer(scored: pd.DataFrame, year: int) -> pd.DataFrame:
    dwp = _load_public_dwp_panel(str(BASE_DIR))
    dwp = dwp[dwp["year"] == year].copy()
    dwp = dwp.drop(columns=["year"], errors="ignore")
    pip_status = _load_public_dwp_pip_suppression(str(BASE_DIR))
    pip_status = pip_status[pip_status["year"].eq(year)].drop(columns=["year"], errors="ignore")
    dwp = dwp.merge(pip_status, on="LSOA_CODE", how="left")
    return scored.merge(dwp, on="LSOA_CODE", how="left")


@lru_cache(maxsize=1)
def _load_dwp_fallback_population(base_dir: str) -> pd.DataFrame:
    """Load 2024 18–64 populations for the supported LSOA21 map geographies."""
    directory = Path(base_dir) / "datasets" / "population_estimates"
    paths = sorted(directory.glob("*_lsoa_population_estimates_2024.csv"))
    columns = ["LSOA_CODE", "ONS_Pop_18to64"]
    frames = [pd.read_csv(path, usecols=columns, dtype={"LSOA_CODE": str}) for path in paths]
    if not frames:
        return pd.DataFrame(columns=columns)
    population = pd.concat(frames, ignore_index=True)
    population["LSOA_CODE"] = population["LSOA_CODE"].map(normalize_code)
    population["ONS_Pop_18to64"] = pd.to_numeric(population["ONS_Pop_18to64"], errors="coerce")
    return population.drop_duplicates("LSOA_CODE", keep="first")


def _parse_float_arg(name: str, default: float) -> float:
    raw = request.args.get(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{name}': {raw}") from exc


def _parse_int_arg(name: str, default: int) -> int:
    raw = request.args.get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for '{name}': {raw}") from exc


def _scores_to_dict(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    valid = df[["LSOA_CODE", score_col]].dropna(subset=["LSOA_CODE", score_col]).copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid.dropna(subset=[score_col])
    return {str(code): float(value) for code, value in zip(valid["LSOA_CODE"], valid[score_col])}


def _num_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _text_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _json_safe(value: object) -> object:
    """Convert pandas/numpy missing values to JSON ``null`` recursively."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    # pandas/numpy scalar values expose item(), which gives Flask a native type.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return value


def _get_lsoa_name_column(df: pd.DataFrame) -> str | None:
    candidates = ["LSOA21NM", "LSOA21NM_x", "LSOA21NM_y", "LSOA_NAME", "NAME"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


@app.get("/")
def home_page() -> str:
    return render_template("home.html")


@app.get("/need_index")
def need_index_page() -> str:
    return render_template(
        "need_index.html",
        geojson_by_scope=NEED_INDEX_GEOJSON_PATHS,
    )


@app.get("/samhi")
def samhi_page() -> str:
    return render_template(
        "samhi.html",
        geojson_by_scope=PUBLIC_SAMHI3_GEOJSON_PATHS,
        min_year=min(SAMHI_YEARS),
        max_year=max(SAMHI_YEARS),
        default_year=max(SAMHI_YEARS),
    )


@app.get("/access_index")
def access_index_page() -> str:
    return render_template(
        "access_index.html",
        geojson_by_scope=NEED_INDEX_GEOJSON_PATHS,
    )


@app.get("/access_gap_index")
def access_gap_index_page() -> str:
    return render_template(
        "access_gap_index.html",
        geojson_by_scope=NEED_INDEX_GEOJSON_PATHS,
        min_year=min(SAMHI_YEARS),
        max_year=max(SAMHI_YEARS),
    )


@app.get("/rural_risk_index")
def rural_risk_index_page() -> str:
    return render_template(
        "rural_risk_index.html",
        geojson_by_scope=NEED_INDEX_GEOJSON_PATHS,
    )


@app.get("/datasets/<path:filename>")
def dataset_files(filename: str):
    return send_from_directory(DATASETS_DIR, filename)


@app.get("/api/gp_locations")
def gp_locations_api():
    scope = str(request.args.get("scope", "lincolnshire") or "lincolnshire").strip().lower()
    if scope not in NEED_INDEX_GEOJSON_PATHS:
        scope = "lincolnshire"
    marker_df = GP_MARKER_DF if scope == "lincolnshire" else _get_current_area_need_base(scope)["markers"]
    cols = [
        "PRACTICE_CODE",
        "Practice_Name",
        "Lat",
        "Lon",
        "NUMBER_OF_PATIENTS",
        "Dep_Register",
        "SMI_Register",
        "Dep_Prevalence_Pct",
        "SMI_Prevalence_Pct",
        "Antidepressant_Items",
        "Antidepressant_Actual_Cost",
        "MH002_Pct",
        "Physical_Health_Review_Avg_Pct",
        "Exception_Rate_Pct",
        "MH021_Pct",
        "Dep_Exception_Rate_Pct",
        "SMI_Exception_Rate_Pct",
        "DEP004_Pct",
        "Effective_LSOA",
    ]
    available_cols = [c for c in cols if c in marker_df.columns]
    out = marker_df[available_cols].copy()
    out["Lat"] = pd.to_numeric(out.get("Lat"), errors="coerce")
    out["Lon"] = pd.to_numeric(out.get("Lon"), errors="coerce")
    out = out.dropna(subset=["Lat", "Lon"]).drop_duplicates("PRACTICE_CODE", keep="first")

    markers: list[dict[str, object]] = []
    for row in out.to_dict(orient="records"):
        marker = {
            "practice_code": str(row.get("PRACTICE_CODE", "")),
            "practice_name": str(row.get("Practice_Name", "") or ""),
            "map_data_note": (
                "NOTSPAR is a Special Allocation Scheme practice. Its location and available practice data are shown here; its registrations and practice-level indicators are excluded from LSOA-wide map calculations and GP registration totals."
                if str(row.get("PRACTICE_CODE", "")).strip() in SPECIAL_ALLOCATION_PRACTICE_CODES
                else None
            ),
            "lat": float(row["Lat"]),
            "lon": float(row["Lon"]),
            "patients": _num_or_none(row.get("NUMBER_OF_PATIENTS")),
            "dep_register": _num_or_none(row.get("Dep_Register")),
            "dep_prev_pct": _num_or_none(row.get("Dep_Prevalence_Pct")),
            "smi_register": _num_or_none(row.get("SMI_Register")),
            "smi_prev_pct": _num_or_none(row.get("SMI_Prevalence_Pct")),
            "antidepressant_items": _num_or_none(row.get("Antidepressant_Items")),
            "antidepressant_actual_cost": _num_or_none(row.get("Antidepressant_Actual_Cost")),
            "mh002_pct": _num_or_none(row.get("MH002_Pct")),
            "physical_health_review_avg_pct": _num_or_none(row.get("Physical_Health_Review_Avg_Pct")),
            "exception_rate_pct": _num_or_none(row.get("Exception_Rate_Pct")),
            "mh021_pct": _num_or_none(row.get("MH021_Pct")),
            "mh_pca_pct": _num_or_none(row.get("SMI_Exception_Rate_Pct")),
            "dep_pca_pct": _num_or_none(row.get("Dep_Exception_Rate_Pct")),
            "dep004_pct": _num_or_none(row.get("DEP004_Pct")),
            "effective_lsoa": str(row.get("Effective_LSOA", "") or ""),
        }
        markers.append(marker)

    return jsonify(markers)


@app.get("/api/need_scores")
def need_scores_api():
    try:
        dep = _parse_float_arg("dep", 16.7)
        smi = _parse_float_arg("smi", 16.7)
        prescribing = _parse_float_arg("prescribing", 16.7)
        samhi_weight = _parse_float_arg("samhi", 16.7)
        dla_weight = _parse_float_arg("dla", 16.7)
        pip_weight = _parse_float_arg("pip", 16.7)
        samhi_year = _parse_int_arg("samhi_year", max(SAMHI_YEARS))
        public_samhi3_year = _parse_int_arg("public_samhi3_year", 2025)
        dwp_year = _parse_int_arg("dwp_year", max(DWP_AVAILABLE_YEARS))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if samhi_year not in SAMHI_YEARS:
        return jsonify({"error": f"samhi_year must be one of {SAMHI_YEARS}"}), 400
    if public_samhi3_year < 2011 or public_samhi3_year > 2025:
        return jsonify({"error": "public_samhi3_year must be between 2011 and 2025"}), 400
    if dwp_year not in DWP_AVAILABLE_YEARS:
        return jsonify({"error": f"dwp_year must be one of {DWP_AVAILABLE_YEARS}"}), 400

    scope = str(request.args.get("scope", "lincolnshire") or "lincolnshire").strip().lower()
    if scope not in NEED_INDEX_GEOJSON_PATHS:
        scope = "lincolnshire"

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    if scope == "lincolnshire":
        if samhi_index_col not in LSOA_METRICS.columns:
            return jsonify({"error": f"Missing SAMHI column: {samhi_index_col}"}), 400
        scored = _attach_public_dwp_layer(LSOA_METRICS, dwp_year)
        out_of_area_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
        out_of_area_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    else:
        scored = _get_current_area_need_metrics(scope, dwp_year)
        area_data = _get_current_area_need_base(scope)
        out_of_area_lsoa_count = area_data["out_of_area_lsoa_count"]
        out_of_area_patients = area_data["out_of_area_patients"]

    out_of_area_note = (
        "GP registration data includes patients registered to practices in this area who live outside its map "
        f"boundary. Distinct out-of-area LSOAs: {out_of_area_lsoa_count:,}. "
        f"Patient records in those LSOAs: {out_of_area_patients:,}. "
        "They are retained in GP allocation weights but excluded from the map polygons."
    )

    public_samhi3 = _get_public_samhi3_df(scope)
    if {"year", "LSOA21CD", "public_samhi3_proxy", "public_samhi3_n_components"}.issubset(public_samhi3.columns):
        proxy_columns = ["LSOA21CD", "public_samhi3_proxy", "public_samhi3_n_components"]
        if "public_samhi3_proxy_2plus" in public_samhi3.columns:
            proxy_columns.append("public_samhi3_proxy_2plus")
        public_samhi3 = public_samhi3[
            public_samhi3["year"].eq(public_samhi3_year)
        ][proxy_columns].copy()
        primary_proxy = pd.to_numeric(public_samhi3["public_samhi3_proxy"], errors="coerce")
        fallback_proxy = pd.to_numeric(
            public_samhi3.get("public_samhi3_proxy_2plus", pd.Series(np.nan, index=public_samhi3.index)),
            errors="coerce",
        )
        public_samhi3["public_samhi3_partial"] = primary_proxy.isna() & fallback_proxy.notna()
        public_samhi3["public_samhi3_proxy"] = primary_proxy.combine_first(fallback_proxy)
    else:
        public_samhi3 = pd.DataFrame(
            columns=["LSOA21CD", "public_samhi3_proxy", "public_samhi3_n_components", "public_samhi3_partial"]
        )
    public_samhi3 = public_samhi3.rename(
        columns={
            "LSOA21CD": "LSOA_CODE",
            "public_samhi3_proxy": "Public_SAMHI3_Proxy",
            "public_samhi3_n_components": "Public_SAMHI3_Components",
            "public_samhi3_partial": "Public_SAMHI3_Partial",
        }
    )
    scored = scored.merge(public_samhi3, on="LSOA_CODE", how="left")
    scored = apply_need_index(
        scored,
        dep,
        smi,
        prescribing,
        samhi_weight,
        samhi_index_col,
        dla_weight,
        pip_weight,
    )
    scored["Depression_Prevalence_Pct"] = pd.to_numeric(scored["Depression_Prevalence"], errors="coerce") * 100.0
    scored["SMI_Prevalence_Pct"] = pd.to_numeric(scored["SMI_Prevalence"], errors="coerce") * 100.0

    layers = {
        "Need_Index": _scores_to_dict(scored, "Need_Index"),
        "Depression_Prevalence": _scores_to_dict(scored, "Depression_Prevalence"),
        "SMI_Prevalence": _scores_to_dict(scored, "SMI_Prevalence"),
        "Antidepressant_Items_Per_Patient": _scores_to_dict(scored, "Antidepressant_Items_Per_Patient"),
        "SAMHI_Selected": _scores_to_dict(scored, "SAMHI_Selected"),
        "Public_SAMHI3_Proxy": _scores_to_dict(scored, "Public_SAMHI3_Proxy"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
        "DWP_DLA_Rate_Pct": _scores_to_dict(scored, "dwp_dla_rate_pct"),
        "DWP_PIP_Rate_Pct": _scores_to_dict(scored, "dwp_pip_rate_pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)

    detail_cols = [
        "LSOA_CODE",
        "Need_Index",
        "Depression_Prevalence_Pct",
        "Depression_Coverage_Pct",
        "SMI_Prevalence_Pct",
        "SMI_Coverage_Pct",
        "Antidepressant_Items_Per_Patient",
        "Antidepressant_Coverage_Pct",
        "SAMHI_Selected",
        "Public_SAMHI3_Proxy",
        "Public_SAMHI3_Components",
        "Public_SAMHI3_Partial",
        "RUC21NM",
        "Urban_rural_flag",
        "ONS_Pop_Total_2024",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "dwp_dla_rate_pct",
        "dwp_pip_rate_pct",
        "dwp_dla_count",
        "dwp_pip_count",
        "dwp_dla_source_status",
        "dwp_pip_source_status",
        "dwp_population_source_year",
        "dla_pip_source",
        "Registration_Gap_Est",
        "List_Inflation_Est",
        "Parkfields_Jan_2026_Patients",
        "Parkfields_July_2026_Patients",
    ]
    if lsoa_name_col:
        detail_cols.insert(1, lsoa_name_col)
    details_df = scored[[c for c in detail_cols if c in scored.columns]].copy()
    lsoa_details: dict[str, dict[str, object]] = {}
    for row in details_df.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        ruc_text = str(row.get("RUC21NM", "") or "")
        flag_text = str(row.get("Urban_rural_flag", "") or ("Rural" if "rural" in ruc_text.lower() else "Urban"))
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "need_index": _num_or_none(row.get("Need_Index")),
            "depression_prevalence_pct": _num_or_none(row.get("Depression_Prevalence_Pct")),
            "depression_coverage_pct": _num_or_none(row.get("Depression_Coverage_Pct")),
            "smi_prevalence_pct": _num_or_none(row.get("SMI_Prevalence_Pct")),
            "smi_coverage_pct": _num_or_none(row.get("SMI_Coverage_Pct")),
            "antidepressant_items_per_patient": _num_or_none(row.get("Antidepressant_Items_Per_Patient")),
            "antidepressant_coverage_pct": _num_or_none(row.get("Antidepressant_Coverage_Pct")),
            "samhi_selected": _num_or_none(row.get("SAMHI_Selected")),
            "public_samhi3_proxy": _num_or_none(row.get("Public_SAMHI3_Proxy")),
            "public_samhi3_components": _num_or_none(row.get("Public_SAMHI3_Components")),
            "public_samhi3_partial": (
                bool(row.get("Public_SAMHI3_Partial"))
                if pd.notna(row.get("Public_SAMHI3_Partial"))
                else False
            ),
            "ruc21nm": ruc_text,
            "urban_rural_flag": flag_text,
            "is_rural": "rural" in ruc_text.lower(),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
            "parkfields_jan_2026_patients": _num_or_none(row.get("Parkfields_Jan_2026_Patients")),
            "parkfields_july_2026_patients": _num_or_none(row.get("Parkfields_July_2026_Patients")),
            "dwp_dla_rate_pct": _num_or_none(row.get("dwp_dla_rate_pct")),
            "dwp_pip_rate_pct": _num_or_none(row.get("dwp_pip_rate_pct")),
            "dwp_dla_source_status": _text_or_none(row.get("dwp_dla_source_status")),
            "dwp_pip_source_status": _text_or_none(row.get("dwp_pip_source_status")),
            "dwp_dla_count": _num_or_none(row.get("dwp_dla_count")),
            "dwp_pip_count": _num_or_none(row.get("dwp_pip_count")),
            "dwp_population_source_year": _num_or_none(row.get("dwp_population_source_year")),
            "dwp_source": _text_or_none(row.get("dla_pip_source")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
            "meta": {
                "samhi_year": samhi_year,
                "samhi_label": f"SAMHI Index ({samhi_year})",
                "public_samhi3_year": public_samhi3_year,
                "public_samhi3_label": f"Public SAMHI-3 proxy ({public_samhi3_year})",
                "dwp_year": dwp_year,
                "dwp_label": f"DWP DLA and PIP rates ({dwp_year})",
                "dwp_component_labels": {"dla": f"DLA rate ({dwp_year})", "pip": f"PIP rate ({dwp_year})"},
                "dwp_population_vintage_note": "The DWP PIP rate is unavailable where the source suppresses its claimant count; the popup identifies those rows. Available component counts use the selected panel denominator where possible and a marked 2024 ONS 18–64 fallback for partial rows.",
                "out_of_area_note": out_of_area_note,
            },
        }
    )


_AREA_METRICS_CACHE: dict[str, pd.DataFrame] = {}


@lru_cache(maxsize=1)
def _prepare_national_deri() -> pd.DataFrame:
    """Calculate DERI across England, then expand its 2011 LSOAs to 2021."""
    source_path = DATASETS_DIR / "digital_exclusion_risk_index" / "DERI dataset_v1.6.csv"
    raw = pd.read_csv(source_path, low_memory=False)
    source_columns = {
        "Percentage of population aged 65 and over": "Aged_65_Plus_Pct",
        "Percentage of residents whose day-to-day activities are limited": "Day_To_Day_Limited_Pct",
        "Percentage of residents aged 16+ with no qualifications": "No_Qualifications_Pct",
        "Guaranteed pension credit (rate per 1,000 aged 65+)": "Pension_Credit_Rate",
        "Unemployment rate": "Unemployment_Rate",
        "Percentage of population in social grade DE": "Social_Grade_DE_Pct",
        "Index of Multiple Deprivation 2019 score - England base": "IMD_2019_Score",
        "Percentage of homes unable to receive at least 30Mbit/s broadband": "No_Superfast_Broadband_Pct",
        "Average download speed (Mbit/s)": "Avg_Download_Speed_Mbps",
        "Percentage of connections receiving less than 10Mbit/s broadband": "Slow_Connections_Pct",
    }
    numeric_columns = list(source_columns)
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw[column] = raw[column].fillna(raw[column].median())

    def score(values: pd.Series, invert: bool = False) -> pd.Series:
        low = values.min()
        high = values.max()
        if pd.isna(low) or pd.isna(high) or high == low:
            return pd.Series(0.0, index=values.index)
        normalized = (values - low) / (high - low)
        return (1.0 - normalized if invert else normalized) * 10.0

    age = "Percentage of population aged 65 and over"
    disability = "Percentage of residents whose day-to-day activities are limited"
    qualifications = "Percentage of residents aged 16+ with no qualifications"
    pension = "Guaranteed pension credit (rate per 1,000 aged 65+)"
    unemployment = "Unemployment rate"
    social_grade = "Percentage of population in social grade DE"
    imd = "Index of Multiple Deprivation 2019 score - England base"
    no_superfast = "Percentage of homes unable to receive at least 30Mbit/s broadband"
    speed = "Average download speed (Mbit/s)"
    slow_connections = "Percentage of connections receiving less than 10Mbit/s broadband"

    raw["Demography_Score"] = (score(raw[age]) + score(raw[disability])) / 2.0
    raw["Deprivation_Score"] = (
        score(raw[qualifications]) + score(raw[pension]) + score(raw[unemployment])
        + score(raw[social_grade]) + score(raw[imd])
    ) / 5.0
    raw["Broadband_Score"] = (
        score(raw[no_superfast]) + score(raw[speed], invert=True) + score(raw[slow_connections])
    ) / 3.0
    raw["DERI_Score"] = (raw["Demography_Score"] + raw["Deprivation_Score"] + raw["Broadband_Score"]) / 3.0
    raw["Digital_Access"] = 1.0 - raw["DERI_Score"] / 10.0
    percentage_outputs = {
        "Aged_65_Plus_Pct",
        "Day_To_Day_Limited_Pct",
        "No_Qualifications_Pct",
        "Social_Grade_DE_Pct",
        "No_Superfast_Broadband_Pct",
        "Slow_Connections_Pct",
        "Unemployment_Rate",
    }
    for source_column, output_column in source_columns.items():
        multiplier = 100.0 if output_column in percentage_outputs else 1.0
        raw[output_column] = raw[source_column] * multiplier

    lookup_path = (
        DATASETS_DIR / "lsoa" / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    lookup = pd.read_csv(lookup_path, usecols=["LSOA11CD", "LSOA21CD"], dtype=str)
    lookup["LSOA11CD"] = lookup["LSOA11CD"].astype(str).str.strip()
    lookup["LSOA21CD"] = lookup["LSOA21CD"].astype(str).str.strip()
    raw["LSOA code"] = raw["LSOA code"].astype(str).str.strip()
    value_columns = [*source_columns.values(), "DERI_Score", "Digital_Access"]
    expanded = lookup.merge(raw[["LSOA code", *value_columns]], left_on="LSOA11CD", right_on="LSOA code", how="left")
    expanded = expanded.groupby("LSOA21CD", as_index=False)[value_columns].mean()
    expanded = expanded.rename(columns={"LSOA21CD": "LSOA_Code"})
    return prepare_digital_exclusion(expanded)


@lru_cache(maxsize=1)
def _prepare_national_ts003() -> pd.DataFrame:
    """Parse the national Nomis TS003 download into the dashboard's prepared shape."""
    source_path = DATASETS_DIR / "TS003_household_composition" / "TS003 - Household composition.csv"
    rows: list[dict[str, object]] = []
    raw_headers: list[str] = []
    with source_path.open(encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.reader(source)):
            if row_number == 7:
                raw_headers = row
                continue
            if row_number < 9 or not row or not row[0].strip():
                continue
            geography = row[0].strip()
            code = geography.split(" : ", 1)[0].strip()
            record: dict[str, object] = {"LSOA_Code": code}
            for column_index, value in enumerate(row[1:], start=1):
                if column_index >= len(raw_headers):
                    break
                header = raw_headers[column_index].strip()
                if header == "%":
                    column_name = f"{raw_headers[column_index - 1].strip()}_Pct"
                elif header == "Total: All households":
                    column_name = "Total_All_Households"
                else:
                    column_name = f"{header}_Count"
                try:
                    number = float(value)
                    record[column_name] = int(number) if number.is_integer() and not column_name.endswith("_Pct") else number
                except ValueError:
                    record[column_name] = value
            rows.append(record)
    return prepare_ts003_household_composition(pd.DataFrame(rows))


@lru_cache(maxsize=1)
def _get_national_access_risk_components() -> dict[str, pd.DataFrame]:
    """Prepare England-wide indicators shared by the non-Lincolnshire maps."""
    journey_dir = DATASETS_DIR / "journey_time_statistics"
    lookup_path = (
        DATASETS_DIR
        / "lsoa"
        / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    lookup_map = build_2011_to_2021_lookup(pd.read_csv(lookup_path, dtype=str))
    travel = prepare_travel_times(
        pd.read_csv(journey_dir / "england" / "clean_gps_2019_england.csv"),
        pd.read_csv(journey_dir / "england" / "clean_hospitals_2019_england.csv"),
        lookup_map,
    )

    lsoac_source = BASE_DIR / "scripts" / "utils" / "source_data" / "lsoa_classification_2021_2"
    lsoac = pd.read_csv(lsoac_source / "lsoa_dz_sdz_lookup.csv", low_memory=False)
    code_names = pd.read_csv(lsoac_source / "classification_codes_and_names_ukoac_lsoac.csv")
    for column in ("geography_code", "Supergroup", "Group", "Subgroup"):
        lsoac[column] = lsoac[column].astype(str).str.strip()
    name_maps = {
        level: dict(zip(group["Classification Code"].astype(str).str.strip(), group["Classification Name"].astype(str).str.strip()))
        for level, group in code_names.groupby("Level")
    }
    lsoac = lsoac.rename(columns={"geography_code": "LSOA_Code"})
    lsoac["Supergroup_Code"] = lsoac["Supergroup"]
    lsoac["Group_Code"] = lsoac["Group"]
    lsoac["Subgroup_Code"] = lsoac["Subgroup"]
    lsoac["Supergroup_Name"] = lsoac["Supergroup_Code"].map(name_maps.get("Supergroup", {}))
    lsoac["Group_Name"] = lsoac["Group_Code"].map(name_maps.get("Group", {}))
    lsoac["Subgroup_Name"] = lsoac["Subgroup_Code"].map(name_maps.get("Subgroup", {}))
    lsoac = prepare_lsoac_classification(lsoac)

    return {
        "travel": travel,
        "rural_urban": prepare_rural_urban(pd.read_csv(
            DATASETS_DIR / "rural_urban_classification_2021_lsoa" / "Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv",
            low_memory=False,
        )),
        "car_van": prepare_car_availability(pd.read_csv(
            DATASETS_DIR / "car_or_van_availability" / "TS045-2021-4-filtered-2026-11-02T14_52_56Z.csv",
            low_memory=False,
        )),
        "deri": _prepare_national_deri(),
        "imd": prepare_imd_2025(pd.read_csv(
            DATASETS_DIR / "indices_of_deprivation_imd" / "2025" / "england_imd_2025_lsoa.csv",
            low_memory=False,
        )),
        "lsoac": lsoac,
        "ts003": _prepare_national_ts003(),
        "tenure": prepare_housing_tenure(pd.read_csv(
            DATASETS_DIR / "TS054_tenure" / "TS054-2021-4-filtered-2026-09-24T09_58_45Z.csv",
            low_memory=False,
        )),
        "occupancy": prepare_bedroom_occupancy(pd.read_csv(
            DATASETS_DIR / "TS052_occupancy_rating_for_bedrooms" / "TS052-2021-5-filtered-2026-09-24T09_59_23Z.csv",
            low_memory=False,
        )),
    }


@lru_cache(maxsize=1)
def _get_lsoa_workbook_extractor():
    extractor_path = BASE_DIR / "scripts" / "utils" / "_lincolnshire_xlsx.py"
    spec = importlib.util.spec_from_file_location("lumhr_lsoa_xlsx", extractor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load workbook extractor: {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_lsoa_rows


def _extract_area_energy_metrics(lsoa_codes: set[str]) -> list[pd.DataFrame]:
    extract_lsoa_rows = _get_lsoa_workbook_extractor()
    utils_source = BASE_DIR / "scripts" / "utils" / "source_data"
    fuel_headers, fuel_rows, _ = extract_lsoa_rows(
        utils_source / "fuel_poverty" / "2024" / "fuel-poverty-sub-regional-2026-2024-data-tables.xlsx",
        "Table 4",
        3,
        "A",
        lsoa_codes,
    )
    fuel = pd.DataFrame(fuel_rows, columns=fuel_headers)
    fuel = fuel[["LSOA Code", "Proportion of households fuel poor (%)"]].copy()
    fuel["LSOA_CODE"] = fuel["LSOA Code"].map(normalize_code)
    fuel["Fuel_Poverty_Pct"] = parse_numeric(fuel["Proportion of households fuel poor (%)"])

    gas_headers, gas_rows, _ = extract_lsoa_rows(
        utils_source / "properties_not_connected_to_gas_network" / "LSOA_estimates_of_properties_not_connected_to_the_gas_network_2015-2024.xlsx",
        "2024",
        4,
        "E",
        lsoa_codes,
    )
    gas = pd.DataFrame(gas_rows, columns=gas_headers)
    gas_code_column = find_column(gas, ["LSOA code"])
    gas_percent_column = find_column(gas, ["Estimated percentage of properties not on the gas grid"])
    gas = gas[[gas_code_column, gas_percent_column]].copy()
    gas["LSOA_CODE"] = gas[gas_code_column].map(normalize_code)
    gas_source_values = gas[gas_percent_column].astype(str).str.strip()
    gas["Properties_Not_On_Gas_Grid_Pct"] = parse_numeric(
        gas_source_values
    ) * 100.0
    gas["Properties_Not_On_Gas_Grid_Status"] = gas_source_values.map(
        lambda value: "Suppressed in source" if value in {"", ".."} else None
    )

    return [
        fuel[["LSOA_CODE", "Fuel_Poverty_Pct"]].drop_duplicates("LSOA_CODE"),
        gas[["LSOA_CODE", "Properties_Not_On_Gas_Grid_Pct", "Properties_Not_On_Gas_Grid_Status"]].drop_duplicates("LSOA_CODE"),
    ]


def _get_area_lsoa_metrics(scope: str) -> pd.DataFrame:
    """Return selected-area metrics for access, gap, and rural-risk scoring."""
    if scope == "lincolnshire":
        if scope in _AREA_METRICS_CACHE:
            return _AREA_METRICS_CACHE[scope]
        metrics = LSOA_METRICS.copy()
        lsoa_codes = set(metrics["LSOA_CODE"].dropna().astype(str))
        for component in _extract_area_energy_metrics(lsoa_codes):
            for column in [name for name in component.columns if name != "LSOA_CODE"]:
                source_column = f"{column}__complete_source"
                supplemental = component[["LSOA_CODE", column]].drop_duplicates("LSOA_CODE")
                supplemental = supplemental.rename(columns={column: source_column})
                metrics = metrics.merge(supplemental, on="LSOA_CODE", how="left")
                if column in metrics.columns:
                    metrics[column] = metrics[column].combine_first(metrics[source_column])
                else:
                    metrics[column] = metrics[source_column]
                metrics = metrics.drop(columns=[source_column])
        _AREA_METRICS_CACHE[scope] = metrics
        return metrics
    if scope in _AREA_METRICS_CACHE:
        return _AREA_METRICS_CACHE[scope]

    area_data = _get_current_area_need_base(scope)
    metrics = area_data["metrics"].copy()
    lsoa_codes = set(metrics["LSOA_CODE"].dropna().astype(str))
    shared = _get_national_access_risk_components()
    components = [*shared.values(), *_extract_area_energy_metrics(lsoa_codes)]
    for component in components:
        if "LSOA_CODE" not in component.columns:
            continue
        component = component[component["LSOA_CODE"].astype(str).isin(lsoa_codes)]
        component = component[[
            "LSOA_CODE",
            *[column for column in component.columns if column != "LSOA_CODE" and column not in metrics.columns],
        ]]
        metrics = metrics.merge(component, on="LSOA_CODE", how="left")
    _AREA_METRICS_CACHE[scope] = metrics
    return metrics


@app.get("/api/access_scores")
def access_scores_api():
    scope = str(request.args.get("scope", "lincolnshire") or "lincolnshire").strip().lower()
    if scope not in NEED_INDEX_GEOJSON_PATHS:
        scope = "lincolnshire"
    try:
        mh002 = _parse_float_arg("mh002", 8.33)
        mh021 = _parse_float_arg("mh021", 8.33)
        mh_pca = _parse_float_arg("mh_pca", 8.33)
        dep_pca = _parse_float_arg("dep_pca", 8.33)
        dep004 = _parse_float_arg("dep004", 8.33)
        gp_pt = _parse_float_arg("gp_pt", 8.33)
        gp_car = _parse_float_arg("gp_car", 8.33)
        hosp_pt = _parse_float_arg("hosp_pt", 8.33)
        hosp_car = _parse_float_arg("hosp_car", 8.33)
        rural = _parse_float_arg("rural", 8.33)
        car = _parse_float_arg("car", 8.33)
        digital = _parse_float_arg("digital", 8.33)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    required_cols = [
        "MH002_Access_Pct",
        "MH021_Access_Pct",
        "MH_PCA_Access_Pct",
        "Dep_PCA_Access_Pct",
        "DEP004_Access_Pct",
        "GP_PT_Time",
        "GP_Car_Time",
        "Hosp_PT_Time",
        "Hosp_Car_Time",
        "Rural_Access",
        "Car_Access",
        "Digital_Access",
    ]
    metrics = _get_area_lsoa_metrics(scope)
    missing = [c for c in required_cols if c not in metrics.columns]
    if missing:
        return jsonify({"error": f"Missing access columns: {', '.join(missing)}"}), 400

    scored = apply_access_index(
        metrics, mh002, mh021, mh_pca, dep_pca, dep004, gp_pt, gp_car, hosp_pt, hosp_car, rural, car, digital
    )

    layers = {
        "Access_Index": _scores_to_dict(scored, "Access_Index"),
        "MH002_Access_Pct": _scores_to_dict(scored, "MH002_Access_Pct"),
        "MH021_Access_Pct": _scores_to_dict(scored, "MH021_Access_Pct"),
        "MH_PCA_Access_Pct": _scores_to_dict(scored, "MH_PCA_Access_Pct"),
        "Dep_PCA_Access_Pct": _scores_to_dict(scored, "Dep_PCA_Access_Pct"),
        "DEP004_Access_Pct": _scores_to_dict(scored, "DEP004_Access_Pct"),
        "GP_PT_Time": _scores_to_dict(scored, "GP_PT_Time"),
        "GP_Car_Time": _scores_to_dict(scored, "GP_Car_Time"),
        "Hosp_PT_Time": _scores_to_dict(scored, "Hosp_PT_Time"),
        "Hosp_Car_Time": _scores_to_dict(scored, "Hosp_Car_Time"),
        "Rural_Access": _scores_to_dict(scored, "Rural_Access"),
        "Car_Access": _scores_to_dict(scored, "Car_Access"),
        "Digital_Access": _scores_to_dict(scored, "Digital_Access"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)

    extra_detail_cols = [
        "Car_Access_Pct",
        "No_Cars_Pct",
        "One_Car_Pct",
        "Two_Cars_Pct",
        "Three_Plus_Cars_Pct",
        "Total_Households",
        "DERI_Score",
        "Demography_Score",
        "Deprivation_Score",
        "Broadband_Score",
        "Avg_Download_Speed_Mbps",
        "No_Superfast_Broadband_Pct",
        "Slow_Connections_Pct",
        "ONS_Pop_Total_2024",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    detail_cols = [
        "LSOA_CODE",
        "Access_Index",
        "RUC21NM",
        "Urban_rural_flag",
        *required_cols,
        *extra_detail_cols,
    ]
    if lsoa_name_col:
        detail_cols.insert(1, lsoa_name_col)
    details_df = scored[[c for c in detail_cols if c in scored.columns]].copy()
    lsoa_details: dict[str, dict[str, object]] = {}
    for row in details_df.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        ruc_text = str(row.get("RUC21NM", "") or "")
        flag_text = str(row.get("Urban_rural_flag", "") or ("Rural" if "rural" in ruc_text.lower() else "Urban"))
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "access_index": _num_or_none(row.get("Access_Index")),
            "mh002_pct": _num_or_none(row.get("MH002_Access_Pct")),
            "mh021_pct": _num_or_none(row.get("MH021_Access_Pct")),
            "mh_pca_pct": _num_or_none(row.get("MH_PCA_Access_Pct")),
            "dep_pca_pct": _num_or_none(row.get("Dep_PCA_Access_Pct")),
            "dep004_pct": _num_or_none(row.get("DEP004_Access_Pct")),
            "gp_pt_time": _num_or_none(row.get("GP_PT_Time")),
            "gp_car_time": _num_or_none(row.get("GP_Car_Time")),
            "hosp_pt_time": _num_or_none(row.get("Hosp_PT_Time")),
            "hosp_car_time": _num_or_none(row.get("Hosp_Car_Time")),
            "rural_access": _num_or_none(row.get("Rural_Access")),
            "ruc21nm": ruc_text,
            "urban_rural_flag": flag_text,
            "is_rural": "rural" in ruc_text.lower(),
            "car_access": _num_or_none(row.get("Car_Access")),
            "car_access_pct": _num_or_none(row.get("Car_Access_Pct")),
            "no_cars_pct": _num_or_none(row.get("No_Cars_Pct")),
            "one_car_pct": _num_or_none(row.get("One_Car_Pct")),
            "two_cars_pct": _num_or_none(row.get("Two_Cars_Pct")),
            "three_plus_cars_pct": _num_or_none(row.get("Three_Plus_Cars_Pct")),
            "total_households": _num_or_none(row.get("Total_Households")),
            "digital_access": _num_or_none(row.get("Digital_Access")),
            "deri_score": _num_or_none(row.get("DERI_Score")),
            "demography_score": _num_or_none(row.get("Demography_Score")),
            "deprivation_score": _num_or_none(row.get("Deprivation_Score")),
            "broadband_score": _num_or_none(row.get("Broadband_Score")),
            "avg_download_speed_mbps": _num_or_none(row.get("Avg_Download_Speed_Mbps")),
            "no_superfast_broadband_pct": _num_or_none(row.get("No_Superfast_Broadband_Pct")),
            "slow_connections_pct": _num_or_none(row.get("Slow_Connections_Pct")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
        }
    )


@app.get("/api/access_gap_scores")
def access_gap_scores_api():
    scope = str(request.args.get("scope", "lincolnshire") or "lincolnshire").strip().lower()
    if scope not in NEED_INDEX_GEOJSON_PATHS:
        scope = "lincolnshire"
    try:
        dep = _parse_float_arg("dep", 16.7)
        smi = _parse_float_arg("smi", 16.7)
        prescribing = _parse_float_arg("prescribing", 16.7)
        samhi_weight = _parse_float_arg("samhi", 16.7)
        dla_weight = _parse_float_arg("dla", 16.7)
        pip_weight = _parse_float_arg("pip", 16.7)
        samhi_year = _parse_int_arg("samhi_year", max(SAMHI_YEARS))
        dwp_year = _parse_int_arg("dwp_year", max(DWP_AVAILABLE_YEARS))
        mh002 = _parse_float_arg("mh002", 8.33)
        mh021 = _parse_float_arg("mh021", 8.33)
        mh_pca = _parse_float_arg("mh_pca", 8.33)
        dep_pca = _parse_float_arg("dep_pca", 8.33)
        dep004 = _parse_float_arg("dep004", 8.33)
        gp_pt = _parse_float_arg("gp_pt", 8.33)
        gp_car = _parse_float_arg("gp_car", 8.33)
        hosp_pt = _parse_float_arg("hosp_pt", 8.33)
        hosp_car = _parse_float_arg("hosp_car", 8.33)
        rural = _parse_float_arg("rural", 8.33)
        car = _parse_float_arg("car", 8.33)
        digital = _parse_float_arg("digital", 8.33)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if samhi_year not in SAMHI_YEARS:
        return jsonify({"error": f"samhi_year must be one of {SAMHI_YEARS}"}), 400
    if dwp_year not in DWP_AVAILABLE_YEARS:
        return jsonify({"error": f"dwp_year must be one of {DWP_AVAILABLE_YEARS}"}), 400

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    metrics = _get_area_lsoa_metrics(scope)
    if samhi_index_col not in metrics.columns:
        return jsonify({"error": f"Missing SAMHI column: {samhi_index_col}"}), 400

    access_required_cols = [
        "MH002_Access_Pct",
        "MH021_Access_Pct",
        "MH_PCA_Access_Pct",
        "Dep_PCA_Access_Pct",
        "DEP004_Access_Pct",
        "GP_PT_Time",
        "GP_Car_Time",
        "Hosp_PT_Time",
        "Hosp_Car_Time",
        "Rural_Access",
        "Car_Access",
        "Digital_Access",
    ]
    missing = [c for c in access_required_cols if c not in metrics.columns]
    if missing:
        return jsonify({"error": f"Missing access columns: {', '.join(missing)}"}), 400

    scored_with_dwp = _attach_public_dwp_layer(metrics, dwp_year)
    need_scored = apply_need_index(
        scored_with_dwp, dep, smi, prescribing, samhi_weight, samhi_index_col, dla_weight, pip_weight
    )
    access_scored = apply_access_index(
        metrics, mh002, mh021, mh_pca, dep_pca, dep004, gp_pt, gp_car, hosp_pt, hosp_car, rural, car, digital
    )

    combined = need_scored[["LSOA_CODE", "Need_Index"]].merge(
        access_scored[["LSOA_CODE", "Access_Index"]], on="LSOA_CODE", how="outer"
    )
    combined["Access_Gap_Index"] = combined["Need_Index"] - combined["Access_Index"]
    dwp_cols = [
        "dwp_dla_count",
        "dwp_pip_count",
        "dwp_dla_rate_pct",
        "dwp_pip_rate_pct",
        "dwp_dla_source_status",
        "dwp_pip_source_status",
        "dwp_population_source_year",
        "dla_pip_source",
    ]
    combined = combined.merge(
        scored_with_dwp[["LSOA_CODE", *dwp_cols]], on="LSOA_CODE", how="left"
    )

    pop_cols = [
        "RUC21NM",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    extra_join_cols = [c for c in pop_cols if c in metrics.columns]
    if extra_join_cols:
        combined = combined.merge(metrics[["LSOA_CODE", *extra_join_cols]], on="LSOA_CODE", how="left")

    lsoa_name_col = _get_lsoa_name_column(metrics)
    if lsoa_name_col:
        combined = combined.merge(metrics[["LSOA_CODE", lsoa_name_col]], on="LSOA_CODE", how="left")

    layers = {
        "Access_Gap_Index": _scores_to_dict(combined, "Access_Gap_Index"),
        "Need_Index": _scores_to_dict(combined, "Need_Index"),
        "Access_Index": _scores_to_dict(combined, "Access_Index"),
        "Pct_65plus": _scores_to_dict(combined, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(combined, "GP_Registration_Rate_Pct"),
        "DWP_DLA_Rate_Pct": _scores_to_dict(combined, "dwp_dla_rate_pct"),
        "DWP_PIP_Rate_Pct": _scores_to_dict(combined, "dwp_pip_rate_pct"),
    }

    lsoa_details: dict[str, dict[str, object]] = {}
    for row in combined.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        ruc_text = str(row.get("RUC21NM", "") or "")
        flag_text = str(row.get("Urban_rural_flag", "") or ("Rural" if "rural" in ruc_text.lower() else "Urban"))
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "access_gap_index": _num_or_none(row.get("Access_Gap_Index")),
            "need_index": _num_or_none(row.get("Need_Index")),
            "access_index": _num_or_none(row.get("Access_Index")),
            "dwp_dla_count": _num_or_none(row.get("dwp_dla_count")),
            "dwp_pip_count": _num_or_none(row.get("dwp_pip_count")),
            "dwp_dla_rate_pct": _num_or_none(row.get("dwp_dla_rate_pct")),
            "dwp_pip_rate_pct": _num_or_none(row.get("dwp_pip_rate_pct")),
            "dwp_dla_source_status": _text_or_none(row.get("dwp_dla_source_status")),
            "dwp_pip_source_status": _text_or_none(row.get("dwp_pip_source_status")),
            "dwp_population_source_year": _num_or_none(row.get("dwp_population_source_year")),
            "dwp_source": _text_or_none(row.get("dla_pip_source")),
            "ruc21nm": ruc_text,
            "urban_rural_flag": flag_text,
            "is_rural": "rural" in ruc_text.lower(),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
            "meta": {
                "samhi_year": samhi_year,
                "dwp_year": dwp_year,
                "dwp_label": f"DWP DLA and PIP rates ({dwp_year})",
                "dwp_component_labels": {"dla": f"DLA rate ({dwp_year})", "pip": f"PIP rate ({dwp_year})"},
            },
        }
    )


_SAMHI_AREA_METRICS_CACHE: dict[str, pd.DataFrame] = {}
_NEED_DWP_PANEL_CACHE: pd.DataFrame | None = None


def _get_samhi_area_metrics(scope: str = "lincolnshire") -> pd.DataFrame:
    clean_scope = _normalise_public_samhi3_scope(scope)
    if clean_scope == "lincolnshire":
        return LSOA_METRICS
    if clean_scope in _SAMHI_AREA_METRICS_CACHE:
        return _SAMHI_AREA_METRICS_CACHE[clean_scope]

    source_path = DATASETS_DIR / "samhi" / "samhi_21_01_v5.00_2011_2022_LSOA.csv"
    lookup_path = (
        DATASETS_DIR
        / "lsoa"
        / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    if not source_path.exists() or not lookup_path.exists():
        _SAMHI_AREA_METRICS_CACHE[clean_scope] = pd.DataFrame()
        return _SAMHI_AREA_METRICS_CACHE[clean_scope]

    source = pd.read_csv(source_path, low_memory=False)
    source["lsoa11"] = source["lsoa11"].astype(str).str.strip()
    lookup = pd.read_csv(
        lookup_path,
        usecols=["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22NM"],
        dtype=str,
    ).rename(columns={"LSOA11CD": "lsoa11"})
    for column in ["lsoa11", "LSOA21CD", "LSOA21NM", "LAD22NM"]:
        lookup[column] = lookup[column].astype(str).str.strip()
    lookup = lookup.drop_duplicates("LSOA21CD")
    if clean_scope in PUBLIC_SAMHI3_SCOPE_LADS:
        lookup = lookup[lookup["LAD22NM"].isin(PUBLIC_SAMHI3_SCOPE_LADS[clean_scope])]

    samhi_columns = [
        column
        for year in SAMHI_YEARS
        for column in (f"samhi_index.{year}", f"samhi_dec.{year}")
        if column in source.columns
    ]
    area_metrics = lookup.merge(source[["lsoa11", *samhi_columns]], on="lsoa11", how="inner")
    rural_path = DATASETS_DIR / "rural_urban_classification_2021_lsoa" / "Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv"
    if rural_path.exists():
        rural = pd.read_csv(rural_path, usecols=["LSOA21CD", "RUC21NM", "Urban_rural_flag"], dtype=str)
        rural["LSOA21CD"] = rural["LSOA21CD"].astype(str).str.strip()
        area_metrics = area_metrics.merge(
            rural.drop_duplicates("LSOA21CD"),
            on="LSOA21CD",
            how="left",
        )
    area_metrics = area_metrics.rename(columns={"LSOA21CD": "LSOA_CODE"})
    _SAMHI_AREA_METRICS_CACHE[clean_scope] = area_metrics.drop_duplicates("LSOA_CODE", keep="last")
    return _SAMHI_AREA_METRICS_CACHE[clean_scope]


_CURRENT_AREA_NEED_BASE_CACHE: dict[str, dict[str, pd.DataFrame]] = {}


def _get_current_area_need_base(scope: str) -> dict[str, pd.DataFrame]:
    """Allocate 2024/25 practice QOF and May 2026 prescriptions to county LSOAs."""
    if scope in _CURRENT_AREA_NEED_BASE_CACHE:
        return _CURRENT_AREA_NEED_BASE_CACHE[scope]

    scopes = AREA_COMPONENT_SCOPES[scope]
    dep_parts = []
    smi_parts = []
    prescribing_parts = []
    for part in scopes:
        dep_path = DATASETS_DIR / "quality_outcomes_framework" / f"qof_depression_2425_{part}.csv"
        smi_path = DATASETS_DIR / "quality_outcomes_framework" / f"qof_mental_health_2425_{part}.csv"
        prescribing_path = DATASETS_DIR / "gp_prescribing_data" / f"need_index_antidepressant_items_per_gp_{part}_may_2026.csv"
        for path in (dep_path, smi_path, prescribing_path):
            if not path.exists():
                raise FileNotFoundError(f"Required county Need Index input is missing: {path.name}")
        dep_parts.append(pd.read_csv(dep_path, low_memory=False))
        smi_parts.append(pd.read_csv(smi_path, low_memory=False))
        prescribing_parts.append(pd.read_csv(prescribing_path, low_memory=False))

    dep_raw = pd.concat(dep_parts, ignore_index=True).drop_duplicates("Practice Code", keep="last")
    smi_raw = pd.concat(smi_parts, ignore_index=True).drop_duplicates("Practice Code", keep="last")
    prescribing_raw = pd.concat(prescribing_parts, ignore_index=True).drop_duplicates("id", keep="last")
    dep_df = prepare_depression(dep_raw)
    smi_df = prepare_smi(smi_raw)
    prescribing_df = prepare_prescribing(prescribing_raw)
    gp_master = build_gp_master(dep_df, smi_df, prescribing_df)

    # Keep all registered LSOAs in each area's practice denominator, including
    # patients who live outside the selected map boundary.
    mapping_path = (
        DATASETS_DIR
        / "patients_registered_gp_practice"
        / f"{scope}_july_2026"
        / "gp-reg-pat-prac-lsoa-all.csv"
    )
    mapping_df = pd.read_csv(mapping_path, low_memory=False)
    mapping_df["PRACTICE_CODE"] = mapping_df["PRACTICE_CODE"].astype(str).str.strip()
    mapping_df["LSOA_CODE"] = mapping_df["LSOA_CODE"].astype(str).str.strip()
    mapping_df["NUMBER_OF_PATIENTS"] = pd.to_numeric(mapping_df["NUMBER_OF_PATIENTS"], errors="coerce")
    mapping_df = mapping_df[
        mapping_df["PRACTICE_CODE"].ne("")
        & mapping_df["LSOA_CODE"].str.fullmatch(r"E\d{8}")
        & mapping_df["NUMBER_OF_PATIENTS"].gt(0)
    ].copy()
    marker_mapping_df = mapping_df
    allocation_mapping_df = mapping_df[
        ~mapping_df["PRACTICE_CODE"].isin(SPECIAL_ALLOCATION_PRACTICE_CODES)
    ].copy()
    parkfields_january = mapping_df[
        mapping_df.get("REGISTRATION_SOURCE_DATE", pd.Series(index=mapping_df.index, dtype=str))
        .astype(str)
        .eq("2026-01-01")
    ]
    if not parkfields_january.empty:
        parkfields_by_lsoa = parkfields_january.groupby("LSOA_CODE", as_index=False).agg(
            Parkfields_Jan_2026_Patients=("NUMBER_OF_PATIENTS", "sum"),
            Parkfields_July_2026_Patients=("JULY_2026_NUMBER_OF_PATIENTS", "sum"),
        )
    else:
        parkfields_by_lsoa = pd.DataFrame(
            columns=["LSOA_CODE", "Parkfields_Jan_2026_Patients", "Parkfields_July_2026_Patients"]
        )
    practice_totals = allocation_mapping_df.groupby("PRACTICE_CODE")["NUMBER_OF_PATIENTS"].transform("sum")
    allocation_mapping_df["Weight"] = (
        allocation_mapping_df["NUMBER_OF_PATIENTS"] / practice_totals.replace(0, np.nan)
    )
    allocated, _ = allocate_registers_to_lsoa(allocation_mapping_df, gp_master)
    access_rates = allocate_access_rates_to_lsoa(allocation_mapping_df, gp_master)
    area = _get_samhi_area_metrics(scope).copy()
    area = area.merge(allocated, on="LSOA_CODE", how="left")
    area = area.merge(access_rates, on="LSOA_CODE", how="left")
    area = area.merge(parkfields_by_lsoa, on="LSOA_CODE", how="left")
    pop_path = DATASETS_DIR / "population_estimates" / f"{scope}_lsoa_population_estimates_2024.csv"
    population = pd.read_csv(pop_path, low_memory=False)
    area = area.merge(population, on="LSOA_CODE", how="left", suffixes=("", "_population"))

    locations = BUNDLE["gp_loc_df"]
    geojson_path = DATASETS_DIR / NEED_INDEX_GEOJSON_PATHS[scope]
    with geojson_path.open(encoding="utf-8") as stream:
        geojson = json.load(stream)
    centroid_rows = []
    try:
        from shapely.geometry import shape

        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            code = props.get("LSOA21CD") or props.get("CODE") or props.get("LSOA_CODE")
            if not code or not feature.get("geometry"):
                continue
            point = shape(feature["geometry"]).representative_point()
            centroid_rows.append({"LSOA_CODE": str(code), "lat": point.y, "lon": point.x})
    except ImportError:
        centroid_rows = []
    centroids = pd.DataFrame(centroid_rows, columns=["LSOA_CODE", "lat", "lon"])
    markers = build_gp_marker_df(
        locations,
        gp_master,
        marker_mapping_df,
        centroids,
        set(area["LSOA_CODE"].dropna().astype(str)),
    )
    # Allocation mappings include practices with only a small number of
    # patients in the area, including practices whose own postcode is elsewhere.
    # Keep those practices in the LSOA calculations, but only pin practices
    # whose ODS postcode falls within this map boundary.
    in_area_lsoas = set(area["LSOA_CODE"].dropna().astype(str))
    local_practice_codes = set(
        locations.loc[locations["GP_LSOA"].astype(str).isin(in_area_lsoas), "PRACTICE_CODE"]
        .astype(str)
        .str.strip()
    )
    if scope in {"nottinghamshire", "lincolnshire_nottinghamshire_derbyshire"}:
        local_practice_codes |= set(SPECIAL_ALLOCATION_PRACTICE_CODES)
    markers = markers[markers["PRACTICE_CODE"].astype(str).isin(local_practice_codes)].copy()
    out_of_area = allocation_mapping_df[
        ~allocation_mapping_df["LSOA_CODE"].isin(set(area["LSOA_CODE"].dropna().astype(str)))
    ]
    out_of_area_stats = {
        "out_of_area_lsoa_count": int(out_of_area["LSOA_CODE"].nunique()),
        "out_of_area_patients": int(round(float(out_of_area["NUMBER_OF_PATIENTS"].sum()))),
    }
    _CURRENT_AREA_NEED_BASE_CACHE[scope] = {
        "metrics": area,
        "markers": markers,
        **out_of_area_stats,
    }
    return _CURRENT_AREA_NEED_BASE_CACHE[scope]


def _get_current_area_need_metrics(scope: str, dwp_year: int) -> pd.DataFrame:
    global _NEED_DWP_PANEL_CACHE
    metrics = _get_current_area_need_base(scope)["metrics"].copy()
    panel = _NEED_DWP_PANEL_CACHE
    if panel is None:
        panel_path = _dashboard_data_path(SAMHI_FORECAST_DATA_DIR, "three_component_panel.csv")
        columns = [
            "lsoa11", "year", "antidep_rate", "qof_dep_pct",
            "dwp_dla_count", "dwp_pip_count", "dwp_dla_rate_pct", "dwp_pip_rate_pct",
        ]
        panel = pd.read_csv(panel_path, usecols=columns, low_memory=False) if panel_path.exists() else pd.DataFrame(columns=columns)
        panel["lsoa11"] = panel["lsoa11"].astype(str).str.strip()
        panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
        _NEED_DWP_PANEL_CACHE = panel
    dwp = panel.loc[
        panel["year"].eq(dwp_year),
        [
            "lsoa11",
            "dwp_dla_count",
            "dwp_pip_count",
            "dwp_dla_rate_pct",
            "dwp_pip_rate_pct",
        ],
    ].drop_duplicates("lsoa11", keep="last")
    metrics = metrics.merge(dwp, on="lsoa11", how="left")
    has_dwp_record = metrics[[
        "dwp_dla_count",
        "dwp_pip_count",
        "dwp_dla_rate_pct",
        "dwp_pip_rate_pct",
    ]].notna().any(axis=1)
    metrics["dla_pip_source"] = np.where(has_dwp_record, "DWP_August", None)
    pip_status = _load_public_dwp_pip_suppression(str(BASE_DIR))
    pip_status = pip_status[pip_status["year"].eq(dwp_year)].drop(columns=["year"], errors="ignore")
    return metrics.merge(pip_status, on="LSOA_CODE", how="left")


@app.get("/api/samhi_scores")
def samhi_scores_api():
    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    scope_metrics = _get_samhi_area_metrics(scope)
    if scope_metrics.empty:
        return jsonify({"error": "SAMHI data are unavailable for this geographic area."}), 404
    mode = str(request.args.get("mode", "Index")).strip()
    analysis_mode = str(request.args.get("analysis_mode", "Single Year")).strip()

    mode_norm = mode.lower()
    if mode_norm not in {"index", "decile"}:
        return jsonify({"error": "mode must be 'Index' or 'Decile'"}), 400

    try:
        year = _parse_int_arg("year", max(SAMHI_YEARS))
        from_year = _parse_int_arg("from_year", min(SAMHI_YEARS))
        to_year = _parse_int_arg("to_year", max(SAMHI_YEARS))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    for val, name in [(year, "year"), (from_year, "from_year"), (to_year, "to_year")]:
        if val not in SAMHI_YEARS:
            return jsonify({"error": f"{name} must be one of {SAMHI_YEARS}"}), 400

    is_change = analysis_mode.lower() in {"change", "change between years"}

    if not is_change:
        index_col, dec_col = get_samhi_columns(year)
        selected_col = index_col if mode_norm == "index" else dec_col
        if selected_col not in scope_metrics.columns:
            return jsonify({"error": f"Missing SAMHI column: {selected_col}"}), 400

        lsoa_name_col = _get_lsoa_name_column(scope_metrics)
        cols = ["LSOA_CODE", selected_col]
        if lsoa_name_col:
            cols.insert(1, lsoa_name_col)
        pop_cols = [
            "RUC21NM",
            "Urban_rural_flag",
            "ONS_Pop_Total_2024",
            "ONS_Pop_18plus",
            "ONS_Pop_65plus",
            "ONS_Pop_0to17",
            "Pct_18plus",
            "Pct_65plus",
            "Pct_0to17",
            "GP_Registered_Patients",
            "GP_Registration_Rate_Pct",
            "Registration_Gap_Est",
            "List_Inflation_Est",
        ]
        cols.extend([c for c in pop_cols if c in scope_metrics.columns])
        out_df = scope_metrics[cols].copy()
        out_df = out_df.rename(columns={selected_col: "score"})

        details: dict[str, dict[str, object]] = {}
        for row in out_df.to_dict(orient="records"):
            code = str(row.get("LSOA_CODE", "") or "")
            if not code:
                continue
            ruc_text = str(row.get("RUC21NM", "") or "")
            flag_text = str(row.get("Urban_rural_flag", "") or ("Rural" if "rural" in ruc_text.lower() else "Urban"))
            details[code] = {
                "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
                "value": _num_or_none(row.get("score")),
                "ruc21nm": ruc_text,
                "urban_rural_flag": flag_text,
                "is_rural": "rural" in ruc_text.lower(),
                "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
                "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
                "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
                "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
                "pct_18plus": _num_or_none(row.get("Pct_18plus")),
                "pct_65plus": _num_or_none(row.get("Pct_65plus")),
                "pct_0to17": _num_or_none(row.get("Pct_0to17")),
                "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
                "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
                "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
                "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
            }

        return jsonify(
            {
                "scores": _scores_to_dict(out_df, "score"),
                "details": details,
                "meta": {
                    "analysis_mode": "Single Year",
                    "mode": mode,
                    "year": year,
                },
            }
        )

    from_index_col, from_dec_col = get_samhi_columns(from_year)
    to_index_col, to_dec_col = get_samhi_columns(to_year)
    from_col = from_index_col if mode_norm == "index" else from_dec_col
    to_col = to_index_col if mode_norm == "index" else to_dec_col

    missing = [col for col in [from_col, to_col] if col not in scope_metrics.columns]
    if missing:
        return jsonify({"error": f"Missing SAMHI columns: {', '.join(missing)}"}), 400

    lsoa_name_col = _get_lsoa_name_column(scope_metrics)
    cols = ["LSOA_CODE", from_col, to_col]
    if lsoa_name_col:
        cols.insert(1, lsoa_name_col)
    pop_cols = [
        "RUC21NM",
        "Urban_rural_flag",
        "ONS_Pop_Total_2024",
        "ONS_Pop_18plus",
        "ONS_Pop_65plus",
        "ONS_Pop_0to17",
        "Pct_18plus",
        "Pct_65plus",
        "Pct_0to17",
        "GP_Registered_Patients",
        "GP_Registration_Rate_Pct",
        "Registration_Gap_Est",
        "List_Inflation_Est",
    ]
    cols.extend([c for c in pop_cols if c in scope_metrics.columns])
    out_df = scope_metrics[cols].copy()
    out_df["from_value"] = pd.to_numeric(out_df[from_col], errors="coerce")
    out_df["to_value"] = pd.to_numeric(out_df[to_col], errors="coerce")
    out_df["score"] = out_df["to_value"] - out_df["from_value"]

    details: dict[str, dict[str, object]] = {}
    for row in out_df.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        ruc_text = str(row.get("RUC21NM", "") or "")
        flag_text = str(row.get("Urban_rural_flag", "") or ("Rural" if "rural" in ruc_text.lower() else "Urban"))
        details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "from_value": _num_or_none(row.get("from_value")),
            "to_value": _num_or_none(row.get("to_value")),
            "change": _num_or_none(row.get("score")),
            "ruc21nm": ruc_text,
            "urban_rural_flag": flag_text,
            "is_rural": "rural" in ruc_text.lower(),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
        }

    return jsonify(
        {
            "scores": _scores_to_dict(out_df, "score"),
            "details": details,
            "meta": {
                "analysis_mode": "Change Between Years",
                "mode": mode,
                "from_year": from_year,
                "to_year": to_year,
            },
        }
    )


@app.get("/api/rural_risk_scores")
def rural_risk_scores_api():
    scope = str(request.args.get("scope", "lincolnshire") or "lincolnshire").strip().lower()
    if scope not in NEED_INDEX_GEOJSON_PATHS:
        scope = "lincolnshire"
    try:
        rural_weight = _parse_float_arg("w_rural", 7.7)
        gp_pt_weight = _parse_float_arg("w_gp_pt", 7.7)
        gp_car_weight = _parse_float_arg("w_gp_car", 7.7)
        no_car_weight = _parse_float_arg("w_no_car", 7.7)
        imd_weight = _parse_float_arg("w_imd", 7.7)
        oac_weight = _parse_float_arg("w_oac", 7.7)
        household_weight = _parse_float_arg("w_household", 7.7)
        fuel_poverty_weight = _parse_float_arg("w_fuel_poverty", 7.7)
        off_gas_grid_weight = _parse_float_arg("w_off_gas_grid", 7.7)
        housing_tenure_weight = _parse_float_arg("w_housing_tenure", 7.7)
        overcrowding_weight = _parse_float_arg("w_overcrowding", 7.7)
        dla_weight = _parse_float_arg("w_dla", 7.7)
        pip_weight = _parse_float_arg("w_pip", 7.7)
        dwp_year = _parse_int_arg("dwp_year", max(DWP_AVAILABLE_YEARS))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if dwp_year not in DWP_AVAILABLE_YEARS:
        return jsonify({"error": f"dwp_year must be one of {DWP_AVAILABLE_YEARS}"}), 400

    scored = _attach_public_dwp_layer(_get_area_lsoa_metrics(scope), dwp_year)
    scored = apply_rural_risk_index(
        scored,
        rural_weight=rural_weight,
        gp_pt_weight=gp_pt_weight,
        gp_car_weight=gp_car_weight,
        no_car_weight=no_car_weight,
        imd_weight=imd_weight,
        oac_weight=oac_weight,
        household_weight=household_weight,
        fuel_poverty_weight=fuel_poverty_weight,
        off_gas_grid_weight=off_gas_grid_weight,
        housing_tenure_weight=housing_tenure_weight,
        overcrowding_weight=overcrowding_weight,
        dla_weight=dla_weight,
        pip_weight=pip_weight,
    )

    layers = {
        "Rural_Risk_Index": _scores_to_dict(scored, "Rural_Risk_Index"),
        "Rural_Isolation": _scores_to_dict(scored, "Rural_Isolation_Normalized"),
        "GP_PT_Travel_Time": _scores_to_dict(scored, "GP_PT_Time"),
        "GP_Car_Travel_Time": _scores_to_dict(scored, "GP_Car_Time"),
        "No_Cars_Pct": _scores_to_dict(scored, "No_Cars_Pct"),
        "IMD_2025_Decile": _scores_to_dict(scored, "IMD_2025_Decile"),
        "IMD_2025_Rank": _scores_to_dict(scored, "IMD_2025_Rank"),
        "LSOAC_Risk": _scores_to_dict(scored, "LSOAC_Risk_Normalized"),
        "Household_Vulnerability": _scores_to_dict(scored, "Household_Vulnerability_Normalized"),
        "Fuel_Poverty_Pct": _scores_to_dict(scored, "Fuel_Poverty_Pct"),
        "Properties_Not_On_Gas_Grid_Pct": _scores_to_dict(scored, "Properties_Not_On_Gas_Grid_Pct"),
        "Owner_Occupied_Pct": _scores_to_dict(scored, "Owner_Occupied_Pct"),
        "Social_Rented_Pct": _scores_to_dict(scored, "Social_Rented_Pct"),
        "Private_Rented_Pct": _scores_to_dict(scored, "Private_Rented_Pct"),
        "Overcrowded_HH_Pct": _scores_to_dict(scored, "Overcrowded_HH_Pct"),
        "Housing_Tenure_Vulnerability": _scores_to_dict(scored, "Housing_Tenure_Vulnerability_Normalized"),
        "Overcrowding_Risk": _scores_to_dict(scored, "Overcrowding_Normalized"),
        "Single_Pensioner_HH_Pct": _scores_to_dict(scored, "Single_Pensioner_HH_Pct"),
        "Non_Couple_HH_Pct": _scores_to_dict(scored, "Non_Couple_HH_Pct"),
        "Pensioner_Couple_HH_Pct": _scores_to_dict(scored, "Pensioner_Couple_HH_Pct"),
        "Lone_Parent_HH_Pct": _scores_to_dict(scored, "Lone_Parent_Dep_Children_HH_Pct"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
        "DWP_DLA_Rate_Pct": _scores_to_dict(scored, "dwp_dla_rate_pct"),
        "DWP_PIP_Rate_Pct": _scores_to_dict(scored, "dwp_pip_rate_pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)
    lsoa_details: dict[str, dict[str, object]] = {}
    for row in scored.to_dict(orient="records"):
        code = str(row.get("LSOA_CODE", "") or "")
        if not code:
            continue
        ruc_text = str(row.get("RUC21NM", "") or "")
        flag_text = str(row.get("Urban_rural_flag", "") or ("Rural" if "rural" in ruc_text.lower() else "Urban"))
        lsoa_details[code] = {
            "lsoa_name": str(row.get(lsoa_name_col, "") or "") if lsoa_name_col else "",
            "rural_risk_index": _num_or_none(row.get("Rural_Risk_Index")),
            "rural_isolation_score": _num_or_none(row.get("Rural_Isolation_Normalized")),
            "ruc21nm": ruc_text,
            "urban_rural_flag": flag_text,
            "is_rural": "rural" in ruc_text.lower(),
            "gp_pt_time": _num_or_none(row.get("GP_PT_Time")),
            "gp_car_time": _num_or_none(row.get("GP_Car_Time")),
            "no_cars_pct": _num_or_none(row.get("No_Cars_Pct")),
            "imd_2025_decile": _num_or_none(row.get("IMD_2025_Decile")),
            "imd_2025_rank": _num_or_none(row.get("IMD_2025_Rank")),
            "supergroup_code": str(row.get("Supergroup_Code", "") or ""),
            "supergroup_name": str(row.get("Supergroup_Name", "") or ""),
            "group_code": str(row.get("Group_Code", "") or ""),
            "group_name": str(row.get("Group_Name", "") or ""),
            "subgroup_code": str(row.get("Subgroup_Code", "") or ""),
            "subgroup_name": str(row.get("Subgroup_Name", "") or ""),
            "total_households_2021": _num_or_none(row.get("Total_Households_2021")),
            "single_pensioner_hh_count": _num_or_none(row.get("Single_Pensioner_HH_Count")),
            "single_pensioner_hh_pct": _num_or_none(row.get("Single_Pensioner_HH_Pct")),
            "pensioner_couple_hh_count": _num_or_none(row.get("Pensioner_Couple_HH_Count")),
            "pensioner_couple_hh_pct": _num_or_none(row.get("Pensioner_Couple_HH_Pct")),
            "lone_parent_dep_hh_count": _num_or_none(row.get("Lone_Parent_Dep_Children_HH_Count")),
            "lone_parent_dep_hh_pct": _num_or_none(row.get("Lone_Parent_Dep_Children_HH_Pct")),
            "non_couple_hh_count": _num_or_none(row.get("Non_Couple_HH_Count")),
            "non_couple_hh_pct": _num_or_none(row.get("Non_Couple_HH_Pct")),
            "household_vulnerability_score": _num_or_none(row.get("Household_Vulnerability_Normalized")),
            "fuel_poverty_pct": _num_or_none(row.get("Fuel_Poverty_Pct")),
            "properties_not_on_gas_grid_pct": _num_or_none(row.get("Properties_Not_On_Gas_Grid_Pct")),
            "properties_not_on_gas_grid_status": _text_or_none(row.get("Properties_Not_On_Gas_Grid_Status")),
            "tenure_households": _num_or_none(row.get("Tenure_Households")),
            "owner_occupied_hh_count": _num_or_none(row.get("Owner_Occupied_HH_Count")),
            "owner_occupied_pct": _num_or_none(row.get("Owner_Occupied_Pct")),
            "social_rented_hh_count": _num_or_none(row.get("Social_Rented_HH_Count")),
            "social_rented_pct": _num_or_none(row.get("Social_Rented_Pct")),
            "private_rented_hh_count": _num_or_none(row.get("Private_Rented_HH_Count")),
            "private_rented_pct": _num_or_none(row.get("Private_Rented_Pct")),
            "overcrowded_hh_count": _num_or_none(row.get("Overcrowded_HH_Count")),
            "overcrowded_hh_pct": _num_or_none(row.get("Overcrowded_HH_Pct")),
            "housing_tenure_vulnerability": _num_or_none(row.get("Housing_Tenure_Vulnerability_Normalized")),
            "overcrowding_risk": _num_or_none(row.get("Overcrowding_Normalized")),
            "ons_pop_total": _num_or_none(row.get("ONS_Pop_Total_2024")),
            "ons_pop_18plus": _num_or_none(row.get("ONS_Pop_18plus")),
            "ons_pop_65plus": _num_or_none(row.get("ONS_Pop_65plus")),
            "ons_pop_0to17": _num_or_none(row.get("ONS_Pop_0to17")),
            "pct_18plus": _num_or_none(row.get("Pct_18plus")),
            "pct_65plus": _num_or_none(row.get("Pct_65plus")),
            "pct_0to17": _num_or_none(row.get("Pct_0to17")),
            "gp_registered_patients": _num_or_none(row.get("GP_Registered_Patients")),
            "gp_registration_rate_pct": _num_or_none(row.get("GP_Registration_Rate_Pct")),
            "registration_gap_est": _num_or_none(row.get("Registration_Gap_Est")),
            "list_inflation_est": _num_or_none(row.get("List_Inflation_Est")),
            "dwp_dla_rate_pct": _num_or_none(row.get("dwp_dla_rate_pct")),
            "dwp_pip_rate_pct": _num_or_none(row.get("dwp_pip_rate_pct")),
            "dwp_dla_count": _num_or_none(row.get("dwp_dla_count")),
            "dwp_pip_count": _num_or_none(row.get("dwp_pip_count")),
            "dwp_dla_source_status": _text_or_none(row.get("dwp_dla_source_status")),
            "dwp_pip_source_status": _text_or_none(row.get("dwp_pip_source_status")),
            "dwp_population_source_year": _num_or_none(row.get("dwp_population_source_year")),
            "dwp_source": _text_or_none(row.get("dla_pip_source")),
        }

    return jsonify(
        {
            "layers": layers,
            "lsoa_details": lsoa_details,
            "weights": {
                "rural_weight": rural_weight,
                "gp_pt_weight": gp_pt_weight,
                "gp_car_weight": gp_car_weight,
                "no_car_weight": no_car_weight,
                "imd_weight": imd_weight,
                "oac_weight": oac_weight,
                "household_weight": household_weight,
                "fuel_poverty_weight": fuel_poverty_weight,
                "off_gas_grid_weight": off_gas_grid_weight,
                "housing_tenure_weight": housing_tenure_weight,
                "overcrowding_weight": overcrowding_weight,
                "dla_weight": dla_weight,
                "pip_weight": pip_weight,
            },
            "meta": {
                "dwp_year": dwp_year,
                "dwp_label": f"DWP DLA and PIP rates ({dwp_year})",
                "dwp_component_labels": {"dla": f"DLA rate ({dwp_year})", "pip": f"PIP rate ({dwp_year})"},
                "dwp_population_vintage_note": "Complete rows use the panel's estimated 16–64 denominator. Disclosure-suppressed component rows use a marked 2024 ONS 18–64 fallback; DLA/PIP are not official SAMHI components.",
            },
        }
    )


# ---------------------------------------------------------------------------
# SAMHI Machine Learning Forecast Page & API
# ---------------------------------------------------------------------------

_ML_PREDICTIONS_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_ML_METRICS_CACHE: dict[int, pd.DataFrame] = {}
_ML_SHAP_CACHE: dict[tuple[str, int, str], pd.DataFrame] = {}
_PUBLIC_SAMHI3_CACHE: dict[str, pd.DataFrame] = {}
_PUBLIC_SAMHI3_METRICS_CACHE: pd.DataFrame | None = None
_PUBLIC_SAMHI3_SHAP_CACHE: dict[str, pd.DataFrame] = {}

# Model-specific explanation files. Baselines do not have learned feature
# drivers, so the page reports their explanation as not applicable.
ML_SHAP_SPECS = {
    "pred_multimodal_elasticnet": ("elasticnet", "ElasticNet", "LinearSHAP"),
    "pred_multimodal_ridge": ("ridge", "Ridge", "LinearSHAP"),
    "pred_multimodal_random_forest": ("random_forest", "Random Forest", "TreeSHAP"),
    "pred_multimodal_extra-trees": ("extra-trees", "Extra-Trees", "TreeSHAP"),
    "pred_multimodal_lightgbm": ("lightgbm", "LightGBM", "TreeSHAP"),
    "pred_multimodal_xgboost": ("xgboost", "XGBoost", "TreeSHAP"),
    "pred_multimodal_catboost": ("catboost", "CatBoost", "TreeSHAP"),
    "pred_explainable_boosting_machine_(ebm)": ("explainable_boosting_machine_(ebm)", "EBM", "Permutation importance"),
    "pred_stacking_ensemble_(super_learner)": ("stacking_ensemble_(super_learner)", "Stacking Ensemble", "Permutation importance"),
}


LINCOLNSHIRE_LADS_SET = {
    "Boston",
    "East Lindsey",
    "Lincoln",
    "North Kesteven",
    "South Holland",
    "South Kesteven",
    "West Lindsey",
}

NOTTINGHAMSHIRE_LADS_SET = {
    "Ashfield", "Bassetlaw", "Broxtowe", "Gedling", "Mansfield",
    "Newark and Sherwood", "Nottingham", "Rushcliffe",
}
DERBYSHIRE_LADS_SET = {
    "Amber Valley", "Bolsover", "Chesterfield", "Derby", "Derbyshire Dales",
    "Erewash", "High Peak", "North East Derbyshire", "South Derbyshire",
}
NORTH_LINCOLNSHIRE_LADS_SET = {"North Lincolnshire"}
NORTH_EAST_LINCOLNSHIRE_LADS_SET = {"North East Lincolnshire"}
GREATER_LINCOLNSHIRE_LADS_SET = (
    LINCOLNSHIRE_LADS_SET | NORTH_LINCOLNSHIRE_LADS_SET | NORTH_EAST_LINCOLNSHIRE_LADS_SET
)
PUBLIC_SAMHI3_SCOPE_LADS = {
    "lincolnshire": LINCOLNSHIRE_LADS_SET,
    "north_lincolnshire": NORTH_LINCOLNSHIRE_LADS_SET,
    "north_east_lincolnshire": NORTH_EAST_LINCOLNSHIRE_LADS_SET,
    "greater_lincolnshire": GREATER_LINCOLNSHIRE_LADS_SET,
    "nottinghamshire": NOTTINGHAMSHIRE_LADS_SET,
    "derbyshire": DERBYSHIRE_LADS_SET,
    "lincolnshire_nottinghamshire_derbyshire": (
        LINCOLNSHIRE_LADS_SET | NOTTINGHAMSHIRE_LADS_SET | DERBYSHIRE_LADS_SET
    ),
}


def _normalise_public_samhi3_scope(scope: str) -> str:
    clean_scope = str(scope or "").strip().lower()
    return clean_scope if clean_scope in {"national", *PUBLIC_SAMHI3_SCOPE_LADS} else "lincolnshire"


def _get_public_samhi3_df(scope: str = "lincolnshire") -> pd.DataFrame:
    """Load dashboard-ready Public SAMHI-3 rows expanded to 2021 LSOAs."""
    clean_scope = _normalise_public_samhi3_scope(scope)
    if clean_scope in _PUBLIC_SAMHI3_CACHE:
        return _PUBLIC_SAMHI3_CACHE[clean_scope]
    path = _dashboard_data_path(PUBLIC_SAMHI3_DATA_DIR, "public_samhi3_web_predictions.csv")
    lookup_path = DATASETS_DIR / "lsoa" / "lsoa_2011_to_2021_lookup" / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    if not path.exists() or not lookup_path.exists():
        _PUBLIC_SAMHI3_CACHE[clean_scope] = pd.DataFrame()
        return _PUBLIC_SAMHI3_CACHE[clean_scope]
    df = pd.read_csv(path, low_memory=False)
    df["lsoa11"] = df["lsoa11"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    lookup = pd.read_csv(lookup_path, usecols=["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22NM"], dtype=str)
    lookup = lookup.rename(columns={"LSOA11CD": "lsoa11", "LSOA21CD": "LSOA21CD", "LSOA21NM": "LSOA21NM", "LAD22NM": "LAD22NM"})
    for column in ["lsoa11", "LSOA21CD", "LSOA21NM", "LAD22NM"]:
        lookup[column] = lookup[column].astype(str).str.strip()
    lookup = lookup.drop_duplicates("LSOA21CD")
    if clean_scope in PUBLIC_SAMHI3_SCOPE_LADS:
        lookup = lookup[lookup["LAD22NM"].isin(PUBLIC_SAMHI3_SCOPE_LADS[clean_scope])]
    expanded = lookup.merge(df, on="lsoa11", how="inner")
    expanded = expanded.drop_duplicates(["LSOA21CD", "year"], keep="last")
    rural_path = DATASETS_DIR / "rural_urban_classification_2021_lsoa" / "Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv"
    if rural_path.exists():
        rural = pd.read_csv(rural_path, usecols=["LSOA21CD", "RUC21NM", "Urban_rural_flag"], dtype=str)
        rural["LSOA21CD"] = rural["LSOA21CD"].astype(str).str.strip()
        rural = rural.drop_duplicates("LSOA21CD")
        expanded = expanded.merge(rural, on="LSOA21CD", how="left")
    _PUBLIC_SAMHI3_CACHE[clean_scope] = expanded
    return expanded


def _get_public_samhi3_metrics() -> pd.DataFrame:
    global _PUBLIC_SAMHI3_METRICS_CACHE
    if _PUBLIC_SAMHI3_METRICS_CACHE is None:
        path = _dashboard_data_path(PUBLIC_SAMHI3_DATA_DIR, "model_summary.csv")
        _PUBLIC_SAMHI3_METRICS_CACHE = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return _PUBLIC_SAMHI3_METRICS_CACHE


def _get_public_samhi3_shap(model: str = "BayesianRidge") -> pd.DataFrame:
    """Load model-specific global SHAP impacts for the Public SAMHI-3 page."""
    allowed = {"BayesianRidge", "ElasticNet", "Ridge", "RandomForest", "ExtraTrees", "LightGBM", "XGBoost", "CatBoost"}
    clean_model = model if model in allowed else "BayesianRidge"
    if clean_model not in _PUBLIC_SAMHI3_SHAP_CACHE:
        path = _dashboard_data_path(PUBLIC_SAMHI3_DATA_DIR, f"public_samhi3_shap_{clean_model.lower()}.csv")
        _PUBLIC_SAMHI3_SHAP_CACHE[clean_model] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return _PUBLIC_SAMHI3_SHAP_CACHE[clean_model]


def _get_ml_predictions_df(scope: str = "lincolnshire", experiment_set: int = 1) -> pd.DataFrame:
    clean_scope = _normalise_public_samhi3_scope(scope)
    source_scope = "lincolnshire" if clean_scope == "lincolnshire" else "national"
    experiment_set = 2 if experiment_set == 2 else 1
    cache_key = (clean_scope, experiment_set)
    if cache_key not in _ML_PREDICTIONS_CACHE:
        suffix = "pre_covid_2018_2019" if experiment_set == 2 else "2020_2022"
        csv_path = _dashboard_data_path(
            SAMHI_FORECAST_DATA_DIR,
            f"multimodal_predictions_{suffix}_{source_scope}.csv",
        )
        if not csv_path.exists():
            csv_path = _dashboard_data_path(
                SAMHI_FORECAST_DATA_DIR,
                f"baseline_predictions_{suffix}_{source_scope}.csv",
            )
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["lsoa11"] = df["lsoa11"].astype(str).str.strip()

            # Expand 2011 predictions to all 2021 LSOA boundaries (handling split LSOAs)
            lookup_path = (
                DATASETS_DIR
                / "lsoa"
                / "lsoa_2011_to_2021_lookup"
                / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
            )
            if lookup_path.exists():
                lookup_df = pd.read_csv(lookup_path)
                lookup_df["LSOA11CD"] = lookup_df["LSOA11CD"].astype(str).str.strip()
                lookup_df["LSOA21CD"] = lookup_df["LSOA21CD"].astype(str).str.strip()
                lookup_df["LSOA21NM"] = lookup_df["LSOA21NM"].astype(str).str.strip()
                lookup_df["LAD22NM"] = lookup_df["LAD22NM"].astype(str).str.strip()

                lookup_21 = lookup_df.drop_duplicates(subset=["LSOA21CD"]).copy()
                if clean_scope in PUBLIC_SAMHI3_SCOPE_LADS:
                    lookup_21 = lookup_21[
                        lookup_21["LAD22NM"].isin(PUBLIC_SAMHI3_SCOPE_LADS[clean_scope])
                    ]

                pred_data_cols = [
                    c for c in df.columns
                    if c not in ["LSOA21CD", "LSOA21NM", "LAD22NM", "ObjectId"]
                ]
                expanded = lookup_21[["LSOA11CD", "LSOA21CD", "LSOA21NM", "LAD22NM"]].merge(
                    df[pred_data_cols],
                    left_on="LSOA11CD",
                    right_on="lsoa11",
                    how="inner"
                )
                _ML_PREDICTIONS_CACHE[cache_key] = expanded
            else:
                df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip()
                if "LAD22NM" in df.columns:
                    df["LAD22NM"] = df["LAD22NM"].astype(str).str.strip()
                    if clean_scope in PUBLIC_SAMHI3_SCOPE_LADS:
                        df = df[df["LAD22NM"].isin(PUBLIC_SAMHI3_SCOPE_LADS[clean_scope])]
                _ML_PREDICTIONS_CACHE[cache_key] = df
        else:
            _ML_PREDICTIONS_CACHE[cache_key] = pd.DataFrame()
    return _ML_PREDICTIONS_CACHE[cache_key]


def _get_ml_metrics_df(experiment_set: int = 1) -> pd.DataFrame:
    experiment_set = 2 if experiment_set == 2 else 1
    if experiment_set not in _ML_METRICS_CACHE:
        suffix = "pre_covid_2018_2019" if experiment_set == 2 else "2020_2022"
        csv_path = _dashboard_data_path(
            SAMHI_FORECAST_DATA_DIR,
            f"multimodal_metrics_comparison_{suffix}.csv",
        )
        if not csv_path.exists():
            csv_path = _dashboard_data_path(
                SAMHI_FORECAST_DATA_DIR,
                f"baseline_metrics_comparison_{suffix}.csv",
            )
        _ML_METRICS_CACHE[experiment_set] = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return _ML_METRICS_CACHE[experiment_set]


def _get_ml_shap_df(
    scope: str = "lincolnshire",
    experiment_set: int = 1,
    model: str = "pred_multimodal_elasticnet",
) -> pd.DataFrame:
    clean_scope = "lincolnshire" if scope == "lincolnshire" else "national"
    experiment_set = 2 if experiment_set == 2 else 1
    shap_spec = ML_SHAP_SPECS.get(model)
    shap_model = shap_spec[0] if shap_spec else "not_applicable"
    cache_key = (clean_scope, experiment_set, shap_model)
    if cache_key not in _ML_SHAP_CACHE:
        if shap_spec is None:
            _ML_SHAP_CACHE[cache_key] = pd.DataFrame()
        else:
            suffix = "_pre_covid_2018_2019" if experiment_set == 2 else "_2020_2022"
            csv_path = _dashboard_data_path(
                SAMHI_FORECAST_DATA_DIR,
                f"shap_feature_importance_{shap_model}{suffix}_{clean_scope}.csv",
            )
            # Preserve compatibility with the legacy LightGBM output.
            if shap_model == "lightgbm" and not csv_path.exists():
                csv_path = _dashboard_data_path(
                    SAMHI_FORECAST_DATA_DIR,
                    f"shap_feature_importance{suffix}_{clean_scope}.csv",
                )
            _ML_SHAP_CACHE[cache_key] = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return _ML_SHAP_CACHE[cache_key]


_ML_PROJECTIONS_CACHE: dict[str, pd.DataFrame] = {}
_ML_PROJ_SUMMARY_CACHE: pd.DataFrame | None = None


def _get_ml_projections_df(scope: str = "lincolnshire") -> pd.DataFrame:
    clean_scope = _normalise_public_samhi3_scope(scope)
    source_scope = "lincolnshire" if clean_scope == "lincolnshire" else "national"
    if clean_scope not in _ML_PROJECTIONS_CACHE:
        # Prefer the extended projection file, retaining the prior dataset as
        # a compatibility fallback for checkouts that have not updated data.
        csv_path = _dashboard_data_path(
            SAMHI_FORECAST_DATA_DIR,
            f"forward_projections_2023_2027_{source_scope}.csv",
        )
        if not csv_path.exists():
            csv_path = _dashboard_data_path(
                SAMHI_FORECAST_DATA_DIR,
                f"forward_projections_2023_2025_{source_scope}.csv",
            )
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip()
            if "year" in df.columns:
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
            if "LAD22NM" in df.columns:
                df["LAD22NM"] = df["LAD22NM"].astype(str).str.strip()
                if clean_scope in PUBLIC_SAMHI3_SCOPE_LADS:
                    df = df[df["LAD22NM"].isin(PUBLIC_SAMHI3_SCOPE_LADS[clean_scope])]
            # Keep the API one-to-one at the map key level even if an older
            # result file contains duplicate LSOA-year rows.
            if {"LSOA21CD", "year"}.issubset(df.columns):
                df = df.drop_duplicates(subset=["LSOA21CD", "year"], keep="last")
            _ML_PROJECTIONS_CACHE[clean_scope] = df
        else:
            _ML_PROJECTIONS_CACHE[clean_scope] = pd.DataFrame()
    return _ML_PROJECTIONS_CACHE[clean_scope]


def _get_ml_proj_summary_df() -> pd.DataFrame:
    global _ML_PROJ_SUMMARY_CACHE
    if _ML_PROJ_SUMMARY_CACHE is None:
        csv_path = _dashboard_data_path(SAMHI_FORECAST_DATA_DIR, "forward_projections_summary.csv")
        if csv_path.exists():
            _ML_PROJ_SUMMARY_CACHE = pd.read_csv(csv_path)
        else:
            _ML_PROJ_SUMMARY_CACHE = pd.DataFrame()
    return _ML_PROJ_SUMMARY_CACHE


@app.get("/api/samhi_ml/export.png")
def samhi_ml_export_png():
    """Render the selected SAMHI map as a publication-quality high-resolution PNG."""
    try:
        import math
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import requests
        from PIL import Image
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
        from matplotlib.patheffects import withStroke
    except ImportError as exc:
        return jsonify({"error": f"Publication map export dependencies are unavailable: {exc}"}), 503

    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    try:
        year = int(request.args.get("year", 2025))
    except (TypeError, ValueError):
        year = 2025
    try:
        experiment_set = int(request.args.get("experiment_set", 1))
    except (TypeError, ValueError):
        experiment_set = 1
    experiment_set = 2 if experiment_set == 2 else 1
    model = request.args.get("model", "pred_multimodal_elasticnet").strip()
    display_mode = request.args.get("display_mode", "pred").strip().lower()
    district = request.args.get("district", "All").strip()
    try:
        dpi = int(request.args.get("dpi", 300))
    except (TypeError, ValueError):
        dpi = 300
    dpi = max(150, min(dpi, 600))
    basemap = request.args.get("basemap", "osm").strip().lower()
    if basemap not in {"osm", "none"}:
        basemap = "osm"
    output_format = request.args.get("format", "png").strip().lower()
    if output_format not in {"png", "pdf"}:
        return jsonify({"error": "Export format must be png or pdf."}), 400

    allowed_models = set(ML_SHAP_SPECS) | {
        "pred_ar_baseline_persistence",
        "pred_ar_baseline_momentum_drift",
    }
    if model not in allowed_models:
        return jsonify({"error": "Unknown SAMHI model selected."}), 400
    if display_mode not in {"pred", "actual", "err", "change", "direction"}:
        return jsonify({"error": "Unknown map display mode selected."}), 400

    is_projection = year >= 2023
    if is_projection and display_mode in {"actual", "err"}:
        return jsonify({"error": "Actual SAMHI and prediction error are unavailable for future projections."}), 400

    df = _get_ml_projections_df(scope) if is_projection else _get_ml_predictions_df(scope, experiment_set)
    if df.empty or "year" not in df.columns or "LSOA21CD" not in df.columns:
        return jsonify({"error": "No SAMHI data are available for this export."}), 404

    df_yr = df[df["year"] == year].copy()
    if district and district != "All" and "LAD22NM" in df_yr.columns:
        df_yr = df_yr[df_yr["LAD22NM"] == district]
    if df_yr.empty:
        return jsonify({"error": "No SAMHI data match the selected year or district."}), 404

    df_yr["LSOA21CD"] = df_yr["LSOA21CD"].astype(str).str.strip()
    if {"LSOA21CD", "year"}.issubset(df_yr.columns):
        df_yr = df_yr.drop_duplicates(subset=["LSOA21CD", "year"], keep="last")

    if model not in df_yr.columns:
        return jsonify({"error": f"No prediction column is available for {model}."}), 404

    prediction = pd.to_numeric(df_yr[model], errors="coerce")
    if is_projection:
        model_suffix = model.removeprefix("pred_")
        change_col = f"projected_change_{model_suffix}_vs_2022"
        if change_col not in df_yr.columns:
            change_col = "projected_change_vs_2022"
        change = pd.to_numeric(df_yr.get(change_col), errors="coerce")
        if display_mode == "pred":
            map_values = prediction
        else:
            map_values = change
    else:
        actual = pd.to_numeric(df_yr.get("actual_samhi"), errors="coerce")
        error_col = model.replace("pred_", "err_", 1)
        error = pd.to_numeric(df_yr.get(error_col), errors="coerce")
        lag_1 = pd.to_numeric(df_yr.get("lag_1"), errors="coerce")
        change = actual - lag_1
        if display_mode == "pred":
            map_values = prediction
        elif display_mode == "actual":
            map_values = actual
        elif display_mode == "err":
            map_values = error
        else:
            map_values = change

    map_data = df_yr[["LSOA21CD"]].copy()
    map_data["_map_value"] = pd.to_numeric(map_values, errors="coerce").to_numpy()
    if "LAD22NM" in df_yr.columns:
        map_data["_district_name"] = df_yr["LAD22NM"].astype(str).to_numpy()
    else:
        map_data["_district_name"] = ""
    map_data = map_data.dropna(subset=["_map_value"]).drop_duplicates("LSOA21CD", keep="last")
    if map_data.empty:
        return jsonify({"error": "No valid numeric values are available for this export."}), 404

    geojson_path = DATASETS_DIR / PUBLIC_SAMHI3_GEOJSON_PATHS[scope]
    if not geojson_path.exists():
        return jsonify({"error": "The selected LSOA boundary file is unavailable."}), 404

    try:
        boundaries = gpd.read_file(geojson_path)
    except Exception as exc:
        return jsonify({"error": f"The LSOA boundary file could not be read: {exc}"}), 500

    geo_code_column = next(
        (column for column in ("LSOA21CD", "CODE", "lsoa21cd") if column in boundaries.columns),
        None,
    )
    if geo_code_column is None:
        return jsonify({"error": "The selected boundary file has no LSOA code column."}), 500
    boundaries["_map_code"] = boundaries[geo_code_column].astype(str).str.strip()
    boundaries = boundaries.merge(
        map_data,
        left_on="_map_code",
        right_on="LSOA21CD",
        how="left",
    )
    boundaries = boundaries[
        boundaries.geometry.notna()
        & ~boundaries.geometry.is_empty
        & boundaries["_map_value"].notna()
    ].copy()
    valid_values = boundaries["_map_value"].dropna().to_numpy(dtype=float)
    if valid_values.size == 0:
        return jsonify({"error": "No selected SAMHI values overlap the boundary file."}), 404

    sorted_values = sorted(valid_values.tolist())
    p5 = sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.05))]
    p95 = sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))]

    if display_mode == "direction":
        abs_max = max(abs(p5), abs(p95), 0.10)
        cmap = LinearSegmentedColormap.from_list(
            "samhi_direction",
            ["#1a9850", "#d9ef8b", "#ffffbf", "#fee08b", "#d73027"],
        )
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        map_label = "Direction of Change"
    elif display_mode in {"err", "change"}:
        abs_max = max(abs(p5), abs(p95), 0.15)
        cmap = LinearSegmentedColormap.from_list(
            "samhi_change",
            ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"],
        )
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        map_label = "Prediction Error" if display_mode == "err" else "Change vs 2022"
    else:
        if p5 == p95:
            p5 -= 0.5
            p95 += 0.5
        cmap = LinearSegmentedColormap.from_list(
            "samhi_level",
            ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
        )
        norm = Normalize(vmin=p5, vmax=p95, clip=True)
        map_label = "Actual SAMHI" if display_mode == "actual" else ("Projected SAMHI" if is_projection else "Predicted SAMHI")

    model_labels = {
        "pred_multimodal_elasticnet": "Multimodal ElasticNet",
        "pred_multimodal_ridge": "Multimodal Ridge",
        "pred_multimodal_random_forest": "Multimodal Random Forest",
        "pred_multimodal_extra-trees": "Multimodal Extra-Trees",
        "pred_multimodal_lightgbm": "Multimodal LightGBM",
        "pred_multimodal_xgboost": "Multimodal XGBoost",
        "pred_multimodal_catboost": "Multimodal CatBoost",
        "pred_explainable_boosting_machine_(ebm)": "Explainable Boosting Machine",
        "pred_stacking_ensemble_(super_learner)": "Stacking Ensemble",
        "pred_ar_baseline_persistence": "Persistence baseline",
        "pred_ar_baseline_momentum_drift": "Momentum drift baseline",
    }
    model_label = model_labels.get(model, model)
    scope_label = {
        "lincolnshire": "Lincolnshire",
        "north_lincolnshire": "North Lincolnshire",
        "north_east_lincolnshire": "North East Lincolnshire",
        "greater_lincolnshire": "Greater Lincolnshire",
        "nottinghamshire": "Nottinghamshire",
        "derbyshire": "Derbyshire",
        "lincolnshire_nottinghamshire_derbyshire": "Lincolnshire, Nottinghamshire and Derbyshire",
        "national": "England",
    }[scope]
    title_prefix = "Projected Small Area Mental Health Index (SAMHI)" if is_projection else "Small Area Mental Health Index (SAMHI)"

    # Reproject the LSOA boundaries to Web Mercator so they align with OSM tiles.
    if boundaries.crs is None:
        boundaries = boundaries.set_crs(epsg=4326)
    boundaries_wgs84 = boundaries.to_crs(epsg=4326)
    boundaries_web = boundaries.to_crs(epsg=3857)

    osm_image = None
    osm_extent = None
    basemap_used = "none"
    if basemap == "osm":
        try:
            # Use a higher tile zoom for Lincolnshire so OSM place names are larger
            # and more legible in a 300-DPI publication figure.
            osm_zoom = 12 if scope == "lincolnshire" else (7 if scope == "national" else 10)
            lon_min, lat_min, lon_max, lat_max = boundaries_wgs84.total_bounds
            lat_min = max(-85.05112878, min(85.05112878, float(lat_min)))
            lat_max = max(-85.05112878, min(85.05112878, float(lat_max)))
            tile_count = 1 << osm_zoom

            def lon_to_tile_x(lon: float) -> float:
                return (lon + 180.0) / 360.0 * tile_count

            def lat_to_tile_y(lat: float) -> float:
                latitude = math.radians(lat)
                return (1.0 - math.asinh(math.tan(latitude)) / math.pi) / 2.0 * tile_count

            x_min = max(0, int(math.floor(lon_to_tile_x(lon_min))))
            x_max = min(tile_count - 1, int(math.floor(lon_to_tile_x(lon_max))))
            y_min = max(0, int(math.floor(lat_to_tile_y(lat_max))))
            y_max = min(tile_count - 1, int(math.floor(lat_to_tile_y(lat_min))))
            tile_size = 256
            osm_image = Image.new("RGB", ((x_max - x_min + 1) * tile_size, (y_max - y_min + 1) * tile_size))

            for tile_x in range(x_min, x_max + 1):
                for tile_y in range(y_min, y_max + 1):
                    tile_url = f"https://tile.openstreetmap.org/{osm_zoom}/{tile_x}/{tile_y}.png"
                    tile_response = requests.get(
                        tile_url,
                        headers={"User-Agent": "LUMHR-SAMHI-publication-map/1.0"},
                        timeout=15,
                    )
                    tile_response.raise_for_status()
                    tile = Image.open(BytesIO(tile_response.content)).convert("RGB")
                    osm_image.paste(tile, ((tile_x - x_min) * tile_size, (tile_y - y_min) * tile_size))

            earth_half_circumference = 20037508.342789244

            def tile_x_to_web_mercator(tile_x: int) -> float:
                return tile_x / tile_count * 40075016.68557849 - earth_half_circumference

            def tile_y_to_web_mercator(tile_y: int) -> float:
                return earth_half_circumference * (1.0 - 2.0 * tile_y / tile_count)

            osm_extent = (
                tile_x_to_web_mercator(x_min),
                tile_x_to_web_mercator(x_max + 1),
                tile_y_to_web_mercator(y_max + 1),
                tile_y_to_web_mercator(y_min),
            )
            basemap_used = "OpenStreetMap"
        except Exception:
            # A publication export should still work if the deployment cannot reach
            # the public tile server; the footer identifies the fallback output.
            osm_image = None
            osm_extent = None

    figure, axis = plt.subplots(figsize=(14, 10) if scope == "national" else (12, 9))
    if osm_image is not None and osm_extent is not None:
        axis.imshow(osm_image, extent=osm_extent, origin="upper", zorder=0)
    boundaries_web.plot(
        ax=axis,
        column="_map_value",
        cmap=cmap,
        norm=norm,
        linewidth=0.12,
        edgecolor="#ffffff",
        alpha=0.78 if osm_image is not None else 1.0,
        zorder=2,
    )

    # Add larger district labels above the basemap and LSOA layer. OSM town and
    # village names remain visible from the higher-zoom raster tiles underneath.
    if scope != "national" and "_district_name" in boundaries_web.columns:
        for district_name, district_boundaries in boundaries_web.groupby("_district_name"):
            if not district_name or district_name == "nan":
                continue
            try:
                label_point = district_boundaries.geometry.union_all().representative_point()
            except AttributeError:
                label_point = district_boundaries.geometry.unary_union.representative_point()
            axis.annotate(
                district_name,
                xy=(label_point.x, label_point.y),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#17313d",
                zorder=4,
                path_effects=[withStroke(linewidth=3, foreground="white", alpha=0.9)],
            )

    axis.set_axis_off()
    if output_format != "pdf":
        axis.set_title(
            f"{title_prefix} scores across {scope_label} LSOAs in {year}",
            fontsize=17,
            fontweight="bold",
            pad=18,
        )
        axis.text(
            0.5,
            1.01,
            f"{model_label} | {map_label} | {len(map_data):,} mapped LSOAs",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#425563",
        )

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(valid_values)
    colorbar = figure.colorbar(scalar_mappable, ax=axis, fraction=0.035, pad=0.02, shrink=0.78)
    colorbar.set_label(map_label, fontsize=10)
    if output_format != "pdf":
        basemap_note = (
            "contributors; "
            if basemap_used == "OpenStreetMap"
            else "OpenStreetMap basemap unavailable; "
        )
        figure.text(
            0.5,
            0.015,
            basemap_note
            + "LSOA 2021 boundaries; values are from the SAMHI machine-learning forecast data. "
            + f"Rendered at {dpi} DPI.",
            ha="center",
            fontsize=8,
            color="#5d717d",
        )
        figure.subplots_adjust(left=0.02, right=0.91, top=0.88, bottom=0.06)
    else:
        # The PDF is deliberately limited to the map and colour legend for use as
        # a paper figure; no title, model subtitle or footer is included.
        figure.subplots_adjust(left=0.01, right=0.91, top=0.99, bottom=0.01)

    image = BytesIO()
    figure.savefig(
        image,
        format=output_format,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05 if output_format == "pdf" else 0.1,
        facecolor="white",
    )
    plt.close(figure)
    image.seek(0)

    safe_model = "".join(character if character.isalnum() else "_" for character in model.removeprefix("pred_"))
    safe_mode = "".join(character if character.isalnum() else "_" for character in display_mode)
    extension = output_format
    mimetype = "application/pdf" if output_format == "pdf" else "image/png"
    filename = f"samhi_{scope}_{year}_{safe_model}_{safe_mode}_{dpi}dpi.{extension}"
    response = send_file(image, mimetype=mimetype, as_attachment=True, download_name=filename)
    response.headers["X-SAMHI-Basemap"] = basemap_used
    return response


@app.get("/samhi_forecast")
def samhi_forecast_page() -> str:
    return render_template(
        "samhi_forecast.html",
        geojson_by_scope=PUBLIC_SAMHI3_GEOJSON_PATHS,
        test_years=[2020, 2021, 2022],
        pre_covid_test_years=[2018, 2019],
        projection_years=[2023, 2024, 2025, 2026, 2027],
        default_year=2023,
    )


@app.get("/public_samhi3")
def public_samhi3_page() -> str:
    return render_template(
        "public_samhi3.html",
        geojson_by_scope=PUBLIC_SAMHI3_GEOJSON_PATHS,
    )


@app.get("/api/public_samhi3")
def public_samhi3_api():
    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    try:
        year = int(request.args.get("year", 2023))
    except (TypeError, ValueError):
        year = 2023
    if year < 2011 or year > 2027:
        return jsonify({"error": "Public SAMHI-3 year must be between 2011 and 2027."}), 400
    district = request.args.get("district", "All").strip()
    model_names = ["bayesianridge", "elasticnet", "ridge", "randomforest", "extratrees", "lightgbm", "xgboost", "catboost"]
    model = request.args.get("model", "bayesianridge").strip().lower()
    if model not in model_names:
        model = "bayesianridge"
    try:
        comparison_year = int(request.args.get("comparison_year", 2020))
    except (TypeError, ValueError):
        comparison_year = 2020
    if comparison_year < 2011 or comparison_year > 2027:
        return jsonify({"error": "Comparison year must be between 2011 and 2027."}), 400
    df = _get_public_samhi3_df(scope)
    if df.empty:
        return jsonify({"error": "Public SAMHI-3 dashboard data are unavailable. Run the web prediction export first."}), 404
    df_year = df[df["year"].eq(year)].copy()
    previous_rows = df[df["year"].eq(year - 1)][["LSOA21CD", f"pred_{model}", "public_samhi3_proxy"]].drop_duplicates("LSOA21CD")
    previous_forecasts = previous_rows.set_index("LSOA21CD")[f"pred_{model}"]
    previous_proxies = previous_rows.set_index("LSOA21CD")["public_samhi3_proxy"]
    comparison_rows = df[df["year"].eq(comparison_year)].copy()
    comparison_column = f"pred_{model}" if comparison_year >= 2026 else "public_samhi3_proxy"
    comparison_values = comparison_rows.drop_duplicates("LSOA21CD").set_index("LSOA21CD")[comparison_column]
    comparison_proxy_values = comparison_rows.drop_duplicates("LSOA21CD").set_index("LSOA21CD")["public_samhi3_proxy"]
    model_comparison_column = f"pred_{model}" if comparison_year >= 2017 else "public_samhi3_proxy"
    model_comparison_values = comparison_rows.drop_duplicates("LSOA21CD").set_index("LSOA21CD")[model_comparison_column]
    # Public SAMHI-3 is a z-score proxy while published SAMHI is on its own
    # index scale.  Compare them only after within-year standardisation.
    proxy_values = pd.to_numeric(df_year["public_samhi3_proxy"], errors="coerce")
    official_values = pd.to_numeric(df_year["samhi_index"], errors="coerce")
    proxy_std = proxy_values.std(ddof=0)
    official_std = official_values.std(ddof=0)
    if proxy_std and official_std and pd.notna(proxy_std) and pd.notna(official_std):
        df_year["proxy_official_std_error"] = (
            (proxy_values - proxy_values.mean()) / proxy_std
            - (official_values - official_values.mean()) / official_std
        )
    else:
        df_year["proxy_official_std_error"] = float("nan")
    df_year["proxy_official_raw_difference"] = proxy_values - official_values
    if district and district != "All":
        df_year = df_year[df_year["LAD22NM"].eq(district)]
        previous_forecasts = previous_forecasts.loc[
            previous_forecasts.index.isin(df_year["LSOA21CD"])
        ]
        comparison_values = comparison_values.loc[
            comparison_values.index.isin(df_year["LSOA21CD"])
        ]
        previous_proxies = previous_proxies.loc[
            previous_proxies.index.isin(df_year["LSOA21CD"])
        ]
        model_comparison_values = model_comparison_values.loc[
            model_comparison_values.index.isin(df_year["LSOA21CD"])
        ]
        comparison_proxy_values = comparison_proxy_values.loc[
            comparison_proxy_values.index.isin(df_year["LSOA21CD"])
        ]
    districts = ["All", *sorted(df["LAD22NM"].dropna().astype(str).unique().tolist())]
    prediction_columns = [
        *(f"pred_{name}" for name in model_names),
        *(bound for name in model_names for bound in (f"ci_lower_{name}", f"ci_upper_{name}")),
    ]
    records = {}
    for _, row in df_year.iterrows():
        code = str(row.get("LSOA21CD", "")).strip()
        if not code:
            continue
        model_prediction = _num_or_none(row.get(f"pred_{model}"))
        previous_model_prediction = _num_or_none(previous_forecasts.get(code))
        previous_proxy = _num_or_none(previous_proxies.get(code))
        current_comparison_value = model_prediction if year >= 2026 else _num_or_none(row.get("public_samhi3_proxy"))
        comparison_value = _num_or_none(comparison_values.get(code))
        current_proxy = _num_or_none(row.get("public_samhi3_proxy"))
        proxy_comparison_value = _num_or_none(comparison_proxy_values.get(code))
        model_comparison_value = _num_or_none(model_comparison_values.get(code))
        previous_model_change = (
            model_prediction - previous_model_prediction
            if model_prediction is not None and previous_model_prediction is not None
            else model_prediction - previous_proxy
            if model_prediction is not None and previous_proxy is not None
            else None
        )
        model_change_baseline = (
            "model estimate"
            if model_prediction is not None and previous_model_prediction is not None
            else "component proxy"
            if model_prediction is not None and previous_proxy is not None
            else None
        )
        proxy_comparison_difference = (
            current_proxy - proxy_comparison_value
            if current_proxy is not None and proxy_comparison_value is not None and year >= comparison_year
            else proxy_comparison_value - current_proxy
            if current_proxy is not None and proxy_comparison_value is not None
            else None
        )
        model_comparison_difference = (
            model_prediction - model_comparison_value
            if model_prediction is not None and model_comparison_value is not None and year >= comparison_year
            else model_comparison_value - model_prediction
            if model_prediction is not None and model_comparison_value is not None
            else None
        )
        item = {
            "code": code,
            "name": str(row.get("LSOA21NM", "") or ""),
            "district": str(row.get("LAD22NM", "") or ""),
            "ruc21nm": str(row.get("RUC21NM", "") or ""),
            "urban_rural": str(row.get("Urban_rural_flag", "") or ""),
            "year": year,
            "proxy": _num_or_none(row.get("public_samhi3_proxy")),
            "proxy_2plus": _num_or_none(row.get("public_samhi3_proxy_2plus")),
            "n_components": _num_or_none(row.get("public_samhi3_n_components")),
            "official_samhi": _num_or_none(row.get("samhi_index")),
            "proxy_official_std_error": _num_or_none(row.get("proxy_official_std_error")),
            "proxy_official_raw_difference": _num_or_none(row.get("proxy_official_raw_difference")),
            "proxy_lag1": _num_or_none(row.get("public_samhi3_proxy_lag1")),
            "proxy_change": _num_or_none(row.get("proxy_change")),
            "model_prediction_change": previous_model_change,
            "model_prediction_change_baseline": model_change_baseline,
            "comparison_year": comparison_year,
            "comparison_difference": (
                current_comparison_value - comparison_value
                if current_comparison_value is not None and comparison_value is not None and year >= comparison_year
                else comparison_value - current_comparison_value
                if current_comparison_value is not None and comparison_value is not None
                else None
            ),
            "proxy_comparison_difference": proxy_comparison_difference,
            "model_comparison_difference": model_comparison_difference,
        }
        for column in prediction_columns:
            item[column] = _num_or_none(row.get(column))
        records[code] = item
    metrics = _get_public_samhi3_metrics()
    metric_records = []
    if not metrics.empty:
        metric_records = [
            _json_safe(record)
            for record in metrics[metrics["target"].eq("PublicSAMHI3_HistoryPlusComponents")].to_dict(orient="records")
        ]
    long_range_projection = year >= 2026
    return jsonify({
        "scope": scope,
        "year": year,
        "model": model,
        "comparison_year": comparison_year,
        "district": district,
        "total_lsoas": len(records),
        "districts": districts,
        "predictions": records,
        "metrics": metric_records,
        "meta": {
            "label": "Public SAMHI-3 proxy",
            "warning": (
                "2026–2027 are exploratory projections using the latest component data available (through 2025). The 2027 score builds on the 2026 projected score; no observed data for those years are used."
                if long_range_projection
                else "This is a public-data proxy built from antidepressant prescribing, QOF depression and DLA/PIP. It excludes the hospital-attendance component and is not official SAMHI."
            ),
            "source": "Public SAMHI-3 component reconstruction",
            "method": "The observed proxy is the equal-weight mean of available components standardised against 2011–2022 reference statistics. Model forecasts use lagged components and recursively predicted index history.",
            "is_projection": year >= 2023,
            "long_range_projection": long_range_projection,
            "observed_proxy_available": year <= 2025,
            "data_sources": {
                "antidepressants": "OpenPrescribing / NHS prescribing data",
                "qof_depression": "NHS England Quality and Outcomes Framework (practice data allocated to LSOAs)",
                "dla_pip": "DWP Stat-Xplore DLA and PIP claims",
                "benchmark": "PLDR published SAMHI (2011–2022)",
            },
            "hospital_component": "Unavailable in public data; not imputed",
        },
    })


@app.get("/api/public_samhi3/history")
def public_samhi3_history_api():
    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    code = request.args.get("code", "").strip()
    if not code:
        return jsonify({"error": "An LSOA code is required."}), 400
    model_names = {"bayesianridge", "elasticnet", "ridge", "randomforest", "extratrees", "lightgbm", "xgboost", "catboost"}
    model = request.args.get("model", "bayesianridge").strip().lower()
    if model not in model_names:
        model = "bayesianridge"
    df = _get_public_samhi3_df(scope)
    if df.empty:
        return jsonify({"error": "Public SAMHI-3 dashboard data are unavailable."}), 404
    area = df[df["LSOA21CD"].eq(code)].sort_values("year")
    if area.empty:
        return jsonify({"error": "No SAMHI history is available for this LSOA."}), 404
    prediction_column = f"pred_{model}"
    series = []
    for _, row in area.iterrows():
        year = _num_or_none(row.get("year"))
        if year is None:
            continue
        series.append({
            "year": int(year),
            "samhi": _num_or_none(row.get("samhi_index")),
            "samhi3": _num_or_none(row.get("public_samhi3_proxy")),
            "prediction": _num_or_none(row.get(prediction_column)),
        })
    return jsonify({"code": code, "model": model, "series": series})


@app.get("/api/public_samhi3/shap")
def public_samhi3_shap_api():
    model = request.args.get("model", "BayesianRidge").strip()
    allowed = {"BayesianRidge", "ElasticNet", "Ridge", "RandomForest", "ExtraTrees", "LightGBM", "XGBoost", "CatBoost"}
    if model not in allowed:
        model = "BayesianRidge"
    shap_df = _get_public_samhi3_shap(model)
    if shap_df.empty:
        return jsonify({"model": model, "method": "Unavailable", "features": [], "error": "SHAP output has not been generated yet."}), 404
    features = [_json_safe(record) for record in shap_df.to_dict(orient="records")]
    domains = (
        shap_df.groupby("domain", as_index=False)["importance_pct"]
        .sum()
        .sort_values("importance_pct", ascending=False)
    )
    return jsonify({
        "model": model,
        "method": str(shap_df["method"].iloc[0]),
        "features": features,
        "domain_summary": [_json_safe(record) for record in domains.to_dict(orient="records")],
        "meta": {
            "target": "Public SAMHI-3 proxy",
            "note": "Global mean absolute SHAP impact from models trained on the 2011–2022 panel; impacts are descriptive, not causal.",
        },
    })


@app.get("/api/samhi_ml/history")
def samhi_ml_history_api():
    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    code = request.args.get("code", "").strip()
    if not code:
        return jsonify({"error": "An LSOA code is required."}), 400
    allowed_models = {
        "pred_multimodal_ridge", "pred_multimodal_elasticnet", "pred_multimodal_random_forest",
        "pred_multimodal_extra-trees", "pred_multimodal_lightgbm", "pred_multimodal_xgboost",
        "pred_multimodal_catboost", "pred_explainable_boosting_machine_(ebm)",
        "pred_stacking_ensemble_(super_learner)", "pred_ar_baseline_persistence",
        "pred_ar_baseline_momentum_drift",
    }
    model = request.args.get("model", "pred_multimodal_elasticnet").strip()
    if model not in allowed_models:
        model = "pred_multimodal_elasticnet"

    actual_df = _get_public_samhi3_df(scope)
    actual_rows = actual_df[actual_df["LSOA21CD"].eq(code)] if not actual_df.empty else pd.DataFrame()
    actual_by_year = {}
    if not actual_rows.empty:
        for _, row in actual_rows.iterrows():
            year = _num_or_none(row.get("year"))
            value = _num_or_none(row.get("samhi_index"))
            if year is not None and value is not None:
                actual_by_year[int(year)] = value

    prediction_by_year = {}
    for experiment_set in (2, 1):
        predictions = _get_ml_predictions_df(scope, experiment_set)
        if predictions.empty or model not in predictions.columns:
            continue
        area_predictions = predictions[predictions["LSOA21CD"].eq(code)]
        for _, row in area_predictions.iterrows():
            year = _num_or_none(row.get("year"))
            if year is not None:
                prediction_by_year[int(year)] = _num_or_none(row.get(model))

    projections = _get_ml_projections_df(scope)
    if not projections.empty and model in projections.columns:
        area_projections = projections[projections["LSOA21CD"].eq(code)]
        for _, row in area_projections.iterrows():
            year = _num_or_none(row.get("year"))
            if year is not None:
                prediction_by_year[int(year)] = _num_or_none(row.get(model))

    years = sorted(set(actual_by_year) | set(prediction_by_year))
    if not years:
        return jsonify({"error": "No SAMHI history is available for this LSOA."}), 404
    return jsonify({
        "code": code,
        "model": model,
        "series": [
            {
                "year": year,
                "actual": actual_by_year.get(year),
                "prediction": prediction_by_year.get(year),
                "is_projection": year >= 2023,
            }
            for year in years
        ],
    })


@app.get("/api/samhi_ml/predictions")
def samhi_ml_predictions_api():
    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    try:
        year = int(request.args.get("year", 2023))
    except (ValueError, TypeError):
        year = 2023
    district = request.args.get("district", "All").strip()
    try:
        experiment_set = int(request.args.get("experiment_set", 1))
    except (ValueError, TypeError):
        experiment_set = 1
    experiment_set = 2 if experiment_set == 2 else 1

    is_projection = year >= 2023
    df = _get_ml_projections_df(scope) if is_projection else _get_ml_predictions_df(scope, experiment_set)
    if df.empty:
        return jsonify({"error": f"No prediction/projection data found for scope: {scope}"}), 404

    df_yr = df[df["year"] == year].copy()
    if district and district != "All" and "LAD22NM" in df_yr.columns:
        df_yr = df_yr[df_yr["LAD22NM"] == district]

    # Get available districts for dropdown
    all_districts = ["All"]
    if "LAD22NM" in df.columns:
        all_districts.extend(sorted(df["LAD22NM"].dropna().unique().tolist()))

    records = {}
    pred_cols = [c for c in df_yr.columns if c.startswith("pred_")]
    err_cols = [c for c in df_yr.columns if c.startswith("err_")]

    for _, row in df_yr.iterrows():
        code = str(row["LSOA21CD"]).strip()
        item = {
            "code": code,
            "name": str(row.get("LSOA21NM", "")),
            "district": str(row.get("LAD22NM", "")),
            "is_projection": is_projection,
        }

        if is_projection:
            item["actual"] = None
            item["baseline_2022"] = _num_or_none(row.get("baseline_2022"))
            item["lag_1"] = _num_or_none(row.get("pred_ar_baseline_persistence"))
            item["projected_change_vs_2022"] = _num_or_none(row.get("projected_change_vs_2022"))
            item["projected_annual_change"] = _num_or_none(row.get("projected_annual_change"))
            item["projected_decile"] = int(row.get("projected_decile", 5))
            item["ci_lower_95"] = _num_or_none(row.get("ci_lower_95"))
            item["ci_upper_95"] = _num_or_none(row.get("ci_upper_95"))
            item["horizon_years"] = int(row.get("horizon_years", 1))
            item["projection_model"] = str(row.get("projection_model", ""))
            item["actual_change"] = item["projected_change_vs_2022"]
            item["prediction_error_available"] = False
            for c in df_yr.columns:
                if c.startswith(("projected_decile_", "ci_lower_", "ci_upper_")):
                    item[c] = _num_or_none(row.get(c))
        else:
            actual_val = _num_or_none(row.get("actual_samhi"))
            lag1_val = _num_or_none(row.get("lag_1"))
            item["actual"] = actual_val
            item["lag_1"] = lag1_val
            item["delta_1"] = _num_or_none(row.get("delta_1"))
            item["actual_change"] = (actual_val - lag1_val) if actual_val is not None and lag1_val is not None else None
            item["prediction_error_available"] = True

        for c in pred_cols:
            item[c] = _num_or_none(row.get(c))
        for c in err_cols:
            item[c] = _num_or_none(row.get(c))

        records[code] = item

    return jsonify({
        "scope": scope,
        "year": year,
        "experiment_set": experiment_set,
        "is_projection": is_projection,
        "district": district,
        "total_lsoas": len(records),
        "districts": all_districts,
        "predictions": records
    })


@app.get("/api/samhi_ml/metrics")
def samhi_ml_metrics_api():
    scope = _normalise_public_samhi3_scope(request.args.get("scope", "lincolnshire"))
    try:
        experiment_set = int(request.args.get("experiment_set", 1))
    except (ValueError, TypeError):
        experiment_set = 1
    experiment_set = 2 if experiment_set == 2 else 1
    model = request.args.get("model", "pred_multimodal_elasticnet").strip()
    if not model.startswith("pred_"):
        model = "pred_multimodal_elasticnet"
    metrics_df = _get_ml_metrics_df(experiment_set)
    shap_df = _get_ml_shap_df(scope, experiment_set, model)
    proj_summary_df = _get_ml_proj_summary_df()

    filtered_metrics = []
    if not metrics_df.empty:
        if "Scope" in metrics_df.columns:
            metric_scope = "lincolnshire" if scope == "lincolnshire" else "national"
            m_sub = metrics_df[metrics_df["Scope"] == metric_scope]
            if m_sub.empty and metric_scope == "lincolnshire":
                m_sub = metrics_df[metrics_df["Scope"].str.contains("linc", case=False, na=False)]
            if m_sub.empty:
                m_sub = metrics_df
            filtered_metrics = m_sub.to_dict(orient="records")
        else:
            filtered_metrics = metrics_df.to_dict(orient="records")

    filtered_proj_summary = []
    if not proj_summary_df.empty and "Scope" in proj_summary_df.columns:
        summary_scope = "lincolnshire" if scope == "lincolnshire" else "national"
        ps_sub = proj_summary_df[proj_summary_df["Scope"] == summary_scope]
        filtered_proj_summary = ps_sub.to_dict(orient="records")

    shap_records = []
    domain_summary = []
    if not shap_df.empty:
        shap_records = shap_df.to_dict(orient="records")
        if "domain" in shap_df.columns and "importance_pct" in shap_df.columns:
            domain_summary = (
                shap_df.groupby("domain")["importance_pct"]
                .sum()
                .reset_index()
                .to_dict(orient="records")
            )

    return jsonify({
        "scope": scope,
        "experiment_set": experiment_set,
        "shap_model": ML_SHAP_SPECS.get(model, (None, "Not applicable", ""))[1],
        "shap_method": ML_SHAP_SPECS.get(model, (None, "Not applicable", ""))[2],
        "metrics": filtered_metrics,
        "shap_features": shap_records,
        "domain_summary": domain_summary,
        "projections_summary": filtered_proj_summary
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)

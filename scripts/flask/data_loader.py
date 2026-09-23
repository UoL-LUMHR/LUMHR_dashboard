from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.allocation import (
    allocate_access_rates_to_lsoa,
    allocate_registers_to_lsoa,
    build_gp_master,
    build_gp_marker_df,
    prepare_depression,
    prepare_gp_locations,
    prepare_mapping_from_sex_split,
    prepare_prescribing,
    prepare_smi,
)
from core.common import find_column, normalize_code, parse_numeric
from core.samhi import join_samhi, prepare_samhi
from core.travel import (
    build_2011_to_2021_lookup,
    prepare_car_availability,
    prepare_digital_exclusion,
    prepare_imd_2025,
    prepare_lsoac_classification,
    prepare_population_estimates,
    prepare_rural_urban,
    prepare_travel_times,
    prepare_ts003_household_composition,
)


def resolve_base_dir(script_file: Path) -> Path:
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


def get_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "depression": base_dir / "datasets" / "quality_outcomes_framework" / "qof_depression_2425_lincolnshire.csv",
        "smi": base_dir / "datasets" / "quality_outcomes_framework" / "qof_mental_health_2425_lincolnshire.csv",
        "prescribing": base_dir
        / "datasets"
        / "gp_prescribing_data"
        / "items_for_antidepressant_drugs_per_gp_lincolnshire_may_2026.csv",
        "mapping_male": base_dir
        / "datasets"
        / "patients_registered_gp_practice"
        / "july_2026"
        / "gp-reg-pat-prac-lsoa-male.csv",
        "mapping_female": base_dir
        / "datasets"
        / "patients_registered_gp_practice"
        / "july_2026"
        / "gp-reg-pat-prac-lsoa-female.csv",
        "gp_locations": base_dir / "datasets" / "gp_locations" / "Lincolnshire_ICB_GPs.csv",
        "lsoa_geo": base_dir / "datasets" / "lincolnshire_lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson",
        "samhi": base_dir / "datasets" / "samhi" / "samhi_lincolnshire_2021_lsoa.csv",
        "gp_travel_time": base_dir
        / "datasets"
        / "journey_time_statistics"
        / "jts0505 LSOA Travel Time to GPs_sheet_2019.csv",
        "hospital_travel_time": base_dir
        / "datasets"
        / "journey_time_statistics"
        / "jts0506 LSOA Travel Time to Hospitals_sheet_2019.csv",
        "rural_urban": base_dir
        / "datasets"
        / "rural_urban_classification_2021_lsoa"
        / "lincolnshire_rural_urban_2021.csv",
        "car_van": base_dir
        / "datasets"
        / "car_or_van_availability"
        / "TS045-2021-4-filtered-2026-08-24T10_54_11Z.csv",
        "deri": base_dir
        / "datasets"
        / "digital_exclusion_risk_index"
        / "deri_lincolnshire_2021_lsoa.csv",
        "population": base_dir
        / "datasets"
        / "population_estimates"
        / "lincolnshire_lsoa_population_estimates_2024.csv",
        "imd_2025": base_dir
        / "datasets"
        / "indices_of_deprivation_imd"
        / "2025"
        / "lincolnshire_imd_2025_lsoa.csv",
        "lsoac_2021_2": base_dir
        / "datasets"
        / "lsoa_classification_2021_2"
        / "lincolnshire_lsoa_classification_2021_2.csv",
        "ts003_household_composition": base_dir
        / "datasets"
        / "TS003_household_composition"
        / "lincolnshire_ts003_household_composition.csv",
        "fuel_poverty": base_dir
        / "datasets"
        / "fuel_poverty"
        / "2024"
        / "lincolnshire_fuel_poverty_2024.csv",
        "properties_not_connected_to_gas_network": base_dir
        / "datasets"
        / "properties_not_connected_to_gas_network"
        / "lincolnshire_properties_not_connected_to_gas_network_2024.csv",
        "lsoa_2011_2021_lookup": base_dir
        / "datasets"
        / "lincolnshire_lsoa"
        / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv",
    }


def _polygon_centroid_area(ring: list[list[float]]) -> tuple[float, float, float]:
    if not ring or len(ring) < 3:
        return 0.0, np.nan, np.nan

    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        cross = (x0 * y1) - (x1 * y0)
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    if np.isclose(area2, 0.0):
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))

    area = area2 / 2.0
    cx = cx / (3.0 * area2)
    cy = cy / (3.0 * area2)
    return float(area), float(cx), float(cy)


def _feature_centroid(geometry: dict[str, object]) -> tuple[float, float]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    best_abs_area = -1.0
    best_lon = np.nan
    best_lat = np.nan

    polygons: list[list[list[float]]] = []
    if gtype == "Polygon":
        polygons = coords
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                polygons.append(poly[0])

    for ring in polygons:
        area, lon, lat = _polygon_centroid_area(ring)
        abs_area = abs(area)
        if abs_area > best_abs_area:
            best_abs_area = abs_area
            best_lon = lon
            best_lat = lat

    return float(best_lat), float(best_lon)


def _load_lsoa_codes_and_centroids(geojson_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(geojson_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    code_rows: list[dict[str, str]] = []
    centroid_rows: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        code = normalize_code(props.get("CODE") or props.get("LSOA_CODE") or props.get("LSOA21CD"))
        if not code:
            continue
        code_rows.append(
            {
                "LSOA_CODE": code,
                "LSOA21NM": str(props.get("NAME") or props.get("LSOA21NM") or "").strip(),
            }
        )
        geometry = feature.get("geometry", {}) or {}
        lat, lon = _feature_centroid(geometry)
        centroid_rows.append({"LSOA_CODE": code, "lat": lat, "lon": lon})

    lsoa_codes_df = pd.DataFrame(code_rows).drop_duplicates(subset=["LSOA_CODE"])
    if lsoa_codes_df.empty:
        raise ValueError(f"No LSOA codes found in GeoJSON: {geojson_path}")

    lsoa_centroids_df = pd.DataFrame(centroid_rows).drop_duplicates(subset=["LSOA_CODE"])
    return lsoa_codes_df, lsoa_centroids_df


def _prepare_energy_metric(
    raw_df: pd.DataFrame,
    code_candidates: list[str],
    value_candidates: list[str],
    output_column: str,
) -> pd.DataFrame:
    code_column = find_column(raw_df, code_candidates)
    value_column = find_column(raw_df, value_candidates)
    out = raw_df[[code_column, value_column]].copy()
    out["LSOA_CODE"] = out[code_column].map(normalize_code)
    out[output_column] = parse_numeric(out[value_column])
    return out[["LSOA_CODE", output_column]].drop_duplicates(subset=["LSOA_CODE"])


def load_raw_data(base_dir_str: str) -> dict[str, object]:
    base_dir = Path(base_dir_str)
    paths = get_paths(base_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found for {name}: {path}")

    lsoa_codes_df, lsoa_centroids_df = _load_lsoa_codes_and_centroids(paths["lsoa_geo"])

    return {
        "dep_raw": pd.read_csv(paths["depression"]),
        "smi_raw": pd.read_csv(paths["smi"]),
        "prescribing_raw": pd.read_csv(paths["prescribing"]),
        "mapping_male_raw": pd.read_csv(paths["mapping_male"]),
        "mapping_female_raw": pd.read_csv(paths["mapping_female"]),
        "gp_loc_raw": pd.read_csv(paths["gp_locations"]),
        "samhi_raw": pd.read_csv(paths["samhi"]),
        "gp_travel_time_raw": pd.read_csv(paths["gp_travel_time"]),
        "hospital_travel_time_raw": pd.read_csv(paths["hospital_travel_time"]),
        "rural_urban_raw": pd.read_csv(paths["rural_urban"]),
        "car_van_raw": pd.read_csv(paths["car_van"]),
        "deri_raw": pd.read_csv(paths["deri"]),
        "pop_raw": pd.read_csv(paths["population"]),
        "imd_2025_raw": pd.read_csv(paths["imd_2025"]),
        "lsoac_2021_2_raw": pd.read_csv(paths["lsoac_2021_2"]),
        "ts003_raw": pd.read_csv(paths["ts003_household_composition"]),
        "fuel_poverty_raw": pd.read_csv(paths["fuel_poverty"]),
        "properties_not_connected_to_gas_network_raw": pd.read_csv(
            paths["properties_not_connected_to_gas_network"]
        ),
        "lsoa_2011_2021_lookup_raw": pd.read_csv(paths["lsoa_2011_2021_lookup"]),
        "lsoa_codes": lsoa_codes_df,
        "lsoa_centroids": lsoa_centroids_df,
    }


@lru_cache(maxsize=2)
def get_prepared_bundle_cached(base_dir_str: str) -> dict[str, object]:
    raw = load_raw_data(base_dir_str)

    dep_df = prepare_depression(raw["dep_raw"])
    smi_df = prepare_smi(raw["smi_raw"])
    prescribing_df = prepare_prescribing(raw["prescribing_raw"])
    mapping_df = prepare_mapping_from_sex_split(raw["mapping_male_raw"], raw["mapping_female_raw"])
    gp_loc_df = prepare_gp_locations(raw["gp_loc_raw"])
    samhi_df = prepare_samhi(raw["samhi_raw"])

    lsoa_codes_df: pd.DataFrame = raw["lsoa_codes"]
    lsoa_centroids_df: pd.DataFrame = raw["lsoa_centroids"]
    in_area_lsoa_codes = set(lsoa_codes_df["LSOA_CODE"].dropna().unique())

    gp_master = build_gp_master(dep_df, smi_df, prescribing_df)
    mapped_lsoa, mismatch_summary = allocate_registers_to_lsoa(mapping_df, gp_master)

    out_of_area_lsoa_codes = set(mapped_lsoa["LSOA_CODE"].dropna().unique()) - in_area_lsoa_codes
    out_of_area_rows = mapping_df[mapping_df["LSOA_CODE"].isin(out_of_area_lsoa_codes)]
    out_of_area_patients = float(out_of_area_rows["NUMBER_OF_PATIENTS"].sum()) if not out_of_area_rows.empty else 0.0

    lsoa_metrics = lsoa_codes_df.merge(mapped_lsoa, on="LSOA_CODE", how="left")
    lsoa_metrics = join_samhi(lsoa_metrics, samhi_df)

    access_lsoa = allocate_access_rates_to_lsoa(mapping_df, gp_master)
    lsoa_metrics = lsoa_metrics.merge(access_lsoa, on="LSOA_CODE", how="left")

    lookup_map = build_2011_to_2021_lookup(raw["lsoa_2011_2021_lookup_raw"])
    travel_times_df = prepare_travel_times(
        raw["gp_travel_time_raw"], raw["hospital_travel_time_raw"], lookup_map
    )
    lsoa_metrics = lsoa_metrics.merge(travel_times_df, on="LSOA_CODE", how="left")

    rural_urban_df = prepare_rural_urban(raw["rural_urban_raw"])
    lsoa_metrics = lsoa_metrics.merge(rural_urban_df, on="LSOA_CODE", how="left")

    car_van_df = prepare_car_availability(raw["car_van_raw"])
    lsoa_metrics = lsoa_metrics.merge(car_van_df, on="LSOA_CODE", how="left")

    deri_df = prepare_digital_exclusion(raw["deri_raw"])
    lsoa_metrics = lsoa_metrics.merge(deri_df, on="LSOA_CODE", how="left")

    pop_df = prepare_population_estimates(raw["pop_raw"])
    lsoa_metrics = lsoa_metrics.merge(pop_df, on="LSOA_CODE", how="left")

    imd_df = prepare_imd_2025(raw["imd_2025_raw"])
    lsoa_metrics = lsoa_metrics.merge(imd_df, on="LSOA_CODE", how="left")

    lsoac_df = prepare_lsoac_classification(raw["lsoac_2021_2_raw"])
    lsoa_metrics = lsoa_metrics.merge(lsoac_df, on="LSOA_CODE", how="left")

    ts003_df = prepare_ts003_household_composition(raw["ts003_raw"])
    lsoa_metrics = lsoa_metrics.merge(ts003_df, on="LSOA_CODE", how="left")

    fuel_poverty_df = _prepare_energy_metric(
        raw["fuel_poverty_raw"],
        ["LSOA Code"],
        ["Proportion of households fuel poor (%)"],
        "Fuel_Poverty_Pct",
    )
    lsoa_metrics = lsoa_metrics.merge(fuel_poverty_df, on="LSOA_CODE", how="left")

    gas_network_df = _prepare_energy_metric(
        raw["properties_not_connected_to_gas_network_raw"],
        ["LSOA code"],
        ["Estimated percentage of properties not on the gas grid"],
        "Properties_Not_On_Gas_Grid_Pct",
    )
    lsoa_metrics = lsoa_metrics.merge(gas_network_df, on="LSOA_CODE", how="left")

    gp_marker_df = build_gp_marker_df(gp_loc_df, gp_master, mapping_df, lsoa_centroids_df, in_area_lsoa_codes)

    return {
        "dep_df": dep_df,
        "smi_df": smi_df,
        "prescribing_df": prescribing_df,
        "mapping_df": mapping_df,
        "gp_loc_df": gp_loc_df,
        "gp_master": gp_master,
        "lsoa_metrics": lsoa_metrics,
        "gp_marker_df": gp_marker_df,
        "mismatch_summary": mismatch_summary,
        "out_of_area_lsoa_codes": out_of_area_lsoa_codes,
        "out_of_area_patients": out_of_area_patients,
    }

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import streamlit as st

from analysis.allocation import (
    allocate_registers_to_lsoa,
    build_gp_marker_df,
    build_gp_master,
    get_lsoa_centroids,
    get_lsoa_centroids_cached,
    prepare_depression,
    prepare_gp_locations,
    prepare_lsoa_geography,
    prepare_lsoa_geography_cached,
    prepare_lsoa_map_base_cached,
    prepare_mapping_from_sex_split,
    prepare_prescribing,
    prepare_smi,
)
from analysis.samhi import join_samhi, prepare_samhi


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
        "gp_locations": base_dir / "datasets" / "gp_locations" / "GP_Locations_epraccur.csv",
        "lsoa_geo": base_dir / "datasets" / "lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson",
        "samhi": base_dir / "datasets" / "samhi" / "samhi_lincolnshire_2021_lsoa.csv",
    }


def _normalise_postcode(value: object) -> str:
    return "".join(str(value or "").upper().split())


def _load_ods_gp_locations(path: Path, base_dir: Path) -> pd.DataFrame:
    locations = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False).fillna("")
    role = locations["RoleID"].astype(str).str.contains(r"(^|\|)RO76($|\|)", regex=True)
    active = locations["StatusName"].astype(str).str.strip().str.upper().eq("ACTIVE")
    locations = locations[role & active].drop_duplicates("OrganisationCode", keep="last").copy()

    postcode_key = locations["Postcode"].map(_normalise_postcode)
    wanted = set(postcode_key[postcode_key.ne("")])
    lookup_path = (
        base_dir
        / "scripts"
        / "utils"
        / "source_data"
        / "Postcode_to_OA_(2021)_to_LSOA_to_MSOA_to_LAD_(February_2026)_Best_Fit_Lookup_in_the_UK"
        / "PCD_OA21_LSOA21_MSOA21_LAD_FEB26_UK_LU.csv"
    )
    matches = []
    for chunk in pd.read_csv(
        lookup_path,
        usecols=["pcd7", "lsoa21cd", "dointr", "doterm"],
        dtype=str,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk["postcode_key"] = chunk["pcd7"].map(_normalise_postcode)
        active_postcodes = chunk["doterm"].isna() | chunk["doterm"].astype(str).str.strip().eq("")
        selected = chunk[active_postcodes & chunk["postcode_key"].isin(wanted)]
        if not selected.empty:
            matches.append(selected[["postcode_key", "lsoa21cd", "dointr"]])

    postcode_lookup = (
        pd.concat(matches, ignore_index=True)
        if matches
        else pd.DataFrame(columns=["postcode_key", "lsoa21cd", "dointr"])
    )
    postcode_lookup["dointr"] = pd.to_numeric(postcode_lookup["dointr"], errors="coerce")
    postcode_lookup = postcode_lookup.sort_values("dointr").drop_duplicates("postcode_key", keep="last")
    postcode_to_lsoa = postcode_lookup.set_index("postcode_key")["lsoa21cd"].to_dict()
    locations["LSOA"] = postcode_key.map(postcode_to_lsoa).fillna("")
    return locations


@st.cache_data(show_spinner=False)
def load_raw_data_cached(base_dir_str: str) -> dict[str, object]:
    base_dir = Path(base_dir_str)
    paths = get_paths(base_dir)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found for {name}: {path}")

    dep_df = pd.read_csv(paths["depression"])
    smi_df = pd.read_csv(paths["smi"])
    prescribing_df = pd.read_csv(paths["prescribing"])
    map_male_df = pd.read_csv(paths["mapping_male"])
    map_female_df = pd.read_csv(paths["mapping_female"])
    gp_loc_df = _load_ods_gp_locations(paths["gp_locations"], base_dir)
    samhi_df = pd.read_csv(paths["samhi"])

    with open(paths["lsoa_geo"], "r", encoding="utf-8") as f:
        gj = json.load(f)
    lsoa_gdf = gpd.GeoDataFrame.from_features(gj["features"])
    if lsoa_gdf.crs is None:
        lsoa_gdf = lsoa_gdf.set_crs("EPSG:4326")

    return {
        "dep_raw": dep_df,
        "smi_raw": smi_df,
        "prescribing_raw": prescribing_df,
        "mapping_male_raw": map_male_df,
        "mapping_female_raw": map_female_df,
        "gp_loc_raw": gp_loc_df,
        "lsoa_raw": lsoa_gdf,
        "samhi_raw": samhi_df,
    }


@st.cache_data(show_spinner=False)
def get_prepared_bundle_cached(base_dir_str: str) -> dict[str, object]:
    raw = load_raw_data_cached(base_dir_str)

    dep_df = prepare_depression(raw["dep_raw"])
    smi_df = prepare_smi(raw["smi_raw"])
    prescribing_df = prepare_prescribing(raw["prescribing_raw"])
    mapping_df = prepare_mapping_from_sex_split(raw["mapping_male_raw"], raw["mapping_female_raw"])
    gp_loc_df = prepare_gp_locations(raw["gp_loc_raw"])
    # Cache the static polygon preprocessing separately from the rest of the bundle.
    lsoa_gdf = prepare_lsoa_geography_cached(raw["lsoa_raw"])
    lsoa_map_base = prepare_lsoa_map_base_cached(raw["lsoa_raw"])
    samhi_df = prepare_samhi(raw["samhi_raw"])

    in_area_lsoa_codes = set(lsoa_gdf["LSOA_CODE"].dropna().unique())
    gp_master = build_gp_master(dep_df, smi_df, prescribing_df)
    mapped_lsoa, mismatch_summary = allocate_registers_to_lsoa(mapping_df, gp_master)

    out_of_area_lsoa_codes = set(mapped_lsoa["LSOA_CODE"].dropna().unique()) - in_area_lsoa_codes
    out_of_area_rows = mapping_df[mapping_df["LSOA_CODE"].isin(out_of_area_lsoa_codes)]
    out_of_area_patients = float(out_of_area_rows["NUMBER_OF_PATIENTS"].sum()) if not out_of_area_rows.empty else 0.0

    lsoa_metrics = lsoa_gdf.merge(mapped_lsoa, on="LSOA_CODE", how="left")
    lsoa_metrics = join_samhi(lsoa_metrics, samhi_df)

    lsoa_centroids = get_lsoa_centroids_cached(lsoa_gdf)
    gp_marker_df = build_gp_marker_df(gp_loc_df, gp_master, mapping_df, lsoa_centroids, in_area_lsoa_codes)

    return {
        "dep_df": dep_df,
        "smi_df": smi_df,
        "prescribing_df": prescribing_df,
        "mapping_df": mapping_df,
        "gp_loc_df": gp_loc_df,
        "gp_master": gp_master,
        "lsoa_metrics": lsoa_metrics,
        "lsoa_map_base": lsoa_map_base,
        "gp_marker_df": gp_marker_df,
        "mismatch_summary": mismatch_summary,
        "out_of_area_lsoa_codes": out_of_area_lsoa_codes,
        "out_of_area_patients": out_of_area_patients,
    }


@st.cache_data(show_spinner=False)
def get_comparison_bundle_cached(base_dir_str: str) -> dict[str, object]:
    """Load only the data needed by the Comparison page.

    This intentionally skips centroid and GP marker generation because those
    geospatial transforms are not required for comparison metrics.
    """
    raw = load_raw_data_cached(base_dir_str)

    dep_df = prepare_depression(raw["dep_raw"])
    smi_df = prepare_smi(raw["smi_raw"])
    prescribing_df = prepare_prescribing(raw["prescribing_raw"])
    mapping_df = prepare_mapping_from_sex_split(raw["mapping_male_raw"], raw["mapping_female_raw"])
    lsoa_gdf = prepare_lsoa_geography(raw["lsoa_raw"])
    samhi_df = prepare_samhi(raw["samhi_raw"])

    gp_master = build_gp_master(dep_df, smi_df, prescribing_df)
    mapped_lsoa, mismatch_summary = allocate_registers_to_lsoa(mapping_df, gp_master)

    lsoa_metrics = lsoa_gdf.merge(mapped_lsoa, on="LSOA_CODE", how="left")
    lsoa_metrics = join_samhi(lsoa_metrics, samhi_df)

    return {
        "lsoa_metrics": lsoa_metrics,
        "mismatch_summary": mismatch_summary,
    }

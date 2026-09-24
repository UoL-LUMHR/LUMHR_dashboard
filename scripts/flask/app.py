from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

from core.need_index import apply_need_index
from core.access_index import apply_access_index
from core.rural_risk_index import apply_rural_risk_index
from core.samhi import SAMHI_YEARS, get_samhi_columns
from data_loader import get_prepared_bundle_cached, resolve_base_dir

app = Flask(__name__, template_folder="templates")

BASE_DIR = resolve_base_dir(Path(__file__))
DATASETS_DIR = BASE_DIR / "datasets"
GEOJSON_REL_PATH = "lincolnshire_lsoa/lower-super-output-areas-2021-5RrVTw.geojson"
ENGLAND_GEOJSON_REL_PATH = "england_lsoa/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_6894679968818356315.geojson"
ML_RESULTS_DIR = BASE_DIR / "scripts" / "machine_learning" / "samhi" / "results"

# Load once at startup so API requests only do lightweight transforms.
BUNDLE = get_prepared_bundle_cached(str(BASE_DIR))
LSOA_METRICS = BUNDLE["lsoa_metrics"].copy()
GP_MARKER_DF = BUNDLE["gp_marker_df"].copy()


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
    out_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
    out_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    return render_template(
        "need_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        out_of_area_lsoa_count=out_lsoa_count,
        out_of_area_lsoa_count_fmt=f"{out_lsoa_count:,}",
        out_of_area_patients=out_patients,
        out_of_area_patients_fmt=f"{out_patients:,}",
    )


@app.get("/samhi")
def samhi_page() -> str:
    return render_template(
        "samhi.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        min_year=min(SAMHI_YEARS),
        max_year=max(SAMHI_YEARS),
        default_year=max(SAMHI_YEARS),
    )


@app.get("/access_index")
def access_index_page() -> str:
    out_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
    out_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    return render_template(
        "access_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        out_of_area_lsoa_count_fmt=f"{out_lsoa_count:,}",
        out_of_area_patients_fmt=f"{out_patients:,}",
    )


@app.get("/access_gap_index")
def access_gap_index_page() -> str:
    out_lsoa_count = len(BUNDLE["out_of_area_lsoa_codes"])
    out_patients = int(round(float(BUNDLE["out_of_area_patients"])))
    return render_template(
        "access_gap_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
        out_of_area_lsoa_count_fmt=f"{out_lsoa_count:,}",
        out_of_area_patients_fmt=f"{out_patients:,}",
        min_year=min(SAMHI_YEARS),
        max_year=max(SAMHI_YEARS),
    )


@app.get("/rural_risk_index")
def rural_risk_index_page() -> str:
    return render_template(
        "rural_risk_index.html",
        geojson_rel_path=GEOJSON_REL_PATH,
    )


@app.get("/datasets/<path:filename>")
def dataset_files(filename: str):
    return send_from_directory(DATASETS_DIR, filename)


@app.get("/api/gp_locations")
def gp_locations_api():
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
    available_cols = [c for c in cols if c in GP_MARKER_DF.columns]
    out = GP_MARKER_DF[available_cols].copy()
    out["Lat"] = pd.to_numeric(out.get("Lat"), errors="coerce")
    out["Lon"] = pd.to_numeric(out.get("Lon"), errors="coerce")
    out = out.dropna(subset=["Lat", "Lon"])

    markers: list[dict[str, object]] = []
    for row in out.to_dict(orient="records"):
        marker = {
            "practice_code": str(row.get("PRACTICE_CODE", "")),
            "practice_name": str(row.get("Practice_Name", "") or ""),
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
        dep = _parse_float_arg("dep", 25.0)
        smi = _parse_float_arg("smi", 25.0)
        prescribing = _parse_float_arg("prescribing", 25.0)
        samhi_weight = _parse_float_arg("samhi", 25.0)
        samhi_year = _parse_int_arg("samhi_year", max(SAMHI_YEARS))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if samhi_year not in SAMHI_YEARS:
        return jsonify({"error": f"samhi_year must be one of {SAMHI_YEARS}"}), 400

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    if samhi_index_col not in LSOA_METRICS.columns:
        return jsonify({"error": f"Missing SAMHI column: {samhi_index_col}"}), 400

    scored = apply_need_index(
        LSOA_METRICS,
        dep,
        smi,
        prescribing,
        samhi_weight,
        samhi_index_col,
    )
    scored["Depression_Prevalence_Pct"] = pd.to_numeric(scored["Depression_Prevalence"], errors="coerce") * 100.0
    scored["SMI_Prevalence_Pct"] = pd.to_numeric(scored["SMI_Prevalence"], errors="coerce") * 100.0

    layers = {
        "Need_Index": _scores_to_dict(scored, "Need_Index"),
        "Depression_Prevalence": _scores_to_dict(scored, "Depression_Prevalence"),
        "SMI_Prevalence": _scores_to_dict(scored, "SMI_Prevalence"),
        "Antidepressant_Items_Per_Patient": _scores_to_dict(scored, "Antidepressant_Items_Per_Patient"),
        "SAMHI_Selected": _scores_to_dict(scored, "SAMHI_Selected"),
        "Pct_65plus": _scores_to_dict(scored, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(scored, "GP_Registration_Rate_Pct"),
    }

    lsoa_name_col = _get_lsoa_name_column(scored)

    detail_cols = [
        "LSOA_CODE",
        "Need_Index",
        "Depression_Prevalence_Pct",
        "SMI_Prevalence_Pct",
        "Antidepressant_Items_Per_Patient",
        "SAMHI_Selected",
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
            "smi_prevalence_pct": _num_or_none(row.get("SMI_Prevalence_Pct")),
            "antidepressant_items_per_patient": _num_or_none(row.get("Antidepressant_Items_Per_Patient")),
            "samhi_selected": _num_or_none(row.get("SAMHI_Selected")),
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
                "samhi_label": f"SAMHI Index ({samhi_year})",
            },
        }
    )


@app.get("/api/access_scores")
def access_scores_api():
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
    missing = [c for c in required_cols if c not in LSOA_METRICS.columns]
    if missing:
        return jsonify({"error": f"Missing access columns: {', '.join(missing)}"}), 400

    scored = apply_access_index(
        LSOA_METRICS, mh002, mh021, mh_pca, dep_pca, dep004, gp_pt, gp_car, hosp_pt, hosp_car, rural, car, digital
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
    try:
        dep = _parse_float_arg("dep", 25.0)
        smi = _parse_float_arg("smi", 25.0)
        prescribing = _parse_float_arg("prescribing", 25.0)
        samhi_weight = _parse_float_arg("samhi", 25.0)
        samhi_year = _parse_int_arg("samhi_year", max(SAMHI_YEARS))
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

    samhi_index_col, _ = get_samhi_columns(samhi_year)
    if samhi_index_col not in LSOA_METRICS.columns:
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
    missing = [c for c in access_required_cols if c not in LSOA_METRICS.columns]
    if missing:
        return jsonify({"error": f"Missing access columns: {', '.join(missing)}"}), 400

    need_scored = apply_need_index(LSOA_METRICS, dep, smi, prescribing, samhi_weight, samhi_index_col)
    access_scored = apply_access_index(
        LSOA_METRICS, mh002, mh021, mh_pca, dep_pca, dep004, gp_pt, gp_car, hosp_pt, hosp_car, rural, car, digital
    )

    combined = need_scored[["LSOA_CODE", "Need_Index"]].merge(
        access_scored[["LSOA_CODE", "Access_Index"]], on="LSOA_CODE", how="outer"
    )
    combined["Access_Gap_Index"] = combined["Need_Index"] - combined["Access_Index"]

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
    extra_join_cols = [c for c in pop_cols if c in LSOA_METRICS.columns]
    if extra_join_cols:
        combined = combined.merge(LSOA_METRICS[["LSOA_CODE", *extra_join_cols]], on="LSOA_CODE", how="left")

    lsoa_name_col = _get_lsoa_name_column(LSOA_METRICS)
    if lsoa_name_col:
        combined = combined.merge(LSOA_METRICS[["LSOA_CODE", lsoa_name_col]], on="LSOA_CODE", how="left")

    layers = {
        "Access_Gap_Index": _scores_to_dict(combined, "Access_Gap_Index"),
        "Need_Index": _scores_to_dict(combined, "Need_Index"),
        "Access_Index": _scores_to_dict(combined, "Access_Index"),
        "Pct_65plus": _scores_to_dict(combined, "Pct_65plus"),
        "GP_Registration_Rate_Pct": _scores_to_dict(combined, "GP_Registration_Rate_Pct"),
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
            },
        }
    )


@app.get("/api/samhi_scores")
def samhi_scores_api():
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
        if selected_col not in LSOA_METRICS.columns:
            return jsonify({"error": f"Missing SAMHI column: {selected_col}"}), 400

        lsoa_name_col = _get_lsoa_name_column(LSOA_METRICS)
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
        cols.extend([c for c in pop_cols if c in LSOA_METRICS.columns])
        out_df = LSOA_METRICS[cols].copy()
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

    missing = [col for col in [from_col, to_col] if col not in LSOA_METRICS.columns]
    if missing:
        return jsonify({"error": f"Missing SAMHI columns: {', '.join(missing)}"}), 400

    lsoa_name_col = _get_lsoa_name_column(LSOA_METRICS)
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
    cols.extend([c for c in pop_cols if c in LSOA_METRICS.columns])
    out_df = LSOA_METRICS[cols].copy()
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
    try:
        rural_weight = _parse_float_arg("w_rural", 9.1)
        gp_pt_weight = _parse_float_arg("w_gp_pt", 9.1)
        gp_car_weight = _parse_float_arg("w_gp_car", 9.1)
        no_car_weight = _parse_float_arg("w_no_car", 9.1)
        imd_weight = _parse_float_arg("w_imd", 9.1)
        oac_weight = _parse_float_arg("w_oac", 9.1)
        household_weight = _parse_float_arg("w_household", 9.1)
        fuel_poverty_weight = _parse_float_arg("w_fuel_poverty", 9.1)
        off_gas_grid_weight = _parse_float_arg("w_off_gas_grid", 9.1)
        housing_tenure_weight = _parse_float_arg("w_housing_tenure", 9.1)
        overcrowding_weight = _parse_float_arg("w_overcrowding", 9.1)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scored = apply_rural_risk_index(
        LSOA_METRICS,
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
            },
        }
    )


# ---------------------------------------------------------------------------
# SAMHI Machine Learning Forecast Page & API
# ---------------------------------------------------------------------------

_ML_PREDICTIONS_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_ML_METRICS_CACHE: dict[int, pd.DataFrame] = {}
_ML_SHAP_CACHE: dict[tuple[str, int, str], pd.DataFrame] = {}

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


def _get_ml_predictions_df(scope: str = "lincolnshire", experiment_set: int = 1) -> pd.DataFrame:
    clean_scope = "lincolnshire" if scope == "lincolnshire" else "national"
    experiment_set = 2 if experiment_set == 2 else 1
    cache_key = (clean_scope, experiment_set)
    if cache_key not in _ML_PREDICTIONS_CACHE:
        suffix = "pre_covid_2018_2019" if experiment_set == 2 else "2020_2022"
        csv_path = ML_RESULTS_DIR / f"multimodal_predictions_{suffix}_{clean_scope}.csv"
        if not csv_path.exists():
            csv_path = ML_RESULTS_DIR / f"baseline_predictions_{suffix}_{clean_scope}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["lsoa11"] = df["lsoa11"].astype(str).str.strip()

            # Expand 2011 predictions to all 2021 LSOA boundaries (handling split LSOAs)
            lookup_path = (
                DATASETS_DIR
                / "lincolnshire_lsoa"
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
                if clean_scope == "lincolnshire":
                    lookup_21 = lookup_21[lookup_21["LAD22NM"].isin(LINCOLNSHIRE_LADS_SET)]

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
                _ML_PREDICTIONS_CACHE[cache_key] = df
        else:
            _ML_PREDICTIONS_CACHE[cache_key] = pd.DataFrame()
    return _ML_PREDICTIONS_CACHE[cache_key]


def _get_ml_metrics_df(experiment_set: int = 1) -> pd.DataFrame:
    experiment_set = 2 if experiment_set == 2 else 1
    if experiment_set not in _ML_METRICS_CACHE:
        suffix = "pre_covid_2018_2019" if experiment_set == 2 else "2020_2022"
        csv_path = ML_RESULTS_DIR / f"multimodal_metrics_comparison_{suffix}.csv"
        if not csv_path.exists():
            csv_path = ML_RESULTS_DIR / f"baseline_metrics_comparison_{suffix}.csv"
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
            csv_path = ML_RESULTS_DIR / f"shap_feature_importance_{shap_model}{suffix}_{clean_scope}.csv"
            # Preserve compatibility with the legacy LightGBM output.
            if shap_model == "lightgbm" and not csv_path.exists():
                csv_path = ML_RESULTS_DIR / f"shap_feature_importance{suffix}_{clean_scope}.csv"
            _ML_SHAP_CACHE[cache_key] = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return _ML_SHAP_CACHE[cache_key]


_ML_PROJECTIONS_CACHE: dict[str, pd.DataFrame] = {}
_ML_PROJ_SUMMARY_CACHE: pd.DataFrame | None = None


def _get_ml_projections_df(scope: str = "lincolnshire") -> pd.DataFrame:
    clean_scope = "lincolnshire" if scope == "lincolnshire" else "national"
    if clean_scope not in _ML_PROJECTIONS_CACHE:
        csv_path = ML_RESULTS_DIR / f"forward_projections_2023_2025_{clean_scope}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip()
            if "year" in df.columns:
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
            if "LAD22NM" in df.columns:
                df["LAD22NM"] = df["LAD22NM"].astype(str).str.strip()
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
        csv_path = ML_RESULTS_DIR / "forward_projections_summary.csv"
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

    scope = request.args.get("scope", "lincolnshire").strip().lower()
    scope = "lincolnshire" if scope == "lincolnshire" else "national"
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

    geojson_path = DATASETS_DIR / (
        GEOJSON_REL_PATH if scope == "lincolnshire" else ENGLAND_GEOJSON_REL_PATH
    )
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
    scope_label = "Lincolnshire" if scope == "lincolnshire" else "England"
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
            osm_zoom = 12 if scope == "lincolnshire" else 7
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

    figure, axis = plt.subplots(figsize=(12, 9) if scope == "lincolnshire" else (14, 10))
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
    if scope == "lincolnshire" and "_district_name" in boundaries_web.columns:
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
        geojson_lincolnshire=GEOJSON_REL_PATH,
        geojson_england=ENGLAND_GEOJSON_REL_PATH,
        test_years=[2020, 2021, 2022],
        pre_covid_test_years=[2018, 2019],
        projection_years=[2023, 2024, 2025],
        default_year=2023,
    )


@app.get("/api/samhi_ml/predictions")
def samhi_ml_predictions_api():
    scope = request.args.get("scope", "lincolnshire").strip().lower()
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
    scope = request.args.get("scope", "lincolnshire").strip().lower()
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
            m_sub = metrics_df[metrics_df["Scope"] == scope]
            if m_sub.empty and scope == "lincolnshire":
                m_sub = metrics_df[metrics_df["Scope"].str.contains("linc", case=False, na=False)]
            if m_sub.empty:
                m_sub = metrics_df
            filtered_metrics = m_sub.to_dict(orient="records")
        else:
            filtered_metrics = metrics_df.to_dict(orient="records")

    filtered_proj_summary = []
    if not proj_summary_df.empty and "Scope" in proj_summary_df.columns:
        ps_sub = proj_summary_df[proj_summary_df["Scope"] == scope]
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

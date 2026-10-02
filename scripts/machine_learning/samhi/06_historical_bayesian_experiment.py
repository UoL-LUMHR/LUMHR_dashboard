"""Leakage-aware rolling-origin SAMHI experiment.

This experiment is intentionally separate from the dashboard's existing
multimodal benchmark.  It uses only covariates that have a dated observation
and selects the most recent observation available before the target year.

The non-spatial Bayesian pilot uses ``sklearn.linear_model.BayesianRidge`` with
LSOA and year indicators.  A second pilot adds a conjugate Gaussian
CAR-style spatial random intercept from the supplied LSOA GeoJSON.  Both are
screening models before a full PyMC/INLA spatial random-effects analysis.

Run from the project root, for example::

    scripts/.venv/bin/python scripts/machine_learning/samhi/06_historical_bayesian_experiment.py \
        --scope lincolnshire

Outputs are written to ``results/historical_bayesian_v2`` by default.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("SAMHI_Historical_Bayesian")

HISTORY_FEATURES = [
    "lag_1", "lag_2", "lag_3", "delta_1", "delta_2", "acceleration",
    "rolling_mean_3yr", "rolling_std_3yr", "rolling_min_3yr", "rolling_max_3yr",
]
HISTORICAL_FEATURES = [
    "fuel_poverty_pct", "gp_pt_time", "gp_car_time", "hosp_pt_time",
    "hosp_car_time", "pop_total", "pop_under16_pct", "pop_working_age_pct",
    "pop_65_plus_pct", "qof_mh002_pct", "qof_mh021_pct", "qof_dep_achievement_pct",
]
ALL_NUMERIC_FEATURES = HISTORY_FEATURES + HISTORICAL_FEATURES


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalise_code(value: object) -> str:
    return str(value).split(":", 1)[0].strip()


def _find_header(path: Path) -> Optional[Tuple[str, int, List[str]]]:
    """Find the LSOA/fuel header in differently formatted Excel releases."""
    try:
        import openpyxl

        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in book.sheetnames:
                ws = book[sheet]
                for row_number, row in enumerate(ws.iter_rows(values_only=True)):
                    values = [str(v).strip() if v is not None else "" for v in row]
                    joined = " | ".join(values).lower()
                    if "lsoa code" in joined and (
                        "percent fuel" in joined or "proportion of households fuel poor" in joined
                    ):
                        return sheet, row_number, values
        finally:
            book.close()
    except Exception as exc:
        LOGGER.warning("Could not inspect %s: %s", path.name, exc)
    return None


def load_historical_fuel_poverty(source_root: Path) -> pd.DataFrame:
    """Load one dated LSOA fuel-poverty observation per year.

    The 2011/2012 releases contain duplicate definition workbooks.  Prefer
    the LIHC-labelled workbook where available; otherwise use the first valid
    workbook for that directory year.  Values are converted to percentage
    points when a release stores proportions in [0, 1].
    """
    rows: List[pd.DataFrame] = []
    base = source_root / "fuel_poverty"
    for year_dir in sorted(base.iterdir() if base.exists() else []):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        paths = sorted(p for p in year_dir.iterdir() if p.suffix.lower() in {".xlsx", ".xlsm"})
        paths.sort(key=lambda p: ("lihc" not in p.name.lower(), "v2" not in p.name.lower(), p.name.lower()))
        selected = None
        for path in paths:
            header = _find_header(path)
            if header is not None:
                selected = (path, header)
                break
        if selected is None:
            continue
        path, (sheet, header_row, header_values) = selected
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=header_row, engine="openpyxl")
            raw.columns = [str(c).strip() for c in raw.columns]
            code_col = next(c for c in raw.columns if c.lower() == "lsoa code")
            value_col = next(
                c for c in raw.columns
                if "percent fuel poor" in c.lower() or "proportion of households fuel poor" in c.lower()
            )
            out = pd.DataFrame({
                "lsoa11": raw[code_col].map(_normalise_code),
                "fuel_poverty_pct": pd.to_numeric(raw[value_col], errors="coerce"),
                "source_year": year,
            })
            out = out[out["lsoa11"].str.startswith("E", na=False)].copy()
            if out["fuel_poverty_pct"].dropna().median() <= 1.0:
                out["fuel_poverty_pct"] *= 100.0
            rows.append(out.groupby(["lsoa11", "source_year"], as_index=False)["fuel_poverty_pct"].mean())
            LOGGER.info("Loaded fuel poverty %s from %s (%s rows)", year, path.name, len(out))
        except Exception as exc:
            LOGGER.warning("Could not parse fuel poverty %s: %s", path, exc)
    if not rows:
        return pd.DataFrame(columns=["lsoa11", "source_year", "fuel_poverty_pct"])
    return pd.concat(rows, ignore_index=True)


def _travel_files(datasets_dir: Path, prefix: str) -> Iterable[Tuple[int, Path]]:
    for path in sorted(datasets_dir.joinpath("journey_time_statistics").glob(f"{prefix}*.csv")):
        match = re.search(r"_(\d{4})(?:_REVISED)?\.csv$", path.name)
        if match:
            yield int(match.group(1)), path


def load_historical_travel(datasets_dir: Path) -> pd.DataFrame:
    """Load cleaned annual GP and hospital travel-time CSVs.

    These files are already restricted to the 2011 LSOA geography and can be
    joined directly to SAMHI's ``lsoa11`` key.
    """
    frames: List[pd.DataFrame] = []
    for year, path in _travel_files(datasets_dir, "jts0505 LSOA Travel Time to GPs"):
        raw = pd.read_csv(path)
        cols = {"lsoa11": "LSOA_code", "gp_pt_time": "GPPTt", "gp_car_time": "GPCart"}
        if not set(cols.values()).issubset(raw.columns):
            continue
        out = raw[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
        out["source_year"] = year
        frames.append(out)
    for year, path in _travel_files(datasets_dir, "jts0506 LSOA Travel Time to Hospitals"):
        raw = pd.read_csv(path)
        cols = {"lsoa11": "LSOA_code", "hosp_pt_time": "HospPTt", "hosp_car_time": "HospCart"}
        if not set(cols.values()).issubset(raw.columns):
            continue
        out = raw[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
        out["source_year"] = year
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["lsoa11", "source_year", *HISTORICAL_FEATURES[1:5]])
    out = pd.concat(frames, ignore_index=True)
    for col in ["gp_pt_time", "gp_car_time", "hosp_pt_time", "hosp_car_time"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.groupby(["lsoa11", "source_year"], as_index=False)[HISTORICAL_FEATURES[1:5]].mean()


def _population_candidates(source_root: Path) -> Dict[int, Path]:
    """Choose one consolidated 2021-geography population workbook per year."""
    candidates: Dict[int, List[Path]] = {}
    for base in source_root.glob("lsoa_population_estimates*"):
        if not base.is_dir():
            continue
        for path in base.rglob("*.xlsx"):
            if not path.name.lower().startswith("sapelsoasyoa"):
                continue
            try:
                book = __import__("openpyxl").load_workbook(path, read_only=True, data_only=True)
                years = []
                for sheet in book.sheetnames:
                    match = re.fullmatch(r"Mid-(\d{4}) LSOA 2021", str(sheet))
                    if match:
                        years.append(int(match.group(1)))
                book.close()
                for year in years:
                    candidates.setdefault(year, []).append(path)
            except Exception as exc:
                LOGGER.warning("Could not inspect population workbook %s: %s", path.name, exc)
    # Prefer the short consolidated path; duplicate 2022–2024 copies contain
    # the same revised release and should not create duplicate panel rows.
    return {year: sorted(paths, key=lambda p: (len(str(p)), str(p)))[0] for year, paths in candidates.items()}


def load_historical_population(source_root: Path, min_year: int | None = None, max_year: int | None = None) -> pd.DataFrame:
    """Load annual persons-by-age estimates for 2021 LSOAs.

    The source folder also contains legacy 2011-geography XLS files.  The
    consolidated XLSX releases cover mid-2011 onward on 2021 LSOA codes,
    which is the geography used by the SAMHI lookup in this experiment.
    """
    import openpyxl

    rows: List[pd.DataFrame] = []
    candidates = _population_candidates(source_root)
    for year, path in sorted(candidates.items()):
        if min_year is not None and year < min_year:
            continue
        if max_year is not None and year > max_year:
            continue
        try:
            book = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet = f"Mid-{year} LSOA 2021"
            if sheet not in book.sheetnames:
                book.close()
                continue
            ws = book[sheet]
            iterator = ws.iter_rows(values_only=True)
            header = None
            for raw in iterator:
                values = [str(v).strip() if v is not None else "" for v in raw]
                if "LSOA 2021 Code" in values and "Total" in values:
                    header = values
                    break
            if header is None:
                book.close()
                continue
            code_idx = header.index("LSOA 2021 Code")
            total_idx = header.index("Total")
            age_cols: Dict[int, int] = {}
            for idx, name in enumerate(header):
                match = re.fullmatch(r"[Ff](\d+)", name)
                if match:
                    age_cols[int(match.group(1))] = idx
            parsed = []
            for raw in iterator:
                if code_idx >= len(raw):
                    continue
                code = _normalise_code(raw[code_idx])
                if not code.startswith("E"):
                    continue
                values = pd.to_numeric(pd.Series(list(raw)), errors="coerce")
                total = values.iloc[total_idx] if total_idx < len(values) else np.nan
                under16 = sum(values.iloc[idx] for age, idx in age_cols.items() if age <= 15 and idx < len(values))
                working = sum(values.iloc[idx] for age, idx in age_cols.items() if 16 <= age <= 64 and idx < len(values))
                older = sum(values.iloc[idx] for age, idx in age_cols.items() if age >= 65 and idx < len(values))
                parsed.append((code, total, under16, working, older))
            book.close()
            out = pd.DataFrame(parsed, columns=["lsoa21", "pop_total", "pop_under16", "pop_working", "pop_65_plus"])
            for col in ["pop_total", "pop_under16", "pop_working", "pop_65_plus"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out["pop_under16_pct"] = out["pop_under16"] / out["pop_total"].replace(0, np.nan) * 100
            out["pop_working_age_pct"] = out["pop_working"] / out["pop_total"].replace(0, np.nan) * 100
            out["pop_65_plus_pct"] = out["pop_65_plus"] / out["pop_total"].replace(0, np.nan) * 100
            out["source_year"] = year
            rows.append(out[["lsoa21", "source_year", "pop_total", "pop_under16_pct", "pop_working_age_pct", "pop_65_plus_pct"]])
            LOGGER.info("Loaded population estimates %s from %s (%s LSOAs)", year, path.name, len(out))
        except Exception as exc:
            LOGGER.warning("Could not parse population estimates %s: %s", path, exc)
    if not rows:
        return pd.DataFrame(columns=["lsoa21", "source_year", "pop_total", "pop_under16_pct", "pop_working_age_pct", "pop_65_plus_pct"])
    return pd.concat(rows, ignore_index=True)


QOF_PUBLICATION_DATES = {
    2017: pd.Timestamp("2018-10-26"),
    2019: pd.Timestamp("2020-08-20"),
    2021: pd.Timestamp("2022-09-22"),
    2022: pd.Timestamp("2023-09-07"),
    2023: pd.Timestamp("2024-08-29"),
    2024: pd.Timestamp("2025-08-28"),
}


def _qof_indicator_position(rows: List[Tuple[object, ...]], header_idx: int, code: str) -> Optional[int]:
    for row in rows[max(0, header_idx - 6):header_idx]:
        for idx, value in enumerate(row):
            if str(value).strip() == code:
                return idx
    return None


def _qof_metric_column(headers: List[str], indicator_idx: int, all_indicator_positions: List[int]) -> Optional[int]:
    end = min([pos for pos in all_indicator_positions if pos > indicator_idx] or [len(headers)])
    window = range(indicator_idx + 1, end)
    priorities = ["underlying achievement", "net of pcas", "achievement rate", "achievement (%)"]
    for priority in priorities:
        for idx in window:
            if priority in headers[idx].lower():
                return idx
    return None


def _read_qof_sheet(path: Path, sheet: str, source_year: int) -> pd.DataFrame:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = book[sheet]
        rows = list(ws.iter_rows(values_only=True))
    finally:
        book.close()
    header_idx = next((i for i, row in enumerate(rows) if any(str(v).strip().lower() == "practice code" for v in row)), None)
    if header_idx is None:
        return pd.DataFrame()
    headers = [str(v).strip() if v is not None else "" for v in rows[header_idx]]
    practice_idx = next(i for i, value in enumerate(headers) if value.lower() == "practice code")
    codes = ["MH002", "MH021"] if sheet == "MH" else ["DEP004", "DEP003"]
    positions = [pos for code in codes if (pos := _qof_indicator_position(rows, header_idx, code)) is not None]
    metric_indices = {
        code: _qof_metric_column(headers, pos, positions)
        for code in codes
        for pos in [_qof_indicator_position(rows, header_idx, code)]
        if pos is not None
    }
    records = []
    for row in rows[header_idx + 1:]:
        if practice_idx >= len(row):
            continue
        practice = _normalise_code(row[practice_idx])
        if not practice or practice.lower() in {"practice code", "nan"}:
            continue
        record = {"PRACTICE_CODE": practice, "source_year": source_year}
        for code, idx in metric_indices.items():
            record[code] = pd.to_numeric(pd.Series([row[idx] if idx is not None and idx < len(row) else np.nan]), errors="coerce").iloc[0]
        records.append(record)
    return pd.DataFrame(records)


def load_historical_qof(source_root: Path, mapping_path: Path) -> pd.DataFrame:
    """Extract current-year indicator rates and allocate them to 2021 LSOAs.

    QOF workbooks expose current and previous reporting years, but the detailed
    indicator blocks are current-year measures.  Publication dates are retained
    so the panel can select only reports available before each target year.
    """
    base = source_root / "quality_outcomes_framework"
    mapping = pd.read_csv(mapping_path)
    mapping = mapping.rename(columns={"PRACTICE_CODE": "PRACTICE_CODE", "LSOA_CODE": "lsoa21", "NUMBER_OF_PATIENTS": "patients"})
    mapping["PRACTICE_CODE"] = mapping["PRACTICE_CODE"].map(_normalise_code)
    mapping["lsoa21"] = mapping["lsoa21"].map(_normalise_code)
    mapping["patients"] = pd.to_numeric(mapping["patients"], errors="coerce")
    mapping = mapping[(mapping["lsoa21"].str.startswith("E", na=False)) & (mapping["patients"] > 0)].copy()
    outputs = []
    for year_dir in sorted(base.iterdir() if base.exists() else []):
        if not year_dir.is_dir() or not year_dir.name[:4].isdigit():
            continue
        source_year = int(year_dir.name[:4])
        publication_date = QOF_PUBLICATION_DATES.get(source_year)
        if publication_date is None:
            continue
        paths = sorted(year_dir.rglob("*.xlsx"))
        if not paths:
            continue
        path = paths[0]
        mh = _read_qof_sheet(path, "MH", source_year)
        dep = _read_qof_sheet(path, "DEP", source_year)
        if mh.empty and dep.empty:
            continue
        qof = mh.merge(dep, on=["PRACTICE_CODE", "source_year"], how="outer")
        qof = qof.rename(columns={"MH002": "qof_mh002_pct", "MH021": "qof_mh021_pct", "DEP003": "qof_dep_achievement_pct", "DEP004": "qof_dep_achievement_pct"})
        merged = mapping.merge(qof, on="PRACTICE_CODE", how="inner")
        if merged.empty:
            continue
        feature_cols = [c for c in ["qof_mh002_pct", "qof_mh021_pct", "qof_dep_achievement_pct"] if c in merged]
        rows = []
        for lsoa, group in merged.groupby("lsoa21"):
            record = {"lsoa21": lsoa, "source_year": source_year, "publication_date": publication_date}
            for col in feature_cols:
                valid = group.dropna(subset=[col])
                record[col] = np.average(valid[col], weights=valid["patients"]) if not valid.empty else np.nan
            rows.append(record)
        out = pd.DataFrame(rows)
        outputs.append(out)
        LOGGER.info("Loaded QOF %s from %s (%s LSOAs; published %s)", source_year, path.name, len(out), publication_date.date())
    if not outputs:
        return pd.DataFrame(columns=["lsoa21", "source_year", "publication_date", "qof_mh002_pct", "qof_mh021_pct", "qof_dep_achievement_pct"])
    return pd.concat(outputs, ignore_index=True)


def _latest_before(panel: pd.DataFrame, history: pd.DataFrame, value_cols: Sequence[str], year: int, key: str, label: str) -> pd.DataFrame:
    """Attach the most recent dated observation strictly before ``year``."""
    missing = {col: np.nan for col in value_cols}
    if history.empty or "source_year" not in history.columns:
        return panel.assign(**missing, **{f"{label}_source_year": np.nan})
    eligible = history[history["source_year"] < year].copy()
    if eligible.empty:
        return panel.assign(**missing, **{f"{label}_source_year": np.nan})
    eligible = eligible.sort_values([key, "source_year"]).drop_duplicates(key, keep="last")
    eligible = eligible.rename(columns={"source_year": f"{label}_source_year"})
    columns = [key, *value_cols, f"{label}_source_year"]
    return panel.merge(eligible[[c for c in columns if c in eligible.columns]], on=key, how="left")


def _latest_qof_before(panel: pd.DataFrame, qof: pd.DataFrame, year: int) -> pd.DataFrame:
    """Attach the latest QOF report whose publication date predates the target."""
    value_cols = ["qof_mh002_pct", "qof_mh021_pct", "qof_dep_achievement_pct"]
    missing = {col: np.nan for col in value_cols}
    missing.update({"qof_source_year": np.nan, "qof_publication_date": pd.NaT})
    if qof.empty:
        return panel.assign(**missing)
    cutoff = pd.Timestamp(f"{year}-01-01")
    eligible = qof[qof["publication_date"] < cutoff].copy()
    if eligible.empty:
        return panel.assign(**missing)
    eligible = eligible.sort_values(["lsoa21", "publication_date", "source_year"]).drop_duplicates("lsoa21", keep="last")
    eligible = eligible.rename(columns={"source_year": "qof_source_year", "publication_date": "qof_publication_date"})
    columns = ["lsoa21", *value_cols, "qof_source_year", "qof_publication_date"]
    return panel.merge(eligible[[c for c in columns if c in eligible.columns]], on="lsoa21", how="left")


def build_panel(root: Path, scope: str = "lincolnshire") -> pd.DataFrame:
    datasets = root / "datasets"
    samhi = pd.read_csv(datasets / "samhi" / "samhi_21_01_v5.00_2011_2022_LSOA.csv")
    samhi["lsoa11"] = samhi["lsoa11"].map(_normalise_code)
    lookup_path = datasets / "lincolnshire_lsoa" / "lsoa_2011_to_2021_lookup" / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    lookup = pd.read_csv(lookup_path, usecols=["LSOA11CD", "LSOA21CD", "LAD22NM"])
    lookup = lookup.rename(columns={"LSOA11CD": "lsoa11", "LSOA21CD": "lsoa21", "LAD22NM": "lad"})
    lookup["lsoa11"] = lookup["lsoa11"].map(_normalise_code)
    lookup["lsoa21"] = lookup["lsoa21"].map(_normalise_code)
    master = samhi.merge(lookup.drop_duplicates("lsoa11"), on="lsoa11", how="left")
    lincolnshire_lads = {"Boston", "East Lindsey", "Lincoln", "North Kesteven", "South Holland", "South Kesteven", "West Lindsey"}
    master["is_lincolnshire"] = master["lad"].isin(lincolnshire_lads).astype(int)
    if scope == "lincolnshire":
        master = master[master["is_lincolnshire"] == 1].copy()

    source_root = root / "scripts" / "utils" / "source_data"
    fuel = load_historical_fuel_poverty(source_root)
    travel = load_historical_travel(datasets)
    population = load_historical_population(source_root)
    qof = load_historical_qof(
        source_root,
        datasets / "patients_registered_gp_practice" / "july_2026" / "gp-reg-pat-prac-lsoa-all.csv",
    )

    records: List[pd.DataFrame] = []
    for year in range(2014, 2023):
        target = f"samhi_index.{year}"
        if target not in master.columns or f"samhi_index.{year - 3}" not in master.columns:
            continue
        l1 = master[f"samhi_index.{year - 1}"]
        l2 = master[f"samhi_index.{year - 2}"]
        l3 = master[f"samhi_index.{year - 3}"]
        row = pd.DataFrame({
            "lsoa11": master["lsoa11"], "lsoa21": master["lsoa21"], "lad": master["lad"],
            "year": year, "target": master[target], "lag_1": l1, "lag_2": l2, "lag_3": l3,
            "delta_1": l1 - l2, "delta_2": l2 - l3,
            "acceleration": (l1 - l2) - (l2 - l3),
            "rolling_mean_3yr": pd.concat([l1, l2, l3], axis=1).mean(axis=1),
            "rolling_std_3yr": pd.concat([l1, l2, l3], axis=1).std(axis=1).fillna(0),
            "rolling_min_3yr": pd.concat([l1, l2, l3], axis=1).min(axis=1),
            "rolling_max_3yr": pd.concat([l1, l2, l3], axis=1).max(axis=1),
        })
        row = _latest_before(row, fuel, ["fuel_poverty_pct"], year, "lsoa11", "fuel")
        row = _latest_before(row, travel, HISTORICAL_FEATURES[1:5], year, "lsoa11", "travel")
        row = _latest_before(row, population, HISTORICAL_FEATURES[5:9], year, "lsoa21", "population")
        row = _latest_qof_before(row, qof, year)
        records.append(row)
    panel = pd.concat(records, ignore_index=True)
    # Imputation is fitted inside each training origin; retaining missingness
    # here prevents future-year values from entering earlier origins.
    return panel


def crps_normal(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> float:
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-6)
    z = (np.asarray(y) - np.asarray(mean)) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))))


def metrics(y: np.ndarray, pred: np.ndarray, lag: np.ndarray, sd: np.ndarray) -> Dict[str, float]:
    lo, hi = pred - 1.96 * sd, pred + 1.96 * sd
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "directional_accuracy_pct": float(np.mean(np.sign(y - lag) == np.sign(pred - lag)) * 100),
        "interval_coverage_pct": float(np.mean((y >= lo) & (y <= hi)) * 100),
        "interval_width": float(np.mean(hi - lo)),
        "crps": crps_normal(y, pred, sd),
    }


def fit_elastic(train: pd.DataFrame, feature_cols: Sequence[str]) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000, random_state=42)),
    ]).fit(train[list(feature_cols)], train["target"])


def fit_bayesian(train: pd.DataFrame, feature_cols: Sequence[str]) -> Pipeline:
    numeric = list(feature_cols)
    categorical = ["lsoa21", "year"]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    model = Pipeline([("features", pre), ("bayesian_ridge", BayesianRidge(compute_score=True))])
    model.fit(train[numeric + categorical], train["target"])
    return model


def load_spatial_adjacency(root: Path, codes: Sequence[str]) -> Dict[str, List[str]]:
    """Build a Queen-contiguity graph from the supplied 2021 LSOA GeoJSON."""
    try:
        import geopandas as gpd

        path = root / "datasets" / "lincolnshire_lsoa" / "lower-super-output-areas-2021-5RrVTw.geojson"
        gdf = gpd.read_file(path)[["CODE", "geometry"]]
        wanted = {str(code) for code in codes if pd.notna(code)}
        gdf = gdf[gdf["CODE"].isin(wanted)].reset_index(drop=True)
        adjacency = {code: [] for code in gdf["CODE"]}
        for idx, row in gdf.iterrows():
            neighbours = gdf.sindex.query(row.geometry, predicate="touches")
            adjacency[row["CODE"]] = [gdf.iloc[j]["CODE"] for j in neighbours if j != idx]
        LOGGER.info("Built spatial adjacency for %s LSOAs (%s edges)", len(adjacency), sum(map(len, adjacency.values())) // 2)
        return adjacency
    except Exception as exc:
        LOGGER.warning("Could not build LSOA adjacency: %s", exc)
        return {}


def fit_spatial_bayesian(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    adjacency: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a conjugate Gaussian CAR-style random-intercept model.

    Fixed effects use the same dated numeric covariates as the other models.
    LSOA effects have a properised intrinsic-CAR prior, so neighbouring LSOAs
    are shrunk towards one another while a small diagonal term keeps the
    precision matrix nonsingular.  This is a lightweight spatial Bayesian
    pilot; it is not a substitute for a fully sampled PyMC/INLA model.
    """
    codes = sorted(set(train["lsoa21"].dropna().astype(str)) | set(test["lsoa21"].dropna().astype(str)))
    code_idx = {code: idx for idx, code in enumerate(codes)}
    m = len(codes)
    medians = train[list(feature_cols)].median(numeric_only=True)
    scales = train[list(feature_cols)].std(numeric_only=True).replace(0, 1).fillna(1)
    x_train = train[list(feature_cols)].fillna(medians).fillna(0).sub(medians).div(scales).to_numpy(float)
    x_test = test[list(feature_cols)].fillna(medians).fillna(0).sub(medians).div(scales).to_numpy(float)
    year_mean = float(train["year"].mean())
    year_sd = max(float(train["year"].std(ddof=0)), 1.0)
    fixed_train = np.column_stack([np.ones(len(train)), x_train, (train["year"].to_numpy(float) - year_mean) / year_sd])
    fixed_test = np.column_stack([np.ones(len(test)), x_test, (test["year"].to_numpy(float) - year_mean) / year_sd])
    z_train = np.zeros((len(train), m))
    z_test = np.zeros((len(test), m))
    for row_idx, code in enumerate(train["lsoa21"].astype(str)):
        if code in code_idx:
            z_train[row_idx, code_idx[code]] = 1.0
    for row_idx, code in enumerate(test["lsoa21"].astype(str)):
        if code in code_idx:
            z_test[row_idx, code_idx[code]] = 1.0

    # Properised CAR precision: tau*I + rho*(D-W).
    w = np.zeros((m, m), dtype=float)
    for code, neighbours in adjacency.items():
        if code not in code_idx:
            continue
        for neighbour in neighbours:
            if neighbour in code_idx:
                w[code_idx[code], code_idx[neighbour]] = 1.0
    degree = np.diag(w.sum(axis=1))
    car_precision = 1.0 * np.eye(m) + 0.75 * (degree - w)
    x_train_full = np.column_stack([fixed_train, z_train])
    x_test_full = np.column_stack([fixed_test, z_test])
    # A weakly informative Gaussian prior for fixed effects and the CAR prior
    # for spatial effects.  Noise is estimated only from the training fold.
    prior = np.zeros((x_train_full.shape[1], x_train_full.shape[1]))
    prior[:fixed_train.shape[1], :fixed_train.shape[1]] = np.eye(fixed_train.shape[1]) * 0.25
    prior[0, 0] = 1e-6
    prior[-m:, -m:] = car_precision
    residual_scale = max(float(np.std(train["target"].to_numpy() - train["lag_1"].to_numpy(), ddof=1)), 0.05)
    sigma2 = residual_scale ** 2
    precision = prior + (x_train_full.T @ x_train_full) / sigma2
    rhs = x_train_full.T @ train["target"].to_numpy(float) / sigma2
    coefficients = np.linalg.solve(precision, rhs)
    pred = x_test_full @ coefficients
    posterior_rows = np.linalg.solve(precision, x_test_full.T).T
    variance = sigma2 + np.sum(posterior_rows * x_test_full, axis=1)
    return pred, np.sqrt(np.maximum(variance, 1e-8))


def run(
    panel: pd.DataFrame,
    output_dir: Path,
    min_train_year: int = 2014,
    adjacency: Optional[Dict[str, List[str]]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    predictions: List[pd.DataFrame] = []
    all_features = HISTORY_FEATURES + HISTORICAL_FEATURES
    origins = range(min_train_year + 3, int(panel["year"].max()) + 1)
    for target_year in origins:
        train = panel[panel["year"] < target_year].copy()
        test = panel[panel["year"] == target_year].copy()
        if train.empty or test.empty:
            continue
        # Ensure no row with an entirely missing history is silently used.
        train = train.dropna(subset=["target", "lag_1", "lag_2", "lag_3"])
        test = test.dropna(subset=["target", "lag_1", "lag_2", "lag_3"])
        if test.empty:
            continue
        train_resid_sd = float(np.std(train["target"] - train["lag_1"], ddof=1))
        baseline_pred = test["lag_1"].to_numpy()
        baseline_sd = np.full(len(test), max(train_resid_sd, 1e-3))
        rows.append({"target_year": target_year, "model": "persistence", **metrics(test["target"].to_numpy(), baseline_pred, test["lag_1"].to_numpy(), baseline_sd)})

        history_model = fit_elastic(train, HISTORY_FEATURES)
        hist_pred = history_model.predict(test[HISTORY_FEATURES])
        hist_sd = np.full(len(test), max(float(np.std(train["target"] - history_model.predict(train[HISTORY_FEATURES]), ddof=1)), 1e-3))
        rows.append({"target_year": target_year, "model": "history_elasticnet", **metrics(test["target"].to_numpy(), hist_pred, test["lag_1"].to_numpy(), hist_sd)})

        # A report-era feature can be structurally unavailable in early folds
        # (for example MH021 was introduced after the 2019 target).  Exclude
        # such columns from that fold rather than asking the imputer to invent
        # a median for an all-missing series.
        available_features = [col for col in all_features if train[col].notna().any()]
        full_model = fit_elastic(train, available_features)
        full_pred = full_model.predict(test[available_features])
        full_sd = np.full(len(test), max(float(np.std(train["target"] - full_model.predict(train[available_features]), ddof=1)), 1e-3))
        rows.append({"target_year": target_year, "model": "history_plus_historical_elasticnet", **metrics(test["target"].to_numpy(), full_pred, test["lag_1"].to_numpy(), full_sd)})

        bayes_model = fit_bayesian(train, available_features)
        bayes_pred, bayes_sd = bayes_model.predict(test[available_features + ["lsoa21", "year"]], return_std=True)
        rows.append({"target_year": target_year, "model": "bayesian_ridge_partial_pooling", **metrics(test["target"].to_numpy(), bayes_pred, test["lag_1"].to_numpy(), bayes_sd)})

        if adjacency:
            spatial_pred, spatial_sd = fit_spatial_bayesian(train, test, available_features, adjacency)
            rows.append({"target_year": target_year, "model": "bayesian_car_spatial_random_effect", **metrics(test["target"].to_numpy(), spatial_pred, test["lag_1"].to_numpy(), spatial_sd)})

        pred = test[["lsoa11", "lsoa21", "lad", "year", "target"]].copy()
        pred["pred_persistence"] = baseline_pred
        pred["pred_history_elasticnet"] = hist_pred
        pred["pred_history_plus_historical_elasticnet"] = full_pred
        pred["pred_bayesian_ridge_partial_pooling"] = bayes_pred
        pred["bayesian_sd"] = bayes_sd
        if adjacency:
            pred["pred_bayesian_car_spatial_random_effect"] = spatial_pred
            pred["bayesian_car_sd"] = spatial_sd
        predictions.append(pred)
        LOGGER.info("Target %s: history RMSE %.4f, historical RMSE %.4f, Bayesian RMSE %.4f", target_year, np.sqrt(mean_squared_error(test["target"], hist_pred)), np.sqrt(mean_squared_error(test["target"], full_pred)), np.sqrt(mean_squared_error(test["target"], bayes_pred)))

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        aggregate = []
        for model, group in metrics_df.groupby("model"):
            # Aggregate point predictions are recomputed from saved rows below;
            # mean fold metrics remain useful for diagnosing time variation.
            aggregate.append({"target_year": "mean_folds", "model": model, **{c: float(group[c].mean()) for c in ["rmse", "mae", "directional_accuracy_pct", "interval_coverage_pct", "interval_width", "crps"]}, "n": int(group["n"].sum())})
        metrics_df = pd.concat([metrics_df, pd.DataFrame(aggregate)], ignore_index=True)
    metrics_df.to_csv(output_dir / "rolling_origin_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(output_dir / "rolling_origin_predictions.csv", index=False)
    panel.to_csv(output_dir / "dated_panel.csv", index=False)
    LOGGER.info("Wrote experiment outputs to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-aware rolling-origin SAMHI experiment")
    parser.add_argument("--scope", choices=["lincolnshire", "national"], default="lincolnshire")
    parser.add_argument("--output-dir", default="scripts/machine_learning/samhi/results/historical_bayesian_v2")
    args = parser.parse_args()
    if args.scope != "lincolnshire":
        raise SystemExit("National Bayesian scaling is intentionally disabled until the Lincolnshire pilot is calibrated.")
    root = project_root()
    panel = build_panel(root, args.scope)
    if panel.empty:
        raise SystemExit("No panel rows were created")
    adjacency = load_spatial_adjacency(root, panel["lsoa21"].dropna().unique())
    run(panel, root / args.output_dir, adjacency=adjacency)


if __name__ == "__main__":
    main()

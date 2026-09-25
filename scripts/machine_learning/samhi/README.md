# SAMHI Machine Learning Forecasting

Machine learning models to predict future Small Area Mental Health Index (SAMHI) scores for Lower Layer Super Output Areas (LSOAs) across Lincolnshire and England.

---

## Overview

The Small Area Mental Health Index (SAMHI) is a composite index of mental health need developed by the Place-Based Longitudinal Data Resource (PLDR), combining NHS mental health prescriptions, inpatient admissions, and incapacity/disability benefits from 2011 to 2022.

This folder contains the machine learning experiments exploring:
1. **Autoregressive baseline models** (Phase 1): Predicting future SAMHI using historical temporal lags, momentum, volatility, and decile history.
2. **Multimodal integration** (Phase 2): Incorporating external socioeconomic, demographic, housing, energy, travel accessibility, service-quality, and spatial-neighbour features.
3. **Forward projections** (Phase 3): Recursively projecting 2023–2025 using ElasticNet as the primary model, with Ridge and LightGBM retained for comparison.

---

## Project Structure

```
scripts/machine_learning/samhi/
├── 01_baseline_autoregressive.py  # Autoregressive baseline ML models & benchmarks
├── README.md                      # Documentation & execution guide
└── results/                       # Exported benchmark metrics & predictions
    ├── baseline_metrics_comparison.csv
    ├── baseline_metrics_lincolnshire.csv
    ├── baseline_metrics_national.csv
    ├── baseline_predictions_2020_2022_lincolnshire.csv
    ├── baseline_predictions_2020_2022_national.csv
    ├── feature_importance_lincolnshire.csv
    └── feature_importance_national.csv
```

---

## Prerequisites & Environment

Make sure you have activated the virtual environment and installed the dependencies:

```bash
cd /home/cheddar/code/LUMHR_dashboard
source scripts/.venv/bin/activate
pip install -r scripts/flask/requirements.txt scikit-learn lightgbm xgboost catboost interpret shap geopandas
```

---

## How to Run It

Run the baseline autoregressive benchmark script [`01_baseline_autoregressive.py`](01_baseline_autoregressive.py):

### 1. Run Lincolnshire Models Only
Trains and evaluates exclusively on Lincolnshire's 420/435 LSOAs:
```bash
python scripts/machine_learning/samhi/01_baseline_autoregressive.py --scope lincolnshire
```

### 2. Run National Models Only
Trains across all 32,844 LSOAs in England (164,000+ training rows) and evaluates nationally and on the Lincolnshire subset:
```bash
python scripts/machine_learning/samhi/01_baseline_autoregressive.py --scope national
```

### 3. Run Both Scopes & Generate Comparison
Executes both experiments and generates a side-by-side comparison file:
```bash
python scripts/machine_learning/samhi/01_baseline_autoregressive.py --scope both
```

### 4. Run Phase 2 Multimodal & Spatial Models
Trains multimodal models combining SAMHI history with IMD 2019 deprivation, disability, unemployment, rural/urban classification, healthcare travel times, QOF service-quality measures, housing tenure and bedroom occupancy, detailed household composition, fuel poverty, annual gas-grid disconnection, and spatial-neighbour lags:

```bash
# Run Lincolnshire multimodal models
python scripts/machine_learning/samhi/02_multimodal_features.py --scope lincolnshire

# Run National multimodal models (164,000+ rows)
python scripts/machine_learning/samhi/02_multimodal_features.py --scope national

# Run both scopes with the normal history-inclusive multimodal model
python scripts/machine_learning/samhi/02_multimodal_features.py --scope both --history with

# Run external-data-only models (no previous SAMHI predictors)
python scripts/machine_learning/samhi/02_multimodal_features.py --scope both --history without

# Run both history-inclusive and external-only experiments
python scripts/machine_learning/samhi/02_multimodal_features.py --scope both --history both

# Run the second, pre-COVID experiment set (2018-2019 test years)
python scripts/machine_learning/samhi/01_baseline_autoregressive.py --scope both --experiment-set 2
python scripts/machine_learning/samhi/02_multimodal_features.py --scope both --experiment-set 2 --history both
```

### 5. Run Phase 3 Multi-Year Forward Projections (2023 - 2025)
Retrains LightGBM, Ridge, and ElasticNet on the complete historical panel up to 2022 (295k+ rows) and recursively forecasts future mental health need for **2023, 2024, and 2025**. ElasticNet is the primary projection model; the other model outputs are retained for comparison:

```bash
# Run Lincolnshire forward projections (435 LSOAs)
python scripts/machine_learning/samhi/03_forward_projections.py --scope lincolnshire

# Run National forward projections (33k+ LSOAs)
python scripts/machine_learning/samhi/03_forward_projections.py --scope national

# Run both scopes and export projection summary
python scripts/machine_learning/samhi/03_forward_projections.py --scope both
```

### 6. Additional Options

* **Predict Change ($\Delta y_t$) instead of Level**:
  ```bash
  python scripts/machine_learning/samhi/02_multimodal_features.py --scope lincolnshire --target change
  ```
* **Custom Output Directory**:
  ```bash
  python scripts/machine_learning/samhi/03_forward_projections.py --scope lincolnshire --output-dir path/to/results
  ```

---

## Interactive Map Dashboard Guide

When using the Flask web application to view SAMHI predictions and projections, you will see a detailed tooltip pop-up when hovering over any LSOA on the map. Here is what those metrics mean:

*   **Predicted SAMHI**: The mental health need score predicted by the currently selected machine learning model for the selected year.
*   **Actual SAMHI**: The ground-truth mental health need score recorded by the PLDR for that year.
    *   *(Note: For future projections like 2023–2025+, this field shows the **2022 Baseline** since the true values have not yet occurred).*
*   **Prediction Error**: The mathematical difference between the model's prediction and reality (`Predicted SAMHI - Actual SAMHI`). 
    *   A **positive value (+)** means the model *over-predicted* the need (it anticipated things being worse than they were).
    *   A **negative value (-)** means the model *under-predicted* the need (it anticipated things being better than they were).
    *   *(Note: This metric is intentionally hidden for future projections where no ground truth exists).*
*   **Direction of Change**: The absolute movement in the SAMHI score compared to the previous year (or the 2022 baseline, for projections). 
    *   A **worsening (▲)** status indicates mental health need is increasing.
    *   An **improving (▼)** status indicates mental health need is decreasing.

---

## Evaluation Methodology

* **Train Split (2014–2018)**: Learns transition dynamics using sliding 3-year history windows ($t-1, t-2, t-3$).
* **Validation Split (2019)**: Uncorrupted pre-COVID baseline benchmark to evaluate model performance under normal conditions.
* **Test Split (2020–2022)**: Stress-test evaluating model resilience across the COVID-19 pandemic disruptions and subsequent recovery.
* **Experiment 2 (pre-COVID)**: Train on 2014–2016, validate on 2017, and test on 2018–2019. This provides a normal-period comparison without 2020–2022 observations in the test set.

### Evaluated Models (11 Architectures)
* **Naive Persistence Baseline**: $\widehat{y}_t = y_{t-1}$ (essential benchmark due to high temporal autocorrelation)
* **Naive Momentum Drift Baseline**: $\widehat{y}_t = y_{t-1} + (y_{t-1} - y_{t-2})$
* **Multimodal ElasticNet**: L1 + L2 regularized linear model with feature pruning (Tournament Winner 🏆)
* **Stacking Ensemble (Super Learner)**: Meta-learner combining LightGBM, CatBoost, XGBoost, and Ridge (Tournament Runner-up 🥈)
* **Multimodal Ridge Regression**: L2-regularized linear autoregressive model
* **Multimodal Random Forest**: Bagged non-linear decision tree ensemble
* **Multimodal Extra-Trees**: Extremely Randomized Trees
* **Multimodal LightGBM**: Fast histogram-based gradient boosted trees
* **Multimodal XGBoost**: Regularized gradient boosted decision trees
* **Multimodal CatBoost**: Symmetric oblivious decision tree ensemble
* **Explainable Boosting Machine (EBM)**: Glass-box Generalized Additive Model with Pairwise Interactions ($y = \sum f(x_i) + \sum f(x_i, x_j)$)

### Added external feature groups

The multimodal benchmark accepts `--history with`, `--history without`, or `--history both`. The `without` setting removes local, temporal, district and neighbour SAMHI-derived predictors while retaining the external data.

The multimodal benchmark and forward-projection pipeline now also include:

* Fuel-poverty percentage.
* Annual gas-grid disconnection percentage, lagged by one year for each SAMHI target year.
* Tenure percentages: owner occupied, shared ownership, social rented, private rented, and rent free.
* Bedroom occupancy percentages, including total overcrowding.
* Detailed Census household-composition percentages, including older people living alone, pensioner couples, family composition, and dependent-child categories.

Tenure, occupancy, and household-composition data are Census 2021 snapshots. Fuel-poverty data are 2024 and cover England at LSOA level. Gas-grid data are annual 2015–2024 from the Great Britain LSOA workbook; the England SAMHI geographies are matched by LSOA code. These availability dates should be considered when interpreting historical backtests.

### Metrics Reported
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **$R^2$** (Coefficient of Determination)
* **Directional Accuracy (%)**: Ability to correctly predict whether mental health need will increase or decrease compared to the previous year
* **Decile Accuracy (%)**: Exact and $\pm 1$ decile classification accuracy

---

## 11-Model Leaderboard

All models were evaluated under identical conditions: trained on **2014–2018 historical transitions**, validated on **2019 pre-COVID baseline**, and stress-tested against the **2020–2022 pandemic shock**:

### Lincolnshire Leaderboard ($N = 1,260$ Test Observations)

| Rank | Model Architecture | RMSE (Lower = Better) | $R^2$ (Higher = Better) | Directional Accuracy | Decile Accuracy ($\pm 1$) | Strengths & Characteristics |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Multimodal ElasticNet** | **0.2491** | **0.8634** | **73.25%** | **80.95%** | **Best overall accuracy.** L1+L2 regularization prunes redundant noise while retaining primary temporal and spatial drivers. |
| 2 | **Stacking Ensemble (Super Learner)** | **0.2560** | **0.8557** | **70.87%** | **81.35%** | **Best decile accuracy.** Blends LightGBM, CatBoost, XGBoost, and Ridge with a meta-regressor. |
| 3 | **Multimodal Ridge** | 0.2575 | 0.8540 | 71.19% | 80.71% | Solid L2 linear baseline; fast, stable, and interpretable. |
| 4 | **Multimodal Random Forest** | 0.2703 | 0.8392 | 71.11% | 79.21% | Resilient non-linear ensemble; robust against outliers. |
| 5 | **Multimodal Extra-Trees** | 0.2711 | 0.8383 | 71.43% | 79.60% | Extremely randomized decision splits. |
| 6 | **Explainable Boosting Machine (EBM)** | 0.2792 | 0.8285 | 70.48% | 80.24% | **100% Glass-Box Explainable.** Exact additive curve graphs ($y = \sum f(x_i) + \sum f(x_i, x_j)$) with no black-box opacity. |
| 7 | **Multimodal CatBoost** | 0.2793 | 0.8283 | 70.56% | 80.56% | Oblivious symmetric decision trees; strong generalization. |
| 8 | **Multimodal XGBoost** | 0.2795 | 0.8280 | 71.35% | 79.52% | Regularized gradient boosted trees with depth controls. |
| 9 | **Multimodal LightGBM** | 0.2816 | 0.8255 | 70.48% | 78.49% | Ultra-fast histogram-based tree boosting. |
| 10 | **AR Baseline (Persistence)** | 0.3138 | 0.7832 | 0.00% | 75.71% | Naive assumption that next year equals last year. |
| 11 | **AR Baseline (Momentum Drift)** | 0.5381 | 0.3627 | 47.54% | 59.21% | Linear trend projection; prone to overshooting during shocks. |

### National England Leaderboard ($N = 98,532$ Test Observations)

| Rank | Model Architecture | RMSE | $R^2$ | Directional Accuracy | Decile Accuracy ($\pm 1$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Multimodal ElasticNet** | **0.3192** | **0.8771** | **61.62%** | **87.65%** |
| 2 | **Multimodal Extra-Trees** | **0.3205** | **0.8761** | 61.36% | 87.38% |
| 3 | **Multimodal CatBoost** | 0.3265 | 0.8715 | 61.52% | 87.63% |
| 4 | **Multimodal Random Forest** | 0.3270 | 0.8710 | 61.44% | 87.31% |
| 5 | **Multimodal XGBoost** | 0.3278 | 0.8704 | 61.64% | 87.65% |
| 6 | **Multimodal LightGBM** | 0.3282 | 0.8701 | 61.55% | 87.66% |
| 7 | **AR Baseline (Persistence)** | 0.3289 | 0.8696 | 0.00% | 84.73% |
| 8 | **Multimodal Ridge** | 0.3330 | 0.8663 | 60.42% | 87.61% |
| 9 | **Stacking Ensemble (Super Learner)** | 0.3353 | 0.8645 | 60.35% | 87.66% |
| 10 | **Explainable Boosting Machine (EBM)** | 0.3592 | 0.8444 | 61.09% | 87.34% |
| 11 | **AR Baseline (Momentum Drift)** | 0.5499 | 0.6353 | 42.23% | 68.86% |

---

## Plain-English Guide to Model Features & Predictive Drivers

In epidemiological forecasting, technical feature names can be unintuitive. Below is a comprehensive plain-English glossary explaining what each predictive driver measures:

| Feature Name | Plain-English Label | Domain | Clinical & Operational Meaning |
| :--- | :--- | :--- | :--- |
| **`rolling_max_3yr`** | **3-Year Peak Demand Level** | Temporal History | Highest mental health index recorded in this LSOA over the past 3 years. Acts as an **"elastic ceiling"**: areas that reached high historical peaks rarely normalize immediately because chronic psychological illness has long recovery cycles. |
| **`lag_1`** | **Previous Year SAMHI Level ($t-1$)** | Temporal History | The most recently recorded official index level. Provides the baseline anchor for the forecast. |
| **`rolling_mean_3yr`** | **3-Year Moving Average** | Temporal History | Smoothed multi-year baseline demand that filters out single-year reporting noise or temporary shocks. |
| **`lag_2`** | **2 Years Prior Level ($t-2$)** | Temporal History | Historical demand level from 2 years prior; gives the model multi-year memory. |
| **`lag_3`** | **3 Years Prior Level ($t-3$)** | Temporal History | Historical demand level from 3 years prior. |
| **`delta_1`** | **1-Year Momentum / Trajectory** | Temporal History | Annual pace and direction of change ($y_{t-1} - y_{t-2}$). |
| **`acceleration`** | **Trend Acceleration** | Temporal History | Rate of change of the momentum ($\Delta y_{t-1} - \Delta y_{t-2}$), capturing speedup or slowdown in demand. |
| **`spatial_lag_1`** | **Bordering Areas' Need (Spillover)** | Geography & Access | Average SAMHI index across all **physically adjacent neighboring LSOAs**. Community distress spills over local borders due to shared local economies and regional service constraints. |
| **`lad_mean_samhi_lag_1`** | **District Authority Average Need** | Geography & Access | Overall baseline demand across the parent Local Authority (e.g. Lincoln, Boston, East Lindsey), capturing district-level funding and infrastructure factors. |
| **`pct_disability_limited`** | **Chronic Illness / Disability Rate (%)** | Demographics | Proportion of residents whose daily activities are limited by chronic health conditions. Physical multi-morbidity and chronic pain are primary biological drivers of anxiety and depressive disorders. |
| **`imd_2019_score`** | **Index of Multiple Deprivation (2019)** | Deprivation | Baseline government composite deprivation score combining income, employment, education, and crime. |
| **`imd_2025_decile`** | **Projected Deprivation Decile (2025)** | Deprivation | Updated 2025 relative deprivation rank (Decile 1 = most deprived 10%, Decile 10 = least deprived 10%). |
| **`hosp_car_time`** | **Hospital Car Travel Time (Minutes)** | Healthcare Access | Average driving time to the nearest acute secondary hospital with emergency psychiatric inpatient services. Captures physical isolation from crisis healthcare. |
| **`hosp_pt_time`** | **Hospital Public Transit Time (Minutes)**| Healthcare Access | Travel time by bus or train to the nearest acute hospital. |
| **`gp_car_time` / `gp_pt_time`** | **GP Surgery Travel Time (Minutes)** | Healthcare Access | Driving and public transit time to nearest primary care general practice surgery. |
| **`no_car_pct`** | **Households Without a Car (%)** | Transport Poverty | Captures transport poverty; residents without private vehicles in rural districts face acute isolation from support networks. |
| **`isolation_scale`** | **Rural Remoteness Scale (1–6)** | Geography & Access | Graduated index from 1 (Major Urban Conurbation) to 6 (Rural Dispersed/Isolated). |
| **`unemployment_rate`** | **Unemployment Rate (%)** | Deprivation | Percentage of working-age population claiming unemployment-related benefits. |
| **`pension_credit_rate`** | **Pension Credit Rate** | Deprivation | Rate per 1,000 pensioners receiving means-tested income support (capturing low-income elderly vulnerability). |
| **`one_person_household_pct`**| **Single Occupancy Living Alone (%)**| Household Structure| Proportion of households where an individual lives alone (key risk factor for social isolation). |
| **`avg_download_speed`** | **Broadband Download Speed (Mbps)** | Digital Infrastructure| Digital infrastructure quality; proxy for digital exclusion and feasibility of remote telehealth consultations. |

---

## Datasets and source links

The experiments use the processed files in `datasets/` so that runs are reproducible. The links in the final column point to the upstream dataset or publisher; local file links point to the exact files loaded by the scripts. Paths are relative to the repository root (`code/LUMHR_dashboard`).

### Core target and geography data

| Dataset and local file | Used by | Model role | Upstream source |
| :--- | :---: | :--- | :--- |
| [SAMHI 2011–2022 CSV](../../../datasets/samhi/samhi_21_01_v5.00_2011_2022_LSOA.csv) | 01, 02, 03 | Target variable and historical SAMHI lags, changes, rolling statistics, and deciles. | [PLDR Small Area Mental Health Index (SAMHI)](https://pldr.org/dataset/small-area-mental-health-index-samhi-2noyv) |
| [2011-to-2021 LSOA / LAD exact-fit lookup](<../../../datasets/lincolnshire_lsoa/lsoa_2011_to_2021_lookup/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv>) | 01, 02, 03 | Joins the 2011 geography used by SAMHI to 2021 LSOA codes, names, and Local Authority Districts; defines the Lincolnshire scope. | [Local geography documentation](../../../datasets/lincolnshire_lsoa/readme.md) |
| [Lincolnshire 2021 LSOA boundary GeoJSON](../../../datasets/lincolnshire_lsoa/lower-super-output-areas-2021-5RrVTw.geojson) | 02, 03 | Builds Queen-contiguity neighbours for spatial lag and spillover features. | [Lincolnshire County Council LSOA data](https://data.lincolnshire.gov.uk/@non-lincolnshire-county-council/boundaries/r/8a175023-7265-492a-8719-60de1173b366) |

The Flask SAMHI map uses the [Lincolnshire SAMHI extract](../../../datasets/samhi/samhi_lincolnshire_2021_lsoa.csv) for its dashboard calculations. The ML scripts instead load the national 2011–2022 SAMHI file above and apply the lookup themselves.

### Auxiliary features used by the multimodal and projection experiments

| Dataset and local file | Used by | Features or transformation | Upstream source |
| :--- | :---: | :--- | :--- |
| [Digital Exclusion Risk Index v1.6](<../../../datasets/digital_exclusion_risk_index/DERI dataset_v1.6.csv>) | 02, 03 | IMD 2019 score, unemployment, age 65+, disability, no qualifications, social grade DE, pension credit, and average download speed. | [GMCA / Greater Manchester Open Data Alliance DERI](https://github.com/GreaterManchesterODA/Digital-Exclusion-Risk-Index) |
| [2021 Rural/Urban Classification](<../../../datasets/rural_urban_classification_2021_lsoa/Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv>) | 02, 03 | Rural flag, isolation scale, and normalized isolation. | [DEFRA / ONS official collection](https://www.gov.uk/government/collections/rural-urban-classification) and [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::rural-urban-classification-2021-of-lsoas-in-ew/about) |
| [Census 2021 TS003 household composition](<../../../datasets/TS003_household_composition/TS003 - Household composition.csv>) | 02, 03 | One-person household and lone-parent household percentages. The Lincolnshire processed file is used as a fallback if the national source file is unavailable. | [ONS Census 2021 TS003](https://www.ons.gov.uk/datasets/TS003/editions/2021/versions/3) |
| [Census 2021 TS045 car or van availability](<../../../datasets/car_or_van_availability/TS045-2021-4-filtered-2026-08-24T10_54_11Z.csv>) | 02, 03 | Percentage of households with no car or van. | [ONS Census 2021 TS045](https://www.ons.gov.uk/datasets/TS045/editions/2021/versions/4) |
| [English Indices of Deprivation 2025 LSOA CSV](../../../datasets/indices_of_deprivation_imd/2025/england_imd_2025_lsoa.csv) | 02, 03 | IMD 2025 decile; the rank is also retained in the assembled data. | [English Indices of Deprivation 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) |
| [2019 GP travel times](../../../datasets/journey_time_statistics/england/clean_gps_2019_england.csv) and [hospital travel times](../../../datasets/journey_time_statistics/england/clean_hospitals_2019_england.csv) | 02, 03 | Public-transport and car travel times to GPs and hospitals. | [DfT Journey Time Statistics](https://www.gov.uk/government/statistical-data-sets/journey-time-statistics-data-tables-jts) |

### QOF-augmented experiment inputs

The QOF features are used by `02_multimodal_features.py` and `03_forward_projections.py`. Practice-level 2024–25 rates are joined to the July 2026 GP registration-to-LSOA mapping and converted to patient-weighted LSOA features. They are held constant across the 2014–2022 SAMHI panel; see the [QOF-augmented experiment](#qof-augmented-experiment-202425-measures) caveat below.

| Dataset and local file | Features created | Upstream source |
| :--- | :--- | :--- |
| [QOF depression 2024–25](../../../datasets/quality_outcomes_framework/qof_depression_2425_lincolnshire.csv) and [QOF mental health 2024–25](../../../datasets/quality_outcomes_framework/qof_mental_health_2425_lincolnshire.csv) | `qof_mh002_pct`, `qof_mh021_pct`, `qof_mh_pca_pct`, `qof_dep_pca_pct`, and `qof_dep004_pct`. | [NHS Digital QOF 2024–25](https://digital.nhs.uk/data-and-information/publications/statistical/quality-and-outcomes-framework-achievement-prevalence-and-exceptions-data/2024-25) |
| [GP registration-to-LSOA mapping, July 2026](../../../datasets/patients_registered_gp_practice/july_2026/gp-reg-pat-prac-lsoa-all.csv) | Patient weights used to allocate QOF rates from GP practices to LSOAs. | [NHS Digital Patients Registered at a GP Practice](https://digital.nhs.uk/data-and-information/publications/statistical/patients-registered-at-a-gp-practice/july-2026) |

### Flask dashboard sources not currently used as ML features

These sources are loaded by the need, access, access-gap, rural-risk, or map-marker pages. They are documented here to distinguish dashboard context from the variables passed to the ML models.

| Dataset and local file | Dashboard use | Upstream source |
| :--- | :--- | :--- |
| [Mid-2024 LSOA population estimates](../../../datasets/population_estimates/lincolnshire_lsoa_population_estimates_2024.csv) | Population context, registration-rate and gap calculations. | [ONS LSOA Mid-Year Population Estimates](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates) |
| [Antidepressant prescribing](../../../datasets/gp_prescribing_data/items_for_antidepressant_drugs_per_gp_lincolnshire_may_2026.csv) | Antidepressant items per patient in the Need and Access Gap indexes. | [OpenPrescribing – Lincolnshire ICB (71E)](https://openprescribing.net/analyse/#org=practice&orgIds=71E&numIds=4.3&denom=nothing&selectedTab=map) |
| [UK LSOA Classification 2021/2](../../../datasets/lsoa_classification_2021_2/lincolnshire_lsoa_classification_2021_2.csv) | Rural Risk Index geodemographic classification. | [GeoDS UK LSOA / DZ / SDZ Classification](https://data.geods.ac.uk/dataset/lsoac) |
| [Lincolnshire GP locations](../../../datasets/gp_locations/Lincolnshire_ICB_GPs.csv) | GP map markers. | [NHS Digital Organisation Data Service](https://digital.nhs.uk/services/organisation-data-service) |

The QOF, travel-time, rural/urban, car-availability, DERI, SAMHI, and LSOA geography sources overlap between the Flask dashboard and the ML pipeline. OpenPrescribing and the population estimates support the dashboard indexes but are not included in the current ML feature lists.

---

## How the Predictions Work: Combining Historical SAMHI with Demographics

The forecasting framework integrates two complementary signals:
1. **Baseline Trajectory & Momentum**: Derived from **historical SAMHI data** up to year $t-1$.
2. **Structural Vulnerability & Shocks**: Derived from **demographic, socioeconomic, and accessibility covariates**.

```
[ Historical SAMHI up to t-1 ]  --> Establishes baseline trajectory & temporal momentum
               +
[ Demographics & Environment ]   --> Adjusts trajectory based on structural risk factors
               =
[ Predicted Year t SAMHI ]       --> Out-of-time forecast
```

### 1. The Inputs (Predicting 2021 as an Example)
When predicting year $t = 2021$, the model is strictly blind to 2021 and uses only signals available prior to 2021:
* **Historical SAMHI**:
  - `lag_1` ($y_{2020}$): Preceding year's level (primary baseline anchor).
  - `lag_2` ($y_{2019}$), `lag_3` ($y_{2018}$): Trailing history.
  - `delta_1` ($y_{2020} - y_{2019}$): 1-year momentum.
  - `acceleration`: Change in momentum ($\Delta y_{2020} - \Delta y_{2019}$).
  - `rolling_mean_3yr`: Trailing 3-year average to filter out annual noise.
  - `spatial_lag_1`: Average 2020 SAMHI score of **bordering neighbor LSOAs** (capturing spatial spillovers).
  - `lad_mean_samhi_lag_1`: District-wide average SAMHI score.
* **Socioeconomic & Demographics**:
  - `imd_2019_score`: Index of Multiple Deprivation.
  - `pct_disability_limited`: Proportion of residents with chronic illness or physical disability.
  - `pct_aged_65_plus`: Proportion of elderly residents.
  - `unemployment_rate` & `pct_social_grade_de`: Economic distress indicators.
  - `is_rural`: 2021 Rural/Urban classification flag.
  - `GPPTt` & `GPCart`: Public transit and car travel times to nearest GP practice.
  - `one_person_household_pct` & `lone_parent_pct`: Social isolation indicators.

### 2. How the Models Combine Them
* **Linear Ridge Regression**: Computes a regularized weighted sum where temporal lags set the baseline score, and demographic/spatial coefficients nudge the prediction up or down according to local vulnerability.
* **Gradient-Boosted Trees (LightGBM / Random Forest)**: Learns non-linear interactions across domains. For instance:
  - *Interaction 1*: If prior year SAMHI showed rapid deterioration ($\Delta y > 0$) AND the area has elevated chronic illness/disability ($> 20\%$) in a rural setting, predict an amplified increase.
  - *Interaction 2*: If prior year SAMHI had a temporary spike, but the area has low deprivation and strong transport accessibility, predict mean reversion back toward baseline.

### 3. Concrete Example: West Lindsey 009F (`E01034707`) in 2021
* **2020 Baseline**: $0.362$
* **Local Risk Profile**: Rural classification with elevated neighboring mental health distress (`spatial_lag_1`).
* **Naive Persistence Guess**: Assumes 2021 remains at $0.362$ (misses the surge by $-0.263$).
* **Multimodal Model Prediction**: Anticipates upward surge and predicts **$0.590$**.
* **Actual 2021 Ground Truth**: Surged to **$0.625$**.
* **Outcome**: The Multimodal model correctly captured the sharp upward spike, missing by only **$0.035$**.




## QOF-augmented experiment (2024–25 measures)

The multimodal pipeline now includes the five QOF measures used by the dashboard Access maps:

| Model feature | Map measure | Direction in the model |
|---|---|---|
| `qof_mh002_pct` | MH002 — SMI Care Plan | Higher achievement rate |
| `qof_mh021_pct` | MH021 — SMI Health Check | Higher achievement rate |
| `qof_mh_pca_pct` | Mental Health PCA — SMI Exceptions | Higher exception rate |
| `qof_dep_pca_pct` | Depression PCA — Exceptions | Higher exception rate |
| `qof_dep004_pct` | DEP004 — Depression Review | Higher achievement rate |

Rates are patient-weighted from practice level to LSOA using the same July 2026 GP-registration allocation used by the dashboard. The source QOF files are 2024–25 cross-sectional data, so they are held constant across the 2014–2022 SAMHI panel. This makes the experiment useful for assessing whether current service-quality/access indicators add signal, but it is not a leakage-free historical causal test. PCA fields and DEP004 are missing for some practices in the supplied files; those features are median-imputed by the existing multimodal preprocessing.

Run the QOF-augmented benchmark and plots with:

```bash
scripts/.venv/bin/python scripts/machine_learning/samhi/02_multimodal_features.py --scope both --experiment-set 1
scripts/.venv/bin/python scripts/machine_learning/samhi/04_plot_results.py --scope both
```

### Results: 2020–2022 stress test

The QOF-augmented Lincolnshire test set contained 1,260 observations. ElasticNet remained the strongest model: RMSE **0.2491**, R² **0.8634**, directional accuracy **73.25%**, and within-one-decile accuracy **80.95%**. This improved on the persistence baseline (RMSE **0.3138**, R² **0.7832**), although the result reflects all multimodal features together and does not establish that QOF alone caused the improvement.

In the LightGBM TreeSHAP analysis, MH021 contributed **0.90%** and MH002 **0.44%** of total mean absolute contribution in Lincolnshire. The three other QOF features had zero contribution in this fitted tree model, which means they did not receive a split-based SHAP contribution here—not that they are clinically unimportant. Temporal SAMHI history remained dominant (**81.09%** of TreeSHAP contribution). The ElasticNet LinearSHAP analysis should be reported separately because it attributes the fitted linear model rather than the tree model.

For the national run, ElasticNet achieved RMSE **0.3192** and R² **0.8771**, narrowly ahead of persistence (RMSE **0.3289**, R² **0.8696**). The supplied QOF files represent Lincolnshire practices rather than a complete England-wide QOF panel, so QOF values outside the linked LSOAs are median-imputed; the national ElasticNet LinearSHAP QOF contribution was zero in this run. The national result should therefore be treated as a pipeline check, not evidence of England-wide QOF effects.

The pre-COVID experiment (train 2014–16; test 2018–19) was also rerun. Ridge was best in Lincolnshire (RMSE **0.2800**, R² **0.8335**) and nationally (RMSE **0.2954**, R² **0.8965**). In the LightGBM TreeSHAP analysis, the Lincolnshire QOF features contributed **1.56%** collectively; the national contribution was **0.009%**, consistent with the coverage limitation above. Results are in `multimodal_metrics_comparison_pre_covid_2018_2019.csv` and the corresponding TreeSHAP and ElasticNet LinearSHAP files.

New outputs are written to `results/` and `results/plots/`:

- [`05_qof_feature_contribution_lincolnshire.png`](results/plots/05_qof_feature_contribution_lincolnshire.png) — QOF SHAP contribution and rate distributions.
- [`05_qof_feature_contribution_national.png`](results/plots/05_qof_feature_contribution_national.png) — national diagnostic, including the limited source coverage.
- [`shap_feature_importance_2020_2022_lincolnshire.csv`](results/shap_feature_importance_2020_2022_lincolnshire.csv) — LightGBM TreeSHAP feature-level results.
- [`shap_feature_importance_elasticnet_2020_2022_lincolnshire.csv`](results/shap_feature_importance_elasticnet_2020_2022_lincolnshire.csv) — ElasticNet LinearSHAP feature-level results.
- [`shap_feature_importance_elasticnet_2020_2022_national.csv`](results/shap_feature_importance_elasticnet_2020_2022_national.csv) — national ElasticNet LinearSHAP results.
- [`shap_feature_importance_elasticnet_pre_covid_2018_2019_lincolnshire.csv`](results/shap_feature_importance_elasticnet_pre_covid_2018_2019_lincolnshire.csv) and [`shap_feature_importance_elasticnet_pre_covid_2018_2019_national.csv`](results/shap_feature_importance_elasticnet_pre_covid_2018_2019_national.csv) — pre-COVID ElasticNet LinearSHAP results.
- [`multimodal_metrics_comparison_2020_2022.csv`](results/multimodal_metrics_comparison_2020_2022.csv) — benchmark metrics.
- [`forward_projections_2023_2025_lincolnshire.csv`](results/forward_projections_2023_2025_lincolnshire.csv) and [`forward_projections_2023_2025_national.csv`](results/forward_projections_2023_2025_national.csv) — ElasticNet-led 2023–2025 projections with Ridge and LightGBM comparison columns.

# GitHub Large Files
```
find . -type f -size +95M \
  -not -path './.git/*' \
  -print0 |
while IFS= read -r -d '' file; do
    git lfs track "$file"
done

git add .gitattributes
```


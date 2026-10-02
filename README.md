# Lincolnshire Mental Health Research (LUMHR) Dashboard

An interactive geospatial decision-support platform for exploring small-area mental health need, healthcare accessibility, and rural risk across Lincolnshire at LSOA level.

Developed by the [Lincolnshire Unit for Mental Health Research (LUMHR)](https://lumhr.org.uk/) at the [University of Lincoln](https://www.lincoln.ac.uk/).

## Features

The dashboard provides interactive mapping and index calculation across Lincolnshire's 435 LSOAs:

- **Need Index**: Models local mental health demand using QOF depression and SMI prevalence, antidepressant prescribing, and SAMHI data. Range 0.0 lowest need to 1.0 most need.
- **Public DWP DLA/PIP indicators**: Separate 2025 DLA and PIP claimant-rate layers are included in the Need, Rural Risk, and Access Gap maps. They are separately weighted inputs to the Need and Rural Risk composites, while remaining clearly labelled as public-data indicators rather than official SAMHI components.
- **Access Index**: Assesses healthcare accessibility combining clinical quality indicators, travel times (car and public transport), vehicle availability, and digital exclusion (DERI). Range 0.0 lowest access to 1.0 most access.
- **Access Gap Index**: Highlights geographic disparities by comparing clinical need directly against accessibility (Need minus Access). Range (+1.0 to -1.0). +1.0 most severe gap, the area has a high need but low access, 0.0 balanced level of access matches its level of need, to -1.0 surplus access the area has lowest need but the highest access.
- The Access Gap page is the existing single-score Need-versus-Access map: it also lets you switch to the separate Need and Access layers, but it is not a bivariate colour-grid map.
- **Small Area Mental Health Index (SAMHI)**: Longitudinal view of composite mental health indicators from 2011 to 2022.
- **SAMHI Machine Learning Forecasting & Projections**: 11-model spatial-temporal machine learning suite (ElasticNet, Stacking Ensemble, CatBoost, LightGBM, Explainable Boosting Machine) with multi-year forward projections (2023–2025) and TreeSHAP explainability. See [scripts/machine_learning/samhi](scripts/machine_learning/samhi/README.md).
- **Rural Risk Index**: Evaluates rural vulnerability using 2021 Rural/Urban classification, travel times, car non-ownership, geodemographics (LSOAC), IMD deprivation, household composition, fuel poverty, and properties not connected to the gas grid. Range 0.0 low risk to 1.0 high risk.

## Data Sources

Supporting datasets are documented and organized in [datasets](datasets/):

- [TS003_household_composition](datasets/TS003_household_composition/) - Census 2021 household composition
- [car_or_van_availability](datasets/car_or_van_availability/) - Census 2021 vehicle ownership
- [digital_exclusion_risk_index](datasets/digital_exclusion_risk_index/) - Digital Exclusion Risk Index (DERI)
- [gp_locations](datasets/gp_locations/) - GP practice coordinates
- [gp_prescribing_data](datasets/gp_prescribing_data/) - Antidepressant prescribing data
- [indices_of_deprivation_imd](datasets/indices_of_deprivation_imd/) - English Indices of Deprivation (IMD)
- [journey_time_statistics](datasets/journey_time_statistics/) - Travel times to health services by mode
- [lincolnshire_lsoa](datasets/lincolnshire_lsoa/) - LSOA boundary geography
- [lsoa_classification_2021_2](datasets/lsoa_classification_2021_2/) - 2021 UK LSOA geodemographic classification
- [patients_registered_gp_practice](datasets/patients_registered_gp_practice/) - GP-to-LSOA patient registration matrices
- [population_estimates](datasets/population_estimates/) - Mid-year population estimates
- [quality_outcomes_framework](datasets/quality_outcomes_framework/) - QOF clinical indicators
- [rural_urban_classification_2021_lsoa](datasets/rural_urban_classification_2021_lsoa/) - Rural-urban classification
- [samhi](datasets/samhi/) - Small Area Mental Health Index data

The DWP map layers read the copied raw public exports from
[`DWP Stat-Xplore`](https://stat-xplore.dwp.gov.uk/), stored locally in
`datasets/DWP_DLA_PIP_data` (with the legacy
`scripts/utils/source_data/DWP_DLA_PIP_data` location retained as a fallback).
The reconstructed panel at
`scripts/machine_learning/samhi/results/component_reconstruction/three_component_panel.csv`
supplies denominator and publication-vintage metadata. The overlay supports
snapshots for 2023–2025, converts the panel's LSOA 2011 codes to the dashboard's
LSOA 2021 geography using the exact-fit ONS lookup, and records the population
denominator vintage in each tooltip/API response. DLA and PIP are
separate benefit claimant series; their rates should not be interpreted as a
unique-person prevalence measure or as a replacement for the missing hospital
component of official SAMHI. The map defaults use the latest 2025 DWP snapshot;
therefore a Need or Access Gap score should be treated as a current public-data
composite, not as a temporally matched historical SAMHI score. When one benefit
is disclosure-suppressed, the available component is retained using a marked
18–64 denominator fallback rather than being silently dropped. For one-to-many
2011-to-2021 splits, the map keeps every 2021 child LSOA: complete DLA/PIP
children use the parent panel denominator allocated by working-age population,
while partially suppressed children use the marked child 18–64 fallback. This
prevents the duplicate-parent rows that previously affected the ML panel and
could cause map children to disappear.

## Getting Started

### Prerequisites

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/flask/requirements.txt
```

### Running the Flask App (Primary Dashboard)

```bash
cd scripts/flask
python app.py
```

The application runs locally at `http://localhost:5005`.

### Running the Streamlit App

```bash
cd scripts/streamlit
pip install -r requirements.txt
streamlit run app.py
```

# Lincolnshire Mental Health Research (LUMHR) Dashboard

An interactive geospatial decision-support platform for exploring small-area mental health need, healthcare accessibility, and rural risk across Lincolnshire at LSOA level.

Developed by the [Lincolnshire Unit for Mental Health Research (LUMHR)](https://lumhr.org.uk/) at the [University of Lincoln](https://www.lincoln.ac.uk/).

## Features

The dashboard provides interactive mapping and index calculation across Lincolnshire's 435 LSOAs:

- **Need Index**: Models local mental health demand using QOF depression and SMI prevalence, antidepressant prescribing, and SAMHI data.
- **Access Index**: Assesses healthcare accessibility combining clinical quality indicators, travel times (car and public transport), vehicle availability, and digital exclusion (DERI).
- **Access Gap Index**: Highlights geographic disparities by comparing clinical need directly against accessibility (Need minus Access).
- **Small Area Mental Health Index (SAMHI)**: Longitudinal view of composite mental health indicators from 2011 to 2022.
- **Rural Risk Index**: Evaluates rural vulnerability using 2021 Rural/Urban classification, travel times, car non-ownership, geodemographics (LSOAC), and IMD deprivation.

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

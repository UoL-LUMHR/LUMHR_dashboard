# UK LSOA / DZ / SDZ Classification (2021/2 LSOAC)

The **2021/2 UK Output Area Classification for Lower Layer Super Output Areas (2021/2 LSOAC)** is a 3-tier hierarchical geodemographic classification developed from the 2021 Census of England & Wales (and Scottish Data Zones / Northern Ireland Super Data Zones). It identifies areas with similar social, economic, demographic, and housing characteristics.

The Lincolnshire extract contains all **435 Lower Layer Super Output Areas (2021 LSOAs)** within Lincolnshire. The England dashboard export contains the classification codes and names for all **33,755 England LSOAs** and is the input used by the Flask maps.

## Source Dataset
* **Source**: [Geographic Data Science (GeoDS) - UK LSOA / DZ / SDZ Classification (2021/2 LSOAC)](https://data.geods.ac.uk/dataset/lsoac)
* **Local Source Directory**: `scripts/utils/source_data/lsoa_classification_2021_2/`
  * `lsoa_dz_sdz_lookup.csv`
  * `classification_codes_and_names_ukoac_lsoac.csv`
* **Boundary Reference**: [`datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson`](../lsoa/lower-super-output-areas-2021-5RrVTw.geojson)

## Files in this Directory
* `lincolnshire_lsoa_classification_2021_2.csv` — CSV containing all 435 Lincolnshire 2021 LSOAs with Supergroup, Group, and Subgroup codes and descriptive names.
* `england_lsoa_classification_2021_2.csv` — compact England-wide CSV used by the Flask maps. Runtime data is loaded from `datasets/`; the app does not read files from `scripts/utils/source_data`.

## Data Columns
1. `LSOA_Code`: 2021 9-character LSOA code (e.g., `E01026007`).
2. `LSOA_Name`: 2021 LSOA name (e.g., `Boston 001A`).
3. `Supergroup_Code`: 1-digit top-level classification code (1 to 8).
4. `Supergroup_Name`: High-level socio-demographic cluster name.
5. `Group_Code`: 2-character middle-tier classification code (e.g., `1a`, `2b`, `7a`).
6. `Group_Name`: Detailed socio-demographic group name.
7. `Subgroup_Code`: 3-character granular classification code (e.g., `1a1`, `2b2`, `7a2`).
8. `Subgroup_Name`: Granular socio-demographic sub-cluster name.

## Supergroup Distribution & Vulnerability Weights in Rural Risk Index

| Supergroup Code & Name | Vulnerability Weight | Count (LSOAs) | % in Lincs | Key Demographic Focus |
| :--- | :---: | :---: | :---: | :--- |
| **8. Legacy Communities** | **`0.90`** | 2 | 0.5% | Severe deprivation, long-term health burdens, low physical mobility. |
| **7. Semi- and Un-Skilled Workforce** | **`0.75`** | 81 | 18.6% | Routine/manual occupations, lower incomes, higher chronic illness risks. |
| **1. Retired Professionals (Ageing Rural)** | **`0.70`** | 162 | 37.2% | High median age, retirees, distant from services, frailty and social isolation. |
| **4. Low-Skilled Migrant & Students** | **`0.60`** | 1 | 0.2% | Transient populations, private renting, potential language/access hurdles. |
| **6. Baseline UK** | **`0.50`** | 48 | 11.0% | National average socioeconomic and demographic baseline. |
| **2. Suburbanites and Peri-Urbanites** | **`0.40`** | 129 | 29.7% | Family commuter areas, higher vehicle availability, moderate access. |
| **5. Ethnically Diverse Suburb-an Professionals** | **`0.30`** | 8 | 1.8% | Higher education and income, strong private transport mobility. |
| **3. Multicultural and Educated Urbanites** | **`0.20`** | 4 | 0.9% | Young, highly educated, dense walkable urban infrastructure. |

## Detailed Group Breakdown in Lincolnshire (16 Groups)

* **Supergroup 1: Retired Professionals**
  * `1a` **Spacious Rural Living**: 109 LSOAs (25.1%) — Isolated countryside, high car dependence, older population.
  * `1b` **Small Town Suburbia**: 34 LSOAs (7.8%) — Market town fringes, semi-rural.
  * `1c` **Established Mature Families**: 19 LSOAs (4.4%) — Stable mature households.
* **Supergroup 2: Suburbanites and Peri-Urbanites**
  * `2b` **Rural Amenity**: 76 LSOAs (17.5%) — Accessible rural villages and peri-urban greenspaces.
  * `2a` **Inner Suburbs and Small Town Living**: 31 LSOAs (7.1%) — Town suburbs.
  * `2c` **Ageing Communities**: 22 LSOAs (5.1%) — Peri-urban ageing enclaves.
* **Supergroup 6: Baseline UK**
  * `6b` **Legacy Industrial and Coastal Communities**: 40 LSOAs (9.2%) — Coastal towns (Skegness, Mablethorpe) and industrial peripheries.
  * `6c` **Multicultural Inner Suburbs**: 5 LSOAs (1.1%)
  * `6a` **Challenged Communities**: 3 LSOAs (0.7%)
* **Supergroup 7: Semi- and Un-Skilled Workforce**
  * `7a` **Established but Challenged**: 42 LSOAs (9.7%)
  * `7b` **Young Families in Industrial Towns**: 39 LSOAs (9.0%)
* **Supergroup 5: Ethnically Diverse Suburban Professionals**
  * `5b` **Suburban Professionals**: 6 LSOAs (1.4%)
  * `5a` **Outer Suburbs**: 2 LSOAs (0.5%)
* **Supergroup 3: Multicultural and Educated Urbanites**
  * `3a` **Student Living and Professional Footholds**: 4 LSOAs (0.9%)
* **Supergroup 8: Legacy Communities**
  * `8b` **Legacy and Demographically Mixed Communities**: 2 LSOAs (0.5%)
* **Supergroup 4: Low-Skilled Migrant and Student Communities**
  * `4a` **Ethnically Diverse Families in Less Connected Locations**: 1 LSOA (0.2%)

## Generation Script
Generated using [`scripts/utils/process_lsoac_2021_2.py`](../../scripts/utils/process_lsoac_2021_2.py).

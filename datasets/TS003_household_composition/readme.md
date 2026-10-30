# TS003 - Household Composition (Census 2021) - Lincolnshire LSOAs

This dataset provides 2021 Census estimates from the Office for National Statistics (ONS) on **Household Composition** (Table TS003) for all **435 Lower Layer Super Output Areas (2021 LSOAs)** in Lincolnshire.

## Source Dataset
* **Source**: [Office for National Statistics (ONS) - Census 2021 Table TS003](https://www.ons.gov.uk/datasets/TS003/editions/2021/versions/3) / Nomis
* **Table**: TS003 - Household composition
* **Census Year**: 2021
* **Geography**: 2021 Lower Layer Super Output Areas (LSOAs)
* **Local Source File**: `scripts/utils/source_data/TS003_household_composition/TS003 - Household composition.csv`
* **Boundary Reference**: [`datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson`](../lsoa/lower-super-output-areas-2021-5RrVTw.geojson)

## Files in this Directory
* `lincolnshire_ts003_household_composition.csv` — Filtered CSV containing all 435 Lincolnshire LSOAs with total household counts, category counts, and percentage shares.

## Data Columns
1. `LSOA_Code`: 2021 9-character LSOA code (e.g. `E01026007`).
2. `LSOA_Name`: 2021 LSOA name (e.g. `Boston 001A`).
3. `Total_All_Households`: Total number of occupied households.
4. **One-person households**:
   * `One-person household_Count` & `_Pct`
   * `One-person household: Aged 66 years and over_Count` & `_Pct` (Single older adults / isolation risk)
   * `One-person household: Other_Count` & `_Pct`
5. **Single family households**:
   * `Single family household_Count` & `_Pct`
   * `Single family household: All aged 66 years and over_Count` & `_Pct` (All pensioner couples)
   * `Single family household: Married or civil partnership couple_Count` & `_Pct`
   * `Single family household: Married or civil partnership couple: No children_Count` & `_Pct`
   * `Single family household: Married or civil partnership couple: Dependent children_Count` & `_Pct`
   * `Single family household: Married or civil partnership couple: All children non-dependent_Count` & `_Pct`
   * `Single family household: Cohabiting couple family_Count` & `_Pct`
   * `Single family household: Cohabiting couple family: No children_Count` & `_Pct`
   * `Single family household: Cohabiting couple family: With dependent children_Count` & `_Pct`
   * `Single family household: Cohabiting couple family: All children non-dependent_Count` & `_Pct`
   * `Single family household: Lone parent family_Count` & `_Pct`
   * `Single family household: Lone parent family: With dependent children_Count` & `_Pct`
   * `Single family household: Lone parent family: All children non-dependent_Count` & `_Pct`
   * `Single family household: Other single family household_Count` & `_Pct`
   * `Single family household: Other single family household: Other family composition_Count` & `_Pct`
6. **Other household types**:
   * `Other household types_Count` & `_Pct`
   * `Other household types: With dependent children_Count` & `_Pct`
   * `Other household types: Other, including all full-time students and all aged 66 years and over_Count` & `_Pct`

## Lincolnshire Summary (435 LSOAs)
* **Total Households**: 337,929 across Lincolnshire
* **One-Person Households Aged 66+**: 51,757 households (15.3% average per LSOA)
* **All Pensioner Couple Households (Aged 66+)**: 41,202 households (12.2% average per LSOA)
* **Lone Parent Families with Dependent Children**: 21,807 households (6.5% average per LSOA)

## Generation Script
Generated using [`scripts/utils/process_ts003_household_composition.py`](../../scripts/utils/process_ts003_household_composition.py).


# English Indices of Deprivation 2025 (IoD2025) - Lincolnshire LSOAs

The English Indices of Deprivation 2025 (IoD2025), published by the Ministry of Housing, Communities & Local Government (MHCLG), measures relative deprivation across all 33,755 Lower Layer Super Output Areas (2021 LSOAs) in England.

This dataset has been filtered for the **435 Lower Layer Super Output Areas (2021 LSOAs)** within Lincolnshire.

## Source Dataset
* **Source**: [Ministry of Housing, Communities & Local Government - English Indices of Deprivation 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)
* **Source File**: `File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx` (Sheet: `IMD25`)
* **Local Source Path**: `scripts/utils/source_data/indices_of_deprivation_imd/2025/File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx`
* **Boundary Reference**: [`datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson`](../../lsoa/lower-super-output-areas-2021-5RrVTw.geojson)

## Files in this Directory
* `lincolnshire_imd_2025_lsoa.csv` — Full filtered CSV containing all 435 Lincolnshire 2021 LSOAs.

## Data Columns
1. `LSOA code (2021)`: 2021 9-character LSOA code (e.g., `E01026007`).
2. `LSOA name (2021)`: 2021 LSOA name (e.g., `Boston 001A`).
3. `Local Authority District code (2024)`: 2024 Local Authority District code.
4. `Local Authority District name (2024)`: District name (Boston, East Lindsey, Lincoln, North Kesteven, South Holland, South Kesteven, West Lindsey).
5. `Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)`: National rank across all 33,755 LSOAs in England (1 = most deprived).
6. `Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOA`: National decile from 1 (most deprived 10% nationally) to 10 (least deprived 10% nationally).

## Summary by District (435 Total LSOAs)
* **South Kesteven**: 84 LSOAs
* **East Lindsey**: 82 LSOAs
* **North Kesteven**: 66 LSOAs
* **Lincoln**: 60 LSOAs
* **West Lindsey**: 53 LSOAs
* **South Holland**: 51 LSOAs
* **Boston**: 39 LSOAs

## Generation Script
Generated using [`scripts/utils/process_imd_2025.py`](../../../scripts/utils/process_imd_2025.py).

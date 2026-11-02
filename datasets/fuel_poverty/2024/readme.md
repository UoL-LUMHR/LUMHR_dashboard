# Sub-regional fuel poverty data 2026 (2024 data)

[Sub-regional fuel poverty data 2026 (2024 data)](https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2026-2024-data)

## Lincolnshire extract

[lincolnshire_fuel_poverty_2024.csv](lincolnshire_fuel_poverty_2024.csv) is the LSOA-level Table 4 data filtered to the 435 LSOA codes in the Lincolnshire 2021 GeoJSON. It was generated with [extract_lincolnshire_fuel_poverty.py](../../../scripts/utils/extract_lincolnshire_fuel_poverty.py).

## England dashboard data

[england_fuel_poverty_2024.csv](england_fuel_poverty_2024.csv) contains the LSOA code and 2024 fuel-poverty percentage for all 33,755 England LSOAs. The Flask maps read this compact CSV from `datasets/`; they do not read the source workbook or files under `scripts/utils/source_data`.

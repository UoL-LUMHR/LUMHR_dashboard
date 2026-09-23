# Properties not connected to the gas network

[LSOA estimates of properties not connected to the gas network](https://www.gov.uk/government/statistics/lsoa-estimates-of-households-not-connected-to-the-gas-network)

## Lincolnshire extracts

The CSV files in this folder contain the 435 LSOAs represented by the Lincolnshire 2021 LSOA GeoJSON. There is one file for each year from 2015 to 2024:

lincolnshire_properties_not_connected_to_gas_network_YYYY.csv

The extracts were generated from the source workbook with [extract_lincolnshire_properties_not_connected_to_gas_network.py](../../scripts/utils/extract_lincolnshire_properties_not_connected_to_gas_network.py). The source workbook and its documentation are in [scripts/utils/source_data/properties_not_connected_to_gas_network](../../scripts/utils/source_data/properties_not_connected_to_gas_network/).
The gas-grid percentage is exported as percentage points: for example, the workbook value 0.062 is represented as 6.2 in the CSV. Blank source values remain blank because they are suppressed for disclosure control.

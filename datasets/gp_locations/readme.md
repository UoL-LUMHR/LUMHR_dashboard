# GP Practice Locations

The Flask and Streamlit dashboards use `GP_Locations_epraccur.csv`, the GP practices export from the [NHS Organisation Data Service (ODS)](https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads/gp-and-gp-practice-related-data).

The loaders keep active records whose `RoleID` includes `RO76` (GP Practice). ODS supplies practice postcodes rather than map coordinates. The postcode is matched to the 2021 LSOA using the February 2026 best-fit lookup in `scripts/utils/source_data/Postcode_to_OA_(2021)_to_LSOA_to_MSOA_to_LAD_(February_2026)_Best_Fit_Lookup_in_the_UK/`. Each map places a practice at a representative point within its postcode LSOA, not at an exact practice coordinate. Only practices whose postcode LSOA is within the selected map boundary are shown as pins. Practices outside that boundary can still contribute registration data to the area's LSOA allocations.

ODS describes this export as prescribing cost-centre data, with GP practices identified as a subset by the non-primary role `RO76`. The ODS report is dynamic and its source data updates nightly; replace this CSV with a fresh export when updating the location data.

The `*_ICB_GPs.csv` files are older county-specific extracts and are no longer used by dashboard code.


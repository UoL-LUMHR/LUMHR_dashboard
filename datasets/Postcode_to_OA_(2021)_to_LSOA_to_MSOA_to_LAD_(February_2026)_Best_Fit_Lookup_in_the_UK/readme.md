# February 2026 postcode-to-LSOA lookup

This folder contains the February 2026 UK postcode-to-OA/LSOA/MSOA/LAD best-fit lookup split into smaller CSV files for GitHub. The original columns and rows are retained, with the header repeated in each part. Each source row is assigned to exactly one file using its `ladcd` value.

The Flask and Streamlit loaders and the county-input preparation script read these split files directly in chunks; they do not need the original large CSV or a concatenated copy.

English rows are divided into the nine English regions. Scotland, Wales and Northern Ireland each have a file. The source's pseudo-LAD rows for the Channel Islands and Isle of Man are kept separately. Rows without a LAD code are in `postcode_to_lsoa_2021_unassigned_lad.csv`.

All parts are below GitHub's 100 MB per-file limit; the largest part is the South East file at about 52 MB. The large original CSV is in the ignored `scripts/utils/source_data/` directory and does not need to be committed.

## Files

- `postcode_to_lsoa_2021_england_<region>.csv` — one file for each of the nine English regions.
- `postcode_to_lsoa_2021_scotland.csv`
- `postcode_to_lsoa_2021_wales.csv`
- `postcode_to_lsoa_2021_northern_ireland.csv`
- `postcode_to_lsoa_2021_channel_islands.csv`
- `postcode_to_lsoa_2021_isle_of_man.csv`
- `postcode_to_lsoa_2021_unassigned_lad.csv`
- `LAD22_to_RGN22_England_December_2022.csv` — ONS local authority to region crosswalk used for England.

The crosswalk's December 2022 vintage predates six English LAD codes introduced in 2023. The splitter assigns those codes to their current regions: Cumberland and Westmorland and Furness to North West; North Yorkshire, Barnsley and Sheffield to Yorkshire and the Humber; Somerset to South West.

## Rebuild

With the full source CSV present at its ignored source-data path, run from the repository root:

```bash
python scripts/utils/split_postcode_to_lsoa_lookup.py
```

The script uses only Python's standard library. It writes all area files to this folder and prints row counts and file sizes.

## Sources

- Input: `scripts/utils/source_data/Postcode_to_OA_(2021)_to_LSOA_to_MSOA_to_LAD_(February_2026)_Best_Fit_Lookup_in_the_UK/PCD_OA21_LSOA21_MSOA21_LAD_FEB26_UK_LU.csv`.
- English region crosswalk: [ONS Local Authority District to Region (December 2022) Lookup in England](https://www.data.gov.uk/dataset/19df2c89-3df5-4ab6-8429-5f6ca38884d7/local-authority-district-to-region-december-2022-lookup-in-en).

import pandas as pd
import geopandas as gpd

# -----------------------------
# Input / Output
# -----------------------------
samhi_csv = "../../datasets/samhi/samhi_21_01_v5.00_2011_2022_LSOA.csv"
lookup_csv = "../../datasets/lsoa/lsoa_2011_to_2021_lookup/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
lincolnshire_geojson = "../../datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson"
output_csv = "../../datasets/samhi/samhi_lincolnshire_2021_lsoa.csv"

# Load Lincolnshire 2021 LSOAs
gdf = gpd.read_file(lincolnshire_geojson)
lincolnshire_lsoa21 = set(gdf["CODE"].astype(str).str.strip())

print(f"Found {len(lincolnshire_lsoa21)} Lincolnshire LSOA 2021 areas")

# Load lookup
lookup = pd.read_csv(lookup_csv)

lookup["LSOA11CD"] = lookup["LSOA11CD"].astype(str).str.strip()
lookup["LSOA21CD"] = lookup["LSOA21CD"].astype(str).str.strip()

# Keep only Lincolnshire 2021 LSOAs
lookup_lincolnshire = lookup[
    lookup["LSOA21CD"].isin(lincolnshire_lsoa21)
].copy()

print(f"Lookup rows: {len(lookup_lincolnshire)}")

# Load SAMHI
samhi = pd.read_csv(samhi_csv)
samhi["lsoa11"] = samhi["lsoa11"].astype(str).str.strip()

# Join 2011 SAMHI onto 2021 LSOAs
samhi_2021 = lookup_lincolnshire.merge(
    samhi,
    left_on="LSOA11CD",
    right_on="lsoa11",
    how="left"
)

print(f"Rows after join: {len(samhi_2021)}")
print(f"Rows with missing SAMHI: {samhi_2021['lsoa11'].isna().sum()}")

# Save
samhi_2021.to_csv(output_csv, index=False)

print(f"Saved {output_csv}")
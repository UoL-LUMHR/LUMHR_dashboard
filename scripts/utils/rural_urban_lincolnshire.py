import pandas as pd
import geopandas as gpd

# --- Configuration ---
# I have kept the file paths consistent with your previous directory structure
geojson_path = '../../datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson'
ruc_csv_path = 'source_data/rural_urban_classification_2021_lsoa/Rural_Urban_Classification_(2021)_of_LSOAs_in_EW.csv'
output_filename = 'lincolnshire_rural_urban_2021.csv'

# 1. Get Lincolnshire LSOA 2021 codes directly from the GeoJSON
print("Loading Lincolnshire LSOAs from GeoJSON...")
gdf = gpd.read_file(geojson_path)
lsoa21_col = 'CODE' if 'CODE' in gdf.columns else 'LSOA21CD'
lincolnshire_lsoa21_codes = gdf[lsoa21_col].unique()
print(f"Loaded {len(lincolnshire_lsoa21_codes)} Lincolnshire LSOA 2021 codes.\n")

# 2. Load the Rural/Urban Classification CSV
print(f"Loading {ruc_csv_path}...")
df = pd.read_csv(ruc_csv_path)

# 3. Automatically hunt for the correct LSOA Code column (e.g., 'LSOA21CD')
lsoa_col_name = None
for col in df.columns:
    col_str = str(col).upper()
    # Looking for the code column, intentionally avoiding name/description columns
    if 'LSOA' in col_str and 'NAME' not in col_str and 'NM' not in col_str:
        lsoa_col_name = col
        break

if lsoa_col_name is None:
    print("Error: Could not automatically detect the LSOA column in the CSV.")
else:
    print(f"Found LSOA column: '{lsoa_col_name}'. Filtering data...")
    
    # 4. Filter the dataset to only include Lincolnshire LSOAs
    filtered_df = df[df[lsoa_col_name].isin(lincolnshire_lsoa21_codes)]
    
    if not filtered_df.empty:
        # 5. Save to a new CSV
        filtered_df.to_csv(output_filename, index=False)
        print(f"Success! Saved {len(filtered_df)} rows to '{output_filename}'.")
    else:
        print("Warning: Filtering resulted in 0 rows. Please check if the LSOA formats match.")

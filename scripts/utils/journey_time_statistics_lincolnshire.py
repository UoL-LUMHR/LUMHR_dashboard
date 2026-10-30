import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm

# --- Configuration ---
geojson_path = '../../datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson'
lookup_path = '../../datasets/lsoa/lsoa_2011_to_2021_lookup/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv'

jts_files = [
    'source_data/journey_time_statistics/jts0505 LSOA Travel Time to GPs.ods',
    'source_data/journey_time_statistics/jts0506 LSOA Travel Time to Hospitals.ods',
    'source_data/journey_time_statistics/jts0509 LSOA Travel Time to Pharmacies.ods'
]

output_dir = 'lincolnshire_jts_extracts'
os.makedirs(output_dir, exist_ok=True)

# 1. Get Lincolnshire LSOA 2021 codes
print("Loading Lincolnshire LSOAs from GeoJSON...")
gdf = gpd.read_file(geojson_path)
lsoa21_col = 'CODE' if 'CODE' in gdf.columns else 'LSOA21CD'
lincolnshire_lsoa21_codes = gdf[lsoa21_col].unique()

# 2. Map LSOA 2021 -> LSOA 2011
print("Loading LSOA 2011 to 2021 lookup...")
lookup_df = pd.read_csv(lookup_path)
lincolnshire_lookup = lookup_df[lookup_df['LSOA21CD'].isin(lincolnshire_lsoa21_codes)]
lincolnshire_lsoa11_codes = lincolnshire_lookup['LSOA11CD'].unique()

print(f"Found {len(lincolnshire_lsoa21_codes)} LSOA21 codes mapping to {len(lincolnshire_lsoa11_codes)} LSOA11 codes.\n")

# 3. Process each JTS dataset with tqdm
for jts_file in tqdm(jts_files, desc="Overall JTS Files", position=0):
    if not os.path.exists(jts_file):
        tqdm.write(f"Warning: File {jts_file} not found. Skipping.")
        continue
        
    xls = pd.ExcelFile(jts_file, engine='calamine')
    sheet_names = xls.sheet_names
    dataset_name = os.path.splitext(os.path.basename(jts_file))[0]
    
    # Inner progress bar for sheets within the current file
    for sheet in tqdm(sheet_names, desc=f"Sheets in {dataset_name}", leave=False, position=1):
        if sheet.lower() in ['notes', 'metadata', 'cover', 'contents']:
            continue
            
        # Read the sheet WITHOUT assuming row 0 is the header
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        
        header_row_idx = None
        lsoa_col_name = None
        
        # Scan the first 20 rows to find the actual LSOA column header
        for idx, row in df.head(20).iterrows():
            for col_val in row.values:
                col_str = str(col_val).strip()
                
                # Check for exact matches including the underscore, or a broad inclusion check
                if col_str.upper() in ['LSOA_CODE', 'LSOA CODE', 'LSOA11', 'LSOA11CD', 'LSOA']:
                    header_row_idx = idx
                    lsoa_col_name = col_val
                    break
            if header_row_idx is not None:
                break
                
        # If we still can't find an LSOA column, warn the user and skip
        if header_row_idx is None:
            tqdm.write(f"  -> Skipping '{sheet}': No LSOA column found in the first 20 rows.")
            continue
            
        # Promote the found row to be the dataframe's official header
        df.columns = df.iloc[header_row_idx]
        df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
        
        # Filter for Lincolnshire matches
        filtered_df = df[df[lsoa_col_name].isin(lincolnshire_lsoa11_codes)]
        
        if not filtered_df.empty:
            out_filename = f"{dataset_name}_sheet_{sheet}.csv"
            out_filepath = os.path.join(output_dir, out_filename)
            filtered_df.to_csv(out_filepath, index=False)
            tqdm.write(f"  -> Saved {len(filtered_df)} rows to {out_filename}")
        else:
            tqdm.write(f"  -> No matches found in '{sheet}'.")

print("\nProcessing complete!")
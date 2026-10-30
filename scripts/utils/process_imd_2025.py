import json
from pathlib import Path
import pandas as pd

# Paths
script_dir = Path(__file__).resolve().parent
source_excel_path = (
    script_dir
    / "source_data"
    / "indices_of_deprivation_imd"
    / "2025"
    / "File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx"
)
geojson_path = script_dir / "../../datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson"
output_dir = script_dir / "../../datasets/indices_of_deprivation_imd/2025"
output_dir.mkdir(parents=True, exist_ok=True)

output_csv_path = output_dir / "lincolnshire_imd_2025_lsoa.csv"

# 1. Load Lincolnshire 2021 LSOA codes from GeoJSON
print(f"Loading Lincolnshire LSOAs from {geojson_path}...")
with open(geojson_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

lsoa_codes = set()
for feat in gj.get("features", []):
    props = feat.get("properties", {})
    code = props.get("LSOA21CD") or props.get("CODE")
    if code:
        lsoa_codes.add(str(code).strip())

print(f"Found {len(lsoa_codes)} distinct Lincolnshire LSOAs in GeoJSON.")

# 2. Read IMD 2025 Excel
print(f"Reading {source_excel_path} (sheet: IMD25)...")
df_imd = pd.read_excel(source_excel_path, sheet_name="IMD25")
print(f"Loaded {len(df_imd)} rows from Excel.")

# 3. Detect LSOA code column
code_col = None
for c in df_imd.columns:
    if "LSOA" in str(c).upper() and "CODE" in str(c).upper():
        code_col = c
        break

if not code_col:
    raise ValueError("Could not find LSOA code column in IMD 2025 Excel.")

print(f"Using LSOA code column: '{code_col}'")

# 4. Filter for Lincolnshire LSOAs
df_lincs = df_imd[df_imd[code_col].astype(str).str.strip().isin(lsoa_codes)].copy()
df_lincs = df_lincs.sort_values(by=code_col).reset_index(drop=True)

print(f"Matched {len(df_lincs)} / {len(lsoa_codes)} Lincolnshire LSOAs.")

missing = lsoa_codes - set(df_lincs[code_col].astype(str).str.strip())
if missing:
    print(f"Warning: {len(missing)} LSOAs not found in IMD 2025: {missing}")
else:
    print("All 435 Lincolnshire LSOAs matched successfully (0 missing).")

# 5. Save output CSV
df_lincs.to_csv(output_csv_path, index=False)
print(f"Saved: {output_csv_path}")

# 6. Print summary
print("\n--- Summary Breakdown by District ---")
dist_col = [c for c in df_lincs.columns if "district name" in str(c).lower() or "local authority district name" in str(c).lower()]
if dist_col:
    print(df_lincs[dist_col[0]].value_counts())

print("\n--- Sample Output ---")
print(df_lincs.head())

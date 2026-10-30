import csv
import json
from pathlib import Path
import pandas as pd

# Paths
script_dir = Path(__file__).resolve().parent
src_file = script_dir / "source_data" / "TS003_household_composition" / "TS003 - Household composition.csv"
geojson_path = script_dir / "../../datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson"

output_dir = script_dir / "../../datasets/TS003_household_composition"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv_path = output_dir / "lincolnshire_ts003_household_composition.csv"

# 1. Load Lincolnshire 2021 LSOA codes from GeoJSON
print(f"Loading Lincolnshire LSOAs from {geojson_path}...")
with open(geojson_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

linc_lsoas: dict[str, str] = {}
for feat in gj.get("features", []):
    props = feat.get("properties", {})
    code = props.get("LSOA21CD") or props.get("CODE")
    name = props.get("LSOA21NM") or props.get("NAME")
    if code:
        linc_lsoas[str(code).strip()] = str(name).strip() if name else ""

print(f"Found {len(linc_lsoas)} distinct Lincolnshire LSOAs in GeoJSON.")

# 2. Parse Nomis TS003 CSV
print(f"Reading source CSV from {src_file}...")
rows = []
raw_headers = []

with open(src_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 7:
            raw_headers = row
            continue
        if i < 9 or not row or not row[0].strip():
            continue

        raw_geo = row[0].strip()
        if " : " in raw_geo:
            code, name = raw_geo.split(" : ", 1)
            code = code.strip()
            name = name.strip()
        else:
            code = raw_geo
            name = ""

        if code in linc_lsoas:
            record: dict[str, object] = {
                "LSOA_Code": code,
                "LSOA_Name": name or linc_lsoas.get(code, ""),
            }

            for col_idx in range(1, len(row)):
                if col_idx >= len(raw_headers):
                    break
                header_name = raw_headers[col_idx].strip()
                if header_name == "%":
                    prev_header = raw_headers[col_idx - 1].strip()
                    col_name = f"{prev_header}_Pct"
                else:
                    if header_name == "Total: All households":
                        col_name = "Total_All_Households"
                    else:
                        col_name = f"{header_name}_Count"

                val_str = row[col_idx].strip()
                try:
                    val = float(val_str)
                    if val.is_integer() and not col_name.endswith("_Pct"):
                        val = int(val)
                except ValueError:
                    val = val_str
                record[col_name] = val

            rows.append(record)

df_out = pd.DataFrame(rows)

# Sort by LSOA_Code
df_out = df_out.sort_values(by="LSOA_Code").reset_index(drop=True)

# Drop redundant 100% total column if present
if "Total: All households_Pct" in df_out.columns:
    df_out = df_out.drop(columns=["Total: All households_Pct"])

# Verify
print(f"Extracted {len(df_out)} LSOAs and {len(df_out.columns)} columns.")
assert len(df_out) == len(linc_lsoas), f"Expected {len(linc_lsoas)} LSOAs, got {len(df_out)}"

# Save CSV
df_out.to_csv(output_csv_path, index=False)
print(f"Saved Lincolnshire TS003 dataset to: {output_csv_path}")


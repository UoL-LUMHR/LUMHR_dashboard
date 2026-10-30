import json
from pathlib import Path
import pandas as pd

# Paths
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "source_data" / "lsoa_classification_2021_2"
lookup_path = src_dir / "lsoa_dz_sdz_lookup.csv"
codes_path = src_dir / "classification_codes_and_names_ukoac_lsoac.csv"
geojson_path = script_dir / "../../datasets/lsoa/lower-super-output-areas-2021-5RrVTw.geojson"

output_dir = script_dir / "../../datasets/lsoa_classification_2021_2"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv_path = output_dir / "lincolnshire_lsoa_classification_2021_2.csv"

# 1. Load Lincolnshire 2021 LSOA codes from GeoJSON
print(f"Loading Lincolnshire LSOAs from {geojson_path}...")
with open(geojson_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

lsoa_geojson = {}
for feat in gj.get("features", []):
    props = feat.get("properties", {})
    code = props.get("LSOA21CD") or props.get("CODE")
    name = props.get("LSOA21NM") or props.get("NAME")
    if code:
        lsoa_geojson[str(code).strip()] = str(name).strip() if name else ""

print(f"Found {len(lsoa_geojson)} distinct Lincolnshire LSOAs in GeoJSON.")

# 2. Read Classification Lookup and Code Names
print(f"Reading lookup from {lookup_path}...")
df_lookup = pd.read_csv(lookup_path)
df_lookup["geography_code"] = df_lookup["geography_code"].astype(str).str.strip()
df_lookup["Supergroup"] = df_lookup["Supergroup"].astype(str).str.strip()
df_lookup["Group"] = df_lookup["Group"].astype(str).str.strip()
df_lookup["Subgroup"] = df_lookup["Subgroup"].astype(str).str.strip()

print(f"Reading classification codes and names from {codes_path}...")
df_codes = pd.read_csv(codes_path)
df_codes["Classification Code"] = df_codes["Classification Code"].astype(str).str.strip()
df_codes["Classification Name"] = df_codes["Classification Name"].astype(str).str.strip()

supergroup_map = dict(
    zip(
        df_codes[df_codes["Level"] == "Supergroup"]["Classification Code"],
        df_codes[df_codes["Level"] == "Supergroup"]["Classification Name"],
    )
)
group_map = dict(
    zip(
        df_codes[df_codes["Level"] == "Group"]["Classification Code"],
        df_codes[df_codes["Level"] == "Group"]["Classification Name"],
    )
)
subgroup_map = dict(
    zip(
        df_codes[df_codes["Level"] == "Subgroup"]["Classification Code"],
        df_codes[df_codes["Level"] == "Subgroup"]["Classification Name"],
    )
)

# 3. Filter for Lincolnshire LSOAs and enrich with names
df_lincs = df_lookup[df_lookup["geography_code"].isin(lsoa_geojson.keys())].copy()
df_lincs["LSOA_Name"] = df_lincs["geography_code"].map(lsoa_geojson)
df_lincs["Supergroup_Name"] = df_lincs["Supergroup"].map(supergroup_map)
df_lincs["Group_Name"] = df_lincs["Group"].map(group_map)
df_lincs["Subgroup_Name"] = df_lincs["Subgroup"].map(subgroup_map)

# Organize columns
out_df = pd.DataFrame(
    {
        "LSOA_Code": df_lincs["geography_code"],
        "LSOA_Name": df_lincs["LSOA_Name"],
        "Supergroup_Code": df_lincs["Supergroup"],
        "Supergroup_Name": df_lincs["Supergroup_Name"],
        "Group_Code": df_lincs["Group"],
        "Group_Name": df_lincs["Group_Name"],
        "Subgroup_Code": df_lincs["Subgroup"],
        "Subgroup_Name": df_lincs["Subgroup_Name"],
    }
).sort_values(by="LSOA_Code").reset_index(drop=True)

print(f"Matched {len(out_df)} / {len(lsoa_geojson)} Lincolnshire LSOAs.")

missing = set(lsoa_geojson.keys()) - set(out_df["LSOA_Code"])
if missing:
    print(f"Warning: {len(missing)} LSOAs missing: {missing}")
else:
    print("All 435 Lincolnshire LSOAs matched successfully (0 missing).")

# 4. Save to CSV
out_df.to_csv(output_csv_path, index=False)
print(f"Saved: {output_csv_path}")

# 5. Print summary
print("\n--- Top Supergroups in Lincolnshire ---")
print(out_df["Supergroup_Name"].value_counts())

print("\n--- Top Groups in Lincolnshire ---")
print(out_df["Group_Name"].value_counts().head(10))

print("\n--- Sample Output ---")
print(out_df.head())


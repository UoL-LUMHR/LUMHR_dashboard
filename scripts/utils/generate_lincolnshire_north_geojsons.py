"""Build North Lincolnshire, North East Lincolnshire, and Greater Lincolnshire LSOA boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


AREA_LADS = {
    "north_lincolnshire": {"North Lincolnshire"},
    "north_east_lincolnshire": {"North East Lincolnshire"},
    "greater_lincolnshire": {
        "Boston", "East Lindsey", "Lincoln", "North Kesteven", "South Holland",
        "South Kesteven", "West Lindsey", "North Lincolnshire", "North East Lincolnshire",
    },
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    lsoa_dir = root / "datasets" / "lsoa"
    source_path = lsoa_dir / "england_lsoa_2021.geojson"
    lookup_path = (
        lsoa_dir / "lsoa_2011_to_2021_lookup"
        / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    with source_path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    lookup = pd.read_csv(lookup_path, usecols=["LSOA21CD", "LAD22NM"], dtype=str)
    lookup["LSOA21CD"] = lookup["LSOA21CD"].astype(str).str.strip()
    lookup["LAD22NM"] = lookup["LAD22NM"].astype(str).str.strip()
    lad_by_lsoa = lookup.drop_duplicates("LSOA21CD").set_index("LSOA21CD")["LAD22NM"].to_dict()

    for scope, lad_names in AREA_LADS.items():
        features = []
        for feature in source.get("features", []):
            code = str(feature.get("properties", {}).get("LSOA21CD", "")).strip()
            lad_name = lad_by_lsoa.get(code)
            if lad_name not in lad_names:
                continue
            selected = dict(feature)
            selected["properties"] = {**feature.get("properties", {}), "LAD22NM": lad_name}
            features.append(selected)
        output = {"type": "FeatureCollection", "features": features}
        output_path = lsoa_dir / f"{scope}_lsoa_2021.geojson"
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False, separators=(",", ":"))
        print(f"{output_path.name}: {len(features):,} LSOAs")


if __name__ == "__main__":
    main()

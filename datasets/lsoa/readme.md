# LSOA boundaries and lookup

This folder contains the 2021 Lower Layer Super Output Area (LSOA) boundaries used by the dashboard and the 2011-to-2021 geography lookup.

- `england_lsoa_2021.geojson` contains England's 33,755 LSOAs and is used by the England National map views. It is filtered from the England-and-Wales boundary file in `../england_lsoa/` using English LAD codes in the lookup below.
- `lower-super-output-areas-2021-5RrVTw.geojson` contains the 435 LSOAs used for the Lincolnshire dashboard scope. Source: [Lincolnshire County Council LSOA data](https://data.lincolnshire.gov.uk/@non-lincolnshire-county-council/boundaries/r/8a175023-7265-492a-8719-60de1173b366).
- `nottinghamshire_lsoa_2021.geojson` contains LSOAs for Nottinghamshire, including Nottingham.
- `derbyshire_lsoa_2021.geojson` contains LSOAs for Derbyshire, including Derby.
- `lincolnshire_nottinghamshire_derbyshire_lsoa_2021.geojson` combines those three areas.
- `north_lincolnshire_lsoa_2021.geojson` contains North Lincolnshire LSOAs.
- `north_east_lincolnshire_lsoa_2021.geojson` contains North East Lincolnshire LSOAs.
- `greater_lincolnshire_lsoa_2021.geojson` combines Lincolnshire, North Lincolnshire, and North East Lincolnshire.
- `lsoa_2011_to_2021_lookup/` contains the England lookup used to map model records from 2011 LSOA codes to 2021 LSOAs and local authority districts.

The Nottinghamshire, Derbyshire, and combined GeoJSON files are selected from the England-wide 2021 LSOA boundaries in this folder using local authority names in the lookup table. The North Lincolnshire, North East Lincolnshire, and Greater Lincolnshire files use the same source and lookup. Their feature properties include `LSOA21CD`, `LSOA21NM`, and `LAD22NM`. The England-only file retains just `LSOA21CD` and `LSOA21NM` with each feature's original geometry to reduce the national map payload.

Regenerate the three Lincolnshire-related files with `scripts/utils/generate_lincolnshire_north_geojsons.py`.

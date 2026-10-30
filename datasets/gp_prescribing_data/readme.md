# OpenPrescribing Data

The dashboard-ready `need_index_antidepressant_items_per_gp_*_may_2026.csv` files select the May 2026 rows and normalize the county exports to `id`, `name`, `items`, and `actual_cost`. The original supplied county series are retained unchanged. Nottinghamshire and the combined Lincolnshire, Nottinghamshire and Derbyshire area also include May 2026 rows from `items_for_antidepressant_drugs_per_gp_bassetlaw_may_2026.csv`, including Bassetlaw GP practice codes absent from the Nottinghamshire export.

May prescribing rows are matched by practice code, including relevant rows from neighboring area exports when a practice serves registrations across an area boundary. For each LSOA, the map calculates items per patient from practices with a matched May prescribing row and reports the share of registered patients covered by those rows. Registrations for practices without a May row are excluded from that metric's denominator; missing prescribing data is never treated as zero. The popup's coverage figure should be read alongside the rate.

* **Source**: OpenPrescribing practice-level antidepressant prescribing exports. [Lincolnshire ICB (71E) example](https://openprescribing.net/analyse/#org=practice&orgIds=71E&numIds=4.3&denom=nothing&selectedTab=map).

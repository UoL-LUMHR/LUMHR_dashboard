# NHS Patients Registered at a GP Practice

The county-specific folders contain July 2026 practice-to-LSOA records for active ODS GP practices whose postcodes fall within each selected 2021 LSOA boundary and every practice with patients registered in an LSOA within that boundary. They retain each selected practice's registrations inside and outside the boundary so its full mapped list is used for allocation. This includes cross-boundary practices and prevents results for the same LSOA changing simply because a broader geographic area was selected.

For Derbyshire and the combined Lincolnshire, Nottinghamshire and Derbyshire scope, Parkfields Surgery (Y05733) is an explicit exception: its practice-to-LSOA allocation uses the January 2026 England registration rows because its July 2026 figures show an identified mismatch. The generated county mapping records both source dates/counts, and affected LSOA popups show the January and July counts. Other practices continue to use July 2026 data.

NOTSPAR (C84720), a Special Allocation Scheme practice, remains in the Nottinghamshire and combined practice-registration mapping so its GP marker can show its location and available practice data. Its registrations and practice-level indicators are excluded from LSOA-level Need Index calculations and GP registration totals. The marker popup explains this exclusion.

Regenerate the county extracts with `scripts/utils/process_county_need_index_inputs.py`.

Quarterly publications in January, April, July and October will include Lower Layer Super Output Area (LSOA) populations.

[NHS Digital Patients Registered at a GP Practice](https://digital.nhs.uk/data-and-information/publications/statistical/patients-registered-at-a-gp-practice)

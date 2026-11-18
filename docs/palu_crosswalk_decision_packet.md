# Palu mapping decision packet (not a frozen mapping)

Status: **PENDING Jlin decision**. This document is a decision packet.
It is not a frozen Palu mapping and must not be copied into
`docs/eval_protocol.md` as a freeze.

- Ledger sha256 (canonical YAML, sorted keys): `d481ba61d474a8f38d74a5aacf5edcf46cc5d35cf8550a2650c5e3972b2daac3`
- Ledger purpose: `decision_packet_for_jlin_not_a_frozen_mapping`
- Palu mapping status: `pending_decision` (frozen=False)
- Cross-hazard transfer established: `False`
- Compute U or model-based conflicts: `False`

## 1. Paulik 2019 files are on WSA; Palu mapping remains PENDING Jlin decision

- Source id: `paulik_2019`
- Citation: Paulik et al., 2019, Pure and Applied Geophysics 176:3305–3321, doi:10.1007/s00024-019-02254-9
- Obtainment status: **OBTAINED_ON_WSA_AWAITING_ADJUDICATION**
- Full text obtained: `True`
- Blocks Palu freeze: `True`
- Notes: Official PDF and publisher survey table are now on WSA under /mnt/data_hdd/datasets/paulik_palu_2018/. Inventory (paths, hashes, sheet counts, and PDF grade definitions) is recorded below. Official docs/eval_protocol.md still says full text not obtained and is not edited here. Palu mapping stays unfrozen until Jlin adjudicates official Y from Paulik's own DL0-DL3 definitions. Do not treat this obtainment as a freeze.

The official 17-page PDF and the publisher Buildings sheet are inventoried
on WSA. File obtainment is not a Palu freeze. Official protocol text is not
edited here. Jlin still has to adjudicate official Y from Paulik's own
DL0-DL3 definitions. Earlier DS1/DS2/DS3 candidate schemes are not the
mapping and must not be applied.

## 2. What stays frozen (do not reopen)

Wind events continue to use the already-frozen StEER wind package in
`docs/eval_protocol.md`. This packet does not change StEER wind, CAL FIRE
DINS Affected → no, or the frozen FEMA mapping.

- `steer_wind` (wind): status=`approved`, frozen=`True`.
  Source: Prevatt and Roueche, 15 June 2020, Hurricane Michael Data Enhancement (Phase II), Report No. 06-20, University of Florida, for Florida DBPR, Project P0157245, Table 4 and Table 5, printed p. 15.
  Code F from component fields, never from the grade.
- `dins_affected_to_no` (wildfire): status=`approved`, frozen=`True`.
- `fema` (multi): status=`approved`, frozen=`True`.

## 3. Official protocol position on Palu (unchanged)

The official protocol already records a Palu non-mapping. This worktree
does not edit that section into a freeze. Quoted from
`docs/eval_protocol.md`:

> Palu is not one-to-one and is not frozen until Paulik et al. (2019) is in hand.
>
> DS1 → unknown or excluded from a strict crosswalk.
>
> DS2/DS3 only as exploratory sensitivity.
>
> Paulik et al., 2019, Pure and Applied Geophysics 176:3305–3321, doi:10.1007/s00024-019-02254-9. Full text not obtained.

## 4. Literature sources, versions, and obtainment

| Id | Role | Version or locus | Obtainment | May define official Y |
| --- | --- | --- | --- | --- |
| paulik_2019 | primary_palu_field_survey | 176:3305-3321 | obtained_on_wsa_awaiting_adjudication | True |
| lahcene_2021 | secondary_palu_only | 21:2315-2316 | cited_in_protocol | False |
| jalayer_2023 | secondary_palu_only | 23:914 | cited_in_protocol | False |
| prevatt_roueche_2020_report_06_20 | steer_wind_frozen_source | Report 06-20 | obtained | False |
| roueche_2024_nhr | steer_component_field_tables | 25(2):04023060 | obtained | False |
| gupta_2019_xbd | official_xbd_wording | CVPR Workshops 2019 | obtained | False |

Citations:

- `paulik_2019`: Paulik et al., 2019, Pure and Applied Geophysics 176:3305–3321, doi:10.1007/s00024-019-02254-9
  Official PDF and publisher survey table are now on WSA under /mnt/data_hdd/datasets/paulik_palu_2018/. Inventory (paths, hashes, sheet counts, and PDF grade definitions) is recorded below. Official docs/eval_protocol.md still says full text not obtained and is not edited here. Palu mapping stays unfrozen until Jlin adjudicates official Y from Paulik's own DL0-DL3 definitions. Do not treat this obtainment as a freeze.
- `lahcene_2021`: Lahcene et al., 2021, NHESS 21:2315–2316, Table 1
  Secondary Palu source only. It does not define official Y and must not be used in place of the official Paulik PDF and publisher Buildings sheet.
- `jalayer_2023`: Jalayer et al., 2023, NHESS 23:914, Table 2
  Secondary Palu source only. Jalayer's phrase "complete structural collapse" is Jalayer's wording, not a verified Paulik definition.
- `prevatt_roueche_2020_report_06_20`: Prevatt and Roueche, 15 June 2020, Hurricane Michael Data Enhancement (Phase II), Report No. 06-20, University of Florida, for Florida DBPR, Project P0157245, Table 4 and Table 5, printed p. 15
  Source of the already-frozen StEER wind grade-4 promotion rule. Wind events continue to use this package.
- `roueche_2024_nhr`: Roueche, Nakayama, Kijewski-Correa, Prevatt, 2024, Natural Hazards Review 25(2):04023060, printed page 04023060-6, Tables 3 and 4
  Source for the component-field tables. Which cells of WiSPD Table 3 are bold is unverified. Do not cite those bold cells as if read.
- `gupta_2019_xbd`: Gupta et al., 2019, Creating xBD, CVPR Workshops, conference proceedings pp. 12–13, Figure 3
  Official xBD labels are never Y. The Joint Damage Scale is not an authoritative ground assessment rating.

## 5. Date and phase alignment

- Event id: `palu-tsunami-2018`
- Catalog window: 2018-09-18 to 2018-09-18
- Catalog source: configs/event_catalog.yaml, sourced as Gupta et al. Table 1, tier 1, xView2 slug palu-tsunami
- Commonly cited event date: 2018-09-28
- Cited-date status: `verified_in_paulik_2019_pdf_page_1_journal_3305`
- Paulik event date (from official PDF): 2018-09-28
- Paulik event-date source: Official PDF p. 1 (journal 3305): On Friday 28 September 2018, at 18:02 p.m. local time, a shallow strike-slip earthquake of moment magnitude 7.5 occurred near Palu City, Central Sulawesi, Indonesia. The paper title also uses after the September 28 2018 Sulawesi Earthquake and Tsunami.
- Survey dates: 2018-11-13 to 2018-11-17
- Survey-date source: Official PDF p. 5 (journal 3309), Table 2 caption: A summary of information collected for tsunami hazard and damaged structures in Palu City between 13th to 17th of November 2018. Abstract (PDF p. 1) also says a field survey was carried out seven weeks after the earthquake.
- Imagery vs survey phase: open_catalog_window_still_2018-09-18_do_not_impute_imagery_dates

Open questions:

- Reconcile configs/event_catalog.yaml window 2018-09-18..2018-09-18 (Gupta Table 1) with Paulik's 28 September 2018 event date. This packet verifies Paulik's date from the official PDF but does not silently rewrite the official catalog.
- Align post-event imagery dates with the 13-17 November 2018 field-survey window once imagery dates are documented. Missing imagery dates stay missing.

## 6. Footprint checks

The general footprint-to-reference rules are already frozen in the
protocol. Palu uses the same numeric gates when a ground table is available.
This packet does not invent a Palu-specific IoU threshold.

- Match if IoU ≥ 0.5
- Ambiguous if two or more candidates have IoU ≥ 0.3, or the best/second-best score ratio is < 1.2
- Drop a ground point when localization error > 30 m
- Palu footprint layer path: `None`
- Palu ground table path: `/mnt/data_hdd/datasets/paulik_palu_2018/ground_tables/24_2019_2254_MOESM1_ESM.xlsx`
- Notes: Numeric gates are the already-frozen protocol rules. The publisher Buildings sheet is the authorized Palu ground table and stays outside git. No Palu-specific official footprint layer path is entered. MSBFP2 inventory is recorded under authorized_storage and is not a Palu mapping source.

## 7. Unresolved classes

Paulik's official classes are DL0-DL3. They stay unresolved as official Y
until Jlin adjudicates. DS0-DS3 are not the official grade system and
must not be applied as the mapping.

| Class | Status | Official definition, guidance, or notes |
| --- | --- | --- |
| DL0 | unresolved | No damage (PDF Table 1, p. 4 / journal 3308) |
| DL1 | unresolved | Partial damage, repairable (PDF Table 1, p. 4 / journal 3308) |
| DL2 | unresolved | Partial damage, unrepairable (PDF Table 1, p. 4 / journal 3308) |
| DL3 | unresolved | Complete damage (PDF Table 1, p. 4 / journal 3308) |
| DS0 | not_the_official_paulik_grade_system | Do not apply. Official Paulik grades are DL0-DL3. |
| DS1 | not_the_official_paulik_grade_system | Do not apply. Official Paulik grades are DL0-DL3. |
| DS2 | not_the_official_paulik_grade_system | Do not apply. Official Paulik grades are DL0-DL3. |
| DS3 | not_the_official_paulik_grade_system | Do not apply. Official Paulik grades are DL0-DL3. |
| other_or_missing | unresolved | Leave unresolved until Jlin adjudicates any residual class. |

## 8. Earlier DS candidate schemes (not the mapping; do not apply)

The official Palu grade system, checked against the official PDF Table 1,
is DL0-DL3 (No damage; Partial damage, repairable; Partial damage,
unrepairable; Complete damage). The earlier DS1/DS2/DS3 schemes below
are retained only as historical review items. They are not the mapping,
are not main conclusions, must not be applied, and must not be written
into the official protocol as frozen.

- Official grade scale: `DL0_to_DL3`
- Applied to official Y: `False`
- Frozen: `False`
- Definitions: `{'DL0': 'No damage', 'DL1': 'Partial damage, repairable', 'DL2': 'Partial damage, unrepairable', 'DL3': 'Complete damage'}`
- PDF source: PDF p. 4 (journal 3308), Table 1, Assignment of damage levels to tsunami affected structures in Palu City

### `strict_ds1_unknown_exclude_ds2_ds3`

- Status: `not_applied`
- applied_as_mapping: `False`
- is_main_conclusion: `False`
- write_to_official_protocol_as_frozen: `False`
- Notes: Historical candidate only. Official Paulik grades are DL0-DL3, not DS1-DS3. This scheme must not be applied as the mapping.
- Rules:

```yaml
DS1: unknown_or_excluded_from_y
DS2: not_mapped_to_y
DS3: not_mapped_to_y
```

### `exploratory_ds2_major_ds3_destroyed`

- Status: `not_applied`
- applied_as_mapping: `False`
- is_main_conclusion: `False`
- write_to_official_protocol_as_frozen: `False`
- Notes: Historical exploratory candidate only. Official Paulik grades are DL0-DL3, not DS2-DS3. This scheme must not be applied as the mapping.
- Rules:

```yaml
DS2: major
DS3: destroyed
```

### `ds0_to_no_listed_separately`

- Status: `not_applied`
- applied_as_mapping: `False`
- is_main_conclusion: `False`
- write_to_official_protocol_as_frozen: `False`
- Notes: Historical listed-separately candidate only. Official Paulik grades are DL0-DL3, not DS0. This scheme must not be applied as the mapping.
- Rules:

```yaml
DS0: 'no'
```

## 9. Authorized storage and WSA inventory

Real Palu microdata, if located, stays in authorized storage and is not
committed to this repository. Paths and hashes below are inventory only.
Origin is reported only when it can be determined from the file or a
sidecar manifest; this packet does not guess.

- `policy`: Commit only schemas, ledgers, and synthetic fixture generators. Point at authorized paths instead of adding real Palu microdata.
- `inventory_computed`: 2026-11-18 (Asia/Shanghai)
### `official_paulik_pdf`

- found: True
- path: /mnt/data_hdd/datasets/paulik_palu_2018/literature/Paulik_et_al_2019_Palu_10.1007_s00024-019-02254-9.pdf
- filename: Paulik_et_al_2019_Palu_10.1007_s00024-019-02254-9.pdf
- sha256: a88cde36279b637f12c72e9fde34ff08c8ec3fcbc6bce601a62877236cf82c71
- pages: 17
- bytes: 3627766
- title: Tsunami Hazard and Built Environment Damage Observations from Palu City after the September 28 2018 Sulawesi Earthquake and Tsunami
- doi: 10.1007/s00024-019-02254-9
- pdfinfo_author: Ryan Paulik
- pdfinfo_creator: Springer
- origin: official_springer_pdf_metadata_and_filename_match_doi
- notes: pdfinfo reports Pages=17, File size=3627766 bytes, PDF version 1.6, Creator=Springer, Subject includes the DOI. sha256 was computed read-only on WSA on 2026-11-18 (Asia/Shanghai). The PDF is not committed to git.

### `publisher_survey_table`

- found: True
- path: /mnt/data_hdd/datasets/paulik_palu_2018/ground_tables/24_2019_2254_MOESM1_ESM.xlsx
- filename: 24_2019_2254_MOESM1_ESM.xlsx
- sha256: fc72889951d5e116e498d2ba841680e69e5297bbedbf2976cd19d097e7f7a04b
- bytes: 86437
- origin: publisher_esm_filename_matches_doi_article_2254_and_sheet_count_matches_paper
- sheets: ['Buildings', 'Roads', 'Utility Poles']
- buildings_sheet: {'header_row': 1, 'max_row': 464, 'non_empty_data_rows': 463, 'used_fields': ['ID', 'Latitude', 'Longitude', 'Flow Depth Above Ground Level (m)', 'Flow Depth Above Building First Floor Level (m)', 'Damage Level', 'Building Use', 'Building Construction Frame', 'Building Storeys', 'Building Foundation', 'Building Floor Height (m)'], 'unused_trailing_columns': 5, 'rows_with_numeric_lat_lon': 463, 'rows_with_lat_lon_and_damage_level': 463, 'unique_ids': 463, 'unique_latitudes': 461, 'unique_longitudes': 460, 'damage_level_frequencies': {'No Damage': 21, 'Partial Damage, Repairable': 250, 'Partial Damage, Unrepairable': 94, 'Complete Damage': 98}, 'building_use_frequencies': {'Residential': 278, 'Commercial': 82, 'Education': 27, 'Unknown': 24, 'Hotel, Resort': 14, 'Industrial': 8, 'Religious': 7, 'Community': 6, 'Government': 6, 'Agricultural': 6, 'Hospital, Clinic': 4, 'Emergency Service': 1}, 'construction_frame_frequencies': {'Masonry, Unreinforced clay brick': 391, 'Timber/Wood': 32, 'Reinforced Concrete': 20, 'Masonry, Unknown': 13, 'Steel': 4, 'Masonry, Unreinforced, with manufacture stone units': 3}, 'storeys_frequencies': {'1': 417, '2': 43, '3': 3}, 'foundation_frequencies': {'Earthen, filled perimeter wall': 448, 'Concrete Slab': 15}, 'notes': 'Verified on WSA 2026-11-18 (Asia/Shanghai) with openpyxl data_only=True. 463 building-level rows all have numeric Latitude, numeric Longitude, and a Damage Level. This matches the paper statement (PDF p. 1 abstract and Table 2 Building storeys frequency 463) that the survey recorded 463 buildings. Damage Level labels match PDF Table 1 definitions. PDF Table 2 lists Commercial 81 and Unknown 25; the Buildings sheet counts Commercial 82 and Unknown 24 (same 463 total). The sheet is used separately from official xBD palu-tsunami satellite labels, which are never Y. The spreadsheet itself is not committed to git.'}

### `palu_ground_tables`

- found: True
- path: /mnt/data_hdd/datasets/paulik_palu_2018/ground_tables/24_2019_2254_MOESM1_ESM.xlsx
- sha256: fc72889951d5e116e498d2ba841680e69e5297bbedbf2976cd19d097e7f7a04b
- origin: publisher_esm_filename_matches_doi_article_2254_and_sheet_count_matches_paper
- notes: Same file as authorized_storage.publisher_survey_table. Earlier searches on this worktree predated Jlin placing these files on WSA.

### `xbd_palu_tsunami_tiles`

- found: True
- path: /mnt/data_hdd/datasets/xbd
- filename_prefix: palu-tsunami
- sha256: not_computed_not_a_ground_table
- origin: official_xbd_xview2_satellite_labels
- notes: Palu-named files at this path are official xBD / xView2 satellite image labels and targets. They are not Paulik 2019 field tables and are never Y.

### `building_footprint_catalog`

- found: True
- path: /mnt/data_hdd/datasets/_catalog/building_footprints
- palu_or_sulawesi_or_paulik_entry: False
- notes: DATASET_INDEX.txt, datasets.csv, resources.csv, issues.csv, catalog.json, SHA256SUMS, and publication_manifest.json have no Palu, Sulawesi, Indonesia, or Paulik entry.

### `catalog_index`

- path: /mnt/data_hdd/datasets/_catalog/index/datasets.csv
- msbfp2_listed: False
- ms_buildings_mx_psj_listed: False
- notes: Searched datasets.csv (46 data rows plus header), catalog.json, SHA256SUMS, and manifests/ under this index. No MSBFP2, msbfp, mx_psj, or ms_buildings entry. The only Microsoft-related id is microsoft_us_legacy17_snapshot20261117 (17 US states, primary path on WSB), which is not this file.

### `msbfp2`

- catalog_index_path: /mnt/data_hdd/datasets/_catalog/index/datasets.csv
- listed_in_catalog_index: False
- found: True
- found_via: known_filename_ms_buildings_mx_psj.geojson
- path: /mnt/data_hdd/audits/third_event_coverage_v0/maxar_michael/ms_buildings_mx_psj.geojson
- filename: ms_buildings_mx_psj.geojson
- sha256: b2d2da34c1d1e8d10ebb59d4a6c9de54e69399ce6f58f646d5c6bd167a4dd2b8
- bytes: 2992993
- geojson_type: FeatureCollection
- n_features: 6035
- polygon_count: 6035
- geometry_types: Polygon: 6035
- sample_property_keys: ['BlockgroupID', 'OBJECTID', 'Shape__Area', 'Shape__Length', 'StateAbbrev', '_query_box']
- not_a_palu_mapping_source: True
- not_written_to_official_footprints_or_git: True
- substitute_not_used: /mnt/data_hdd/datasets/building_footprints/michael_florida_ms_legacy_v2_steer_bbox.gpkg
- notes: Fresh read-only sha256 and polygon count recomputed on WSA CPU on 2026-11-18 (Asia/Shanghai) from the actual file. Inventory only, not a frozen protocol fact. The file was not listed in /mnt/data_hdd/datasets/_catalog/index/datasets.csv and was located by the known filename ms_buildings_mx_psj.geojson. Sample coordinates are around 29.95 N, 85.43 W (Mexico Beach / Port St. Joe, Florida). This is not a Palu footprint layer. The gpkg michael_florida_ms_legacy_v2_steer_bbox.gpkg was not used as a substitute, is a Michael basemap only (mixed capture dates, survey extent), and is not written into official footprints or into git.

## 11. What this packet does not claim

- It does not freeze a Palu mapping. File obtainment is not a freeze.
- It does not compute U or prediction-based conflict counts.
- It does not treat transfer across hazards as shown or settled.
- It does not apply the earlier DS1/DS2/DS3 candidate schemes as the mapping.
- Official xBD palu-tsunami labels, if present on disk, are never Y.
- Secondary sources (Lahcene 2021, Jalayer 2023) do not define official Y.
- Jalayer's phrase "complete structural collapse" is Jalayer's wording,
  not a verified Paulik definition.
- MSBFP2 inventory is not a Palu mapping source and is not a protocol freeze.

## 12. Decisions requested from Jlin

- Keep Palu unfrozen in this round. File obtainment is not a freeze.
- Adjudicate official Y from Paulik Table 1 DL0-DL3 (No damage; Partial damage, repairable; Partial damage, unrepairable; Complete damage) and the publisher Buildings-sheet labels, not from Jalayer wording and not from the earlier DS0-DS3 schemes.
- Do not apply the earlier DS1/DS2/DS3 candidate schemes as the mapping.
- Decide how, if at all, DL0-DL3 enter official four-class Y (no, minor, major, destroyed) or stay unknown or excluded.
- Note that official docs/eval_protocol.md still says full text not obtained. Decide when that protocol sentence should be updated; this worktree does not edit it into a freeze.
- Catalog event window remains 2018-09-18 (Gupta Table 1). Paulik PDF p. 1 (journal 3305) states the earthquake and tsunami date as 28 September 2018, 18:02 local time, and PDF p. 5 (journal 3309) Table 2 caption states the Palu City survey as 13-17 November 2018. Reconcile the catalog window against Gupta Table 1 without silently rewriting event_catalog.yaml in this packet.
- Do not write any candidate scheme or DL-to-Y map into the official protocol as frozen in this round.

## 13. Validator

- Validator result: `PASS`
- Issue count: 0

The validator refuses to mark Palu approved or frozen until Jlin
adjudicates official Y, refuses to unfreeze StEER wind, refuses applying
the earlier DS schemes as the mapping, refuses candidate schemes labeled
as main conclusions or as official protocol freezes, and refuses ground
tables that include model predictions or U/conflict fields.

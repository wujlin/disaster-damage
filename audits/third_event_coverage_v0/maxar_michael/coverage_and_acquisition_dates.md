# Maxar coverage and acquisition-date table (Mexico Beach / Port Saint Joe)

Generated: 2026-11-17 20:08:33 Asia/Shanghai (UTC+8)
CPU only. No imagery, tiles, or rasters were downloaded. The Michael matcher was not rerun.

## Scope

Jlin allowed Maxar only for Mexico Beach and Port Saint Joe, pending a coverage and date table. This audit queries catalog *metadata* against the StEER / xBD / MSBFP2 area of interest for those two towns, and records pre- vs post-Michael landfall.

## Landfall instant used for pre/post

- Calendar date (as specified): 2018-10-10
- Instant (NHC TCR, Beven et al. 2019; already in the StEER Michael audit script): 2018-10-10T17:30:00Z UTC = 2018-10-11 01:30:00 Asia/Shanghai (UTC+8)
- A scene is **pre** if its acquisition instant is strictly before landfall, **post** if at or after landfall, and **unknown** if the date cannot be confirmed. Unknown-date scenes are kept in the table, not dropped.

## Catalog source

No Maxar / Discovery / MGP / SecureWatch credentials or keys were found in `~/.codex/skills` (desktop or WSA) or in the WSA environment. The only already-configured Maxar catalog method on this machine is the public Maxar Open Data STAC, previously used for Hurricane Helene metadata in Flood_Road_Criticality (`MAXAR_OPEN_DATA_STAC_ITEM_FOOTPRINT_METADATA`, `asset_opening: PROHIBITED__ASSET_KEYS_ONLY`).

- Root catalog: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
- Query time: 2026-11-17 20:08:33 Asia/Shanghai (UTC+8)
- STAC id / license: maxar-opendata / CC-BY-NC-4.0
- Child event collections in the catalog at query time: 55
- Hurricane Michael collection present: **False**

## Coverage / date table

The table has **no rows**. The current Maxar ARD Open Data STAC catalog does not contain a Hurricane Michael event collection, so there are no catalog scenes, strips, acquisition dates, cloud-cover fields, or AOI intersections to list for Mexico Beach / Port Saint Joe from this source.

Historical collection paths that would follow the Helene naming pattern were probed on the same S3 host (metadata `collection.json` / `catalog.json` only). All returned HTTP 404. They are listed in the JSON under `historical_stac_path_probes`.

## Blockers (no workaround download was attempted)

- The current Maxar Open Data STAC root catalog (https://maxar-opendata.s3.amazonaws.com/events/catalog.json) was reachable (HTTP 200) at 2026-11-17 20:08:33 Asia/Shanghai (UTC+8) but lists no Hurricane Michael child collection among 55 event collections. This is an absence in the live ARD Open Data catalog, not a failed fetch of the root catalog.
- All probed historical Hurricane Michael collection.json / catalog.json paths on maxar-opendata.s3.amazonaws.com returned HTTP 404. No metadata items were retrieved from those paths.
- The historical DigitalGlobe Open Data host opendata.digitalglobe.com failed TLS certificate verification (certificate notAfter=Jun  5 23:59:59 2026 GMT; today is after that date). HTTP URLs redirected to HTTPS and failed the same way. TLS was not bypassed, and no imagery or metadata was read from that host.
- No Maxar Discovery / MGP / SecureWatch / API keys or accounts were found in ~/.codex/skills on DESKTOP-UQ2MLK2 or on WSA, and no MAXAR* environment variables were set. The commercial catalog at https://api.maxar.com/discovery/v1/ was not queried.

## Area of interest (already-local extracts; matcher not rerun)

The search AOI is the union of the two envelopes already used for the MSBFP2 / USA Structures extracts of Mexico Beach and Port Saint Joe:

- `mx_psj_coast`: lon -85.435 to -85.245, lat 29.675 to 29.958
- `psj_outlier_north`: lon -85.615 to -85.599, lat 30.075 to 30.090

Observed bounds of already-local layers are in the JSON (`aoi`). Official xBD hurricane-michael tiles do not cover the 356 StEER town points (prior result, not rerun).

## Already-local xBD label metadata (not Open Data catalog coverage of the towns)

xBD `hurricane-michael` labels already on WSA include `catalog_id`, `capture_date`, and `sensor`. They describe official xBD tiles, not Maxar Open Data scenes over Mexico Beach / Port Saint Joe. Counts are in the JSON under `already_local_xbd_hurricane_michael_metadata`.

## Imagery download statement

- Imagery / tiles / rasters downloaded in this audit: **False** (must be, and was, false).
- Matcher rerun: **False**.
- Commercial Maxar Discovery API queried: **False**.

Full machine-readable table, query log, and AOI bounds: `coverage_and_acquisition_dates.json`.

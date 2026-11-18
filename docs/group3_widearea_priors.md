# Group 3: neighborhood alerts and public geographic layers

This branch implements the Group-3 feature machinery on CPU. It does not
change the official training recipe, does not start official scoring or
retraining, and does not read official GPU checkpoints.

The scientific role of these features is described in the mainline plan:
Group 3 asks how far neighborhood satellite alarms and public geography go
as a control, before any claim that a visual mismatch score adds something
more. Neighborhood features never include the focal building's own alert.

## What is computed

For each footprint, in a local azimuthal-equidistant metric frame:

1. Satellite-side alert density inside a 30 m disk.
2. Satellite-side alert density in the 100–300 m annulus. The 30–100 m gap
   is deliberate so the two named scales do not overlap.
3. Distance to the nearest *other* building that the satellite called
   collapsed (destroyed).
4. Elevation at the footprint coordinate, when a DEM layer is supplied and
   its acquisition window is valid for the requested as-of date.
   Publication or product-release dates are recorded separately and are not
   used as that cutoff.
5. Distance to the nearest supplied water or coastline geometry, under the
   same acquisition-window rule.

Densities are counts of satellite-side alerts divided by the *observed*
window area, reported per km². If an imagery tile or map extent is
provided and the window is clipped, `window_*_edge` is set so a low count
is not silently treated as a complete density. If no coverage map or
imagery extent is provided, `window_*_edge` is left unknown (missing),
not false: false would be read as "not on the edge". Observed area still
equals the full disk or annulus in that tooling fallback, but completeness
is unknown, and official tables should attach an imagery extent before a
density is interpreted.

## Leave-one-building-out and the neighbor universe

Queries always exclude the focal `building_uid`. A building that the
satellite called destroyed does not report a zero distance to itself.

Neighbors are buildings that, for the *same event and phase*:

- have finite coordinates,
- have imagery coverage (`in_imagery=true`), and
- have a usable satellite-side grade that is not the official-export
  placeholder.

Survey-only buildings are not entered as zero-alert neighbors. Current-
disaster ground labels (`ground_grade` and other names listed in
`GROUND_LABEL_FIELDS`) are never used to decide an alert or a collapse.
Passing those field names as the alert source raises `ConfigError`.

Default vocabulary, aligned with the four-class protocol:

- alert = satellite grade in `{minor, major, destroyed}`
- collapse = satellite grade `destroyed` (`collapsed` is accepted as an alias)

Pre-event rows do not enter post-event neighborhood statistics.

## Official export placeholder

`configs/group3_layers.yaml` marks both `michael_satellite_alerts` and
`palu_satellite_alerts` as `OFFICIAL_EXPORT_PLACEHOLDER`. Until the
official export is joined, alert densities and nearest-collapse distances
are not computed from real Michael or Palu grades. Geographic layers may
still be sampled. Tests use synthetic satellite grades only.

## Palu

The same adapters and the same global DEM / water ledger entries may be
used to *prepare* Palu geographic layers. That preparation does not
announce that Michael-to-Palu transfer or migration has been established,
and it does not freeze a Palu ground-mapping crosswalk.

## Public-layer obtainability

`configs/group3_layers.yaml` records when each listed public source was
actually reachable on 2026-11-18 (Asia/Shanghai), by HTTP HEAD of a public
page or portal, plus the product publication date and acquisition window
when they are known. Acquisition and publication are stored as separate
fields. Layer usability uses the acquisition window, not the publication
date, as the cutoff. No raster or official table is stored in git.

A short reading of that ledger:

- Copernicus GLO-30 (OpenTopography portal) was reachable (HTTP 200). The
  TanDEM-X acquisition (2011–2015) predates Michael and Palu, but the
  recorded OpenTopography publication date is 2021-04-22, after both
  events.
- HydroRivers, OSM land polygons, and JRC Global Surface Water portals
  were reachable (HTTP 200). OSM still needs a frozen extract date.
- USGS 3DEP returned HTTP 403 on this fetch path; status is inconclusive,
  and 3DEP is not a Palu source.

## Tests

`tests/test_group3_priors.py` covers leave-self-out, survey versus imagery
coverage, tile edges, no-alert cases, dates and phases, missing fields,
ground-label leakage, the official-export placeholder, and ledger parsing.
All fixtures are synthetic.

## Ambiguities left open

These are recorded rather than silently decided as official protocol:

1. Whether "alert" should be any non-intact satellite call (the default
   here) or only major/destroyed. This can be changed via `alert_grades`.
2. Whether a post-2018 public DSM may be used as a static terrain prior in
   a 2018 retrospective, given that acquisition is earlier but publication
   is later, and that a DSM includes buildings and vegetation.
3. The exact water definition (coast only versus coast plus inland water)
   once a layer pack is frozen. The code accepts any polyline pack.
4. Elevation is a point sample at the coordinate. Slope or relative
   elevation is not computed yet.
5. When no coverage map is supplied, the implementation leaves edge flags
   unknown (missing) rather than false. Official tables should still attach
   an imagery extent before those flags or densities are interpreted.
6. HydroRivers `product_available_from` is year-only (2019) pending a
   tighter citation if that layer is frozen.

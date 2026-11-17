# third_event_coverage_v0 update — Maxar coverage and acquisition dates

Added on 2026-11-17 20:07:23 Asia/Shanghai (UTC+8) by scripts/audit_maxar_michael_coverage.py (sha256 8597347eea2695af2b50a2289e02152fbdc08a269089197bdb7b1fedf7ea6b55).

This adds a Maxar catalog coverage and acquisition-date table for Mexico Beach and Port Saint Joe under maxar_michael/coverage_and_acquisition_dates.json and .md (copies also under michael_maxar_opendata/).
The current Maxar Open Data STAC catalog has no Hurricane Michael collection, so the scene table is empty.
No imagery, tiles, or rasters were downloaded. The Michael matcher was not rerun. No commercial Maxar API was queried (no credentials found). The historical DigitalGlobe Open Data host failed TLS (expired certificate) and was not bypassed.

#!/usr/bin/env python3
"""CPU-only, read-only third-event coverage audit.

Uses the frozen matcher audit_dins_match.py (sha256
dd23327bc372da18186b7504f56c369dcf90e95fc221cc801ff59c4b972a8ae4)
without modifying it: 10 m point-to-polygon, inside exactly one
footprint matches, >=2 ambiguous dropped, IoU>0.5 dedup, no offset.

Does not read model predictions, does not compute U, does not download
imagery, and does not write outside the third-event audit directory.
Uncertain grades are marked unknown and reported separately. No new
filters are applied.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Frozen matcher (byte-identical; imported, never edited).
sys.path.insert(0, "/home/jinlin/projects/disaster-damage-wt/audit-third-event-coverage/scripts")
import audit_dins_match as matcher

SH = timezone(timedelta(hours=8))
HTTP_UA = "third-event-coverage-audit/0.1"

OUT_ROOT = Path("/mnt/data_hdd/audits/third_event_coverage_v0")
XBD_ROOT = Path("/mnt/data_hdd/datasets/xbd")
FROZEN_MATCHER = Path(
    "/home/jinlin/projects/disaster-damage-wt/audit-third-event-coverage/scripts/audit_dins_match.py"
)
EXPECTED_MATCHER_SHA = "dd23327bc372da18186b7504f56c369dcf90e95fc221cc801ff59c4b972a8ae4"

DINS_FULL = Path("/mnt/data_hdd/datasets/cal_fire_dins/cal_fire_dins_tubbs_nuns_2017.geojson")
DINS_README = Path("/mnt/data_hdd/datasets/cal_fire_dins/README.txt")
TILE_EXTENTS_SR = Path(
    "/mnt/data_hdd/datasets/cal_fire_dins/xbd_santa_rosa_wildfire_post_tile_extents.geojson"
)

FEMA_HIST = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
    "FEMA_Historical_Geospatial_Damage_Assessment_Database/FeatureServer/0"
)
DAT_LAYER = (
    "https://services.dat.noaa.gov/arcgis/rest/services/"
    "nws_damageassessmenttoolkit/DamageViewer/FeatureServer/0"
)

# Frozen mappings (origin/main eval protocol, 2026-11-17).
# DINS: No Damage / Unaffected / Affected -> no-damage; Minor/Major/Destroyed as named.
# FEMA: No Damage/Unaffected -> no-damage; Affected and Minor -> minor;
#       Major -> major; Destroyed -> destroyed.
DINS_TO_XBD = {
    "no damage": "no-damage",
    "unaffected": "no-damage",
    "affected": "no-damage",
    "minor": "minor",
    "major": "major",
    "destroyed": "destroyed",
    "inaccessible": "unknown",
}
FEMA_TO_XBD = {
    "no damage": "no-damage",
    "nod": "no-damage",
    "unaffected": "no-damage",
    "affected": "minor",
    "aff": "minor",
    "minor": "minor",
    "min": "minor",
    "major": "major",
    "maj": "major",
    "destroyed": "destroyed",
    "des": "destroyed",
    "unk": "unknown",
    "unknown": "unknown",
}
XBD_ORDER = ["no-damage", "minor", "major", "destroyed", "unknown", "unmapped"]


def now_sh() -> str:
    return datetime.now(SH).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai (UTC+8)")


def http_json(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def arcgis_query(base: str, **params):
    params.setdefault("f", "pjson")
    return http_json(base + "/query?" + urllib.parse.urlencode(params))


def arcgis_page_features(base: str, where: str, out_fields: str = "*",
                         geometry=None, in_sr=4326, out_sr=4326, page=2000):
    feats = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": out_sr,
            "resultOffset": offset,
            "resultRecordCount": page,
        }
        if geometry is not None:
            params.update({
                "geometry": geometry,
                "geometryType": "esriGeometryEnvelope",
                "inSR": in_sr,
                "spatialRel": "esriSpatialRelIntersects",
            })
        d = arcgis_query(base, **params)
        if d.get("error"):
            raise RuntimeError(d["error"])
        batch = d.get("features") or []
        feats.extend(batch)
        if not d.get("exceededTransferLimit") and len(batch) < page:
            break
        if not batch:
            break
        offset += len(batch)
        print(f"  paged {offset}", flush=True)
        if offset > 5_000_000:
            raise RuntimeError("pagination safety stop")
    return feats


def load_tile_polygons_from_geojson(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    geoms = []
    for feat in data.get("features") or []:
        g = feat.get("geometry")
        if g:
            geom = shape(g)
            if not geom.is_empty:
                geoms.append(geom)
    return geoms


def tile_polygons_from_labels(xbd_root: Path, event: str):
    """Tile extent = bounding box of post-disaster label coordinates.

    xBD official PNGs are used only if a sidecar/world file exists; we do
    not open raster pixels. Feature bboxes are a slightly tight proxy for
    the true image footprint and are labelled as such.
    """
    tiles = []
    for split in ("train", "test", "hold"):
        lab = xbd_root / split / "labels"
        if not lab.is_dir():
            continue
        for path in sorted(lab.glob(f"{event}_*_post_disaster.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            feats = (data.get("features") or {}).get("lng_lat") or []
            xs, ys = [], []
            for feat in feats:
                wkt = (feat or {}).get("wkt") or ""
                geom = matcher.normalize_wkt_poly(wkt)
                if geom is None or geom.is_empty:
                    continue
                minx, miny, maxx, maxy = geom.bounds
                xs.extend([minx, maxx])
                ys.extend([miny, maxy])
            if not xs:
                continue
            tiles.append({
                "tile": path.name,
                "split": split,
                "geom": box(min(xs), min(ys), max(xs), max(ys)),
                "extent_method": "post_label_feature_bbox",
            })
    return tiles


def point_to_tiles(lon, lat, tile_geoms, tree):
    pt = Point(lon, lat)
    hits = tree.query(pt)
    inside = False
    cands = list(hits) if hits is not None else []
    for g in cands:
        geom = g if hasattr(g, "covers") else tile_geoms[int(g)]
        if geom.covers(pt) or geom.contains(pt) or geom.intersects(pt):
            inside = True
            break
    # nearest distance (deg is useless); caller should pass projected tiles
    return inside


def distance_stats_m(points_xy, tiles_union):
    """points_xy: list of (x,y) in the same metric CRS as tiles_union."""
    dists = []
    n_inside = 0
    for x, y in points_xy:
        pt = Point(x, y)
        if tiles_union.covers(pt) or tiles_union.contains(pt) or tiles_union.intersects(pt):
            dists.append(0.0)
            n_inside += 1
        else:
            dists.append(float(pt.distance(tiles_union)))
    if not dists:
        return {"n": 0, "n_inside": 0}
    arr = np.array(dists, dtype=float)
    return {
        "n": int(len(arr)),
        "n_inside": int(n_inside),
        "n_outside": int(len(arr) - n_inside),
        "min_m": float(arr.min()),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.percentile(arr, 90)),
        "p95_m": float(np.percentile(arr, 95)),
        "max_m": float(arr.max()),
        "n_within_1km": int((arr <= 1000).sum()),
        "n_within_5km": int((arr <= 5000).sum()),
        "n_within_10km": int((arr <= 10000).sum()),
        "n_within_26.1km": int((arr <= 26100).sum()),
    }


def map_grade(raw: str, table: dict) -> str:
    key = (raw or "").strip().lower()
    # strip common prefixes / punctuation
    key = key.replace("*", "").strip()
    if not key:
        return "unknown"
    if key in table:
        return table[key]
    # allow values like "Destroyed (>50%)"
    for src, dst in table.items():
        if key.startswith(src):
            return dst
    return "unmapped"


def tally(values):
    c = Counter(values)
    return {k: c.get(k, 0) for k in list(dict.fromkeys([*XBD_ORDER, *c.keys()]))}


def match_points(points, footprints, id_key="id"):
    """Adapt generic points to frozen matcher classify_points()."""
    adapted = []
    for p in points:
        rec = {
            "dins_id": p.get(id_key) or p.get("dins_id") or "",
            "objectid": str(p.get("objectid") or ""),
            "globalid": str(p.get("globalid") or ""),
            "grade_raw": p.get("grade_raw") or "",
            "grade": p.get("grade") or p.get("grade_raw") or "",
            "incident": p.get("incident") or "",
            "lon": p["lon"],
            "lat": p["lat"],
            "point": p["point"],
        }
        rec.update({k: v for k, v in p.items() if k not in rec})
        adapted.append(rec)
    return matcher.classify_points(adapted, footprints)


def status_summary(results):
    counts = Counter(r["status"] for r in results)
    matched = sum(counts[s] for s in matcher.MATCH_TYPES)
    ambiguous = counts.get("ambiguous_inside", 0) + counts.get("ambiguous_10m", 0)
    return {
        "n_points": len(results),
        "unique_matches": matched,
        "inside": counts.get("inside", 0),
        "nearest_10m": counts.get("nearest_10m", 0),
        "ambiguous_inside": counts.get("ambiguous_inside", 0),
        "ambiguous_10m": counts.get("ambiguous_10m", 0),
        "ambiguous_total": ambiguous,
        "no_footprint_10m": counts.get("no_footprint_10m", 0),
        "status_counts": dict(counts),
    }


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"WROTE {path}", flush=True)


def santa_rosa():
    print("===== Santa Rosa / CAL FIRE DINS =====", flush=True)
    out = OUT_ROOT / "santa_rosa_dins"
    out.mkdir(parents=True, exist_ok=True)
    crs = "EPSG:32610"
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_m = lambda lon, lat: transformer.transform(lon, lat)

    points, n_skipped = matcher.load_dins_points(DINS_FULL, transformer)
    print(f"DINS full points={len(points)} skipped={n_skipped}", flush=True)

    # Tile extents: prefer the already-built official tile-extent file, also
    # recompute feature-bbox extents for comparison.
    if TILE_EXTENTS_SR.exists():
        tile_geoms_ll = load_tile_polygons_from_geojson(TILE_EXTENTS_SR)
        extent_source = str(TILE_EXTENTS_SR)
        extent_sha = matcher.sha256_file(TILE_EXTENTS_SR)
    else:
        tiles = tile_polygons_from_labels(XBD_ROOT, "santa-rosa-wildfire")
        tile_geoms_ll = [t["geom"] for t in tiles]
        extent_source = "post_label_feature_bbox"
        extent_sha = None
    tile_geoms_m = []
    for g in tile_geoms_ll:
        # project by transforming the exterior bbox / polygon
        gj = mapping(g)
        proj = matcher.project_geom(g, transformer)
        if proj is not None and not proj.is_empty:
            tile_geoms_m.append(proj)
    tiles_union = unary_union(tile_geoms_m)
    pts_xy = [(p["point"].x, p["point"].y) for p in points]
    dist = distance_stats_m(pts_xy, tiles_union)

    inside_pts, outside_pts = [], []
    for p in points:
        pt = p["point"]
        if tiles_union.covers(pt) or tiles_union.contains(pt) or tiles_union.intersects(pt):
            inside_pts.append(p)
        else:
            outside_pts.append(p)

    # Incident split for coverage (outside points are coverage, not match drops).
    inc_all = Counter((p.get("incident") or "").strip() or "(empty)" for p in points)
    inc_in = Counter((p.get("incident") or "").strip() or "(empty)" for p in inside_pts)
    inc_out = Counter((p.get("incident") or "").strip() or "(empty)" for p in outside_pts)

    raw, load_stats = matcher.load_post_footprints(XBD_ROOT, transformer, event="santa-rosa-wildfire")
    footprints, dedupe_stats = matcher.dedupe_footprints(raw, matcher.IOU_DEDUP)
    results = match_points(inside_pts, footprints)
    match_sum = status_summary(results)

    native = Counter(p.get("grade") or "(empty)" for p in inside_pts)
    mapped = [map_grade(p.get("grade") or p.get("grade_raw") or "", DINS_TO_XBD) for p in inside_pts]
    mapped_all_full = [map_grade(p.get("grade") or p.get("grade_raw") or "", DINS_TO_XBD) for p in points]

    summary = {
        "candidate": "Santa Rosa (Tubbs) / CAL FIRE DINS",
        "pipeline_check_fold": True,
        "note": "Santa Rosa is a pipeline check fold, not Claim-2 evidence. The protocol still requires at least 3 events.",
        "created": now_sh(),
        "host": os.uname().nodename,
        "crs": crs,
        "frozen_matcher": {
            "path": str(FROZEN_MATCHER),
            "sha256": matcher.sha256_file(FROZEN_MATCHER),
            "expected_sha256": EXPECTED_MATCHER_SHA,
            "sha256_ok": matcher.sha256_file(FROZEN_MATCHER) == EXPECTED_MATCHER_SHA,
            "rule": "10 m point-to-polygon; inside exactly one footprint matches; >=2 ambiguous dropped; IoU>0.5 dedup; no offset",
        },
        "inputs": {
            "dins_full": {"path": str(DINS_FULL), "sha256": matcher.sha256_file(DINS_FULL)},
            "xbd_root": str(XBD_ROOT),
            "event": "santa-rosa-wildfire",
            "tile_extent_source": extent_source,
            "tile_extent_sha256": extent_sha,
            "n_tile_polygons": len(tile_geoms_m),
        },
        "gate1_independent_reference": {
            "verdict": "PASS (official CAL FIRE DINS field inspection points; official statewide CSV subset Tubbs+Nuns 2017)",
            "source": "Official CAL FIRE DINS release POSTFIRE_MASTER_DATA_SHARE, downloaded 2026-11-17 17:44 Asia/Shanghai (see dataset README).",
        },
        "full_extent_distances_to_tiles_m": dist,
        "inside_tile_count": len(inside_pts),
        "outside_tile_count": len(outside_pts),
        "incident_counts": {"all": dict(inc_all), "inside_tile": dict(inc_in), "outside_tile": dict(inc_out)},
        "outside_tile_reported_as_coverage_not_match_drops": True,
        "n_skipped_bad_coords": n_skipped,
        "xbd_footprints": load_stats,
        "dedupe": dedupe_stats,
        "frozen_matcher_inside_tile_points": match_sum,
        "dins_label_tally_mapping": "No Damage/Unaffected/Affected -> no-damage; Minor/Major/Destroyed as named; Inaccessible/empty -> unknown; anything else unmapped (no new filters)",
        "dins_native_grades_inside_tile": dict(native),
        "dins_mapped_xbd_inside_tile": tally(mapped),
        "dins_mapped_xbd_full_extent": tally(mapped_all_full),
        "gate2_spatial_alignment": {
            "inside_tile_points": len(inside_pts),
            "unique_matches": match_sum["unique_matches"],
            "ambiguous_dropped": match_sum["ambiguous_total"],
            "no_footprint_10m": match_sum["no_footprint_10m"],
            "verdict": "PASS" if match_sum["unique_matches"] > 0 else "FAIL",
        },
        "gate3_can_be_evaluated_yet": True,
        "gate3_note": "Not computed in this coverage audit (no U, no conflict labels). Unique matches exist, so a later Q pass is possible. Santa Rosa remains a pipeline-check fold.",
        "blockers": [],
    }
    if match_sum["unique_matches"] == 0:
        summary["blockers"].append("No unique footprint matches inside tiles.")
    write_json(out / "summary.json", summary)
    return summary


def harvey_fema():
    print("===== Hurricane Harvey / FEMA historical geospatial assessments =====", flush=True)
    out = OUT_ROOT / "harvey_fema"
    out.mkdir(parents=True, exist_ok=True)
    crs = "EPSG:32615"
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    tiles = tile_polygons_from_labels(XBD_ROOT, "hurricane-harvey")
    tile_geoms_m = [matcher.project_geom(t["geom"], transformer) for t in tiles]
    tile_geoms_m = [g for g in tile_geoms_m if g is not None and not g.is_empty]
    tiles_union = unary_union(tile_geoms_m)
    # geographic envelope of tiles (with a small pad) for the service query
    xs, ys = [], []
    for t in tiles:
        minx, miny, maxx, maxy = t["geom"].bounds
        xs.extend([minx, maxx]); ys.extend([miny, maxy])
    pad = 0.02
    tile_env = {
        "xmin": min(xs) - pad, "ymin": min(ys) - pad,
        "xmax": max(xs) + pad, "ymax": max(ys) + pad,
    }
    print(f"Harvey tiles={len(tiles)} envelope={tile_env}", flush=True)

    where = "EVENT_NAME = '2017 Hurricane Harvey'"
    # Full-event count / type already known; fetch ALL Harvey points (attributes+geom).
    # This is a vector query, not an imagery download.
    print("Fetching all Harvey FEMA historical points (vector only)...", flush=True)
    feats = arcgis_page_features(
        FEMA_HIST, where=where,
        out_fields="OBJECTID,DMG_LEVEL,DMG_TYPE,ASMT_TYPE,SOURCE,EVENT_NAME,EVENT_DATE,PROD_DATE,IMG_DATE,STATE,COUNTY,LONGITUDE,LATITUDE,DIS_NUMBER",
    )
    print(f"fetched {len(feats)}", flush=True)
    points = []
    skipped = 0
    for f in feats:
        a = f.get("attributes") or {}
        g = f.get("geometry") or {}
        lon = a.get("LONGITUDE") if a.get("LONGITUDE") is not None else g.get("x")
        lat = a.get("LATITUDE") if a.get("LATITUDE") is not None else g.get("y")
        try:
            lon, lat = float(lon), float(lat)
        except (TypeError, ValueError):
            skipped += 1
            continue
        x, y = transformer.transform(lon, lat)
        raw = str(a.get("DMG_LEVEL") or "")
        points.append({
            "id": str(a.get("OBJECTID")),
            "objectid": a.get("OBJECTID"),
            "grade_raw": raw,
            "grade": raw,
            "mapped": map_grade(raw, FEMA_TO_XBD),
            "asmt_type": a.get("ASMT_TYPE"),
            "dmg_type": a.get("DMG_TYPE"),
            "source": a.get("SOURCE"),
            "state": a.get("STATE"),
            "lon": lon, "lat": lat,
            "point": Point(x, y),
        })

    dist = distance_stats_m([(p["point"].x, p["point"].y) for p in points], tiles_union)
    inside_pts = []
    for p in points:
        pt = p["point"]
        if tiles_union.covers(pt) or tiles_union.contains(pt) or tiles_union.intersects(pt):
            inside_pts.append(p)

    raw_fp, load_stats = matcher.load_post_footprints(XBD_ROOT, transformer, event="hurricane-harvey")
    footprints, dedupe_stats = matcher.dedupe_footprints(raw_fp, matcher.IOU_DEDUP)
    results = match_points(inside_pts, footprints) if inside_pts else []
    match_sum = status_summary(results) if results else {
        "n_points": 0, "unique_matches": 0, "inside": 0, "nearest_10m": 0,
        "ambiguous_inside": 0, "ambiguous_10m": 0, "ambiguous_total": 0,
        "no_footprint_10m": 0, "status_counts": {},
    }

    native_all = Counter(p["grade_raw"] or "(empty)" for p in points)
    native_in = Counter(p["grade_raw"] or "(empty)" for p in inside_pts)
    mapped_all = tally(p["mapped"] for p in points)
    mapped_in = tally(p["mapped"] for p in inside_pts)
    asmt = Counter(p["asmt_type"] or "(empty)" for p in points)

    blockers = [
        "All 156,099 Harvey records in the FEMA Historical Geospatial Damage Assessment Database have ASMT_TYPE=MOD (modeled) and DMG_TYPE=FL (flood). The service description states assessments were conducted 'either using high-resolution imagery or by means of geospatial modeling.' IMG_DATE is null for every Harvey record. This is not a field or UAV independent reference, so it cannot satisfy Gate 1 as frozen in the eval protocol (conflict ground truth must come from a non-satellite field/UAV source). Label tallies and matcher counts below are reported for coverage completeness only.",
        "The former gis.fema.gov FEMA_Damage_Assessments service returns HTTP 404. disasters.geoplatform.gov Harvey DamageAssessments folders returned HTTP 500 on 2026-11-17. Live FEMA IA services on gis.fema.gov are zip/county/USNG aggregates, not building-level grades.",
    ]
    if match_sum["unique_matches"] == 0:
        blockers.append("No unique frozen-matcher footprint matches among inside-tile points.")

    summary = {
        "candidate": "Hurricane Harvey / FEMA historical geospatial damage assessments",
        "created": now_sh(),
        "host": os.uname().nodename,
        "crs": crs,
        "frozen_matcher": {
            "path": str(FROZEN_MATCHER),
            "sha256": matcher.sha256_file(FROZEN_MATCHER),
            "expected_sha256": EXPECTED_MATCHER_SHA,
            "sha256_ok": matcher.sha256_file(FROZEN_MATCHER) == EXPECTED_MATCHER_SHA,
        },
        "source_layer": {
            "item_id": "e56e3d16fa144684bd532edbf298042c",
            "title": "FEMA Historical Geospatial Damage Assessment Database",
            "layer_url": FEMA_HIST,
            "layer_id": 0,
            "event_filter": where,
            "n_fetched": len(feats),
            "n_usable_points": len(points),
            "n_skipped_bad_coords": skipped,
            "asmt_type_counts": dict(asmt),
            "self_description": "Geospatial damage assessments from past National Disaster events conducted either using high-resolution imagery or by means of geospatial modeling.",
        },
        "gate1_independent_reference": {
            "verdict": "FAIL — modeled flood-depth assessments (ASMT_TYPE=MOD), not field/UAV inspections",
        },
        "tile_extent_method": "post_label_feature_bbox (no raster pixels read)",
        "n_post_tiles": len(tiles),
        "tile_envelope_lonlat_padded": tile_env,
        "full_extent_distances_to_tiles_m": dist,
        "inside_tile_count": len(inside_pts),
        "xbd_footprints": load_stats,
        "dedupe": dedupe_stats,
        "frozen_matcher_inside_tile_points": match_sum,
        "fema_label_tally_mapping": "Frozen FEMA mapping in the repo (origin/main eval protocol): No Damage/Unaffected -> no-damage; Affected and Minor -> minor; Major -> major; Destroyed -> destroyed; UNK/empty -> unknown. Native codes NOD/AFF/MIN/MAJ/DES/UNK are mapped accordingly. No new filters.",
        "fema_native_grades_all": dict(native_all),
        "fema_native_grades_inside_tile": dict(native_in),
        "fema_mapped_xbd_all": mapped_all,
        "fema_mapped_xbd_inside_tile": mapped_in,
        "gate2_spatial_alignment": {
            "inside_tile_points": len(inside_pts),
            "unique_matches": match_sum["unique_matches"],
            "ambiguous_dropped": match_sum["ambiguous_total"],
            "no_footprint_10m": match_sum["no_footprint_10m"],
            "verdict": (
                "COUNTS_ONLY — Gate 2 geometry can be measured, but Gate 1 failed so these matches are not eligible as independent-reference overlap"
            ),
        },
        "gate3_can_be_evaluated_yet": False,
        "gate3_note": "Not evaluable as a third-event independent-reference set while Gate 1 fails (modeled / non-field source).",
        "blockers": blockers,
    }
    write_json(out / "summary.json", summary)
    # do not write the 156k-point dump into the git worktree; optional compact counts only
    return summary


def florence():
    print("===== Florence FEMA + NWS DAT =====", flush=True)
    out_fema = OUT_ROOT / "florence_fema"
    out_nws = OUT_ROOT / "tornado_nws_dat"
    out_fema.mkdir(parents=True, exist_ok=True)
    # Florence NWS lives more naturally next to florence; also write a florence_nws file
    # under third_event_coverage_v0 and a pointer in tornado_nws_dat if that dir is empty.
    out_fn = OUT_ROOT / "florence_nws_dat"
    out_fn.mkdir(parents=True, exist_ok=True)

    crs = "EPSG:32618"
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    tiles = tile_polygons_from_labels(XBD_ROOT, "hurricane-florence")
    tile_geoms_m = [matcher.project_geom(t["geom"], transformer) for t in tiles]
    tile_geoms_m = [g for g in tile_geoms_m if g is not None and not g.is_empty]
    tiles_union = unary_union(tile_geoms_m)
    xs, ys = [], []
    for t in tiles:
        minx, miny, maxx, maxy = t["geom"].bounds
        xs.extend([minx, maxx]); ys.extend([miny, maxy])
    pad = 0.05
    tile_env = {
        "xmin": min(xs) - pad, "ymin": min(ys) - pad,
        "xmax": max(xs) + pad, "ymax": max(ys) + pad,
    }
    print(f"Florence tiles={len(tiles)} envelope={tile_env}", flush=True)

    # FEMA historical DB: expect 0 Florence records.
    fema_count = arcgis_query(FEMA_HIST, where="EVENT_NAME LIKE '%Florence%'", returnCountOnly="true")
    # also list event names already known; record Michael-only presence etc.
    fema_summary = {
        "candidate": "Hurricane Florence / FEMA historical geospatial damage assessments",
        "created": now_sh(),
        "source_layer": {
            "item_id": "e56e3d16fa144684bd532edbf298042c",
            "layer_url": FEMA_HIST,
            "event_filter": "EVENT_NAME LIKE '%Florence%'",
            "count": fema_count.get("count"),
            "error": fema_count.get("error"),
        },
        "related_context": {
            "florence_steer_vs_xbd": {
                "audit": "/mnt/data_hdd/audits/florence_xbd_overlap_v0/summary.json",
                "gate2": "FAIL — 0 StEER on-site records inside any xBD hurricane-florence POST tile; nearest 26,080.9 m (26.1 km); median 34.4 km",
            },
            "note": "StEER is a separate reference (already failed Gate 2) and is not a FEMA source.",
        },
        "n_post_tiles": len(tiles),
        "tile_envelope_lonlat_padded": tile_env,
        "tile_extent_method": "post_label_feature_bbox (no raster pixels read)",
        "gate1_independent_reference": {
            "verdict": "NO COVERAGE — 0 Florence records in the FEMA Historical Geospatial Damage Assessment Database (the only building-level FEMA damage-assessment point layer found still online). Live FEMA IA services are zip/county/USNG aggregates. The former FEMA_Damage_Assessments service is 404; disasters.geoplatform.gov Florence folders returned HTTP 500 on 2026-11-17."
        },
        "gate2_spatial_alignment": {
            "inside_tile_points": 0,
            "unique_matches": 0,
            "verdict": "NOT EVALUABLE — no Florence FEMA building-level points to align",
        },
        "gate3_can_be_evaluated_yet": False,
        "fema_label_tally": {},
        "blockers": [
            "No Hurricane Florence features in FEMA Historical Geospatial Damage Assessment Database (count=0).",
            "No substitute official FEMA field-inspection point layer is currently reachable.",
        ],
    }
    write_json(out_fema / "summary.json", fema_summary)

    # NWS DAT: query date window + exact tile envelope (and a slightly wider box).
    where_date = (
        "stormdate >= timestamp '2018-09-13 00:00:00' "
        "AND stormdate <= timestamp '2018-10-15 23:59:59'"
    )
    geom = f'{tile_env["xmin"]},{tile_env["ymin"]},{tile_env["xmax"]},{tile_env["ymax"]}'
    fields = "objectid,stormdate,surveydate,event_id,damage,damage_txt,dod,dod_txt,efscale,lat,lon,office,qc,surveytype,comments"
    # nationwide window count + event_id breakdown already probed; fetch points in tile envelope and a wide Carolinas box.
    wide = "-81.0,32.5,-75.5,36.5"
    dat_tile = arcgis_page_features(DAT_LAYER, where=where_date, out_fields=fields, geometry=geom)
    dat_wide = arcgis_page_features(DAT_LAYER, where=where_date, out_fields=fields, geometry=wide)
    nationwide = arcgis_query(DAT_LAYER, where=where_date, returnCountOnly="true")

    def pack(feats):
        rows = []
        for f in feats:
            a = f.get("attributes") or {}
            g = f.get("geometry") or {}
            lon = a.get("lon") if a.get("lon") is not None else g.get("x")
            lat = a.get("lat") if a.get("lat") is not None else g.get("y")
            try:
                lon, lat = float(lon), float(lat)
            except (TypeError, ValueError):
                continue
            x, y = transformer.transform(lon, lat)
            rows.append({
                "objectid": a.get("objectid"),
                "event_id": a.get("event_id"),
                "stormdate_ms": a.get("stormdate"),
                "surveydate_ms": a.get("surveydate"),
                "office": a.get("office"),
                "damage_txt": a.get("damage_txt"),
                "dod_txt": a.get("dod_txt"),
                "efscale": a.get("efscale"),
                "surveytype": a.get("surveytype"),
                "comments": a.get("comments"),
                "lon": lon, "lat": lat,
                "point": Point(x, y),
                "grade_raw": a.get("damage_txt") or a.get("dod_txt") or "",
            })
        return rows

    rows_tile = pack(dat_tile)
    rows_wide = pack(dat_wide)
    inside = []
    for p in rows_wide:
        pt = p["point"]
        if tiles_union.covers(pt) or tiles_union.contains(pt) or tiles_union.intersects(pt):
            inside.append(p)

    # Matcher only if any inside-tile or envelope points exist.
    raw_fp, load_stats = matcher.load_post_footprints(XBD_ROOT, transformer, event="hurricane-florence")
    footprints, dedupe_stats = matcher.dedupe_footprints(raw_fp, matcher.IOU_DEDUP)
    results = match_points(inside if inside else rows_tile, footprints) if (inside or rows_tile) else []
    match_sum = status_summary(results) if results else {
        "n_points": 0, "unique_matches": 0, "inside": 0, "nearest_10m": 0,
        "ambiguous_inside": 0, "ambiguous_10m": 0, "ambiguous_total": 0,
        "no_footprint_10m": 0, "status_counts": {},
    }
    dist = None
    if rows_wide:
        dist = distance_stats_m([(p["point"].x, p["point"].y) for p in rows_wide], tiles_union)

    # name-search collisions (not 2018 Hurricane Florence) recorded as a negative control
    name_hits = arcgis_query(DAT_LAYER, where="event_id LIKE '%Florence%'", returnCountOnly="true")

    event_ids_wide = Counter((p.get("event_id") or "(empty)") for p in rows_wide)
    nws_summary = {
        "candidate": "Hurricane Florence / tier-3 NWS Damage Assessment Toolkit",
        "created": now_sh(),
        "source_layer": {
            "url": DAT_LAYER,
            "layer_name": "Damage Points SDE",
            "date_window": ["2018-09-13", "2018-10-15"],
            "date_where": where_date,
            "nationwide_window_count": nationwide.get("count"),
            "event_id_LIKE_Florence_count": name_hits.get("count"),
            "name_search_note": "The 36 event_id LIKE '%Florence%' hits are name collisions from 2020-2024 (SE Florence, Florence Thomasville Tornado, etc.), not 2018 Hurricane Florence.",
        },
        "tile_extent_method": "post_label_feature_bbox (no raster pixels read)",
        "n_post_tiles": len(tiles),
        "tile_envelope_lonlat_padded": tile_env,
        "n_points_in_tile_envelope_and_date_window": len(rows_tile),
        "n_points_in_wide_carolinas_box_and_date_window": len(rows_wide),
        "wide_box": {"xmin": -81.0, "ymin": 32.5, "xmax": -75.5, "ymax": 36.5},
        "event_ids_in_wide_box": dict(event_ids_wide),
        "points_in_wide_box": [
            {k: v for k, v in p.items() if k != "point"} for p in rows_wide
        ],
        "full_extent_distances_to_tiles_m_wide_box": dist,
        "inside_tile_count": len(inside),
        "xbd_footprints": load_stats,
        "dedupe": dedupe_stats,
        "frozen_matcher_inside_tile_or_envelope_points": match_sum,
        "gate1_independent_reference": {
            "verdict": "NO USABLE 2018 HURRICANE FLORENCE COVERAGE in the public NWS DAT DamageViewer layer",
            "note": "DAT is a field/aerial wind-survey toolkit and could in principle be an independent reference, but the public layer has no identifiable Hurricane Florence 2018 event. The only two Carolinas points in the date window are a 2018-09-17/18 RAH survey tagged event_id=ROSE (one comment says rotation marks were noted from the air). Nationwide, the same window is dominated by Hurricane Michael (landfall 10 Oct 2018) and unlabeled events.",
        },
        "gate2_spatial_alignment": {
            "inside_tile_points": len(inside),
            "unique_matches": match_sum["unique_matches"],
            "verdict": "NOT EVALUABLE as Florence coverage — no 2018 Hurricane Florence DAT event to align; incidental ROSE points are reported but are not a Florence survey",
        },
        "gate3_can_be_evaluated_yet": False,
        "blockers": [
            "Public NWS DAT DamageViewer layer has no 2018 Hurricane Florence event (name search hits are later storms with Florence in the name).",
            "Only 2 DAT points fall in a wide Carolinas box in 13 Sep–15 Oct 2018, both from a RAH 'ROSE' survey, not a hurricane damage assessment of Florence.",
        ],
        "steer_context": "Florence StEER already failed Gate 2 with nearest 26.1 km (0 inside-tile records).",
    }
    write_json(out_fn / "summary.json", nws_summary)
    # keep the pre-created tornado_nws_dat folder as a pointer, not a tornado redo
    write_json(out_nws / "florence_pointer.json", {
        "note": "This directory was pre-created for NWS DAT reconnaissance. The Florence tier-3 NWS result is in ../florence_nws_dat/summary.json. Prior tornado DAT reconnaissance (Joplin/Moore/Tuscaloosa) is not part of this four-candidate table.",
        "florence_nws_summary": str(out_fn / "summary.json"),
    })
    return fema_summary, nws_summary


def four_candidate_table(sr, harvey, flor_fema, flor_nws):
    def row(s, name=None):
        g2 = s.get("gate2_spatial_alignment") or {}
        g1 = s.get("gate1_independent_reference") or {}
        return {
            "candidate": name or s.get("candidate"),
            "gate1_independent_reference": g1.get("verdict"),
            "gate2_inside_tile_points": g2.get("inside_tile_points", s.get("inside_tile_count")),
            "gate2_unique_matches": g2.get("unique_matches"),
            "gate2_verdict": g2.get("verdict"),
            "gate3_can_be_evaluated_yet": s.get("gate3_can_be_evaluated_yet"),
            "blockers": s.get("blockers") or [],
        }
    table = [
        row(sr),
        row(harvey),
        row(flor_fema),
        row(flor_nws),
    ]
    return table


def main():
    print(f"START {now_sh()}", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    sha = matcher.sha256_file(FROZEN_MATCHER)
    if sha != EXPECTED_MATCHER_SHA:
        raise SystemExit(f"Frozen matcher sha256 mismatch: {sha}")
    print("matcher sha256 OK", sha, flush=True)
    # do not touch PID 1202931; this process is CPU-only.
    sr = santa_rosa()
    hv = harvey_fema()
    ff, fn = florence()
    table = four_candidate_table(sr, hv, ff, fn)
    root = {
        "created": now_sh(),
        "timezone": "Asia/Shanghai (UTC+8)",
        "host": os.uname().nodename,
        "cpu_only": True,
        "no_imagery_download": True,
        "no_predictions_or_U": True,
        "frozen_matcher_sha256": sha,
        "msbfp2_provenance": str(OUT_ROOT / "maxar_michael" / "msbfp2_provenance.json"),
        "maxar_michael_note": "MSBFP2 / USA Structures footprint matching for Maxar Michael is already complete under maxar_michael/ and was not rerun. Provenance of the MSBFP2 extract is pinned in msbfp2_provenance.json.",
        "four_candidate_table": table,
        "paths": {
            "santa_rosa_dins": str(OUT_ROOT / "santa_rosa_dins" / "summary.json"),
            "harvey_fema": str(OUT_ROOT / "harvey_fema" / "summary.json"),
            "florence_fema": str(OUT_ROOT / "florence_fema" / "summary.json"),
            "florence_nws_dat": str(OUT_ROOT / "florence_nws_dat" / "summary.json"),
        },
    }
    write_json(OUT_ROOT / "four_candidate_table.json", root)
    print(f"DONE {now_sh()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

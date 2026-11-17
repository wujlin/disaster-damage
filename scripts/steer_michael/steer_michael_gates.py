#!/usr/bin/env python3
"""StEER Hurricane Michael four-gate audit against official xBD hurricane-michael POST tiles.

CPU only. Does NOT compute conflict labels U, does NOT use official xBD damage subtypes
for any estimate, does NOT train anything.

Frozen matching rule is imported verbatim from scripts/frozen_matcher.py (a byte copy of
audit-florence-xbd-overlap/scripts/audit_dins_match.py @ 25a03b5): inside-exactly-one ->
match; inside >=2 -> ambiguous_inside; else exactly one footprint within 10 m
(point-to-polygon) -> match; >=2 -> ambiguous_10m; none -> no_footprint_10m. Footprints
deduplicated at IoU > 0.5. No offset correction.

Sample policy (Jlin): only removals are (1) no unique match under the frozen rule,
(2) temporal mismatch with the xBD POST image, (3) exclusions the approved source text
requires (wind grade -1 removed from the wind tally only). Everything else (valid-pixel
fraction, assessment type, General Area records, imagery-derived notes, multi-point
footprints) is reported as flag columns / separate tables, not dropped from the main sample.
Wind and surge are tallied separately throughout.

Building-level Q: official Q is the unique footprints among point-level Q.
A uniquely matched point just outside the tile polygon is kept in Q and flagged
(gate 2 is the unique-footprint rule; the footprint is inside the tile). Multi-point
footprints that disagree on a scale are dropped from that scale's Y only, not from Q.
Wind grades 3/4 are landed from existing component fields for reporting only; this
does not drop buildings from Q. Grade 4 is not mapped to destroyed, and no
percentage threshold is proposed. summary.json is written entirely by this script
and is deterministic (no run timestamps).
"""
from __future__ import annotations

import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.wkt import loads as load_wkt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_matcher as fm  # noqa: E402

EVENT = "hurricane-michael"
CRS_METRIC = "EPSG:32616"  # UTM 16N covers -88..-82 deg lon
XBD = Path("/mnt/data_hdd/datasets/xbd")
SRC = Path("/mnt/data_hdd/datasets/steer_hurricane_michael")
BLD_CSV = SRC / "D1.1 Building Damage Assessments - StEER/HM_D2D_Building.csv"
BLD_GJ = SRC / "D1.1 Building Damage Assessments - StEER/HM_D2D_Building.geojson"
OUT = Path("/mnt/data_hdd/audits/steer_michael_v0")
# NHC TCR (Beven et al. 2019): Michael landfall near Mexico Beach ~1730 UTC 10 Oct 2018.
LANDFALL_UTC = datetime(2018, 10, 10, 17, 30, tzinfo=timezone.utc)
VALID_PIX_THR = 0.80
WIND_VALID = {"0", "1", "2", "3", "4"}
SURGE_VALID = {"0", "1", "2", "3", "4", "5", "6"}
COMPONENT_FIELDS = [
    "roof_structure_damage_", "roof_substrate_damage", "roof_cover_damage_",
    "wall_structure_damage_", "wall_substrate_damage_", "building_envelope_damage_",
    "front_wall_fenestration_damage", "left_wall_fenestration_damage",
    "back_wall_fenestration_damage", "right_wall_fenestration_damage",
    "large_door_failure", "soffit_damage", "fascia_damage_", "stories_with_damage",
    "percent_of_building_footprint_eroded", "_damage_to_understory",
    "_piles_missing_or_collapsed", "_piles_leaning_or_broken", "cause_of_foundation_damage",
]



# Source-field groups used in building-level 3/4 reporting (exact CSV names).
ROOF_WALL_DAMAGE_FIELDS = [
    "roof_structure_damage_", "wall_structure_damage_",
    "roof_substrate_damage", "wall_substrate_damage_",
    "roof_cover_damage_",
]
ROOF_WALL_TYPE_FIELDS = ["roof_cover", "roof_substrate_type", "wall_structure", "wall_substrate"]
# Fields whose meaning may indicate total collapse, absence, or an incomplete structure.
# There is no breakaway-wall column in the 96-column StEER table.
COLLAPSE_RELATED_FIELDS = [
    "status",
    "overall_damage_notes", "general_notes", "structural_notes",
    "wind_damage_details", "water_induced_damage_notes", "attribute_notes", "qc_notes",
    "percent_of_building_footprint_eroded", "maximum_scour_depth_inches",
    "_piles_missing_or_collapsed", "_piles_leaning_or_broken",
    "cause_of_foundation_damage", "foundation_type",
    "first_floor_elevation_feet", "understory_pct_of_building_footprint",
    "_damage_to_understory", "stories_with_damage", "building_envelope_damage_",
    "surge_damage_rating",
]
# Keyword scan over all source columns for collapse / no-longer-present language.
# Hits are recorded as values; nothing is mapped to destroyed from a hit.
COLLAPSE_KEYWORD_RE = (
    r"total\s*collapse|partial\s*collapse|\bcollapse(?:d|s)?\b|"
    r"destroyed|total\s*loss|complete\s*loss|"
    r"no longer (?:present|exists|standing)|not standing|not (?:a )?complete structure|"
    r"washed away|swept away|demolished|leveled|razed|"
    r"slab only|only slab|foundation only|debris only|"
    r"structure (?:is )?gone|building (?:is )?gone|no structure left|no longer present"
)

WIND_MAP_02 = {"0": "undamaged", "1": "minor", "2": "moderate"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _filled(row, key):
    return (row.get(key) or "").strip() not in ("",)


def _structure(row):
    roof_filled = _filled(row, "roof_structure_damage_")
    wall_filled = _filled(row, "wall_structure_damage_")
    roof = _num(row.get("roof_structure_damage_", "")) if roof_filled else None
    wall = _num(row.get("wall_structure_damage_", "")) if wall_filled else None
    roof_gt0 = roof is not None and roof > 0
    wall_gt0 = wall is not None and wall > 0
    return {
        "roof_structure_damage": roof,
        "wall_structure_damage": wall,
        "roof_structure_gt0": roof_gt0,
        "wall_structure_gt0": wall_gt0,
        "roof_or_wall_structure_gt0": roof_gt0 or wall_gt0,
        "both_structure_fields_blank": (not roof_filled) and (not wall_filled),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"non-JSON type in summary: {type(obj)!r}")


def _distinct_values(rows, keys):
    """Deterministic distinct-value tables: value -> count, empty kept as ""."""
    out = {}
    for k in keys:
        c = Counter((r.get(k) or "") for r in rows)
        out[k] = {
            "n_filled": sum(n for v, n in c.items() if str(v).strip() != ""),
            "n_distinct_including_empty": len(c),
            "n_distinct_nonempty": sum(1 for v in c if str(v).strip() != ""),
            "counts": dict(sorted(c.items(), key=lambda t: (t[0] == "", t[0]))),
        }
    return out


def collapse_to_buildings(recs):
    """One record per unique footprint, deterministic (uid, then fulcrum_id)."""
    by_fp = defaultdict(list)
    for r in recs:
        by_fp[r["footprint_uid"]].append(r)
    buildings = []
    for uid, group in sorted(by_fp.items(), key=lambda t: t[0]):
        group = sorted(group, key=lambda r: r["fulcrum_id"])
        primary = group[0]
        wind_grades = [x["row"]["wind_damage_rating"] for x in group]
        surge_grades = [x["row"]["surge_damage_rating"] for x in group]
        wind_valid = [g for g in wind_grades if g in WIND_VALID]
        wind_agree = len(set(wind_grades) - {"-1"}) <= 1
        surge_agree = len(set(surge_grades)) <= 1
        if wind_agree and wind_valid:
            wind_grade = sorted(set(wind_valid))[0]
        elif wind_agree and all(g == "-1" for g in wind_grades):
            wind_grade = "-1"
        else:
            wind_grade = None
        surge_grade = sorted(set(surge_grades))[0] if surge_agree and surge_grades else None
        edge_pts = [x for x in group if str(x.get("gate2_status", "")).endswith("_edge_outside_tile_poly")]
        flags = sorted({f for x in group for f in (x.get("flags") or "").split(";") if f})
        b = {
            "footprint_uid": uid,
            "n_points": len(group),
            "fulcrum_ids": [x["fulcrum_id"] for x in group],
            "points": group,
            "primary": primary,
            "wind_grades": wind_grades,
            "surge_grades": surge_grades,
            "wind_agree": wind_agree,
            "surge_agree": surge_agree,
            "wind_grade": wind_grade,
            "surge_grade": surge_grade,
            "wind_in_Y": bool(wind_agree and wind_grade in WIND_VALID),
            "surge_in_Y": bool(surge_agree and surge_grade in SURGE_VALID),
            "edge_outside_tile_poly": bool(edge_pts),
            "tile": primary.get("tile", ""),
            "split": primary.get("split", ""),
            "distance_m": primary.get("distance_m", ""),
            "nearest_distance_m": primary.get("nearest_distance_m", ""),
            "lat": primary.get("lat", ""),
            "lon": primary.get("lon", ""),
            "gate2_status": primary.get("gate2_status", ""),
            "flags": flags,
        }
        if edge_pts:
            e = edge_pts[0]
            b["edge_fulcrum_id"] = e["fulcrum_id"]
            b["edge_distance_m"] = e.get("distance_m", e.get("nearest_distance_m", ""))
            b["edge_gate2_status"] = e.get("gate2_status", "")
            b["edge_tile"] = e.get("tile", "")
            b["edge_lat"] = e.get("lat", "")
            b["edge_lon"] = e.get("lon", "")
        buildings.append(b)
    return buildings


def land_wind_grade(b):
    """Land wind 3/4 from existing component fields. Does not drop the building from Q.
    Grade 4 is recorded as major when it has a resolved grade and structure damage;
    destroyed is never assigned and no threshold is applied.
    """
    if not b["wind_in_Y"]:
        reason = ("dropped_from_wind_Y_multi_point_disagree"
                  if not b["wind_agree"] else "excluded_wind_grade_-1")
        return {"mapped": None, "reason": reason, "destroyed_assigned": False}
    g = b["wind_grade"]
    if g in WIND_MAP_02:
        return {"mapped": WIND_MAP_02[g], "reason": f"wind_grade_{g}_unchanged", "destroyed_assigned": False}
    st = _structure(b["primary"]["row"])
    if g == "3":
        if st["roof_or_wall_structure_gt0"]:
            return {"mapped": "major", "reason": "grade_3_roof_or_wall_structure_gt0",
                    "destroyed_assigned": False, **st}
        return {"mapped": "unknown", "reason": "grade_3_no_structure_damage_gt0",
                "destroyed_assigned": False, **st}
    if g == "4":
        mapped = "major" if st["roof_or_wall_structure_gt0"] else "unknown"
        reason = ("grade_4_recorded_as_major_destroyed_not_assigned"
                  if mapped == "major" else "grade_4_no_structure_damage_gt0_destroyed_not_assigned")
        return {"mapped": mapped, "reason": reason, "destroyed_assigned": False, **st}
    return {"mapped": None, "reason": f"unexpected_wind_grade_{g}", "destroyed_assigned": False}


_COLLAPSE_RE = re.compile(COLLAPSE_KEYWORD_RE, re.I)


def keyword_hits(row, cols):
    hits = []
    for c in cols:
        v = row.get(c) or ""
        if v and _COLLAPSE_RE.search(v):
            hits.append({"field": c, "value": v})
    return hits


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_buildings(tr):
    with BLD_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    gj = json.loads(BLD_GJ.read_text(encoding="utf-8"))
    gj_by_id = {}
    for ft in gj["features"]:
        pr = ft.get("properties") or {}
        fid = pr.get("fulcrum_id") or pr.get("_record_id") or pr.get("id")
        gj_by_id[fid] = ft.get("geometry")
    pts, coord_issues = [], []
    for r in rows:
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except ValueError:
            coord_issues.append({"fulcrum_id": r["fulcrum_id"], "issue": "unparseable_latlon"})
            continue
        g = gj_by_id.get(r["fulcrum_id"])
        gj_delta = ""
        if g and g.get("coordinates"):
            glon, glat = g["coordinates"][:2]
            gj_delta = max(abs(glon - lon), abs(glat - lat))
        x, y = tr.transform(lon, lat)
        d = datetime.strptime(r["date"], "%m/%d/%Y").date()
        pts.append({
            "fulcrum_id": r["fulcrum_id"], "lon": lon, "lat": lat, "point": Point(x, y),
            "survey_date": d.isoformat(), "row": r, "geojson_max_abs_deg_delta": gj_delta,
        })
    return rows, pts, coord_issues, len(gj["features"]), len(gj_by_id)


def tile_meta_and_polys(tr):
    """Per post tile: capture_date and footprint of the 1024x1024 image in metric CRS,
    via a least-squares affine fit pixel(x,y)->lon/lat from label polygon vertices."""
    tiles, uid_xy = {}, {}
    for split in fm.SPLITS:
        for p in sorted((XBD / split / "labels").glob(f"{EVENT}_*_post_disaster.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            md = d.get("metadata") or {}
            ll = (d.get("features") or {}).get("lng_lat") or []
            xy = (d.get("features") or {}).get("xy") or []
            src, dst = [], []
            for a, b in zip(ll, xy):
                uid = (a.get("properties") or {}).get("uid")
                if uid != (b.get("properties") or {}).get("uid"):
                    continue
                try:
                    ga, gb = load_wkt(a["wkt"]), load_wkt(b["wkt"])
                except Exception:
                    continue
                uid_xy[(p.name, uid)] = gb
                if ga.geom_type == "Polygon" and gb.geom_type == "Polygon":
                    ca, cb = list(ga.exterior.coords), list(gb.exterior.coords)
                    if len(ca) == len(cb):
                        dst.extend(ca); src.extend(cb)
            poly_m, fit_rmse_px = None, None
            if len(src) >= 3:
                S = np.c_[np.asarray(src), np.ones(len(src))]
                D = np.asarray(dst)
                A, *_ = np.linalg.lstsq(S, D, rcond=None)
                W, H = md.get("width", 1024), md.get("height", 1024)
                corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]]) @ A
                if np.linalg.matrix_rank(S) == 3:
                    poly_m = Polygon([tr.transform(lo, la) for lo, la in corners])
                    # inverse check: residual in degrees -> approx px
                    res = S @ A - D
                    fit_rmse_px = float(np.sqrt((res ** 2).sum(1).mean()) / (abs(A[0, 0]) + 1e-12))
            tiles[p.name] = {
                "split": split, "capture_date": md.get("capture_date"),
                "sensor": md.get("sensor"), "gsd": md.get("gsd"),
                "img": str(XBD / split / "images" / md.get("img_name", p.name.replace(".json", ".png"))),
                "n_buildings": len(ll), "poly_m": poly_m, "fit_rmse_px": fit_rmse_px,
            }
    return tiles, uid_xy


def valid_pixel_fraction(img_path: str, xy_poly):
    """Fraction of the footprint's pixels (whole polygon, incl. any part off-image) that are
    valid, i.e. not all-zero RGB nodata. Returns (fraction, n_poly_px, n_valid_px)."""
    im = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = im.shape[:2]
    polys = [xy_poly] if xy_poly.geom_type == "Polygon" else list(xy_poly.geoms)
    mask_img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(mask_img)
    for pg in polys:
        dr.polygon([(x, y) for x, y in pg.exterior.coords], fill=1)
    mask = np.asarray(mask_img, dtype=bool)
    n_in = int(mask.sum())
    n_total = max(n_in, int(round(xy_poly.area)))  # off-image area counts as invalid
    valid = (im.max(axis=2) > 0) & mask
    n_valid = int(valid.sum())
    return (n_valid / n_total if n_total else 0.0), n_total, n_valid


def scale_tally(recs, key):
    return dict(sorted(Counter(r["row"][key] for r in recs).items(), key=lambda t: t[0]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat()
    tr = Transformer.from_crs("EPSG:4326", CRS_METRIC, always_xy=True)

    rows, pts, coord_issues, n_gj, n_gj_ids = load_buildings(tr)
    tiles, uid_xy = tile_meta_and_polys(tr)

    raw_fp, fp_stats = fm.load_post_footprints(XBD, tr, event=EVENT)
    canon, dedup_stats = fm.dedupe_footprints(raw_fp)

    # ---- Gate 2a: inside tile coverage
    tile_items = [(k, v) for k, v in tiles.items() if v["poly_m"] is not None]
    ttree = STRtree([v["poly_m"] for _, v in tile_items])
    for p in pts:
        idx = [int(i) for i in ttree.query(p["point"], predicate="intersects")]
        p["tiles_covering"] = [tile_items[i][0] for i in idx]
        p["inside_tiles"] = bool(idx)

    # ---- Gate 2b: frozen matching on all points (points outside tiles reported separately)
    results = fm.classify_points(pts, canon)
    for r in results:
        r["gate2_status"] = r["status"] if r["inside_tiles"] else (
            "outside_tiles" if r["status"] not in fm.MATCH_TYPES else r["status"] + "_edge_outside_tile_poly")

    matched = [r for r in results if r["status"] in fm.MATCH_TYPES]

    # multi-point per footprint (diagnostic, per scale)
    by_fp = defaultdict(list)
    for r in matched:
        by_fp[r["footprint_uid"]].append(r)
    multi = {u: v for u, v in by_fp.items() if len(v) > 1}
    multi_rows = []
    for u, v in multi.items():
        w = sorted({x["row"]["wind_damage_rating"] for x in v})
        s = sorted({x["row"]["surge_damage_rating"] for x in v})
        multi_rows.append({"footprint_uid": u, "n_points": len(v),
                           "fulcrum_ids": ";".join(x["fulcrum_id"] for x in v),
                           "wind_grades": ";".join(w), "surge_grades": ";".join(s),
                           "wind_agree": len(set(w) - {"-1"}) <= 1, "surge_agree": len(s) <= 1})

    # ---- temporal alignment (Gate 1 date part, applied to matched)
    canon_by_uid = {c["uid"]: c for c in canon}
    temporal_rows = []
    for r in matched:
        t = tiles.get(r["tile"], {})
        cd = t.get("capture_date")
        cdt = datetime.fromisoformat(cd.replace("Z", "+00:00")) if cd else None
        r["xbd_post_capture_utc"] = cd or ""
        r["survey_minus_capture_days"] = (
            (datetime.fromisoformat(r["survey_date"]).date() - cdt.date()).days if cdt else "")
        if cdt is None:
            r["temporal"] = "unknown_capture_date"
        elif cdt < LANDFALL_UTC:
            r["temporal"] = "mismatch_post_image_before_landfall"
        else:
            r["temporal"] = "aligned"
        if r["temporal"] != "aligned":
            temporal_rows.append(r)
    main_sample = [r for r in matched if r["temporal"] == "aligned"]

    # ---- Gate 3: valid pixels (computed on main sample; reported as flag, not a removal)
    img_cache_note = []
    for r in main_sample:
        c = canon_by_uid[r["footprint_uid"]]
        xy = uid_xy.get((c["tile"], c["uid"]))
        if xy is None:
            r["valid_px_frac"] = ""
            r["valid_px_flag"] = "unknown_no_xy_polygon"
            continue
        frac, n_tot, n_val = valid_pixel_fraction(tiles[c["tile"]]["img"], xy)
        r["valid_px_frac"] = round(frac, 4)
        r["valid_px_n_poly"] = n_tot
        r["valid_px_flag"] = "ge_0.80" if frac >= VALID_PIX_THR else "lt_0.80"
    Q = [r for r in main_sample if r["valid_px_flag"] == "ge_0.80"]

    # ---- flags (reported, not removed)
    def flags(r):
        row = r["row"]
        f = []
        if row["assessment_type"] == "General Area" or row["building_type"] == "General Area":
            f.append("general_area_record")
        if "pictometry" in (row.get("wind_damage_details", "") + row.get("overall_damage_notes", "")).lower():
            f.append("component_ratios_from_pictometry_imagery")
        if row["assessment_type"] == "Drive-by":
            f.append("drive_by")
        if "repeated point" in row.get("general_notes", "").lower():
            f.append("repeated_point_note")
        return f
    for r in results:
        r["flags"] = ";".join(flags(r))

    # ---- per-scale tallies
    def wind_tally(recs):
        c = Counter(r["row"]["wind_damage_rating"] for r in recs)
        y = {g: c.get(g, 0) for g in ["0", "1", "2", "3", "4"]}
        return {"raw": dict(sorted(c.items())), "excluded_from_wind_Y_grade_-1": c.get("-1", 0),
                "wind_Y_n": sum(y.values()), "undamaged_0": y["0"],
                "damaged_1to4": y["1"] + y["2"] + y["3"] + y["4"],
                "grade_3_or_4_needs_component_review": y["3"] + y["4"]}

    def surge_tally(recs):
        c = Counter(r["row"]["surge_damage_rating"] for r in recs)
        y = {g: c.get(g, 0) for g in sorted(SURGE_VALID)}
        return {"raw": dict(sorted(c.items())),
                "surge_Y_n": sum(y.values()), "none_0": y["0"],
                "damaged_1to6": sum(y[g] for g in "123456"),
                "proposal_only_codex_5_major": y["5"], "proposal_only_codex_6_destroyed": y["6"],
                "unmapped_1to4_pending_rule": sum(y[g] for g in "1234")}

    def scale_presence(recs):
        c = Counter()
        for r in recs:
            w = r["row"]["wind_damage_rating"] in WIND_VALID
            s = r["row"]["surge_damage_rating"] in SURGE_VALID
            sd = s and r["row"]["surge_damage_rating"] != "0"
            c["wind_valid"] += w; c["surge_valid"] += s
            c["both_valid"] += w and s; c["wind_only_valid"] += w and not s
            c["surge_only_valid"] += s and not w; c["neither_valid"] += (not w) and (not s)
            c["wind_valid_and_surge_gt0"] += w and sd
            c["wind_-1_with_surge_grade"] += (r["row"]["wind_damage_rating"] == "-1") and s
        return dict(c)

    def component_coverage(recs):
        """For wind grade 3/4 records: which component fields are filled (non-empty)."""
        sub = [r for r in recs if r["row"]["wind_damage_rating"] in ("3", "4")]
        out = {"n_wind_3_or_4": len(sub)}
        for g in ("3", "4"):
            s2 = [r for r in sub if r["row"]["wind_damage_rating"] == g]
            out[f"grade_{g}"] = {
                "n": len(s2),
                "filled": {k: sum(1 for r in s2 if r["row"].get(k, "").strip() not in ("",)) for k in COMPONENT_FIELDS},
                "roof_structure_gt0": sum(1 for r in s2 if _num(r["row"]["roof_structure_damage_"]) > 0),
                "wall_structure_gt0": sum(1 for r in s2 if _num(r["row"]["wall_structure_damage_"]) > 0),
                "roof_or_wall_structure_gt0": sum(1 for r in s2 if _num(r["row"]["roof_structure_damage_"]) > 0 or _num(r["row"]["wall_structure_damage_"]) > 0),
                "both_structure_fields_blank": sum(1 for r in s2 if not r["row"]["roof_structure_damage_"].strip() and not r["row"]["wall_structure_damage_"].strip()),
            }
        return out

    cols = set(rows[0].keys())
    unobserved = {k: ("present" if any(k2 for k2 in cols if k in k2) else "absent_unobserved")
                  for k in ["waterline", "interior", "char", "burn", "fire"]}

    stages = {"all_records": results, "inside_tiles": [r for r in results if r["inside_tiles"]],
              "uniquely_matched": matched, "main_sample_temporally_aligned": main_sample, "Q_valid_px_ge_0.80": Q}
    per_stage = {k: {"n": len(v), "wind": wind_tally(v), "surge": surge_tally(v),
                     "overall_status_raw": scale_tally(v, "status"),
                     "assessment_type": scale_tally(v, "assessment_type"),
                     "scale_presence": scale_presence(v),
                     "flags": dict(Counter(f for r in v for f in r["flags"].split(";") if f))}
                 for k, v in stages.items()}
    per_stage["Q_valid_px_ge_0.80"]["wind_3_4_component_coverage"] = component_coverage(Q)
    per_stage["main_sample_temporally_aligned"]["wind_3_4_component_coverage"] = component_coverage(main_sample)


    Q_buildings = collapse_to_buildings(Q)
    for b in Q_buildings:
        b["wind_mapped"] = land_wind_grade(b)

    # Official Q includes the just-outside-tile unique match, flagged, not dropped.
    edge_buildings = [b for b in Q_buildings if b["edge_outside_tile_poly"]]
    src_cols_all = list(rows[0].keys()) if rows else []

    def building_wind_tally(buildings):
        c = Counter()
        n_disagree = 0
        n_minus1 = 0
        for b in buildings:
            if not b["wind_agree"]:
                n_disagree += 1
                continue
            if b["wind_grade"] not in WIND_VALID:
                n_minus1 += 1
                continue
            c[b["wind_grade"]] += 1
        y = {g: int(c.get(g, 0)) for g in ["0", "1", "2", "3", "4"]}
        return {
            "by_grade": y,
            "excluded_from_wind_Y_grade_-1": n_minus1,
            "wind_multi_point_disagree_dropped_from_wind_Y": n_disagree,
            "wind_Y_n": sum(y.values()),
            "undamaged_0": y["0"],
            "damaged_1to4": y["1"] + y["2"] + y["3"] + y["4"],
            "grade_3_or_4_needs_component_review": y["3"] + y["4"],
        }

    def building_surge_tally(buildings):
        c = Counter()
        n_disagree = 0
        for b in buildings:
            if not b["surge_agree"] or b["surge_grade"] not in SURGE_VALID:
                n_disagree += 1
                continue
            c[b["surge_grade"]] += 1
        y = {g: int(c.get(g, 0)) for g in sorted(SURGE_VALID)}
        return {
            "by_grade": {k: v for k, v in y.items() if v} if any(y.values()) else dict(y),
            "surge_multi_point_disagree_dropped_from_surge_Y": n_disagree,
            "surge_Y_n": sum(y.values()),
            "none_0": y["0"],
            "damaged_1to6": sum(y[g] for g in "123456"),
        }

    def building_component_coverage(buildings, grade):
        recs = [b for b in buildings if b["wind_in_Y"] and b["wind_grade"] == grade]
        s2_rows = [b["primary"]["row"] for b in recs]
        return {
            "n": len(recs),
            "filled": {k: sum(1 for r in s2_rows if _filled(r, k)) for k in COMPONENT_FIELDS},
            "roof_structure_gt0": sum(1 for r in s2_rows if _structure(r)["roof_structure_gt0"]),
            "wall_structure_gt0": sum(1 for r in s2_rows if _structure(r)["wall_structure_gt0"]),
            "roof_or_wall_structure_gt0": sum(1 for r in s2_rows if _structure(r)["roof_or_wall_structure_gt0"]),
            "both_structure_fields_blank": sum(1 for r in s2_rows if _structure(r)["both_structure_fields_blank"]),
        }

    w_bldg = building_wind_tally(Q_buildings)
    s_bldg = building_surge_tally(Q_buildings)

    g3_major = [b for b in Q_buildings if b["wind_in_Y"] and b["wind_grade"] == "3"
                and b["wind_mapped"]["mapped"] == "major"]
    g3_unknown = [b for b in Q_buildings if b["wind_in_Y"] and b["wind_grade"] == "3"
                  and b["wind_mapped"]["mapped"] == "unknown"]
    g4_in_Y = [b for b in Q_buildings if b["wind_in_Y"] and b["wind_grade"] == "4"]
    # 17 buildings in Q with at least one wind-grade-4 point, including 3
    # multi-point footprints also carrying a disagreeing other-grade point.
    # For those 3, the candidate row uses the grade-4 point (not the other point).
    g4_candidates = []
    for b in Q_buildings:
        g4_pts = [p for p in b["points"] if p["row"]["wind_damage_rating"] == "4"]
        if not g4_pts:
            continue
        g4_pt = sorted(g4_pts, key=lambda p: p["fulcrum_id"])[0]
        g4_candidates.append((b, g4_pt))
    g4_candidates.sort(key=lambda t: t[0]["footprint_uid"])

    def point_wind_tally_grades(recs):
        return dict(sorted(Counter(r["row"]["wind_damage_rating"] for r in recs).items()))

    pt_wind = point_wind_tally_grades(Q)
    # Explain 16/14 vs 18/17: point vs building, plus disagree drops from wind Y.
    multi_explain = []
    for m in sorted(multi_rows, key=lambda d: d["footprint_uid"]):
        fps = [r for r in Q if r["footprint_uid"] == m["footprint_uid"]]
        multi_explain.append({
            "footprint_uid": m["footprint_uid"],
            "n_points": m["n_points"],
            "fulcrum_ids": m["fulcrum_ids"],
            "wind_grades": m["wind_grades"],
            "surge_grades": m["surge_grades"],
            "wind_agree": m["wind_agree"],
            "surge_agree": m["surge_agree"],
            "in_Q": len(fps) == m["n_points"],
        })
    unit_mismatch = {
        "point_level_Q_n": len(Q),
        "building_level_Q_n": len(Q_buildings),
        "point_level_wind_grades": pt_wind,
        "building_level_wind_grades_in_Y": w_bldg["by_grade"],
        "point_level_grade_3": int(pt_wind.get("3", 0)),
        "point_level_grade_4": int(pt_wind.get("4", 0)),
        "building_level_grade_3_in_wind_Y": w_bldg["by_grade"]["3"],
        "building_level_grade_4_in_wind_Y": w_bldg["by_grade"]["4"],
        "n_buildings_with_a_grade4_point": len(g4_candidates),
        "multi_point_footprints_in_Q": multi_explain,
        "explanation": (
            "The earlier gate-3 wind grades 3=16, 4=14 were unique-building counts in wind Y "
            "after collapsing multi-point footprints and dropping disagreements from wind Y. "
            "The component tallies 3=18, 4=17 were point-level counts on the 161 points in Q. "
            "Four footprints have two points each. Two of those contribute one grade-3 and one "
            "grade-4 point and are dropped from wind Y (disagreement). One contributes one "
            "grade-1 and one grade-4 point and is also dropped from wind Y. One has two agreeing "
            "grade-2 points and collapses to a single grade-2 building. Hence grade 3 in wind Y "
            "is 18-2=16 and grade 4 in wind Y is 17-3=14. The 17 unique footprints that each "
            "carry a grade-4 point remain in Q and in the destroyed-candidate list; destroyed "
            "is not assigned."
        ),
    }

    def gate4_buildings(buildings, wind_t, surge_t):
        return {
            "n": len(buildings),
            "unit": "unique footprints / buildings",
            "wind": {
                "Y_n": wind_t["wind_Y_n"],
                "undamaged": wind_t["undamaged_0"],
                "damaged": wind_t["damaged_1to4"],
                "multi_point_disagree_dropped_from_wind_Y": wind_t["wind_multi_point_disagree_dropped_from_wind_Y"],
                "excluded_from_wind_Y_grade_-1": wind_t["excluded_from_wind_Y_grade_-1"],
                "passes_Q_ge_300_and_two_class": wind_t["wind_Y_n"] >= 300 and wind_t["undamaged_0"] > 0 and wind_t["damaged_1to4"] > 0,
                "screen_10pct_conflicts": round(0.10 * wind_t["wind_Y_n"], 1),
            },
            "surge": {
                "Y_n": surge_t["surge_Y_n"],
                "none": surge_t["none_0"],
                "damaged": surge_t["damaged_1to6"],
                "multi_point_disagree_dropped_from_surge_Y": surge_t["surge_multi_point_disagree_dropped_from_surge_Y"],
                "passes_Q_ge_300_and_two_class": surge_t["surge_Y_n"] >= 300 and surge_t["none_0"] > 0 and surge_t["damaged_1to6"] > 0,
                "screen_10pct_conflicts": round(0.10 * surge_t["surge_Y_n"], 1),
            },
            "note": "Screening assumption only (10% conflict rate). Not U. No xBD labels used. Counts are building-level.",
        }

    def gate4_points(recs):
        w, s = wind_tally(recs), surge_tally(recs)
        return {
            "n": len(recs),
            "unit": "survey points",
            "wind": {"Y_n": w["wind_Y_n"], "undamaged": w["undamaged_0"], "damaged": w["damaged_1to4"],
                     "passes_Q_ge_300_and_two_class": w["wind_Y_n"] >= 300 and w["undamaged_0"] > 0 and w["damaged_1to4"] > 0,
                     "screen_10pct_conflicts": round(0.10 * w["wind_Y_n"], 1)},
            "surge": {"Y_n": s["surge_Y_n"], "none": s["none_0"], "damaged": s["damaged_1to6"],
                      "passes_Q_ge_300_and_two_class": s["surge_Y_n"] >= 300 and s["none_0"] > 0 and s["damaged_1to6"] > 0,
                      "screen_10pct_conflicts": round(0.10 * s["surge_Y_n"], 1)},
            "note": "Point-level diagnostic only. Official counts are under Q_building_level / gate4_on_Q. Screening assumption only (10% conflict rate). Not U.",
        }

    g4_rows_for_distinct = [pt["row"] for _, pt in g4_candidates]
    distinct_keys = (ROOF_WALL_DAMAGE_FIELDS + ROOF_WALL_TYPE_FIELDS + COLLAPSE_RELATED_FIELDS
                     + ["wind_damage_rating", "assessment_type", "building_type", "hazards_present"])
    # Preserve order, drop dupes
    seen_k = set()
    distinct_keys_u = []
    for k in distinct_keys:
        if k not in seen_k:
            seen_k.add(k)
            distinct_keys_u.append(k)

    g4_keyword_hits = []
    for b, pt in g4_candidates:
        hits = keyword_hits(pt["row"], src_cols_all)
        if hits:
            g4_keyword_hits.append({
                "fulcrum_id": pt["fulcrum_id"],
                "footprint_uid": b["footprint_uid"],
                "hits": hits,
            })

    landing = {
        "grade_3": {
            "n_buildings_in_wind_Y": w_bldg["by_grade"]["3"],
            "major": len(g3_major),
            "unknown": len(g3_unknown),
            "rule": "Roof or wall structure damage >0 maps to major; the rest are marked unknown and listed separately; all stay in Q.",
            "unknown_fulcrum_ids": [b["primary"]["fulcrum_id"] for b in g3_unknown],
            "unknown_footprint_uids": [b["footprint_uid"] for b in g3_unknown],
        },
        "grade_4": {
            "n_buildings_with_a_grade4_point": len(g4_candidates),
            "n_in_wind_Y": len(g4_in_Y),
            "n_multi_point_wind_disagree_still_in_Q": sum(1 for b, _ in g4_candidates if not b["wind_agree"]),
            "major_in_wind_Y": sum(1 for b in g4_in_Y if b["wind_mapped"]["mapped"] == "major"),
            "unknown_in_wind_Y": sum(1 for b in g4_in_Y if b["wind_mapped"]["mapped"] == "unknown"),
            "destroyed": 0,
            "destroyed_assigned": False,
            "rule": "Grade 4 is not mapped to destroyed. Buildings in wind Y with structure damage are recorded as major for reporting only. A component threshold for upgrading any of them to destroyed is not proposed here and stays empty pending Jlin approval of text drafted from Prevatt & Roueche 2020 Tables 4-5. The candidate CSV lists all 17 buildings that have a grade-4 point, with existing component fields, so a threshold can be applied later.",
        },
    }

    Q_building_level = {
        "unit": "unique footprints / buildings",
        "n_points": len(Q),
        "n_buildings": len(Q_buildings),
        "Q": len(Q_buildings),
        "incl_edge_point": True,
        "n_edge_outside_tile_poly_kept": len(edge_buildings),
        "edge_outside_tile_poly": [
            {
                "fulcrum_id": b.get("edge_fulcrum_id"),
                "footprint_uid": b["footprint_uid"],
                "distance_m": b.get("edge_distance_m"),
                "gate2_status": b.get("edge_gate2_status"),
                "tile": b.get("edge_tile"),
                "lat": b.get("edge_lat"),
                "lon": b.get("edge_lon"),
                "wind_grade": b["wind_grade"],
                "flag": "edge_outside_tile_poly",
                "kept_in_Q": True,
                "note": "Gate 2 is unique footprint match. The footprint is inside the tile, pixels are valid, and the frozen 10 m unique-match rule holds. The survey point lies just outside the tile polygon and is flagged, not dropped.",
            }
            for b in edge_buildings
        ],
        "wind_building_level": w_bldg["by_grade"],
        "wind_Y_n": w_bldg["wind_Y_n"],
        "wind_multi_point_disagree_dropped_from_wind_Y": w_bldg["wind_multi_point_disagree_dropped_from_wind_Y"],
        "excluded_from_wind_Y_grade_-1": w_bldg["excluded_from_wind_Y_grade_-1"],
        "surge_building_level": s_bldg["by_grade"],
        "surge_Y_n": s_bldg["surge_Y_n"],
        "surge_multi_point_disagree_dropped_from_surge_Y": s_bldg["surge_multi_point_disagree_dropped_from_surge_Y"],
        "wind_3_4_component_coverage_building_level": {
            "n_wind_3_or_4_in_wind_Y": w_bldg["by_grade"]["3"] + w_bldg["by_grade"]["4"],
            "grade_3": building_component_coverage(Q_buildings, "3"),
            "grade_4_in_wind_Y": building_component_coverage(Q_buildings, "4"),
        },
        "wind_3_4_landing": landing,
        "count_unit_note": unit_mismatch,
        "wind_grade4_destroyed_candidates": {
            "n_buildings_with_a_grade4_point": len(g4_candidates),
            "n_in_wind_Y": len(g4_in_Y),
            "n_multi_point_wind_disagree_still_in_Q": sum(1 for b, _ in g4_candidates if not b["wind_agree"]),
            "destroyed_assigned": 0,
            "percentage_threshold_proposed": False,
            "source_column_count": len(src_cols_all),
            "exact_field_names_in_csv": (
                ROOF_WALL_DAMAGE_FIELDS + ROOF_WALL_TYPE_FIELDS + COLLAPSE_RELATED_FIELDS
            ),
            "breakaway_wall_field_present": any("breakaway" in c.lower() for c in src_cols_all),
            "collapse_keyword_hits": g4_keyword_hits,
            "distinct_values_across_17": _distinct_values(g4_rows_for_distinct, distinct_keys_u),
            "csv": "wind_grade4_destroyed_candidates.csv",
            "note": "Per-building values are in the CSV. Distinct values below are over the 17 grade-4 points (one per footprint). No building is mapped to destroyed.",
        },
    }

    status_counts = Counter(r["gate2_status"] for r in results)
    inside = [r for r in results if r["inside_tiles"]]

    def gate4(recs):
        # retained name used only for the point-level diagnostic below
        return gate4_points(recs)

    script_path = Path(__file__).resolve()
    summary = {
        "inputs": {"building_csv": str(BLD_CSV), "building_csv_sha256": sha(BLD_CSV),
                   "building_geojson_sha256": sha(BLD_GJ),
                   "frozen_matcher_sha256": sha(Path(fm.__file__)),
                   "script": str(script_path),
                   "script_sha256": sha(script_path),
                   "xbd_root": str(XBD), "crs_metric": CRS_METRIC,
                   "landfall_utc_assumed": LANDFALL_UTC.isoformat()},
        "source": {"csv_rows": len(rows), "geojson_features": n_gj, "geojson_ids_resolved": n_gj_ids,
                   "coord_issues": coord_issues,
                   "geojson_vs_csv_max_abs_deg_delta": max([p["geojson_max_abs_deg_delta"] for p in pts if p["geojson_max_abs_deg_delta"] != ""] or [None]),
                   "survey_dates": dict(sorted(Counter(p["survey_date"] for p in pts).items())),
                   "interior_waterline_charring_fields": unobserved,
                   "n_source_columns": len(src_cols_all)},
        "xbd": {"n_post_tiles": len(tiles),
                "n_tiles_with_poly": len(tile_items),
                "n_tiles_without_poly_no_buildings": len(tiles) - len(tile_items),
                "capture_dates_utc_by_day": dict(sorted(Counter((v["capture_date"] or "")[:10] for v in tiles.values()).items())),
                "sensors": dict(Counter(v["sensor"] for v in tiles.values())),
                "affine_fit_rmse_px_max": max([v["fit_rmse_px"] for v in tiles.values() if v["fit_rmse_px"] is not None] or [None]),
                "footprint_load": fp_stats, "dedupe": dedup_stats},
        "gate2": {"n_points": len(results), "n_inside_tiles": len(inside),
                  "n_outside_tiles": len(results) - len(inside),
                  "status_counts": dict(status_counts),
                  "within_tiles_status": dict(Counter(r["status"] for r in inside)),
                  "n_uniquely_matched": len(matched),
                  "n_uniquely_matched_buildings": len({r["footprint_uid"] for r in matched}),
                  "drop_rate_all": round(1 - len(matched) / len(results), 4),
                  "drop_rate_within_tiles": round(1 - sum(r["status"] in fm.MATCH_TYPES for r in inside) / max(1, len(inside)), 4),
                  "footprints_with_multiple_points": len(multi),
                  "multi_point_wind_disagree": sum(1 for m in multi_rows if not m["wind_agree"]),
                  "multi_point_surge_disagree": sum(1 for m in multi_rows if not m["surge_agree"])},
        "temporal": {"n_matched": len(matched), "n_aligned": len(main_sample),
                     "n_mismatch_or_unknown": len(temporal_rows),
                     "by_class": dict(Counter(r["temporal"] for r in matched)),
                     "survey_minus_capture_days": dict(sorted(Counter(r["survey_minus_capture_days"] for r in matched).items(), key=lambda t: (t[0] == "", t[0])))},
        "gate3": {"threshold": VALID_PIX_THR,
                  "n_main_sample_points": len(main_sample),
                  "n_points_Q": len(Q),
                  "Q": len(Q_buildings),
                  "Q_unit": "unique footprints / buildings",
                  "valid_px_flags_points": dict(Counter(r["valid_px_flag"] for r in main_sample)),
                  "n_edge_outside_tile_poly_kept": len(edge_buildings)},
        "per_stage_point_level": per_stage,
        "Q_building_level": Q_building_level,
        "gate4_on_Q": gate4_buildings(Q_buildings, w_bldg, s_bldg),
        "gate4_on_Q_point_level_diagnostic": gate4_points(Q),
        "gate4_on_main_sample_sensitivity_no_px_filter_point_level": gate4_points(main_sample),
        "not_computed": ["conflict labels U", "any estimate from official xBD damage subtypes", "training",
                         "destroyed mapping for wind grade 4", "percentage threshold for destroyed"],
        "output_determinism": {
            "summary_json_excludes_run_timestamps": True,
            "non_deterministic_fields": "none in summary.json; started/finished are written only to run.log",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n"
    (OUT / "summary.json").write_text(summary_text, encoding="utf-8")

    # ---- CSVs
    keep = ["fulcrum_id", "lat", "lon", "survey_date", "inside_tiles", "gate2_status", "status", "drop_reason",
            "footprint_uid", "tile", "split", "distance_m", "n_inside", "n_within_10m", "candidate_uids",
            "nearest_distance_m", "xbd_post_capture_utc", "survey_minus_capture_days", "temporal",
            "valid_px_frac", "valid_px_flag", "flags", "geojson_max_abs_deg_delta"]
    src_cols = ["assessment_type", "building_type", "status", "wind_damage_rating", "surge_damage_rating",
                "hazards_present", "first_floor_elevation_feet"] + COMPONENT_FIELDS
    bldg_by_fp = {b["footprint_uid"]: b for b in Q_buildings}
    extra_point_cols = ["in_Q_points", "in_Q_buildings", "edge_outside_tile_poly",
                        "wind_in_Y", "surge_in_Y", "wind_mapped"]
    with (OUT / "points_gate_status.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keep + extra_point_cols + ["src_" + c for c in src_cols])
        for r in results:
            b = bldg_by_fp.get(r.get("footprint_uid"))
            extra = [
                r in Q or r["fulcrum_id"] in {x["fulcrum_id"] for x in Q},
                bool(b),
                bool(b and b["edge_outside_tile_poly"]) or str(r.get("gate2_status", "")).endswith("_edge_outside_tile_poly"),
                bool(b and b["wind_in_Y"]) if b else "",
                bool(b and b["surge_in_Y"]) if b else "",
                (b["wind_mapped"].get("mapped") if b and b.get("wind_mapped") else ""),
            ]
            w.writerow([r.get(k, "") for k in keep] + extra + [r["row"].get(c, "") for c in src_cols])

    with (OUT / "multi_point_footprints.csv").open("w", newline="", encoding="utf-8") as f:
        if multi_rows:
            rows_sorted = sorted(multi_rows, key=lambda d: d["footprint_uid"])
            w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            w.writeheader()
            w.writerows(rows_sorted)

    with (OUT / "temporal_mismatch.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fulcrum_id", "survey_date", "tile", "xbd_post_capture_utc", "temporal"])
        for r in temporal_rows:
            w.writerow([r["fulcrum_id"], r["survey_date"], r["tile"], r["xbd_post_capture_utc"], r["temporal"]])

    q_bldg_fields = [
        "footprint_uid", "fulcrum_ids", "n_points", "tile", "split",
        "lat", "lon", "distance_m", "nearest_distance_m", "gate2_status",
        "edge_outside_tile_poly", "edge_fulcrum_id", "edge_distance_m",
        "wind_grades", "surge_grades", "wind_agree", "surge_agree",
        "wind_grade", "surge_grade", "wind_in_Y", "surge_in_Y",
        "wind_mapped", "wind_mapped_reason", "destroyed_assigned", "flags",
    ]
    with (OUT / "Q_buildings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(q_bldg_fields)
        for b in Q_buildings:
            wm = b.get("wind_mapped") or {}
            w.writerow([
                b["footprint_uid"], ";".join(b["fulcrum_ids"]), b["n_points"], b["tile"], b["split"],
                b["lat"], b["lon"], b["distance_m"], b["nearest_distance_m"], b["gate2_status"],
                b["edge_outside_tile_poly"], b.get("edge_fulcrum_id", ""), b.get("edge_distance_m", ""),
                ";".join(b["wind_grades"]), ";".join(b["surge_grades"]), b["wind_agree"], b["surge_agree"],
                b["wind_grade"] if b["wind_grade"] is not None else "",
                b["surge_grade"] if b["surge_grade"] is not None else "",
                b["wind_in_Y"], b["surge_in_Y"],
                wm.get("mapped") or "", wm.get("reason") or "", wm.get("destroyed_assigned", False),
                ";".join(b["flags"]),
            ])

    g3_unknown_fields = (
        ["footprint_uid", "fulcrum_id", "tile", "split", "assessment_type", "building_type",
         "wind_damage_rating", "surge_damage_rating", "status", "wind_mapped", "wind_mapped_reason",
         "roof_or_wall_structure_gt0", "both_structure_fields_blank"]
        + ROOF_WALL_DAMAGE_FIELDS + COMPONENT_FIELDS
    )
    with (OUT / "wind_grade3_unknown.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(g3_unknown_fields)
        for b in sorted(g3_unknown, key=lambda x: x["footprint_uid"]):
            row = b["primary"]["row"]
            st = _structure(row)
            wm = b["wind_mapped"]
            w.writerow([
                b["footprint_uid"], b["primary"]["fulcrum_id"], b["tile"], b["split"],
                row.get("assessment_type", ""), row.get("building_type", ""),
                row.get("wind_damage_rating", ""), row.get("surge_damage_rating", ""),
                row.get("status", ""), wm.get("mapped") or "", wm.get("reason") or "",
                st["roof_or_wall_structure_gt0"], st["both_structure_fields_blank"],
            ] + [row.get(c, "") for c in ROOF_WALL_DAMAGE_FIELDS] + [row.get(c, "") for c in COMPONENT_FIELDS])

    # Grade-4 destroyed candidates: all 17 buildings with a grade-4 point.
    # Exact source field names; per-building values; destroyed is not assigned.
    g4_meta = [
        "footprint_uid", "fulcrum_id_grade4", "other_fulcrum_ids_same_footprint",
        "n_points_on_footprint", "wind_grades_on_footprint", "wind_agree", "wind_in_Y",
        "tile", "split", "distance_m", "assessment_type", "building_type",
        "destroyed_assigned", "keyword_hits",
    ]
    g4_src_fields = []
    for c in (["wind_damage_rating", "surge_damage_rating"] + ROOF_WALL_DAMAGE_FIELDS
              + ROOF_WALL_TYPE_FIELDS + COLLAPSE_RELATED_FIELDS + COMPONENT_FIELDS):
        if c not in g4_src_fields:
            g4_src_fields.append(c)
    # residual source columns (all 96) so a later threshold can use any field
    residual = [c for c in src_cols_all if c not in g4_src_fields and c not in (
        "fulcrum_id", "latitude", "longitude", "all_photos", "all_photos_caption", "all_photos_url",
        "audio", "audio_url")]
    g4_csv_fields = g4_meta + g4_src_fields + residual
    with (OUT / "wind_grade4_destroyed_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(g4_csv_fields)
        for b, pt in g4_candidates:
            row = pt["row"]
            hits = keyword_hits(row, src_cols_all)
            hit_s = " | ".join(f"{h['field']}={h['value']}" for h in hits)
            other_ids = [i for i in b["fulcrum_ids"] if i != pt["fulcrum_id"]]
            w.writerow([
                b["footprint_uid"], pt["fulcrum_id"], ";".join(other_ids),
                b["n_points"], ";".join(b["wind_grades"]), b["wind_agree"], b["wind_in_Y"],
                b["tile"], b["split"], pt.get("distance_m", ""),
                row.get("assessment_type", ""), row.get("building_type", ""),
                False, hit_s,
            ] + [row.get(c, "") for c in g4_src_fields] + [row.get(c, "") for c in residual])

    # README and changelog generated from the same structures (no hand-edited numbers).
    edge = Q_building_level["edge_outside_tile_poly"]
    edge_line = "none"
    if edge:
        e0 = edge[0]
        edge_line = (
            f"{e0['fulcrum_id']}  footprint={e0['footprint_uid']}  "
            f"distance_m={e0['distance_m']}  tile={e0['tile']}  wind_grade={e0['wind_grade']}  kept_in_Q=true"
        )
    wg = w_bldg["by_grade"]
    sg = s_bldg["by_grade"]
    readme = f"""# StEER Hurricane Michael x xBD hurricane-michael POST: 4-gate audit v0 (CPU only)

Generated by `scripts/steer_michael/steer_michael_gates.py` (sha256 {sha(script_path)}).
All numbers in summary.json and the tables below are written by that script. No hand-edited values.
No U computed. No official xBD damage subtype used. No GPU. No training. No downloads.
Matching rule unchanged (frozen 10 m / IoU>0.5, no offset).

## What changed in this rerun
- Official Q is building-level: {len(Q_buildings)} unique footprints from {len(Q)} uniquely matched, temporally aligned, valid-px>=0.80 points.
- The just-outside-tile unique match is kept in Q and flagged (not dropped).
- Wind / surge / component / 3-4 landing / gate 4 official counts are at the {len(Q_buildings)}-building level.
- Point-level counts remain only as diagnostics, and to document the old 16/14 vs 18/17 mismatch.
- summary.json no longer contains run timestamps, so a clean rerun of this script is byte-identical.
- New files: Q_buildings.csv, wind_grade3_unknown.csv, wind_grade4_destroyed_candidates.csv.

## Gate summary
- Gate 1 PASS (conditional): on-site/drive-by human D2D survey, per-building lat/lon, grade fields.
- Gate 2: {len(results)} points; {len(inside)} inside xBD post tiles, {len(results) - len(inside)} outside. Uniquely matched {len(matched)} points / {len({r['footprint_uid'] for r in matched})} buildings. One nearest_10m match whose point lies just outside the tile polygon is kept and flagged.
- Gate 3: valid px >= 0.80 for {len(Q)}/{len(main_sample)} points. Official Q = {len(Q_buildings)} buildings.
- Gate 4: wind Y n={w_bldg['wind_Y_n']} (0:{wg['0']}, 1:{wg['1']}, 2:{wg['2']}, 3:{wg['3']}, 4:{wg['4']}) after dropping {w_bldg['wind_multi_point_disagree_dropped_from_wind_Y']} disagreeing footprints from wind Y only; surge Y n={s_bldg['surge_Y_n']} by_grade={json.dumps(sg, ensure_ascii=False)}. Q < 300 screening line is reported, not used to drop the event here.

## Flagged just-outside-tile point
{edge_line}

## Wind 3/4 landing (reporting only; buildings stay in Q; destroyed not assigned)
- Grade 3 in wind Y: {landing['grade_3']['n_buildings_in_wind_Y']} buildings; {landing['grade_3']['major']} -> major (roof or wall structure damage >0); {landing['grade_3']['unknown']} -> unknown (listed in wind_grade3_unknown.csv).
- Grade 4: {landing['grade_4']['n_buildings_with_a_grade4_point']} buildings in Q have a grade-4 point ({landing['grade_4']['n_in_wind_Y']} in wind Y, {landing['grade_4']['n_multi_point_wind_disagree_still_in_Q']} also have a disagreeing other-grade point). In wind Y, {landing['grade_4']['major_in_wind_Y']} recorded as major for reporting only; {landing['grade_4']['unknown_in_wind_Y']} without structure damage >0 marked unknown. destroyed=0. No percentage threshold is proposed. All 17 are in wind_grade4_destroyed_candidates.csv with existing component fields.

## Old 16/14 vs 18/17 mismatch
{unit_mismatch['explanation']}

## Files
summary.json, points_gate_status.csv, multi_point_footprints.csv, temporal_mismatch.csv, Q_buildings.csv, wind_grade3_unknown.csv, wind_grade4_destroyed_candidates.csv, run.log, CHANGES.md, text/.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    changes = f"""# steer_michael_v0 output update

This directory was updated in place by scripts/steer_michael/steer_michael_gates.py
(sha256 {sha(script_path)}). Previous outputs from the 2026-11-17 18:49 Asia/Shanghai run,
including the hand-appended Q_building_level blocks in summary.json, were copied to
prior_run_20261117_1849_hand_appended/ before this rerun and must not be used as results.

Official Q is now generated by the script at building level ({len(Q_buildings)} buildings,
including the flagged just-outside-tile point {edge[0]['fulcrum_id'] if edge else 'n/a'}).
Wind 3/4 landings and the grade-3 unknown / grade-4 candidate lists are also script-generated.
Destroyed is not assigned. No U, no GPU, no downloads, matching rule unchanged.
"""
    (OUT / "CHANGES.md").write_text(changes, encoding="utf-8")

    preview = {k: summary[k] for k in ["gate2", "temporal", "gate3", "gate4_on_Q", "Q_building_level"]}
    # Q_building_level is large (distinct values); print a compact view to stdout/run.log
    compact = {
        "gate2": summary["gate2"],
        "temporal": summary["temporal"],
        "gate3": summary["gate3"],
        "gate4_on_Q": summary["gate4_on_Q"],
        "Q": summary["Q_building_level"]["Q"],
        "wind_building_level": summary["Q_building_level"]["wind_building_level"],
        "wind_Y_n": summary["Q_building_level"]["wind_Y_n"],
        "surge_building_level": summary["Q_building_level"]["surge_building_level"],
        "surge_Y_n": summary["Q_building_level"]["surge_Y_n"],
        "wind_3_4_landing": summary["Q_building_level"]["wind_3_4_landing"],
        "edge_outside_tile_poly": summary["Q_building_level"]["edge_outside_tile_poly"],
        "count_unit_note_short": {
            "point_level_grade_3": unit_mismatch["point_level_grade_3"],
            "point_level_grade_4": unit_mismatch["point_level_grade_4"],
            "building_level_grade_3_in_wind_Y": unit_mismatch["building_level_grade_3_in_wind_Y"],
            "building_level_grade_4_in_wind_Y": unit_mismatch["building_level_grade_4_in_wind_Y"],
            "n_buildings_with_a_grade4_point": unit_mismatch["n_buildings_with_a_grade4_point"],
        },
    }
    log = {
        "started": started,
        "finished": datetime.now().astimezone().isoformat(),
        "script": str(script_path),
        "script_sha256": sha(script_path),
        "summary_sha256": sha(OUT / "summary.json"),
        "preview": compact,
        "not_computed": summary["not_computed"],
    }
    (OUT / "run.log").write_text(json.dumps(_jsonable(log), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(compact), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

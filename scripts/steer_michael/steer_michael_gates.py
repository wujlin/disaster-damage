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
"""
from __future__ import annotations

import csv, hashlib, json, sys
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

    def gate4(recs):
        w, s = wind_tally(recs), surge_tally(recs)
        return {"n": len(recs),
                "wind": {"Y_n": w["wind_Y_n"], "undamaged": w["undamaged_0"], "damaged": w["damaged_1to4"],
                         "passes_Q_ge_300_and_two_class": w["wind_Y_n"] >= 300 and w["undamaged_0"] > 0 and w["damaged_1to4"] > 0,
                         "screen_10pct_conflicts": round(0.10 * w["wind_Y_n"], 1)},
                "surge": {"Y_n": s["surge_Y_n"], "none": s["none_0"], "damaged": s["damaged_1to6"],
                          "passes_Q_ge_300_and_two_class": s["surge_Y_n"] >= 300 and s["none_0"] > 0 and s["damaged_1to6"] > 0,
                          "screen_10pct_conflicts": round(0.10 * s["surge_Y_n"], 1)},
                "note": "Screening assumption only (10% conflict rate). Not U. No xBD labels used."}

    status_counts = Counter(r["gate2_status"] for r in results)
    inside = [r for r in results if r["inside_tiles"]]
    summary = {
        "started": started, "finished": datetime.now().astimezone().isoformat(),
        "inputs": {"building_csv": str(BLD_CSV), "building_csv_sha256": sha(BLD_CSV),
                   "building_geojson_sha256": sha(BLD_GJ),
                   "frozen_matcher_sha256": sha(Path(fm.__file__)),
                   "xbd_root": str(XBD), "crs_metric": CRS_METRIC,
                   "landfall_utc_assumed": LANDFALL_UTC.isoformat()},
        "source": {"csv_rows": len(rows), "geojson_features": n_gj, "geojson_ids_resolved": n_gj_ids,
                   "coord_issues": coord_issues,
                   "geojson_vs_csv_max_abs_deg_delta": max([p["geojson_max_abs_deg_delta"] for p in pts if p["geojson_max_abs_deg_delta"] != ""] or [None]),
                   "survey_dates": dict(sorted(Counter(p["survey_date"] for p in pts).items())),
                   "interior_waterline_charring_fields": unobserved},
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
                  "drop_rate_all": round(1 - len(matched) / len(results), 4),
                  "drop_rate_within_tiles": round(1 - sum(r["status"] in fm.MATCH_TYPES for r in inside) / max(1, len(inside)), 4),
                  "footprints_with_multiple_points": len(multi),
                  "multi_point_wind_disagree": sum(1 for m in multi_rows if not m["wind_agree"]),
                  "multi_point_surge_disagree": sum(1 for m in multi_rows if not m["surge_agree"])},
        "temporal": {"n_matched": len(matched), "n_aligned": len(main_sample),
                     "n_mismatch_or_unknown": len(temporal_rows),
                     "by_class": dict(Counter(r["temporal"] for r in matched)),
                     "survey_minus_capture_days": dict(sorted(Counter(r["survey_minus_capture_days"] for r in matched).items(), key=lambda t: (t[0] == "", t[0])))},
        "gate3": {"threshold": VALID_PIX_THR, "n_main_sample": len(main_sample), "Q": len(Q),
                  "valid_px_flags": dict(Counter(r["valid_px_flag"] for r in main_sample))},
        "per_stage": per_stage,
        "gate4_on_Q": gate4(Q),
        "gate4_on_main_sample_sensitivity_no_px_filter": gate4(main_sample),
        "not_computed": ["conflict labels U", "any estimate from official xBD damage subtypes", "training"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    keep = ["fulcrum_id", "lat", "lon", "survey_date", "inside_tiles", "gate2_status", "status", "drop_reason",
            "footprint_uid", "tile", "split", "distance_m", "n_inside", "n_within_10m", "candidate_uids",
            "nearest_distance_m", "xbd_post_capture_utc", "survey_minus_capture_days", "temporal",
            "valid_px_frac", "valid_px_flag", "flags", "geojson_max_abs_deg_delta"]
    src_cols = ["assessment_type", "building_type", "status", "wind_damage_rating", "surge_damage_rating",
                "hazards_present", "first_floor_elevation_feet"] + COMPONENT_FIELDS
    with (OUT / "points_gate_status.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keep + ["src_" + c for c in src_cols])
        for r in results:
            w.writerow([r.get(k, "") for k in keep] + [r["row"].get(c, "") for c in src_cols])
    with (OUT / "multi_point_footprints.csv").open("w", newline="", encoding="utf-8") as f:
        if multi_rows:
            w = csv.DictWriter(f, fieldnames=list(multi_rows[0].keys())); w.writeheader(); w.writerows(multi_rows)
    with (OUT / "temporal_mismatch.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fulcrum_id", "survey_date", "tile", "xbd_post_capture_utc", "temporal"])
        for r in temporal_rows:
            w.writerow([r["fulcrum_id"], r["survey_date"], r["tile"], r["xbd_post_capture_utc"], r["temporal"]])
    print(json.dumps({k: summary[k] for k in ["gate2", "temporal", "gate3", "gate4_on_Q"]}, indent=1, default=str))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
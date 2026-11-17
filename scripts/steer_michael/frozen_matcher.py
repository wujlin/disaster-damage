#!/usr/bin/env python3
"""Santa Rosa DINS-to-xBD footprint match audit (CPU only).

Frozen matching rule (do not change):
  - Point inside exactly one footprint -> match type ``inside``, even if
    other footprints lie within 10 m.
  - Point inside >= 2 footprints -> drop ``ambiguous_inside``.
  - Otherwise, using shortest point-to-polygon distance:
      exactly one footprint within 10 m -> match type ``nearest_10m``;
      >= 2 within 10 m -> drop ``ambiguous_10m``;
      none -> drop ``no_footprint_10m``.

Footprints that overlap across tiles/splits with IoU > 0.5 are treated as
the same building (union-find, then geometry union). Distances are in
a chosen metric CRS (default UTM 10N / EPSG:32610). This script does not compute conflict labels
and does not start training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import mapping, point as shp_point, shape
from shapely.ops import unary_union
from shapely.wkt import loads as load_wkt

TZ = ZoneInfo("Asia/Shanghai")
CRS_METRIC = "EPSG:32610"
IOU_DEDUP = 0.5
NEAR_M = 10.0
SPLITS = ("train", "test", "hold")
EVENT = "santa-rosa-wildfire"

GRADE_MAP = {
    "destroyed (>50%)": "Destroyed",
    "major (25-50%)": "Major",
    "minor (10-25%)": "Minor",
    "affected (>0-10%)": "Affected",
    "no damage": "No Damage",
    "inaccessible": "Inaccessible",
}
GRADE_ORDER = ["Destroyed", "Major", "Minor", "Affected", "No Damage", "Inaccessible"]
STATUS_ORDER = [
    "inside",
    "nearest_10m",
    "ambiguous_inside",
    "ambiguous_10m",
    "no_footprint_10m",
]
MATCH_TYPES = {"inside", "nearest_10m"}
DROP_REASONS = {"ambiguous_inside", "ambiguous_10m", "no_footprint_10m"}


def now_shanghai() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


def sha256_file(path: Path, buf: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
        )
    except Exception as exc:  # pragma: no cover
        return f"<unavailable: {exc}>"


def short_grade(raw: str) -> str:
    key = (raw or "").strip()
    mapped = GRADE_MAP.get(key.lower())
    return mapped if mapped else (key or "(empty)")


def normalize_wkt_poly(wkt: str):
    if not wkt or not str(wkt).strip():
        return None
    try:
        geom = load_wkt(wkt)
    except Exception:
        return None
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon" and not geom.exterior.is_ccw:
        geom = geom.normalize()
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom


def project_geom(geom, transformer: Transformer):
    def _xy(xs, ys):
        xx, yy = transformer.transform(xs, ys)
        return xx, yy

    return shape(
        {
            "type": mapping(geom)["type"],
            "coordinates": _project_coords(mapping(geom)["coordinates"], transformer),
        }
    )


def _project_coords(coords, transformer: Transformer):
    if not coords:
        return coords
    if isinstance(coords[0], (int, float)):
        x, y = transformer.transform(coords[0], coords[1])
        return (x, y) + tuple(coords[2:])
    return tuple(_project_coords(c, transformer) for c in coords)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            self.parent[ri] = rj
        elif self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
        else:
            self.parent[rj] = ri
            self.rank[ri] += 1


def load_dins_points(path: Path, transformer: Transformer):
    with path.open(encoding="utf-8") as f:
        fc = json.load(f)
    points = []
    skipped = 0
    for feat in fc.get("features", []):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") if geom else None
        lon = lat = None
        if coords and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
        else:
            try:
                lon = float(props.get("Longitude") or props.get("LONGITUDE"))
                lat = float(props.get("Latitude") or props.get("LATITUDE"))
            except (TypeError, ValueError):
                skipped += 1
                continue
        x, y = transformer.transform(lon, lat)
        grade_raw = (
            props.get("* Damage")
            or props.get("DAMAGE")
            or props.get("Damage")
            or ""
        )
        points.append(
            {
                "dins_id": props.get("GLOBALID") or props.get("OBJECTID") or "",
                "objectid": str(props.get("OBJECTID") or ""),
                "globalid": str(props.get("GLOBALID") or ""),
                "grade_raw": str(grade_raw),
                "grade": short_grade(str(grade_raw)),
                "incident": (
                    props.get("* Incident Name")
                    or props.get("INCIDENTNAME")
                    or ""
                ),
                "lon": lon,
                "lat": lat,
                "point": shp_point.Point(x, y),
            }
        )
    return points, skipped


def load_post_footprints(xbd_root: Path, transformer: Transformer, event: str = EVENT):
    raw = []
    n_files = 0
    n_empty_files = 0
    n_bad_wkt = 0
    n_non_building = 0
    fn_re = re.compile(
        rf"^{re.escape(event)}_(\d+)_(pre|post)_disaster\.json$", re.I
    )
    for split in SPLITS:
        lab = xbd_root / split / "labels"
        if not lab.is_dir():
            continue
        for path in sorted(lab.glob(f"{event}_*_post_disaster.json")):
            n_files += 1
            m = fn_re.match(path.name)
            tile_id = m.group(1) if m else path.stem
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            feats = (data.get("features") or {}).get("lng_lat") or []
            kept = 0
            for feat in feats:
                props = feat.get("properties") or {}
                ftype = (props.get("feature_type") or "building").lower()
                if ftype and ftype != "building":
                    n_non_building += 1
                    continue
                geom = normalize_wkt_poly(feat.get("wkt") or "")
                if geom is None:
                    n_bad_wkt += 1
                    continue
                try:
                    geom_m = project_geom(geom, transformer)
                except Exception:
                    n_bad_wkt += 1
                    continue
                if geom_m is None or geom_m.is_empty or geom_m.area <= 0:
                    n_bad_wkt += 1
                    continue
                if not geom_m.is_valid:
                    geom_m = geom_m.buffer(0)
                if geom_m.is_empty or geom_m.area <= 0:
                    n_bad_wkt += 1
                    continue
                raw.append(
                    {
                        "uid": str(props.get("uid") or ""),
                        "subtype": str(props.get("subtype") or ""),
                        "tile": path.name,
                        "tile_id": tile_id,
                        "split": split,
                        "geom": geom_m,
                        "area": float(geom_m.area),
                    }
                )
                kept += 1
            if kept == 0:
                n_empty_files += 1
    stats = {
        "n_post_label_files": n_files,
        "n_empty_or_unusable_files": n_empty_files,
        "n_raw_footprints": len(raw),
        "n_bad_wkt": n_bad_wkt,
        "n_non_building_skipped": n_non_building,
    }
    return raw, stats


def dedupe_footprints(raw: list[dict], iou_thr: float = IOU_DEDUP):
    n = len(raw)
    if n == 0:
        return [], {"n_raw": 0, "n_after": 0, "n_merged_clusters": 0}
    geoms = [r["geom"] for r in raw]
    tree = STRtree(geoms)
    uf = UnionFind(n)
    n_pairs = 0
    for i, geom in enumerate(geoms):
        try:
            cand_idx = tree.query(geom, predicate="intersects")
        except Exception:
            cand_idx = tree.query(geom)
        for j in cand_idx:
            j = int(j)
            if j <= i:
                continue
            other = geoms[j]
            try:
                inter = geom.intersection(other)
                inter_area = inter.area if not inter.is_empty else 0.0
            except Exception:
                continue
            if inter_area <= 0:
                continue
            union_area = geom.area + other.area - inter_area
            if union_area <= 0:
                continue
            iou = inter_area / union_area
            if iou > iou_thr:
                uf.union(i, j)
                n_pairs += 1
    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    canonical = []
    n_merged = 0
    max_cluster = 1
    for members in clusters.values():
        max_cluster = max(max_cluster, len(members))
        recs = [raw[i] for i in members]
        recs.sort(key=lambda r: (-r["area"], r["split"], r["tile"], r["uid"]))
        rep = recs[0]
        if len(recs) > 1:
            n_merged += 1
            try:
                merged_geom = unary_union([r["geom"] for r in recs])
            except Exception:
                merged_geom = rep["geom"]
            if merged_geom is None or merged_geom.is_empty:
                merged_geom = rep["geom"]
            if not merged_geom.is_valid:
                merged_geom = merged_geom.buffer(0) or rep["geom"]
        else:
            merged_geom = rep["geom"]
        subtypes = sorted({r["subtype"] for r in recs if r["subtype"]})
        canonical.append(
            {
                "uid": rep["uid"],
                "subtype": rep["subtype"],
                "tile": rep["tile"],
                "tile_id": rep["tile_id"],
                "split": rep["split"],
                "geom": merged_geom,
                "n_sources": len(recs),
                "source_uids": [r["uid"] for r in recs],
                "source_tiles": [r["tile"] for r in recs],
                "source_splits": sorted({r["split"] for r in recs}),
                "source_subtypes": subtypes,
            }
        )
    stats = {
        "method": (
            "Union-find on projected polygons; IoU > "
            f"{iou_thr} treated as the same building. Canonical geometry is "
            "the unary union (fallback: largest-area member). Canonical uid / "
            "tile / split / subtype come from the largest-area member "
            "(tie-break: split order train/test/hold, then tile name, uid)."
        ),
        "iou_threshold": iou_thr,
        "n_raw": n,
        "n_after": len(canonical),
        "n_removed_by_dedupe": n - len(canonical),
        "n_merged_clusters": n_merged,
        "n_iou_pairs_above_threshold": n_pairs,
        "max_cluster_size": max_cluster,
        "n_singleton_kept": len(canonical) - n_merged,
    }
    return canonical, stats


def classify_points(points: list[dict], footprints: list[dict]):
    if not footprints:
        results = []
        for p in points:
            results.append(
                {
                    **p,
                    "status": "no_footprint_10m",
                    "match_type": "",
                    "drop_reason": "no_footprint_10m",
                    "footprint_uid": "",
                    "tile": "",
                    "split": "",
                    "distance_m": "",
                    "xbd_subtype": "",
                    "n_inside": 0,
                    "n_within_10m": 0,
                    "candidate_uids": "",
                    "nearest_uid": "",
                    "nearest_distance_m": "",
                }
            )
        return results
    geoms = [fp["geom"] for fp in footprints]
    tree = STRtree(geoms)
    results = []
    for p in points:
        pt = p["point"]
        # Candidates that may cover the point, plus anything within 10 m.
        query_geom = pt.buffer(NEAR_M + 1e-6)
        try:
            cand_idx = [int(i) for i in tree.query(query_geom, predicate="intersects")]
        except Exception:
            cand_idx = [int(i) for i in tree.query(query_geom)]
        # Also query the bare point in case a covering polygon was missed.
        try:
            extra = [int(i) for i in tree.query(pt, predicate="intersects")]
        except Exception:
            extra = [int(i) for i in tree.query(pt)]
        cand_idx = sorted(set(cand_idx) | set(extra))

        inside = []
        within = []  # (idx, distance) for 0 < d <= 10, and d == 0 already in inside
        nearest_i = None
        nearest_d = None
        for i in cand_idx:
            g = geoms[i]
            try:
                is_inside = bool(g.covers(pt)) or g.distance(pt) <= 1e-9
            except Exception:
                is_inside = False
            try:
                dist = 0.0 if is_inside else float(g.distance(pt))
            except Exception:
                continue
            if nearest_d is None or dist < nearest_d:
                nearest_d = dist
                nearest_i = i
            if is_inside:
                inside.append(i)
            elif dist <= NEAR_M:
                within.append((i, dist))

        rec = {
            **p,
            "status": "",
            "match_type": "",
            "drop_reason": "",
            "footprint_uid": "",
            "tile": "",
            "split": "",
            "distance_m": "",
            "xbd_subtype": "",
            "n_inside": len(inside),
            "n_within_10m": len(inside) + len(within),
            "candidate_uids": "",
            "nearest_uid": footprints[nearest_i]["uid"] if nearest_i is not None else "",
            "nearest_distance_m": (
                f"{nearest_d:.6f}" if nearest_d is not None else ""
            ),
        }

        def attach(idx, dist):
            fp = footprints[idx]
            rec["footprint_uid"] = fp["uid"]
            rec["tile"] = fp["tile"]
            rec["split"] = fp["split"]
            rec["distance_m"] = f"{dist:.6f}"
            rec["xbd_subtype"] = fp["subtype"]
            rec["source_splits"] = ",".join(fp["source_splits"])
            rec["n_sources"] = fp["n_sources"]

        if len(inside) == 1:
            rec["status"] = "inside"
            rec["match_type"] = "inside"
            attach(inside[0], 0.0)
        elif len(inside) >= 2:
            rec["status"] = "ambiguous_inside"
            rec["drop_reason"] = "ambiguous_inside"
            rec["candidate_uids"] = ";".join(footprints[i]["uid"] for i in inside)
            # diagnostics: nearest of the containing footprints (distance 0)
            attach(inside[0], 0.0)
        elif len(within) == 1:
            idx, dist = within[0]
            rec["status"] = "nearest_10m"
            rec["match_type"] = "nearest_10m"
            attach(idx, dist)
        elif len(within) >= 2:
            rec["status"] = "ambiguous_10m"
            rec["drop_reason"] = "ambiguous_10m"
            within.sort(key=lambda t: t[1])
            rec["candidate_uids"] = ";".join(
                f"{footprints[i]['uid']}:{d:.3f}m" for i, d in within
            )
            attach(within[0][0], within[0][1])
        else:
            rec["status"] = "no_footprint_10m"
            rec["drop_reason"] = "no_footprint_10m"
            if nearest_i is not None:
                rec["distance_m"] = f"{nearest_d:.6f}"
                rec["footprint_uid"] = footprints[nearest_i]["uid"]
                rec["tile"] = footprints[nearest_i]["tile"]
                rec["split"] = footprints[nearest_i]["split"]
                rec["xbd_subtype"] = footprints[nearest_i]["subtype"]
        results.append(rec)
    return results


def ordered_keys(counter_keys, preferred, keep_preferred=False):
    seen = []
    for k in preferred:
        if keep_preferred or k in counter_keys:
            if k not in seen:
                seen.append(k)
    for k in sorted(counter_keys):
        if k not in seen:
            seen.append(k)
    return seen


def crosstab(results, row_key, col_key, row_order, col_order, keep_row_order=False, keep_col_order=True):
    table = Counter((r[row_key], r[col_key]) for r in results)
    rows = ordered_keys({k[0] for k in table}, row_order, keep_preferred=keep_row_order)
    cols = ordered_keys({k[1] for k in table}, col_order, keep_preferred=keep_col_order)
    out = {"rows": rows, "cols": cols, "counts": {}, "row_totals": {}, "col_totals": {}, "total": 0}
    for i in rows:
        out["counts"][i] = {}
        rt = 0
        for j in cols:
            n = table.get((i, j), 0)
            out["counts"][i][j] = n
            rt += n
        out["row_totals"][i] = rt
        out["total"] += rt
    for j in cols:
        out["col_totals"][j] = sum(table.get((i, j), 0) for i in rows)
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def render_table(ct: dict) -> str:
    rows, cols = ct["rows"], ct["cols"]
    header = ["", *cols, "TOTAL"]
    widths = [max(16, max((len(str(r)) for r in rows), default=4))]
    for j, c in enumerate(cols):
        w = len(str(c))
        for r in rows:
            w = max(w, len(str(ct["counts"][r].get(c, 0))))
        w = max(w, len(str(ct["col_totals"].get(c, 0))))
        widths.append(w)
    widths.append(max(5, len(str(ct["total"]))))
    lines = []

    def fmt(vals):
        return "  ".join(str(v).ljust(widths[i]) if i == 0 else str(v).rjust(widths[i]) for i, v in enumerate(vals))

    lines.append(fmt(header))
    lines.append(fmt(["-" * widths[0], *["-" * w for w in widths[1:]]]))
    for r in rows:
        lines.append(fmt([r, *[ct["counts"][r].get(c, 0) for c in cols], ct["row_totals"][r]]))
    lines.append(fmt(["TOTAL", *[ct["col_totals"].get(c, 0) for c in cols], ct["total"]]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dins",
        default="/mnt/data_hdd/datasets/cal_fire_dins/cal_fire_dins_tubbs_nuns_2017_xbd_santa_rosa_extents.geojson",
    )
    parser.add_argument("--xbd-root", default="/mnt/data_hdd/datasets/xbd")
    parser.add_argument("--out", default="/mnt/data_hdd/audits/dins_santa_rosa_v0")
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--event",
        default=EVENT,
        help="xBD disaster name used in label filenames (e.g. santa-rosa-wildfire, socal-fire)",
    )
    parser.add_argument(
        "--crs",
        default=CRS_METRIC,
        help="Metric CRS for distances (e.g. EPSG:32610 UTM 10N, EPSG:32611 UTM 11N)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title written at the top of README.txt",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dins_path = Path(args.dins)
    xbd_root = Path(args.xbd_root)
    repo = Path(args.repo)
    script_path = Path(__file__).resolve()

    event = args.event
    crs = args.crs
    title = args.title or f"{event} DINS footprint match audit"
    crs_label = {
        "EPSG:32610": "UTM 10N",
        "EPSG:32611": "UTM 11N",
    }.get(crs, crs)

    print(f"START {now_shanghai()}", flush=True)
    print(f"EVENT={event} CRS={crs} ({crs_label})", flush=True)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    points, n_skipped = load_dins_points(dins_path, transformer)
    print(f"DINS points={len(points)} skipped={n_skipped}", flush=True)
    raw, load_stats = load_post_footprints(xbd_root, transformer, event=event)
    print(f"RAW footprints {load_stats}", flush=True)
    footprints, dedupe_stats = dedupe_footprints(raw, IOU_DEDUP)
    print(f"DEDUPE {dedupe_stats}", flush=True)
    results = classify_points(points, footprints)
    print(f"CLASSIFIED {len(results)}", flush=True)

    matches = [r for r in results if r["status"] in MATCH_TYPES]
    drops = [r for r in results if r["status"] in DROP_REASONS]

    # Footprints claimed by more than one matched DINS point.
    claimed = defaultdict(list)
    for r in matches:
        if r["footprint_uid"]:
            claimed[r["footprint_uid"]].append(r)
    multi = {uid: recs for uid, recs in claimed.items() if len(recs) > 1}
    n_multi_fp = len(multi)
    n_multi_pts = sum(len(v) for v in multi.values())
    multi_examples = []
    for uid, recs in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:20]:
        multi_examples.append(
            {
                "footprint_uid": uid,
                "n_points": len(recs),
                "dins_ids": [r["dins_id"] for r in recs],
                "grades": [r["grade"] for r in recs],
                "match_types": [r["status"] for r in recs],
                "tile": recs[0]["tile"],
                "split": recs[0]["split"],
            }
        )

    status_ct = crosstab(results, "grade", "status", GRADE_ORDER, STATUS_ORDER)
    split_counts = Counter(r["split"] for r in matches)
    matched_by_split = {s: split_counts.get(s, 0) for s in SPLITS}
    matched_by_split["(empty_split)"] = split_counts.get("", 0)
    confusion = crosstab(
        matches,
        "grade",
        "xbd_subtype",
        GRADE_ORDER,
        ["destroyed", "major-damage", "minor-damage", "no-damage", "un-classified"],
    )

    match_fields = [
        "dins_id",
        "objectid",
        "incident",
        "grade",
        "grade_raw",
        "match_type",
        "footprint_uid",
        "tile",
        "split",
        "distance_m",
        "xbd_subtype",
        "n_sources",
        "source_splits",
    ]
    drop_fields = [
        "dins_id",
        "objectid",
        "incident",
        "grade",
        "grade_raw",
        "drop_reason",
        "n_inside",
        "n_within_10m",
        "candidate_uids",
        "nearest_uid",
        "nearest_distance_m",
        "footprint_uid",
        "tile",
        "split",
        "distance_m",
        "xbd_subtype",
    ]
    write_csv(out / "matches.csv", matches, match_fields)
    write_csv(out / "drops.csv", drops, drop_fields)

    dins_sha = sha256_file(dins_path)
    script_sha = sha256_file(script_path)
    commit = git_commit(repo)
    status_counts = Counter(r["status"] for r in results)
    grade_counts = Counter(r["grade"] for r in results)

    summary = {
        "created": now_shanghai(),
        "timezone": "Asia/Shanghai",
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "crs": crs,
        "near_threshold_m": NEAR_M,
        "git_commit": commit,
        "script": str(script_path),
        "script_sha256": script_sha,
        "inputs": {
            "dins_points": {
                "path": str(dins_path),
                "sha256": dins_sha,
                "n_features_loaded": len(points),
                "n_skipped": n_skipped,
            },
            "xbd_labels": {
                "root": str(xbd_root),
                "event": event,
                "when": "post_disaster",
                "splits": list(SPLITS),
                **load_stats,
            },
        },
        "dedupe": dedupe_stats,
        "matching_rule": {
            "inside_exactly_one": "match type inside (even if other footprints within 10 m)",
            "inside_two_or_more": "drop ambiguous_inside",
            "outside_exactly_one_within_10m": "match type nearest_10m, shortest point-to-polygon distance",
            "outside_two_or_more_within_10m": "drop ambiguous_10m",
            "outside_none_within_10m": "drop no_footprint_10m",
            "distance": "shortest point-to-polygon distance in {} meters".format(crs),
        },
        "n_points": len(results),
        "n_matched": len(matches),
        "n_dropped": len(drops),
        "status_counts": {k: status_counts.get(k, 0) for k in STATUS_ORDER},
        "grade_counts": {k: grade_counts.get(k, 0) for k in ordered_keys(grade_counts, GRADE_ORDER)},
        "crosstab_grade_by_status": status_ct,
        "matched_by_official_split": matched_by_split,
        "multi_claimed_footprints": {
            "n_footprints_claimed_by_gt1_dins_point": n_multi_fp,
            "n_dins_points_involved": n_multi_pts,
            "examples_top20": multi_examples,
        },
        "confusion_dins_grade_vs_xbd_post_subtype_matched_only": confusion,
        "note": (
            "Confusion table is raw co-occurrence counts for matched pairs only. "
            "No conflict label was computed and no conflict interpretation is made."
        ),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    readme = []
    readme.append(title)
    readme.append("")
    readme.append(f"Created: {summary['created']}")
    readme.append(f"Host: {summary['host']}")
    readme.append(f"Worktree / script: {script_path}")
    readme.append(f"Git commit of this script: {commit}")
    readme.append(f"Script sha256: {script_sha}")
    readme.append("CPU only. No training, no GPU, no conflict computation.")
    readme.append("")
    readme.append("Inputs")
    readme.append(f"  DINS in-extent points: {dins_path}")
    readme.append(f"    sha256: {dins_sha}")
    readme.append(f"    loaded: {len(points)}  skipped: {n_skipped}")
    readme.append(
        f"  xBD footprints: {event} post_disaster label JSONs under "
        f"{xbd_root}/{{train,test,hold}}/labels"
    )
    readme.append(f"    post label files: {load_stats['n_post_label_files']}")
    readme.append(f"    raw building footprints: {load_stats['n_raw_footprints']}")
    readme.append(f"    bad/empty WKT skipped: {load_stats['n_bad_wkt']}")
    readme.append(
        f"    non-building features skipped: {load_stats['n_non_building_skipped']}"
    )
    readme.append(f"  Metric CRS: {crs} ({crs_label})")
    readme.append("")
    readme.append("Footprint dedupe across tiles/splits")
    readme.append(f"  {dedupe_stats['method']}")
    readme.append(f"  before: {dedupe_stats['n_raw']}")
    readme.append(f"  after:  {dedupe_stats['n_after']}")
    readme.append(f"  removed as duplicates: {dedupe_stats['n_removed_by_dedupe']}")
    readme.append(f"  clusters with 2+ members: {dedupe_stats['n_merged_clusters']}")
    readme.append(
        f"  IoU pairs above threshold: {dedupe_stats['n_iou_pairs_above_threshold']}"
    )
    readme.append(f"  max cluster size: {dedupe_stats['max_cluster_size']}")
    readme.append("")
    readme.append("Frozen matching rule")
    for k, v in summary["matching_rule"].items():
        readme.append(f"  {k}: {v}")
    readme.append("")
    readme.append("Results")
    readme.append(f"  points: {len(results)}  matched: {len(matches)}  dropped: {len(drops)}")
    readme.append(f"  status counts: { {k: status_counts.get(k, 0) for k in STATUS_ORDER} }")
    readme.append("")
    readme.append("Crosstab: DINS grade x match type / drop reason")
    readme.append(render_table(status_ct))
    readme.append("")
    readme.append("Matched counts by official xBD split (of the canonical footprint)")
    for s in SPLITS:
        readme.append(f"  {s}: {matched_by_split.get(s, 0)}")
    readme.append("")
    readme.append("Footprints claimed by more than one matched DINS point")
    readme.append(f"  footprints: {n_multi_fp}")
    readme.append(f"  DINS points involved: {n_multi_pts}")
    readme.append("")
    readme.append("DINS grade vs xBD post subtype (matched pairs only, raw counts)")
    readme.append("No conflict interpretation is attached to this table.")
    readme.append(render_table(confusion))
    readme.append("")
    readme.append("Output files")
    readme.append("  matches.csv  — matched DINS points (inside, nearest_10m)")
    readme.append("  drops.csv    — dropped DINS points (ambiguous_*, no_footprint_10m)")
    readme.append("  summary.json — machine-readable counts, including both crosstabs")
    readme.append("  README.txt   — this file")
    text = "\n".join(readme) + "\n"
    (out / "README.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"WROTE {out}", flush=True)
    print(f"END {now_shanghai()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

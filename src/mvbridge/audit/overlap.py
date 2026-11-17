"""Event-level and spatial overlap between SAGIN groups and official xBD splits."""

from __future__ import annotations

import json
from pathlib import Path

from mvbridge.catalog import EventCatalog, EventMatch, load_catalog, match_event
from mvbridge.data.exclusions import write_id_list
from mvbridge.data.sagin_index import load_sagin_groups
from mvbridge.data.xbd_index import XBDImageGeo, index_xbd
from mvbridge.data.xbd_labels import OFFICIAL_SPLIT_NAMES
from mvbridge.errors import ConfigError, DataError
from mvbridge.geo import haversine_m, parse_date_span, point_in_expanded_bbox, spans_overlap, valid_lat_lon

# Literature context printed with every report. These numbers are not a check
# that the local metadata file is complete.
SAGIN_LITERATURE = {
    "groups": 2080,
    "images": 118560,
    "simulated_uav_videos": 2080,
    "views": ["vgi", "panoramic svi", "uav", "rsi"],
    "sources": [
        "Yin et al., SAGINGeo, IEEE JSTARS, https://doi.org/10.1109/jstars.2026.3712316",
        "Li et al., ProbGLC, ISPRS JPRS 2026, https://github.com/bobleegogogo/ProbGLC",
        "Gupta et al., xBD, arXiv:1911.09296",
    ],
}

RULE_TEXT = (
    "Event overlap requires both records to resolve to the same catalog event id. "
    "Ids are date-specific, so Midwest flooding 2015 and the xBD Midwest floods of 2019 "
    "are different events. Token rules and exact aliases live in the event catalog; "
    "a bare shared word such as flood or fire is not a match. Names listed as ambiguous, "
    "and names that hit more than one id, are reported and are not exclusions. "
    "When a SAGIN record has a date and the catalog event has a date range, the spans "
    "must overlap after date_window_days of slack. A SAGIN date outside that window "
    "is a date disagreement and is not an event exclusion. "
    "Spatial overlap is independent: a SAGIN coordinate hits an xBD image when the "
    "haversine distance to the nearest building centroid is <= buffer_m. "
    "A building centroid is the mean of the lng_lat WKT vertices. "
    "The candidate images are those whose vertex-centroid bounding box, expanded by "
    "buffer_m, contains the SAGIN point. "
    "The primary recommendation excludes SAGIN groups that hit the configured "
    "downstream splits (test and holdout by default) from continued pretraining, "
    "and leaves the official xBD test split unchanged. "
    "The alternative file lists those xBD test/holdout image ids so they can be "
    "removed from evaluation instead. Applying that alternative means the reported "
    "test set is no longer the untouched official split."
)


def run_audit(config: dict) -> dict:
    buffer_m = float(config.get("buffer_m", 1000))
    if buffer_m < 0:
        raise ConfigError("buffer_m must be >= 0")
    window_days = int(config.get("date_window_days", 14))
    catalog = load_catalog(config["catalog"])
    sagin_cfg = config.get("sagin") or {}
    xbd_cfg = config.get("xbd") or {}
    if not xbd_cfg.get("root"):
        raise ConfigError("xbd.root is required. Pass the local xBD/xView2 directory with --set xbd.root=...")
    groups = load_sagin_groups(sagin_cfg, check_files=False)
    if not groups:
        raise DataError("SAGIN metadata loaded zero place groups.")
    split_dirs = xbd_cfg.get("split_dirs") or {"train": "train", "test": "test", "holdout": "holdout"}
    xbd_index = index_xbd(xbd_cfg["root"], split_dirs)
    pretrain_splits = set(config.get("pretrain_exclusion_splits") or ["test", "holdout"])
    downstream_splits = set(config.get("downstream_exclusion_splits") or ["test", "holdout"])
    _check_split_names(pretrain_splits)
    _check_split_names(downstream_splits)
    payload = _compare(
        groups=groups,
        images=xbd_index.images,
        catalog=catalog,
        buffer_m=buffer_m,
        window_days=window_days,
        pretrain_splits=pretrain_splits,
        downstream_splits=downstream_splits,
    )
    payload["xbd_index"] = {
        "label_files": xbd_index.label_files,
        "bad_wkt": xbd_index.bad_wkt,
        "missing_splits": xbd_index.missing_splits,
    }
    payload["rule"] = {
        "text": RULE_TEXT,
        "buffer_m": buffer_m,
        "date_window_days": window_days,
        "catalog": catalog.path,
        "pretrain_exclusion_splits": sorted(pretrain_splits),
        "downstream_exclusion_splits": sorted(downstream_splits),
        "literature": SAGIN_LITERATURE,
    }
    output_dir = config.get("output_dir")
    if output_dir:
        paths = write_audit_outputs(Path(output_dir), payload)
        payload["outputs"] = {key: str(path) for key, path in paths.items()}
    return payload


def write_audit_outputs(output_dir: Path, payload: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = format_report(payload)
    json_path = output_dir / "overlap_report.json"
    text_path = output_dir / "overlap_report.txt"
    pretrain_path = output_dir / "exclude_pretrain_sagin_groups.txt"
    downstream_path = output_dir / "alternative_exclude_downstream_xbd_images.txt"
    train_path = output_dir / "train_split_spatial_hits.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    write_id_list(
        pretrain_path,
        payload["pretrain_exclusion_ids"],
        "SAGIN group ids to drop from continued pretraining. Official xBD test/holdout can stay intact.",
    )
    write_id_list(
        downstream_path,
        payload["downstream_exclusion_ids"],
        "Alternative: xBD test/holdout image ids to drop from evaluation. Using this file departs from the official split.",
    )
    train_ids = sorted({hit["sagin_group_id"] for hit in payload["train_only_spatial_hits"]})
    write_id_list(
        train_path,
        train_ids,
        "SAGIN groups that only meet xBD train geography. Not removed by the default pretraining exclusion.",
    )
    return {
        "json": json_path,
        "text": text_path,
        "pretrain": pretrain_path,
        "downstream": downstream_path,
        "train_only": train_path,
    }


def format_report(payload: dict) -> str:
    counts = payload["counts"]
    rule = payload["rule"]
    lines = [
        "SAGIN–xBD overlap audit",
        "",
        "Rule",
        rule["text"],
        f"buffer_m: {rule['buffer_m']}",
        f"date_window_days: {rule['date_window_days']}",
        f"catalog: {rule['catalog']}",
        f"pretrain exclusion splits: {', '.join(rule['pretrain_exclusion_splits'])}",
        f"downstream exclusion splits: {', '.join(rule['downstream_exclusion_splits'])}",
        "",
        "Literature context (not a file-count check)",
        (
            "SAGINGeo reports 2080 groups, 118560 images, and 2080 simulated UAV videos "
            "across VGI, panoramic street view, UAV, and remote sensing. "
            "ProbGLC says the archive draws on more than 50 NOAA events and names seven examples: "
            "California wildfires (2025), Maui wildfires (2023), Hurricane Ian (2022; the ProbGLC PDF "
            "also writes 'IAN (2025)' in one sentence, which disagrees with the 28 September 2022 landfall "
            "in that same section), Nashville tornado (2020), Illinois tornado (2015), "
            "Louisiana flood (2016), Midwest flood (2015). "
            "xBD (Gupta et al., Table 2) is train/test/holdout = 18336/1866/1866 images."
        ),
        "",
        "Counts",
        f"sagin_groups: {counts['sagin_groups']}",
        f"sagin_groups_with_coordinates: {counts['sagin_groups_with_coordinates']}",
        f"sagin_groups_missing_coordinates: {counts['sagin_groups_missing_coordinates']}",
        f"sagin_invalid_coordinates: {counts['sagin_invalid_coordinates']}",
        f"xbd_images: {counts['xbd_images']}",
        f"xbd_images_with_coordinates: {counts['xbd_images_with_coordinates']}",
        f"xbd_images_by_split: {counts['xbd_images_by_split']}",
        f"spatial_hits: {counts['spatial_hits']}",
        f"event_hits: {counts['event_hits']}",
        f"date_disagreements: {counts['date_disagreements']}",
        f"pretrain_exclusions: {counts['pretrain_exclusions']}",
        f"downstream_test_exclusions: {counts['downstream_exclusions']}",
        f"missing_xbd_splits: {payload['xbd_index']['missing_splits']}",
        "",
        "Spatial hits",
    ]
    lines.extend(_format_hits(payload["spatial_hits"]) or ["  none"])
    lines.append("")
    lines.append("Event hits")
    lines.extend(_format_hits(payload["event_hits"]) or ["  none"])
    lines.append("")
    lines.append("Date disagreements (not excluded)")
    lines.extend(_format_hits(payload["date_disagreements"]) or ["  none"])
    lines.append("")
    lines.append("Train-split spatial hits (reported, not in the default pretraining exclusion)")
    lines.extend(_format_hits(payload["train_only_spatial_hits"]) or ["  none"])
    lines.append("")
    lines.append("Unmapped xBD disaster strings")
    lines.extend([f"  {name}" for name in payload["unmapped_xbd_disasters"]] or ["  none"])
    lines.append("Unmapped SAGIN event names")
    lines.extend([f"  {name}" for name in payload["unmapped_sagin_events"]] or ["  none"])
    lines.append("Ambiguous SAGIN event names")
    lines.extend([f"  {name}" for name in payload["ambiguous_sagin_events"]] or ["  none"])
    lines.append("")
    lines.append("Reviewed name collisions (not exclusions)")
    for item in payload["review_only"]:
        right = item.get("right") or "no xBD event"
        lines.append(f"  {item.get('left')} vs {right}: {item.get('reason', '')}")
    lines.append("")
    lines.append("Recommended exclusions")
    lines.append(
        "Primary: drop the SAGIN groups in exclude_pretrain_sagin_groups.txt from continued "
        "pretraining. Keep the official xBD test and holdout splits unchanged."
    )
    lines.append(f"  pretrain group ids ({counts['pretrain_exclusions']}): {', '.join(payload['pretrain_exclusion_ids']) or 'none'}")
    lines.append(
        "Alternative: drop the xBD ids in alternative_exclude_downstream_xbd_images.txt from "
        "evaluation if those places already entered pretraining. Say so when reporting numbers; "
        "that evaluation is no longer the full official test split."
    )
    lines.append(
        f"  downstream image ids ({counts['downstream_exclusions']}): {', '.join(payload['downstream_exclusion_ids']) or 'none'}"
    )
    return "\n".join(lines) + "\n"


def _compare(
    *,
    groups,
    images: list[XBDImageGeo],
    catalog: EventCatalog,
    buffer_m: float,
    window_days: int,
    pretrain_splits: set[str],
    downstream_splits: set[str],
) -> dict:
    image_events: dict[str, EventMatch] = {}
    unmapped_xbd = set()
    for image in images:
        match = match_event(image.disaster_raw, catalog)
        image_events[image.image_id] = match
        if match.status == "unmapped" and image.disaster_raw:
            unmapped_xbd.add(image.disaster_raw)
    images_by_event: dict[str, list[XBDImageGeo]] = {}
    for image in images:
        match = image_events[image.image_id]
        if match.event_id:
            images_by_event.setdefault(match.event_id, []).append(image)

    spatial_hits = []
    event_hits = []
    date_disagreements = []
    ambiguous = set()
    unmapped_sagin = set()
    invalid_coords = []
    pretrain_ids = set()
    downstream_ids = set()
    groups_with_coords = 0

    for group in groups:
        match = match_event(group.event_name, catalog)
        if match.status == "ambiguous":
            ambiguous.add(group.event_name or "")
        elif match.status == "unmapped" and group.event_name:
            unmapped_sagin.add(group.event_name)
        event_ok = False
        if match.event_id:
            catalog_event = catalog.by_id[match.event_id]
            sagin_span = parse_date_span(group.event_date)
            if sagin_span and catalog_event.start and catalog_event.end:
                if spans_overlap(sagin_span, (catalog_event.start, catalog_event.end), window_days):
                    event_ok = True
                else:
                    event_ok = False
                    date_disagreements.append(
                        {
                            "kind": "date_disagreement",
                            "sagin_group_id": group.group_id,
                            "event_id": match.event_id,
                            "sagin_event_date": group.event_date,
                            "xbd_image_id": None,
                            "xbd_split": None,
                            "distance_m": None,
                            "detail": "SAGIN date falls outside the catalog window for this event id",
                        }
                    )
            else:
                event_ok = True
        if event_ok and match.event_id:
            for image in images_by_event.get(match.event_id, []):
                event_hits.append(
                    {
                        "kind": "event",
                        "sagin_group_id": group.group_id,
                        "event_id": match.event_id,
                        "xbd_image_id": image.image_id,
                        "xbd_split": image.split,
                        "distance_m": None,
                        "detail": group.event_name or "",
                    }
                )
                if image.split in pretrain_splits:
                    pretrain_ids.add(group.group_id)
                if image.split in downstream_splits:
                    downstream_ids.add(image.image_id)

        if group.lat is None or group.lon is None:
            continue
        if not valid_lat_lon(group.lat, group.lon):
            invalid_coords.append(group.group_id)
            continue
        groups_with_coords += 1
        for image in images:
            if not image.has_coordinates:
                continue
            if not point_in_expanded_bbox(
                group.lat,
                group.lon,
                image.min_lat,
                image.max_lat,
                image.min_lon,
                image.max_lon,
                buffer_m,
            ):
                continue
            distances = haversine_m(group.lat, group.lon, image.centroids[:, 0], image.centroids[:, 1])
            nearest = int(distances.argmin())
            distance = float(distances[nearest])
            if distance > buffer_m:
                continue
            spatial_hits.append(
                {
                    "kind": "spatial",
                    "sagin_group_id": group.group_id,
                    "event_id": match.event_id,
                    "xbd_image_id": image.image_id,
                    "xbd_split": image.split,
                    "distance_m": round(distance, 3),
                    "nearest_building_uid": image.uids[nearest],
                    "detail": f"{distance:.1f} m to building {image.uids[nearest]}",
                }
            )
            if image.split in pretrain_splits:
                pretrain_ids.add(group.group_id)
            if image.split in downstream_splits:
                downstream_ids.add(image.image_id)

    train_only = []
    for hit in spatial_hits:
        group_id = hit["sagin_group_id"]
        group_hits = [item for item in spatial_hits if item["sagin_group_id"] == group_id]
        splits = {item["xbd_split"] for item in group_hits}
        if splits <= {"train"}:
            train_only.append(hit)

    by_split = {name: 0 for name in OFFICIAL_SPLIT_NAMES}
    with_coords = 0
    for image in images:
        by_split[image.split] = by_split.get(image.split, 0) + 1
        if image.has_coordinates:
            with_coords += 1
    return {
        "counts": {
            "sagin_groups": len(groups),
            "sagin_groups_with_coordinates": groups_with_coords,
            "sagin_groups_missing_coordinates": len(groups) - groups_with_coords - len(invalid_coords),
            "sagin_invalid_coordinates": len(invalid_coords),
            "xbd_images": len(images),
            "xbd_images_with_coordinates": with_coords,
            "xbd_images_by_split": by_split,
            "spatial_hits": len(spatial_hits),
            "event_hits": len(event_hits),
            "date_disagreements": len(date_disagreements),
            "pretrain_exclusions": len(pretrain_ids),
            "downstream_exclusions": len(downstream_ids),
        },
        "spatial_hits": spatial_hits,
        "event_hits": event_hits,
        "date_disagreements": date_disagreements,
        "train_only_spatial_hits": train_only,
        "pretrain_exclusion_ids": sorted(pretrain_ids),
        "downstream_exclusion_ids": sorted(downstream_ids),
        "invalid_coordinate_ids": invalid_coords,
        "unmapped_xbd_disasters": sorted(unmapped_xbd),
        "unmapped_sagin_events": sorted(unmapped_sagin),
        "ambiguous_sagin_events": sorted(ambiguous),
        "review_only": catalog.review_only,
    }


def _format_hits(hits: list[dict]) -> list[str]:
    lines = []
    for hit in hits:
        lines.append(
            "  "
            + " ".join(
                [
                    f"group={hit.get('sagin_group_id')}",
                    f"xbd={hit.get('xbd_image_id')}",
                    f"split={hit.get('xbd_split')}",
                    f"event={hit.get('event_id')}",
                    f"distance_m={hit.get('distance_m')}",
                ]
            )
        )
    return lines


def _check_split_names(names: set[str]) -> None:
    illegal = sorted(name for name in names if name not in OFFICIAL_SPLIT_NAMES)
    if illegal:
        raise ConfigError(f"Exclusion splits must be official xBD splits, got {illegal}")

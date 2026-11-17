"""Parse official xBD / xView2 JSON labels. No images are required."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_COORD = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_PAIR_NAME = re.compile(r"^(?P<prefix>.+)_(?P<when>pre|post)_disaster$", re.IGNORECASE)
_STEM_EVENT = re.compile(r"^(?P<event>.+?)_(?P<num>\d+)$")

DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
CLASS_TO_INDEX = {name: index for index, name in enumerate(DAMAGE_CLASSES)}
SKIP_SUBTYPES = {"un-classified", "unclassified", ""}
OFFICIAL_SPLIT_NAMES = ("train", "test", "holdout")


@dataclass
class BuildingLabel:
    uid: str
    subtype: str
    xy_box: tuple[float, float, float, float] | None
    lat: float | None
    lon: float | None


@dataclass
class XBDLabelFile:
    path: Path
    image_id: str
    when: str
    disaster_raw: str
    capture_date: str | None
    buildings: list[BuildingLabel] = field(default_factory=list)
    bad_wkt: int = 0


def parse_wkt_pairs(wkt: str | None) -> list[tuple[float, float]]:
    if not wkt:
        return []
    return [(float(first), float(second)) for first, second in _COORD.findall(wkt)]


def load_label_file(path: Path) -> XBDLabelFile:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    image_id, when = image_id_from_filename(path.name)
    metadata = payload.get("metadata") or {}
    img_name = str(metadata.get("img_name") or "")
    if img_name:
        parsed_id, parsed_when = image_id_from_filename(img_name)
        image_id = parsed_id or image_id
        when = parsed_when or when
    disaster = metadata.get("disaster") or event_from_image_id(image_id)
    features = payload.get("features") or {}
    xy_by_uid = _index_features(features.get("xy") or [], geographic=False)
    geo_by_uid = _index_features(features.get("lng_lat") or [], geographic=True)
    uids = list(dict.fromkeys([*xy_by_uid.keys(), *geo_by_uid.keys()]))
    buildings = []
    bad_wkt = 0
    for uid in uids:
        xy = xy_by_uid.get(uid)
        geo = geo_by_uid.get(uid)
        source = xy or geo
        if source is None:
            continue
        subtype = str(source["subtype"] or "").strip().lower()
        buildings.append(
            BuildingLabel(
                uid=uid,
                subtype=subtype,
                xy_box=None if xy is None else xy["box"],
                lat=None if geo is None else geo["lat"],
                lon=None if geo is None else geo["lon"],
            )
        )
        if xy is not None and xy["box"] is None:
            bad_wkt += 1
        if geo is not None and geo["lat"] is None:
            bad_wkt += 1
    return XBDLabelFile(
        path=path,
        image_id=image_id,
        when=when,
        disaster_raw=str(disaster or ""),
        capture_date=_blank(metadata.get("capture_date")),
        buildings=buildings,
        bad_wkt=bad_wkt,
    )


def image_id_from_filename(name: str) -> tuple[str, str]:
    stem = Path(name).stem
    if stem.endswith("_target"):
        stem = stem[: -len("_target")]
    match = _PAIR_NAME.match(stem)
    if not match:
        return stem, ""
    return match.group("prefix"), match.group("when").lower()


def event_from_image_id(image_id: str) -> str:
    match = _STEM_EVENT.match(image_id)
    if match:
        return match.group("event")
    return image_id


def _index_features(features: list, *, geographic: bool) -> dict[str, dict]:
    indexed = {}
    for ordinal, feature in enumerate(features):
        properties = feature.get("properties") or {}
        feature_type = str(properties.get("feature_type") or "").strip().lower()
        if feature_type and feature_type != "building":
            continue
        uid = str(properties.get("uid") or f"idx-{ordinal}")
        pairs = parse_wkt_pairs(feature.get("wkt"))
        if len(pairs) >= 2 and pairs[0] == pairs[-1]:
            pairs = pairs[:-1]
        record = {
            "subtype": properties.get("subtype"),
            "box": None,
            "lat": None,
            "lon": None,
        }
        if len(pairs) >= 3:
            xs = [pair[0] for pair in pairs]
            ys = [pair[1] for pair in pairs]
            if geographic:
                record["lon"] = float(sum(xs) / len(xs))
                record["lat"] = float(sum(ys) / len(ys))
            else:
                record["box"] = (min(xs), min(ys), max(xs), max(ys))
        indexed[uid] = record
    return indexed


def _blank(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def iter_label_paths(labels_dir: Path) -> list[Path]:
    if not labels_dir.is_dir():
        return []
    paths = []
    for path in sorted(labels_dir.iterdir()):
        if path.suffix.lower() != ".json":
            continue
        if path.name.startswith("."):
            continue
        paths.append(path)
    return paths

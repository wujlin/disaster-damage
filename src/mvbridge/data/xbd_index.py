"""Torch-free xBD geography index used by the overlap audit.

Only the split directories named in the config are read. A directory such as
``val/`` is ignored, so a custom split cannot enter the audit by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mvbridge.data.xbd_labels import (
    OFFICIAL_SPLIT_NAMES,
    image_id_from_filename,
    iter_label_paths,
    load_label_file,
)
from mvbridge.errors import ConfigError, DataError
from mvbridge.geo import valid_lat_lon


@dataclass
class XBDImageGeo:
    image_id: str
    split: str
    disaster_raw: str
    capture_date: str | None
    centroids: np.ndarray
    uids: list[str]
    min_lat: float | None = None
    max_lat: float | None = None
    min_lon: float | None = None
    max_lon: float | None = None

    @property
    def has_coordinates(self) -> bool:
        return self.centroids.size > 0


@dataclass
class XBDIndex:
    images: list[XBDImageGeo]
    missing_splits: list[str] = field(default_factory=list)
    bad_wkt: int = 0
    label_files: int = 0


def index_xbd(root: str | Path, split_dirs: dict, *, splits: tuple[str, ...] = OFFICIAL_SPLIT_NAMES) -> XBDIndex:
    root_path = Path(root)
    if not root_path.is_dir():
        raise DataError(f"xBD root does not exist: {root_path}")
    unknown = [name for name in split_dirs if name not in OFFICIAL_SPLIT_NAMES]
    if unknown:
        raise ConfigError(
            f"xBD split names {unknown} are not official. Use train, test, and holdout. "
            "Point split_dirs.holdout at the folder name the release used, such as holdout or hold."
        )
    images: list[XBDImageGeo] = []
    missing = []
    seen: dict[str, str] = {}
    bad_wkt = 0
    label_files = 0
    found_any = False
    for split in splits:
        if split not in split_dirs:
            missing.append(split)
            continue
        split_dir = root_path / str(split_dirs[split])
        labels_dir = split_dir / "labels"
        if not labels_dir.is_dir():
            missing.append(split)
            continue
        found_any = True
        by_image: dict[str, dict[str, Path]] = {}
        for path in iter_label_paths(labels_dir):
            image_id, when = image_id_from_filename(path.name)
            label_files += 1
            by_image.setdefault(image_id, {})[when or path.stem] = path
        for image_id, files in sorted(by_image.items()):
            if image_id in seen:
                raise DataError(
                    f"xBD image id {image_id} appears in both {seen[image_id]!r} and {split!r}. "
                    "Official splits are disjoint; this scaffold will not merge them."
                )
            seen[image_id] = split
            chosen = files.get("post") or files.get("pre") or next(iter(files.values()))
            label = load_label_file(chosen)
            bad_wkt += label.bad_wkt
            lats = []
            lons = []
            uids = []
            for building in label.buildings:
                if building.lat is None or building.lon is None:
                    continue
                if not valid_lat_lon(building.lat, building.lon):
                    continue
                lats.append(building.lat)
                lons.append(building.lon)
                uids.append(building.uid)
            centroids = np.column_stack([lats, lons]) if lats else np.zeros((0, 2), dtype=np.float64)
            images.append(
                XBDImageGeo(
                    image_id=image_id,
                    split=split,
                    disaster_raw=label.disaster_raw or "",
                    capture_date=label.capture_date,
                    centroids=centroids,
                    uids=uids,
                    min_lat=float(min(lats)) if lats else None,
                    max_lat=float(max(lats)) if lats else None,
                    min_lon=float(min(lons)) if lons else None,
                    max_lon=float(max(lons)) if lons else None,
                )
            )
    if not found_any:
        expected = ", ".join(f"{name} -> {root_path / str(split_dirs.get(name, name)) / 'labels'}" for name in splits)
        raise DataError(f"No xBD label directories were found. Expected one of: {expected}")
    return XBDIndex(images=images, missing_splits=missing, bad_wkt=bad_wkt, label_files=label_files)

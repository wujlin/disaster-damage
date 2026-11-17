"""Index SAGIN / SAGINGeo place groups from local metadata.

Two layouts are supported. Neither layout is downloaded by this module.

* ``manifest``: a CSV with one image per row.
* ``probglc``: the MultiDisaster tables shipped with ProbGLC
  (``RSI,SVI,UAV,VGI`` CSVs plus ``Other/allData_20250320.xlsx``).

The ProbGLC 80/20 and 70/30 files are geolocalization splits. ``which_tables: all``
joins them. They are not an xBD split and this loader does not invent one.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from mvbridge.data.records import (
    CANONICAL_VIEWS,
    PlaceGroup,
    ViewAsset,
    media_kind,
    role_for_asset,
)
from mvbridge.errors import ConfigError, DataError

PROBGLC_TABLES = {
    "train80": "RSI_SVI_UAV_VGI/80%train_RSI_SVI_UAV_VGI.csv",
    "test20": "RSI_SVI_UAV_VGI/20%test_RSI_SVI_UAV_VGI.csv",
    "train70": "RSI_SVI_UAV_VGI/70%train_RSI_SVI_UAV_VGI.csv",
    "test30": "RSI_SVI_UAV_VGI/30%test_RSI_SVI_UAV_VGI.csv",
}
EVENT_COLUMNS = ("event_name", "event", "disaster", "disaster_name")
DATE_COLUMNS = ("event_date", "date", "capture_date")
MANIFEST_COLUMNS = (
    "group_id",
    "event_name",
    "event_date",
    "lat",
    "lon",
    "view",
    "path",
    "is_simulated",
)


def load_sagin_groups(sagin_cfg: dict, *, check_files: bool = False) -> list[PlaceGroup]:
    layout = str(sagin_cfg.get("format", "manifest")).lower()
    if layout == "manifest":
        groups = _load_manifest(sagin_cfg)
    elif layout == "probglc":
        groups = _load_probglc(sagin_cfg)
    else:
        raise ConfigError("sagin.format must be 'manifest' or 'probglc'")
    if check_files:
        _stat_assets(groups)
    return groups


def _load_manifest(sagin_cfg: dict) -> list[PlaceGroup]:
    raw_path = sagin_cfg.get("manifest_csv")
    if not raw_path:
        raise ConfigError("sagin.manifest_csv is required when sagin.format is manifest")
    path = Path(raw_path)
    if not path.is_file():
        raise DataError(f"SAGIN manifest does not exist: {path}")
    root = Path(sagin_cfg["root"]) if sagin_cfg.get("root") else path.parent
    groups: dict[str, PlaceGroup] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [column for column in MANIFEST_COLUMNS if column not in header]
        if missing:
            raise DataError(
                f"SAGIN manifest {path} is missing columns {missing}. Expected {list(MANIFEST_COLUMNS)}."
            )
        for line_number, row in enumerate(reader, start=2):
            view = str(row["view"]).strip().lower()
            if view not in CANONICAL_VIEWS:
                raise DataError(f"{path}:{line_number} has view {view!r}; expected one of {CANONICAL_VIEWS}")
            group_id = str(row["group_id"]).strip()
            if not group_id:
                raise DataError(f"{path}:{line_number} has an empty group_id")
            group = groups.get(group_id)
            if group is None:
                group = PlaceGroup(
                    group_id=group_id,
                    lat=_optional_float(row.get("lat")),
                    lon=_optional_float(row.get("lon")),
                    event_name=_blank_to_none(row.get("event_name")),
                    event_date=_blank_to_none(row.get("event_date")),
                    source=str(path),
                )
                groups[group_id] = group
            is_simulated = _as_bool(row.get("is_simulated"))
            if view == "uav" and sagin_cfg.get("uav_default_is_simulated", True) and row.get("is_simulated") in (None, ""):
                is_simulated = True
            rel = _blank_to_none(row.get("path"))
            if rel is None:
                continue
            resolved = _resolve_path(root, rel)
            group.assets.append(
                ViewAsset(
                    view=view,
                    path=str(resolved),
                    role=role_for_asset(view, is_simulated),
                    is_simulated=is_simulated,
                    kind=media_kind(str(resolved)),
                )
            )
    return list(groups.values())


def _load_probglc(sagin_cfg: dict) -> list[PlaceGroup]:
    raw_root = sagin_cfg.get("probglc_root")
    if not raw_root:
        raise ConfigError("sagin.probglc_root is required when sagin.format is probglc")
    root = Path(raw_root)
    if not root.is_dir():
        raise DataError(f"ProbGLC / SAGIN root does not exist: {root}")
    which = sagin_cfg.get("which_tables", "all")
    if which == "all":
        selected = list(PROBGLC_TABLES)
    elif isinstance(which, list):
        selected = [str(item) for item in which]
    else:
        selected = [str(which)]
    unknown = [name for name in selected if name not in PROBGLC_TABLES]
    if unknown:
        raise ConfigError(f"Unknown ProbGLC tables {unknown}. Known tables: {sorted(PROBGLC_TABLES)}")
    frames = []
    missing_tables = []
    for name in selected:
        csv_path = root / PROBGLC_TABLES[name]
        if not csv_path.is_file():
            missing_tables.append(str(csv_path))
            continue
        frame = pd.read_csv(csv_path, header=None, names=["RSI", "SVI", "UAV", "VGI"])
        frame["__table"] = name
        frames.append(frame)
    if not frames:
        raise DataError(
            "No ProbGLC pairing CSVs were found. Expected one or more of: "
            + ", ".join(str(root / rel) for rel in PROBGLC_TABLES.values())
        )
    paired = pd.concat(frames, ignore_index=True)
    for column in ("RSI", "SVI", "UAV", "VGI"):
        paired[column] = paired[column].map(_clean_cell)
    paired = paired.drop_duplicates(subset=["RSI", "SVI", "UAV", "VGI"])
    metadata_rel = sagin_cfg.get("metadata_file", "Other/allData_20250320.xlsx")
    metadata_path = root / str(metadata_rel)
    if not metadata_path.is_file():
        raise DataError(
            f"SAGIN metadata file does not exist: {metadata_path}. "
            "ProbGLC uses Other/allData_20250320.xlsx with columns newnewID, Latitude, Longitude."
        )
    meta = _read_table(metadata_path)
    id_column = str(sagin_cfg.get("id_column", "newnewID"))
    lookup = _metadata_lookup(meta, id_column)
    uav_default_simulated = bool(sagin_cfg.get("uav_default_is_simulated", True))
    groups: dict[str, PlaceGroup] = {}
    for row in paired.itertuples(index=False):
        vgi = row.VGI
        if not vgi:
            continue
        group_id = Path(str(vgi).replace("\\", "/")).stem
        meta_row = _lookup_meta(lookup, group_id)
        group = groups.get(group_id)
        if group is None:
            lat = lon = None
            event_name = event_date = None
            if meta_row is not None:
                lat = _optional_float(meta_row.get("latitude"))
                lon = _optional_float(meta_row.get("longitude"))
                event_name = meta_row.get("event_name")
                event_date = meta_row.get("event_date")
            group = PlaceGroup(
                group_id=group_id,
                lat=lat,
                lon=lon,
                event_name=event_name,
                event_date=event_date,
                source=str(root),
            )
            groups[group_id] = group
        for view, raw_value in (("rsi", row.RSI), ("svi", row.SVI), ("uav", row.UAV), ("vgi", row.VGI)):
            value = _clean_cell(raw_value)
            if value is None:
                continue
            resolved = _resolve_path(root, value)
            is_simulated = view == "uav" and uav_default_simulated
            group.assets.append(
                ViewAsset(
                    view=view,
                    path=str(resolved),
                    role=role_for_asset(view, is_simulated),
                    is_simulated=is_simulated,
                    kind=media_kind(str(resolved)),
                )
            )
    if missing_tables and which != "all":
        raise DataError(f"Missing ProbGLC pairing CSV: {missing_tables[0]}")
    return list(groups.values())


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _metadata_lookup(frame: pd.DataFrame, id_column: str) -> dict[str, dict]:
    columns = {column.lower(): column for column in frame.columns}
    if id_column.lower() not in columns:
        raise DataError(
            f"SAGIN metadata is missing id column {id_column!r}. Columns found: {list(frame.columns)}"
        )
    if "latitude" not in columns or "longitude" not in columns:
        raise DataError(
            "SAGIN metadata is missing Latitude/Longitude columns. "
            f"Columns found: {list(frame.columns)}"
        )
    event_col = next((columns[name] for name in EVENT_COLUMNS if name in columns), None)
    date_col = next((columns[name] for name in DATE_COLUMNS if name in columns), None)
    id_col = columns[id_column.lower()]
    lookup = {}
    for record in frame.to_dict(orient="records"):
        key = str(record[id_col]).strip()
        if key.endswith(".0") and key[:-2].isdigit():
            key = key[:-2]
        payload = {
            "latitude": record[columns["latitude"]],
            "longitude": record[columns["longitude"]],
            "event_name": _blank_to_none(record[event_col]) if event_col else None,
            "event_date": _blank_to_none(record[date_col]) if date_col else None,
        }
        lookup[key] = payload
        if key.isdigit():
            lookup[str(int(key))] = payload
    return lookup


def _lookup_meta(lookup: dict[str, dict], group_id: str) -> dict | None:
    if group_id in lookup:
        return lookup[group_id]
    if group_id.isdigit() and str(int(group_id)) in lookup:
        return lookup[str(int(group_id))]
    return None


def _stat_assets(groups: list[PlaceGroup]) -> None:
    for group in groups:
        for asset in group.assets:
            asset.exists = Path(asset.path).is_file()


def _resolve_path(root: Path, value: str) -> Path:
    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def _clean_cell(value) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _blank_to_none(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _optional_float(value) -> float | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _as_bool(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "simulated"}

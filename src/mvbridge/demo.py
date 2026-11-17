"""Tiny local fixtures for --demo. Nothing here is a real dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


def prepare_audit_demo(root: Path, catalog: str, output_dir: Path, buffer_m: float = 1000) -> dict:
    """Build a fake SAGIN manifest and fake xBD labels, then return an audit config."""
    sagin_dir = root / "sagin"
    xbd_dir = root / "xbd"
    sagin_dir.mkdir(parents=True, exist_ok=True)
    _write_xbd_pair(
        xbd_dir,
        "test",
        "hurricane-harvey_00000001",
        "hurricane-harvey",
        lat=10.0,
        lon=10.0,
        uid="harvey-b",
    )
    _write_xbd_pair(
        xbd_dir,
        "test",
        "spatial-target_00000007",
        "spatial-target",
        lat=29.76,
        lon=-95.37,
        uid="near-b",
    )
    _write_xbd_pair(
        xbd_dir,
        "train",
        "train-only_00000003",
        "train-only",
        lat=40.0,
        lon=-74.0,
        uid="train-b",
    )
    _write_xbd_pair(
        xbd_dir,
        "train",
        "midwest-flooding_00000009",
        "midwest-flooding",
        lat=45.0,
        lon=-90.0,
        uid="mw-b",
    )
    manifest = sagin_dir / "manifest.csv"
    rows = [
        ("near-test", "", "", 29.761, -95.37, "rsi", "near.jpg", "false"),
        ("midwest-2015", "Midwest U.S. Flooding (2015)", "2015", 0.0, 0.0, "vgi", "mw.jpg", "false"),
        ("harvey-group", "hurricane-harvey", "2017-08-30", 55.0, 10.0, "svi", "harvey.jpg", "false"),
        ("near-train", "", "", 40.001, -74.0, "vgi", "train.jpg", "false"),
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "event_name", "event_date", "lat", "lon", "view", "path", "is_simulated"])
        writer.writerows(rows)
    return {
        "buffer_m": buffer_m,
        "date_window_days": 14,
        "catalog": catalog,
        "sagin": {"format": "manifest", "manifest_csv": str(manifest), "uav_default_is_simulated": True},
        "xbd": {
            "root": str(xbd_dir),
            "split_dirs": {"train": "train", "test": "test", "holdout": "holdout"},
        },
        "pretrain_exclusion_splits": ["test", "holdout"],
        "downstream_exclusion_splits": ["test", "holdout"],
        "output_dir": str(output_dir),
    }


def prepare_sagin_demo(root: Path) -> dict:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name, color in (("a_vgi.png", (200, 20, 20)), ("a_svi.png", (20, 200, 20)), ("a_rsi.png", (20, 20, 200)), ("b_vgi.png", (200, 200, 20)), ("b_svi.png", (20, 200, 200))):
        Image.new("RGB", (8, 8), color).save(image_dir / name)
    (image_dir / "a_uav.mp4").write_bytes(b"not a real video")
    manifest = root / "manifest.csv"
    rows = [
        ("place-a", "Maui fire 2023", "2023", 20.8, -156.3, "vgi", "images/a_vgi.png", "false"),
        ("place-a", "Maui fire 2023", "2023", 20.8, -156.3, "svi", "images/a_svi.png", "false"),
        ("place-a", "Maui fire 2023", "2023", 20.8, -156.3, "rsi", "images/a_rsi.png", "false"),
        ("place-a", "Maui fire 2023", "2023", 20.8, -156.3, "uav", "images/a_uav.mp4", "true"),
        ("place-b", "Maui fire 2023", "2023", 20.9, -156.4, "vgi", "images/b_vgi.png", "false"),
        ("place-b", "Maui fire 2023", "2023", 20.9, -156.4, "svi", "images/b_svi.png", "false"),
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "event_name", "event_date", "lat", "lon", "view", "path", "is_simulated"])
        writer.writerows(rows)
    return {
        "format": "manifest",
        "manifest_csv": str(manifest),
        "root": str(root),
        "image_size": 32,
        "min_views": 2,
        "missing_view_policy": "allow_partial",
        "check_files": True,
        "uav_default_is_simulated": True,
        "seed": 0,
    }


def summarize_sagin_dataset(dataset) -> str:
    bridge = 0
    damage = 0
    lines = [
        f"sagin_groups_in_metadata: {dataset.source_group_count}",
        f"sagin_groups_used: {len(dataset)}",
        f"dropped_missing_views: {dataset.dropped_missing}",
        f"dropped_excluded: {dataset.dropped_excluded}",
    ]
    for index in range(len(dataset)):
        sample = dataset[index]
        lines.append(
            f"group={sample['group_id']} identity={sample['identity_views']} "
            f"damage_evidence={sample['damage_evidence_views']} "
            f"bridge_videos={len(sample['bridge_videos'])} present={sample['view_present']}"
        )
        damage += len(sample["damage_evidence_views"])
        bridge += len(sample["bridge_videos"])
        if "uav" in sample["damage_evidence_views"]:
            raise RuntimeError("UAV viewpoint bridge was listed as damage evidence")
    lines.append(f"damage_evidence_views: {damage}")
    lines.append(f"bridge_videos: {bridge}")
    lines.append("simulated UAV stays on the viewpoint bridge and is absent from damage evidence")
    return "\n".join(lines) + "\n"


def _write_xbd_pair(root: Path, split: str, image_id: str, disaster: str, lat: float, lon: float, uid: str) -> None:
    labels = root / split / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    delta = 0.0002
    ring = (
        f"{lon - delta} {lat - delta}, {lon + delta} {lat - delta}, "
        f"{lon + delta} {lat + delta}, {lon - delta} {lat + delta}, {lon - delta} {lat - delta}"
    )
    for when in ("pre", "post"):
        payload = {
            "features": {
                "lng_lat": [
                    {
                        "properties": {"feature_type": "building", "subtype": "no-damage", "uid": uid},
                        "wkt": f"POLYGON (({ring}))",
                    }
                ],
                "xy": [],
            },
            "metadata": {
                "disaster": disaster,
                "capture_date": "2017-08-31T00:00:00.000Z" if "harvey" in disaster else "2019-03-01T00:00:00.000Z",
                "img_name": f"{image_id}_{when}_disaster.png",
            },
        }
        (labels / f"{image_id}_{when}_disaster.json").write_text(json.dumps(payload), encoding="utf-8")

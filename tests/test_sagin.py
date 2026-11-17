"""SAGIN place-group loader, including the fake-metadata smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image

from mvbridge.cli import main
from mvbridge.data.records import VIEWPOINT_BRIDGE
from mvbridge.data.sagin import SAGINMultiViewDataset, assert_damage_evidence, collate_sagin
from mvbridge.data.sagin_index import load_sagin_groups
from mvbridge.errors import DataError

ROOT = Path(__file__).resolve().parents[1]


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _manifest(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "group_id,event_name,event_date,lat,lon,view,path,is_simulated\n"
    body = "\n".join(",".join(str(item) for item in row) for row in rows)
    path.write_text(header + body + "\n", encoding="utf-8")


def _config(root: Path, manifest: Path, **overrides) -> dict:
    config = {
        "format": "manifest",
        "manifest_csv": str(manifest),
        "root": str(root),
        "image_size": 16,
        "min_views": 2,
        "missing_view_policy": "allow_partial",
        "check_files": True,
        "uav_default_is_simulated": True,
        "seed": 0,
    }
    config.update(overrides)
    return config


def test_groups_views_and_keeps_simulated_uav_off_damage(tmp_path):
    _write_png(tmp_path / "a_vgi.png", (255, 0, 0))
    _write_png(tmp_path / "a_svi.png", (0, 255, 0))
    _write_png(tmp_path / "a_rsi.png", (0, 0, 255))
    _write_png(tmp_path / "a_uav.png", (255, 255, 0))
    _write_png(tmp_path / "b_vgi.png", (10, 10, 10))
    manifest = tmp_path / "manifest.csv"
    _manifest(
        manifest,
        [
            ("place-a", "Maui fire 2023", "2023-08-01", 20.8, -156.3, "vgi", "a_vgi.png", "false"),
            ("place-a", "Maui fire 2023", "2023-08-01", 20.8, -156.3, "svi", "a_svi.png", "false"),
            ("place-a", "Maui fire 2023", "2023-08-01", 20.8, -156.3, "rsi", "a_rsi.png", "false"),
            ("place-a", "Maui fire 2023", "2023-08-01", 20.8, -156.3, "uav", "a_uav.png", "true"),
            ("place-b", "Maui fire 2023", "2023-08-01", 20.9, -156.4, "vgi", "b_vgi.png", "false"),
            ("place-b", "Maui fire 2023", "2023-08-01", 20.9, -156.4, "svi", "missing_svi.png", "false"),
        ],
    )
    dataset = SAGINMultiViewDataset(_config(tmp_path, manifest))
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["group_id"] == "place-a"
    assert set(sample["identity_views"]) == {"vgi", "svi", "rsi", "uav"}
    assert "uav" not in sample["damage_evidence_views"]
    assert sample["roles"]["uav"] == VIEWPOINT_BRIDGE
    assert sample["images"]["vgi"].shape == (3, 16, 16)
    assert_damage_evidence(sample)
    poisoned = dict(sample)
    poisoned["damage_evidence_views"] = ["uav"]
    poisoned["roles"] = {"uav": VIEWPOINT_BRIDGE}
    with pytest.raises(DataError, match="viewpoint-bridge"):
        assert_damage_evidence(poisoned)
    batch = collate_sagin([sample, sample])
    assert batch["vgi"].shape == (2, 3, 16, 16)
    assert bool(batch["vgi_present"].all())


def test_missing_view_policies(tmp_path):
    for name, color in (
        ("full_vgi.png", (1, 0, 0)),
        ("full_svi.png", (0, 1, 0)),
        ("full_uav.png", (0, 0, 1)),
        ("full_rsi.png", (1, 1, 0)),
        ("only_vgi.png", (1, 2, 3)),
        ("only_svi.png", (3, 2, 1)),
    ):
        _write_png(tmp_path / name, color)
    manifest = tmp_path / "manifest.csv"
    _manifest(
        manifest,
        [
            ("full", "", "", 1, 2, "vgi", "full_vgi.png", "false"),
            ("full", "", "", 1, 2, "svi", "full_svi.png", "false"),
            ("full", "", "", 1, 2, "uav", "full_uav.png", "false"),
            ("full", "", "", 1, 2, "rsi", "full_rsi.png", "false"),
            ("partial", "", "", 1, 2, "vgi", "only_vgi.png", "false"),
            ("partial", "", "", 1, 2, "svi", "only_svi.png", "false"),
        ],
    )
    kept = SAGINMultiViewDataset(_config(tmp_path, manifest, missing_view_policy="allow_partial", min_views=2))
    assert {sample["group_id"] for sample in (kept[index] for index in range(len(kept)))} == {"full", "partial"}
    partial = kept[1] if kept[0]["group_id"] == "full" else kept[0]
    assert partial["view_present"]["uav"] is False
    skipped = SAGINMultiViewDataset(_config(tmp_path, manifest, missing_view_policy="skip_group"))
    assert len(skipped) == 1
    assert skipped[0]["group_id"] == "full"
    # Explicit is_simulated=false keeps a real UAV frame available as damage evidence.
    assert skipped[0]["roles"]["uav"] != VIEWPOINT_BRIDGE
    assert "uav" in skipped[0]["damage_evidence_views"]
    with pytest.raises(DataError, match="partial"):
        SAGINMultiViewDataset(_config(tmp_path, manifest, missing_view_policy="error"))


def test_exclusion_file_drops_group(tmp_path):
    _write_png(tmp_path / "vgi.png", (9, 9, 9))
    _write_png(tmp_path / "svi.png", (8, 8, 8))
    manifest = tmp_path / "manifest.csv"
    _manifest(
        manifest,
        [
            ("drop-me", "", "", 1, 1, "vgi", "vgi.png", "false"),
            ("drop-me", "", "", 1, 1, "svi", "svi.png", "false"),
        ],
    )
    exclusion = tmp_path / "exclude.txt"
    exclusion.write_text("# comment\ndrop-me\n", encoding="utf-8")
    dataset = SAGINMultiViewDataset(_config(tmp_path, manifest, exclude_ids_file=str(exclusion)))
    assert len(dataset) == 0
    assert dataset.dropped_excluded == 1


def test_probglc_layout_groups_by_vgi_id(tmp_path):
    pair_dir = tmp_path / "RSI_SVI_UAV_VGI"
    pair_dir.mkdir()
    (pair_dir / "80%train_RSI_SVI_UAV_VGI.csv").write_text(
        "RSI\\img\\rsi.png,SVI\\img\\svi.png,UAV\\img\\uav.png,VGI\\img\\1636.jpg\n"
        "RSI\\img\\rsi.png,SVI\\img\\svi.png,UAV\\img\\uav.png,VGI\\img\\1636.jpg\n",
        encoding="utf-8",
    )
    (pair_dir / "20%test_RSI_SVI_UAV_VGI.csv").write_text(
        "RSI/img/rsi2.png,SVI/img/svi2.png,,VGI/img/99.jpg\n",
        encoding="utf-8",
    )
    other = tmp_path / "Other"
    other.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["newnewID", "Latitude", "Longitude", "event_name"])
    sheet.append([1636, 29.7, -95.3, "hurricane-harvey"])
    sheet.append([99, 20.8, -156.3, "Maui fire 2023"])
    workbook.save(other / "allData_20250320.xlsx")
    groups = load_sagin_groups(
        {
            "format": "probglc",
            "probglc_root": str(tmp_path),
            "which_tables": "all",
            "uav_default_is_simulated": True,
        },
        check_files=False,
    )
    by_id = {group.group_id: group for group in groups}
    assert set(by_id) == {"1636", "99"}
    assert by_id["1636"].lat == pytest.approx(29.7)
    assert by_id["1636"].event_name == "hurricane-harvey"
    uav = by_id["1636"].assets_for("uav")
    assert len(uav) == 1
    assert uav[0].is_simulated is True
    assert uav[0].role == VIEWPOINT_BRIDGE
    assert by_id["99"].assets_for("uav") == []
    assert "img/rsi.png" in by_id["1636"].assets_for("rsi")[0].path.replace("\\", "/")


def test_sagin_smoke_cli_demo():
    code = main(["sagin-smoke", "--config", str(ROOT / "configs" / "sagin_loader.yaml"), "--demo"])
    assert code == 0


def test_missing_manifest_names_the_path(tmp_path):
    missing = tmp_path / "no-such-manifest.csv"
    with pytest.raises(DataError, match="no-such-manifest.csv"):
        load_sagin_groups({"format": "manifest", "manifest_csv": str(missing)})

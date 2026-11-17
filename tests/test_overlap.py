"""Overlap audit, catalog identity, and geographic helpers."""

from __future__ import annotations

import json
from pathlib import Path

from mvbridge.audit.overlap import run_audit
from mvbridge.catalog import load_catalog, match_event
from mvbridge.cli import main
from mvbridge.demo import prepare_audit_demo
from mvbridge.errors import ConfigError, DataError
from mvbridge.geo import haversine_m, parse_date_span, spans_overlap

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "event_catalog.yaml"


def test_haversine_one_degree_latitude():
    distance = float(haversine_m(0.0, 0.0, 1.0, 0.0))
    assert abs(distance - 111194.9) < 1.0


def test_date_span_year_and_window():
    year = parse_date_span("2015")
    day = parse_date_span("2015-06-01")
    assert year == (parse_date_span("2015-01-01")[0], parse_date_span("2015-12-31")[1])
    assert spans_overlap(year, day, window_days=0)
    later = parse_date_span("2019-03-01")
    assert not spans_overlap(year, later, window_days=14)


def test_catalog_keeps_midwest_years_apart():
    catalog = load_catalog(CATALOG)
    assert match_event("Midwest U.S. Flooding (2015)", catalog).event_id == "midwest-us-flooding-2015"
    assert match_event("midwest-flooding", catalog).event_id == "midwest-us-floods-2019"
    assert match_event("midwest flooding", catalog).status == "ambiguous"
    assert match_event("hurricane-harvey", catalog).event_id == "hurricane-harvey-2017"
    assert match_event("Hurricane Ian (2025)", catalog).status == "unmapped"
    assert match_event("hurricane ian", catalog).event_id == "hurricane-ian-2022"


def test_audit_spatial_event_and_year_collision(tmp_path):
    output = tmp_path / "out"
    config = prepare_audit_demo(tmp_path / "fixture", str(CATALOG), output, buffer_m=1000)
    # A custom val/ tree must not be read.
    val_labels = tmp_path / "fixture" / "xbd" / "val" / "labels"
    val_labels.mkdir(parents=True)
    (val_labels / "leaky_00000001_post_disaster.json").write_text(
        json.dumps(
            {
                "features": {"lng_lat": [], "xy": []},
                "metadata": {"disaster": "leaky-split", "img_name": "leaky_00000001_post_disaster.png"},
            }
        ),
        encoding="utf-8",
    )
    payload = run_audit(config)
    pretrain = set(payload["pretrain_exclusion_ids"])
    downstream = set(payload["downstream_exclusion_ids"])
    assert "near-test" in pretrain
    assert "harvey-group" in pretrain
    assert "midwest-2015" not in pretrain
    assert "near-train" not in pretrain
    assert "spatial-target_00000007" in downstream
    assert "hurricane-harvey_00000001" in downstream
    assert "train-only_00000003" not in downstream
    assert "leaky_00000001" not in {hit["xbd_image_id"] for hit in payload["spatial_hits"]}
    assert payload["counts"]["spatial_hits"] >= 2
    text = (output / "overlap_report.txt").read_text(encoding="utf-8")
    assert "midwest-2015" in text or "Midwest" in text
    assert "Primary:" in text
    assert "near-test" in (output / "exclude_pretrain_sagin_groups.txt").read_text(encoding="utf-8")
    assert "near-train" in (output / "train_split_spatial_hits.txt").read_text(encoding="utf-8")


def test_tighter_buffer_drops_the_spatial_hit(tmp_path):
    output = tmp_path / "out"
    config = prepare_audit_demo(tmp_path / "fixture", str(CATALOG), output, buffer_m=50)
    payload = run_audit(config)
    assert "near-test" not in payload["pretrain_exclusion_ids"]
    assert "harvey-group" in payload["pretrain_exclusion_ids"]


def test_date_disagreement_is_not_an_event_exclusion(tmp_path):
    output = tmp_path / "out"
    config = prepare_audit_demo(tmp_path / "fixture", str(CATALOG), output, buffer_m=1000)
    manifest = Path(config["sagin"]["manifest_csv"])
    text = manifest.read_text(encoding="utf-8").replace("2017-08-30", "2022-01-01")
    manifest.write_text(text, encoding="utf-8")
    payload = run_audit(config)
    assert "harvey-group" not in payload["pretrain_exclusion_ids"]
    assert any(hit["sagin_group_id"] == "harvey-group" for hit in payload["date_disagreements"])


def test_missing_roots_name_the_path(tmp_path):
    config = {
        "buffer_m": 1000,
        "date_window_days": 14,
        "catalog": str(CATALOG),
        "sagin": {"format": "manifest", "manifest_csv": str(tmp_path / "missing.csv")},
        "xbd": {"root": str(tmp_path / "missing-xbd"), "split_dirs": {"train": "train", "test": "test", "holdout": "holdout"}},
        "output_dir": str(tmp_path / "out"),
    }
    try:
        run_audit(config)
    except DataError as exc:
        assert "missing.csv" in str(exc)
    else:
        raise AssertionError("expected DataError")


def test_audit_cli_demo(tmp_path):
    code = main(
        [
            "audit",
            "--config",
            str(ROOT / "configs" / "overlap_audit.yaml"),
            "--demo",
            "--set",
            f"output_dir={tmp_path / 'demo-out'}",
        ]
    )
    assert code == 0
    assert (tmp_path / "demo-out" / "overlap_report.json").is_file()


def test_custom_exclusion_split_rejected(tmp_path):
    output = tmp_path / "out"
    config = prepare_audit_demo(tmp_path / "fixture", str(CATALOG), output)
    config["pretrain_exclusion_splits"] = ["val"]
    try:
        run_audit(config)
    except ConfigError as exc:
        assert "val" in str(exc)
    else:
        raise AssertionError("expected ConfigError")

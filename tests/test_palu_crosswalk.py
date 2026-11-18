"""Palu mapping ledger, pending/approved validator, and decision packet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mvbridge.cli import main
from mvbridge.crosswalk.checks import (
    PREDICTION_FIELDS,
    box_iou,
    map_source_class,
    run_ground_side_checks,
)
from mvbridge.crosswalk.fixtures import FIXTURE_NOTE, write_synthetic_ground_fixture
from mvbridge.crosswalk.ledger import (
    PACKAGED_LEDGER_PATH,
    PALU_PROTOCOL_NON_MAPPING_SENTENCES,
    PALU_PROTOCOL_OBTAINMENT_CITATION,
    STEER_WIND_FROZEN_RULES,
    clone_ledger,
    load_ledger,
)
from mvbridge.crosswalk.packet import PACKET_TITLE, render_decision_packet, write_decision_packet
from mvbridge.crosswalk.validate import validate_ledger
from mvbridge.errors import CrosswalkError

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "palu_crosswalk" / "ledger.yaml"
PROTOCOL = ROOT / "docs" / "eval_protocol.md"
PACKET = ROOT / "docs" / "palu_crosswalk_decision_packet.md"


def test_packaged_ledger_path_matches_repo():
    assert PACKAGED_LEDGER_PATH.resolve() == LEDGER.resolve()


def test_ledger_records_paulik_2019_on_wsa_but_mapping_unfrozen():
    ledger = load_ledger(LEDGER)
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert report.ok, [f"{item.code}: {item.message}" for item in report.issues]
    paulik = next(row for row in ledger["sources"] if row["id"] == "paulik_2019")
    assert paulik["obtainment_status"] == "obtained_on_wsa_awaiting_adjudication"
    assert paulik["full_text_obtained"] is True
    assert paulik["blocks_palu_freeze"] is True
    assert "doi:10.1007/s00024-019-02254-9" in paulik["citation"]
    assert ledger["palu_mapping"]["status"] == "pending_decision"
    assert ledger["palu_mapping"]["frozen"] is False
    assert ledger["meta"]["cross_hazard_transfer_established"] is False
    assert ledger["meta"]["compute_u_or_model_conflicts"] is False
    grade = ledger["official_grade_system"]
    assert grade["scale"] == "DL0_to_DL3"
    assert grade["frozen"] is False
    assert grade["applied_to_official_y"] is False
    assert grade["definitions"] == {
        "DL0": "No damage",
        "DL1": "Partial damage, repairable",
        "DL2": "Partial damage, unrepairable",
        "DL3": "Complete damage",
    }
    pdf = ledger["authorized_storage"]["official_paulik_pdf"]
    assert pdf["found"] is True
    assert pdf["pages"] == 17
    assert pdf["sha256"] == "a88cde36279b637f12c72e9fde34ff08c8ec3fcbc6bce601a62877236cf82c71"
    table = ledger["authorized_storage"]["publisher_survey_table"]
    assert table["sha256"] == "fc72889951d5e116e498d2ba841680e69e5297bbedbf2976cd19d097e7f7a04b"
    buildings = table["buildings_sheet"]
    assert buildings["non_empty_data_rows"] == 463
    assert buildings["rows_with_lat_lon_and_damage_level"] == 463
    assert buildings["unique_ids"] == 463
    assert buildings["unique_latitudes"] == 461
    assert buildings["unique_longitudes"] == 460
    assert buildings["unique_lat_lon_pairs"] == 463
    assert buildings["shared_exact_lat_lon_pairs"] == 0
    coords = ledger["inventory_differences"]["coordinate_uniqueness"]
    assert coords["unique_ids"] == 463
    assert coords["unique_latitudes"] == 461
    assert coords["unique_longitudes"] == 460
    assert coords["unique_lat_lon_pairs"] == 463
    assert coords["shared_exact_lat_lon_pairs"] == 0
    freqs = buildings["damage_level_frequencies"]
    assert freqs == {
        "No Damage": 21,
        "Partial Damage, Repairable": 250,
        "Partial Damage, Unrepairable": 94,
        "Complete Damage": 98,
    }
    assert sum(freqs.values()) == 463


def test_steer_wind_stays_frozen_and_matches_package():
    ledger = load_ledger(LEDGER)
    mapping = next(row for row in ledger["frozen_mappings"] if row["id"] == "steer_wind")
    assert mapping["status"] == "approved"
    assert mapping["frozen"] is True
    assert mapping["rules"] == STEER_WIND_FROZEN_RULES
    assert mapping["code_F_from"] == "component_fields_never_from_the_grade"
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "StEER wind is frozen as follows." in protocol
    assert "4 is not automatically destroyed." in protocol


def test_official_protocol_palu_section_is_not_a_freeze():
    text = PROTOCOL.read_text(encoding="utf-8")
    for sentence in PALU_PROTOCOL_NON_MAPPING_SENTENCES:
        assert sentence in text
    # Current protocol citation, not a frozen validator invariant.
    assert PALU_PROTOCOL_OBTAINMENT_CITATION in text
    assert "Palu mapping is frozen" not in text
    assert "Palu is frozen" not in text


def test_candidate_schemes_are_not_conclusions_or_protocol_freezes():
    ledger = load_ledger(LEDGER)
    schemes = {row["id"]: row for row in ledger["palu_mapping"]["candidate_schemes"]}
    assert set(schemes) == {
        "strict_ds1_unknown_exclude_ds2_ds3",
        "exploratory_ds2_major_ds3_destroyed",
        "ds0_to_no_listed_separately",
    }
    assert map_source_class("DS1", "strict_ds1_unknown_exclude_ds2_ds3") == "unknown_or_excluded_from_y"
    assert map_source_class("DS2", "strict_ds1_unknown_exclude_ds2_ds3") == "not_mapped_to_y"
    assert map_source_class("DS3", "strict_ds1_unknown_exclude_ds2_ds3") == "not_mapped_to_y"
    assert map_source_class("DS2", "exploratory_ds2_major_ds3_destroyed") == "major"
    assert map_source_class("DS3", "exploratory_ds2_major_ds3_destroyed") == "destroyed"
    assert map_source_class("DS0", "ds0_to_no_listed_separately") == "no"
    assert map_source_class("DS1", "ds0_to_no_listed_separately") == "not_in_this_scheme"
    for row in schemes.values():
        assert row["is_main_conclusion"] is False
        assert row["write_to_official_protocol_as_frozen"] is False
        assert row.get("frozen") is not True
        assert row.get("approved") is not True
        assert row.get("applied_as_mapping") is False
        assert row.get("status") == "not_applied"


def test_validator_rejects_approving_palu_before_jlin_adjudication():
    ledger = clone_ledger(load_ledger(LEDGER))
    ledger["palu_mapping"]["status"] = "approved"
    ledger["palu_mapping"]["frozen"] = True
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert not report.ok
    assert {item.code for item in report.issues} >= {"palu.status", "palu.frozen"}


def test_validator_rejects_clearing_the_jlin_adjudication_block():
    ledger = clone_ledger(load_ledger(LEDGER))
    paulik = next(row for row in ledger["sources"] if row["id"] == "paulik_2019")
    paulik["blocks_palu_freeze"] = False
    item = next(row for row in ledger["palu_mapping"]["blocking_items"] if row["id"] == "jlin_adjudication_of_dl_mapping")
    item["status"] = "cleared"
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert not report.ok
    assert {item.code for item in report.issues} >= {"paulik.blocks_freeze", "palu.adjudication_status"}


def test_validator_rejects_applying_ds_schemes_as_the_mapping():
    ledger = clone_ledger(load_ledger(LEDGER))
    schemes = ledger["palu_mapping"]["candidate_schemes"]
    schemes[0]["applied_as_mapping"] = True
    schemes[0]["status"] = "approved"
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert not report.ok
    assert {item.code for item in report.issues} >= {"schemes.applied"}


def test_validator_rejects_unfreezing_or_editing_steer_wind():
    ledger = clone_ledger(load_ledger(LEDGER))
    mapping = next(row for row in ledger["frozen_mappings"] if row["id"] == "steer_wind")
    mapping["frozen"] = False
    mapping["status"] = "pending_decision"
    mapping["rules"] = dict(mapping["rules"])
    mapping["rules"]["4"] = "destroyed"
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert not report.ok
    codes = {item.code for item in report.issues}
    assert "steer.status" in codes
    assert "steer.rules" in codes


def test_validator_rejects_candidate_scheme_as_main_conclusion():
    ledger = clone_ledger(load_ledger(LEDGER))
    ledger["palu_mapping"]["candidate_schemes"][0]["is_main_conclusion"] = True
    ledger["palu_mapping"]["candidate_schemes"][0]["write_to_official_protocol_as_frozen"] = True
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert not report.ok
    codes = {item.code for item in report.issues}
    assert "schemes.not_conclusion" in codes
    assert "schemes.not_protocol" in codes


def test_validator_rejects_transfer_claim_and_u_flag():
    ledger = clone_ledger(load_ledger(LEDGER))
    ledger["meta"]["cross_hazard_transfer_established"] = True
    ledger["meta"]["compute_u_or_model_conflicts"] = True
    report = validate_ledger(ledger, protocol_path=PROTOCOL)
    assert not report.ok
    codes = {item.code for item in report.issues}
    assert "meta.no_transfer" in codes
    assert "meta.no_u" in codes


def test_protocol_obtainment_citation_is_not_a_frozen_invariant(tmp_path):
    current = PROTOCOL.read_text(encoding="utf-8")
    assert PALU_PROTOCOL_OBTAINMENT_CITATION in current
    report = validate_ledger(load_ledger(LEDGER), protocol_path=PROTOCOL)
    assert report.ok, [f"{item.code}: {item.message}" for item in report.issues]

    corrected = current.replace(
        PALU_PROTOCOL_OBTAINMENT_CITATION,
        "Full text obtained on WSA; Palu mapping remains unfrozen.",
    )
    assert PALU_PROTOCOL_OBTAINMENT_CITATION not in corrected
    for sentence in PALU_PROTOCOL_NON_MAPPING_SENTENCES:
        assert sentence in corrected
    assert "Palu mapping is frozen" not in corrected
    assert "Palu is frozen" not in corrected
    path = tmp_path / "eval_protocol_obtainment_corrected.md"
    path.write_text(corrected, encoding="utf-8")
    report = validate_ledger(load_ledger(LEDGER), protocol_path=path)
    assert report.ok, [f"{item.code}: {item.message}" for item in report.issues]


def test_validator_detects_protocol_palu_sentence_removal(tmp_path):
    text = PROTOCOL.read_text(encoding="utf-8")
    altered = text.replace(
        "Palu is not one-to-one and is not frozen until Paulik et al. (2019) is in hand.",
        "Palu mapping is frozen.",
    )
    path = tmp_path / "eval_protocol.md"
    path.write_text(altered, encoding="utf-8")
    report = validate_ledger(load_ledger(LEDGER), protocol_path=path)
    assert not report.ok
    codes = {item.code for item in report.issues}
    assert "protocol.palu" in codes
    assert "protocol.not_frozen" in codes


def test_ground_checks_cover_dates_footprints_and_schemes(tmp_path):
    ledger = load_ledger(LEDGER)
    fixture_path = write_synthetic_ground_fixture(tmp_path / "synthetic_palu_ground.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert FIXTURE_NOTE in fixture["note"]
    assert "Paulik 2019 microdata" in fixture["note"]
    payload = run_ground_side_checks(fixture, ledger)
    assert payload["computes_u"] is False
    assert payload["computes_prediction_conflicts"] is False
    assert payload["cross_hazard_transfer_established"] is False
    assert payload["n_records"] == 8
    assert payload["footprint_match_counts"]["matched"] >= 5
    assert payload["footprint_match_counts"]["ambiguous"] == 1
    assert payload["footprint_match_counts"]["dropped_localization"] == 1
    issues = {item["footprint_id"]: item for item in payload["multipoint_inconsistencies"]}
    assert set(issues) == {"fp-a", "fp-b"}
    assert set(issues["fp-a"]["classes"]) == {"DS0", "DS1"}
    assert set(issues["fp-b"]["classes"]) == {"DS0", "DS1"}
    assert set(issues["fp-a"]["record_ids"]) == {"g-ds0", "g-ds1-on-a"}
    assert set(issues["fp-b"]["record_ids"]) == {"g-ds1", "g-poly-unique"}
    flags = payload["date_phase_flag_counts"]
    assert flags["missing_survey_date"] == 1
    assert flags["catalog_window_differs_from_cited_event_date"] >= 1
    assert flags["survey_before_catalog_start"] == 1
    inventories = {row["scheme_id"]: row for row in payload["candidate_scheme_inventories"]}
    strict = inventories["strict_ds1_unknown_exclude_ds2_ds3"]
    assert strict["is_main_conclusion"] is False
    assert strict["mapped_counts"]["unknown_or_excluded_from_y"] == 2
    assert strict["mapped_counts"]["not_mapped_to_y"] == 4
    assert strict["n_with_y_in_four_class_scale"] == 0
    exploratory = inventories["exploratory_ds2_major_ds3_destroyed"]
    assert exploratory["mapped_counts"]["major"] == 2
    assert exploratory["mapped_counts"]["destroyed"] == 2
    assert exploratory["n_with_y_in_four_class_scale"] == 4
    ds0 = inventories["ds0_to_no_listed_separately"]
    assert ds0["mapped_counts"]["no"] == 2
    assert ds0["n_with_y_in_four_class_scale"] == 2


def test_box_iou_and_unique_polygon_match():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)
    overlap = box_iou((200, 0, 230, 30), (210, 0, 240, 30))
    assert overlap > 0.3
    assert overlap < 0.9


def test_ground_checks_refuse_prediction_or_u_fields():
    ledger = load_ledger(LEDGER)
    fixture = {
        "note": "bad",
        "footprints": [{"id": "fp", "box": [0, 0, 10, 10]}],
        "ground_records": [{"id": "g1", "source_class": "DS1", "x": 1, "y": 1, "U": 1}],
    }
    with pytest.raises(CrosswalkError, match="must not compute U"):
        run_ground_side_checks(fixture, ledger)
    assert "u" in PREDICTION_FIELDS


def test_decision_packet_is_pending_and_matches_renderer(tmp_path):
    ledger = load_ledger(LEDGER)
    text = PACKET.read_text(encoding="utf-8")
    expected = render_decision_packet(ledger)
    assert text == expected
    assert PACKET_TITLE in text
    assert "PENDING Jlin decision" in text
    assert "Paulik 2019 files are on WSA" in text
    assert "must not be applied" in text
    assert "not a frozen mapping" in text.lower()
    assert "Wind events continue to use the already-frozen StEER wind package" in text
    assert "is_main_conclusion: `False`" in text
    assert "It does not treat transfer across hazards as shown or settled." in text
    assert "## 10. Inventory differences (pre-freeze findings, not frozen protocol facts)" in text
    assert "### PDF Table 2 vs Buildings sheet Commercial/Unknown" in text
    assert "### Catalog window vs Paulik event date (must be adjudicated before any freeze)" in text
    assert "### Palu City survey period is not the event date" in text
    assert "### Coordinate uniqueness" in text
    assert "unique IDs: 463" in text
    assert "unique latitudes: 461" in text
    assert "unique longitudes: 460" in text
    assert "unique (lat, lon) pairs: 463" in text
    assert "must be adjudicated before any freeze" in text
    assert "must not be conflated with the event date" in text
    assert "cross-hazard transfer is established" not in text.lower()
    assert "Palu mapping is frozen" not in text
    lowered = text.lower()
    assert "palu is frozen" not in lowered
    dest = write_decision_packet(ledger, tmp_path / "packet.md")
    assert dest.read_text(encoding="utf-8") == text


def test_authorized_storage_records_verified_paulik_and_msbfp2_inventory():
    ledger = load_ledger(LEDGER)
    tables = ledger["authorized_storage"]["palu_ground_tables"]
    assert tables["found"] is True
    assert tables["path"].endswith("24_2019_2254_MOESM1_ESM.xlsx")
    assert tables["sha256"] == "fc72889951d5e116e498d2ba841680e69e5297bbedbf2976cd19d097e7f7a04b"
    xbd = ledger["authorized_storage"]["xbd_palu_tsunami_tiles"]
    assert xbd["origin"] == "official_xbd_xview2_satellite_labels"
    assert "never Y" in xbd["notes"]
    msbfp2 = ledger["authorized_storage"]["msbfp2"]
    assert msbfp2["listed_in_catalog_index"] is False
    assert msbfp2["path"].endswith("ms_buildings_mx_psj.geojson")
    assert msbfp2["sha256"] == "b2d2da34c1d1e8d10ebb59d4a6c9de54e69399ce6f58f646d5c6bd167a4dd2b8"
    assert msbfp2["polygon_count"] == 6035
    assert msbfp2["n_features"] == 6035
    assert msbfp2["not_a_palu_mapping_source"] is True
    assert "michael_florida_ms_legacy_v2_steer_bbox.gpkg" in msbfp2["substitute_not_used"]


def test_cli_validate_and_synthetic_fixture(tmp_path):
    fixture = tmp_path / "fixture.json"
    packet_out = tmp_path / "packet.md"
    checks_out = tmp_path / "checks.json"
    code = main(
        [
            "palu-crosswalk",
            "--ledger",
            str(LEDGER),
            "--protocol",
            str(PROTOCOL),
            "--write-synthetic-fixture",
            str(fixture),
            "--packet-out",
            str(packet_out),
            "--checks-out",
            str(checks_out),
        ]
    )
    assert code == 0
    payload = json.loads(checks_out.read_text(encoding="utf-8"))
    assert payload["computes_u"] is False
    assert packet_out.read_text(encoding="utf-8").startswith(f"# {PACKET_TITLE}")
    assert "Synthetic fixture" in fixture.read_text(encoding="utf-8")


def test_cli_refuses_a_ground_table_with_predictions(tmp_path):
    path = tmp_path / "preds.json"
    path.write_text(
        json.dumps(
            {
                "footprints": [{"id": "fp", "box": [0, 0, 10, 10]}],
                "ground_records": [{"id": "g", "source_class": "DS1", "x": 1, "y": 1, "y_hat": "no"}],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "palu-crosswalk",
            "--ledger",
            str(LEDGER),
            "--protocol",
            str(PROTOCOL),
            "--ground-table",
            str(path),
        ]
    )
    assert code == 2

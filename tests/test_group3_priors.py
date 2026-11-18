"""Group-3 neighborhood priors. Fixtures are synthetic; no official scores."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from mvbridge.errors import ConfigError
from mvbridge.geo import haversine_m
from mvbridge.priors.coordinates import MetricFrame
from mvbridge.priors.coverage import CoverageMap, annulus_area_m2, disk_area_m2
from mvbridge.priors.features import compute_group3_features
from mvbridge.priors.fixtures import (
    SYNTHETIC_EVENT,
    build_default_fixture,
    future_acquired_elevation_grid,
    isolated_collapse_records,
    late_published_elevation_grid,
    no_alert_records,
    offset_lat_lon,
)
from mvbridge.priors.layers import ElevationGrid, WaterGeometries, layer_usable
from mvbridge.priors.neighborhood import NeighborhoodIndex
from mvbridge.priors.provenance import LayerLedger
from mvbridge.priors.schema import (
    GRADE_DESTROYED,
    GRADE_NO,
    OFFICIAL_EXPORT_PLACEHOLDER,
    BuildingRecord,
)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "group3_layers.yaml"


def _by_uid(rows):
    return {row.building_uid: row for row in rows}


def _post_records(fixture):
    return [row for row in fixture.records if row.event_id == SYNTHETIC_EVENT]


def test_metric_frame_matches_haversine_at_neighborhood_scales():
    frame = MetricFrame(29.95, -85.43)
    for east_m, north_m in ((30.0, 0.0), (0.0, 30.0), (150.0, 200.0), (300.0, 0.0)):
        lat, lon = frame.from_xy(east_m, north_m)
        lat = float(np.asarray(lat))
        lon = float(np.asarray(lon))
        x, y = frame.to_xy(lat, lon)
        assert float(np.asarray(x)) == pytest.approx(east_m, abs=1e-6)
        assert float(np.asarray(y)) == pytest.approx(north_m, abs=1e-6)
        dist = frame.distance_m(frame.lat0, frame.lon0, lat, lon)
        haver = float(haversine_m(frame.lat0, frame.lon0, lat, lon))
        assert float(np.asarray(dist)) == pytest.approx(haver, rel=1e-4, abs=0.05)
        assert haver == pytest.approx(math.hypot(east_m, north_m), rel=1e-3, abs=0.2)


def test_neighborhood_index_excludes_focal_id_even_when_coincident():
    index = NeighborhoodIndex(
        x=np.array([0.0, 0.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0]),
        ids=np.array(["focal", "other", "far"], dtype=object),
    )
    ids, dist = index.query_disk(0.0, 0.0, 1.0, exclude_ids={"focal"})
    assert set(ids) == {"other"}
    assert dist[0] == pytest.approx(0.0)
    nearest_id, nearest_d = index.nearest(0.0, 0.0, exclude_ids={"focal"})
    assert nearest_id == "other"
    assert nearest_d == pytest.approx(0.0)


def test_leave_self_out_isolated_collapse_is_not_its_own_neighbor():
    records = isolated_collapse_records()
    features = _by_uid(compute_group3_features(records))
    row = features["SOLO"]
    assert row.alert_count_30m == 0
    assert row.alert_count_100_300m == 0
    assert row.nearest_collapse_found is False
    assert row.nearest_collapse_distance_m is None
    assert row.alert_features_status == "computed"


def test_leave_self_out_and_ground_label_are_not_read_as_alerts():
    fixture = build_default_fixture()
    features = _by_uid(
        compute_group3_features(
            _post_records(fixture),
            elevation=fixture.elevation,
            water=fixture.water,
            coverage=fixture.coverage,
            frame=fixture.frame,
        )
    )
    # A is satellite intact and survey-destroyed. Self and the survey label
    # must not create an alert or a collapse at distance 0.
    assert fixture.records[0].ground_grade == GRADE_DESTROYED
    assert fixture.records[0].satellite_grade == GRADE_NO
    a = features["A"]
    assert a.alert_count_30m == 3  # B, F, G. Not A, not survey-only E, not H.
    assert a.alert_count_100_300m == 1  # C only. H is in the 30-100 m gap.
    assert a.nearest_collapse_found is True
    assert a.nearest_collapse_distance_m == pytest.approx(20.0, abs=0.05)
    # B is satellite-destroyed and survey-intact. A must not appear as a
    # collapse just because the ground label says destroyed.
    b = features["B"]
    assert b.nearest_collapse_found is True
    assert b.nearest_collapse_distance_m == pytest.approx(70.0, abs=0.1)
    assert b.alert_count_30m == 2  # F and G, not A and not self.
    with pytest.raises(ConfigError, match="ground_grade"):
        compute_group3_features(_post_records(fixture), alert_source_name="ground_grade")


def test_survey_only_building_is_not_a_satellite_neighbor():
    fixture = build_default_fixture()
    records = _post_records(fixture)
    e = next(row for row in records if row.building_uid == "E")
    assert e.in_survey is True and e.in_imagery is False
    assert e.satellite_grade == GRADE_DESTROYED
    features = _by_uid(
        compute_group3_features(
            records,
            coverage=fixture.coverage,
            frame=fixture.frame,
        )
    )
    # E is within 30 m of A but has no imagery, so it must not enter A's density.
    assert features["A"].alert_count_30m == 3
    # E can still receive features as a focal building, from imagery neighbors.
    assert features["E"].focal_in_survey is True
    assert features["E"].focal_in_imagery is False
    assert features["E"].alert_count_30m == 2  # B and G
    assert features["E"].nearest_collapse_found is True
    assert features["F"].focal_in_survey is False
    assert features["F"].focal_in_imagery is True


def test_tile_edge_flags_and_observed_area_are_not_silent():
    fixture = build_default_fixture()
    features = _by_uid(
        compute_group3_features(
            _post_records(fixture),
            coverage=fixture.coverage,
            frame=fixture.frame,
        )
    )
    a = features["A"]
    k = features["K"]
    d = features["D"]
    full_30 = disk_area_m2(30.0)
    full_ring = annulus_area_m2(100.0, 300.0)
    # A is in the interior of the 30 m scale, but its 300 m annulus leaves the tile.
    assert a.window_30m_edge is False
    assert a.window_100_300m_edge is True
    assert a.window_30m_observed_area_m2 == pytest.approx(full_30, rel=0.02)
    assert a.window_100_300m_observed_area_m2 < full_ring * 0.8
    # K sits 2 m south of the tile's north edge, so even the 30 m disk is clipped.
    assert k.window_30m_edge is True
    assert k.window_100_300m_edge is True
    assert k.window_30m_observed_area_m2 < full_30 * 0.7
    assert k.alert_density_30m_per_km2 == pytest.approx(
        k.alert_count_30m / (k.window_30m_observed_area_m2 / 1_000_000.0)
    )
    # D lives on a separate tile and is more than 300 m from the main cluster.
    assert d.window_30m_edge is False
    assert d.alert_count_30m == 0
    assert d.alert_count_100_300m == 0


def test_no_alert_case_has_zero_density_and_no_collapse():
    features = _by_uid(compute_group3_features(no_alert_records()))
    for uid in ("N1", "N2"):
        row = features[uid]
        assert row.alert_count_30m == 0
        assert row.alert_count_100_300m == 0
        assert row.alert_density_30m_per_km2 == 0
        assert row.nearest_collapse_found is False
        assert row.nearest_collapse_distance_m is None


def test_pre_event_alerts_do_not_enter_post_event_neighborhoods():
    fixture = build_default_fixture()
    pre = next(row for row in fixture.records if row.building_uid == "PRE")
    assert pre.phase == "pre"
    assert pre.satellite_grade == GRADE_DESTROYED
    features = _by_uid(
        compute_group3_features(
            list(fixture.records),
            coverage=fixture.coverage,
            frame=fixture.frame,
            phase="post",
        )
    )
    assert "PRE" not in features
    # Without PRE, A's nearest collapse is still B at 20 m, not a doubled count.
    assert features["A"].alert_count_30m == 3
    assert features["A"].nearest_collapse_distance_m == pytest.approx(20.0, abs=0.05)


def test_layer_usable_ignores_publication_date_and_uses_acquisition_window():
    as_of = date(2018, 10, 12)
    assert layer_usable(
        as_of,
        acquired_from=date(2011, 1, 1),
        acquired_until=date(2015, 12, 31),
        available_from=date(2021, 4, 22),
        product_available_from=date(2021, 4, 22),
    ) is True
    assert layer_usable(
        as_of,
        acquired_from=date(2020, 1, 1),
        acquired_until=date(2022, 12, 31),
        available_from=date(2015, 1, 1),
    ) is False


def test_late_published_dem_remains_usable_when_acquired_before_as_of():
    fixture = build_default_fixture()
    late = late_published_elevation_grid(fixture)
    assert late.available_from == date(2025, 1, 1)
    assert layer_usable(
        date(2018, 10, 12),
        acquired_from=late.acquired_from,
        acquired_until=late.acquired_until,
        available_from=late.available_from,
    ) is True
    features = _by_uid(
        compute_group3_features(
            _post_records(fixture),
            elevation=late,
            water=fixture.water,
            coverage=fixture.coverage,
            frame=fixture.frame,
            as_of=date(2018, 10, 12),
        )
    )
    a = features["A"]
    assert a.elevation_missing is False
    assert a.elevation_m is not None


def test_future_acquired_dem_is_unusable_as_of_earlier_event():
    fixture = build_default_fixture()
    future = future_acquired_elevation_grid(fixture)
    assert layer_usable(date(2018, 10, 12), acquired_from=future.acquired_from) is False
    features = _by_uid(
        compute_group3_features(
            _post_records(fixture),
            elevation=future,
            water=fixture.water,
            coverage=fixture.coverage,
            frame=fixture.frame,
            as_of=date(2018, 10, 12),
        )
    )
    a = features["A"]
    assert a.elevation_missing is True
    assert a.elevation_m is None
    assert any(note.startswith("elevation_unavailable_as_of") for note in a.layer_notes)
    assert a.water_missing is False
    assert a.distance_to_water_m == pytest.approx(80.0, abs=0.5)


def test_static_dem_and_coast_sample_at_the_origin():
    fixture = build_default_fixture()
    features = _by_uid(
        compute_group3_features(
            _post_records(fixture),
            elevation=fixture.elevation,
            water=fixture.water,
            coverage=fixture.coverage,
            frame=fixture.frame,
        )
    )
    a = features["A"]
    assert a.elevation_missing is False
    assert a.elevation_m == pytest.approx(10.0, abs=0.15)
    assert a.distance_to_water_m == pytest.approx(80.0, abs=0.5)
    d = features["D"]
    # D is 400 m north of the origin: 10 + 0.01*400 = 14 m.
    assert d.elevation_m == pytest.approx(14.0, abs=0.2)
    assert d.distance_to_water_m == pytest.approx(480.0, abs=1.0)


def test_missing_coordinates_and_missing_layers_are_flagged():
    fixture = build_default_fixture()
    features = _by_uid(
        compute_group3_features(
            _post_records(fixture),
            coverage=fixture.coverage,
            frame=fixture.frame,
        )
    )
    miss = features["MISS"]
    assert miss.alert_features_status == "missing_coordinates"
    assert miss.alert_count_30m is None
    assert miss.elevation_missing is True
    assert miss.water_missing is True
    a = features["A"]
    assert a.elevation_missing is True
    assert a.water_missing is True
    assert "elevation_layer_absent" in a.layer_notes
    assert "water_layer_absent" in a.layer_notes


def test_official_export_placeholder_does_not_invent_alerts():
    fixture = build_default_fixture()
    placeholder = next(row for row in fixture.records if row.building_uid == "PH")
    assert placeholder.satellite_alert_status == OFFICIAL_EXPORT_PLACEHOLDER
    features = _by_uid(
        compute_group3_features(
            [placeholder],
            elevation=fixture.elevation,
            water=fixture.water,
            frame=fixture.frame,
        )
    )
    row = features["PH"]
    assert row.alert_features_status == OFFICIAL_EXPORT_PLACEHOLDER
    assert row.alert_count_30m is None
    assert row.alert_density_30m_per_km2 is None
    assert row.alert_count_100_300m is None
    assert row.nearest_collapse_found is False
    assert row.nearest_collapse_distance_m is None
    # Geographic layers may still be sampled while the alert column waits.
    assert row.elevation_missing is False
    assert row.distance_to_water_m == pytest.approx(80.0, abs=0.5)


def test_layer_ledger_records_obtainability_and_keeps_alerts_placeholder():
    ledger = LayerLedger.from_path(LEDGER)
    assert ledger.checked_timezone == "Asia/Shanghai"
    assert ledger.get("copernicus_dem_glo30").obtainable is True
    assert ledger.get("copernicus_dem_glo30").http_status == 200
    assert ledger.get("copernicus_dem_glo30").obtained_into_repo is False
    assert ledger.get("hydrosheds_hydrorivers").obtainable is True
    assert ledger.get("osm_land_polygons").obtainable is True
    assert ledger.get("jrc_global_surface_water").obtainable is True
    # 3DEP was checked, but this fetch path did not confirm access.
    assert ledger.get("usgs_3dep").obtainable is None
    assert ledger.get("usgs_3dep").http_status == 403
    placeholders = {layer.id: layer for layer in ledger.placeholders()}
    assert "michael_satellite_alerts" in placeholders
    assert "palu_satellite_alerts" in placeholders
    assert placeholders["michael_satellite_alerts"].usable_as_of(date(2018, 10, 12)) is False
    # Preparing Palu geographic layers does not make the Palu alert column usable.
    assert ledger.usable_layers(
        "satellite_alerts", event_id="palu-tsunami-2018", as_of=date(2018, 9, 18)
    ) == ()
    glo = ledger.get("copernicus_dem_glo30")
    assert glo.product_available_from == date(2021, 4, 22)
    assert glo.acquired_from == date(2011, 1, 1)
    assert glo.acquired_until == date(2015, 12, 31)
    dems = ledger.usable_layers(
        "elevation", event_id="hurricane-michael-2018", as_of=date(2026, 11, 18)
    )
    assert any(layer.id == "copernicus_dem_glo30" for layer in dems)
    # Publication after 2018 must not reject GLO-30 when acquisition is 2011-2015.
    early = ledger.usable_layers(
        "elevation", event_id="hurricane-michael-2018", as_of=date(2018, 10, 12)
    )
    assert any(layer.id == "copernicus_dem_glo30" for layer in early)
    assert glo.usable_as_of(date(2018, 10, 12)) is True
    with pytest.raises(ConfigError, match="official-export placeholders"):
        ledger.assert_no_official_alerts(["hurricane-michael-2018"])


def test_ledger_rejects_obtained_rasters_in_the_repository():
    payload = {
        "schema_version": 1,
        "checked_timezone": "Asia/Shanghai",
        "layers": [
            {
                "id": "should-not-commit",
                "theme": "elevation",
                "obtained_into_repo": True,
                "obtainable": True,
            }
        ],
    }
    with pytest.raises(ConfigError, match="stay out of the repository"):
        LayerLedger.from_mapping(payload)


def test_ledger_usable_as_of_uses_acquisition_not_publication():
    payload = {
        "schema_version": 1,
        "checked_timezone": "Asia/Shanghai",
        "layers": [
            {
                "id": "early_acq_late_pub",
                "theme": "elevation",
                "acquired_from": "2011-01-01",
                "acquired_until": "2015-12-31",
                "product_available_from": "2021-04-22",
                "obtainable": True,
            },
            {
                "id": "late_acq",
                "theme": "elevation",
                "acquired_from": "2020-01-01",
                "acquired_until": "2022-12-31",
                "product_available_from": "2015-01-01",
                "obtainable": True,
            },
        ],
    }
    ledger = LayerLedger.from_mapping(payload)
    as_of = date(2018, 10, 12)
    ids = {layer.id for layer in ledger.usable_layers("elevation", as_of=as_of)}
    assert ids == {"early_acq_late_pub"}


def test_missing_coverage_leaves_edge_flags_unknown():
    records = no_alert_records()
    assert CoverageMap().disk_is_edge(0.0, 0.0, 30.0) is None
    features = _by_uid(compute_group3_features(records, coverage=CoverageMap()))
    assert features["N1"].window_30m_edge is None
    assert features["N1"].window_100_300m_edge is None
    assert "window_edge_unknown_missing_coverage" in features["N1"].layer_notes
    assert features["N1"].window_30m_observed_area_m2 == pytest.approx(disk_area_m2(30.0))


def test_elevation_out_of_grid_is_missing_and_water_without_segments_is_missing():
    frame = MetricFrame(10.0, 20.0)
    grid = ElevationGrid(
        lats=np.array([9.99, 10.01]),
        lons=np.array([19.99, 20.01]),
        values_m=np.array([[1.0, 2.0], [3.0, 4.0]]),
        available_from=date(2015, 1, 1),
        layer_id="tiny",
    )
    water = WaterGeometries(polylines=[], frame=frame, available_from=date(2015, 1, 1))
    lat, lon = offset_lat_lon(0.0, 0.0, frame)
    far_lat, far_lon = offset_lat_lon(0.0, 50_000.0, frame)
    records = [
        BuildingRecord(
            building_uid="IN",
            event_id="tiny",
            lat=lat,
            lon=lon,
            satellite_grade=GRADE_NO,
            in_imagery=True,
        ),
        BuildingRecord(
            building_uid="OUT",
            event_id="tiny",
            lat=far_lat,
            lon=far_lon,
            satellite_grade=GRADE_NO,
            in_imagery=True,
        ),
    ]
    features = _by_uid(
        compute_group3_features(records, elevation=grid, water=water, frame=frame)
    )
    assert features["IN"].elevation_missing is False
    assert features["OUT"].elevation_missing is True
    assert features["IN"].water_missing is True
    assert features["OUT"].water_missing is True

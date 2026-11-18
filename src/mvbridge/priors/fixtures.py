"""Synthetic Group-3 fixtures. No real buildings, scores, or rasters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np

from mvbridge.priors.coordinates import MetricFrame
from mvbridge.priors.coverage import AxisAlignedBox, CoverageMap
from mvbridge.priors.layers import ElevationGrid, WaterGeometries
from mvbridge.priors.schema import (
    GRADE_DESTROYED,
    GRADE_MAJOR,
    GRADE_MINOR,
    GRADE_NO,
    OFFICIAL_EXPORT_PLACEHOLDER,
    PHASE_POST,
    PHASE_PRE,
    BuildingRecord,
)

# A made-up origin. Not a real event coordinate catalog.
SYNTHETIC_ORIGIN_LAT = 10.0
SYNTHETIC_ORIGIN_LON = 20.0
SYNTHETIC_EVENT = "synthetic-group3-fixture"

# Acquisition windows and publication dates are stored separately.
STATIC_DEM_ACQUIRED_FROM = date(2011, 1, 1)
STATIC_DEM_ACQUIRED_UNTIL = date(2015, 12, 31)
STATIC_DEM_AVAILABLE_FROM = date(2015, 1, 1)
LATE_DEM_PUBLISHED_FROM = date(2025, 1, 1)
FUTURE_DEM_ACQUIRED_FROM = date(2025, 1, 1)
WATER_ACQUIRED_FROM = date(2016, 1, 1)
WATER_AVAILABLE_FROM = date(2016, 1, 1)


def synthetic_frame() -> MetricFrame:
    return MetricFrame(SYNTHETIC_ORIGIN_LAT, SYNTHETIC_ORIGIN_LON)


def offset_lat_lon(x_m: float, y_m: float, frame: MetricFrame | None = None) -> tuple[float, float]:
    frame = frame or synthetic_frame()
    lat, lon = frame.from_xy(x_m, y_m)
    return float(np.asarray(lat)), float(np.asarray(lon))


def _record(
    uid: str,
    x_m: float,
    y_m: float,
    *,
    grade: str | None,
    in_imagery: bool = True,
    in_survey: bool = True,
    phase: str = PHASE_POST,
    tile_id: str = "tile-a",
    status: str = "fixture",
    ground_grade: str | None = None,
    acquisition_date: date | None = date(2018, 10, 12),
    event_id: str = SYNTHETIC_EVENT,
    frame: MetricFrame | None = None,
) -> BuildingRecord:
    lat, lon = offset_lat_lon(x_m, y_m, frame)
    return BuildingRecord(
        building_uid=uid,
        event_id=event_id,
        lat=lat,
        lon=lon,
        phase=phase,
        satellite_grade=grade,
        satellite_alert_status=status,
        in_survey=in_survey,
        in_imagery=in_imagery,
        tile_id=tile_id,
        acquisition_date=acquisition_date,
        as_of_date=acquisition_date,
        ground_grade=ground_grade,
    )


@dataclass(frozen=True)
class Group3Fixture:
    """A small synthetic town with known metric offsets, used only in tests."""

    name: str
    records: tuple[BuildingRecord, ...]
    coverage: CoverageMap
    elevation: ElevationGrid
    water: WaterGeometries
    frame: MetricFrame
    notes: str


def build_default_fixture() -> Group3Fixture:
    """Build the standard leave-one-out, coverage, edge, and leakage fixture.

    Layout in local meters (origin at the synthetic frame):

    * A at (0, 0): imagery+survey, satellite no-damage, but ground=destroyed.
      A must never be treated as an alert or collapse because of the survey label.
    * B at (20, 0): satellite destroyed (true close collapse / 30 m alert).
    * C at (150, 0): satellite major (inside the 100-300 m annulus, not 30 m).
    * D at (0, 400): satellite destroyed, outside 300 m of A.
    * E at (20, 20): survey-only, satellite grade destroyed — excluded from the
      neighbor universe because there is no imagery coverage.
    * F at (0, -10): imagery only (no survey), satellite minor.
    * G at (29, 0): just inside 30 m of A, satellite minor.
    * H at (90, 0): in the 30-100 m gap, satellite destroyed — counted in
      neither the 30 m disk nor the 100-300 m annulus, but it is the nearest
      collapse for a no-damage building at the origin if B is ignored.
    * EDGE at (95, 0) is not used. Instead K sits at (0, 48) near the north
      imagery tile edge (tile covers y in [-50, 50], x in [-50, 200]).
    * PRE at (20, 0) shares B's coordinates but is a pre-event destroyed call
      and must not enter post-event densities.
    * MISS has no coordinates.
    * PH is a placeholder official-alert row on a second event id.
    """
    frame = synthetic_frame()
    records = [
        _record("A", 0.0, 0.0, grade=GRADE_NO, ground_grade=GRADE_DESTROYED),
        _record("B", 20.0, 0.0, grade=GRADE_DESTROYED, ground_grade=GRADE_NO),
        _record("C", 150.0, 0.0, grade=GRADE_MAJOR),
        _record("D", 0.0, 400.0, grade=GRADE_DESTROYED, tile_id="tile-b"),
        _record("E", 20.0, 20.0, grade=GRADE_DESTROYED, in_imagery=False, in_survey=True),
        _record("F", 0.0, -10.0, grade=GRADE_MINOR, in_survey=False, in_imagery=True),
        _record("G", 29.0, 0.0, grade=GRADE_MINOR),
        _record("H", 90.0, 0.0, grade=GRADE_DESTROYED),
        _record("K", 0.0, 48.0, grade=GRADE_NO, tile_id="tile-a"),
        _record(
            "PRE",
            20.0,
            0.0,
            grade=GRADE_DESTROYED,
            phase=PHASE_PRE,
            acquisition_date=date(2018, 10, 1),
        ),
        BuildingRecord(
            building_uid="MISS",
            event_id=SYNTHETIC_EVENT,
            lat=None,
            lon=None,
            satellite_grade=GRADE_DESTROYED,
            satellite_alert_status="fixture",
            in_imagery=True,
            in_survey=True,
        ),
        BuildingRecord(
            building_uid="PH",
            event_id="hurricane-michael-2018",
            lat=SYNTHETIC_ORIGIN_LAT,
            lon=SYNTHETIC_ORIGIN_LON,
            satellite_grade=None,
            satellite_alert_status=OFFICIAL_EXPORT_PLACEHOLDER,
            in_imagery=True,
            in_survey=True,
        ),
    ]
    # Imagery covers most of the cluster, but not D (400 m north) and it cuts
    # off just beyond K (y=48), so K's 30 m disk and 300 m annulus are clipped.
    coverage = CoverageMap(
        imagery_boxes=(
            AxisAlignedBox(xmin=-50.0, xmax=200.0, ymin=-50.0, ymax=50.0),
            AxisAlignedBox(xmin=-50.0, xmax=50.0, ymin=350.0, ymax=450.0),
        ),
        survey_boxes=(
            AxisAlignedBox(xmin=-50.0, xmax=200.0, ymin=-50.0, ymax=80.0),
            AxisAlignedBox(xmin=-50.0, xmax=50.0, ymin=350.0, ymax=450.0),
        ),
    )
    # A gentle north-sloping plane: elevation_m = 10 + 0.01 * y_m, constant in x.
    # At the origin this is 10 m. Water is an east-west coastline at y = -80 m.
    lats = []
    lons = []
    ys = np.linspace(-500.0, 500.0, 21)
    xs = np.linspace(-500.0, 500.0, 21)
    values = np.zeros((ys.size, xs.size), dtype=np.float64)
    for i, y in enumerate(ys):
        lat, _ = offset_lat_lon(0.0, float(y), frame)
        lats.append(lat)
        for j, x in enumerate(xs):
            _, lon = offset_lat_lon(float(x), 0.0, frame)
            if i == 0:
                lons.append(lon)
            values[i, j] = 10.0 + 0.01 * y
    elevation = ElevationGrid(
        lats=np.asarray(lats),
        lons=np.asarray(lons),
        values_m=values,
        available_from=STATIC_DEM_AVAILABLE_FROM,
        acquired_from=STATIC_DEM_ACQUIRED_FROM,
        acquired_until=STATIC_DEM_ACQUIRED_UNTIL,
        layer_id="synthetic_dem_static",
    )
    coast = [offset_lat_lon(float(x), -80.0, frame) for x in np.linspace(-500.0, 500.0, 21)]
    water = WaterGeometries(
        polylines=[coast],
        frame=frame,
        available_from=WATER_AVAILABLE_FROM,
        acquired_from=WATER_ACQUIRED_FROM,
        layer_id="synthetic_coast",
    )
    return Group3Fixture(
        name="default-synthetic-town",
        records=tuple(records),
        coverage=coverage,
        elevation=elevation,
        water=water,
        frame=frame,
        notes=(
            "Synthetic only. Ground labels on A/B are deliberately opposite the "
            "satellite grades so leakage tests can fail if survey Y is read."
        ),
    )


def late_published_elevation_grid(fixture: Group3Fixture | None = None) -> ElevationGrid:
    """Early acquisition, but a publication date after the 2018 fixture as-of."""
    fixture = fixture or build_default_fixture()
    return ElevationGrid(
        lats=fixture.elevation.lats.copy(),
        lons=fixture.elevation.lons.copy(),
        values_m=fixture.elevation.values_m.copy(),
        available_from=LATE_DEM_PUBLISHED_FROM,
        acquired_from=STATIC_DEM_ACQUIRED_FROM,
        acquired_until=STATIC_DEM_ACQUIRED_UNTIL,
        layer_id="synthetic_dem_late_published",
    )


def future_acquired_elevation_grid(fixture: Group3Fixture | None = None) -> ElevationGrid:
    """A DEM whose collection window starts after the 2018 fixture as-of date."""
    fixture = fixture or build_default_fixture()
    return ElevationGrid(
        lats=fixture.elevation.lats.copy(),
        lons=fixture.elevation.lons.copy(),
        values_m=fixture.elevation.values_m.copy(),
        available_from=FUTURE_DEM_ACQUIRED_FROM,
        acquired_from=FUTURE_DEM_ACQUIRED_FROM,
        layer_id="synthetic_dem_future_acquisition",
    )


def isolated_collapse_records() -> list[BuildingRecord]:
    """A single destroyed building: leave-self-out must not find a collapse."""
    return [_record("SOLO", 0.0, 0.0, grade=GRADE_DESTROYED)]


def no_alert_records() -> list[BuildingRecord]:
    """Two close intact buildings and no satellite alerts."""
    return [
        _record("N1", 0.0, 0.0, grade=GRADE_NO),
        _record("N2", 15.0, 0.0, grade=GRADE_NO),
    ]


def records_to_dicts(records: Iterable[BuildingRecord]) -> list[dict]:
    """JSON-friendly dump of synthetic records (still not real event data)."""
    rows = []
    for row in records:
        rows.append(
            {
                "building_uid": row.building_uid,
                "event_id": row.event_id,
                "lat": row.lat,
                "lon": row.lon,
                "phase": row.phase,
                "satellite_grade": row.satellite_grade,
                "satellite_alert_status": row.satellite_alert_status,
                "in_survey": row.in_survey,
                "in_imagery": row.in_imagery,
                "tile_id": row.tile_id,
                "acquisition_date": None if row.acquisition_date is None else row.acquisition_date.isoformat(),
                "ground_grade": row.ground_grade,
            }
        )
    return rows

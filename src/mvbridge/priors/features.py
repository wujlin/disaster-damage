"""Group-3 neighborhood and geographic features, leave-one-building-out."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

import numpy as np

from mvbridge.errors import ConfigError
from mvbridge.priors.coordinates import MetricFrame
from mvbridge.priors.coverage import CoverageMap, density_per_km2
from mvbridge.priors.layers import ElevationGrid, WaterGeometries
from mvbridge.priors.neighborhood import NeighborhoodIndex
from mvbridge.priors.schema import (
    DEFAULT_ALERT_GRADES,
    DEFAULT_COLLAPSE_GRADES,
    GROUND_LABEL_FIELDS,
    OFFICIAL_EXPORT_PLACEHOLDER,
    PHASE_POST,
    STATUS_COMPUTED,
    STATUS_MISSING_ALERTS,
    STATUS_MISSING_COORDINATES,
    STATUS_PLACEHOLDER,
    BuildingRecord,
    Group3FeatureRow,
)

NEAR_RADIUS_M = 30.0
ANNULUS_INNER_M = 100.0
ANNULUS_OUTER_M = 300.0


def _refuse_ground_label_source(source_name: str | None) -> None:
    if source_name is None:
        return
    if source_name in GROUND_LABEL_FIELDS or source_name.lower() in GROUND_LABEL_FIELDS:
        raise ConfigError(
            f"neighborhood stats cannot be built from {source_name!r}: "
            "use satellite-side alerts only, never the current disaster's ground labels"
        )


def _same_event_phase(
    records: Sequence[BuildingRecord],
    event_id: str,
    phase: str,
) -> list[BuildingRecord]:
    return [row for row in records if row.event_id == event_id and row.phase == phase]


def _neighbor_universe(
    records: Sequence[BuildingRecord],
    *,
    alert_grades: Iterable[str],
    collapse_grades: Iterable[str],
) -> tuple[list[BuildingRecord], bool, bool]:
    """Imagery-covered rows with a usable satellite grade, never survey-only.

    Returns (neighbors, any_placeholder, any_usable_alert_field).
    """
    _ = (alert_grades, collapse_grades)  # grades are applied per row below
    placeholders = False
    usable_field = False
    neighbors: list[BuildingRecord] = []
    for row in records:
        if row.alerts_are_placeholder:
            placeholders = True
            continue
        if not row.in_imagery or not row.has_coordinates:
            continue
        if row.satellite_grade is None:
            continue
        usable_field = True
        neighbors.append(row)
    return neighbors, placeholders, usable_field


def _row_as_of(row: BuildingRecord, default: date | None) -> date | None:
    return row.as_of_date or row.acquisition_date or default


def compute_group3_features(
    records: Sequence[BuildingRecord],
    *,
    elevation: ElevationGrid | None = None,
    water: WaterGeometries | None = None,
    coverage: CoverageMap | None = None,
    frame: MetricFrame | None = None,
    phase: str = PHASE_POST,
    as_of: date | None = None,
    near_radius_m: float = NEAR_RADIUS_M,
    annulus_inner_m: float = ANNULUS_INNER_M,
    annulus_outer_m: float = ANNULUS_OUTER_M,
    alert_grades: Iterable[str] = DEFAULT_ALERT_GRADES,
    collapse_grades: Iterable[str] = DEFAULT_COLLAPSE_GRADES,
    alert_source_name: str = "satellite_grade",
) -> list[Group3FeatureRow]:
    """Compute Group-3 features for every record in the requested phase.

    Neighborhood densities and the nearest-collapse distance are leave-one-
    building-out and use only satellite-side alerts of the same event and
    phase. Survey-only footprints are not treated as zero-alert neighbors.
    Geographic layers may be sampled even while official alert columns remain
    placeholders.
    """
    _refuse_ground_label_source(alert_source_name)
    if annulus_inner_m <= near_radius_m:
        # Allowed physically, but the documented default is a 30 m disk and a
        # 100-300 m annulus with a deliberate gap. Do not silently overlap.
        if not (annulus_inner_m > 0 and annulus_outer_m > annulus_inner_m):
            raise ConfigError("annulus radii must satisfy 0 <= near, and inner < outer")
    if near_radius_m <= 0 or annulus_inner_m < 0 or annulus_outer_m <= annulus_inner_m:
        raise ConfigError("neighborhood radii must be positive with inner < outer")

    alert_grade_set = frozenset(alert_grades)
    collapse_grade_set = frozenset(collapse_grades)
    if not records:
        return []

    event_ids = tuple(dict.fromkeys(row.event_id for row in records))
    rows_out: list[Group3FeatureRow] = []
    for event_id in event_ids:
        rows_out.extend(
            _compute_event(
                _same_event_phase(records, event_id, phase),
                event_id=event_id,
                elevation=elevation,
                water=water,
                coverage=coverage,
                frame=frame,
                as_of=as_of,
                near_radius_m=near_radius_m,
                annulus_inner_m=annulus_inner_m,
                annulus_outer_m=annulus_outer_m,
                alert_grades=alert_grade_set,
                collapse_grades=collapse_grade_set,
            )
        )
    return rows_out


def _compute_event(
    records: Sequence[BuildingRecord],
    *,
    event_id: str,
    elevation: ElevationGrid | None,
    water: WaterGeometries | None,
    coverage: CoverageMap | None,
    frame: MetricFrame | None,
    as_of: date | None,
    near_radius_m: float,
    annulus_inner_m: float,
    annulus_outer_m: float,
    alert_grades: frozenset[str],
    collapse_grades: frozenset[str],
) -> list[Group3FeatureRow]:
    if not records:
        return []

    coord_rows = [row for row in records if row.has_coordinates]
    if frame is None:
        if not coord_rows:
            frame = MetricFrame(0.0, 0.0)
        else:
            frame = MetricFrame.from_points(
                [row.lat for row in coord_rows],
                [row.lon for row in coord_rows],
            )

    neighbors, placeholders, usable_field = _neighbor_universe(
        records, alert_grades=alert_grades, collapse_grades=collapse_grades
    )
    if neighbors:
        xs, ys = frame.to_xy(
            np.asarray([row.lat for row in neighbors], dtype=np.float64),
            np.asarray([row.lon for row in neighbors], dtype=np.float64),
        )
        index = NeighborhoodIndex(
            x=np.asarray(xs, dtype=np.float64),
            y=np.asarray(ys, dtype=np.float64),
            ids=np.asarray([row.building_uid for row in neighbors], dtype=object),
        )
        alert_mask = np.asarray([row.satellite_is_alert(alert_grades) for row in neighbors], dtype=bool)
        collapse_mask = np.asarray([row.satellite_is_collapse(collapse_grades) for row in neighbors], dtype=bool)
        neighbor_by_id = {row.building_uid: i for i, row in enumerate(neighbors)}
    else:
        index = None
        alert_mask = None
        collapse_mask = None
        neighbor_by_id = {}

    if placeholders and not usable_field:
        alert_status_default = STATUS_PLACEHOLDER
    elif not usable_field:
        alert_status_default = STATUS_MISSING_ALERTS
    else:
        alert_status_default = STATUS_COMPUTED

    coverage = coverage or CoverageMap()
    out: list[Group3FeatureRow] = []
    for row in records:
        out.append(
            _feature_for_row(
                row,
                event_id=event_id,
                frame=frame,
                index=index,
                alert_mask=alert_mask,
                collapse_mask=collapse_mask,
                neighbor_by_id=neighbor_by_id,
                coverage=coverage,
                elevation=elevation,
                water=water,
                as_of=_row_as_of(row, as_of),
                near_radius_m=near_radius_m,
                annulus_inner_m=annulus_inner_m,
                annulus_outer_m=annulus_outer_m,
                alert_grades=alert_grades,
                alert_status_default=alert_status_default,
            )
        )
    return out


def _feature_for_row(
    row: BuildingRecord,
    *,
    event_id: str,
    frame: MetricFrame,
    index: NeighborhoodIndex | None,
    alert_mask: np.ndarray | None,
    collapse_mask: np.ndarray | None,
    neighbor_by_id: dict[str, int],
    coverage: CoverageMap,
    elevation: ElevationGrid | None,
    water: WaterGeometries | None,
    as_of: date | None,
    near_radius_m: float,
    annulus_inner_m: float,
    annulus_outer_m: float,
    alert_grades: frozenset[str],
    alert_status_default: str,
) -> Group3FeatureRow:
    notes: list[str] = []
    elevation_m = None
    elevation_missing = True
    water_m = None
    water_missing = True
    if row.has_coordinates:
        if elevation is None:
            notes.append("elevation_layer_absent")
        elif not elevation.usable(as_of):
            notes.append(f"elevation_unavailable_as_of:{elevation.layer_id}")
        else:
            elevation_m = elevation.sample(row.lat, row.lon, as_of=as_of)
            elevation_missing = elevation_m is None
            if elevation_missing:
                notes.append("elevation_sample_missing")
        if water is None:
            notes.append("water_layer_absent")
        elif not water.usable(as_of):
            notes.append(f"water_unavailable_as_of:{water.layer_id}")
        else:
            water_m = water.distance_m(row.lat, row.lon, as_of=as_of)
            water_missing = water_m is None
            if water_missing:
                notes.append("water_sample_missing")
    else:
        notes.append("missing_coordinates")

    if not row.has_coordinates:
        return Group3FeatureRow(
            building_uid=row.building_uid,
            event_id=event_id,
            alert_count_30m=None,
            alert_density_30m_per_km2=None,
            alert_count_100_300m=None,
            alert_density_100_300m_per_km2=None,
            nearest_collapse_distance_m=None,
            nearest_collapse_found=False,
            elevation_m=elevation_m,
            elevation_missing=elevation_missing,
            distance_to_water_m=water_m,
            water_missing=water_missing,
            window_30m_edge=False,
            window_100_300m_edge=False,
            window_30m_observed_area_m2=None,
            window_100_300m_observed_area_m2=None,
            focal_in_survey=row.in_survey,
            focal_in_imagery=row.in_imagery,
            alert_features_status=STATUS_MISSING_COORDINATES,
            layer_notes=tuple(notes),
        )

    x_arr, y_arr = frame.to_xy(row.lat, row.lon)
    x0 = float(np.asarray(x_arr))
    y0 = float(np.asarray(y_arr))
    edge_30 = coverage.disk_is_edge(x0, y0, near_radius_m)
    edge_300 = coverage.disk_is_edge(x0, y0, annulus_outer_m)
    area_30 = coverage.observed_disk_area_m2(x0, y0, near_radius_m)
    area_ring = coverage.observed_annulus_area_m2(x0, y0, annulus_inner_m, annulus_outer_m)
    if edge_30:
        notes.append("window_30m_tile_or_map_edge")
    if edge_300:
        notes.append("window_100_300m_tile_or_map_edge")

    if alert_status_default in {STATUS_PLACEHOLDER, STATUS_MISSING_ALERTS} or index is None:
        if alert_status_default == STATUS_PLACEHOLDER:
            notes.append("michael_or_event_alert_column_is_official_export_placeholder")
        elif index is None:
            notes.append("no_imagery_covered_satellite_neighbors")
        return Group3FeatureRow(
            building_uid=row.building_uid,
            event_id=event_id,
            alert_count_30m=None if alert_status_default == STATUS_PLACEHOLDER else 0.0,
            alert_density_30m_per_km2=(
                None if alert_status_default == STATUS_PLACEHOLDER else density_per_km2(0.0, area_30)
            ),
            alert_count_100_300m=None if alert_status_default == STATUS_PLACEHOLDER else 0.0,
            alert_density_100_300m_per_km2=(
                None if alert_status_default == STATUS_PLACEHOLDER else density_per_km2(0.0, area_ring)
            ),
            nearest_collapse_distance_m=None,
            nearest_collapse_found=False,
            elevation_m=elevation_m,
            elevation_missing=elevation_missing,
            distance_to_water_m=water_m,
            water_missing=water_missing,
            window_30m_edge=edge_30,
            window_100_300m_edge=edge_300,
            window_30m_observed_area_m2=area_30,
            window_100_300m_observed_area_m2=area_ring,
            focal_in_survey=row.in_survey,
            focal_in_imagery=row.in_imagery,
            alert_features_status=alert_status_default,
            layer_notes=tuple(notes),
        )

    exclude = {row.building_uid}
    near_ids, _ = index.query_disk(x0, y0, near_radius_m, exclude_ids=exclude)
    ring_ids, _ = index.query_annulus(
        x0, y0, annulus_inner_m, annulus_outer_m, exclude_ids=exclude
    )
    count_30 = _count_alerts(near_ids, neighbor_by_id, alert_mask)
    count_ring = _count_alerts(ring_ids, neighbor_by_id, alert_mask)
    collapse_found_id, collapse_dist = index.nearest(
        x0, y0, exclude_ids=exclude, mask=collapse_mask
    )
    found = collapse_found_id is not None
    # Ground labels are never consulted here. alert_mask and collapse_mask
    # are built only from satellite_is_alert / satellite_is_collapse.
    _ = alert_grades

    return Group3FeatureRow(
        building_uid=row.building_uid,
        event_id=event_id,
        alert_count_30m=float(count_30),
        alert_density_30m_per_km2=density_per_km2(count_30, area_30),
        alert_count_100_300m=float(count_ring),
        alert_density_100_300m_per_km2=density_per_km2(count_ring, area_ring),
        nearest_collapse_distance_m=None if not found else float(collapse_dist),
        nearest_collapse_found=found,
        elevation_m=elevation_m,
        elevation_missing=elevation_missing,
        distance_to_water_m=water_m,
        water_missing=water_missing,
        window_30m_edge=edge_30,
        window_100_300m_edge=edge_300,
        window_30m_observed_area_m2=area_30,
        window_100_300m_observed_area_m2=area_ring,
        focal_in_survey=row.in_survey,
        focal_in_imagery=row.in_imagery,
        alert_features_status=STATUS_COMPUTED,
        layer_notes=tuple(notes),
    )


def _count_alerts(ids: np.ndarray, neighbor_by_id: dict[str, int], alert_mask: np.ndarray) -> int:
    total = 0
    for item in ids:
        index = neighbor_by_id.get(item)
        if index is not None and alert_mask[index]:
            total += 1
    return total


def placeholder_records(
    uids: Iterable[str],
    event_id: str,
    lats: Iterable[float],
    lons: Iterable[float],
    **kwargs,
) -> list[BuildingRecord]:
    """Build records whose satellite alert column is the official-export placeholder."""
    rows = []
    for uid, lat, lon in zip(uids, lats, lons, strict=True):
        rows.append(
            BuildingRecord(
                building_uid=str(uid),
                event_id=event_id,
                lat=lat,
                lon=lon,
                satellite_grade=None,
                satellite_alert_status=OFFICIAL_EXPORT_PLACEHOLDER,
                **kwargs,
            )
        )
    return rows

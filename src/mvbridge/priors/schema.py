"""Records, feature rows, and satellite-alert vocabulary for Group 3.

Neighborhood statistics may use only satellite-side alert fields. Ground
labels for the current disaster are carried on the record so tests can prove
they are never read. Michael's real official alert column is a placeholder
until the official export arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from mvbridge.errors import ConfigError

PHASE_PRE = "pre"
PHASE_POST = "post"
VALID_PHASES = (PHASE_PRE, PHASE_POST)

# Official export has not been joined. Do not fill this from ground labels.
OFFICIAL_EXPORT_PLACEHOLDER = "OFFICIAL_EXPORT_PLACEHOLDER"

STATUS_COMPUTED = "computed"
STATUS_PLACEHOLDER = OFFICIAL_EXPORT_PLACEHOLDER
STATUS_MISSING_ALERTS = "missing_alerts"
STATUS_MISSING_COORDINATES = "missing_coordinates"

GRADE_NO = "no"
GRADE_MINOR = "minor"
GRADE_MAJOR = "major"
GRADE_DESTROYED = "destroyed"
SATELLITE_GRADES = (GRADE_NO, GRADE_MINOR, GRADE_MAJOR, GRADE_DESTROYED)

# Default: any non-intact satellite call is an "alert".
DEFAULT_ALERT_GRADES = frozenset({GRADE_MINOR, GRADE_MAJOR, GRADE_DESTROYED})
DEFAULT_COLLAPSE_GRADES = frozenset({GRADE_DESTROYED})

_GRADE_ALIASES = {
    "no": GRADE_NO,
    "no-damage": GRADE_NO,
    "no_damage": GRADE_NO,
    "none": GRADE_NO,
    "intact": GRADE_NO,
    "undamaged": GRADE_NO,
    "minor": GRADE_MINOR,
    "minor-damage": GRADE_MINOR,
    "minor_damage": GRADE_MINOR,
    "major": GRADE_MAJOR,
    "major-damage": GRADE_MAJOR,
    "major_damage": GRADE_MAJOR,
    "destroyed": GRADE_DESTROYED,
    "destroy": GRADE_DESTROYED,
    "collapsed": GRADE_DESTROYED,
    "collapse": GRADE_DESTROYED,
}

# Names that look like current-disaster ground labels. Never use them as alerts.
GROUND_LABEL_FIELDS = frozenset(
    {
        "ground_grade",
        "ground_label",
        "survey_grade",
        "survey_label",
        "y",
        "Y",
        "reference_grade",
        "reference_label",
        "uav_grade",
    }
)


def normalize_grade(value: str | None) -> str | None:
    """Map a satellite grade string onto the four-class protocol vocabulary."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", ""}:
        return None
    if text not in _GRADE_ALIASES:
        raise ConfigError(f"unrecognized satellite grade {value!r}")
    return _GRADE_ALIASES[text]


def is_satellite_alert(grade: str | None, alert_grades: Iterable[str] = DEFAULT_ALERT_GRADES) -> bool:
    normalized = normalize_grade(grade)
    return normalized is not None and normalized in set(alert_grades)


def is_satellite_collapse(grade: str | None, collapse_grades: Iterable[str] = DEFAULT_COLLAPSE_GRADES) -> bool:
    normalized = normalize_grade(grade)
    return normalized is not None and normalized in set(collapse_grades)


@dataclass(frozen=True)
class BuildingRecord:
    """One footprint used to build or query Group-3 features.

    ``satellite_grade`` and the derived alert flags are the only damage
    fields the neighborhood index may read. ``ground_grade`` is accepted so
    that leakage tests can put a contradictory survey label on the record.
    """

    building_uid: str
    event_id: str
    lat: float | None
    lon: float | None
    phase: str = PHASE_POST
    satellite_grade: str | None = None
    satellite_alert_status: str = "fixture"
    in_survey: bool = False
    in_imagery: bool = False
    tile_id: str | None = None
    acquisition_date: date | None = None
    as_of_date: date | None = None
    ground_grade: str | None = None

    def __post_init__(self) -> None:
        if not self.building_uid:
            raise ConfigError("building_uid is required")
        if not self.event_id:
            raise ConfigError("event_id is required")
        if self.phase not in VALID_PHASES:
            raise ConfigError(f"phase must be one of {VALID_PHASES}, got {self.phase!r}")
        # Normalize the stored satellite grade once. Ground grades are not normalized
        # here so a leakage test can leave arbitrary survey strings untouched.
        if self.satellite_grade is not None:
            object.__setattr__(self, "satellite_grade", normalize_grade(self.satellite_grade))

    @property
    def has_coordinates(self) -> bool:
        return (
            self.lat is not None
            and self.lon is not None
            and np_isfinite(self.lat)
            and np_isfinite(self.lon)
        )

    @property
    def alerts_are_placeholder(self) -> bool:
        return self.satellite_alert_status == OFFICIAL_EXPORT_PLACEHOLDER

    def satellite_is_alert(self, alert_grades: Iterable[str] = DEFAULT_ALERT_GRADES) -> bool:
        if self.alerts_are_placeholder or not self.in_imagery:
            return False
        return is_satellite_alert(self.satellite_grade, alert_grades)

    def satellite_is_collapse(self, collapse_grades: Iterable[str] = DEFAULT_COLLAPSE_GRADES) -> bool:
        if self.alerts_are_placeholder or not self.in_imagery:
            return False
        return is_satellite_collapse(self.satellite_grade, collapse_grades)


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


@dataclass
class Group3FeatureRow:
    """Leave-one-building-out Group-3 features for one footprint."""

    building_uid: str
    event_id: str
    alert_count_30m: float | None
    alert_density_30m_per_km2: float | None
    alert_count_100_300m: float | None
    alert_density_100_300m_per_km2: float | None
    nearest_collapse_distance_m: float | None
    nearest_collapse_found: bool
    elevation_m: float | None
    elevation_missing: bool
    distance_to_water_m: float | None
    water_missing: bool
    window_30m_edge: bool
    window_100_300m_edge: bool
    window_30m_observed_area_m2: float | None
    window_100_300m_observed_area_m2: float | None
    focal_in_survey: bool
    focal_in_imagery: bool
    alert_features_status: str
    layer_notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["layer_notes"] = list(self.layer_notes)
        return data

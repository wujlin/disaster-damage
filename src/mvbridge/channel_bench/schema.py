"""Analysis-table contract used by the channel bench.

Column names are intended to join later with score-contract and
widearea-priors-group3 exports. This module does not read those trees or
any official Michael dump.

``elevation`` and ``dist_to_water`` are unfrozen/candidate input columns.
They must not be treated as already-frozen Group-3 features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mvbridge.channel_bench.locks import (
    DAMAGE_CLASSES,
    INTACT_CLASS,
    UNFROZEN_CANDIDATE_TRACE_FIELDS,
)

# Re-exported so callers can depend on one schema module.
__all__ = [
    "ANALYSIS_TABLE_COLUMNS",
    "DAMAGE_CLASSES",
    "INTACT_CLASS",
    "BuildingAnalysisRow",
    "class_index",
    "imprint_bin",
    "require_known_grade",
    "UNFROZEN_CANDIDATE_TRACE_FIELDS",
]

ANALYSIS_TABLE_COLUMNS: tuple[str, ...] = (
    "building_uid",
    "image_id",
    "event_id",
    "seed_id",
    "block_id",
    "eligible",
    "surveyed",
    "y",
    "s",
    "visibility_bin",
    "clear_pixel_fraction",
    "gsd_m",
    "off_nadir_deg",
    "sat_alarm_density_30m",
    "sat_alarm_density_100_300m",
    "dist_to_nearest_sat_collapsed",
    "elevation",
    "dist_to_water",
    "imprint_score",
    "index_building_alarm_excluded",
    "ranking_group3_neighborhood",
    "ranking_group3_public_geo",
    "ranking_group3_combined",
    "ranking_visual_mismatch",
    "ranking_group3_plus_visual",
    "ranking_entropy",
    "ranking_one_minus_max_p",
)


def require_known_grade(name: str) -> str:
    if name not in DAMAGE_CLASSES:
        raise ValueError(f"unknown grade {name!r}, expected one of {DAMAGE_CLASSES}")
    return name


def class_index(name: str) -> int:
    return DAMAGE_CLASSES.index(require_known_grade(name))


def imprint_bin(score: float, edges: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)) -> int:
    """Assign a pre-fixed imprint bin. Edges are not fit on test labels."""
    value = float(score)
    if value < edges[0] or value > edges[-1]:
        raise ValueError(f"imprint_score {value} is outside locked edges {edges}")
    for index in range(len(edges) - 1):
        upper = edges[index + 1]
        if value < upper or index == len(edges) - 2:
            return index
    return len(edges) - 2


@dataclass(frozen=True)
class BuildingAnalysisRow:
    """One eligible or ineligible building on the analysis sheet."""

    building_uid: str
    event_id: str
    seed_id: str
    block_id: str
    y: str
    s: str
    visibility_bin: str
    imprint_score: float
    image_id: str | None = None
    eligible: bool = True
    surveyed: bool = True
    clear_pixel_fraction: float | None = None
    gsd_m: float | None = None
    off_nadir_deg: float | None = None
    sat_alarm_density_30m: float | None = None
    sat_alarm_density_100_300m: float | None = None
    dist_to_nearest_sat_collapsed: float | None = None
    elevation: float | None = None  # unfrozen/candidate, not a frozen Group-3 feature
    dist_to_water: float | None = None  # unfrozen/candidate, not a frozen Group-3 feature
    index_building_alarm_excluded: bool = True
    rankings: Mapping[str, float] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_known_grade(self.y)
        require_known_grade(self.s)
        if not self.index_building_alarm_excluded:
            raise ValueError(
                "neighborhood traces must exclude the index building's own alarm"
            )

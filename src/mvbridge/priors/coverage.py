"""Imagery, survey, and tile-extent coverage used to flag window edges."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AxisAlignedBox:
    """Rectangle in the local metric plane (meters)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError("box max must be >= min on both axes")

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def contains_disk(self, x: float, y: float, radius_m: float) -> bool:
        return (
            x - radius_m >= self.xmin
            and x + radius_m <= self.xmax
            and y - radius_m >= self.ymin
            and y + radius_m <= self.ymax
        )

    def intersects_disk(self, x: float, y: float, radius_m: float) -> bool:
        dx = 0.0 if self.xmin <= x <= self.xmax else min(abs(x - self.xmin), abs(x - self.xmax))
        dy = 0.0 if self.ymin <= y <= self.ymax else min(abs(y - self.ymin), abs(y - self.ymax))
        return dx * dx + dy * dy <= radius_m * radius_m


def disk_area_m2(radius_m: float) -> float:
    return math.pi * radius_m * radius_m


def annulus_area_m2(inner_m: float, outer_m: float) -> float:
    return disk_area_m2(outer_m) - disk_area_m2(inner_m)


def observed_ring_area_m2(
    x0: float,
    y0: float,
    inner_m: float,
    outer_m: float,
    covers,
    *,
    n_radial: int = 48,
    n_angular: int = 96,
) -> float:
    """Midpoint polar quadrature of a disk or annulus inside ``covers(x, y)``."""
    if outer_m <= 0 or outer_m <= inner_m:
        return 0.0
    if inner_m < 0:
        raise ValueError("inner_m must be non-negative")
    dr = (outer_m - inner_m) / n_radial
    dtheta = 2.0 * math.pi / n_angular
    area = 0.0
    for i in range(n_radial):
        r = inner_m + (i + 0.5) * dr
        weight = r * dr * dtheta
        for j in range(n_angular):
            theta = (j + 0.5) * dtheta
            if covers(x0 + r * math.cos(theta), y0 + r * math.sin(theta)):
                area += weight
    return area


@dataclass
class CoverageMap:
    """Union of imagery tiles and an optional survey extent, in meters.

    Neighborhood windows that extend outside the imagery union are marked as
    tile/map edges so a count is not silently treated as a complete density.
    If no imagery boxes are supplied, coverage is treated as unbounded and
    edge flags stay false. That case is explicit: completeness is unknown.
    """

    imagery_boxes: tuple[AxisAlignedBox, ...] = ()
    survey_boxes: tuple[AxisAlignedBox, ...] = ()
    unbounded: bool = field(init=False)

    def __post_init__(self) -> None:
        self.imagery_boxes = tuple(self.imagery_boxes)
        self.survey_boxes = tuple(self.survey_boxes)
        self.unbounded = len(self.imagery_boxes) == 0

    def in_imagery(self, x: float, y: float) -> bool:
        if self.unbounded:
            return True
        return any(box.contains(x, y) for box in self.imagery_boxes)

    def in_survey(self, x: float, y: float) -> bool:
        if not self.survey_boxes:
            return False
        return any(box.contains(x, y) for box in self.survey_boxes)

    def disk_is_edge(self, x: float, y: float, radius_m: float) -> bool:
        if self.unbounded:
            return False
        if not any(box.intersects_disk(x, y, radius_m) for box in self.imagery_boxes):
            return True
        return not any(box.contains_disk(x, y, radius_m) for box in self.imagery_boxes)

    def observed_disk_area_m2(self, x: float, y: float, radius_m: float) -> float:
        if self.unbounded:
            return disk_area_m2(radius_m)
        return observed_ring_area_m2(x, y, 0.0, radius_m, self.in_imagery)

    def observed_annulus_area_m2(self, x: float, y: float, inner_m: float, outer_m: float) -> float:
        if self.unbounded:
            return annulus_area_m2(inner_m, outer_m)
        return observed_ring_area_m2(x, y, inner_m, outer_m, self.in_imagery)


def density_per_km2(count: float, area_m2: float) -> float | None:
    if area_m2 is None or not np.isfinite(area_m2) or area_m2 <= 0:
        return None
    return float(count) / (area_m2 / 1_000_000.0)

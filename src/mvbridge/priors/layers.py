"""Elevation and distance-to-water lookups on the local metric plane.

These adapters accept already-prepared grids and geometries. They do not
download public rasters. A layer may be withheld when its provenance says it
was not obtainable as of the requested date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from mvbridge.priors.coordinates import MetricFrame


def _as_date(value: date | None) -> date | None:
    return value


def layer_usable(available_from: date | None, as_of: date | None, available_until: date | None = None) -> bool:
    """True when a layer's stated vintage covers the analysis as-of date."""
    if as_of is None:
        return True
    if available_from is not None and as_of < available_from:
        return False
    if available_until is not None and as_of > available_until:
        return False
    return True


@dataclass
class ElevationGrid:
    """Regular lat/lon grid with bilinear sampling. Missing cells are NaN."""

    lats: np.ndarray
    lons: np.ndarray
    values_m: np.ndarray
    available_from: date | None = None
    available_until: date | None = None
    layer_id: str = "elevation"

    def __post_init__(self) -> None:
        self.lats = np.asarray(self.lats, dtype=np.float64)
        self.lons = np.asarray(self.lons, dtype=np.float64)
        self.values_m = np.asarray(self.values_m, dtype=np.float64)
        if self.lats.ndim != 1 or self.lons.ndim != 1:
            raise ValueError("lats and lons must be 1-D axes")
        if self.values_m.shape != (self.lats.size, self.lons.size):
            raise ValueError("values_m must have shape (len(lats), len(lons))")
        if self.lats.size < 2 or self.lons.size < 2:
            raise ValueError("elevation grid needs at least a 2x2 lattice")

    def usable(self, as_of: date | None) -> bool:
        return layer_usable(self.available_from, as_of, self.available_until)

    def sample(self, lat: float, lon: float, *, as_of: date | None = None) -> float | None:
        if not self.usable(as_of):
            return None
        if not np.isfinite(lat) or not np.isfinite(lon):
            return None
        # Axes may be ascending or descending.
        lat_asc = self.lats[-1] >= self.lats[0]
        lon_asc = self.lons[-1] >= self.lons[0]
        lat_axis = self.lats if lat_asc else self.lats[::-1]
        lon_axis = self.lons if lon_asc else self.lons[::-1]
        values = self.values_m
        if not lat_asc:
            values = values[::-1, :]
        if not lon_asc:
            values = values[:, ::-1]
        if lat < lat_axis[0] or lat > lat_axis[-1] or lon < lon_axis[0] or lon > lon_axis[-1]:
            return None
        i = int(np.searchsorted(lat_axis, lat, side="right") - 1)
        j = int(np.searchsorted(lon_axis, lon, side="right") - 1)
        i = min(max(i, 0), lat_axis.size - 2)
        j = min(max(j, 0), lon_axis.size - 2)
        t = (lat - lat_axis[i]) / (lat_axis[i + 1] - lat_axis[i])
        u = (lon - lon_axis[j]) / (lon_axis[j + 1] - lon_axis[j])
        v00 = values[i, j]
        v10 = values[i + 1, j]
        v01 = values[i, j + 1]
        v11 = values[i + 1, j + 1]
        if not np.all(np.isfinite([v00, v10, v01, v11])):
            # Fall back to the nearest finite corner, otherwise missing.
            corners = np.array(
                [
                    [0.0, 0.0, v00],
                    [1.0, 0.0, v10],
                    [0.0, 1.0, v01],
                    [1.0, 1.0, v11],
                ]
            )
            finite = np.isfinite(corners[:, 2])
            if not np.any(finite):
                return None
            d2 = (corners[finite, 0] - t) ** 2 + (corners[finite, 1] - u) ** 2
            return float(corners[finite][int(np.argmin(d2)), 2])
        return float((1 - t) * (1 - u) * v00 + t * (1 - u) * v10 + (1 - t) * u * v01 + t * u * v11)


def _segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return float(np.hypot(px - x1, py - y1))
    t = ((px - x1) * dx + (py - y1) * dy) / length2
    t = min(1.0, max(0.0, t))
    return float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))


@dataclass
class WaterGeometries:
    """Coast or water polylines in geographic degrees, queried in meters."""

    polylines: Sequence[Sequence[tuple[float, float]]]
    frame: MetricFrame
    available_from: date | None = None
    available_until: date | None = None
    layer_id: str = "water"

    def __post_init__(self) -> None:
        segments: list[tuple[float, float, float, float]] = []
        for line in self.polylines:
            pts = list(line)
            if len(pts) < 2:
                continue
            lats = np.asarray([p[0] for p in pts], dtype=np.float64)
            lons = np.asarray([p[1] for p in pts], dtype=np.float64)
            xs, ys = self.frame.to_xy(lats, lons)
            for i in range(len(pts) - 1):
                if np.isfinite(xs[i]) and np.isfinite(ys[i]) and np.isfinite(xs[i + 1]) and np.isfinite(ys[i + 1]):
                    segments.append((float(xs[i]), float(ys[i]), float(xs[i + 1]), float(ys[i + 1])))
        self._segments = tuple(segments)

    def usable(self, as_of: date | None) -> bool:
        return layer_usable(self.available_from, as_of, self.available_until)

    def distance_m(self, lat: float, lon: float, *, as_of: date | None = None) -> float | None:
        if not self.usable(as_of):
            return None
        if not np.isfinite(lat) or not np.isfinite(lon):
            return None
        if not self._segments:
            return None
        x_arr, y_arr = self.frame.to_xy(lat, lon)
        x = float(np.asarray(x_arr))
        y = float(np.asarray(y_arr))
        best = min(_segment_distance(x, y, *seg) for seg in self._segments)
        return float(best)

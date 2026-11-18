"""Local metric coordinates for neighborhood queries.

An azimuthal equidistant frame is centered on an event reference point so
that Euclidean distances in the (x, y) plane are meters on the sphere, with
low distortion at city scale. Neighborhood radii (30 m, 100-300 m) are
applied in this plane, not in degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mvbridge.geo import EARTH_RADIUS_M

_EPS = 1e-12


def _as_float_arrays(*values):
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    return arrays


@dataclass(frozen=True)
class MetricFrame:
    """Azimuthal equidistant projection about (lat0, lon0), in meters."""

    lat0: float
    lon0: float
    radius_m: float = EARTH_RADIUS_M

    def __post_init__(self) -> None:
        if not math.isfinite(self.lat0) or not math.isfinite(self.lon0):
            raise ValueError("MetricFrame origin must be finite lat/lon")
        if not -90.0 <= self.lat0 <= 90.0:
            raise ValueError(f"latitude out of range: {self.lat0}")
        if not -180.0 <= self.lon0 <= 180.0:
            raise ValueError(f"longitude out of range: {self.lon0}")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("radius_m must be a positive finite number")

    @classmethod
    def from_points(cls, lats, lons, *, radius_m: float = EARTH_RADIUS_M) -> "MetricFrame":
        """Center the frame on the finite-coordinate centroid of the points."""
        lat_arr = np.asarray(lats, dtype=np.float64)
        lon_arr = np.asarray(lons, dtype=np.float64)
        if lat_arr.shape != lon_arr.shape:
            raise ValueError("lats and lons must have the same shape")
        ok = np.isfinite(lat_arr) & np.isfinite(lon_arr)
        if not np.any(ok):
            raise ValueError("need at least one finite lat/lon to place the frame")
        return cls(float(np.mean(lat_arr[ok])), float(np.mean(lon_arr[ok])), radius_m=radius_m)

    def to_xy(self, lats, lons) -> tuple[np.ndarray, np.ndarray]:
        """Forward project geographic degrees to local meters (x east, y north)."""
        lat, lon = _as_float_arrays(lats, lons)
        phi = np.radians(lat)
        lam = np.radians(lon)
        phi0 = math.radians(self.lat0)
        lam0 = math.radians(self.lon0)
        dlam = lam - lam0
        cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dlam)
        cos_c = np.clip(cos_c, -1.0, 1.0)
        c = np.arccos(cos_c)
        sin_c = np.sin(c)
        k = np.ones_like(c)
        ok = sin_c > _EPS
        k[ok] = c[ok] / sin_c[ok]
        x = self.radius_m * k * np.cos(phi) * np.sin(dlam)
        y = self.radius_m * k * (
            np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlam)
        )
        x = np.where(np.isfinite(lat) & np.isfinite(lon), x, np.nan)
        y = np.where(np.isfinite(lat) & np.isfinite(lon), y, np.nan)
        if np.ndim(lats) == 0 and np.ndim(lons) == 0:
            return np.asarray(x).reshape([]), np.asarray(y).reshape([])
        return x, y

    def from_xy(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        """Inverse project local meters back to geographic degrees."""
        x, y = _as_float_arrays(xs, ys)
        rho = np.hypot(x, y)
        phi0 = math.radians(self.lat0)
        lam0 = math.radians(self.lon0)
        lat = np.full_like(x, self.lat0)
        lon = np.full_like(x, self.lon0)
        nz = rho > _EPS
        c = rho[nz] / self.radius_m
        sin_c = np.sin(c)
        cos_c = np.cos(c)
        phi = np.arcsin(
            np.clip(cos_c * math.sin(phi0) + y[nz] * sin_c * math.cos(phi0) / np.maximum(rho[nz], _EPS), -1.0, 1.0)
        )
        lam = lam0 + np.arctan2(
            x[nz] * sin_c,
            rho[nz] * math.cos(phi0) * cos_c - y[nz] * math.sin(phi0) * sin_c,
        )
        lat[nz] = np.degrees(phi)
        lon[nz] = np.degrees(lam)
        lon = (lon + 180.0) % 360.0 - 180.0
        lat = np.where(np.isfinite(x) & np.isfinite(y), lat, np.nan)
        lon = np.where(np.isfinite(x) & np.isfinite(y), lon, np.nan)
        if np.ndim(xs) == 0 and np.ndim(ys) == 0:
            return np.asarray(lat).reshape([]), np.asarray(lon).reshape([])
        return lat, lon

    def distance_m(self, lat1, lon1, lat2, lon2) -> np.ndarray:
        """Euclidean distance in the local metric plane, in meters."""
        x1, y1 = self.to_xy(lat1, lon1)
        x2, y2 = self.to_xy(lat2, lon2)
        return np.hypot(x2 - x1, y2 - y1)

"""Small geographic helpers. Distances are haversine meters on a sphere."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_DEGREE_LAT = 2.0 * math.pi * EARTH_RADIUS_M / 360.0


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine distance in meters. Inputs may be scalars or numpy arrays."""
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_M * c


def valid_lat_lon(lat: float, lon: float) -> bool:
    return math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def buffer_degrees(lat: float, buffer_m: float) -> tuple[float, float]:
    """Axis-aligned degree padding that contains a `buffer_m` disk at this latitude."""
    dlat = buffer_m / METERS_PER_DEGREE_LAT
    cos_lat = max(0.2, abs(math.cos(math.radians(lat))))
    dlon = buffer_m / (METERS_PER_DEGREE_LAT * cos_lat)
    return dlat, dlon


def point_in_expanded_bbox(
    lat: float,
    lon: float,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    buffer_m: float,
) -> bool:
    dlat, dlon = buffer_degrees(lat, buffer_m)
    return (min_lat - dlat) <= lat <= (max_lat + dlat) and (min_lon - dlon) <= lon <= (max_lon + dlon)


def _parse_one_day(text: str) -> date | None:
    text = text.strip()
    if not text:
        return None
    if len(text) == 4 and text.isdigit():
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text[:26].replace("Z", ""), fmt.replace("Z", "")).date()
        except ValueError:
            continue
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def parse_date_span(value: str | None) -> tuple[date, date] | None:
    """Parse a day, a month, a year, or an explicit start/end pair.

    A bare year covers that calendar year. That matches sources that only
    publish a year for a SAGIN example event.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    if "/" in text and text.count("-") >= 2:
        # Already an ISO date; do not treat the slash as a range separator.
        pass
    for separator in ("..", " to "):
        if separator in text:
            left, right = text.split(separator, 1)
            start = parse_date_span(left)
            end = parse_date_span(right)
            if start and end:
                return start[0], end[1]
    if len(text) == 4 and text.isdigit():
        year = int(text)
        return date(year, 1, 1), date(year, 12, 31)
    if len(text) == 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit():
        year = int(text[:4])
        month = int(text[5:7])
        if not 1 <= month <= 12:
            return None
        if month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, month + 1, 1) - timedelta(days=1)
        return date(year, month, 1), end
    day = _parse_one_day(text)
    if day is None:
        return None
    return day, day


def spans_overlap(left: tuple[date, date], right: tuple[date, date], window_days: int) -> bool:
    """True when the two inclusive spans meet, allowing `window_days` of slack."""
    pad = timedelta(days=int(window_days))
    return left[0] <= right[1] + pad and right[0] <= left[1] + pad

"""Metric-plane neighborhood index with leave-one-building-out queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cell(coord: np.ndarray, size: float) -> np.ndarray:
    return np.floor(coord / size).astype(np.int64)


@dataclass
class NeighborhoodIndex:
    """Grid index over local (x, y) meters.

    Queries always accept an ``exclude_ids`` set. Callers must pass the focal
    building id so that a building never enters its own neighborhood
    statistics (leave-one-building-out).
    """

    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    cell_size_m: float = 50.0

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.ids = np.asarray(self.ids)
        if not (self.x.shape == self.y.shape == self.ids.shape):
            raise ValueError("x, y, and ids must have the same shape")
        if self.x.ndim != 1:
            raise ValueError("x, y, and ids must be one-dimensional")
        if self.cell_size_m <= 0 or not np.isfinite(self.cell_size_m):
            raise ValueError("cell_size_m must be a positive finite number")
        self._valid = np.isfinite(self.x) & np.isfinite(self.y)
        self._grid: dict[tuple[int, int], np.ndarray] = {}
        if not np.any(self._valid):
            return
        ix = _cell(self.x[self._valid], self.cell_size_m)
        iy = _cell(self.y[self._valid], self.cell_size_m)
        positions = np.nonzero(self._valid)[0]
        for cell, pos in zip(zip(ix.tolist(), iy.tolist()), positions.tolist()):
            self._grid.setdefault(cell, []).append(pos)
        self._grid = {key: np.asarray(val, dtype=np.int64) for key, val in self._grid.items()}

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def _candidate_indices(self, x0: float, y0: float, radius_m: float) -> np.ndarray:
        if not np.isfinite(x0) or not np.isfinite(y0) or not np.isfinite(radius_m):
            return np.empty(0, dtype=np.int64)
        pad = int(np.ceil(radius_m / self.cell_size_m)) + 1
        cx = int(np.floor(x0 / self.cell_size_m))
        cy = int(np.floor(y0 / self.cell_size_m))
        chunks = []
        for dx in range(-pad, pad + 1):
            for dy in range(-pad, pad + 1):
                hits = self._grid.get((cx + dx, cy + dy))
                if hits is not None:
                    chunks.append(hits)
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)

    def query_radius(
        self,
        x0: float,
        y0: float,
        radius_m: float,
        *,
        exclude_ids: set | frozenset | None = None,
        r_min_m: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances_m) with r_min_m < d <= radius_m, excluding ids.

        The default r_min_m=0 keeps coincident *other* buildings, but the
        focal id still has to be listed in ``exclude_ids``.
        """
        if radius_m < 0:
            raise ValueError("radius_m must be non-negative")
        if r_min_m < 0:
            raise ValueError("r_min_m must be non-negative")
        cand = self._candidate_indices(x0, y0, radius_m)
        if cand.size == 0:
            return self.ids[:0], np.empty(0, dtype=np.float64)
        dist = np.hypot(self.x[cand] - x0, self.y[cand] - y0)
        # A zero inner radius must keep coincident *other* buildings. The
        # focal building is removed only through exclude_ids, never by
        # dropping distance 0. An annulus uses a strict inner bound so a
        # point on the inner radius is not counted in two windows.
        if r_min_m == 0.0:
            mask = np.isfinite(dist) & (dist <= radius_m)
        else:
            mask = np.isfinite(dist) & (dist <= radius_m) & (dist > r_min_m)
        if exclude_ids:
            # Compare as strings so fixture ints and official uids can mix safely.
            excluded = {str(item) for item in exclude_ids}
            keep = np.fromiter((str(item) not in excluded for item in self.ids[cand]), dtype=bool, count=cand.size)
            mask &= keep
        hit = cand[mask]
        return self.ids[hit], dist[mask]

    def query_disk(
        self,
        x0: float,
        y0: float,
        radius_m: float,
        *,
        exclude_ids: set | frozenset | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.query_radius(x0, y0, radius_m, exclude_ids=exclude_ids, r_min_m=0.0)

    def query_annulus(
        self,
        x0: float,
        y0: float,
        inner_m: float,
        outer_m: float,
        *,
        exclude_ids: set | frozenset | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if inner_m >= outer_m:
            raise ValueError("annulus inner radius must be smaller than the outer radius")
        return self.query_radius(x0, y0, outer_m, exclude_ids=exclude_ids, r_min_m=inner_m)

    def nearest(
        self,
        x0: float,
        y0: float,
        *,
        exclude_ids: set | frozenset | None = None,
        mask: np.ndarray | None = None,
        max_radius_m: float | None = None,
    ) -> tuple[object | None, float]:
        """Return (id, distance_m) of the nearest kept point, or (None, inf)."""
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.ids.shape:
                raise ValueError("mask must align with the index ids")
        valid = self._valid if mask is None else self._valid & mask
        if exclude_ids:
            excluded = {str(item) for item in exclude_ids}
            keep = np.fromiter((str(item) not in excluded for item in self.ids), dtype=bool, count=self.ids.size)
            valid = valid & keep
        if not np.any(valid):
            return None, float("inf")
        dist = np.hypot(self.x - x0, self.y - y0)
        dist = np.where(valid & np.isfinite(dist), dist, np.inf)
        if max_radius_m is not None:
            dist = np.where(dist <= max_radius_m, dist, np.inf)
        j = int(np.argmin(dist))
        best = float(dist[j])
        if not np.isfinite(best):
            return None, float("inf")
        return self.ids[j], best

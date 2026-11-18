"""Paired spatial-block intervals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class PairedBlockInterval:
    mean: float
    lower: float
    upper: float
    level: float
    n_blocks: int
    n_replicates: int
    rng_seed: int
    resample_sha256: str
    block_ids: tuple[str, ...]
    block_values: tuple[float, ...]

    @property
    def lower_bound_positive(self) -> bool:
        return self.lower > 0.0

    @property
    def upper_bound_negative(self) -> bool:
        return self.upper < 0.0

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def _block_means(
    block_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(block_ids) != len(values):
        raise ValueError("block_ids and values must have the same length")
    if len(block_ids) == 0:
        raise ValueError("cannot form a block interval from an empty sample")
    unique = np.array(sorted(set(map(str, block_ids))), dtype=object)
    means = np.empty(len(unique), dtype=np.float64)
    for i, block in enumerate(unique):
        mask = np.asarray(block_ids, dtype=object) == block
        if not np.any(mask):
            raise ValueError(f"block {block!r} has no rows")
        means[i] = float(np.mean(np.asarray(values)[mask], dtype=np.float64))
    return unique, means


def paired_block_interval(
    block_ids: np.ndarray | list[str],
    values_a: np.ndarray,
    values_b: np.ndarray | None = None,
    *,
    n_replicates: int = 1000,
    level: float = 0.95,
    rng_seed: int = 0,
    resample_indices: np.ndarray | None = None,
) -> PairedBlockInterval:
    """Percentile interval for a paired block-mean contrast.

    If ``values_b`` is supplied, the contrast is ``mean(a-b)`` at the block
    level. If omitted, ``values_a`` is already the per-row contrast. The same
    ``rng_seed`` and block id set yield the same resample draws, so two seeds
    that share buildings and blocks can share a bootstrap design.

    ``resample_indices`` may be passed in to force an exact shared design.
    """
    ids = np.asarray(block_ids, dtype=object)
    left = np.asarray(values_a, dtype=np.float64)
    if values_b is None:
        contrast = left
    else:
        right = np.asarray(values_b, dtype=np.float64)
        if right.shape != left.shape:
            raise ValueError("paired values must have the same shape")
        contrast = left - right
    unique, means = _block_means(ids, contrast)
    n_blocks = len(means)
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if resample_indices is None:
        rng = np.random.default_rng(int(rng_seed))
        draws = rng.integers(0, n_blocks, size=(int(n_replicates), n_blocks))
    else:
        draws = np.asarray(resample_indices)
        if draws.ndim != 2 or draws.shape[1] != n_blocks:
            raise ValueError(
                f"resample_indices must have shape (n_replicates, {n_blocks})"
            )
        n_replicates = int(draws.shape[0])
    replicates = means[draws].mean(axis=1)
    alpha = 1.0 - float(level)
    lower = float(np.quantile(replicates, alpha / 2.0, method="linear"))
    upper = float(np.quantile(replicates, 1.0 - alpha / 2.0, method="linear"))
    payload = np.ascontiguousarray(draws, dtype=np.int64).tobytes() + repr(draws.shape).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return PairedBlockInterval(
        mean=float(means.mean()),
        lower=lower,
        upper=upper,
        level=float(level),
        n_blocks=n_blocks,
        n_replicates=int(n_replicates),
        rng_seed=int(rng_seed),
        resample_sha256=digest,
        block_ids=tuple(str(block) for block in unique),
        block_values=tuple(float(value) for value in means),
    )

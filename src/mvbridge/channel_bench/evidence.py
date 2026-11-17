"""Intact-report evidence ratios under a historically locked mixture."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from mvbridge.channel_bench.intervals import PairedBlockInterval
from mvbridge.channel_bench.locks import (
    DAMAGE_CLASSES,
    INTACT_CLASS,
    PROPOSED_BOOTSTRAP_REPLICATES,
    PROPOSED_DIRICHLET_ALPHA,
    PROPOSED_FIXED_DAMAGE_MIXTURE,
    PROPOSED_INTERVAL_LEVEL,
)
from mvbridge.channel_bench.schema import BuildingAnalysisRow, class_index, imprint_bin, require_known_grade
from mvbridge.channel_bench.tables import estimate_q_tables, normalize_q_rows


def _mixture_vector(
    mixture: Mapping[str, float] | None,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
) -> np.ndarray:
    source = PROPOSED_FIXED_DAMAGE_MIXTURE if mixture is None else mixture
    vec = np.zeros(len(grades), dtype=np.float64)
    for name, weight in source.items():
        require_known_grade(name)
        if name == INTACT_CLASS:
            raise ValueError("the locked damaged mixture cannot include the intact class")
        if weight < 0:
            raise ValueError("mixture weights must be non-negative")
        vec[class_index(name)] = float(weight)
    if vec.sum() <= 0:
        raise ValueError("mixture has no mass")
    return vec / vec.sum()


def intact_evidence_log_ratio(
    q_table: np.ndarray,
    *,
    mixture: Mapping[str, float] | None = None,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
    intact_class: str = INTACT_CLASS,
) -> float:
    """Log ratio of intact-report probability vs a locked damaged mixture.

    ``q_table`` is ``Q(s|y)`` with rows ``Y`` and columns ``S`` in ``grades``
    order. The table is row-normalized before the readout.
    """
    table = normalize_q_rows(np.asarray(q_table, dtype=np.float64))
    if table.shape != (len(grades), len(grades)):
        raise ValueError(
            f"expected Q shape {(len(grades), len(grades))}, got {table.shape}"
        )
    intact = class_index(intact_class)
    num = float(table[intact, intact])
    mix = _mixture_vector(mixture, grades)
    den = float(table[:, intact] @ mix)
    if num <= 0.0 or den <= 0.0:
        raise ValueError("intact evidence ratio is undefined for a zero cell")
    return float(np.log(num) - np.log(den))


def intact_report_rates(
    q_table: np.ndarray,
    *,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
    intact_class: str = INTACT_CLASS,
) -> dict[str, float]:
    """Intact-report probability from every true-Y row, including damaged rows."""
    table = normalize_q_rows(np.asarray(q_table, dtype=np.float64))
    intact = class_index(intact_class)
    return {name: float(table[class_index(name), intact]) for name in grades}


def evidence_ratio_by_bin(
    tables: Mapping[str, np.ndarray],
    pooled: np.ndarray,
    bin_keys: list[str],
    *,
    mixture: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Read the locked intact evidence ratio at each pre-specified bin key."""
    out: dict[str, float] = {}
    for key in bin_keys:
        table = tables.get(key, pooled)
        out[key] = intact_evidence_log_ratio(table, mixture=mixture)
    return out


def _ratios_from_rows(
    rows: Sequence[BuildingAnalysisRow],
    *,
    mixture: Mapping[str, float] | None,
    alpha: float,
    grades: tuple[str, ...],
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot estimate evidence ratios from an empty sample")
    keys = np.array([f"t{imprint_bin(row.imprint_score)}" for row in rows], dtype=object)
    tables, pooled = estimate_q_tables(
        np.array([row.y for row in rows], dtype=object),
        np.array([row.s for row in rows], dtype=object),
        keys,
        grades=grades,
        alpha=alpha,
    )
    bin_keys = sorted(set(keys.tolist()), key=lambda name: int(str(name)[1:]))
    return evidence_ratio_by_bin(tables, pooled, bin_keys, mixture=mixture)


def bootstrap_evidence_ratio_shift(
    rows: Sequence[BuildingAnalysisRow],
    *,
    mixture: Mapping[str, float] | None = None,
    alpha: float = PROPOSED_DIRICHLET_ALPHA,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
    n_replicates: int = PROPOSED_BOOTSTRAP_REPLICATES,
    level: float = PROPOSED_INTERVAL_LEVEL,
    rng_seed: int = 0,
) -> tuple[dict[str, float], PairedBlockInterval]:
    """Observed bin ratios plus a block-bootstrap interval for LR(low) - LR(high).

    A positive lower bound means the intact evidence log-ratio falls along the
    pre-fixed imprint gradient. An interval that contains 0 is a stable ratio.
    """
    observed = _ratios_from_rows(rows, mixture=mixture, alpha=alpha, grades=grades)
    keys = sorted(observed, key=lambda name: int(name[1:]))
    if len(keys) < 2:
        raise ValueError("need at least two imprint bins to test an evidence-ratio shift")
    low_key, high_key = keys[0], keys[-1]
    groups: dict[str, list[BuildingAnalysisRow]] = {}
    for row in rows:
        groups.setdefault(row.block_id, []).append(row)
    block_ids = np.array(sorted(groups), dtype=object)
    rng = np.random.default_rng(int(rng_seed))
    draws = rng.integers(0, len(block_ids), size=(int(n_replicates), len(block_ids)))
    shifts: list[float] = []
    valid_draws: list[np.ndarray] = []
    for draw in draws:
        sample = [row for index in draw for row in groups[str(block_ids[index])]]
        try:
            ratios = _ratios_from_rows(sample, mixture=mixture, alpha=alpha, grades=grades)
            if low_key not in ratios or high_key not in ratios:
                continue
            shifts.append(float(ratios[low_key] - ratios[high_key]))
            valid_draws.append(np.asarray(draw))
        except ValueError:
            continue
    if len(shifts) < max(50, int(0.5 * n_replicates)):
        raise ValueError("too few bootstrap replicates produced an evidence-ratio shift")
    values = np.asarray(shifts, dtype=np.float64)
    alpha_tail = 1.0 - float(level)
    interval = PairedBlockInterval(
        mean=float(observed[low_key] - observed[high_key]),
        lower=float(np.quantile(values, alpha_tail / 2.0, method="linear")),
        upper=float(np.quantile(values, 1.0 - alpha_tail / 2.0, method="linear")),
        level=float(level),
        n_blocks=len(block_ids),
        n_replicates=len(shifts),
        rng_seed=int(rng_seed),
        resample_sha256="",
        block_ids=tuple(str(block) for block in block_ids),
        block_values=tuple(),
    )
    return observed, interval


def classify_evidence_ratio_trend(shift: PairedBlockInterval) -> str:
    """Locked direction: a positive shift means the log-ratio falls toward 0."""
    if shift.lower_bound_positive:
        return "decreasing_toward_zero_or_below"
    if shift.lower <= 0.0 <= shift.upper:
        return "stable"
    return "other"

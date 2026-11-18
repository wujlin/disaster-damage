"""Row-normalized observation tables and Bayes updates."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from mvbridge.channel_bench.locks import DAMAGE_CLASSES, PROPOSED_DIRICHLET_ALPHA
from mvbridge.channel_bench.schema import class_index, require_known_grade


def normalize_q_rows(table: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Normalize the last axis so each ``Y`` row is a distribution over ``S``.

    Zero rows and negative entries are rejected. The result is at least as
    many dimensions as the input and sums to 1 along the last axis.
    """
    array = np.asarray(table, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("Q table must have at least one dimension")
    if np.any(~np.isfinite(array)):
        raise ValueError("Q table contains non-finite values")
    if np.any(array < -atol):
        raise ValueError("Q table contains negative likelihoods")
    array = np.clip(array, 0.0, None)
    totals = array.sum(axis=-1, keepdims=True)
    if np.any(totals <= atol):
        raise ValueError("Q table has a zero row and cannot be normalized")
    return array / totals


def posterior_from_likelihood(
    prior: np.ndarray | Mapping[str, float],
    likelihood_by_grade: np.ndarray | Mapping[str, float],
    *,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
) -> np.ndarray:
    """Return ``P(Y|S,Z)`` from a prior and class-conditional likelihoods.

    If every class likelihood is equal, the posterior equals the prior.
    """
    prior_vec = _as_grade_vector(prior, grades)
    like_vec = _as_grade_vector(likelihood_by_grade, grades)
    if prior_vec.size != len(grades) or like_vec.size != len(grades):
        raise ValueError("prior and likelihood must cover every grade")
    if np.any(prior_vec < 0) or not np.isfinite(prior_vec).all():
        raise ValueError("prior must be a finite non-negative vector")
    if prior_vec.sum() <= 0:
        raise ValueError("prior must have positive mass")
    prior_vec = prior_vec / prior_vec.sum()
    if np.any(like_vec < 0) or not np.isfinite(like_vec).all():
        raise ValueError("likelihoods must be finite and non-negative")
    unnormalized = prior_vec * like_vec
    total = float(unnormalized.sum())
    if total <= 0:
        raise ValueError("posterior has no mass under the supplied likelihoods")
    return unnormalized / total


def equal_likelihood_leaves_prior_unchanged(
    prior: np.ndarray | Mapping[str, float],
    common_likelihood: float = 1.0,
    *,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
    atol: float = 1e-12,
) -> bool:
    """Return True when equal class likelihoods leave the prior unchanged."""
    prior_vec = _as_grade_vector(prior, grades)
    prior_vec = prior_vec / prior_vec.sum()
    likelihood = np.full(len(grades), float(common_likelihood), dtype=np.float64)
    posterior = posterior_from_likelihood(prior_vec, likelihood, grades=grades)
    return bool(np.allclose(posterior, prior_vec, atol=atol, rtol=0.0))


def estimate_q_tables(
    y: np.ndarray,
    s: np.ndarray,
    keys: np.ndarray,
    *,
    grades: tuple[str, ...] = DAMAGE_CLASSES,
    alpha: float = PROPOSED_DIRICHLET_ALPHA,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Estimate a row-normalized Q table per feature key, plus a pooled table.

    ``alpha`` is Dirichlet smoothing (proposed, not an official freeze).
    """
    n = len(grades)
    if len(y) != len(s) or len(y) != len(keys):
        raise ValueError("y, s, and keys must have the same length")
    counts: dict[str, np.ndarray] = {}
    pooled = np.full((n, n), float(alpha), dtype=np.float64)
    for grade, report, key in zip(y, s, keys, strict=True):
        y_index = class_index(str(grade))
        s_index = class_index(str(report))
        label = str(key)
        if label not in counts:
            counts[label] = np.full((n, n), float(alpha), dtype=np.float64)
        counts[label][y_index, s_index] += 1.0
        pooled[y_index, s_index] += 1.0
    tables = {key: normalize_q_rows(table) for key, table in counts.items()}
    return tables, normalize_q_rows(pooled)


def lookup_q_row(
    tables: Mapping[str, np.ndarray],
    pooled: np.ndarray,
    key: str,
    y: str,
) -> np.ndarray:
    table = tables.get(str(key), pooled)
    return np.asarray(table[class_index(y)], dtype=np.float64)


def log_score_for_rows(
    tables: Mapping[str, np.ndarray],
    pooled: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    scores = np.empty(len(y), dtype=np.float64)
    for i, (grade, report, key) in enumerate(zip(y, s, keys, strict=True)):
        row = lookup_q_row(tables, pooled, str(key), str(grade))
        prob = float(row[class_index(str(report))])
        if prob <= 0.0:
            scores[i] = -np.inf
        else:
            scores[i] = float(np.log(prob))
    return scores


def _as_grade_vector(
    values: np.ndarray | Mapping[str, float],
    grades: tuple[str, ...],
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = [name for name in grades if name not in values]
        if missing:
            raise ValueError(f"missing grades in mapping: {missing}")
        for name in values:
            require_known_grade(name)
        return np.asarray([values[name] for name in grades], dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(grades),):
        raise ValueError(f"expected a vector of length {len(grades)}, got {array.shape}")
    return array

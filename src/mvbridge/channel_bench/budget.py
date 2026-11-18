"""Fixed-budget audit hit rates on a locked eligible list.

In this bench a hit is ``S != Y``, not official U (``U_E or U_F``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvbridge.channel_bench.locks import PROPOSED_AUDIT_BUDGET


@dataclass(frozen=True)
class BudgetHitRate:
    hit_rate: float
    n_audited: int
    n_hits: int
    n_eligible: int
    budget: float
    ranking_name: str
    # Explicit so reports cannot be read as official U (U_E or U_F).
    hit_definition: str = "S_neq_Y_not_official_U"


def _audit_count(n_eligible: int, budget: float) -> int:
    if n_eligible < 0:
        raise ValueError("n_eligible must be non-negative")
    if not 0.0 < budget <= 1.0:
        raise ValueError("budget must be in (0, 1]")
    if n_eligible == 0:
        return 0
    return max(1, int(np.floor(budget * n_eligible)))


def budget_hit_rate(
    scores: np.ndarray,
    hits: np.ndarray,
    *,
    building_uids: np.ndarray | list[str] | None = None,
    budget: float = PROPOSED_AUDIT_BUDGET,
    ranking_name: str = "unnamed",
) -> BudgetHitRate:
    """Hit rate in the top ``floor(budget * N)`` buildings, ties broken by uid.

    A hit is a precomputed boolean. In this bench the caller must pass
    ``S != Y`` (frozen satellite grade differs from the aligned ground
    grade). That disagreement is not official U (``U_E or U_F``). Do not
    report or interpret this hit rate as official U. Scores are ranked
    descending. The eligible list must already be filtered by the caller;
    this function does not drop rows.
    """
    score_vec = np.asarray(scores, dtype=np.float64)
    hit_vec = np.asarray(hits, dtype=bool)
    if score_vec.shape != hit_vec.shape or score_vec.ndim != 1:
        raise ValueError("scores and hits must be 1-D arrays of the same length")
    if np.any(~np.isfinite(score_vec)):
        raise ValueError("ranking scores must be finite")
    n_eligible = int(score_vec.size)
    if building_uids is None:
        uids = np.array([f"{index:08d}" for index in range(n_eligible)])
    else:
        uids = np.asarray(list(map(str, building_uids)))
        if uids.shape != score_vec.shape:
            raise ValueError("building_uids must align with scores")
        if len(set(uids.tolist())) != n_eligible:
            raise ValueError("building_uids must be unique on the eligible list")
    n_audited = _audit_count(n_eligible, budget)
    if n_audited == 0:
        return BudgetHitRate(
            hit_rate=float("nan"),
            n_audited=0,
            n_hits=0,
            n_eligible=0,
            budget=float(budget),
            ranking_name=ranking_name,
        )
    # Rank by score descending, then uid ascending for a stable, audit-safe tie break.
    order = np.lexsort((uids, -score_vec))
    chosen = hit_vec[order[:n_audited]]
    n_hits = int(chosen.sum())
    return BudgetHitRate(
        hit_rate=float(n_hits / n_audited),
        n_audited=n_audited,
        n_hits=n_hits,
        n_eligible=n_eligible,
        budget=float(budget),
        ranking_name=ranking_name,
    )


def paired_budget_increment(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    hits: np.ndarray,
    *,
    building_uids: np.ndarray | list[str] | None = None,
    budget: float = PROPOSED_AUDIT_BUDGET,
    name_a: str = "a",
    name_b: str = "b",
) -> dict[str, float | int | str | BudgetHitRate]:
    """Compare two rankings on the same eligible list and the same budget."""
    left = budget_hit_rate(
        scores_a, hits, building_uids=building_uids, budget=budget, ranking_name=name_a
    )
    right = budget_hit_rate(
        scores_b, hits, building_uids=building_uids, budget=budget, ranking_name=name_b
    )
    return {
        "a": left,
        "b": right,
        "increment_a_minus_b": float(left.hit_rate - right.hit_rate),
        "budget": float(budget),
        "n_eligible": left.n_eligible,
    }

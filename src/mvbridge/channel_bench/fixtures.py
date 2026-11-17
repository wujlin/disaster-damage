"""Synthetic fixture families for identification checks.

These generators never read official Michael outputs. They exist to test that
the analysis recovers prior-only hotspots versus signed Q-change, and that
equal class likelihoods leave a prior unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvbridge.channel_bench.locks import DAMAGE_CLASSES
from mvbridge.channel_bench.schema import BuildingAnalysisRow, imprint_bin
from mvbridge.channel_bench.tables import normalize_q_rows, posterior_from_likelihood


def _rank_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _rank_one_minus_max(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs.max(axis=1)


def prior_by_imprint(imprint: np.ndarray) -> np.ndarray:
    """Damage prior that shifts sharply toward severe grades as imprint grows."""
    low = np.array([0.70, 0.18, 0.08, 0.04], dtype=np.float64)
    high = np.array([0.12, 0.18, 0.30, 0.40], dtype=np.float64)
    weight = np.clip(np.asarray(imprint, dtype=np.float64), 0.0, 1.0)[:, None]
    prior = (1.0 - weight) * low + weight * high
    return prior / prior.sum(axis=1, keepdims=True)


def prior_change_q_table() -> np.ndarray:
    """A reporting table that does not depend on traces."""
    # Rows are Y, columns are S, in DAMAGE_CLASSES order.
    raw = np.array(
        [
            [0.82, 0.12, 0.04, 0.02],
            [0.28, 0.52, 0.14, 0.06],
            [0.10, 0.22, 0.52, 0.16],
            [0.04, 0.08, 0.22, 0.66],
        ],
        dtype=np.float64,
    )
    return normalize_q_rows(raw)


def q_change_table(imprint: float, visibility: str = "clear") -> np.ndarray:
    """Reporting table whose intact row and intact column move with imprint.

    At low imprint, an intact report supports undamaged (log-ratio well above
    0). At high imprint the intact-report likelihoods converge, so the
    log-ratio moves toward 0. A degraded visibility slice is a small row-mix
    toward uniform, used only as a C factor, not as the imprint gradient.
    """
    # Piecewise-constant on the locked imprint bins so Q1 is well specified
    # and a C-only Q0 is a visibly worse compromise.
    low = np.array(
        [
            [0.92, 0.05, 0.02, 0.01],
            [0.05, 0.80, 0.10, 0.05],
            [0.02, 0.10, 0.78, 0.10],
            [0.01, 0.04, 0.15, 0.80],
        ],
        dtype=np.float64,
    )
    mid = np.array(
        [
            [0.70, 0.16, 0.09, 0.05],
            [0.22, 0.56, 0.16, 0.06],
            [0.18, 0.16, 0.52, 0.14],
            [0.14, 0.12, 0.20, 0.54],
        ],
        dtype=np.float64,
    )
    high = np.array(
        [
            [0.40, 0.25, 0.20, 0.15],
            [0.38, 0.32, 0.18, 0.12],
            [0.36, 0.22, 0.24, 0.18],
            [0.34, 0.20, 0.22, 0.24],
        ],
        dtype=np.float64,
    )
    tables = (low, mid, high)
    table = tables[imprint_bin(float(np.clip(imprint, 0.0, 1.0)))]
    table = normalize_q_rows(table)
    if visibility != "clear":
        uniform = np.full_like(table, 1.0 / table.shape[-1])
        table = normalize_q_rows(0.85 * table + 0.15 * uniform)
    return table


@dataclass(frozen=True)
class EqualLikelihoodCase:
    prior: np.ndarray
    likelihood_by_grade: np.ndarray
    posterior: np.ndarray
    prior_unchanged: bool


def generate_equal_likelihood_case(
    prior: np.ndarray | None = None,
    common_likelihood: float = 0.25,
) -> EqualLikelihoodCase:
    """Fixture: identical class likelihoods leave the damage prior unchanged."""
    if prior is None:
        prior_vec = np.array([0.5, 0.25, 0.15, 0.10], dtype=np.float64)
    else:
        prior_vec = np.asarray(prior, dtype=np.float64)
    prior_vec = prior_vec / prior_vec.sum()
    likelihood = np.full(len(DAMAGE_CLASSES), float(common_likelihood), dtype=np.float64)
    posterior = posterior_from_likelihood(prior_vec, likelihood)
    unchanged = bool(np.allclose(posterior, prior_vec, atol=1e-12, rtol=0.0))
    return EqualLikelihoodCase(
        prior=prior_vec,
        likelihood_by_grade=likelihood,
        posterior=posterior,
        prior_unchanged=unchanged,
    )


@dataclass
class FixtureDataset:
    name: str
    family: str
    rows: list[BuildingAnalysisRow]
    event_id: str = "synthetic-event"
    seed_id: str = "1"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not row.index_building_alarm_excluded for row in self.rows):
            raise ValueError("fixtures must not put the index building alarm into T")

    def eligible_rows(self) -> list[BuildingAnalysisRow]:
        return [row for row in self.rows if row.eligible]

    def arrays(self) -> dict[str, np.ndarray]:
        rows = self.eligible_rows()
        return {
            "building_uid": np.array([row.building_uid for row in rows], dtype=object),
            "block_id": np.array([row.block_id for row in rows], dtype=object),
            "y": np.array([row.y for row in rows], dtype=object),
            "s": np.array([row.s for row in rows], dtype=object),
            "visibility_bin": np.array([row.visibility_bin for row in rows], dtype=object),
            "imprint_score": np.array([row.imprint_score for row in rows], dtype=np.float64),
            "imprint_bin": np.array([imprint_bin(row.imprint_score) for row in rows], dtype=np.int64),
            "mismatch": np.array([row.s != row.y for row in rows], dtype=bool),
        }


def _sample_rows(
    *,
    name: str,
    family: str,
    q_lookup,
    n_blocks: int,
    buildings_per_block: int,
    rng: np.random.Generator,
    event_id: str,
    seed_id: str,
    degraded_fraction: float,
    notes: tuple[str, ...],
) -> FixtureDataset:
    n = n_blocks * buildings_per_block
    block_ids = np.repeat(np.arange(n_blocks), buildings_per_block)
    # Block-level imprint field, pre-specified without looking at Y.
    block_imprint = np.linspace(0.05, 0.95, n_blocks)
    imprint = block_imprint[block_ids]
    visibility = np.array(
        ["degraded" if u < degraded_fraction else "clear" for u in rng.random(n)],
        dtype=object,
    )
    priors = prior_by_imprint(imprint)
    y_index = np.array(
        [rng.choice(len(DAMAGE_CLASSES), p=priors[i]) for i in range(n)],
        dtype=np.int64,
    )
    s_index = np.empty(n, dtype=np.int64)
    q_used = np.empty((n, len(DAMAGE_CLASSES)), dtype=np.float64)
    for i in range(n):
        table = q_lookup(float(imprint[i]), str(visibility[i]))
        q_used[i] = table[y_index[i]]
        s_index[i] = int(rng.choice(len(DAMAGE_CLASSES), p=table[y_index[i]]))

    # Rankings: Group-3 tracks imprint / public-geo proxies. Visual mismatch
    # is informative only when Q actually varies with imprint.
    neighborhood = imprint + 0.02 * rng.normal(size=n)
    public_geo = 0.5 * imprint + 0.05 * rng.normal(size=n)
    combined = 0.7 * neighborhood + 0.3 * public_geo
    # A "visual" score that looks at residual report surprise under a T-blind Q.
    pooled = q_lookup(0.5, "clear")
    visual = np.empty(n, dtype=np.float64)
    for i in range(n):
        p_s = float(pooled[y_index[i], s_index[i]])
        visual[i] = -np.log(max(p_s, 1e-8)) + 0.05 * rng.normal()
    # In the prior-change family Q is constant, so this visual score is mostly
    # a noisy function of (Y,S) and should not beat combined Group 3 much.
    plus_visual = combined + 0.15 * (visual - visual.mean()) / (visual.std() + 1e-8)

    # Entropy / 1-max p from the true prior (a hesitation baseline that knows
    # the local damage rate, not the channel). Higher = more hesitation.
    entropy = _rank_entropy(priors)
    one_minus_max = _rank_one_minus_max(priors)

    rows: list[BuildingAnalysisRow] = []
    for i in range(n):
        uid = f"{event_id}-{seed_id}-b{i:05d}"
        rows.append(
            BuildingAnalysisRow(
                building_uid=uid,
                event_id=event_id,
                seed_id=str(seed_id),
                block_id=f"block-{int(block_ids[i]):03d}",
                y=DAMAGE_CLASSES[int(y_index[i])],
                s=DAMAGE_CLASSES[int(s_index[i])],
                visibility_bin=str(visibility[i]),
                imprint_score=float(imprint[i]),
                sat_alarm_density_30m=float(imprint[i]),
                sat_alarm_density_100_300m=float(0.8 * imprint[i]),
                dist_to_nearest_sat_collapsed=float(1.0 - imprint[i]),
                elevation=10.0,
                dist_to_water=float(2.0 - imprint[i]),
                rankings={
                    "group3_neighborhood_only": float(neighborhood[i]),
                    "group3_public_geo_only": float(public_geo[i]),
                    "group3_neighborhood_and_public_geo": float(combined[i]),
                    "ranking_visual_mismatch": float(visual[i]),
                    "group3_plus_frozen_visual_mismatch": float(plus_visual[i]),
                    "entropy": float(entropy[i]),
                    "one_minus_max_p": float(one_minus_max[i]),
                },
            )
        )
    return FixtureDataset(
        name=name,
        family=family,
        rows=rows,
        event_id=event_id,
        seed_id=str(seed_id),
        notes=notes,
    )


def generate_prior_change_fixture(
    *,
    n_blocks: int = 36,
    buildings_per_block: int = 40,
    rng_seed: int = 11,
    event_id: str = "synthetic-prior-change",
    seed_id: str = "1",
    degraded_fraction: float = 0.2,
) -> FixtureDataset:
    """Family 1: spatial priors change, the observation table does not."""
    table = prior_change_q_table()

    def q_lookup(imprint: float, visibility: str) -> np.ndarray:
        del imprint, visibility
        return table

    return _sample_rows(
        name="prior_change_only",
        family="prior_change",
        q_lookup=q_lookup,
        n_blocks=n_blocks,
        buildings_per_block=buildings_per_block,
        rng=np.random.default_rng(rng_seed),
        event_id=event_id,
        seed_id=seed_id,
        degraded_fraction=degraded_fraction,
        notes=(
            "Q is constant across imprint. Mismatch hotspots may follow the prior.",
            "The sheet must not claim channel information loss from this family.",
        ),
    )


def generate_q_change_fixture(
    *,
    n_blocks: int = 36,
    buildings_per_block: int = 40,
    rng_seed: int = 17,
    event_id: str = "synthetic-q-change",
    seed_id: str = "1",
    degraded_fraction: float = 0.2,
) -> FixtureDataset:
    """Family 2: the intact row of Q also changes along the imprint gradient."""

    def q_lookup(imprint: float, visibility: str) -> np.ndarray:
        return q_change_table(imprint, visibility)

    return _sample_rows(
        name="q_change_intact_row",
        family="q_change",
        q_lookup=q_lookup,
        n_blocks=n_blocks,
        buildings_per_block=buildings_per_block,
        rng=np.random.default_rng(rng_seed),
        event_id=event_id,
        seed_id=seed_id,
        degraded_fraction=degraded_fraction,
        notes=(
            "Intact-report evidence log-ratio should fall toward 0 as imprint grows.",
            "Both undamaged-row and damaged-row intact-report rates must be reported.",
        ),
    )


def true_q_for_row(row: BuildingAnalysisRow, family: str) -> np.ndarray:
    if family == "prior_change":
        return prior_change_q_table()
    if family == "q_change":
        return q_change_table(row.imprint_score, row.visibility_bin)
    raise ValueError(f"unknown fixture family {family!r}")


def ranking_matrix(dataset: FixtureDataset, name: str) -> np.ndarray:
    return np.array([float(row.rankings[name]) for row in dataset.eligible_rows()], dtype=np.float64)

"""Run the locked channel stress bench on an analysis table.

Official Michael dumps are not accepted as a special input type. Callers
pass in-memory building rows, which in this branch come from synthetic
fixtures only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from mvbridge.channel_bench.budget import BudgetHitRate, budget_hit_rate
from mvbridge.channel_bench.evidence import (
    bootstrap_evidence_ratio_shift,
    classify_evidence_ratio_trend,
    intact_report_rates,
)
from mvbridge.channel_bench.intervals import PairedBlockInterval, paired_block_interval
from mvbridge.channel_bench.locks import (
    ANALYSIS_LOCK,
    AnalysisLock,
    NegligibleGainBand,
)
from mvbridge.channel_bench.schema import BuildingAnalysisRow, imprint_bin
from mvbridge.channel_bench.support import SupportReport, support_report
from mvbridge.channel_bench.tables import estimate_q_tables, log_score_for_rows


CLAIM_CHANNEL_EVIDENCE_CHANGE = "channel_evidence_change"
CLAIM_REPORTING_CHANGES_WITH_ENVIRONMENT = "reporting_changes_with_environment"
CLAIM_PRIOR_EXPLANATION = "prior_only_no_channel_information_loss"
CLAIM_INCONCLUSIVE = "inconclusive"


def _feature_key(row: BuildingAnalysisRow, *, include_traces: bool) -> str:
    if include_traces:
        return f"{row.visibility_bin}|t{imprint_bin(row.imprint_score)}"
    return str(row.visibility_bin)


def _grade_weighted_mean(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    grades: tuple[str, ...],
) -> tuple[float, dict[str, float], tuple[str, ...]]:
    per_grade: dict[str, float] = {}
    used: list[str] = []
    dropped: list[str] = []
    for name in grades:
        mask = y == name
        if not np.any(mask):
            dropped.append(name)
            continue
        per_grade[name] = float(np.mean(scores[mask]))
        used.append(name)
    if not used:
        raise ValueError("no grade has evaluation support")
    weight = 1.0 / len(used)
    return float(sum(per_grade[name] * weight for name in used)), per_grade, tuple(dropped)


def leave_block_out_scores(
    rows: Sequence[BuildingAnalysisRow],
    *,
    include_traces: bool,
    alpha: float,
    grades: tuple[str, ...],
) -> np.ndarray:
    """Score each building from Q estimated on all other blocks."""
    block_ids = [row.block_id for row in rows]
    unique_blocks = sorted(set(block_ids))
    scores = np.empty(len(rows), dtype=np.float64)
    for block in unique_blocks:
        train = [row for row in rows if row.block_id != block]
        test_index = [i for i, row in enumerate(rows) if row.block_id == block]
        if not train or not test_index:
            scores[np.array(test_index, dtype=int)] = np.nan
            continue
        tables, pooled = estimate_q_tables(
            np.array([row.y for row in train], dtype=object),
            np.array([row.s for row in train], dtype=object),
            np.array([_feature_key(row, include_traces=include_traces) for row in train], dtype=object),
            grades=grades,
            alpha=alpha,
        )
        test_rows = [rows[i] for i in test_index]
        fold_scores = log_score_for_rows(
            tables,
            pooled,
            np.array([row.y for row in test_rows], dtype=object),
            np.array([row.s for row in test_rows], dtype=object),
            np.array(
                [_feature_key(row, include_traces=include_traces) for row in test_rows],
                dtype=object,
            ),
        )
        scores[np.array(test_index, dtype=int)] = fold_scores
    return scores


def _comparable_rows(
    rows: Sequence[BuildingAnalysisRow],
    visibility: str,
) -> list[BuildingAnalysisRow]:
    return [row for row in rows if row.eligible and row.visibility_bin == visibility]


@dataclass
class ChannelBenchResult:
    family: str
    claim: str
    official_ceiling_claim_allowed: bool
    log_score_q0: float
    log_score_q1: float
    gain: float
    gain_interval: PairedBlockInterval
    evidence_log_ratio_by_imprint_bin: dict[str, float]
    evidence_ratio_shift: PairedBlockInterval
    evidence_ratio_trend: str
    intact_report_rates_by_bin: dict[str, dict[str, float]]
    budget_hit_rates: dict[str, BudgetHitRate]
    support: SupportReport
    dropped_grades: tuple[str, ...]
    negligible_gain_band: NegligibleGainBand
    notes: list[str] = field(default_factory=list)
    lock_design_version: str = ANALYSIS_LOCK.design_version


def classify_channel_result(
    *,
    gain_interval: PairedBlockInterval,
    evidence_ratio_trend: str,
    mismatch_rises_with_imprint: bool,
    band: NegligibleGainBand,
    support: SupportReport,
) -> tuple[str, list[str]]:
    """Map locked metrics onto a claim label. No official ceiling while unlocked."""
    notes: list[str] = []
    comparable = [
        slice_
        for slice_ in support.slices
        if slice_.slice_name.startswith("visibility_clear_imprint_bin_")
    ]
    weak_support = any(not slice_.adequate for slice_ in comparable) if comparable else True
    if weak_support:
        notes.append("support flags present in a comparable-visibility imprint slice")
    if gain_interval.lower <= 0.0 < gain_interval.upper or (
        gain_interval.lower < 0.0 and gain_interval.upper > 0.0
    ):
        if evidence_ratio_trend == "stable" and mismatch_rises_with_imprint:
            return CLAIM_PRIOR_EXPLANATION, notes
        if not gain_interval.lower_bound_positive:
            notes.append("channel gain interval does not have a positive lower bound")
            return CLAIM_INCONCLUSIVE if evidence_ratio_trend != "stable" else CLAIM_PRIOR_EXPLANATION, notes
    if gain_interval.lower_bound_positive:
        if evidence_ratio_trend == "decreasing_toward_zero_or_below":
            return CLAIM_CHANNEL_EVIDENCE_CHANGE, notes
        if evidence_ratio_trend == "stable":
            notes.append("table fit improved but intact evidence ratio stayed stable")
            return CLAIM_REPORTING_CHANGES_WITH_ENVIRONMENT, notes
        notes.append("gain lower bound is positive but evidence-ratio direction is not locked-pass")
        return CLAIM_INCONCLUSIVE, notes
    if gain_interval.upper_bound_negative:
        notes.append("Q1 scored worse than Q0")
        return CLAIM_INCONCLUSIVE, notes
    notes.append("channel gain interval is inconclusive")
    del band  # width stays unused for official ceiling claims while unlocked
    return CLAIM_INCONCLUSIVE, notes


def run_channel_stress_bench(
    rows: Sequence[BuildingAnalysisRow],
    *,
    family: str = "unspecified",
    lock: AnalysisLock = ANALYSIS_LOCK,
    comparable_visibility: str = "clear",
    rng_seed: int = 0,
) -> ChannelBenchResult:
    """Estimate Q0/Q1, evidence ratios, budget hit rates, and a claim label."""
    eligible = [row for row in rows if row.eligible]
    if not eligible:
        raise ValueError("no eligible rows")
    comparable = _comparable_rows(eligible, comparable_visibility)
    if not comparable:
        raise ValueError(f"no eligible rows with visibility_bin={comparable_visibility!r}")

    scores_q0 = leave_block_out_scores(
        comparable,
        include_traces=False,
        alpha=lock.dirichlet_alpha,
        grades=lock.grade_names,
    )
    scores_q1 = leave_block_out_scores(
        comparable,
        include_traces=True,
        alpha=lock.dirichlet_alpha,
        grades=lock.grade_names,
    )
    y = np.array([row.y for row in comparable], dtype=object)
    mean0, _, dropped0 = _grade_weighted_mean(y, scores_q0, grades=lock.grade_names)
    mean1, _, dropped1 = _grade_weighted_mean(y, scores_q1, grades=lock.grade_names)
    dropped = tuple(sorted(set(dropped0) | set(dropped1)))
    gain_interval = paired_block_interval(
        np.array([row.block_id for row in comparable], dtype=object),
        scores_q1,
        scores_q0,
        n_replicates=lock.bootstrap_replicates,
        level=lock.interval_level,
        rng_seed=rng_seed,
    )

    # Evidence-ratio point estimates use a full-sample Q fitted on all
    # comparable-visibility rows. This is a different convention from the
    # grade-weighted log scores above, which are leave-block-out. The shift
    # interval is a block bootstrap of that full-sample estimator.
    ratios, ratio_shift = bootstrap_evidence_ratio_shift(
        comparable,
        mixture=lock.fixed_damage_mixture,
        alpha=lock.dirichlet_alpha,
        grades=lock.grade_names,
        n_replicates=lock.bootstrap_replicates,
        level=lock.interval_level,
        rng_seed=rng_seed,
    )
    trend = classify_evidence_ratio_trend(ratio_shift)
    bin_ids = sorted({int(imprint_bin(row.imprint_score)) for row in comparable})
    keys = np.array([f"t{imprint_bin(row.imprint_score)}" for row in comparable], dtype=object)
    tables, pooled = estimate_q_tables(
        y,
        np.array([row.s for row in comparable], dtype=object),
        keys,
        grades=lock.grade_names,
        alpha=lock.dirichlet_alpha,
    )
    rates = {f"t{bin_id}": intact_report_rates(tables.get(f"t{bin_id}", pooled)) for bin_id in bin_ids}

    mismatch_by_bin: dict[int, float] = {}
    for bin_id in bin_ids:
        subset = [row for row in comparable if imprint_bin(row.imprint_score) == bin_id]
        mismatch_by_bin[bin_id] = float(np.mean([row.s != row.y for row in subset]))
    mismatch_values = [mismatch_by_bin[bin_id] for bin_id in bin_ids]
    # Heuristic only: proposed/unlocked rise cutoff, not a locked official threshold.
    mismatch_rises = (
        len(mismatch_values) >= 2
        and mismatch_values[-1] - mismatch_values[0] > lock.mismatch_rise_threshold
    )

    # Hits are S != Y, not official U (U_E or U_F).
    hits = np.array([row.s != row.y for row in eligible], dtype=bool)
    uids = np.array([row.building_uid for row in eligible], dtype=object)
    budget_rates: dict[str, BudgetHitRate] = {}
    for contrast in lock.primary_contrasts:
        if contrast == "q0_visibility_only_vs_q1_visibility_and_traces":
            continue
        scores = np.array(
            [float(row.rankings[contrast]) for row in eligible],
            dtype=np.float64,
        )
        budget_rates[contrast] = budget_hit_rate(
            scores,
            hits,
            building_uids=uids,
            budget=lock.audit_budget,
            ranking_name=contrast,
        )

    support = support_report(rows, comparable_visibility=None)
    claim, notes = classify_channel_result(
        gain_interval=gain_interval,
        evidence_ratio_trend=trend,
        mismatch_rises_with_imprint=mismatch_rises,
        band=lock.negligible_gain_band,
        support=support,
    )
    notes.append(
        "Negligible-gain band width is unlocked_pending_jlin and is not used "
        "as an official ceiling threshold."
    )
    notes.append(
        "Budget hit rate uses S != Y, not official U (U_E or U_F)."
    )
    notes.append(
        "Evidence-ratio point estimates use full-sample Q, which is a "
        "different convention from leave-block-out log scores."
    )
    notes.append(
        "mismatch_rises_with_imprint threshold is proposed_unlocked "
        f"({lock.mismatch_rise_threshold:g}), not a locked official cutoff."
    )
    if dropped:
        notes.append("grades dropped from the weighted log score: " + ",".join(dropped))
    return ChannelBenchResult(
        family=family,
        claim=claim,
        official_ceiling_claim_allowed=lock.negligible_gain_band.is_officially_locked(),
        log_score_q0=mean0,
        log_score_q1=mean1,
        gain=float(mean1 - mean0),
        gain_interval=gain_interval,
        evidence_log_ratio_by_imprint_bin=ratios,
        evidence_ratio_shift=ratio_shift,
        evidence_ratio_trend=trend,
        intact_report_rates_by_bin=rates,
        budget_hit_rates=budget_rates,
        support=support,
        dropped_grades=dropped,
        negligible_gain_band=lock.negligible_gain_band,
        notes=notes,
    )


def refuse_official_michael_path(path: str) -> None:
    """Guard rail: this bench does not treat official Michael artifacts as inputs."""
    text = path.replace("\\", "/").lower()
    markers = (
        "hurricane" + "-michael",
        "hurricane" + "_michael",
        "/official/",
        "worktrees" + "/folds-",
        "michael" + "-seed",
    )
    if any(marker in text for marker in markers):
        raise ValueError(
            "official Michael outputs are not accepted as channel-bench inputs; "
            "pass synthetic fixtures or a later frozen export through a dedicated loader"
        )


def iter_building_uids(rows: Iterable[BuildingAnalysisRow]) -> list[str]:
    return [row.building_uid for row in rows]

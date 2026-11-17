"""Locked analysis sheet, channel stress bench, and blinded review sampler.

This package is analysis development only. It uses synthetic fixtures, runs
on CPU, and does not read official Michael outputs or start an image head.
"""

from mvbridge.channel_bench.analysis import (
    ChannelBenchResult,
    classify_channel_result,
    run_channel_stress_bench,
)
from mvbridge.channel_bench.budget import budget_hit_rate, paired_budget_increment
from mvbridge.channel_bench.evidence import intact_evidence_log_ratio, intact_report_rates
from mvbridge.channel_bench.fixtures import (
    FixtureDataset,
    generate_equal_likelihood_case,
    generate_prior_change_fixture,
    generate_q_change_fixture,
)
from mvbridge.channel_bench.intervals import paired_block_interval
from mvbridge.channel_bench.locks import ANALYSIS_LOCK, NegligibleGainBand
from mvbridge.channel_bench.sampler import BlindedReviewSampler, ScoreLock
from mvbridge.channel_bench.schema import (
    ANALYSIS_TABLE_COLUMNS,
    DAMAGE_CLASSES,
    INTACT_CLASS,
    BuildingAnalysisRow,
)
from mvbridge.channel_bench.support import support_report
from mvbridge.channel_bench.tables import (
    equal_likelihood_leaves_prior_unchanged,
    normalize_q_rows,
    posterior_from_likelihood,
)

__all__ = [
    "ANALYSIS_LOCK",
    "ANALYSIS_TABLE_COLUMNS",
    "DAMAGE_CLASSES",
    "INTACT_CLASS",
    "BlindedReviewSampler",
    "BuildingAnalysisRow",
    "ChannelBenchResult",
    "FixtureDataset",
    "NegligibleGainBand",
    "ScoreLock",
    "budget_hit_rate",
    "classify_channel_result",
    "equal_likelihood_leaves_prior_unchanged",
    "generate_equal_likelihood_case",
    "generate_prior_change_fixture",
    "generate_q_change_fixture",
    "intact_evidence_log_ratio",
    "intact_report_rates",
    "normalize_q_rows",
    "paired_block_interval",
    "paired_budget_increment",
    "posterior_from_likelihood",
    "run_channel_stress_bench",
    "support_report",
]

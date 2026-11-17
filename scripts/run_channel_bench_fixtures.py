#!/usr/bin/env python3
"""CPU-only synthetic fixture run for the locked channel bench.

This script does not read official scores, does not start an image head, and
does not launch a fusion experiment. It prints a compact identification
summary for the two fixture families.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from mvbridge.channel_bench.analysis import run_channel_stress_bench
from mvbridge.channel_bench.fixtures import (
    generate_equal_likelihood_case,
    generate_prior_change_fixture,
    generate_q_change_fixture,
)
from mvbridge.channel_bench.locks import ANALYSIS_LOCK


def _interval(interval) -> dict:
    return {
        "mean": interval.mean,
        "lower": interval.lower,
        "upper": interval.upper,
        "lower_bound_positive": interval.lower_bound_positive,
        "n_blocks": interval.n_blocks,
        "n_replicates": interval.n_replicates,
    }


def _summary(result) -> dict:
    return {
        "family": result.family,
        "claim": result.claim,
        "official_ceiling_claim_allowed": result.official_ceiling_claim_allowed,
        "gain": result.gain,
        "gain_interval": _interval(result.gain_interval),
        "evidence_log_ratio_by_imprint_bin": result.evidence_log_ratio_by_imprint_bin,
        "evidence_ratio_shift": _interval(result.evidence_ratio_shift),
        "evidence_ratio_trend": result.evidence_ratio_trend,
        "intact_report_rates_by_bin": result.intact_report_rates_by_bin,
        "budget_hit_rates": {
            name: asdict(rate) for name, rate in result.budget_hit_rates.items()
        },
        "dropped_grades": list(result.dropped_grades),
        "lock_design_version": result.lock_design_version,
        "negligible_gain_band": {
            "width_nats": result.negligible_gain_band.width_nats,
            "status": result.negligible_gain_band.status,
            "unit": result.negligible_gain_band.unit,
        },
        "notes": result.notes,
    }


def main() -> None:
    equal = generate_equal_likelihood_case()
    prior = generate_prior_change_fixture()
    q_change = generate_q_change_fixture()
    payload = {
        "design_version": ANALYSIS_LOCK.design_version,
        "official_recipe_untouched": ANALYSIS_LOCK.official_recipe_untouched,
        "accepts_official_michael_inputs": ANALYSIS_LOCK.accepts_official_michael_inputs,
        "equal_likelihood_leaves_prior_unchanged": equal.prior_unchanged,
        "prior_change": _summary(run_channel_stress_bench(prior.rows, family=prior.family)),
        "q_change": _summary(run_channel_stress_bench(q_change.rows, family=q_change.family)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

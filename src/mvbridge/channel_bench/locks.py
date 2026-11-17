"""Pre-unblinding analysis lock. Numeric official freezes stay with Jlin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

try:
    from mvbridge.score_contract.protocol import OFFICIAL_CLASS_ORDER
except ImportError:  # Sibling package is present after the main merge.
    OFFICIAL_CLASS_ORDER = ("no-damage", "minor-damage", "major-damage", "destroyed")


# Example only. Not an official freeze. See NegligibleGainBand and the lock doc.
EXAMPLE_NEGLIGIBLE_GAIN_BAND_NATS = 0.02
NEGLIGIBLE_GAIN_BAND_STATUS = "unlocked_pending_jlin"

# Same four-class order as the official score-contract / xBD labels.
DAMAGE_CLASSES = OFFICIAL_CLASS_ORDER
INTACT_CLASS = "no-damage"

# Mild/moderate mixture for the intact-report evidence-ratio denominator.
# Proposed for this design version; not silently an official freeze.
PROPOSED_FIXED_DAMAGE_MIXTURE: dict[str, float] = {
    "minor-damage": 0.5,
    "major-damage": 0.5,
}

PROPOSED_AUDIT_BUDGET = 0.10
PROPOSED_DIRICHLET_ALPHA = 0.5
PROPOSED_BOOTSTRAP_REPLICATES = 1000
PROPOSED_INTERVAL_LEVEL = 0.95
PROPOSED_IMPRINT_EDGES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
PROPOSED_HIGH_SCORE_TAIL = 0.20
PROPOSED_MISSED_ERROR_TAIL = 0.50
PROPOSED_REVIEW_QUOTA = 8
DESIGN_VERSION = "channel-bench-lock-v1"


@dataclass(frozen=True)
class NegligibleGainBand:
    """Width of the practical negligible-gain band, in nats per building.

    While ``status`` is ``unlocked_pending_jlin``, the width may be used to
    test machinery on synthetic fixtures but must not be reported as an
    official frozen threshold.
    """

    width_nats: float = EXAMPLE_NEGLIGIBLE_GAIN_BAND_NATS
    status: str = NEGLIGIBLE_GAIN_BAND_STATUS
    unit: str = "nats_per_building_grade_balanced_log_score"

    def is_officially_locked(self) -> bool:
        return self.status == "locked_by_jlin"

    def contains(self, value: float) -> bool:
        return abs(float(value)) <= float(self.width_nats)


@dataclass(frozen=True)
class AnalysisLock:
    """Locked directions, contrasts, splits, and strata for this design."""

    design_version: str = DESIGN_VERSION
    grade_names: tuple[str, ...] = DAMAGE_CLASSES
    intact_class: str = INTACT_CLASS
    log_score_direction: str = "higher_is_better"
    channel_gain_pass_direction: str = "positive_lower_bound"
    intact_evidence_ratio_direction: str = (
        "along_increasing_imprint_in_comparable_visibility_toward_zero_from_above"
    )
    budget_hit_rate_direction: str = "higher_is_better"
    primary_contrasts: tuple[str, ...] = (
        "group3_neighborhood_only",
        "group3_public_geo_only",
        "group3_neighborhood_and_public_geo",
        "group3_plus_frozen_visual_mismatch",
        "entropy",
        "one_minus_max_p",
        "q0_visibility_only_vs_q1_visibility_and_traces",
    )
    visibility_fields: tuple[str, ...] = (
        "visibility_bin",
        "clear_pixel_fraction",
        "gsd_m",
        "off_nadir_deg",
    )
    trace_fields: tuple[str, ...] = (
        "sat_alarm_density_30m",
        "sat_alarm_density_100_300m",
        "dist_to_nearest_sat_collapsed",
        "elevation",
        "dist_to_water",
    )
    trace_excludes_index_building_alarm: bool = True
    imprint_field: str = "imprint_score"
    imprint_bin_edges: tuple[float, ...] = PROPOSED_IMPRINT_EDGES
    imprint_from_raw_traces_status: str = "pending_jlin"
    review_strata: tuple[str, ...] = (
        "high_score_error",
        "missed_error",
        "matched_condition_correct",
    )
    high_score_tail: float = PROPOSED_HIGH_SCORE_TAIL
    missed_error_tail: float = PROPOSED_MISSED_ERROR_TAIL
    review_quota_per_stratum: int = PROPOSED_REVIEW_QUOTA
    audit_budget: float = PROPOSED_AUDIT_BUDGET
    fixed_damage_mixture: Mapping[str, float] = field(
        default_factory=lambda: dict(PROPOSED_FIXED_DAMAGE_MIXTURE)
    )
    grade_weighting: str = "equal_over_supported_grades"
    dirichlet_alpha: float = PROPOSED_DIRICHLET_ALPHA
    bootstrap_replicates: int = PROPOSED_BOOTSTRAP_REPLICATES
    interval_level: float = PROPOSED_INTERVAL_LEVEL
    leave_out_scheme: str = "leave_block_out"
    negligible_gain_band: NegligibleGainBand = field(default_factory=NegligibleGainBand)
    official_recipe_untouched: bool = True
    accepts_official_michael_inputs: bool = False


ANALYSIS_LOCK = AnalysisLock()

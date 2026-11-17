"""CPU-only tests for the locked channel bench, using synthetic fixtures only."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from mvbridge.channel_bench.analysis import (
    CLAIM_CHANNEL_EVIDENCE_CHANGE,
    CLAIM_PRIOR_EXPLANATION,
    refuse_official_michael_path,
    run_channel_stress_bench,
)
from mvbridge.channel_bench.budget import budget_hit_rate, paired_budget_increment
from mvbridge.channel_bench.evidence import intact_evidence_log_ratio, intact_report_rates
from mvbridge.channel_bench.fixtures import (
    generate_equal_likelihood_case,
    generate_prior_change_fixture,
    generate_q_change_fixture,
    prior_change_q_table,
    q_change_table,
    ranking_matrix,
    true_q_for_row,
)
from mvbridge.channel_bench.intervals import paired_block_interval
from mvbridge.channel_bench.locks import (
    ANALYSIS_LOCK,
    EXAMPLE_NEGLIGIBLE_GAIN_BAND_NATS,
    NEGLIGIBLE_GAIN_BAND_STATUS,
    NegligibleGainBand,
)
from mvbridge.channel_bench.sampler import BlindedReviewSampler
from mvbridge.channel_bench.schema import DAMAGE_CLASSES, INTACT_CLASS, imprint_bin
from mvbridge.channel_bench.support import support_report
from mvbridge.channel_bench.tables import (
    equal_likelihood_leaves_prior_unchanged,
    normalize_q_rows,
    posterior_from_likelihood,
)


def test_row_normalization_and_rejects_bad_tables():
    raw = np.array([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    # Pad to 4x4 for a full table test of the helper on a rectangular slice.
    table = np.vstack([raw, raw])
    normed = normalize_q_rows(table)
    assert np.allclose(normed.sum(axis=1), 1.0)
    assert np.allclose(normed[0], [0.5, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="negative"):
        normalize_q_rows(np.array([[1.0, -0.2, 0.1, 0.1]]))
    with pytest.raises(ValueError, match="zero row"):
        normalize_q_rows(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_q_rows(np.array([[1.0, np.nan, 0.0, 0.0]]))


def test_equal_class_likelihoods_leave_prior_unchanged():
    prior = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    case = generate_equal_likelihood_case(prior, common_likelihood=0.25)
    assert case.prior_unchanged
    assert np.allclose(case.posterior, case.prior)
    assert equal_likelihood_leaves_prior_unchanged(prior, 1.0)
    # A non-equal likelihood must move the prior.
    moved = posterior_from_likelihood(prior, np.array([0.8, 0.1, 0.05, 0.05]))
    assert not np.allclose(moved, prior / prior.sum())
    assert moved[0] > prior[0]


def test_intact_evidence_log_ratio_formula_and_signs():
    # Construct Q where intact-report probs are known exactly.
    table = normalize_q_rows(
        np.array(
            [
                [0.80, 0.10, 0.06, 0.04],
                [0.20, 0.60, 0.15, 0.05],
                [0.10, 0.20, 0.55, 0.15],
                [0.05, 0.10, 0.25, 0.60],
            ]
        )
    )
    mixture = {"minor-damage": 0.5, "major-damage": 0.5}
    den = 0.5 * table[1, 0] + 0.5 * table[2, 0]
    expected = float(np.log(table[0, 0]) - np.log(den))
    assert intact_evidence_log_ratio(table, mixture=mixture) == pytest.approx(expected)
    assert expected > 0.0
    rates = intact_report_rates(table)
    assert set(rates) == set(DAMAGE_CLASSES)
    assert rates[INTACT_CLASS] == pytest.approx(table[0, 0])

    # Equal intact-report rates => log-ratio 0.
    equal = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (4, 1))
    assert intact_evidence_log_ratio(equal, mixture=mixture) == pytest.approx(0.0)

    inverted = normalize_q_rows(
        np.array(
            [
                [0.10, 0.30, 0.30, 0.30],
                [0.40, 0.30, 0.20, 0.10],
                [0.50, 0.20, 0.20, 0.10],
                [0.20, 0.20, 0.20, 0.40],
            ]
        )
    )
    assert intact_evidence_log_ratio(inverted, mixture=mixture) < 0.0


def test_q_change_true_tables_move_evidence_ratio_toward_zero():
    low = intact_evidence_log_ratio(q_change_table(0.05, "clear"))
    mid = intact_evidence_log_ratio(q_change_table(0.50, "clear"))
    high = intact_evidence_log_ratio(q_change_table(0.95, "clear"))
    assert low > mid > high
    assert high == pytest.approx(0.0, abs=0.35)
    low_rates = intact_report_rates(q_change_table(0.05, "clear"))
    high_rates = intact_report_rates(q_change_table(0.95, "clear"))
    assert low_rates[INTACT_CLASS] > 0.5
    damaged = ("minor-damage", "major-damage", "destroyed")
    assert all(high_rates[name] > low_rates[name] for name in damaged)


def test_paired_block_interval_covers_mean_and_is_seed_stable():
    block_ids = np.array([f"b{i // 5}" for i in range(50)], dtype=object)
    rng = np.random.default_rng(0)
    values_a = rng.normal(0.10, 0.02, size=50)
    values_b = rng.normal(0.00, 0.02, size=50)
    first = paired_block_interval(block_ids, values_a, values_b, rng_seed=123, n_replicates=800)
    second = paired_block_interval(block_ids, values_a, values_b, rng_seed=123, n_replicates=800)
    assert first.resample_sha256 == second.resample_sha256
    assert first.lower == pytest.approx(second.lower)
    assert first.contains(first.mean)
    # Shared resampling design: explicit indices must be honored.
    draws = np.random.default_rng(9).integers(0, first.n_blocks, size=(200, first.n_blocks))
    shared_a = paired_block_interval(block_ids, values_a, values_b, resample_indices=draws)
    shared_b = paired_block_interval(block_ids, values_a, values_b, resample_indices=draws)
    assert shared_a.resample_sha256 == shared_b.resample_sha256
    assert first.lower_bound_positive


def test_budget_hit_rate_perfect_and_anti_rankers():
    n = 200
    hits = np.array([i < 40 for i in range(n)])
    uids = [f"id{i:03d}" for i in range(n)]
    perfect = np.array([1.0 if hit else 0.0 for hit in hits], dtype=np.float64)
    anti = -perfect
    good = budget_hit_rate(perfect, hits, building_uids=uids, budget=0.10)
    bad = budget_hit_rate(anti, hits, building_uids=uids, budget=0.10)
    assert good.n_audited == 20
    assert good.hit_rate == pytest.approx(1.0)
    assert bad.hit_rate == pytest.approx(0.0)
    tied = np.ones(n)
    # Tie break is by uid ascending, which here follows construction order.
    tied_rate = budget_hit_rate(tied, hits, building_uids=uids, budget=0.10)
    assert tied_rate.n_hits == hits[:20].sum()
    increment = paired_budget_increment(
        perfect, anti, hits, building_uids=uids, budget=0.10, name_a="perfect", name_b="anti"
    )
    assert increment["increment_a_minus_b"] == pytest.approx(1.0)


def test_support_report_counts_and_flags():
    data = generate_prior_change_fixture(n_blocks=12, buildings_per_block=8, rng_seed=3)
    report = support_report(data.rows, min_buildings=5, min_blocks=2, min_per_grade=1)
    eligible = next(s for s in report.slices if s.slice_name == "eligible_all")
    assert eligible.n_buildings == 12 * 8
    assert eligible.n_blocks == 12
    assert eligible.n_surveyed == eligible.n_buildings
    assert sum(eligible.y_counts.values()) == eligible.n_buildings
    assert eligible.mismatch_count == sum(row.s != row.y for row in data.rows)
    tiny = support_report(data.rows[:3], min_buildings=20, min_blocks=5, min_per_grade=5)
    assert any(not slice_.adequate for slice_ in tiny.slices)


def test_prior_change_fixture_recovers_hotspots_not_channel_loss():
    data = generate_prior_change_fixture(n_blocks=36, buildings_per_block=40, rng_seed=11)
    arrays = data.arrays()
    # Mismatch rate should rise with imprint because the prior moves toward
    # grades that the constant Q reports less accurately.
    low = arrays["mismatch"][arrays["imprint_bin"] == 0].mean()
    high = arrays["mismatch"][arrays["imprint_bin"] == 2].mean()
    assert high > low + 0.05
    # True Q does not change, so evidence ratios are identical.
    ratio = intact_evidence_log_ratio(prior_change_q_table())
    for row in data.rows[:8]:
        assert intact_evidence_log_ratio(true_q_for_row(row, "prior_change")) == pytest.approx(ratio)
    result = run_channel_stress_bench(data.rows, family=data.family, rng_seed=0)
    assert result.claim == CLAIM_PRIOR_EXPLANATION
    assert result.evidence_ratio_trend == "stable"
    assert not result.gain_interval.lower_bound_positive
    # Do not claim official ceiling while the band is unlocked.
    assert not result.official_ceiling_claim_allowed
    assert result.negligible_gain_band.status == "unlocked_pending_jlin"
    # Group-3 combined should beat a reversed exposure ranking as a hotspot check.
    hits = arrays["mismatch"]
    uids = arrays["building_uid"]
    group3 = ranking_matrix(data, "group3_neighborhood_and_public_geo")
    anti = -group3
    better = budget_hit_rate(group3, hits, building_uids=uids, budget=0.10)
    worse = budget_hit_rate(anti, hits, building_uids=uids, budget=0.10)
    assert better.hit_rate > worse.hit_rate


def test_q_change_fixture_recovers_signed_evidence_change():
    data = generate_q_change_fixture(n_blocks=36, buildings_per_block=40, rng_seed=17)
    result = run_channel_stress_bench(data.rows, family=data.family, rng_seed=0)
    ratios = result.evidence_log_ratio_by_imprint_bin
    keys = sorted(ratios, key=lambda name: int(name[1:]))
    values = [ratios[key] for key in keys]
    assert len(values) == 3
    assert values[0] > values[-1]
    assert result.evidence_ratio_trend == "decreasing_toward_zero_or_below"
    assert result.gain_interval.lower_bound_positive
    assert result.claim == CLAIM_CHANNEL_EVIDENCE_CHANGE
    # Both undamaged and damaged intact-report rates are present in every bin.
    for rates in result.intact_report_rates_by_bin.values():
        assert set(rates) == set(DAMAGE_CLASSES)
        assert all(rate >= 0.0 for rate in rates.values())
        assert rates[INTACT_CLASS] > 0.0
        assert any(rates[name] > 0.0 for name in DAMAGE_CLASSES if name != INTACT_CLASS)


def test_lock_scores_then_view_images_sampler():
    data = generate_prior_change_fixture(n_blocks=20, buildings_per_block=20, rng_seed=5)
    scores = {
        row.building_uid: float(row.rankings["group3_neighborhood_and_public_geo"])
        for row in data.rows
    }
    sampler = BlindedReviewSampler()
    with pytest.raises(RuntimeError, match="locked"):
        sampler.sample(rng_seed=1)
    lock = sampler.lock_scores(data.rows, scores)
    assert lock.sha256 == hashlib.sha256(lock.payload).hexdigest()
    again = BlindedReviewSampler()
    assert again.lock_scores(data.rows, scores).sha256 == lock.sha256
    manifest = sampler.sample(rng_seed=7, expected_lock_sha256=lock.sha256)
    assert manifest.lock_sha256 == lock.sha256
    assert all(case.image_ref is None for case in manifest.cases)
    assert not manifest.images_attached
    strata = {case.stratum for case in manifest.cases}
    assert "high_score_error" in strata
    assert "missed_error" in strata
    assert "matched_condition_correct" in strata
    for case in manifest.cases:
        if case.stratum == "high_score_error":
            assert case.mismatch
        if case.stratum == "missed_error":
            assert case.mismatch
        if case.stratum == "matched_condition_correct":
            assert not case.mismatch
            assert case.matched_to is not None
    # Changing scores after the fact must not silently relabel the lock.
    tampered = dict(scores)
    first_uid = next(iter(tampered))
    tampered[first_uid] = tampered[first_uid] + 3.0
    other = BlindedReviewSampler()
    other_lock = other.lock_scores(data.rows, tampered)
    assert other_lock.sha256 != lock.sha256
    with pytest.raises(ValueError, match="lock hash"):
        sampler.sample(rng_seed=7, expected_lock_sha256=other_lock.sha256)
    # Images may be attached only after the recorded lock is confirmed.
    image_index = {row.building_uid: f"synthetic://{row.building_uid}.png" for row in data.rows}
    viewed = sampler.attach_images(manifest, image_index, expected_lock_sha256=lock.sha256)
    assert viewed.images_attached
    assert viewed.uids() == manifest.uids()
    assert all(case.image_ref is not None for case in viewed.cases)
    with pytest.raises(ValueError, match="lock hash"):
        sampler.attach_images(manifest, image_index, expected_lock_sha256="0" * 64)


def test_analysis_lock_marks_negligible_band_unofficial():
    assert ANALYSIS_LOCK.accepts_official_michael_inputs is False
    assert ANALYSIS_LOCK.official_recipe_untouched is True
    assert ANALYSIS_LOCK.trace_excludes_index_building_alarm is True
    assert NEGLIGIBLE_GAIN_BAND_STATUS == "unlocked_pending_jlin"
    assert EXAMPLE_NEGLIGIBLE_GAIN_BAND_NATS == 0.02
    band = NegligibleGainBand()
    assert not band.is_officially_locked()
    assert band.contains(0.01)
    assert not band.contains(0.05)
    locked = NegligibleGainBand(width_nats=0.03, status="locked_by_jlin")
    assert locked.is_officially_locked()
    assert "group3_plus_frozen_visual_mismatch" in ANALYSIS_LOCK.primary_contrasts
    assert ANALYSIS_LOCK.review_strata == (
        "high_score_error",
        "missed_error",
        "matched_condition_correct",
    )


def test_imprint_bins_use_prefixed_edges_not_labels():
    assert imprint_bin(0.0) == 0
    assert imprint_bin(1.0 / 3.0 - 1e-12) == 0
    assert imprint_bin(1.0 / 3.0) == 1
    assert imprint_bin(1.0) == 2
    with pytest.raises(ValueError):
        imprint_bin(-0.01)


def test_refuse_official_michael_paths_and_source_has_no_default_dump():
    with pytest.raises(ValueError, match="official Michael"):
        refuse_official_michael_path("/mnt/data_hdd/worktrees/folds-michael/scores.json")
    with pytest.raises(ValueError, match="official Michael"):
        refuse_official_michael_path("C:/runs/hurricane-michael-seed1/preds.csv")
    root = Path(__file__).resolve().parents[1] / "src" / "mvbridge" / "channel_bench"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "train-xbd" not in text
    # The package must not default to an official export path.
    assert "/mnt/data_hdd/worktrees/folds-" not in text
    source = inspect.getsource(run_channel_stress_bench)
    assert "michael" not in source.lower()


def test_two_fixture_seeds_can_share_block_resampling():
    a = generate_q_change_fixture(n_blocks=24, buildings_per_block=20, rng_seed=1, seed_id="1")
    b = generate_q_change_fixture(n_blocks=24, buildings_per_block=20, rng_seed=2, seed_id="2")
    assert [row.block_id for row in a.rows] == [row.block_id for row in b.rows]
    # Same geometry, different draws: uids differ by seed id, blocks match.
    interval_a = paired_block_interval(
        np.array([row.block_id for row in a.eligible_rows()], dtype=object),
        np.ones(len(a.eligible_rows())),
        np.zeros(len(a.eligible_rows())),
        rng_seed=99,
        n_replicates=400,
    )
    interval_b = paired_block_interval(
        np.array([row.block_id for row in b.eligible_rows()], dtype=object),
        np.ones(len(b.eligible_rows())),
        np.zeros(len(b.eligible_rows())),
        rng_seed=99,
        n_replicates=400,
    )
    assert interval_a.resample_sha256 == interval_b.resample_sha256


def test_class_order_matches_score_contract():
    """Align with the official score-contract constant without importing torch.

    ``mvbridge.score_contract.protocol`` imports fold inventory helpers that
    pull in ``torch``. This analysis worktree is CPU-only, so the test reads
    ``OFFICIAL_CLASS_ORDER`` from the protocol source with ``ast``.
    """
    import ast

    protocol_path = Path(__file__).resolve().parents[1] / "src" / "mvbridge" / "score_contract" / "protocol.py"
    if not protocol_path.is_file():
        pytest.skip("score-contract protocol is not present in this checkout")
    official = None
    tree = ast.parse(protocol_path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        if any(isinstance(target, ast.Name) and target.id == "OFFICIAL_CLASS_ORDER" for target in targets):
            official = ast.literal_eval(value)
            break
    assert official == ("no-damage", "minor-damage", "major-damage", "destroyed")
    assert DAMAGE_CLASSES == official
    assert INTACT_CLASS == "no-damage"

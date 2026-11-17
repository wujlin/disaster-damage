"""Score contract tests. Fixtures are synthetic; no official scores or GPUs."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest

from mvbridge.data.folds import load_folds
from mvbridge.errors import ConfigError
from mvbridge.score_contract.adapter import (
    align_inference_and_labels,
    build_inference_table,
    entropy,
    inference_payload_unchanged,
    one_minus_max_p,
    predict_satellite_class,
    score_probabilities,
)
from mvbridge.score_contract.eligibility import (
    assign_coverage_class,
    four_class_eligibility,
    is_four_class_label,
    sufficient_satellite_evidence,
)
from mvbridge.score_contract.fixtures import (
    synthetic_contract_folds,
    synthetic_evaluation_labels,
    synthetic_historical_records,
    synthetic_probability_rows,
)
from mvbridge.score_contract.inventory import (
    inventory_historical_pairs_by_fold,
    reporting_model_defines_u,
    reporting_model_version,
)
from mvbridge.score_contract.protocol import (
    OFFICIAL_CLASS_ORDER,
    P4_BASELINES,
    authorized_artifact_inventory,
    load_protocol_contract,
    p4_comparison_baselines,
    p5_comparison_baselines,
)

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_reader_loads_shipped_documents():
    contract = load_protocol_contract(ROOT)
    assert Path(contract.protocol_path).name == "eval_protocol.md"
    assert contract.official_class_order == OFFICIAL_CLASS_ORDER
    assert contract.p4_baselines == ("entropy", "one_minus_max_p")
    assert contract.p5_baselines == ("g0",)
    assert contract.satellite_clear_pixel_min == 0.80
    assert contract.min_conflicts_total == 300
    assert contract.min_qualified_events == 3
    assert contract.min_conflicts_per_event == 30
    assert contract.u_defining_seed == 1
    assert contract.recipe_from_config["image_size_data"] == 518
    assert contract.recipe_from_config["image_size_backbone"] == 518
    assert contract.recipe_from_config["freeze_backbone"] is True
    assert contract.recipe_from_config["batch_size"] == 32
    assert contract.recipe_from_config["num_classes"] == 4


def test_authorized_artifact_inventory_does_not_invent_hashes():
    inventory = authorized_artifact_inventory(ROOT / "configs" / "folds" / "event_folds.yaml")
    by_name = {record.fold_name: record for record in inventory.records}
    assert "minus-hurricane-michael" in by_name
    michael = by_name["minus-hurricane-michael"]
    assert michael.seed == 1
    assert michael.role == "main"
    assert michael.claim_evidence is True
    assert michael.manifest_sha256 is None
    assert michael.checkpoint_sha256 is None
    seed2 = by_name["minus-hurricane-michael-seed2"]
    assert seed2.seed == 2
    assert seed2.role == "seed-repeat"
    assert seed2.claim_evidence is False
    loaded = inventory.weight_provenance["loaded"]["sha256"]
    assert loaded == "6cce660bf1cdf01bc70ad96d53c3ad4d7d02fd4211a669b04abba9aaa848e66c"
    official_folds = load_folds(ROOT / "configs" / "folds" / "event_folds.yaml")
    assert [fold.name for fold in official_folds] == [record.fold_name for record in inventory.records]


def test_p4_main_test_excludes_g0():
    assert p4_comparison_baselines() == P4_BASELINES
    with pytest.raises(ConfigError, match="g0 belongs only to P5"):
        p4_comparison_baselines(("entropy", "one_minus_max_p", "g0"))
    assert p5_comparison_baselines() == ("g0",)


def test_entropy_and_one_minus_max_p_on_synthetic_probabilities():
    uniform = [0.25, 0.25, 0.25, 0.25]
    onehot = [0.0, 0.0, 0.0, 1.0]
    peaked = [0.05, 0.10, 0.80, 0.05]
    assert entropy(uniform) == pytest.approx(math.log(4))
    assert one_minus_max_p(uniform) == pytest.approx(0.75)
    assert entropy(onehot) == pytest.approx(0.0)
    assert one_minus_max_p(onehot) == pytest.approx(0.0)
    scored = score_probabilities(peaked)
    expected_entropy = -sum(p * math.log(p) for p in peaked)
    assert scored.entropy == pytest.approx(expected_entropy)
    assert scored.one_minus_max_p == pytest.approx(0.20)
    assert scored.prediction.class_name == "major-damage"
    assert scored.prediction.tied is False


def test_official_class_order_is_used_for_argmax():
    # Index 0 no-damage, 1 minor, 2 major, 3 destroyed.
    assert predict_satellite_class([0.51, 0.20, 0.20, 0.09]).class_name == "no-damage"
    assert predict_satellite_class([0.09, 0.51, 0.20, 0.20]).class_name == "minor-damage"
    assert predict_satellite_class([0.10, 0.20, 0.60, 0.10]).class_name == "major-damage"
    assert predict_satellite_class([0.05, 0.05, 0.10, 0.80]).class_name == "destroyed"
    with pytest.raises(ConfigError, match="length 4"):
        predict_satellite_class([0.5, 0.5, 0.0])
    with pytest.raises(ConfigError, match="sum to 1"):
        score_probabilities([0.2, 0.2, 0.2, 0.2])


def test_ties_break_to_lowest_official_index():
    no_and_minor = predict_satellite_class([0.4, 0.4, 0.2, 0.0])
    assert no_and_minor.class_name == "no-damage"
    assert no_and_minor.class_index == 0
    assert no_and_minor.tied is True
    assert no_and_minor.tied_indices == (0, 1)

    major_and_destroyed = predict_satellite_class([0.1, 0.1, 0.4, 0.4])
    assert major_and_destroyed.class_name == "major-damage"
    assert major_and_destroyed.tied_indices == (2, 3)

    four_way = predict_satellite_class([0.25, 0.25, 0.25, 0.25])
    assert four_way.class_name == "no-damage"
    assert four_way.tied is True
    assert four_way.tied_indices == (0, 1, 2, 3)


def test_seed_numbering_for_reporting_heads():
    folds = {fold.name: fold for fold in synthetic_contract_folds()}
    seed1 = folds["minus-storm-alpha"]
    seed2 = folds["minus-storm-alpha-seed2"]
    assert seed1.seed == 1
    assert seed2.seed == 2
    assert seed2.role == "seed-repeat"
    assert reporting_model_defines_u(seed1) is True
    assert reporting_model_defines_u(seed2) is False
    assert "seed=1" in reporting_model_version(seed1)
    assert "seed=2" in reporting_model_version(seed2)
    assert "checkpoint_sha256=TBD" in reporting_model_version(seed1)
    official = {fold.name: fold for fold in load_folds(ROOT / "configs" / "folds" / "event_folds.yaml")}
    assert official["minus-hurricane-michael"].seed == 1
    assert reporting_model_defines_u(official["minus-hurricane-michael"]) is True
    assert official["minus-hurricane-michael-seed2"].seed == 2
    assert reporting_model_defines_u(official["minus-hurricane-michael-seed2"]) is False


def test_four_class_eligibility_conditions():
    base = {
        "y": "minor-damage",
        "s": "no-damage",
        "reference_available": True,
        "post_event_intersects_footprint": True,
        "clear_valid_pixel_fraction": 0.80,
        "match_accepted": True,
    }
    ok = four_class_eligibility(base)
    assert ok.eligible is True
    assert ok.in_q is True
    assert ok.coverage_class == "satellite+ground"
    assert is_four_class_label("no") is True
    assert is_four_class_label("Destroyed") is True
    assert is_four_class_label("unknown") is False
    assert is_four_class_label("un-classified") is False
    assert is_four_class_label("ds1") is False
    assert is_four_class_label("-1") is False

    unknown = four_class_eligibility({**base, "y": "unknown"})
    assert unknown.eligible is False
    assert "y_not_four_class" in unknown.reasons

    cloudy = four_class_eligibility({**base, "clear_valid_pixel_fraction": 0.79})
    assert cloudy.eligible is False
    assert cloudy.in_q is False
    assert "insufficient_satellite_evidence" in cloudy.reasons
    assert cloudy.coverage_class == "ground-only"

    no_ref = four_class_eligibility({**base, "reference_available": False})
    assert no_ref.coverage_class == "satellite-only"
    assert no_ref.in_q is False
    assert no_ref.eligible is False

    no_post = four_class_eligibility({**base, "post_event_intersects_footprint": False})
    assert no_post.coverage_class == "ground-only"
    assert sufficient_satellite_evidence(
        post_event_intersects_footprint=True,
        clear_valid_pixel_fraction=0.80,
    )
    assert not sufficient_satellite_evidence(
        post_event_intersects_footprint=True,
        clear_valid_pixel_fraction=0.799,
    )
    assert assign_coverage_class(reference_available=False, sufficient_satellite=False) == "insufficient evidence"

    dropped = four_class_eligibility({**base, "ambiguous_match": True})
    assert dropped.eligible is False
    assert "ambiguous_or_dropped_match" in dropped.reasons


def test_one_to_one_alignment_and_label_decoupling():
    rows = synthetic_probability_rows()
    table = build_inference_table(rows)
    assert table.keys() == tuple(sorted((row["image_id"], row["building_uid"]) for row in rows))
    peak = table.by_key()[("storm-alpha_0001", "b-peak-major")]
    assert peak.y_hat_s == "major-damage"
    assert peak.tied is False
    tied = table.by_key()[("storm-alpha_0004", "b-tie-no-minor")]
    assert tied.y_hat_s == "no-damage"
    assert tied.tied is True
    # Label fields on the input must not leak into inference extras.
    assert all("y" not in dict(row.extras) for row in table.rows)

    labels = synthetic_evaluation_labels(rows)
    aligned = align_inference_and_labels(table, labels)
    assert len(aligned) == len(table.rows)
    assert [row.key for row, _label in aligned] == list(table.keys())

    shuffled = list(reversed(labels))
    shuffled_aligned = align_inference_and_labels(table, shuffled)
    assert [row.scores_only() for row, _label in shuffled_aligned] == [row.scores_only() for row in table.rows]

    dropped_labels = labels[1:]
    with pytest.raises(ConfigError, match="not one-to-one"):
        align_inference_and_labels(table, dropped_labels)

    shuffled_inputs = list(reversed(deepcopy(rows)))
    for row, label in zip(shuffled_inputs, shuffled):
        row["y"] = f"shuffled-{label['y']}"
    rebuilt = build_inference_table(shuffled_inputs)
    assert inference_payload_unchanged(table, rebuilt)

    no_labels = deepcopy(rows)
    for row in no_labels:
        row.pop("y", None)
    without_labels = build_inference_table(no_labels)
    assert inference_payload_unchanged(table, without_labels)

    duplicate = labels + labels[:1]
    with pytest.raises(ConfigError, match="Duplicate evaluation-label key"):
        align_inference_and_labels(table, duplicate)


def test_historical_inventory_by_outer_fold_and_empty_source_is_a_deliverable():
    folds = synthetic_contract_folds()
    inventories = {
        item.fold_name: item
        for item in inventory_historical_pairs_by_fold(folds, synthetic_historical_records())
    }

    alpha = inventories["minus-storm-alpha"]
    assert alpha.has_eligible_historical_source is True
    assert alpha.historically_trained_q_allowed is True
    assert alpha.table_role == "historically_trained_q"
    assert alpha.eligible_source_events == ("storm-beta", "storm-gamma")
    assert alpha.reporting_model_version.endswith("checkpoint_sha256=TBD")
    assert alpha.seed == 1
    assert alpha.reporting_model_defines_u is True
    # The in-sample beta report, unknown Y, and cloudy row are not eligible sources.
    assert alpha.in_sample_pair_count == 1
    eligible_ids = {pair.building_uid for pair in alpha.pairs if pair.used_as_historical_source}
    assert eligible_ids == {"src-beta-1", "src-beta-2", "src-gamma-1"}
    assert all(pair.out_of_sample for pair in alpha.pairs if pair.used_as_historical_source)
    assert all(pair.event not in alpha.holdout_events for pair in alpha.pairs)

    seed2 = inventories["minus-storm-alpha-seed2"]
    assert seed2.seed == 2
    assert seed2.reporting_model_defines_u is False
    assert seed2.has_eligible_historical_source is True

    orphan = inventories["minus-storm-orphan"]
    # Alpha, beta, and gamma exist as candidate sources, but only the eligible
    # OOS four-class pairs authorize Q. Those exist here from alpha/beta/gamma.
    # The orphan event itself is the holdout, so the fold does have sources.
    # A no-source case is minus-storm-orphan when we restrict records to orphan only.
    assert "storm-orphan" not in orphan.eligible_source_events

    only_orphan = [
        record
        for record in synthetic_historical_records()
        if record["event"] == "storm-orphan"
    ]
    empty = inventory_historical_pairs_by_fold(
        [fold for fold in folds if fold.name == "minus-storm-orphan"],
        only_orphan,
    )[0]
    assert empty.has_eligible_historical_source is False
    assert empty.historically_trained_q_allowed is False
    assert empty.table_role == "descriptive"
    assert empty.eligible_source_events == ()
    assert empty.eligible_pair_count == 0
    assert "no eligible historical source event" in empty.notes
    assert "descriptive" in empty.notes


def test_declared_out_of_sample_must_match_the_reporting_head():
    folds = synthetic_contract_folds()
    record = deepcopy(synthetic_historical_records()[0])
    record["out_of_sample"] = False
    record["reporting_fold"] = "minus-storm-beta"
    with pytest.raises(ConfigError, match="out_of_sample"):
        inventory_historical_pairs_by_fold(folds, [record])


def test_duplicate_inference_keys_are_rejected():
    rows = synthetic_probability_rows()[:1]
    with pytest.raises(ConfigError, match="Duplicate inference key"):
        build_inference_table(rows + rows)

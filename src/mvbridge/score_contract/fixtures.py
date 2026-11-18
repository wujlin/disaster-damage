"""Synthetic probability and label fixtures for the score contract.

These generators do not read real records, official scores, or checkpoints.
"""

from __future__ import annotations

from typing import Any

from mvbridge.data.folds import FoldSpec


def make_fold(
    name: str,
    holdout_event: str,
    *,
    seed: int = 1,
    role: str = "main",
    enabled: bool = True,
    claim_evidence: bool = True,
) -> FoldSpec:
    return FoldSpec(
        name=name,
        holdout_events=(holdout_event,),
        role=role,
        enabled=enabled,
        claim_evidence=claim_evidence,
        seed=seed,
    )


def synthetic_contract_folds() -> tuple[FoldSpec, ...]:
    """Three claim folds plus one seed repeat and one fold with no source event."""
    return (
        make_fold("minus-storm-alpha", "storm-alpha", role="main", seed=1),
        make_fold(
            "minus-storm-alpha-seed2",
            "storm-alpha",
            role="seed-repeat",
            seed=2,
            claim_evidence=False,
        ),
        make_fold("minus-storm-beta", "storm-beta", role="secondary", seed=1),
        make_fold("minus-storm-gamma", "storm-gamma", role="secondary", seed=1),
        make_fold("minus-storm-orphan", "storm-orphan", role="secondary", seed=1),
    )


def softmax4(weights: tuple[float, float, float, float]) -> list[float]:
    import math

    max_weight = max(weights)
    exps = [math.exp(weight - max_weight) for weight in weights]
    total = sum(exps)
    return [value / total for value in exps]


def synthetic_probability_rows() -> list[dict[str, Any]]:
    """Rows that exercise entropy, 1-max p, class order, ties, and seeds."""
    return [
        {
            "image_id": "storm-alpha_0001",
            "building_uid": "b-peak-major",
            "disaster": "storm-alpha",
            "fold": "minus-storm-alpha",
            "seed": 1,
            "reporting_model_version": "fold=minus-storm-alpha/seed=1/role=main/claim_evidence=true/checkpoint_sha256=TBD",
            "out_of_sample": True,
            "probabilities": [0.05, 0.10, 0.80, 0.05],
            "y": "major-damage",
        },
        {
            "image_id": "storm-alpha_0002",
            "building_uid": "b-uniform",
            "disaster": "storm-alpha",
            "fold": "minus-storm-alpha",
            "seed": 1,
            "probabilities": [0.25, 0.25, 0.25, 0.25],
            "y": "no-damage",
        },
        {
            "image_id": "storm-alpha_0003",
            "building_uid": "b-onehot-destroyed",
            "disaster": "storm-alpha",
            "fold": "minus-storm-alpha",
            "seed": 1,
            "probabilities": [0.0, 0.0, 0.0, 1.0],
            "y": "destroyed",
        },
        {
            "image_id": "storm-alpha_0004",
            "building_uid": "b-tie-no-minor",
            "disaster": "storm-alpha",
            "fold": "minus-storm-alpha",
            "seed": 1,
            "probabilities": [0.4, 0.4, 0.2, 0.0],
            "y": "minor-damage",
        },
        {
            "image_id": "storm-alpha_0005",
            "building_uid": "b-tie-all",
            "disaster": "storm-alpha",
            "fold": "minus-storm-alpha-seed2",
            "seed": 2,
            "probabilities": [0.25, 0.25, 0.25, 0.25],
            "y": "no-damage",
        },
    ]


def synthetic_historical_records() -> list[dict[str, Any]]:
    """Eligible, ineligible, in-sample, and orphan-event historical records."""
    common_ok = {
        "reference_available": True,
        "post_event_intersects_footprint": True,
        "clear_valid_pixel_fraction": 0.93,
        "match_accepted": True,
    }
    records = [
        {
            "image_id": "storm-beta_0100",
            "building_uid": "src-beta-1",
            "event": "storm-beta",
            "s": "no-damage",
            "y": "minor-damage",
            "z": {"clear_sky": True, "neighborhood_alarm_30m": 0.1},
            "reporting_fold": "minus-storm-beta",
            **common_ok,
        },
        {
            "image_id": "storm-beta_0101",
            "building_uid": "src-beta-2",
            "event": "storm-beta",
            "s": "destroyed",
            "y": "destroyed",
            "z": {"clear_sky": True, "neighborhood_alarm_30m": 0.8},
            "reporting_fold": "minus-storm-beta",
            **common_ok,
        },
        {
            "image_id": "storm-gamma_0200",
            "building_uid": "src-gamma-1",
            "event": "storm-gamma",
            "s": "minor-damage",
            "y": "major-damage",
            "z": {"clear_sky": False, "neighborhood_alarm_30m": 0.4},
            "reporting_fold": "minus-storm-gamma",
            **common_ok,
        },
        {
            "image_id": "storm-gamma_0201",
            "building_uid": "src-gamma-unknown",
            "event": "storm-gamma",
            "s": "no-damage",
            "y": "unknown",
            "z": {"clear_sky": True},
            "reporting_fold": "minus-storm-gamma",
            **common_ok,
        },
        {
            "image_id": "storm-gamma_0202",
            "building_uid": "src-gamma-cloud",
            "event": "storm-gamma",
            "s": "major-damage",
            "y": "major-damage",
            "z": {"clear_sky": False},
            "reporting_fold": "minus-storm-gamma",
            "reference_available": True,
            "post_event_intersects_footprint": True,
            "clear_valid_pixel_fraction": 0.42,
            "match_accepted": True,
        },
        {
            "image_id": "storm-alpha_0300",
            "building_uid": "src-alpha-1",
            "event": "storm-alpha",
            "s": "no-damage",
            "y": "no-damage",
            "z": {"clear_sky": True},
            "reporting_fold": "minus-storm-alpha",
            **common_ok,
        },
        {
            "image_id": "storm-beta_0102",
            "building_uid": "src-beta-insample",
            "event": "storm-beta",
            "s": "minor-damage",
            "y": "minor-damage",
            "z": {"clear_sky": True},
            "reporting_fold": "minus-storm-alpha",
            **common_ok,
        },
        {
            "image_id": "storm-orphan_0400",
            "building_uid": "orphan-unmapped",
            "event": "storm-orphan",
            "s": "no-damage",
            "y": "ds1",
            "z": {"clear_sky": True},
            "reporting_fold": "minus-storm-orphan",
            **common_ok,
        },
        {
            "image_id": "storm-orphan_0401",
            "building_uid": "orphan-no-ref",
            "event": "storm-orphan",
            "s": "major-damage",
            "y": "major-damage",
            "z": {},
            "reporting_fold": "minus-storm-orphan",
            "reference_available": False,
            "post_event_intersects_footprint": True,
            "clear_valid_pixel_fraction": 0.99,
            "match_accepted": True,
        },
    ]
    return records


def synthetic_evaluation_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for row in rows:
        labels.append(
            {
                "image_id": row["image_id"],
                "building_uid": row["building_uid"],
                "y": row.get("y"),
                "f": False,
            }
        )
    return labels

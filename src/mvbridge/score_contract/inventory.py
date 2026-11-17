"""Split inventory of historically eligible (S, Y, Z) pairs by outer fold.

A fold with no eligible historical source is a valid deliverable: that fold's
table may stay descriptive, and a historically trained Q is not allowed until
eligible source events exist. This module records the reporting-model version
and whether each historical report was out-of-sample. It does not train Q and
does not score official events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from mvbridge.data.folds import FoldSpec, route_oof
from mvbridge.errors import ConfigError
from mvbridge.score_contract.eligibility import four_class_eligibility
from mvbridge.score_contract.protocol import (
    SEED_REPEAT_ROLE,
    U_DEFINING_SEED,
    ArtifactInventory,
    ArtifactRecord,
)

DESCRIPTIVE_ROLE = "descriptive"
TRAINABLE_Q_ROLE = "historically_trained_q"


@dataclass(frozen=True)
class HistoricalPair:
    image_id: str
    building_uid: str
    event: str
    s: str | None
    y: str | None
    z: dict[str, Any]
    coverage_class: str
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    in_q: bool
    reporting_model_version: str
    reporting_fold: str
    reporting_seed: int | None
    out_of_sample: bool
    used_as_historical_source: bool
    source_for_folds: tuple[str, ...]


@dataclass(frozen=True)
class FoldHistoricalInventory:
    fold_name: str
    holdout_events: tuple[str, ...]
    seed: int
    role: str
    claim_evidence: bool
    reporting_model_version: str
    reporting_model_defines_u: bool
    checkpoint_sha256: str | None
    manifest_sha256: str | None
    eligible_source_events: tuple[str, ...]
    ineligible_or_missing_source_events: tuple[str, ...]
    eligible_pair_count: int
    inventoried_pair_count: int
    out_of_sample_eligible_pair_count: int
    in_sample_pair_count: int
    coverage_counts: dict[str, int]
    has_eligible_historical_source: bool
    table_role: str
    historically_trained_q_allowed: bool
    notes: str
    pairs: tuple[HistoricalPair, ...] = field(default_factory=tuple)


def reporting_model_defines_u(fold: FoldSpec) -> bool:
    """Seed-1 claim heads define U. Seed repeats do not, even at the same holdout."""
    return bool(
        fold.enabled
        and fold.claim_evidence
        and int(fold.seed) == U_DEFINING_SEED
        and fold.role != SEED_REPEAT_ROLE
    )


def reporting_model_version(fold: FoldSpec, artifact: ArtifactRecord | None = None) -> str:
    """Stable version string from the authorized inventory, never an invented hash."""
    digest = None
    if artifact is not None:
        digest = artifact.checkpoint_sha256
    parts = [
        f"fold={fold.name}",
        f"seed={int(fold.seed)}",
        f"role={fold.role}",
        f"claim_evidence={'true' if fold.claim_evidence else 'false'}",
        f"checkpoint_sha256={digest or 'TBD'}",
    ]
    return "/".join(parts)


def inventory_historical_pairs_by_fold(
    folds: Sequence[FoldSpec],
    records: Iterable[Mapping[str, Any]],
    *,
    artifacts: ArtifactInventory | None = None,
    claim_only: bool = True,
    include_pairs: bool = True,
) -> tuple[FoldHistoricalInventory, ...]:
    """Inventory historical (S, Y, Z) pairs for every outer fold.

    For fold F, a pair from event e is a historical source only when e is not
    held out by F, the pair is four-class eligible, and the satellite report
    was produced out-of-sample by the authorized reporting head for e.
    """
    fold_list = list(folds)
    parsed = [_parse_history_record(record, fold_list, claim_only=claim_only) for record in records]
    inventories = [
        _inventory_one_fold(fold, parsed, artifacts=artifacts, include_pairs=include_pairs)
        for fold in fold_list
    ]
    return tuple(inventories)


def _inventory_one_fold(
    fold: FoldSpec,
    parsed: Sequence[HistoricalPair],
    *,
    artifacts: ArtifactInventory | None,
    include_pairs: bool,
) -> FoldHistoricalInventory:
    holdout = tuple(fold.holdout_events)
    artifact = artifacts.record(fold.name) if artifacts is not None else None
    version = reporting_model_version(fold, artifact)
    coverage_counts: dict[str, int] = {}
    source_pairs: list[HistoricalPair] = []
    eligible_events: set[str] = set()
    seen_events: set[str] = set()
    eligible_count = 0
    oos_eligible = 0
    in_sample = 0
    for pair in parsed:
        if pair.event in holdout:
            continue
        seen_events.add(pair.event)
        annotated = HistoricalPair(
            **{
                **pair.__dict__,
                "source_for_folds": tuple(
                    name for name in pair.source_for_folds if name == fold.name
                )
                or pair.source_for_folds,
                "used_as_historical_source": pair.event not in holdout
                and pair.eligible
                and pair.out_of_sample,
            }
        )
        source_pairs.append(annotated)
        coverage_counts[annotated.coverage_class] = coverage_counts.get(annotated.coverage_class, 0) + 1
        if annotated.out_of_sample is False:
            in_sample += 1
        if annotated.eligible and annotated.out_of_sample:
            eligible_count += 1
            oos_eligible += 1
            eligible_events.add(annotated.event)
    missing_or_ineligible = tuple(sorted(seen_events - eligible_events))
    has_source = bool(eligible_events)
    if has_source:
        table_role = TRAINABLE_Q_ROLE
        notes = (
            f"Fold {fold.name} has {len(eligible_events)} eligible historical source event(s) "
            f"and {eligible_count} eligible out-of-sample (S, Y, Z) pair(s)."
        )
    else:
        table_role = DESCRIPTIVE_ROLE
        notes = (
            f"Fold {fold.name} has no eligible historical source event. "
            "This inventory is itself a deliverable: the table stays descriptive, "
            "and a historically trained Q is not allowed until eligible source events exist."
        )
    return FoldHistoricalInventory(
        fold_name=fold.name,
        holdout_events=holdout,
        seed=int(fold.seed),
        role=fold.role,
        claim_evidence=bool(fold.claim_evidence),
        reporting_model_version=version,
        reporting_model_defines_u=reporting_model_defines_u(fold),
        checkpoint_sha256=None if artifact is None else artifact.checkpoint_sha256,
        manifest_sha256=None if artifact is None else artifact.manifest_sha256,
        eligible_source_events=tuple(sorted(eligible_events)),
        ineligible_or_missing_source_events=missing_or_ineligible,
        eligible_pair_count=eligible_count,
        inventoried_pair_count=len(source_pairs),
        out_of_sample_eligible_pair_count=oos_eligible,
        in_sample_pair_count=in_sample,
        coverage_counts=dict(sorted(coverage_counts.items())),
        has_eligible_historical_source=has_source,
        table_role=table_role,
        historically_trained_q_allowed=has_source,
        notes=notes,
        pairs=tuple(source_pairs) if include_pairs else tuple(),
    )


def _parse_history_record(
    record: Mapping[str, Any],
    folds: Sequence[FoldSpec],
    *,
    claim_only: bool,
) -> HistoricalPair:
    image_id = str(record.get("image_id") or "").strip()
    building_uid = str(record.get("building_uid") or "").strip()
    event = str(record.get("event") or record.get("disaster") or "").strip()
    if not image_id or not building_uid or not event:
        raise ConfigError("Historical pair needs image_id, building_uid, and event/disaster")
    eligibility = four_class_eligibility(record)
    z_raw = record.get("z", record.get("Z", {}))
    if z_raw is None:
        z: dict[str, Any] = {}
    elif isinstance(z_raw, Mapping):
        z = dict(z_raw)
    else:
        raise ConfigError("Z must be a mapping of pre-survey covariates, or omitted")
    reporting_fold_name = str(record.get("reporting_fold") or record.get("fold") or "").strip()
    reporting_fold = _resolve_reporting_fold(event, reporting_fold_name, folds, claim_only=claim_only)
    declared_oos = record.get("out_of_sample")
    if reporting_fold is None:
        out_of_sample = False
        version = str(record.get("reporting_model_version") or "").strip() or "unroutable"
        reporting_name = reporting_fold_name
        reporting_seed = _optional_int(record.get("seed"))
    else:
        out_of_sample = event in reporting_fold.holdout_events
        version = str(record.get("reporting_model_version") or "").strip() or reporting_model_version(reporting_fold)
        reporting_name = reporting_fold.name
        reporting_seed = int(reporting_fold.seed)
        if declared_oos is not None and bool(declared_oos) != out_of_sample:
            raise ConfigError(
                f"Pair {image_id}/{building_uid} declared out_of_sample={declared_oos}, "
                f"but reporting fold {reporting_fold.name} "
                f"{'holds out' if out_of_sample else 'did not hold out'} {event!r}."
            )
    source_for = tuple(
        fold.name
        for fold in folds
        if event not in fold.holdout_events
    )
    used = eligibility.eligible and out_of_sample
    return HistoricalPair(
        image_id=image_id,
        building_uid=building_uid,
        event=event,
        s=eligibility.s_class,
        y=eligibility.y_class,
        z=z,
        coverage_class=eligibility.coverage_class,
        eligible=eligibility.eligible,
        eligibility_reasons=eligibility.reasons,
        in_q=eligibility.in_q,
        reporting_model_version=version,
        reporting_fold=reporting_name,
        reporting_seed=reporting_seed,
        out_of_sample=out_of_sample,
        used_as_historical_source=used,
        source_for_folds=source_for,
    )


def _resolve_reporting_fold(
    event: str,
    requested_name: str,
    folds: Sequence[FoldSpec],
    *,
    claim_only: bool,
) -> FoldSpec | None:
    if requested_name:
        match = next((fold for fold in folds if fold.name == requested_name), None)
        if match is None:
            raise ConfigError(f"Unknown reporting fold {requested_name!r}")
        return match
    eligible = route_oof(event, folds, claim_only=claim_only)
    if not eligible:
        return None
    if len(eligible) > 1:
        # Default routing keeps seed-1 claim heads only. If more than one remains,
        # the record must name the reporting fold explicitly.
        define_u = [fold for fold in eligible if reporting_model_defines_u(fold)]
        if len(define_u) == 1:
            return define_u[0]
        names = ", ".join(fold.name for fold in eligible)
        raise ConfigError(
            f"Event {event!r} is held out by {names}. "
            "Name reporting_fold explicitly rather than averaging heads."
        )
    return eligible[0]


def _optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)

"""Support reporting for contrast slices and review strata."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from mvbridge.channel_bench.locks import DAMAGE_CLASSES
from mvbridge.channel_bench.schema import BuildingAnalysisRow, imprint_bin


PROPOSED_MIN_BUILDINGS = 20
PROPOSED_MIN_BLOCKS = 2
PROPOSED_MIN_PER_GRADE = 5


@dataclass(frozen=True)
class SliceSupport:
    slice_name: str
    n_buildings: int
    n_surveyed: int
    n_eligible: int
    n_blocks: int
    y_counts: dict[str, int]
    s_counts: dict[str, int]
    mismatch_count: int
    flags: tuple[str, ...] = ()

    @property
    def adequate(self) -> bool:
        return not self.flags


@dataclass
class SupportReport:
    slices: list[SliceSupport] = field(default_factory=list)
    proposed_min_buildings: int = PROPOSED_MIN_BUILDINGS
    proposed_min_blocks: int = PROPOSED_MIN_BLOCKS
    proposed_min_per_grade: int = PROPOSED_MIN_PER_GRADE
    notes: tuple[str, ...] = (
        "Support thresholds are proposed for this design version, not an official freeze.",
    )

    def as_dict(self) -> dict:
        return {
            "slices": [slice_.__dict__ for slice_ in self.slices],
            "proposed_min_buildings": self.proposed_min_buildings,
            "proposed_min_blocks": self.proposed_min_blocks,
            "proposed_min_per_grade": self.proposed_min_per_grade,
            "notes": list(self.notes),
        }


def _count_grades(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: int(counts.get(name, 0)) for name in DAMAGE_CLASSES}


def _flags(
    n_buildings: int,
    n_blocks: int,
    y_counts: Mapping[str, int],
    *,
    min_buildings: int,
    min_blocks: int,
    min_per_grade: int,
) -> tuple[str, ...]:
    flags: list[str] = []
    if n_buildings < min_buildings:
        flags.append("low_building_count")
    if n_blocks < min_blocks:
        flags.append("low_block_count")
    missing = [name for name, count in y_counts.items() if count < min_per_grade]
    if missing:
        flags.append("low_grade_support:" + ",".join(missing))
    return tuple(flags)


def support_report(
    rows: Sequence[BuildingAnalysisRow],
    *,
    comparable_visibility: str | None = None,
    min_buildings: int = PROPOSED_MIN_BUILDINGS,
    min_blocks: int = PROPOSED_MIN_BLOCKS,
    min_per_grade: int = PROPOSED_MIN_PER_GRADE,
) -> SupportReport:
    """Count grades, reports, and independent blocks in locked contrast slices."""
    groups: dict[str, list[BuildingAnalysisRow]] = defaultdict(list)
    for row in rows:
        if comparable_visibility is not None and row.visibility_bin != comparable_visibility:
            continue
        if not row.eligible:
            groups["ineligible"].append(row)
            continue
        bin_id = imprint_bin(row.imprint_score)
        groups[f"imprint_bin_{bin_id}"].append(row)
        groups[f"visibility_{row.visibility_bin}"].append(row)
        groups[f"visibility_{row.visibility_bin}_imprint_bin_{bin_id}"].append(row)
        groups["eligible_all"].append(row)
    slices: list[SliceSupport] = []
    for name in sorted(groups):
        items = groups[name]
        y_counts = _count_grades(item.y for item in items)
        s_counts = _count_grades(item.s for item in items)
        n_blocks = len({item.block_id for item in items})
        n_buildings = len(items)
        slices.append(
            SliceSupport(
                slice_name=name,
                n_buildings=n_buildings,
                n_surveyed=sum(1 for item in items if item.surveyed),
                n_eligible=sum(1 for item in items if item.eligible),
                n_blocks=n_blocks,
                y_counts=y_counts,
                s_counts=s_counts,
                mismatch_count=sum(1 for item in items if item.s != item.y),
                flags=_flags(
                    n_buildings,
                    n_blocks,
                    y_counts,
                    min_buildings=min_buildings,
                    min_blocks=min_blocks,
                    min_per_grade=min_per_grade,
                ),
            )
        )
    return SupportReport(
        slices=slices,
        proposed_min_buildings=min_buildings,
        proposed_min_blocks=min_blocks,
        proposed_min_per_grade=min_per_grade,
    )

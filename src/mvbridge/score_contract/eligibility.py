"""Four-class eligibility and the four coverage classes.

A pair is four-class eligible only when both the satellite report S and the
independent reference Y sit on the official four-class scale, an independent
reference is present, and satellite evidence meets the frozen 80% rule. Those
are the conditions for a historical (S, Y, Z) pair and for a scorable Q row.

Coverage classes are inventory statistics. They are assigned only from R and V,
never from entropy, 1-max p, or any other model score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mvbridge.errors import ConfigError
from mvbridge.score_contract.adapter import normalize_class_name
from mvbridge.score_contract.protocol import OFFICIAL_CLASS_ORDER, SATELLITE_CLEAR_PIXEL_MIN

COVERAGE_CLASSES = (
    "satellite+ground",
    "satellite-only",
    "ground-only",
    "insufficient evidence",
)

# Labels that may appear in source data but are not on the official 4-class scale.
INELIGIBLE_Y_TOKENS = frozenset(
    {
        "un-classified",
        "unclassified",
        "unknown",
        "excluded",
        "-1",
        "ds1",
        "none",
        "null",
        "nan",
    }
)


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...]
    y_class: str | None
    s_class: str | None
    reference_available: bool
    sufficient_satellite: bool
    coverage_class: str
    in_q: bool

    @property
    def scorable(self) -> bool:
        return self.in_q


def is_four_class_label(name: str | None, class_order: tuple[str, ...] = OFFICIAL_CLASS_ORDER) -> bool:
    normalized = normalize_class_name(name)
    if normalized is None or normalized in INELIGIBLE_Y_TOKENS:
        return False
    return normalized in class_order


def sufficient_satellite_evidence(
    *,
    post_event_intersects_footprint: bool,
    clear_valid_pixel_fraction: float | None,
    clear_pixel_min: float = SATELLITE_CLEAR_PIXEL_MIN,
) -> bool:
    """Frozen V=1 rule: a post-event intersection and at least 80% clear pixels."""
    if not post_event_intersects_footprint:
        return False
    if clear_valid_pixel_fraction is None:
        return False
    try:
        fraction = float(clear_valid_pixel_fraction)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"clear_valid_pixel_fraction must be a number, got {clear_valid_pixel_fraction!r}."
        ) from exc
    if fraction < 0.0 or fraction > 1.0:
        raise ConfigError(f"clear_valid_pixel_fraction must be in [0, 1], got {fraction}.")
    return fraction >= clear_pixel_min


def assign_coverage_class(*, reference_available: bool, sufficient_satellite: bool) -> str:
    if reference_available and sufficient_satellite:
        return "satellite+ground"
    if sufficient_satellite and not reference_available:
        return "satellite-only"
    if reference_available and not sufficient_satellite:
        return "ground-only"
    return "insufficient evidence"


def four_class_eligibility(
    record: Mapping[str, Any],
    *,
    class_order: tuple[str, ...] = OFFICIAL_CLASS_ORDER,
    clear_pixel_min: float = SATELLITE_CLEAR_PIXEL_MIN,
) -> Eligibility:
    """Decide whether a record may enter a four-class (S, Y) comparison."""
    y_class = _optional_class(record.get("y", record.get("Y", record.get("label"))))
    s_class = _optional_class(record.get("s", record.get("S", record.get("y_hat_s"))))
    reference_available = _required_bool(record, "reference_available", alts=("r", "R"))
    post_intersects = _required_bool(
        record,
        "post_event_intersects_footprint",
        alts=("post_intersects",),
        default=record.get("sufficient_satellite"),
    )
    fraction = record.get("clear_valid_pixel_fraction", record.get("clear_pixel_fraction"))
    if "sufficient_satellite" in record and fraction is None and record.get("post_event_intersects_footprint") is None:
        sufficient = _required_bool(record, "sufficient_satellite", alts=("v", "V"))
    else:
        # A missing intersection flag is treated as not intersecting unless V was
        # supplied as a precomputed boolean above.
        intersects = bool(post_intersects) if post_intersects is not None else False
        sufficient = sufficient_satellite_evidence(
            post_event_intersects_footprint=intersects,
            clear_valid_pixel_fraction=None if fraction is None else float(fraction),
            clear_pixel_min=clear_pixel_min,
        )
    match_ok = _match_accepted(record)
    coverage = assign_coverage_class(
        reference_available=reference_available,
        sufficient_satellite=sufficient,
    )
    reasons: list[str] = []
    if not is_four_class_label(y_class, class_order=class_order):
        reasons.append("y_not_four_class")
    if not is_four_class_label(s_class, class_order=class_order):
        reasons.append("s_not_four_class")
    if not reference_available:
        reasons.append("reference_unavailable")
    if not sufficient:
        reasons.append("insufficient_satellite_evidence")
    if not match_ok:
        reasons.append("ambiguous_or_dropped_match")
    in_q = reference_available and sufficient
    eligible = not reasons
    return Eligibility(
        eligible=eligible,
        reasons=tuple(reasons),
        y_class=normalize_class_name(y_class) if y_class is not None else None,
        s_class=normalize_class_name(s_class) if s_class is not None else None,
        reference_available=reference_available,
        sufficient_satellite=sufficient,
        coverage_class=coverage,
        in_q=in_q,
    )


def _optional_class(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_bool(record: Mapping[str, Any], key: str, *, alts: tuple[str, ...] = (), default=None) -> bool:
    for name in (key, *alts):
        if name in record and record[name] is not None:
            value = record[name]
            if isinstance(value, bool):
                return value
            raise ConfigError(f"{name} must be a boolean, got {value!r}")
    if default is not None:
        if isinstance(default, bool):
            return default
        raise ConfigError(f"{key} must be a boolean, got {default!r}")
    raise ConfigError(f"Record needs boolean {key}")


def _match_accepted(record: Mapping[str, Any]) -> bool:
    if record.get("ambiguous_match") is True or record.get("dropped_match") is True:
        return False
    if "match_accepted" in record and record["match_accepted"] is not None:
        value = record["match_accepted"]
        if not isinstance(value, bool):
            raise ConfigError(f"match_accepted must be a boolean, got {value!r}")
        return value
    return True

"""Ground-side Palu checks: dates, footprints, unresolved classes.

These checks never consume model predictions and never compute U or
prediction-based conflict counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from mvbridge.crosswalk.ledger import candidate_schemes
from mvbridge.errors import CrosswalkError

# Columns that would turn a ground-side table into a model-conflict computation.
PREDICTION_FIELDS = frozenset(
    {
        "u",
        "u_e",
        "u_f",
        "conflict",
        "conflicts",
        "y_hat",
        "yhat",
        "y_s_hat",
        "y_hat_s",
        "hat_y_s",
        "predicted_class",
        "predicted_label",
        "prediction",
        "logits",
        "prob_conflict",
    }
)

SCHEME_Y_RULES = {
    "strict_ds1_unknown_exclude_ds2_ds3": {
        "DS0": None,
        "DS1": "unknown_or_excluded_from_y",
        "DS2": "not_mapped_to_y",
        "DS3": "not_mapped_to_y",
    },
    "exploratory_ds2_major_ds3_destroyed": {
        "DS0": None,
        "DS1": "unknown_or_excluded_from_y",
        "DS2": "major",
        "DS3": "destroyed",
    },
    "ds0_to_no_listed_separately": {
        "DS0": "no",
        "DS1": None,
        "DS2": None,
        "DS3": None,
    },
}

Y_GRADES = {"no", "minor", "major", "destroyed"}


def assert_no_prediction_fields(records: list[dict[str, Any]], *, context: str) -> None:
    """Refuse tables that include model outputs or U / conflict labels."""
    banned: set[str] = set()
    for row in records:
        for key in row:
            normalized = str(key).strip().lower().replace(" ", "_")
            if normalized in PREDICTION_FIELDS:
                banned.add(str(key))
    if banned:
        names = ", ".join(sorted(banned))
        raise CrosswalkError(
            f"{context} includes model-prediction or U/conflict fields ({names}). "
            "Palu crosswalk checks are ground-reference only and must not compute U "
            "or prediction-based conflict counts."
        )


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """IoU of two axis-aligned boxes (min_x, min_y, max_x, max_y) in a planar CRS."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def point_to_box_distance(x: float, y: float, box: tuple[float, float, float, float]) -> float:
    """Euclidean distance from a point to an axis-aligned box, in the box CRS."""
    x0, y0, x1, y1 = box
    clamped_x = min(max(x, x0), x1)
    clamped_y = min(max(y, y0), y1)
    return float(((x - clamped_x) ** 2 + (y - clamped_y) ** 2) ** 0.5)


def match_records_to_footprints(
    records: list[dict[str, Any]],
    footprints: list[dict[str, Any]],
    *,
    iou_match_min: float = 0.5,
    ambiguity_iou: float = 0.3,
    best_second_ratio_min: float = 1.2,
    localization_error_m_max: float = 30.0,
) -> list[dict[str, Any]]:
    """Attach frozen footprint-matching outcomes to ground records.

    A polygon record uses IoU against open footprints. A point record uses
    planar distance, with 30 m as the localization-error limit only.
    """
    results: list[dict[str, Any]] = []
    for record in records:
        if record.get("box") is not None:
            scored = _score_polygon(record["box"], footprints)
            decision = _decide_polygon_match(scored, iou_match_min, ambiguity_iou, best_second_ratio_min)
        else:
            scored = _score_point(float(record["x"]), float(record["y"]), footprints)
            decision = _decide_point_match(scored, localization_error_m_max, best_second_ratio_min)
        row = dict(record)
        row["match"] = decision
        results.append(row)
    return results


def _as_box(item: dict[str, Any]) -> tuple[float, float, float, float]:
    box = item.get("box")
    if box is None:
        raise CrosswalkError(f"Footprint {item.get('id')!r} is missing a box.")
    x0, y0, x1, y1 = (float(part) for part in box)
    return (x0, y0, x1, y1)


def _score_polygon(box: list[float] | tuple[float, ...], footprints: list[dict[str, Any]]) -> list[tuple[str, float]]:
    src = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    scored = [(str(item["id"]), box_iou(src, _as_box(item))) for item in footprints]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _score_point(x: float, y: float, footprints: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    scored: list[tuple[str, float, float]] = []
    for item in footprints:
        box = _as_box(item)
        distance = point_to_box_distance(x, y, box)
        # Higher is better: inside the box scores 1, otherwise decay with distance.
        score = 1.0 if distance == 0.0 else 1.0 / (1.0 + distance)
        scored.append((str(item["id"]), score, distance))
    scored.sort(key=lambda row: (row[1], -row[2]), reverse=True)
    return scored


def _decide_polygon_match(
    scored: list[tuple[str, float]],
    iou_match_min: float,
    ambiguity_iou: float,
    best_second_ratio_min: float,
) -> dict[str, Any]:
    above = [(fid, iou) for fid, iou in scored if iou >= ambiguity_iou]
    if len(above) >= 2:
        # Protocol: ambiguous when two or more candidates have IoU >= 0.3.
        return {
            "status": "ambiguous",
            "reason": "two_or_more_candidates_iou_ge_0_3_or_ratio_lt_1_2",
            "candidates": [{"id": fid, "iou": iou} for fid, iou in above],
        }
    if len(scored) >= 2 and scored[1][1] > 0 and (scored[0][1] / scored[1][1]) < best_second_ratio_min:
        return {
            "status": "ambiguous",
            "reason": "two_or_more_candidates_iou_ge_0_3_or_ratio_lt_1_2",
            "candidates": [{"id": fid, "iou": iou} for fid, iou in scored[:2]],
        }
    if not scored or scored[0][1] < iou_match_min:
        return {"status": "unmatched", "reason": "iou_below_0_5", "candidates": []}
    return {
        "status": "matched",
        "footprint_id": scored[0][0],
        "iou": scored[0][1],
        "reason": "iou_ge_0_5_unique",
    }


def _decide_point_match(
    scored: list[tuple[str, float, float]],
    localization_error_m_max: float,
    best_second_ratio_min: float,
) -> dict[str, Any]:
    within = [(fid, score, dist) for fid, score, dist in scored if dist <= localization_error_m_max]
    if not within:
        best = scored[0] if scored else None
        return {
            "status": "dropped_localization",
            "reason": "localization_error_gt_30m",
            "distance_m": None if best is None else best[2],
        }
    if len(within) >= 2:
        best, second = within[0], within[1]
        # Two footprints equally contain the point, or the score ratio is weak.
        both_inside = best[2] == 0.0 and second[2] == 0.0
        ratio = (best[1] / second[1]) if second[1] > 0 else float("inf")
        if both_inside or ratio < best_second_ratio_min:
            return {
                "status": "ambiguous",
                "reason": "two_or_more_candidates_or_ratio_lt_1_2",
                "candidates": [{"id": fid, "distance_m": dist} for fid, _, dist in within],
            }
    return {
        "status": "matched",
        "footprint_id": within[0][0],
        "distance_m": within[0][2],
        "reason": "unique_within_30m",
    }


def multipoint_inconsistencies(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Footprints that received two or more disagreeing ground classes."""
    by_footprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched:
        match = row.get("match") or {}
        if match.get("status") != "matched":
            continue
        by_footprint[str(match["footprint_id"])].append(row)
    issues: list[dict[str, Any]] = []
    for footprint_id, rows in sorted(by_footprint.items()):
        classes = {str(row.get("source_class")) for row in rows}
        if len(rows) >= 2 and len(classes) >= 2:
            issues.append(
                {
                    "footprint_id": footprint_id,
                    "n_points": len(rows),
                    "classes": sorted(classes),
                    "record_ids": [row.get("id") for row in rows],
                }
            )
    return issues


def date_phase_flags(record: dict[str, Any], alignment: dict[str, Any]) -> list[str]:
    """Return date/phase flags. Missing values stay unresolved, not imputed."""
    flags: list[str] = []
    survey = record.get("survey_date")
    imagery = record.get("imagery_date")
    phase = record.get("phase")
    catalog_start = alignment.get("catalog_start")
    catalog_end = alignment.get("catalog_end")
    cited = alignment.get("commonly_cited_event_date")
    if not survey:
        flags.append("missing_survey_date")
    if not imagery:
        flags.append("missing_imagery_date")
    if not phase:
        flags.append("missing_phase")
    if survey and catalog_start and survey < str(catalog_start):
        flags.append("survey_before_catalog_start")
    if survey and catalog_end and survey > str(catalog_end):
        flags.append("survey_after_catalog_end")
    if survey and cited and survey < str(cited):
        flags.append("survey_before_cited_event_date")
    if catalog_start and cited and str(catalog_start) != str(cited):
        flags.append("catalog_window_differs_from_cited_event_date")
    if survey and imagery and str(imagery) < str(survey):
        flags.append("imagery_precedes_survey")
    return flags


def map_source_class(source_class: str | None, scheme_id: str) -> str:
    """Apply a candidate scheme. Unlisted or None means not_in_this_scheme."""
    rules = SCHEME_Y_RULES.get(scheme_id)
    if rules is None:
        raise CrosswalkError(f"Unknown candidate scheme {scheme_id!r}.")
    if source_class not in rules or rules[source_class] is None:
        return "not_in_this_scheme"
    return str(rules[source_class])


def scheme_inventory(records: list[dict[str, Any]], scheme_id: str) -> dict[str, Any]:
    """Ground-side counts under one candidate scheme. Not a main conclusion."""
    mapped = Counter()
    for row in records:
        mapped[map_source_class(row.get("source_class"), scheme_id)] += 1
    scorable = sum(mapped[grade] for grade in Y_GRADES)
    return {
        "scheme_id": scheme_id,
        "is_main_conclusion": False,
        "write_to_official_protocol_as_frozen": False,
        "n_records": len(records),
        "mapped_counts": dict(mapped),
        "n_with_y_in_four_class_scale": scorable,
        "note": (
            "Candidate-scheme inventory only. These counts are not official Y, "
            "not U, and not a frozen mapping."
        ),
    }


def run_ground_side_checks(
    fixture: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Run date, footprint, and class inventories on a ground-reference table."""
    records = list(fixture.get("ground_records") or [])
    footprints = list(fixture.get("footprints") or [])
    assert_no_prediction_fields(records, context="Ground records")
    assert_no_prediction_fields(footprints, context="Footprints")
    rules = ledger.get("footprint_checks") or {}
    matched = match_records_to_footprints(
        records,
        footprints,
        iou_match_min=float(rules.get("iou_match_min", 0.5)),
        ambiguity_iou=float(rules.get("ambiguity_iou", 0.3)),
        best_second_ratio_min=float(rules.get("best_second_ratio_min", 1.2)),
        localization_error_m_max=float(rules.get("localization_error_m_max", 30.0)),
    )
    alignment = ledger.get("date_phase_alignment") or {}
    date_rows = []
    date_counts: Counter[str] = Counter()
    for row in matched:
        flags = date_phase_flags(row, alignment)
        date_counts.update(flags)
        date_rows.append({"id": row.get("id"), "flags": flags})
    match_counts = Counter(row["match"]["status"] for row in matched)
    class_counts = Counter(str(row.get("source_class")) for row in matched)
    schemes = []
    for row in candidate_schemes(ledger):
        schemes.append(scheme_inventory(matched, str(row["id"])))
    unresolved = []
    for row in ledger.get("unresolved_classes") or []:
        if isinstance(row, dict):
            unresolved.append(
                {
                    "class": row.get("class"),
                    "status": row.get("status"),
                    "n_in_fixture": class_counts.get(str(row.get("class")), 0),
                    "official_guidance": row.get("official_guidance"),
                }
            )
    return {
        "kind": "ground_side_checks",
        "computes_u": False,
        "computes_prediction_conflicts": False,
        "cross_hazard_transfer_established": False,
        "n_records": len(matched),
        "n_footprints": len(footprints),
        "source_class_distribution": dict(class_counts),
        "footprint_match_counts": dict(match_counts),
        "multipoint_inconsistencies": multipoint_inconsistencies(matched),
        "date_phase_flag_counts": dict(date_counts),
        "date_phase_rows": date_rows,
        "candidate_scheme_inventories": schemes,
        "unresolved_classes": unresolved,
        "fixture_note": fixture.get("note"),
    }

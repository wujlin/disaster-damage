"""Pending-decision / approved validator for the Palu mapping ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mvbridge.crosswalk.ledger import (
    OFFICIAL_PROTOCOL_PATH,
    PALU_PROTOCOL_SENTENCES,
    PAULIK_2019_ID,
    STEER_PROTOCOL_SENTENCES,
    STEER_WIND_FROZEN_RULES,
    STEER_WIND_FROZEN_SOURCE,
    STEER_WIND_ID,
    candidate_schemes,
    frozen_mapping_by_id,
    source_by_id,
)
from mvbridge.errors import CrosswalkError

APPROVED = "approved"
PENDING = "pending"
PENDING_DECISION = "pending_decision"
OBTAINED = "obtained"
OBTAINED_AWAITING_ADJUDICATION = "obtained_on_wsa_awaiting_adjudication"
NOT_APPLIED = "not_applied"
ALLOWED_PAULIK_OBTAINMENT = {OBTAINED, OBTAINED_AWAITING_ADJUDICATION}
JLIN_ADJUDICATION_ID = "jlin_adjudication_of_dl_mapping"

FORBIDDEN_TRANSFER_KEYS = (
    "cross_hazard_transfer_established",
    "cross_disaster_transfer_established",
    "cross_hazard_transfer",
)


@dataclass
class ValidationIssue:
    """One ledger or protocol violation."""

    code: str
    message: str
    path: str = ""


@dataclass
class ValidationReport:
    """Result of checking a ledger against the pending/approved rules."""

    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        lines = [f"{item.code}: {item.message}" for item in self.issues]
        raise CrosswalkError("Palu crosswalk validation failed:\n" + "\n".join(lines))


def validate_ledger(
    ledger: dict[str, Any],
    *,
    protocol_path: str | Path | None = None,
) -> ValidationReport:
    """Validate statuses, frozen wind rules, and Palu pending constraints."""
    issues: list[ValidationIssue] = []
    issues.extend(_check_meta(ledger))
    issues.extend(_check_paulik_pending(ledger))
    issues.extend(_check_palu_not_frozen(ledger))
    issues.extend(_check_candidate_schemes(ledger))
    issues.extend(_check_steer_frozen(ledger))
    issues.extend(_check_unresolved_classes(ledger))
    issues.extend(_check_protocol(protocol_path or OFFICIAL_PROTOCOL_PATH))
    return ValidationReport(ok=not issues, issues=issues)


def _issue(code: str, message: str, path: str = "") -> ValidationIssue:
    return ValidationIssue(code=code, message=message, path=path)


def _check_meta(ledger: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    meta = ledger.get("meta")
    if not isinstance(meta, dict):
        return [_issue("meta.missing", "Ledger 'meta' must be a mapping.", "meta")]
    status = meta.get("status")
    if status != PENDING_DECISION:
        issues.append(
            _issue(
                "meta.status",
                f"Ledger meta.status must be {PENDING_DECISION!r} while Palu is unfrozen, got {status!r}.",
                "meta.status",
            )
        )
    if meta.get("purpose") != "decision_packet_for_jlin_not_a_frozen_mapping":
        issues.append(
            _issue(
                "meta.purpose",
                "Ledger purpose must be a Jlin decision packet, not a frozen mapping.",
                "meta.purpose",
            )
        )
    if meta.get("compute_u_or_model_conflicts") is not False:
        issues.append(
            _issue(
                "meta.no_u",
                "Ledger must set compute_u_or_model_conflicts to false.",
                "meta.compute_u_or_model_conflicts",
            )
        )
    if meta.get("cross_hazard_transfer_established") is not False:
        issues.append(
            _issue(
                "meta.no_transfer",
                "Ledger must not announce that cross-hazard transfer is established.",
                "meta.cross_hazard_transfer_established",
            )
        )
    for key in FORBIDDEN_TRANSFER_KEYS:
        value = meta.get(key)
        if value is True or value == "established":
            issues.append(
                _issue(
                    "meta.no_transfer",
                    f"Forbidden transfer claim: meta.{key}={value!r}.",
                    f"meta.{key}",
                )
            )
    return issues


def _check_paulik_pending(ledger: dict[str, Any]) -> list[ValidationIssue]:
    try:
        source = source_by_id(ledger, PAULIK_2019_ID)
    except CrosswalkError as exc:
        return [_issue("paulik.missing", str(exc), "sources")]
    issues: list[ValidationIssue] = []
    status = str(source.get("obtainment_status", "")).lower()
    if status not in ALLOWED_PAULIK_OBTAINMENT:
        issues.append(
            _issue(
                "paulik.pending",
                "Paulik 2019 obtainment_status must record that the official PDF and "
                "publisher table are on WSA and that Jlin adjudication is still required. "
                f"Got obtainment_status={source.get('obtainment_status')!r}.",
                f"sources.{PAULIK_2019_ID}.obtainment_status",
            )
        )
    if source.get("full_text_obtained") is not True:
        issues.append(
            _issue(
                "paulik.full_text",
                "Paulik 2019 full_text_obtained must be true once the official PDF is "
                "inventoried on WSA. File obtainment is not a Palu freeze.",
                f"sources.{PAULIK_2019_ID}.full_text_obtained",
            )
        )
    if source.get("blocks_palu_freeze") is not True:
        issues.append(
            _issue(
                "paulik.blocks_freeze",
                "Paulik 2019 must remain marked as blocking a Palu freeze until Jlin "
                "adjudicates official Y from the DL0-DL3 definitions.",
                f"sources.{PAULIK_2019_ID}.blocks_palu_freeze",
            )
        )
    return issues


def _check_palu_not_frozen(ledger: dict[str, Any]) -> list[ValidationIssue]:
    palu = ledger.get("palu_mapping")
    if not isinstance(palu, dict):
        return [_issue("palu.missing", "Ledger 'palu_mapping' must be a mapping.", "palu_mapping")]
    issues: list[ValidationIssue] = []
    if palu.get("status") != PENDING_DECISION:
        issues.append(
            _issue(
                "palu.status",
                "Palu mapping status must be pending_decision until Jlin freezes it. "
                f"Got {palu.get('status')!r}.",
                "palu_mapping.status",
            )
        )
    if palu.get("frozen") is not False:
        issues.append(
            _issue(
                "palu.frozen",
                "Palu mapping must not be marked frozen until Jlin adjudicates official Y.",
                "palu_mapping.frozen",
            )
        )
    if palu.get("write_to_official_protocol_as_frozen") is not False:
        issues.append(
            _issue(
                "palu.protocol",
                "Palu mapping must not be written into the official protocol as frozen.",
                "palu_mapping.write_to_official_protocol_as_frozen",
            )
        )
    if palu.get("is_main_conclusion") is True:
        issues.append(
            _issue(
                "palu.conclusion",
                "Palu candidate material must not be labeled a main conclusion.",
                "palu_mapping.is_main_conclusion",
            )
        )
    blocking = palu.get("blocking_items") or []
    if not any(isinstance(item, dict) and item.get("id") == "paulik_2019_full_text" for item in blocking):
        issues.append(
            _issue(
                "palu.blocking",
                "Palu mapping must list paulik_2019_full_text as a blocking item.",
                "palu_mapping.blocking_items",
            )
        )
    else:
        item = next(item for item in blocking if item.get("id") == "paulik_2019_full_text")
        if str(item.get("status", "")).lower() not in ALLOWED_PAULIK_OBTAINMENT:
            issues.append(
                _issue(
                    "palu.blocking_status",
                    "Blocking item paulik_2019_full_text must record that the official "
                    "PDF and publisher table are on WSA.",
                    "palu_mapping.blocking_items.paulik_2019_full_text.status",
                )
            )
    if not any(isinstance(item, dict) and item.get("id") == JLIN_ADJUDICATION_ID for item in blocking):
        issues.append(
            _issue(
                "palu.adjudication",
                "Palu mapping must list jlin_adjudication_of_dl_mapping as a blocking item.",
                "palu_mapping.blocking_items",
            )
        )
    else:
        item = next(item for item in blocking if item.get("id") == JLIN_ADJUDICATION_ID)
        if str(item.get("status", "")).lower() != PENDING:
            issues.append(
                _issue(
                    "palu.adjudication_status",
                    "Blocking item jlin_adjudication_of_dl_mapping must be PENDING. "
                    "File obtainment is not a Palu freeze.",
                    "palu_mapping.blocking_items.jlin_adjudication_of_dl_mapping.status",
                )
            )
    return issues


def _check_candidate_schemes(ledger: dict[str, Any]) -> list[ValidationIssue]:
    try:
        schemes = candidate_schemes(ledger)
    except CrosswalkError as exc:
        return [_issue("schemes.missing", str(exc), "palu_mapping.candidate_schemes")]
    issues: list[ValidationIssue] = []
    if not schemes:
        issues.append(
            _issue(
                "schemes.empty",
                "Ledger must list Palu candidate schemes, unlabeled as main conclusions.",
                "palu_mapping.candidate_schemes",
            )
        )
    required = {
        "strict_ds1_unknown_exclude_ds2_ds3",
        "exploratory_ds2_major_ds3_destroyed",
        "ds0_to_no_listed_separately",
    }
    found = {row.get("id") for row in schemes if isinstance(row, dict)}
    missing = sorted(required - found)
    if missing:
        issues.append(
            _issue(
                "schemes.required",
                "Missing required candidate scheme ids: " + ", ".join(missing),
                "palu_mapping.candidate_schemes",
            )
        )
    for row in schemes:
        if not isinstance(row, dict):
            issues.append(_issue("schemes.type", "Each candidate scheme must be a mapping."))
            continue
        scheme_id = str(row.get("id") or "unknown")
        if row.get("is_main_conclusion") is not False:
            issues.append(
                _issue(
                    "schemes.not_conclusion",
                    f"Candidate scheme {scheme_id} must set is_main_conclusion=false.",
                    f"palu_mapping.candidate_schemes.{scheme_id}.is_main_conclusion",
                )
            )
        if row.get("write_to_official_protocol_as_frozen") is not False:
            issues.append(
                _issue(
                    "schemes.not_protocol",
                    f"Candidate scheme {scheme_id} must not be written into the official protocol as frozen.",
                    f"palu_mapping.candidate_schemes.{scheme_id}.write_to_official_protocol_as_frozen",
                )
            )
        if row.get("status") not in {"candidate", "exploratory_sensitivity", "listed_separately", NOT_APPLIED}:
            issues.append(
                _issue(
                    "schemes.status",
                    f"Candidate scheme {scheme_id} status must be candidate, "
                    "exploratory_sensitivity, listed_separately, or not_applied, "
                    f"got {row.get('status')!r}.",
                    f"palu_mapping.candidate_schemes.{scheme_id}.status",
                )
            )
        if row.get("frozen") is True or row.get("approved") is True:
            issues.append(
                _issue(
                    "schemes.frozen",
                    f"Candidate scheme {scheme_id} must not be marked frozen or approved.",
                    f"palu_mapping.candidate_schemes.{scheme_id}",
                )
            )
        if row.get("applied_as_mapping") is True:
            issues.append(
                _issue(
                    "schemes.applied",
                    f"Candidate scheme {scheme_id} must not be applied as the Palu mapping. "
                    "Official Paulik grades are DL0-DL3, not DS0-DS3.",
                    f"palu_mapping.candidate_schemes.{scheme_id}.applied_as_mapping",
                )
            )
        elif row.get("id") in {
            "strict_ds1_unknown_exclude_ds2_ds3",
            "exploratory_ds2_major_ds3_destroyed",
            "ds0_to_no_listed_separately",
        } and row.get("applied_as_mapping") is not False:
            issues.append(
                _issue(
                    "schemes.applied",
                    f"Candidate scheme {scheme_id} must set applied_as_mapping=false.",
                    f"palu_mapping.candidate_schemes.{scheme_id}.applied_as_mapping",
                )
            )
    issues.extend(_check_official_grade_system(ledger))
    return issues


def _check_official_grade_system(ledger: dict[str, Any]) -> list[ValidationIssue]:
    grade = ledger.get("official_grade_system")
    if not isinstance(grade, dict):
        return [_issue("grades.missing", "Ledger must include official_grade_system from the Paulik PDF.", "official_grade_system")]
    issues: list[ValidationIssue] = []
    if grade.get("scale") != "DL0_to_DL3":
        issues.append(
            _issue(
                "grades.scale",
                "Official Palu grade system must be Paulik DL0-DL3, not the earlier DS schemes.",
                "official_grade_system.scale",
            )
        )
    if grade.get("frozen") is True or grade.get("applied_to_official_y") is True:
        issues.append(
            _issue(
                "grades.frozen",
                "The inventoried DL0-DL3 definitions must not be treated as a frozen official-Y map.",
                "official_grade_system",
            )
        )
    if grade.get("is_main_conclusion") is True or grade.get("write_to_official_protocol_as_frozen") is True:
        issues.append(
            _issue(
                "grades.protocol",
                "DL0-DL3 inventory must not be labeled a main conclusion or written into the official protocol as frozen.",
                "official_grade_system",
            )
        )
    definitions = grade.get("definitions") or {}
    expected = {
        "DL0": "No damage",
        "DL1": "Partial damage, repairable",
        "DL2": "Partial damage, unrepairable",
        "DL3": "Complete damage",
    }
    if definitions != expected:
        issues.append(
            _issue(
                "grades.definitions",
                "official_grade_system.definitions must quote Paulik Table 1 DL0-DL3 exactly.",
                "official_grade_system.definitions",
            )
        )
    return issues


def _check_steer_frozen(ledger: dict[str, Any]) -> list[ValidationIssue]:
    try:
        mapping = frozen_mapping_by_id(ledger, STEER_WIND_ID)
    except CrosswalkError as exc:
        return [_issue("steer.missing", str(exc), "frozen_mappings")]
    issues: list[ValidationIssue] = []
    if mapping.get("status") != APPROVED or mapping.get("frozen") is not True:
        issues.append(
            _issue(
                "steer.status",
                "StEER wind must remain approved and frozen. Wind events continue to use this package.",
                "frozen_mappings.steer_wind",
            )
        )
    rules = mapping.get("rules")
    if rules != STEER_WIND_FROZEN_RULES:
        issues.append(
            _issue(
                "steer.rules",
                "StEER wind frozen rules do not match the approved package. Do not edit them here.",
                "frozen_mappings.steer_wind.rules",
            )
        )
    if mapping.get("source") != STEER_WIND_FROZEN_SOURCE:
        issues.append(
            _issue(
                "steer.source",
                "StEER wind frozen source line does not match Report 06-20, printed p. 15.",
                "frozen_mappings.steer_wind.source",
            )
        )
    if mapping.get("code_F_from") != "component_fields_never_from_the_grade":
        issues.append(
            _issue(
                "steer.F",
                "StEER wind must code F from component fields, never from the grade.",
                "frozen_mappings.steer_wind.code_F_from",
            )
        )
    return issues


def _check_unresolved_classes(ledger: dict[str, Any]) -> list[ValidationIssue]:
    rows = ledger.get("unresolved_classes")
    if not isinstance(rows, list):
        return [_issue("classes.missing", "Ledger 'unresolved_classes' must be a list.", "unresolved_classes")]
    issues: list[ValidationIssue] = []
    found = {row.get("class") for row in rows if isinstance(row, dict)}
    for name in ("DL0", "DL1", "DL2", "DL3", "DS0", "DS1", "DS2", "DS3"):
        if name not in found:
            issues.append(
                _issue(
                    "classes.required",
                    f"Unresolved class {name} must be listed until Paulik 2019 is adjudicated.",
                    "unresolved_classes",
                )
            )
        else:
            row = next(row for row in rows if isinstance(row, dict) and row.get("class") == name)
            allowed = {"unresolved"} if str(name).startswith("DL") else {"unresolved", "not_the_official_paulik_grade_system"}
            if row.get("status") not in allowed:
                issues.append(
                    _issue(
                        "classes.status",
                        f"Class {name} must not be applied as official Y or as the official Palu grade system.",
                        f"unresolved_classes.{name}.status",
                    )
                )
    return issues


def _check_protocol(protocol_path: str | Path) -> list[ValidationIssue]:
    path = Path(protocol_path)
    if not path.is_file():
        return [_issue("protocol.missing", f"Official protocol not found: {path}", str(path))]
    text = path.read_text(encoding="utf-8")
    issues: list[ValidationIssue] = []
    for sentence in PALU_PROTOCOL_SENTENCES:
        if sentence not in text:
            issues.append(
                _issue(
                    "protocol.palu",
                    "Official protocol no longer contains the Palu non-mapping sentence: "
                    f"{sentence}",
                    str(path),
                )
            )
    for sentence in STEER_PROTOCOL_SENTENCES:
        if sentence not in text:
            issues.append(
                _issue(
                    "protocol.steer",
                    "Official protocol no longer contains a frozen StEER wind sentence: "
                    f"{sentence}",
                    str(path),
                )
            )
    # Candidate exploratory maps must not have been copied in as a freeze.
    if "Palu mapping is frozen" in text or "Palu is frozen" in text:
        issues.append(
            _issue(
                "protocol.not_frozen",
                "Official protocol appears to freeze Palu, which this worktree must not do.",
                str(path),
            )
        )
    return issues

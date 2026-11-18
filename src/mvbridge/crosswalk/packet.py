"""Render the Palu mapping decision packet for Jlin. This is not a freeze."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from mvbridge.crosswalk.ledger import PAULIK_2019_ID, source_by_id
from mvbridge.crosswalk.validate import ValidationReport, validate_ledger
from mvbridge.errors import CrosswalkError

PACKET_TITLE = "Palu mapping decision packet (not a frozen mapping)"

FORBIDDEN_PACKET_PHRASES = (
    "palu mapping is frozen",
    "palu is frozen",
    "cross-hazard transfer is established",
    "cross-disaster transfer is established",
    "cross-hazard transfer has been established",
    "cross-disaster transfer has been established",
)


def _yaml_dump(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip()


def _ledger_digest(ledger: dict[str, Any]) -> str:
    payload = {key: value for key, value in ledger.items() if not str(key).startswith("_")}
    canonical = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()



def _inventory_differences_section(ledger: dict[str, Any]) -> list[str]:
    """Render inventory findings that are not frozen protocol facts."""
    diff = ledger.get("inventory_differences") or {}
    coords = diff.get("coordinate_uniqueness") or {}
    table2 = diff.get("pdf_table2_vs_buildings_commercial_unknown") or {}
    event = diff.get("catalog_window_vs_paulik_event_date") or {}
    survey = diff.get("palu_city_survey_period") or {}
    lines = [
        "## 10. Inventory differences (pre-freeze findings, not frozen protocol facts)",
        "",
        "These items are inventory findings and pre-freeze issues. They are not",
        "frozen protocol facts, not a Palu mapping, and not a silent edit of",
        "`docs/eval_protocol.md` or `configs/event_catalog.yaml`. Palu mapping",
        "stays `pending_decision` / unfrozen / `obtained_on_wsa_awaiting_adjudication`.",
        "Purpose remains a decision packet, not a frozen mapping.",
        "",
        "### PDF Table 2 vs Buildings sheet Commercial/Unknown",
        "",
        str(table2.get("statement") or "").strip(),
        "",
        "### Catalog window vs Paulik event date (must be adjudicated before any freeze)",
        "",
        str(event.get("statement") or "").strip(),
        "",
        "### Palu City survey period is not the event date",
        "",
        str(survey.get("statement") or "").strip(),
        "",
        "### Coordinate uniqueness",
        "",
        f"- unique IDs: {coords.get('unique_ids')}",
        f"- unique latitudes: {coords.get('unique_latitudes')}",
        f"- unique longitudes: {coords.get('unique_longitudes')}",
        f"- unique (lat, lon) pairs: {coords.get('unique_lat_lon_pairs')}",
        f"- shared exact (lat, lon) pairs: {coords.get('shared_exact_lat_lon_pairs')}",
        f"- computation: {coords.get('computation')}",
        "",
        str(coords.get("statement") or "").strip(),
        "",
    ]
    return lines


def render_decision_packet(
    ledger: dict[str, Any],
    *,
    report: ValidationReport | None = None,
    ground_checks: dict[str, Any] | None = None,
) -> str:
    """Render a Markdown decision packet from a validated ledger."""
    report = report or validate_ledger(ledger)
    if not report.ok:
        raise CrosswalkError("Cannot render a decision packet from an invalid ledger.")
    paulik = source_by_id(ledger, PAULIK_2019_ID)
    meta = ledger["meta"]
    palu = ledger["palu_mapping"]
    alignment = ledger["date_phase_alignment"]
    storage = ledger["authorized_storage"]
    lines: list[str] = [
        f"# {PACKET_TITLE}",
        "",
        "Status: **PENDING Jlin decision**. This document is a decision packet.",
        "It is not a frozen Palu mapping and must not be copied into",
        "`docs/eval_protocol.md` as a freeze.",
        "",
        f"- Ledger sha256 (canonical YAML, sorted keys): `{_ledger_digest(ledger)}`",
        f"- Ledger purpose: `{meta.get('purpose')}`",
        f"- Palu mapping status: `{palu.get('status')}` (frozen={palu.get('frozen')})",
        f"- Cross-hazard transfer established: `{meta.get('cross_hazard_transfer_established')}`",
        f"- Compute U or model-based conflicts: `{meta.get('compute_u_or_model_conflicts')}`",
        "",
        "## 1. Paulik 2019 files are on WSA; Palu mapping remains PENDING Jlin decision",
        "",
        f"- Source id: `{paulik['id']}`",
        f"- Citation: {paulik['citation']}",
        f"- Obtainment status: **{str(paulik['obtainment_status']).upper()}**",
        f"- Full text obtained: `{paulik['full_text_obtained']}`",
        f"- Blocks Palu freeze: `{paulik['blocks_palu_freeze']}`",
        f"- Notes: {paulik.get('notes', '')}",
        "",
        "The official 17-page PDF and the publisher Buildings sheet are inventoried",
        "on WSA. File obtainment is not a Palu freeze. Official protocol text is not",
        "edited here. Jlin still has to adjudicate official Y from Paulik's own",
        "DL0-DL3 definitions. Earlier DS1/DS2/DS3 candidate schemes are not the",
        "mapping and must not be applied.",
        "",
        "## 2. What stays frozen (do not reopen)",
        "",
        "Wind events continue to use the already-frozen StEER wind package in",
        "`docs/eval_protocol.md`. This packet does not change StEER wind, CAL FIRE",
        "DINS Affected → no, or the frozen FEMA mapping.",
        "",
    ]
    for row in ledger["frozen_mappings"]:
        lines.append(
            f"- `{row['id']}` ({row.get('hazard', 'n/a')}): status=`{row['status']}`, "
            f"frozen=`{row['frozen']}`."
        )
        if row.get("id") == "steer_wind":
            lines.append(f"  Source: {row.get('source')}")
            lines.append("  Code F from component fields, never from the grade.")
    lines.extend(
        [
            "",
            "## 3. Official protocol position on Palu (unchanged)",
            "",
            "The official protocol already records a Palu non-mapping. This worktree",
            "does not edit that section into a freeze. Quoted from",
            "`docs/eval_protocol.md`:",
            "",
        ]
    )
    for sentence in palu.get("official_protocol_statements") or []:
        lines.append(f"> {sentence}")
        lines.append(">")
    if lines[-1] == ">":
        lines.pop()
    lines.extend(
        [
            "",
            "## 4. Literature sources, versions, and obtainment",
            "",
            "| Id | Role | Version or locus | Obtainment | May define official Y |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in ledger["sources"]:
        lines.append(
            "| {id} | {role} | {version} | {status} | {may} |".format(
                id=row.get("id"),
                role=row.get("role"),
                version=row.get("version") or row.get("locus") or "",
                status=row.get("obtainment_status"),
                may=row.get("may_define_official_y"),
            )
        )
    lines.extend(["", "Citations:", ""])
    for row in ledger["sources"]:
        lines.append(f"- `{row['id']}`: {row['citation']}")
        if row.get("notes"):
            lines.append(f"  {row['notes']}")
    lines.extend(
        [
            "",
            "## 5. Date and phase alignment",
            "",
            f"- Event id: `{alignment.get('event_id')}`",
            f"- Catalog window: {alignment.get('catalog_start')} to {alignment.get('catalog_end')}",
            f"- Catalog source: {alignment.get('catalog_source')}",
            f"- Commonly cited event date: {alignment.get('commonly_cited_event_date')}",
            f"- Cited-date status: `{alignment.get('commonly_cited_event_date_status')}`",
            f"- Paulik event date (from official PDF): {alignment.get('paulik_event_date')}",
            f"- Paulik event-date source: {alignment.get('paulik_event_date_source')}",
            f"- Survey dates: {alignment.get('survey_dates')}",
            f"- Survey-date source: {alignment.get('survey_dates_source')}",
            f"- Imagery vs survey phase: {alignment.get('imagery_vs_survey_phase')}",
            "",
            "Open questions:",
            "",
        ]
    )
    for question in alignment.get("open_questions") or []:
        lines.append(f"- {question}")
    footprint = ledger["footprint_checks"]
    lines.extend(
        [
            "",
            "## 6. Footprint checks",
            "",
            "The general footprint-to-reference rules are already frozen in the",
            "protocol. Palu uses the same numeric gates when a ground table is available.",
            "This packet does not invent a Palu-specific IoU threshold.",
            "",
            f"- Match if IoU ≥ {footprint.get('iou_match_min')}",
            f"- Ambiguous if two or more candidates have IoU ≥ {footprint.get('ambiguity_iou')}, "
            f"or the best/second-best score ratio is < {footprint.get('best_second_ratio_min')}",
            f"- Drop a ground point when localization error > {footprint.get('localization_error_m_max')} m",
            f"- Palu footprint layer path: `{footprint.get('palu_footprint_layer_path')}`",
            f"- Palu ground table path: `{footprint.get('palu_ground_table_path')}`",
            f"- Notes: {footprint.get('notes')}",
            "",
            "## 7. Unresolved classes",
            "",
            "Paulik's official classes are DL0-DL3. They stay unresolved as official Y",
            "until Jlin adjudicates. DS0-DS3 are not the official grade system and",
            "must not be applied as the mapping.",
            "",
            "| Class | Status | Official definition, guidance, or notes |",
            "| --- | --- | --- |",
        ]
    )
    for row in ledger["unresolved_classes"]:
        guidance = row.get("official_definition") or row.get("official_guidance") or row.get("candidate_treatment") or ""
        lines.append(f"| {row.get('class')} | {row.get('status')} | {guidance} |")
    lines.extend(
        [
            "",
            "## 8. Earlier DS candidate schemes (not the mapping; do not apply)",
            "",
            "The official Palu grade system, checked against the official PDF Table 1,",
            "is DL0-DL3 (No damage; Partial damage, repairable; Partial damage,",
            "unrepairable; Complete damage). The earlier DS1/DS2/DS3 schemes below",
            "are retained only as historical review items. They are not the mapping,",
            "are not main conclusions, must not be applied, and must not be written",
            "into the official protocol as frozen.",
            "",
            f"- Official grade scale: `{ledger.get('official_grade_system', {}).get('scale')}`",
            f"- Applied to official Y: `{ledger.get('official_grade_system', {}).get('applied_to_official_y')}`",
            f"- Frozen: `{ledger.get('official_grade_system', {}).get('frozen')}`",
            f"- Definitions: `{ledger.get('official_grade_system', {}).get('definitions')}`",
            f"- PDF source: {ledger.get('official_grade_system', {}).get('table')}",
            "",
        ]
    )
    for row in palu.get("candidate_schemes") or []:
        lines.append(f"### `{row.get('id')}`")
        lines.append("")
        lines.append(f"- Status: `{row.get('status')}`")
        lines.append(f"- applied_as_mapping: `{row.get('applied_as_mapping')}`")
        lines.append(f"- is_main_conclusion: `{row.get('is_main_conclusion')}`")
        lines.append(
            f"- write_to_official_protocol_as_frozen: `{row.get('write_to_official_protocol_as_frozen')}`"
        )
        if row.get("notes"):
            lines.append(f"- Notes: {row['notes']}")
        lines.append("- Rules:")
        lines.append("")
        lines.append("```yaml")
        lines.append(_yaml_dump(row.get("rules")))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## 9. Authorized storage and WSA inventory",
            "",
            "Real Palu microdata, if located, stays in authorized storage and is not",
            "committed to this repository. Paths and hashes below are inventory only.",
            "Origin is reported only when it can be determined from the file or a",
            "sidecar manifest; this packet does not guess.",
            "",
        ]
    )
    for key, value in storage.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            lines.append(f"### `{key}`")
            lines.append("")
            for sub_key, sub_value in value.items():
                lines.append(f"- {sub_key}: {sub_value}")
            lines.append("")
        else:
            lines.append(f"- `{key}`: {value}")
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(_inventory_differences_section(ledger))
    next_section = 11
    if ground_checks is not None:
        lines.extend(
            [
                f"## {next_section}. Ground-side checks on a synthetic fixture",
                "",
                "These numbers come from a synthetic fixture generator, not from Palu",
                "field records. They demonstrate the check machinery (date/phase flags,",
                "footprint match, multipoint inconsistency, and per-scheme inventories).",
                "They are not results.",
                "",
                f"- computes_u: `{ground_checks.get('computes_u')}`",
                f"- computes_prediction_conflicts: `{ground_checks.get('computes_prediction_conflicts')}`",
                f"- n_records: {ground_checks.get('n_records')}",
                f"- footprint_match_counts: `{ground_checks.get('footprint_match_counts')}`",
                f"- multipoint_inconsistencies: {len(ground_checks.get('multipoint_inconsistencies') or [])}",
                f"- date_phase_flag_counts: `{ground_checks.get('date_phase_flag_counts')}`",
                "",
            ]
        )
        next_section += 1
    claim_section = next_section
    decisions_section = claim_section + 1
    validator_section = decisions_section + 1
    lines.extend(
        [
            f"## {claim_section}. What this packet does not claim",
            "",
            "- It does not freeze a Palu mapping. File obtainment is not a freeze.",
            "- It does not compute U or prediction-based conflict counts.",
            "- It does not treat transfer across hazards as shown or settled.",
            "- It does not apply the earlier DS1/DS2/DS3 candidate schemes as the mapping.",
            "- Official xBD palu-tsunami labels, if present on disk, are never Y.",
            "- Secondary sources (Lahcene 2021, Jalayer 2023) do not define official Y.",
            "- Jalayer's phrase \"complete structural collapse\" is Jalayer's wording,",
            "  not a verified Paulik definition.",
            "- MSBFP2 inventory is not a Palu mapping source and is not a protocol freeze.",
            "",
            f"## {decisions_section}. Decisions requested from Jlin",
            "",
        ]
    )
    for item in palu.get("decisions_requested") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"## {validator_section}. Validator",
            "",
            f"- Validator result: `{'PASS' if report.ok else 'FAIL'}`",
            f"- Issue count: {len(report.issues)}",
            "",
            "The validator refuses to mark Palu approved or frozen until Jlin",
            "adjudicates official Y, refuses to unfreeze StEER wind, refuses applying",
            "the earlier DS schemes as the mapping, refuses candidate schemes labeled",
            "as main conclusions or as official protocol freezes, and refuses ground",
            "tables that include model predictions or U/conflict fields.",
            "",
        ]
    )
    text = "\n".join(lines).rstrip() + "\n"
    lowered = text.lower()
    for phrase in FORBIDDEN_PACKET_PHRASES:
        if phrase in lowered:
            raise CrosswalkError(
                f"Decision packet contains a forbidden claim: {phrase!r}."
            )
    return text


def write_decision_packet(
    ledger: dict[str, Any],
    path: str | Path,
    *,
    report: ValidationReport | None = None,
    ground_checks: dict[str, Any] | None = None,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        render_decision_packet(ledger, report=report, ground_checks=ground_checks),
        encoding="utf-8",
    )
    return dest

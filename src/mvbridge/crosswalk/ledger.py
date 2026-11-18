"""Load and inspect the Palu / StEER mapping ledger."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mvbridge.errors import CrosswalkError

DEFAULT_LEDGER_PATH = Path("configs/palu_crosswalk/ledger.yaml")
REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGED_LEDGER_PATH = REPO_ROOT / "configs" / "palu_crosswalk" / "ledger.yaml"
OFFICIAL_PROTOCOL_PATH = REPO_ROOT / "docs" / "eval_protocol.md"

REQUIRED_TOP_KEYS = (
    "meta",
    "sources",
    "frozen_mappings",
    "palu_mapping",
    "date_phase_alignment",
    "footprint_checks",
    "official_grade_system",
    "unresolved_classes",
    "authorized_storage",
)

PAULIK_2019_ID = "paulik_2019"
STEER_WIND_ID = "steer_wind"

# Structured copy of the frozen StEER wind package in docs/eval_protocol.md.
# The validator rejects any ledger that changes these rules.
STEER_WIND_FROZEN_RULES = {
    "0": "no",
    "1": "minor_does_not_satisfy_F",
    "2": "minor_never_major",
    "3": "major_only_with_recorded_partial_roof_or_wall_structural_failure_otherwise_minor_or_unknown",
    "4": (
        "default_major_in_Q; promote_to_destroyed_only_with_explicit_collapse_"
        "or_no_longer_present_or_living_area_wall_structure_failure_and_no_intact_roof; "
        "table4_shading_is_not_a_promotion_rule; missing_fields_stay_major"
    ),
    "-1": "excluded_from_wind_Y_not_no_damage",
}

STEER_WIND_FROZEN_SOURCE = (
    "Prevatt and Roueche, 15 June 2020, Hurricane Michael Data Enhancement "
    "(Phase II), Report No. 06-20, University of Florida, for Florida DBPR, "
    "Project P0157245, Table 4 and Table 5, printed p. 15."
)

PALU_PROTOCOL_SENTENCES = (
    "Palu is not one-to-one and is not frozen until Paulik et al. (2019) is in hand.",
    "DS1 → unknown or excluded from a strict crosswalk.",
    "DS2/DS3 only as exploratory sensitivity.",
    "Full text not obtained.",
)

STEER_PROTOCOL_SENTENCES = (
    "StEER wind is frozen as follows.",
    "4 is not automatically destroyed.",
    "Table 4 shaded indicators for grade 4",
)


def load_ledger(path: str | Path | None = None) -> dict[str, Any]:
    """Load a mapping ledger YAML file and check required sections."""
    ledger_path = Path(path) if path is not None else PACKAGED_LEDGER_PATH
    if not ledger_path.is_file():
        raise CrosswalkError(f"Mapping ledger not found: {ledger_path}")
    payload = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CrosswalkError(f"Mapping ledger must be a mapping: {ledger_path}")
    missing = [key for key in REQUIRED_TOP_KEYS if key not in payload]
    if missing:
        raise CrosswalkError(f"Mapping ledger is missing sections: {', '.join(missing)}")
    payload["_path"] = str(ledger_path)
    return payload


def clone_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a ledger, dropping loader bookkeeping."""
    copied = deepcopy(ledger)
    copied.pop("_path", None)
    return copied


def source_by_id(ledger: dict[str, Any], source_id: str) -> dict[str, Any]:
    sources = ledger.get("sources")
    if not isinstance(sources, list):
        raise CrosswalkError("Ledger 'sources' must be a list.")
    for row in sources:
        if isinstance(row, dict) and row.get("id") == source_id:
            return row
    raise CrosswalkError(f"Ledger has no source id {source_id!r}.")


def frozen_mapping_by_id(ledger: dict[str, Any], mapping_id: str) -> dict[str, Any]:
    mappings = ledger.get("frozen_mappings")
    if not isinstance(mappings, list):
        raise CrosswalkError("Ledger 'frozen_mappings' must be a list.")
    for row in mappings:
        if isinstance(row, dict) and row.get("id") == mapping_id:
            return row
    raise CrosswalkError(f"Ledger has no frozen mapping id {mapping_id!r}.")


def candidate_schemes(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    palu = ledger.get("palu_mapping")
    if not isinstance(palu, dict):
        raise CrosswalkError("Ledger 'palu_mapping' must be a mapping.")
    schemes = palu.get("candidate_schemes") or []
    if not isinstance(schemes, list):
        raise CrosswalkError("Ledger 'palu_mapping.candidate_schemes' must be a list.")
    return schemes

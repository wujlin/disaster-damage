"""Event identity for the overlap audit.

An event id is date-specific. "Midwest flooding 2015" and xBD's 2019
Midwest floods do not share an id. Matching is an explicit alias table
plus token rules, not a fuzzy name search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from mvbridge.errors import ConfigError
from mvbridge.geo import parse_date_span

STEMS = {
    "floods": "flood",
    "flooding": "flood",
    "fires": "fire",
    "wildfire": "fire",
    "wildfires": "fire",
    "tornadoes": "tornado",
    "hurricanes": "hurricane",
}


@dataclass
class CatalogEvent:
    event_id: str
    start: date | None
    end: date | None
    exact: set[str] = field(default_factory=set)
    match_all: list[str] = field(default_factory=list)
    match_none: list[str] = field(default_factory=list)
    filename_prefixes: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def prefix_norms(self) -> set[str]:
        return {normalize_name(prefix) for prefix in self.filename_prefixes}


@dataclass
class EventCatalog:
    events: list[CatalogEvent]
    by_id: dict[str, CatalogEvent]
    ambiguous: set[tuple[str, ...]]
    review_only: list[dict]
    path: str


@dataclass
class EventMatch:
    event_id: str | None
    status: str  # matched, missing, ambiguous, unmapped


def normalize_name(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def tokens(value: str) -> list[str]:
    stemmed = []
    for token in normalize_name(value).split():
        if token.isdigit() and len(token) == 4:
            stemmed.append(token)
        else:
            stemmed.append(STEMS.get(token, token))
    return stemmed


def load_catalog(path: str | Path) -> EventCatalog:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Event catalog does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    events = []
    seen = set()
    for item in raw.get("events", []):
        event_id = str(item["id"])
        if event_id in seen:
            raise ConfigError(f"Duplicate event id in catalog: {event_id}")
        seen.add(event_id)
        span = None
        if item.get("start") and item.get("end"):
            span = parse_date_span(f"{item['start']}..{item['end']}")
        exact = {normalize_name(name) for name in item.get("exact", [])}
        prefixes = [str(prefix) for prefix in item.get("filename_prefixes", [])]
        events.append(
            CatalogEvent(
                event_id=event_id,
                start=span[0] if span else None,
                end=span[1] if span else None,
                exact=exact,
                match_all=[STEMS.get(token, token) for token in item.get("match_all", [])],
                match_none=[STEMS.get(token, token) for token in item.get("match_none", [])],
                filename_prefixes=prefixes,
                source=str(item.get("source", "")),
            )
        )
    ambiguous = set()
    for name in raw.get("ambiguous_names", []):
        ambiguous.add(tuple(tokens(str(name))))
    return EventCatalog(
        events=events,
        by_id={event.event_id: event for event in events},
        ambiguous=ambiguous,
        review_only=list(raw.get("review_only", [])),
        path=str(file_path.resolve()),
    )


def match_event(name: str | None, catalog: EventCatalog) -> EventMatch:
    if name is None or str(name).strip() == "" or str(name).lower() in {"nan", "none"}:
        return EventMatch(None, "missing")
    text = str(name).strip()
    norm = normalize_name(text)
    looks_like_slug = "-" in text or "_" in text
    hits: list[str] = []
    for event in catalog.events:
        prefix_hit = text.lower() in {prefix.lower() for prefix in event.filename_prefixes}
        if not prefix_hit and looks_like_slug and norm in event.prefix_norms:
            prefix_hit = True
        if norm in event.exact or prefix_hit:
            hits.append(event.event_id)
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return EventMatch(unique[0], "matched")
    if len(unique) > 1:
        return EventMatch(None, "ambiguous")
    token_key = tuple(tokens(text))
    if token_key in catalog.ambiguous:
        return EventMatch(None, "ambiguous")
    for event in catalog.events:
        if event.match_all and all(token in token_key for token in event.match_all):
            if any(token in token_key for token in event.match_none):
                continue
            hits.append(event.event_id)
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return EventMatch(unique[0], "matched")
    if len(unique) > 1:
        return EventMatch(None, "ambiguous")
    return EventMatch(None, "unmapped")

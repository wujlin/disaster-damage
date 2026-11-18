"""Public-layer obtainability ledger for Group-3 geographic features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from mvbridge.errors import ConfigError

LEDGER_SCHEMA_VERSION = 1
PLACEHOLDER_STATUS = "OFFICIAL_EXPORT_PLACEHOLDER"


def _parse_optional_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    return date.fromisoformat(text[:10])


@dataclass(frozen=True)
class LayerRecord:
    id: str
    theme: str
    intended_events: tuple[str, ...]
    title: str
    access_url: str | None
    citation: str | None
    license: str | None
    product_available_from: date | None
    product_available_until: date | None
    acquired_from: date | None
    acquired_until: date | None
    checked_at: datetime | None
    check_method: str | None
    http_status: int | None
    obtainable: bool | None
    obtained_into_repo: bool
    status: str | None
    notes: str

    @property
    def is_official_export_placeholder(self) -> bool:
        return self.status == PLACEHOLDER_STATUS

    def usable_as_of(self, as_of: date | None) -> bool:
        if self.is_official_export_placeholder:
            return False
        if self.obtainable is False:
            return False
        if as_of is not None and self.acquired_from is not None and self.acquired_from > as_of:
            return False
        return self.obtainable is True


def _as_event_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _parse_checked_at(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _layer_from_mapping(raw: dict[str, Any]) -> LayerRecord:
    missing = [key for key in ("id", "theme") if not raw.get(key)]
    if missing:
        raise ConfigError(f"layer entry missing required fields: {missing}")
    obtained = bool(raw.get("obtained_into_repo", False))
    if obtained:
        raise ConfigError(
            f"layer {raw['id']!r} sets obtained_into_repo true, but real rasters "
            "and official alert tables must stay out of the repository"
        )
    status = raw.get("status")
    return LayerRecord(
        id=str(raw["id"]),
        theme=str(raw["theme"]),
        intended_events=_as_event_tuple(raw.get("intended_events")),
        title=str(raw.get("title") or raw["id"]),
        access_url=raw.get("access_url"),
        citation=raw.get("citation"),
        license=raw.get("license"),
        product_available_from=_parse_optional_date(raw.get("product_available_from")),
        product_available_until=_parse_optional_date(raw.get("product_available_until")),
        acquired_from=_parse_optional_date(raw.get("acquired_from")),
        acquired_until=_parse_optional_date(raw.get("acquired_until")),
        checked_at=_parse_checked_at(raw.get("checked_at")),
        check_method=raw.get("check_method"),
        http_status=int(raw["http_status"]) if raw.get("http_status") is not None else None,
        obtainable=raw.get("obtainable"),
        obtained_into_repo=False,
        status=str(status) if status is not None else None,
        notes=str(raw.get("notes") or "").strip(),
    )


@dataclass
class LayerLedger:
    schema_version: int
    checked_timezone: str
    layers: tuple[LayerRecord, ...]
    notes: str = ""

    @classmethod
    def from_path(cls, path: str | Path) -> "LayerLedger":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "LayerLedger":
        if not isinstance(payload, dict):
            raise ConfigError("layer ledger must be a mapping")
        version = int(payload.get("schema_version", LEDGER_SCHEMA_VERSION))
        if version != LEDGER_SCHEMA_VERSION:
            raise ConfigError(f"unsupported layer ledger schema_version {version}")
        raw_layers = payload.get("layers") or []
        if not isinstance(raw_layers, list) or not raw_layers:
            raise ConfigError("layer ledger needs a non-empty layers list")
        layers = tuple(_layer_from_mapping(item) for item in raw_layers)
        ids = [layer.id for layer in layers]
        if len(ids) != len(set(ids)):
            raise ConfigError("layer ledger ids must be unique")
        return cls(
            schema_version=version,
            checked_timezone=str(payload.get("checked_timezone") or ""),
            layers=layers,
            notes=str(payload.get("notes") or "").strip(),
        )

    def get(self, layer_id: str) -> LayerRecord:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(layer_id)

    def by_theme(self, theme: str) -> tuple[LayerRecord, ...]:
        return tuple(layer for layer in self.layers if layer.theme == theme)

    def placeholders(self) -> tuple[LayerRecord, ...]:
        return tuple(layer for layer in self.layers if layer.is_official_export_placeholder)

    def usable_layers(self, theme: str, *, event_id: str | None = None, as_of: date | None = None) -> tuple[LayerRecord, ...]:
        found = []
        for layer in self.by_theme(theme):
            if event_id is not None and layer.intended_events and event_id not in layer.intended_events:
                continue
            if layer.usable_as_of(as_of):
                found.append(layer)
        return tuple(found)

    def assert_no_official_alerts(self, event_ids: Iterable[str] | None = None) -> None:
        wanted = None if event_ids is None else set(event_ids)
        pending = []
        for layer in self.placeholders():
            if wanted is None or not layer.intended_events or wanted.intersection(layer.intended_events):
                pending.append(layer.id)
        if pending:
            raise ConfigError(
                "satellite-side alert columns are still official-export placeholders: "
                + ", ".join(pending)
            )

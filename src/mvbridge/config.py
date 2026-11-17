"""YAML config loading with dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

from mvbridge.errors import ConfigError


def load_yaml(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Config file does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {file_path}")
    return data


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        _assign(config, key.strip(), value)
    return config


def _assign(config: dict, dotted: str, value) -> None:
    cursor = config
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        raise ConfigError(f"Empty override key in {dotted!r}")
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ConfigError(f"Cannot override {dotted!r}; {part!r} is not a mapping")
        cursor = current
    cursor[parts[-1]] = value


def require_mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config is missing a mapping at {key!r}")
    return value

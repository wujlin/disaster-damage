"""Errors raised before any training work starts."""


class ConfigError(ValueError):
    """A config value is missing or inconsistent."""


class DataError(FileNotFoundError):
    """Configured metadata or imagery cannot be read."""


class CrosswalkError(ValueError):
    """A mapping ledger, protocol check, or ground-side table is invalid."""

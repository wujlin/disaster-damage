"""Read the frozen evaluation protocol and the authorized artifact inventory.

The source of truth for metric names, P4/P5 split, and eligibility cutoffs is
``docs/eval_protocol.md`` together with the shipped fold list. This module does
not invent checkpoint hashes. Null or TBD digests stay null.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mvbridge.data.folds import DEFAULT_FOLDS_PATH, FoldSpec, load_folds, load_weight_provenance
from mvbridge.errors import ConfigError

OFFICIAL_CLASS_ORDER = ("no-damage", "minor-damage", "major-damage", "destroyed")
# Protocol shorthand used in U and F definitions.
CLASS_ALIASES = {
    "no": "no-damage",
    "no-damage": "no-damage",
    "nodamage": "no-damage",
    "unaffected": "no-damage",
    "minor": "minor-damage",
    "minor-damage": "minor-damage",
    "major": "major-damage",
    "major-damage": "major-damage",
    "destroyed": "destroyed",
}

P4_BASELINES = ("entropy", "one_minus_max_p")
P5_BASELINES = ("g0",)

SATELLITE_CLEAR_PIXEL_MIN = 0.80
MIN_CONFLICTS_TOTAL = 300
MIN_QUALIFIED_EVENTS = 3
MIN_CONFLICTS_PER_EVENT = 30
U_DEFINING_SEED = 1
SEED_REPEAT_ROLE = "seed-repeat"

OFFICIAL_RECIPE = {
    "backbone": "DINOv2 ViT-B/14",
    "leave_one_event": True,
    "image_size": 518,
    "pos_embed_interpolation": False,
    "freeze_backbone": True,
    "damage_head_only": True,
    "batch_size": 32,
    "amp_dtype": "bfloat16",
    "num_workers": 16,
}

# Phrases that must remain in the committed protocol. Checks are typographic
# flexible so math markup does not have to be copied character-for-character.
_REQUIRED_PROTOCOL_PATTERNS = (
    (r"##\s+3\.\s+Baselines", "Baselines section"),
    (r"satellite damage head entropy", "entropy baseline"),
    (r"1\s*-\s*\\?\\max\s*p|1\s*-\s*max\s*prob", "1-max p baseline"),
    (r"##\s+4\.\s+Metrics", "Metrics section"),
    (r"\*\*P4\.\*\*", "P4 prediction"),
    (r"\*\*P5\.\*\*", "P5 prediction"),
    (r"is not part of the P4 pass criterion", "g0 excluded from P4"),
    (r"The method must beat", "g0 belongs to P5"),
    (r"clear, valid pixels cover at least 80%", "80% clear-pixel rule"),
    (r"at least 300 conflict footprints", "fallback N=300"),
    (r"at least 3 events each with", "fallback 3 events"),
    (r"Frozen seed 1|seed-1 fold head|seed 1", "seed-1 reporting head"),
)


def default_repo_root() -> Path:
    """Return the worktree root that contains ``docs/eval_protocol.md``."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ArtifactRecord:
    """One authorized fold artifact as recorded in the repo, not on disk."""

    fold_name: str
    role: str
    holdout_events: tuple[str, ...]
    seed: int
    enabled: bool
    claim_evidence: bool
    host: str | None
    output_dir: str | None
    manifest_sha256: str | None
    checkpoint_sha256: str | None
    weights_sha256: str | None


@dataclass(frozen=True)
class ArtifactInventory:
    source_path: str
    weight_provenance: dict[str, Any]
    records: tuple[ArtifactRecord, ...]

    def record(self, fold_name: str) -> ArtifactRecord:
        match = next((item for item in self.records if item.fold_name == fold_name), None)
        if match is None:
            known = ", ".join(item.fold_name for item in self.records)
            raise ConfigError(f"No authorized artifact record for {fold_name!r}. Known: {known}")
        return match


@dataclass(frozen=True)
class ProtocolContract:
    protocol_path: str
    folds_path: str
    recipe_path: str
    official_class_order: tuple[str, ...]
    p4_baselines: tuple[str, ...]
    p5_baselines: tuple[str, ...]
    satellite_clear_pixel_min: float
    min_conflicts_total: int
    min_qualified_events: int
    min_conflicts_per_event: int
    u_defining_seed: int
    official_recipe: dict[str, Any]
    recipe_from_config: dict[str, Any]
    artifacts: ArtifactInventory


def load_protocol_contract(
    repo_root: str | Path | None = None,
    *,
    protocol_path: str | Path | None = None,
    folds_path: str | Path | None = None,
    recipe_path: str | Path | None = None,
) -> ProtocolContract:
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    protocol = Path(protocol_path) if protocol_path else root / "docs" / "eval_protocol.md"
    folds = Path(folds_path) if folds_path else root / DEFAULT_FOLDS_PATH
    recipe = Path(recipe_path) if recipe_path else root / "configs" / "xbd_dinov2_vitb.yaml"
    if not protocol.is_file():
        raise ConfigError(f"Evaluation protocol is missing: {protocol}")
    text = protocol.read_text(encoding="utf-8")
    _assert_protocol_text(text)
    recipe_values = _read_official_recipe_config(recipe)
    artifacts = authorized_artifact_inventory(folds)
    return ProtocolContract(
        protocol_path=str(protocol),
        folds_path=str(folds),
        recipe_path=str(recipe),
        official_class_order=OFFICIAL_CLASS_ORDER,
        p4_baselines=P4_BASELINES,
        p5_baselines=P5_BASELINES,
        satellite_clear_pixel_min=SATELLITE_CLEAR_PIXEL_MIN,
        min_conflicts_total=MIN_CONFLICTS_TOTAL,
        min_qualified_events=MIN_QUALIFIED_EVENTS,
        min_conflicts_per_event=MIN_CONFLICTS_PER_EVENT,
        u_defining_seed=U_DEFINING_SEED,
        official_recipe=dict(OFFICIAL_RECIPE),
        recipe_from_config=recipe_values,
        artifacts=artifacts,
    )


def authorized_artifact_inventory(path: str | Path | None = None) -> ArtifactInventory:
    """Read the authorized fold/artifact list from the shipped YAML.

    Manifest and checkpoint digests are taken from the file as written. This
    function does not hash checkpoints, scan official output directories, or
    fill TBD entries.
    """
    file_path = Path(path or DEFAULT_FOLDS_PATH)
    data = _load_yaml_mapping(file_path)
    folds = load_folds(file_path)
    raw_folds = data.get("folds")
    if not isinstance(raw_folds, list):
        raise ConfigError(f"{file_path} needs a folds list")
    by_name = {str(entry.get("name") or "").strip(): entry for entry in raw_folds if isinstance(entry, dict)}
    records = []
    for fold in folds:
        entry = by_name.get(fold.name) or {}
        records.append(
            ArtifactRecord(
                fold_name=fold.name,
                role=fold.role,
                holdout_events=tuple(fold.holdout_events),
                seed=int(fold.seed),
                enabled=bool(fold.enabled),
                claim_evidence=bool(fold.claim_evidence),
                host=fold.host,
                output_dir=fold.output_dir,
                manifest_sha256=_optional_digest(entry.get("manifest_sha256")),
                checkpoint_sha256=_optional_digest(entry.get("checkpoint_sha256")),
                weights_sha256=fold.weights_sha256,
            )
        )
    return ArtifactInventory(
        source_path=str(file_path),
        weight_provenance=load_weight_provenance(file_path),
        records=tuple(records),
    )


def p4_comparison_baselines(names: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    """Return the P4 main-test baselines: entropy and 1-max p only.

    ``g0`` is refused here. It belongs to P5.
    """
    requested = tuple(P4_BASELINES if names is None else names)
    if "g0" in requested or "g_0" in requested:
        raise ConfigError("g0 belongs only to P5. The P4 main test compares only entropy and 1-max p.")
    unexpected = [name for name in requested if name not in P4_BASELINES]
    if unexpected:
        raise ConfigError(
            f"P4 baselines must be a subset of {P4_BASELINES}, got extra {tuple(unexpected)}."
        )
    if set(requested) != set(P4_BASELINES):
        raise ConfigError(
            f"P4 must compare both entropy and 1-max p. Got {requested}."
        )
    return P4_BASELINES


def p5_comparison_baselines(names: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    requested = tuple(P5_BASELINES if names is None else names)
    unexpected = [name for name in requested if name not in P5_BASELINES]
    if unexpected:
        raise ConfigError(f"P5 baselines must be a subset of {P5_BASELINES}, got extra {tuple(unexpected)}.")
    return P5_BASELINES


def _assert_protocol_text(text: str) -> None:
    for pattern, label in _REQUIRED_PROTOCOL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is None:
            raise ConfigError(
                f"docs/eval_protocol.md is missing required protocol text for {label} "
                f"(pattern {pattern!r})."
            )


def _read_official_recipe_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Official recipe config is missing: {path}")
    data = _load_yaml_mapping(path)
    data_cfg = data.get("data") or {}
    backbone = data.get("backbone") or {}
    if not isinstance(data_cfg, dict) or not isinstance(backbone, dict):
        raise ConfigError(f"{path} has invalid data or backbone sections")
    values = {
        "backbone_name": backbone.get("name"),
        "image_size_data": data_cfg.get("image_size"),
        "image_size_backbone": backbone.get("image_size"),
        "freeze_backbone": backbone.get("freeze"),
        "batch_size": data.get("batch_size"),
        "num_classes": (data.get("damage") or {}).get("num_classes") if isinstance(data.get("damage"), dict) else None,
    }
    if values["image_size_data"] != OFFICIAL_RECIPE["image_size"]:
        raise ConfigError(
            f"{path} data.image_size is {values['image_size_data']}, "
            f"but the official recipe is {OFFICIAL_RECIPE['image_size']}."
        )
    if values["image_size_backbone"] != OFFICIAL_RECIPE["image_size"]:
        raise ConfigError(
            f"{path} backbone.image_size is {values['image_size_backbone']}, "
            f"but the official recipe is {OFFICIAL_RECIPE['image_size']}."
        )
    if values["freeze_backbone"] is not True:
        raise ConfigError(f"{path} backbone.freeze must be true in the official recipe")
    if values["batch_size"] != OFFICIAL_RECIPE["batch_size"]:
        raise ConfigError(
            f"{path} batch_size is {values['batch_size']}, "
            f"but the official recipe is {OFFICIAL_RECIPE['batch_size']}."
        )
    if values["num_classes"] not in (None, len(OFFICIAL_CLASS_ORDER)):
        raise ConfigError(
            f"{path} damage.num_classes is {values['num_classes']}, expected {len(OFFICIAL_CLASS_ORDER)}."
        )
    return values


def _load_yaml_mapping(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Cannot read {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def _optional_digest(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "TBD":
        return None
    return text.lower()


def fold_artifact(fold: FoldSpec, inventory: ArtifactInventory) -> ArtifactRecord:
    return inventory.record(fold.name)

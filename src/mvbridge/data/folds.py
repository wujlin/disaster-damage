"""Leave-one-reference-event-out folds for the satellite damage head.

Folds are an appendable list in a YAML file. Routing reads that list. It does
not hardcode fold names or a fixed number of reference events. Adding a fold
is a new list entry.

A fold head drops every image whose ``metadata.disaster`` is a holdout event
from the official train, test, and hold splits. Those images are a separate
eval set. They are not train, val, or that head's in-fold test or hold.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from mvbridge.config import load_yaml
from mvbridge.data.xbd import XBDBuildingDataset
from mvbridge.data.xbd_labels import OFFICIAL_SPLIT_NAMES, image_id_from_filename, iter_label_paths
from mvbridge.errors import ConfigError, DataError

DEFAULT_FOLDS_PATH = "configs/folds/event_folds.yaml"
_SPLIT_LABEL = {"train": "train", "test": "test", "holdout": "hold"}


@dataclass(frozen=True)
class FoldSpec:
    name: str
    holdout_events: tuple[str, ...]
    role: str
    enabled: bool
    claim_evidence: bool
    seed: int
    notes: str = ""
    data_root: str | None = None
    holdout_dirname: str | None = None
    weights_path: str | None = None
    weights_sha256: str | None = None
    weights_bytes: int | None = None
    host: str | None = None
    host_gpu: str | None = None
    output_dir: str | None = None
    batch_size: int | None = None
    amp_dtype: str | None = None
    torch_version: str | None = None


@dataclass(frozen=True)
class HostSpec:
    gpu: str
    data_root: str
    holdout_dirname: str
    weights_path: str
    output_root: str


@dataclass(frozen=True)
class TileRecord:
    split: str
    image_id: str
    disaster: str


def _load_train_pin(data: dict, file_path: Path) -> dict:
    raw = data.get("train") or {}
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"train in {file_path} must be a mapping")
    for key in ("batch_size", "amp_dtype", "torch_version"):
        if key not in raw:
            raise ConfigError(f"train.{key} is required in {file_path}")
    batch_size = raw["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigError(f"train.batch_size must be a positive integer in {file_path}")
    amp_dtype = _optional_amp(raw["amp_dtype"])
    if amp_dtype is None:
        raise ConfigError(f"train.amp_dtype must be bfloat16 or float16 in {file_path}")
    return {
        "batch_size": int(batch_size),
        "amp_dtype": amp_dtype,
        "torch_version": str(raw["torch_version"]).strip(),
    }


def _load_hosts(data: dict, file_path: Path) -> dict[str, HostSpec]:
    raw = data.get("hosts") or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"hosts in {file_path} must be a mapping")
    hosts = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"hosts.{name} must be a mapping")
        try:
            hosts[str(name)] = HostSpec(
                gpu=str(entry["gpu"]).strip(),
                data_root=str(entry["data_root"]).strip(),
                holdout_dirname=str(entry["holdout_dirname"]).strip(),
                weights_path=str(entry["weights_path"]).strip(),
                output_root=str(entry["output_root"]).strip().rstrip("/"),
            )
        except KeyError as exc:
            raise ConfigError(f"hosts.{name} is missing {exc.args[0]}") from exc
    return hosts


def _same_path(fold_name: str, label: str, explicit: str | None, host_value: str) -> str:
    if explicit and explicit != host_value:
        raise ConfigError(
            f"Fold {fold_name} {label} is {explicit}, but its host config says {host_value}."
        )
    return host_value


def _matching_pin(fold_name: str, label: str, explicit, shared):
    if explicit is None:
        return shared
    if shared is not None and explicit != shared:
        raise ConfigError(
            f"Fold {fold_name} {label}={explicit} does not match the shared pin {shared}. "
            "Every fold uses the same batch size, AMP dtype, and torch version. "
            "The GPU does not change them."
        )
    return explicit


def _optional_int(value, fold_name: str, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Fold {fold_name} {label} must be an integer")
    return int(value)


def _optional_amp(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text not in {"bfloat16", "float16"}:
        raise ConfigError(f"amp_dtype must be bfloat16 or float16, got {value!r}")
    return text


def _assert_shared_pins(folds: list[FoldSpec]) -> None:
    for label in ("batch_size", "amp_dtype", "torch_version"):
        values = {getattr(fold, label) for fold in folds if getattr(fold, label) is not None}
        if len(values) > 1:
            raise ConfigError(
                f"Folds disagree on {label}: {sorted(values, key=str)}. "
                "Batch size, AMP dtype, and torch version are pinned once for every host."
            )


def load_folds(path: str | Path | None = None) -> tuple[FoldSpec, ...]:
    file_path = Path(path or DEFAULT_FOLDS_PATH)
    data = load_yaml(file_path)
    raw_folds = data.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ConfigError(f"Fold config {file_path} needs a non-empty folds list")
    train_pin = _load_train_pin(data, file_path)
    hosts = _load_hosts(data, file_path)
    loaded_sha = str(((data.get("weight_provenance") or {}).get("loaded") or {}).get("sha256") or "").strip().lower()
    specs = []
    names: set[str] = set()
    for index, entry in enumerate(raw_folds):
        if not isinstance(entry, dict):
            raise ConfigError(f"Fold entry {index} in {file_path} must be a mapping")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ConfigError(f"Fold entry {index} in {file_path} needs a name")
        if name in names:
            raise ConfigError(f"Duplicate fold name {name!r} in {file_path}")
        names.add(name)
        events = entry.get("holdout_events")
        if not isinstance(events, list) or not events:
            raise ConfigError(f"Fold {name} needs a non-empty holdout_events list")
        holdout = tuple(str(event).strip() for event in events)
        if any(not event for event in holdout):
            raise ConfigError(f"Fold {name} has an empty holdout event")
        if len(set(holdout)) != len(holdout):
            raise ConfigError(f"Fold {name} repeats a holdout event")
        role = str(entry.get("role") or "").strip()
        if not role:
            raise ConfigError(f"Fold {name} needs a role")
        for key in ("enabled", "claim_evidence", "seed"):
            if key not in entry:
                raise ConfigError(f"Fold {name} needs {key}")
        if not isinstance(entry["enabled"], bool) or not isinstance(entry["claim_evidence"], bool):
            raise ConfigError(f"Fold {name} enabled and claim_evidence must be booleans")
        seed = entry["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigError(f"Fold {name} seed must be an integer")
        notes = str(entry.get("notes") or "").strip()
        data_root, holdout_dirname = _fold_data_paths(entry.get("data"), name)
        weights_path, weights_sha256, weights_bytes = _fold_weight_paths(entry.get("backbone"), name)
        host_name = str(entry.get("host") or "").strip() or None
        host = hosts.get(host_name) if host_name else None
        if host_name and host is None:
            raise ConfigError(f"Fold {name} host {host_name!r} is not defined under hosts")
        output_dir = str(entry["output_dir"]).strip() if entry.get("output_dir") else None
        host_gpu = None
        if host is not None:
            data_root = _same_path(name, "data.root", data_root, host.data_root)
            holdout_dirname = _same_path(name, "data.split_dirs.holdout", holdout_dirname, host.holdout_dirname)
            weights_path = _same_path(name, "backbone.weights_path", weights_path, host.weights_path)
            expected_output = str(Path(host.output_root) / name)
            if output_dir and output_dir != expected_output:
                raise ConfigError(
                    f"Fold {name} output_dir {output_dir} must be {expected_output}. "
                    "Checkpoints and logs stay on that host's data disk."
                )
            output_dir = expected_output
            host_gpu = host.gpu
        if weights_sha256 and loaded_sha and weights_sha256 != loaded_sha:
            raise ConfigError(
                f"Fold {name} weights_sha256 {weights_sha256} is not the shared timm-converted sha256 {loaded_sha}."
            )
        batch_size = _matching_pin(name, "batch_size", _optional_int(entry.get("batch_size"), name, "batch_size"), train_pin.get("batch_size"))
        amp_dtype = _matching_pin(name, "amp_dtype", _optional_amp(entry.get("amp_dtype")), train_pin.get("amp_dtype"))
        torch_version = _matching_pin(
            name,
            "torch_version",
            str(entry["torch_version"]).strip() if entry.get("torch_version") else None,
            train_pin.get("torch_version"),
        )
        specs.append(
            FoldSpec(
                name=name,
                holdout_events=holdout,
                role=role,
                enabled=bool(entry["enabled"]),
                claim_evidence=bool(entry["claim_evidence"]),
                seed=int(seed),
                notes=notes,
                data_root=data_root,
                holdout_dirname=holdout_dirname,
                weights_path=weights_path,
                weights_sha256=weights_sha256,
                weights_bytes=weights_bytes,
                host=host_name,
                host_gpu=host_gpu,
                output_dir=output_dir,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
                torch_version=torch_version,
            )
        )
    _assert_shared_pins(specs)
    return tuple(specs)


def load_weight_provenance(path: str | Path | None = None) -> dict:
    data = load_yaml(path or DEFAULT_FOLDS_PATH)
    provenance = data.get("weight_provenance") or {}
    if not isinstance(provenance, dict):
        raise ConfigError(f"weight_provenance in {path or DEFAULT_FOLDS_PATH} must be a mapping")
    return provenance


def apply_fold_data_paths(config: dict, fold: FoldSpec) -> None:
    """Copy the fold's data root, hold directory, and local encoder file onto the run config.

    These are paths, not hyperparameters. They replace a null training-config
    value and a CLI override of the same keys.
    """
    if fold.data_root:
        config.setdefault("data", {})["root"] = fold.data_root
    if fold.holdout_dirname:
        split_dirs = config.setdefault("data", {}).setdefault("split_dirs", {})
        if not isinstance(split_dirs, dict):
            raise ConfigError("data.split_dirs must be a mapping")
        split_dirs["holdout"] = fold.holdout_dirname
    if fold.weights_path:
        backbone = config.setdefault("backbone", {})
        if not isinstance(backbone, dict):
            raise ConfigError("backbone must be a mapping")
        backbone["weights_path"] = fold.weights_path
        backbone["pretrained"] = False
        if fold.weights_sha256:
            backbone["weights_sha256"] = fold.weights_sha256
        if fold.weights_bytes is not None:
            backbone["weights_bytes"] = fold.weights_bytes
    if fold.output_dir:
        config["output_dir"] = fold.output_dir
    if fold.batch_size is not None:
        config["batch_size"] = fold.batch_size
    if fold.amp_dtype:
        config["amp_dtype"] = fold.amp_dtype
    if fold.torch_version:
        config["torch_version"] = fold.torch_version
    if fold.host:
        config["host"] = fold.host
        config["host_gpu"] = fold.host_gpu
    provenance = load_weight_provenance(config.get("folds_config"))
    if provenance:
        config["weight_provenance"] = provenance


def _fold_data_paths(data_cfg, name: str) -> tuple[str | None, str | None]:
    if data_cfg is None:
        return None, None
    if not isinstance(data_cfg, dict):
        raise ConfigError(f"Fold {name} data must be a mapping")
    root = data_cfg.get("root")
    data_root = str(root).strip() if root else None
    split_dirs = data_cfg.get("split_dirs") or {}
    if split_dirs and not isinstance(split_dirs, dict):
        raise ConfigError(f"Fold {name} data.split_dirs must be a mapping")
    holdout = split_dirs.get("holdout") if isinstance(split_dirs, dict) else None
    holdout_dirname = str(holdout).strip() if holdout else None
    return data_root, holdout_dirname


def _fold_weight_paths(backbone_cfg, name: str) -> tuple[str | None, str | None, int | None]:
    if backbone_cfg is None:
        return None, None, None
    if not isinstance(backbone_cfg, dict):
        raise ConfigError(f"Fold {name} backbone must be a mapping")
    weights_path = backbone_cfg.get("weights_path")
    weights_sha = backbone_cfg.get("weights_sha256")
    weights_bytes = backbone_cfg.get("weights_bytes")
    if weights_bytes is not None and (isinstance(weights_bytes, bool) or not isinstance(weights_bytes, int)):
        raise ConfigError(f"Fold {name} backbone.weights_bytes must be an integer")
    return (
        str(weights_path).strip() if weights_path else None,
        str(weights_sha).strip().lower() if weights_sha else None,
        None if weights_bytes is None else int(weights_bytes),
    )


def resolve_fold(config: dict) -> FoldSpec:
    name = str(config.get("fold") or "").strip()
    if not name:
        raise ConfigError("fold is required")
    folds = load_folds(config.get("folds_config") or DEFAULT_FOLDS_PATH)
    match = next((fold for fold in folds if fold.name == name), None)
    if match is None:
        known = ", ".join(fold.name for fold in folds)
        raise ConfigError(f"Unknown fold {name!r}. Known folds: {known}")
    if not match.enabled:
        raise ConfigError(
            f"Fold {name} is disabled. Set enabled: true on that list entry to use it. "
            "Routing does not special-case fold names."
        )
    return match


def route_oof(disaster: str, folds: Sequence[FoldSpec], *, claim_only: bool = False) -> list[FoldSpec]:
    """Enabled heads that held this event out.

    A head is eligible only when the exact disaster string is one of its
    holdout events. Heads that saw the event are not returned. The result is
    not an average. Callers that need one head should use ``select_oof_head``.
    """
    event = str(disaster).strip()
    if not event:
        raise ConfigError("OOF routing needs an exact metadata.disaster string")
    eligible = []
    for fold in folds:
        if not fold.enabled:
            continue
        if claim_only and not fold.claim_evidence:
            continue
        if event in fold.holdout_events:
            eligible.append(fold)
    return eligible


def assert_head_may_score(fold: FoldSpec, disaster: str) -> None:
    event = str(disaster).strip()
    if event not in fold.holdout_events:
        raise ConfigError(
            f"Head {fold.name} was trained with {event!r} still in its splits. "
            "It must not produce the satellite damage grade for that event."
        )


def select_oof_head(disaster: str, folds: Sequence[FoldSpec], *, claim_only: bool = False) -> FoldSpec:
    """Return the one head that may score this event. Refuse to average."""
    eligible = route_oof(disaster, folds, claim_only=claim_only)
    if not eligible:
        raise ConfigError(
            f"No enabled fold holds out {disaster!r}. "
            "Refusing to score it with a head that saw the event."
        )
    if len(eligible) > 1:
        names = ", ".join(fold.name for fold in eligible)
        raise ConfigError(f"Folds {names} all hold out {disaster!r}. Refusing to average their scores.")
    head = eligible[0]
    assert_head_may_score(head, disaster)
    return head


def plan_oof(
    tiles: Sequence[TileRecord],
    folds: Sequence[FoldSpec],
    *,
    claim_only: bool = True,
) -> dict[str, list[str]]:
    """Assign each tile to the single head that never saw its event.

    Tiles whose event is not held out by an eligible fold are omitted. They
    are not given to a head that saw the event. Two eligible heads is an error.
    """
    groups: dict[str, list[str]] = {}
    for tile in tiles:
        eligible = route_oof(tile.disaster, folds, claim_only=claim_only)
        if not eligible:
            continue
        if len(eligible) > 1:
            names = ", ".join(fold.name for fold in eligible)
            raise ConfigError(
                f"Image {tile.image_id} ({tile.disaster}) is held out by {names}. Refusing to average."
            )
        head = eligible[0]
        assert_head_may_score(head, tile.disaster)
        groups.setdefault(head.name, []).append(tile.image_id)
    return {name: sorted(set(image_ids)) for name, image_ids in groups.items()}


def read_metadata_disaster(path: Path) -> str:
    """Return the exact ``metadata.disaster`` string. Filenames are not consulted."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or "disaster" not in metadata or metadata.get("disaster") is None:
        raise DataError(f"{path} has no metadata.disaster. Fold membership does not use the filename.")
    text = str(metadata["disaster"]).strip()
    if not text:
        raise DataError(f"{path} has an empty metadata.disaster. Fold membership does not use the filename.")
    return text


def index_official_tiles(root: str | Path, split_dirs: dict) -> dict[str, list[TileRecord]]:
    """Image ids and exact disaster names for train, test, and hold.

    A tile counts when the official split has a pre/post image pair and a post
    label. ``holdout`` is this repo's name for the official hold split.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise DataError(f"xBD root does not exist: {root_path}")
    if not isinstance(split_dirs, dict):
        raise ConfigError("xbd.split_dirs must be a mapping")
    unknown = [name for name in split_dirs if name not in OFFICIAL_SPLIT_NAMES]
    if unknown:
        raise ConfigError(
            f"xBD split names {unknown} are not official. Use train, test, and holdout. "
            "holdout is the official hold split."
        )
    seen: dict[str, str] = {}
    indexed: dict[str, list[TileRecord]] = {}
    for split in OFFICIAL_SPLIT_NAMES:
        if split not in split_dirs:
            raise ConfigError(
                f"xbd.split_dirs.{split} is required to apply a fold. "
                "The holdout key is the official hold split."
            )
        split_path = root_path / str(split_dirs[split])
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise DataError(
                f"Official {split} split needs images/ and labels/ under {split_path}. "
                "If the release named the hold directory hold, set split_dirs.holdout to hold."
            )
        pairs = _group_images(images_dir)
        labels = _group_labels(labels_dir)
        records = []
        unlabeled = []
        for image_id, images in sorted(pairs.items()):
            if "pre" not in images or "post" not in images:
                continue
            if image_id in seen:
                raise DataError(
                    f"xBD image id {image_id} appears in both {seen[image_id]!r} and {split!r}. "
                    "Official splits are disjoint."
                )
            seen[image_id] = split
            label_files = labels.get(image_id) or {}
            post = label_files.get("post")
            if post is None:
                unlabeled.append(image_id)
                continue
            disaster = read_metadata_disaster(post)
            pre = label_files.get("pre")
            if pre is not None:
                pre_disaster = read_metadata_disaster(pre)
                if pre_disaster != disaster:
                    raise DataError(
                        f"{image_id} pre metadata.disaster {pre_disaster!r} does not match post {disaster!r}."
                    )
            records.append(TileRecord(split=split, image_id=image_id, disaster=disaster))
        if unlabeled:
            preview = ", ".join(unlabeled[:5])
            raise DataError(
                f"Official {split} split has image pairs without a post label, so metadata.disaster "
                f"cannot be read: {preview}"
            )
        indexed[split] = records
    return indexed


def count_disaster_images(records: Sequence[TileRecord], disaster: str) -> int:
    event = str(disaster).strip()
    return sum(1 for record in records if record.disaster == event)


def kept_tiles(records: Sequence[TileRecord], holdout_events: Sequence[str]) -> list[TileRecord]:
    holdout = set(holdout_events)
    return [record for record in records if record.disaster not in holdout]


def assert_zero_holdout(kept_by_split: dict[str, Sequence[TileRecord]], fold: FoldSpec) -> None:
    """Count kept images by exact metadata.disaster and require zero holdout hits."""
    for split, records in kept_by_split.items():
        label = _SPLIT_LABEL.get(split, split)
        for event in fold.holdout_events:
            count = count_disaster_images(records, event)
            if count != 0:
                raise DataError(
                    f"Fold {fold.name} left {count} images with metadata.disaster={event!r} "
                    f"in the official {label} split. Expected 0."
                )


def assert_kept_ids_have_zero_holdout(
    root: str | Path,
    split_dirs: dict,
    kept_ids_by_split: dict[str, Sequence[str]],
    fold: FoldSpec,
) -> None:
    """Re-read metadata.disaster for the image ids a head keeps, and require zero."""
    root_path = Path(root)
    for split in OFFICIAL_SPLIT_NAMES:
        image_ids = list(kept_ids_by_split.get(split, ()))
        labels = _group_labels(root_path / str(split_dirs[split]) / "labels")
        label = _SPLIT_LABEL.get(split, split)
        for event in fold.holdout_events:
            count = 0
            for image_id in image_ids:
                post = (labels.get(image_id) or {}).get("post")
                if post is None:
                    raise DataError(f"Kept image {image_id} in the official {label} split has no post label")
                if read_metadata_disaster(post) == event:
                    count += 1
            if count != 0:
                raise DataError(
                    f"Fold {fold.name} left {count} images with metadata.disaster={event!r} "
                    f"in the official {label} split. Expected 0."
                )


def assert_dataset_excludes_events(examples, events: Sequence[str], *, fold_name: str, split: str) -> None:
    label = _SPLIT_LABEL.get(split, split)
    for event in events:
        image_ids = {example.image_id for example in examples if example.disaster == event}
        count = len(image_ids)
        if count != 0:
            raise DataError(
                f"Fold {fold_name} left {count} images with metadata.disaster={event!r} "
                f"in the official {label} split. Expected 0."
            )


def build_manifest(fold: FoldSpec, tiles: dict[str, list[TileRecord]], split_dirs: dict) -> dict:
    holdout = set(fold.holdout_events)
    splits_out = {}
    per_event: dict[str, dict[str, dict[str, int]]] = {}
    held_out_counts = {}
    used_counts = {}
    eval_ids: list[str] = []
    kept_by_split: dict[str, list[TileRecord]] = {}
    for split in OFFICIAL_SPLIT_NAMES:
        used_ids = []
        held_ids = []
        counts: dict[str, dict[str, int]] = {}
        kept: list[TileRecord] = []
        for tile in tiles[split]:
            bucket = counts.setdefault(tile.disaster, {"used": 0, "held_out": 0})
            if tile.disaster in holdout:
                bucket["held_out"] += 1
                held_ids.append(tile.image_id)
                eval_ids.append(tile.image_id)
            else:
                bucket["used"] += 1
                used_ids.append(tile.image_id)
                kept.append(tile)
        overlap = set(used_ids) & set(held_ids)
        if overlap:
            raise DataError(f"Fold {fold.name} listed images as both used and held out in {split}: {sorted(overlap)[:5]}")
        splits_out[split] = {
            "used_image_ids": sorted(used_ids),
            "held_out_image_ids": sorted(held_ids),
        }
        per_event[split] = counts
        held_out_counts[split] = len(held_ids)
        used_counts[split] = len(used_ids)
        kept_by_split[split] = kept
    assert_zero_holdout(kept_by_split, fold)
    return {
        "fold": fold.name,
        "role": fold.role,
        "claim_evidence": fold.claim_evidence,
        "enabled": fold.enabled,
        "seed": fold.seed,
        "holdout_events": list(fold.holdout_events),
        "official_splits": list(OFFICIAL_SPLIT_NAMES),
        "split_dirs": {split: str(split_dirs[split]) for split in OFFICIAL_SPLIT_NAMES},
        "split_note": (
            "holdout is the official hold split. "
            "Held-out images are removed from train, test, and hold, and listed as fold_eval_image_ids."
        ),
        "splits": splits_out,
        "fold_eval_image_ids": sorted(eval_ids),
        "per_event_counts": per_event,
        "held_out_counts": held_out_counts,
        "used_counts": used_counts,
    }


def canonical_manifest_bytes(payload: dict) -> bytes:
    """UTF-8 JSON with sorted keys and compact separators. The sha256 field is omitted."""
    body = {key: value for key, value in payload.items() if key != "sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_manifest_bytes(payload)).hexdigest()


def finalize_manifest(payload: dict) -> dict:
    body = {key: value for key, value in payload.items() if key != "sha256"}
    body["sha256"] = manifest_sha256(body)
    return body


def write_manifest(path: Path, manifest: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def bind_event_fold(config: dict, logger, *, data_filter: str) -> tuple[FoldSpec, dict]:
    """Index train/test/hold, write the manifest, and assert the holdout count is 0.

    ``data_filter`` is ``exclude-holdout`` for train and in-fold test/hold, or
    ``held-out-only`` for the separate eval set. The manifest hash does not
    depend on which of those this process is running.
    """
    fold = resolve_fold(config)
    data = config.setdefault("data", {})
    root = data.get("root")
    if not root:
        raise ConfigError("data.root is required. Set it in the config or pass --set data.root=...")
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ConfigError("output_dir is required")
    split_dirs = data.get("split_dirs") or {"train": "train", "test": "test", "holdout": "holdout"}
    tiles = index_official_tiles(root, split_dirs)
    manifest = finalize_manifest(build_manifest(fold, tiles, split_dirs))
    kept_ids = {split: manifest["splits"][split]["used_image_ids"] for split in OFFICIAL_SPLIT_NAMES}
    assert_kept_ids_have_zero_holdout(root, split_dirs, kept_ids, fold)
    held_total = sum(manifest["held_out_counts"].values())
    if held_total == 0:
        raise DataError(
            f"Fold {fold.name} matched no images with metadata.disaster in {list(fold.holdout_events)} "
            "across train, test, and hold. Membership is the exact disaster string. "
            "Refusing to continue with a fold that holds nothing out."
        )
    manifest_path = Path(output_dir) / f"fold_{fold.name}_manifest.json"
    write_manifest(manifest_path, manifest)
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    if written.get("sha256") != manifest["sha256"]:
        raise DataError(f"Wrote {manifest_path} but the sha256 did not round-trip")
    config["seed"] = fold.seed
    config["fold_manifest_sha256"] = manifest["sha256"]
    config["fold_role"] = fold.role
    config["fold_claim_evidence"] = fold.claim_evidence
    config["holdout_events"] = list(fold.holdout_events)
    if data_filter == "exclude-holdout":
        data["exclude_disasters"] = list(fold.holdout_events)
        data.pop("include_only_disasters", None)
    elif data_filter == "held-out-only":
        data.pop("exclude_disasters", None)
        data["include_only_disasters"] = list(fold.holdout_events)
    else:
        raise ConfigError(f"Unknown fold data filter {data_filter!r}")
    logger.info(
        "fold=%s role=%s claim_evidence=%s holdout_events=%s seed=%s manifest=%s sha256=%s held_out_counts=%s",
        fold.name,
        fold.role,
        fold.claim_evidence,
        list(fold.holdout_events),
        fold.seed,
        manifest_path,
        manifest["sha256"],
        manifest["held_out_counts"],
    )
    for split in OFFICIAL_SPLIT_NAMES:
        logger.info("fold per-event counts %s %s", _SPLIT_LABEL.get(split, split), manifest["per_event_counts"][split])
    return fold, manifest


class FoldHeldOutDataset(Dataset):
    """Images of the held-out event, taken from train, test, and hold together.

    This is the only eval set that contains that event for this head.
    """

    def __init__(self, config: dict, holdout_events: Sequence[str]):
        events = [str(event).strip() for event in holdout_events if str(event).strip()]
        if not events:
            raise ConfigError("fold eval needs holdout_events")
        super().__init__()
        self.holdout_events = tuple(events)
        self.parts = []
        self.examples = []
        for split in OFFICIAL_SPLIT_NAMES:
            part_cfg = dict(config)
            part_cfg.pop("exclude_disasters", None)
            part_cfg["include_only_disasters"] = list(self.holdout_events)
            part = XBDBuildingDataset(part_cfg, split=split, allow_empty=True)
            self.parts.append(part)
            self.examples.extend(part.examples)
        if not self.examples:
            raise DataError(
                "Fold eval set is empty. No held-out event images with classified buildings "
                "were found in train, test, or hold."
            )
        extra = sorted({example.disaster for example in self.examples if example.disaster not in self.holdout_events})
        if extra:
            raise DataError(f"Fold eval set contains events outside the holdout: {extra}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        for part in self.parts:
            if index < len(part):
                return part[index]
            index -= len(part)
        raise IndexError(index)

    @property
    def image_ids(self) -> set[str]:
        return {example.image_id for example in self.examples}


def _group_images(images_dir: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        image_id, when = image_id_from_filename(path.name)
        if when not in {"pre", "post"}:
            continue
        pairs.setdefault(image_id, {})[when] = path
    return pairs


def _group_labels(labels_dir: Path) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in iter_label_paths(labels_dir):
        image_id, when = image_id_from_filename(path.name)
        grouped.setdefault(image_id, {})[when or "post"] = path
    return grouped

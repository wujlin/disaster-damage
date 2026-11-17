"""Location-grouped SAGIN dataloader.

Each index is one place (group id). The four views of that place are the
DINO multi-view positives. Simulated UAV assets stay in the identity sample
and are omitted from damage evidence.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from mvbridge.data.exclusions import read_id_list
from mvbridge.data.images import IMAGENET_MEAN, IMAGENET_STD, load_rgb_tensor
from mvbridge.data.records import CANONICAL_VIEWS, DAMAGE_OBSERVATION, VIEWPOINT_BRIDGE, PlaceGroup
from mvbridge.data.sagin_index import load_sagin_groups
from mvbridge.errors import ConfigError, DataError


def assert_damage_evidence(sample: dict) -> None:
    """Raise if a simulated or viewpoint-bridge view is being treated as damage evidence."""
    roles = sample.get("roles") or {}
    claimed = list(sample.get("damage_evidence_views") or [])
    illegal = [view for view in claimed if roles.get(view) == VIEWPOINT_BRIDGE]
    if illegal:
        raise DataError(
            "Simulated UAV / viewpoint-bridge imagery cannot be used as damage evidence: "
            + ", ".join(illegal)
        )


class SAGINMultiViewDataset(Dataset):
    def __init__(self, config: dict):
        policy = str(config.get("missing_view_policy", "allow_partial"))
        if policy not in {"allow_partial", "skip_group", "error"}:
            raise ConfigError("missing_view_policy must be allow_partial, skip_group, or error")
        self.policy = policy
        self.min_views = int(config.get("min_views", 2))
        self.image_size = int(config.get("image_size", 224))
        self.seed = int(config.get("seed", 0))
        self.mean = tuple(config.get("mean", IMAGENET_MEAN))
        self.std = tuple(config.get("std", IMAGENET_STD))
        check_files = bool(config.get("check_files", True))
        groups = load_sagin_groups(config, check_files=check_files)
        excluded = read_id_list(config.get("exclude_ids_file"))
        self.dropped_excluded = 0
        self.dropped_missing = 0
        self.groups: list[PlaceGroup] = []
        for group in groups:
            if group.group_id in excluded:
                self.dropped_excluded += 1
                continue
            _dedupe_assets(group)
            if not _keep_group(group, policy=policy, min_views=self.min_views):
                self.dropped_missing += 1
                continue
            self.groups.append(group)
        if groups and not self.groups and self.dropped_excluded < len(groups) and check_files:
            sample = next((asset.path for group in groups for asset in group.assets if asset.exists is False), None)
            raise DataError(
                "No SAGIN groups were usable. Check data roots, missing views, and the missing_view_policy. "
                f"First missing file: {sample}"
            )
        self.source_group_count = len(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict:
        group = self.groups[index]
        rng = random.Random(self.seed + index)
        images = {}
        roles = {}
        paths = {}
        bridge_videos = []
        for view in CANONICAL_VIEWS:
            assets = [asset for asset in group.assets_for(view) if asset.exists is not False]
            if not assets:
                images[view] = None
                continue
            images_only = [asset for asset in assets if asset.kind == "image"]
            videos = [asset for asset in assets if asset.kind == "video"]
            bridge_videos.extend(asset.path for asset in videos if asset.is_viewpoint_bridge)
            chosen = rng.choice(images_only) if images_only else None
            roles[view] = chosen.role if chosen is not None else assets[0].role
            paths[view] = chosen.path if chosen is not None else assets[0].path
            if chosen is None:
                images[view] = None
                continue
            if chosen.exists is False or not chosen.path:
                images[view] = None
                continue
            images[view] = load_rgb_tensor(chosen.path, self.image_size, self.mean, self.std)
        present = [view for view in CANONICAL_VIEWS if images[view] is not None]
        sample = {
            "group_id": group.group_id,
            "lat": None if group.lat is None else float(group.lat),
            "lon": None if group.lon is None else float(group.lon),
            "event_name": group.event_name or "",
            "event_date": group.event_date or "",
            "images": images,
            "roles": {view: roles[view] for view in present},
            "paths": {view: paths[view] for view in CANONICAL_VIEWS if view in paths},
            "identity_views": present,
            "damage_evidence_views": [view for view in present if roles[view] == DAMAGE_OBSERVATION],
            "bridge_videos": bridge_videos,
            "view_present": {view: images[view] is not None for view in CANONICAL_VIEWS},
        }
        assert_damage_evidence(sample)
        return sample


def collate_sagin(batch: list[dict]) -> dict:
    """Stack views independently. A missing view is a zero tensor and a False mask."""
    if not batch:
        raise DataError("Cannot collate an empty SAGIN batch")
    reference = next((image for sample in batch for image in sample["images"].values() if image is not None), None)
    if reference is None:
        raise DataError("SAGIN batch has no images to define a tensor shape")
    out = {
        "group_id": [sample["group_id"] for sample in batch],
        "identity_views": [sample["identity_views"] for sample in batch],
        "damage_evidence_views": [sample["damage_evidence_views"] for sample in batch],
    }
    for view in CANONICAL_VIEWS:
        tensors = []
        present = []
        for sample in batch:
            image = sample["images"][view]
            present.append(image is not None)
            tensors.append(image if image is not None else torch.zeros_like(reference))
        out[view] = torch.stack(tensors, dim=0)
        out[f"{view}_present"] = torch.tensor(present, dtype=torch.bool)
    return out


def _keep_group(group: PlaceGroup, *, policy: str, min_views: int) -> bool:
    existing = []
    loadable = []
    for view in CANONICAL_VIEWS:
        assets = [asset for asset in group.assets_for(view) if asset.exists is not False]
        if not assets:
            continue
        existing.append(view)
        if any(asset.kind == "image" for asset in assets):
            loadable.append(view)
    if policy == "error" and len(existing) < len(CANONICAL_VIEWS):
        missing = [view for view in CANONICAL_VIEWS if view not in existing]
        raise DataError(f"SAGIN group {group.group_id} is missing views {missing} and missing_view_policy is error")
    if policy == "skip_group" and len(existing) < len(CANONICAL_VIEWS):
        return False
    return len(loadable) >= min_views


def _dedupe_assets(group: PlaceGroup) -> None:
    seen = set()
    kept = []
    for asset in group.assets:
        key = (asset.view, asset.path)
        if key in seen:
            continue
        seen.add(key)
        kept.append(asset)
    group.assets = kept

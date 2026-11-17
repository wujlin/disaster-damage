"""Torch-free records shared by the audit and the dataloaders."""

from __future__ import annotations

from dataclasses import dataclass, field


VIEWPOINT_BRIDGE = "viewpoint_bridge"
DAMAGE_OBSERVATION = "damage_observation"
CANONICAL_VIEWS = ("vgi", "svi", "uav", "rsi")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class ViewAsset:
    view: str
    path: str
    role: str
    is_simulated: bool
    kind: str  # image, video, other
    exists: bool | None = None

    @property
    def is_viewpoint_bridge(self) -> bool:
        return self.role == VIEWPOINT_BRIDGE or self.is_simulated


@dataclass
class PlaceGroup:
    group_id: str
    lat: float | None
    lon: float | None
    event_name: str | None = None
    event_date: str | None = None
    assets: list[ViewAsset] = field(default_factory=list)
    source: str = ""

    def assets_for(self, view: str) -> list[ViewAsset]:
        return [asset for asset in self.assets if asset.view == view]

    def present_views(self, *, loadable_images_only: bool = False) -> list[str]:
        found = []
        for view in CANONICAL_VIEWS:
            assets = self.assets_for(view)
            if loadable_images_only:
                assets = [asset for asset in assets if asset.kind == "image" and asset.exists is not False]
            if assets:
                found.append(view)
        return found


def role_for_asset(view: str, is_simulated: bool) -> str:
    """Simulated imagery, including the simulated UAV bridge, is not damage evidence."""
    if is_simulated:
        return VIEWPOINT_BRIDGE
    return DAMAGE_OBSERVATION


def media_kind(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    dotted = f".{suffix}"
    if dotted in IMAGE_SUFFIXES:
        return "image"
    if dotted in VIDEO_SUFFIXES:
        return "video"
    return "other"

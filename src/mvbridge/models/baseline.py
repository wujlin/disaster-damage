"""Frozen DINOv2 ViT-B plus a separate damage head."""

from __future__ import annotations

import torch
from torch import nn

from mvbridge.errors import ConfigError
from mvbridge.models.pathways import (
    DamageHead,
    DamageStatePathway,
    IdentityGeometryPathway,
    TimmDinoEncoder,
)


class BuildingDamageBaseline(nn.Module):
    def __init__(self, identity: IdentityGeometryPathway, damage: DamageStatePathway, head: DamageHead):
        super().__init__()
        self.identity = identity
        self.damage = damage
        self.head = head

    def forward(self, pre_images: torch.Tensor, post_images: torch.Tensor) -> torch.Tensor:
        state = self.damage_state(pre_images, post_images)
        return self.head(state)

    def damage_state(self, pre_images: torch.Tensor, post_images: torch.Tensor):
        pre_features = self.identity(pre_images)
        post_features = self.identity(post_images)
        return self.damage(pre_features, post_features)

    def encode_identity(self, images: torch.Tensor):
        """Identity features for a future multi-view loss. The damage loss does not call this."""
        return self.identity(images)

    def damage_parameters(self):
        yield from self.damage.parameters()
        yield from self.head.parameters()

    def identity_parameters(self):
        yield from self.identity.parameters()


def build_baseline(config: dict, *, pretrained: bool | None = None) -> BuildingDamageBaseline:
    backbone_cfg = config.get("backbone") or {}
    damage_cfg = config.get("damage") or {}
    data_cfg = config.get("data") or {}
    name = str(backbone_cfg.get("name", "vit_base_patch14_dinov2"))
    image_size = int(backbone_cfg.get("image_size", data_cfg.get("image_size", 224)))
    data_image_size = data_cfg.get("image_size")
    if data_image_size is not None and int(data_image_size) != image_size:
        raise ConfigError(
            f"backbone.image_size ({image_size}) and data.image_size ({data_image_size}) must match"
        )
    if image_size % 14 != 0:
        raise ConfigError(f"image_size must be a multiple of 14 for DINOv2 ViT-B/14, got {image_size}")
    use_pretrained = bool(backbone_cfg.get("pretrained", False)) if pretrained is None else pretrained
    weights_path = backbone_cfg.get("weights_path") or None
    encoder = TimmDinoEncoder(
        name=name,
        image_size=image_size,
        pretrained=use_pretrained,
        weights_path=weights_path,
    )
    identity = IdentityGeometryPathway(encoder, freeze=bool(backbone_cfg.get("freeze", True)))
    proj_dim = int(damage_cfg.get("proj_dim", 256))
    num_classes = int(damage_cfg.get("num_classes", 4))
    if num_classes != 4:
        raise ConfigError("The week-1 xBD head has 4 damage classes: no-damage, minor, major, destroyed")
    damage = DamageStatePathway(encoder.out_dim, proj_dim)
    head = DamageHead(proj_dim, num_classes=num_classes, dropout=float(damage_cfg.get("dropout", 0.1)))
    return BuildingDamageBaseline(identity, damage, head)


def save_checkpoint(path, model: BuildingDamageBaseline, config: dict, epoch: int, step: int) -> None:
    payload = {
        "damage_state_pathway": model.damage.state_dict(),
        "damage_head": model.head.state_dict(),
        "config": config,
        "epoch": epoch,
        "step": step,
        "classes": ["no-damage", "minor-damage", "major-damage", "destroyed"],
        "backbone_name": (config.get("backbone") or {}).get("name", "vit_base_patch14_dinov2"),
    }
    torch.save(payload, path)


def load_damage_weights(model: BuildingDamageBaseline, path) -> dict:
    blob = torch.load(path, map_location="cpu")
    model.damage.load_state_dict(blob["damage_state_pathway"])
    model.head.load_state_dict(blob["damage_head"])
    return blob

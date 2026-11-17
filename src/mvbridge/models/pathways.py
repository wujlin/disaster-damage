"""Identity/geometry and damage-state pathways.

The identity pathway is the only place a future DINO multi-view loss should
attach. The damage pathway reads detached features and owns its own
parameters, so a damage loss cannot wash damage cues out of the identity
representation. A later residual or change-detection head replaces
``DamageHead`` and can read ``DamageState.residual`` plus detached patch tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mvbridge.errors import ConfigError

ALLOWED_BACKBONES = {"vit_base_patch14_dinov2", "vit_base_patch14_dinov2.lvd142m"}


@dataclass
class IdentityFeatures:
    cls_token: torch.Tensor
    patch_tokens: torch.Tensor


@dataclass
class DamageState:
    pre: torch.Tensor
    post: torch.Tensor
    residual: torch.Tensor
    pre_patches: torch.Tensor | None = None
    post_patches: torch.Tensor | None = None


class IdentityGeometryPathway(nn.Module):
    """DINOv2 ViT-B, frozen in the week-1 baseline.

    ``forward`` is what the damage pathway calls. With ``freeze=True`` the
    encoder runs under ``no_grad``. The damage pathway also detaches, so the
    damage loss stays off this module even if a later experiment unfreezes it
    and trains it with a separate invariance loss.
    """

    def __init__(self, encoder: nn.Module, freeze: bool = True):
        super().__init__()
        self.encoder = encoder
        self.freeze = freeze
        if freeze:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> IdentityFeatures:
        if self.freeze:
            with torch.no_grad():
                return self.encoder(images)
        return self.encoder(images)


class TimmDinoEncoder(nn.Module):
    """DINOv2 ViT-B/14 via timm. ViT-L and larger names are rejected."""

    def __init__(self, name: str, image_size: int, pretrained: bool = False, weights_path: str | None = None):
        super().__init__()
        lowered = name.lower()
        if any(token in lowered for token in ("large", "giant", "huge")):
            raise ConfigError("ViT-L and larger backbones are out of scope for this scaffold. Use DINOv2 ViT-B.")
        if name not in ALLOWED_BACKBONES:
            raise ConfigError(f"Backbone {name!r} is not DINOv2 ViT-B. Allowed: {sorted(ALLOWED_BACKBONES)}")
        if image_size % 14 != 0:
            raise ConfigError(f"image_size must be a multiple of 14 for DINOv2 ViT-B/14, got {image_size}")
        import timm

        self.backbone = timm.create_model(
            name,
            pretrained=bool(pretrained) and not weights_path,
            num_classes=0,
            img_size=image_size,
        )
        if weights_path:
            _load_encoder_weights(self.backbone, weights_path)
        self.out_dim = int(self.backbone.embed_dim)

    def forward(self, images: torch.Tensor) -> IdentityFeatures:
        features = self.backbone.forward_features(images)
        if features.ndim != 3:
            raise RuntimeError(f"Expected DINOv2 tokens [B, N, C], got shape {tuple(features.shape)}")
        n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        return IdentityFeatures(cls_token=features[:, 0], patch_tokens=features[:, n_prefix:])


class DamageStatePathway(nn.Module):
    """Project pre and post features into a damage space and form a residual.

    Inputs are detached. None of these parameters belong to the identity pathway.
    """

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.pre_proj = nn.Linear(in_dim, proj_dim)
        self.post_proj = nn.Linear(in_dim, proj_dim)

    def forward(self, pre: IdentityFeatures, post: IdentityFeatures) -> DamageState:
        pre_hidden = self.pre_proj(pre.cls_token.detach())
        post_hidden = self.post_proj(post.cls_token.detach())
        pre_patches = None if pre.patch_tokens is None else pre.patch_tokens.detach()
        post_patches = None if post.patch_tokens is None else post.patch_tokens.detach()
        return DamageState(
            pre=pre_hidden,
            post=post_hidden,
            residual=post_hidden - pre_hidden,
            pre_patches=pre_patches,
            post_patches=post_patches,
        )


class DamageHead(nn.Module):
    """Baseline classifier on the residual and the post-event projection.

    Replace this module with a change-detection head. The contract is
    ``forward(DamageState) -> logits [B, num_classes]``.
    """

    def __init__(self, proj_dim: int, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, state: DamageState) -> torch.Tensor:
        return self.net(torch.cat([state.residual, state.post], dim=-1))


def _load_encoder_weights(module: nn.Module, path: str) -> None:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise ConfigError(f"Encoder weights at {path} are not a state dict")
    cleaned = {}
    for key, value in blob.items():
        new_key = key
        for prefix in ("module.", "backbone.", "encoder."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    module.load_state_dict(cleaned, strict=True)

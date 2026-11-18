"""Official-split xBD baseline, pathway separation, and the ViT-B dry run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from mvbridge.cli import main
from mvbridge.data.xbd import XBDBuildingDataset
from mvbridge.data.xbd_labels import DAMAGE_CLASSES
from mvbridge.engine.metrics import classification_metrics
from mvbridge.engine.train import dry_run_forward, run_training
from mvbridge.errors import ConfigError, DataError
from mvbridge.models.baseline import BuildingDamageBaseline, build_baseline, load_damage_weights, save_checkpoint
from mvbridge.models.pathways import DamageState, DamageStatePathway, IdentityFeatures, IdentityGeometryPathway

ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.out_dim = dim
        self.fc = nn.Linear(3, dim)

    def forward(self, images: torch.Tensor) -> IdentityFeatures:
        pooled = images.mean(dim=(2, 3))
        cls = self.fc(pooled)
        return IdentityFeatures(cls_token=cls, patch_tokens=cls.unsqueeze(1))


def _baseline(dim: int = 8, freeze: bool = False) -> BuildingDamageBaseline:
    identity = IdentityGeometryPathway(TinyEncoder(dim), freeze=freeze)
    damage = DamageStatePathway(dim, proj_dim=4)
    head = nn.Identity()  # replaced immediately

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(8, 4)

        def forward(self, state: DamageState) -> torch.Tensor:
            return self.net(torch.cat([state.residual, state.post], dim=-1))

    return BuildingDamageBaseline(identity, damage, _Head())


def _write_pair(root: Path, split: str, image_id: str, subtype: str, extra_subtype: str | None = "un-classified") -> None:
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for when in ("pre", "post"):
        Image.new("RGB", (64, 64), (20, 40, 60) if when == "pre" else (80, 10, 10)).save(
            images / f"{image_id}_{when}_disaster.png"
        )
        buildings = [
            {
                "properties": {"feature_type": "building", "subtype": subtype if when == "post" else "no-damage", "uid": "b1"},
                "wkt": "POLYGON ((8 8, 40 8, 40 36, 8 36, 8 8))",
            }
        ]
        if extra_subtype and when == "post":
            buildings.append(
                {
                    "properties": {"feature_type": "building", "subtype": extra_subtype, "uid": "b2"},
                    "wkt": "POLYGON ((48 48, 60 48, 60 60, 48 60, 48 48))",
                }
            )
        buildings.append(
            {
                "properties": {"feature_type": "flood", "subtype": "water", "uid": "water"},
                "wkt": "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            }
        )
        payload = {
            "features": {"xy": buildings, "lng_lat": []},
            "metadata": {
                "disaster": image_id.split("_")[0],
                "img_name": f"{image_id}_{when}_disaster.png",
            },
        }
        (labels / f"{image_id}_{when}_disaster.json").write_text(json.dumps(payload), encoding="utf-8")


def _xbd_config(root: Path) -> dict:
    return {
        "root": str(root),
        "image_size": 32,
        "crop_padding": 2,
        "split_dirs": {"train": "train", "test": "test", "holdout": "holdout"},
    }


def test_dataset_uses_only_the_official_split(tmp_path):
    _write_pair(tmp_path, "train", "hurricane-harvey_00000001", "minor-damage")
    _write_pair(tmp_path, "test", "hurricane-harvey_00000099", "destroyed")
    _write_pair(tmp_path, "val", "custom-split_00000001", "major-damage")
    # Incomplete official pair: pre image only.
    lone = tmp_path / "train" / "images" / "hurricane-harvey_00000002_pre_disaster.png"
    Image.new("RGB", (64, 64), (1, 1, 1)).save(lone)
    dataset = XBDBuildingDataset(_xbd_config(tmp_path), split="train")
    assert dataset.image_ids == {"hurricane-harvey_00000001"}
    assert "custom-split_00000001" not in dataset.image_ids
    assert dataset.class_counts["minor-damage"] == 1
    assert dataset.class_counts["destroyed"] == 0
    sample = dataset[0]
    assert sample["pre"].shape == (3, 32, 32)
    assert sample["post"].shape == (3, 32, 32)
    assert int(sample["label"]) == DAMAGE_CLASSES.index("minor-damage")
    assert sample["split"] == "train"
    with pytest.raises(ConfigError, match="official"):
        XBDBuildingDataset(_xbd_config(tmp_path), split="val")


def test_exclusion_file_removes_official_test_ids(tmp_path):
    _write_pair(tmp_path, "test", "hurricane-michael_00000004", "major-damage", extra_subtype=None)
    exclusion = tmp_path / "drop.txt"
    exclusion.write_text("hurricane-michael_00000004\n", encoding="utf-8")
    config = _xbd_config(tmp_path)
    config["exclude_ids_file"] = str(exclusion)
    with pytest.raises(DataError, match="no classified buildings"):
        XBDBuildingDataset(config, split="test")


def test_damage_loss_does_not_update_identity():
    model = _baseline(freeze=False)
    before = model.identity.encoder.fc.weight.detach().clone()
    pre = torch.rand(4, 3, 8, 8)
    post = torch.rand(4, 3, 8, 8)
    state = model.damage_state(pre, post)
    assert torch.allclose(state.residual, state.post - state.pre)
    logits = model(pre, post)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert model.identity.encoder.fc.weight.grad is None
    optimizer = torch.optim.SGD(list(model.damage_parameters()), lr=0.1)
    optimizer.step()
    assert torch.equal(model.identity.encoder.fc.weight, before)
    assert model.damage.pre_proj.weight.grad is not None

    class ResidualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, damage_state: DamageState) -> torch.Tensor:
            assert damage_state.post_patches is not None
            return self.fc(damage_state.residual)

    model.head = ResidualHead()
    replaced = model(pre, post)
    assert replaced.shape == (4, 4)


def test_weighted_f1_matches_a_hand_count():
    metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["per_class"]["no-damage"]["f1"] == pytest.approx(0.8)
    assert metrics["weighted_f1"] == pytest.approx((0.8 * 3 + (2.0 / 3.0)) / 4.0)


def test_checkpoint_round_trip_keeps_damage_weights(tmp_path):
    model = _baseline()
    path = tmp_path / "checkpoint_last.pt"
    save_checkpoint(path, model, {"backbone": {"name": "tiny"}}, epoch=0, step=1)
    other = _baseline()
    with torch.no_grad():
        other.damage.pre_proj.weight.zero_()
    load_damage_weights(other, path)
    assert torch.allclose(other.damage.pre_proj.weight, model.damage.pre_proj.weight)


def test_vitb_dry_run_and_rejects_larger_backbones():
    # 224 here is the short-side probe. The official U recipe is image size 518.
    logger = torch_logger()
    result = dry_run_forward(
        {
            "backbone": {"name": "vit_base_patch14_dinov2", "image_size": 224, "pretrained": True, "freeze": True},
            "damage": {"proj_dim": 32, "num_classes": 4, "dropout": 0.0},
            "data": {"image_size": 224},
        },
        logger,
    )
    assert result["logits_shape"] == [2, 4]
    assert result["identity_trainable"] == 0
    assert 80_000_000 < result["identity_params"] < 95_000_000
    assert result["damage_params"] > 0
    with pytest.raises(ConfigError, match="ViT-L"):
        build_baseline({"backbone": {"name": "vit_large_patch14_dinov2", "image_size": 224, "pretrained": False}, "damage": {}})
    with pytest.raises(ConfigError, match="multiple of 14"):
        build_baseline(
            {
                "backbone": {"name": "vit_base_patch14_dinov2", "image_size": 225, "pretrained": False},
                "data": {"image_size": 225},
                "damage": {"num_classes": 4},
            }
        )


def test_train_cli_missing_root_does_not_write_a_checkpoint(tmp_path):
    code = main(
        [
            "train-xbd",
            "--config",
            str(ROOT / "configs" / "xbd_dinov2_vitb.yaml"),
            "--set",
            "data.root=null",
            "--set",
            f"output_dir={tmp_path / 'run'}",
        ]
    )
    assert code == 2
    assert not (tmp_path / "run" / "checkpoint_last.pt").exists()


def test_dry_run_cli(tmp_path, capsys):
    code = main(
        [
            "train-xbd",
            "--config",
            str(ROOT / "configs" / "xbd_dinov2_vitb.yaml"),
            "--dry-run",
            "--set",
            "damage.proj_dim=32",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err or "dry-run" in captured.out
    assert not (tmp_path / "checkpoint_last.pt").exists()


def test_one_training_step_updates_only_damage_params(tmp_path):
    model = _baseline(freeze=True)
    before = model.identity.encoder.fc.weight.detach().clone()
    batch = {
        "pre": torch.rand(2, 3, 8, 8),
        "post": torch.rand(2, 3, 8, 8),
        "label": torch.tensor([1, 3]),
        "image_id": ["a", "b"],
    }
    optimizer = torch.optim.SGD(list(model.damage_parameters()), lr=0.05)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(batch["pre"], batch["post"]), batch["label"])
    loss.backward()
    optimizer.step()
    assert torch.equal(model.identity.encoder.fc.weight, before)
    assert loss.ndim == 0


def torch_logger():
    import logging

    logger = logging.getLogger("mvbridge-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def test_run_training_is_not_invoked_by_import():
    # The import of the training module must not require a dataset.
    assert callable(run_training)

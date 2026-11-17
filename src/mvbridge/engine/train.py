"""Training loop for the frozen DINOv2 ViT-B xBD probe.

Importing this module does not start training.
"""

from __future__ import annotations

import copy
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvbridge.data.xbd import XBDBuildingDataset
from mvbridge.errors import ConfigError
from mvbridge.models.baseline import build_baseline, save_checkpoint


def run_training(config: dict, logger, *, dry_run: bool = False, max_steps: int | None = None) -> dict:
    if dry_run:
        return dry_run_forward(config, logger)
    data_cfg = config.get("data") or {}
    if not data_cfg.get("root"):
        raise ConfigError("data.root is required. Set it in the config or pass --set data.root=...")
    _check_image_size(config)
    dataset = XBDBuildingDataset(data_cfg, split="train")
    logger.info("official train buildings=%s class_counts=%s stats=%s", len(dataset), dataset.class_counts, dataset.stats)
    logger.info("first train image ids=%s", sorted(dataset.image_ids)[:5])
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        drop_last=False,
    )
    device = _device(config)
    logger.info("python=%s torch=%s device=%s", platform.python_version(), torch.__version__, device)
    _log_weight_policy(config, logger)
    model = build_baseline(config).to(device)
    _log_parameters(model, logger)
    optimizer = torch.optim.AdamW(
        list(model.damage_parameters()),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    criterion = torch.nn.CrossEntropyLoss()
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ConfigError("output_dir is required")
    epochs = int(config.get("epochs", 1))
    log_every = int(config.get("log_every", 20))
    step = 0
    last_loss = None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        correct = 0
        for batch_index, batch in enumerate(loader):
            if epoch == 0 and batch_index == 0:
                logger.info(
                    "first batch pre=%s post=%s labels=%s image_ids=%s",
                    tuple(batch["pre"].shape),
                    tuple(batch["post"].shape),
                    batch["label"].tolist(),
                    list(batch["image_id"])[:4],
                )
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pre, post)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            running += float(loss.item()) * batch_size
            seen += batch_size
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            last_loss = float(loss.item())
            if step % log_every == 0:
                logger.info(
                    "epoch=%s step=%s loss=%.4f batch_acc=%.3f lr=%s",
                    epoch,
                    step,
                    last_loss,
                    correct / max(seen, 1),
                    optimizer.param_groups[0]["lr"],
                )
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        logger.info(
            "epoch=%s mean_loss=%.4f acc=%.3f seen=%s",
            epoch,
            running / max(seen, 1),
            correct / max(seen, 1),
            seen,
        )
        checkpoint_path = Path(output_dir) / "checkpoint_last.pt"
        save_checkpoint(checkpoint_path, model, config, epoch, step)
        logger.info("wrote %s", checkpoint_path)
        if max_steps is not None and step >= max_steps:
            break
    return {"steps": step, "last_loss": last_loss}


def dry_run_forward(config: dict, logger) -> dict:
    """Build ViT-B with random weights and run one forward. No dataset and no download."""
    cfg = copy.deepcopy(config)
    backbone = cfg.setdefault("backbone", {})
    backbone["pretrained"] = False
    backbone["weights_path"] = None
    _check_image_size(cfg)
    logger.info("dry-run: pretrained download skipped, xBD root not read, no checkpoint written")
    model = build_baseline(cfg, pretrained=False)
    _log_parameters(model, logger)
    image_size = int(backbone.get("image_size", 224))
    pre = torch.zeros(2, 3, image_size, image_size)
    post = torch.zeros(2, 3, image_size, image_size)
    model.eval()
    with torch.no_grad():
        logits = model(pre, post)
    logger.info("dry-run logits shape=%s", tuple(logits.shape))
    if tuple(logits.shape) != (2, 4):
        raise ConfigError(f"Expected logits [2, 4], got {tuple(logits.shape)}")
    identity = sum(parameter.numel() for parameter in model.identity_parameters())
    identity_trainable = sum(parameter.numel() for parameter in model.identity_parameters() if parameter.requires_grad)
    damage = sum(parameter.numel() for parameter in model.damage_parameters())
    return {
        "logits_shape": list(logits.shape),
        "dry_run": True,
        "identity_params": identity,
        "identity_trainable": identity_trainable,
        "damage_params": damage,
    }


def _check_image_size(config: dict) -> None:
    backbone = config.get("backbone") or {}
    data = config.get("data") or {}
    image_size = int(backbone.get("image_size", data.get("image_size", 224)))
    if image_size % 14 != 0:
        raise ConfigError(f"image_size must be a multiple of 14 for DINOv2 ViT-B/14, got {image_size}")


def _device(config: dict) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _log_weight_policy(config: dict, logger) -> None:
    backbone = config.get("backbone") or {}
    if backbone.get("weights_path"):
        logger.info("loading DINOv2 ViT-B weights from local file %s", backbone["weights_path"])
    elif backbone.get("pretrained", False):
        logger.info(
            "pretrained=true will ask timm for DINOv2 ViT-B weights. "
            "They go to the torch/timm cache, not into this git repository."
        )
    else:
        logger.info("pretrained=false, backbone starts from random weights")


def _log_parameters(model, logger) -> None:
    identity = sum(parameter.numel() for parameter in model.identity_parameters())
    identity_trainable = sum(parameter.numel() for parameter in model.identity_parameters() if parameter.requires_grad)
    damage = sum(parameter.numel() for parameter in model.damage_parameters())
    logger.info(
        "identity_params=%s identity_trainable=%s damage_pathway_and_head_trainable=%s",
        identity,
        identity_trainable,
        damage,
    )


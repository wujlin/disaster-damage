"""Evaluation on an official xBD test or holdout split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvbridge.data.xbd import XBDBuildingDataset
from mvbridge.engine.metrics import classification_metrics
from mvbridge.errors import ConfigError, DataError
from mvbridge.models.baseline import build_baseline, load_damage_weights


def run_evaluation(config: dict, logger, *, split: str, checkpoint: str) -> dict:
    if split not in {"test", "holdout"}:
        raise ConfigError("eval-xbd scores the official test or holdout split. It does not score train.")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise DataError(f"Checkpoint does not exist: {checkpoint_path}")
    data_cfg = config.get("data") or {}
    if data_cfg.get("exclude_ids_file"):
        logger.info(
            "exclude_ids_file is set. This score drops those official-split images and is not the full official %s split.",
            split,
        )
    dataset = XBDBuildingDataset(data_cfg, split=split)
    logger.info("official %s buildings=%s class_counts=%s", split, len(dataset), dataset.class_counts)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    model = build_baseline(config).to(device)
    load_damage_weights(model, checkpoint_path)
    metrics = evaluate_loader(model, loader, device)
    metrics["split"] = split
    metrics["official_split"] = not bool(data_cfg.get("exclude_ids_file"))
    metrics["checkpoint"] = str(checkpoint_path)
    output_dir = Path(config.get("output_dir") or checkpoint_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics_{split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "split=%s count=%s accuracy=%.4f weighted_f1=%.4f wrote %s",
        split,
        metrics["count"],
        metrics["accuracy"],
        metrics["weighted_f1"],
        metrics_path,
    )
    logger.info("confusion_matrix=%s", metrics["confusion_matrix"])
    return metrics


def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    truths = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            logits = model(pre, post)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(batch["label"].tolist())
    return classification_metrics(truths, predictions)

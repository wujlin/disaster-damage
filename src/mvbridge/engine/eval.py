"""Evaluation on an official xBD test or holdout split.

``fold-eval`` scores only the images a fold head held out of train, test, and hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvbridge.data.folds import (
    FoldHeldOutDataset,
    apply_fold_data_paths,
    assert_dataset_excludes_events,
    assert_head_may_score,
    bind_event_fold,
    resolve_fold,
)
from mvbridge.engine.runtime import collect_run_env, require_same_fold_env, write_run_env
from mvbridge.models.pathways import prepare_local_backbone
from mvbridge.data.xbd import XBDBuildingDataset
from mvbridge.engine.metrics import classification_metrics
from mvbridge.errors import ConfigError, DataError
from mvbridge.models.baseline import build_baseline, load_damage_weights


def run_evaluation(config: dict, logger, *, split: str, checkpoint: str) -> dict:
    if split not in {"test", "holdout", "fold-eval"}:
        raise ConfigError(
            "eval-xbd scores the official test or holdout split, or fold-eval for a fold head. "
            "It does not score train."
        )
    if split == "fold-eval" and not config.get("fold"):
        raise ConfigError("fold-eval requires --fold")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise DataError(f"Checkpoint does not exist: {checkpoint_path}")
    if config.get("fold"):
        apply_fold_data_paths(config, resolve_fold(config))
        if not config.get("training_output_dir"):
            config["training_output_dir"] = config.get("output_dir")
        _separate_cross_env_output(config)
    if config.get("output_dir") or config.get("fold"):
        current_env = collect_run_env()
        if config.get("fold"):
            require_same_fold_env(config, logger, current_env)
        if config.get("output_dir"):
            write_run_env(config["output_dir"], current_env, logger)
    prepare_local_backbone(config, logger)
    fold = None
    manifest = None
    if config.get("fold"):
        if split == "fold-eval":
            fold, manifest = bind_event_fold(config, logger, data_filter="held-out-only")
            dataset = FoldHeldOutDataset(config.get("data") or {}, fold.holdout_events)
            for event in sorted({example.disaster for example in dataset.examples}):
                assert_head_may_score(fold, event)
            logger.info(
                "fold-eval images=%s holdout_events=%s. These images were removed from train, test, and hold.",
                len(dataset.image_ids),
                list(fold.holdout_events),
            )
        else:
            fold, manifest = bind_event_fold(config, logger, data_filter="exclude-holdout")
            dataset = XBDBuildingDataset(config.get("data") or {}, split=split)
            assert_dataset_excludes_events(
                dataset.examples,
                fold.holdout_events,
                fold_name=fold.name,
                split=split,
            )
            logger.info(
                "in-fold %s excludes %s. This is not out-of-fold scoring. Use --split fold-eval for the held-out event.",
                split,
                list(fold.holdout_events),
            )
    else:
        data_cfg = config.get("data") or {}
        if data_cfg.get("exclude_ids_file"):
            logger.info(
                "exclude_ids_file is set. This score drops those official-split images and is not the full official %s split.",
                split,
            )
        dataset = XBDBuildingDataset(data_cfg, split=split)
    data_cfg = config.get("data") or {}
    if not config.get("fold"):
        logger.info("official %s buildings=%s class_counts=%s", split, len(dataset), dataset.class_counts)
    elif hasattr(dataset, "class_counts"):
        logger.info("fold %s buildings=%s class_counts=%s", split, len(dataset), dataset.class_counts)
    else:
        logger.info("fold %s buildings=%s", split, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    model = build_baseline(config).to(device)
    blob = load_damage_weights(model, checkpoint_path)
    assert_checkpoint_fold(blob, fold)
    metrics, prediction_rows = evaluate_loader(model, loader, device)
    disaster_by_image = {}
    if hasattr(dataset, "examples"):
        for example in dataset.examples:
            disaster_by_image.setdefault(example.image_id, getattr(example, "disaster", "") or "")
    for row in prediction_rows:
        row["disaster"] = disaster_by_image.get(row["image_id"], "")
    metrics["per_event"] = _per_event_counts(prediction_rows, truths_by_key=None)
    metrics["split"] = split
    metrics["official_split"] = not bool(data_cfg.get("exclude_ids_file")) and not config.get("fold")
    metrics["checkpoint"] = str(checkpoint_path)
    if fold is not None and manifest is not None:
        metrics["fold"] = fold.name
        metrics["fold_manifest_sha256"] = manifest["sha256"]
        metrics["holdout_events"] = list(fold.holdout_events)
    output_dir = Path(config.get("output_dir") or checkpoint_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics_{split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions_path = output_dir / f"predictions_{split}.json"
    predictions_path.write_text(
        json.dumps({"records": prediction_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("predictions wrote %s", predictions_path)
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


def assert_checkpoint_fold(blob: dict, fold) -> None:
    """Refuse a head that would score an event it saw, or a different fold's images."""
    recorded = (blob.get("config") or {}).get("fold")
    recorded_name = str(recorded).strip() if recorded else ""
    if fold is None:
        if recorded_name:
            raise DataError(
                f"Checkpoint was trained with fold {recorded_name!r}. "
                "Pass --fold so this head is not applied to an event it saw."
            )
        return
    if recorded_name != fold.name:
        raise DataError(
            f"Checkpoint fold {recorded_name or None!r} does not match --fold {fold.name!r}. "
            "Refusing to score these images with a different head."
        )


def evaluate_loader(model, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    truths = []
    predictions = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            logits = model(pre, post)
            batch_predictions = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(batch_predictions)
            truths.extend(batch["label"].tolist())
            image_ids = list(batch.get("image_id") or [])
            building_uids = list(batch.get("building_uid") or [])
            for index, prediction in enumerate(batch_predictions):
                rows.append(
                    {
                        "image_id": "" if index >= len(image_ids) else str(image_ids[index]),
                        "building_uid": "" if index >= len(building_uids) else str(building_uids[index]),
                        "prediction": int(prediction),
                        "label": int(batch["label"][index]),
                    }
                )
    rows.sort(key=lambda row: (row["image_id"], row["building_uid"]))
    return classification_metrics(truths, predictions), rows


def _separate_cross_env_output(config: dict) -> None:
    if not config.get("cross_env_check"):
        return
    training_dir = config.get("training_output_dir")
    output_dir = config.get("output_dir")
    if not training_dir or not output_dir:
        raise ConfigError("--cross-env-check needs the fold training directory and a separate output_dir.")
    if Path(output_dir).resolve() == Path(training_dir).resolve():
        raise ConfigError(
            "--cross-env-check must write to a different directory than the fold training output "
            f"({training_dir})."
        )


def _per_event_counts(rows: list[dict], truths_by_key) -> dict:
    del truths_by_key
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        event = str(row.get("disaster") or "")
        slot = grouped.setdefault(event, {"count": 0, "correct": 0})
        slot["count"] += 1
        if row.get("label") is not None and int(row["prediction"]) == int(row["label"]):
            slot["correct"] += 1
    return {
        event: {
            "count": slot["count"],
            "accuracy": slot["correct"] / slot["count"] if slot["count"] else 0.0,
        }
        for event, slot in sorted(grouped.items())
    }

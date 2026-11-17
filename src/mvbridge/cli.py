"""Command line entry points for the week-1 scaffold.

audit, sagin-smoke, train-xbd, eval-xbd, and oof-route never start work at import time.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from mvbridge.audit.overlap import format_report, run_audit
from mvbridge.config import apply_overrides, load_yaml
from mvbridge.data.folds import (
    DEFAULT_FOLDS_PATH,
    apply_fold_data_paths,
    load_folds,
    resolve_fold,
    select_oof_head,
)
from mvbridge.engine.runtime import prediction_agreement
from mvbridge.data.sagin import SAGINMultiViewDataset
from mvbridge.demo import prepare_audit_demo, prepare_sagin_demo, summarize_sagin_dataset
from mvbridge.engine.eval import run_evaluation
from mvbridge.engine.logutil import configure_logging, log_config
from mvbridge.engine.train import run_training
from mvbridge.errors import ConfigError, DataError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mvbridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit event and spatial overlap between SAGIN and xBD")
    _add_config_args(audit, "configs/overlap_audit.yaml")
    audit.add_argument("--demo", action="store_true", help="Run the audit on a tiny in-memory fixture")

    smoke = subparsers.add_parser("sagin-smoke", help="Load SAGIN place groups and print one multi-view sample")
    _add_config_args(smoke, "configs/sagin_loader.yaml")
    smoke.add_argument("--demo", action="store_true", help="Run the loader on fake metadata and tiny images")

    train = subparsers.add_parser("train-xbd", help="Train the DINOv2 ViT-B damage probe on the official xBD train split")
    _add_config_args(train, "configs/xbd_dinov2_vitb.yaml")
    _add_fold_args(train)
    train.add_argument("--dry-run", action="store_true", help="Build ViT-B and run one forward without data or downloads")
    train.add_argument("--max-steps", type=int, default=None)

    evaluate = subparsers.add_parser("eval-xbd", help="Evaluate a damage-head checkpoint on official test, holdout, or a fold eval set")
    _add_config_args(evaluate, "configs/xbd_dinov2_vitb.yaml")
    _add_fold_args(evaluate)
    evaluate.add_argument("--split", required=True, choices=["test", "holdout", "fold-eval"])
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--cross-env-check",
        action="store_true",
        help=(
            "Score a fold checkpoint in a different environment. "
            "Requires an output_dir that is not the fold training directory."
        ),
    )

    agree = subparsers.add_parser(
        "cross-env-agree",
        help="Agreement of satellite grades between two environment prediction files",
    )
    agree.add_argument("--left", required=True, help="predictions JSON from the training environment")
    agree.add_argument("--right", required=True, help="predictions JSON from the other environment")

    route = subparsers.add_parser(
        "oof-route",
        help="Route one metadata.disaster value to the head that never saw that event",
    )
    route.add_argument("--disaster", required=True, help="Exact xBD metadata.disaster string")
    route.add_argument("--folds", default=DEFAULT_FOLDS_PATH, help="Appendable fold list")
    route.add_argument(
        "--claim-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reference folds only. Pipeline-check folds are excluded unless --no-claim-only is set.",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            return _audit(args)
        if args.command == "sagin-smoke":
            return _smoke(args)
        if args.command == "train-xbd":
            return _train(args)
        if args.command == "eval-xbd":
            return _eval(args)
        if args.command == "oof-route":
            return _oof_route(args)
        if args.command == "cross-env-agree":
            return _cross_env_agree(args)
    except (ConfigError, DataError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"error: unknown command {args.command}")
    return 2


def _add_config_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a config key, as in data.root=/data/xbd")


def _add_fold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fold",
        default=None,
        help=(
            "Leave-one-reference-event-out fold name. "
            "Omit to keep the full official split. "
            f"Names come from {DEFAULT_FOLDS_PATH}."
        ),
    )
    parser.add_argument(
        "--folds",
        default=DEFAULT_FOLDS_PATH,
        help="Appendable YAML fold list. Adding a fold is a new entry in this file.",
    )


def _audit(args) -> int:
    base = load_yaml(args.config)
    apply_overrides(base, args.overrides)
    if args.demo:
        output_dir = Path(base.get("output_dir") or tempfile.mkdtemp(prefix="mvbridge-audit-"))
        output_dir.mkdir(parents=True, exist_ok=True)
        catalog = base.get("catalog", "configs/event_catalog.yaml")
        config = prepare_audit_demo(output_dir / "fixture", catalog, output_dir, buffer_m=float(base.get("buffer_m", 1000)))
    else:
        config = base
    payload = run_audit(config)
    print(format_report(payload), end="")
    return 0


def _smoke(args) -> int:
    base = load_yaml(args.config)
    apply_overrides(base, args.overrides)
    if args.demo:
        output_dir = Path(tempfile.mkdtemp(prefix="mvbridge-sagin-"))
        config = prepare_sagin_demo(output_dir)
    else:
        config = base
    dataset = SAGINMultiViewDataset(config)
    print(summarize_sagin_dataset(dataset), end="")
    return 0


def _train(args) -> int:
    config = apply_overrides(load_yaml(args.config), args.overrides)
    _attach_fold(config, args)
    logger = configure_logging(None if args.dry_run else config.get("output_dir"))
    log_config(logger, config)
    run_training(config, logger, dry_run=args.dry_run, max_steps=args.max_steps)
    return 0


def _eval(args) -> int:
    config = apply_overrides(load_yaml(args.config), args.overrides)
    _attach_fold(config, args)
    logger = configure_logging(config.get("output_dir"))
    log_config(logger, config)
    run_evaluation(config, logger, split=args.split, checkpoint=args.checkpoint)
    return 0


def _oof_route(args) -> int:
    folds = load_folds(args.folds)
    head = select_oof_head(args.disaster, folds, claim_only=args.claim_only)
    print(f"disaster={args.disaster}")
    print(f"fold={head.name}")
    print(f"seed={head.seed}")
    print(f"role={head.role}")
    print(f"claim_evidence={str(head.claim_evidence).lower()}")
    print("holdout_events=" + ",".join(head.holdout_events))
    print("averaged=false")
    return 0


def _cross_env_agree(args) -> int:
    left = _load_prediction_records(args.left)
    right = _load_prediction_records(args.right)
    report = prediction_agreement(left, right)
    rate = report["agreement"]
    print(f"compared={report['compared']}")
    print(f"matched={report['matched']}")
    print(f"agreement={rate:.6f}")
    print(f"threshold={report['threshold']}")
    print(f"no_effect={str(report['no_effect']).lower()}")
    for event, slot in report["per_event"].items():
        label = event or "(missing-event)"
        print(f"event={label} compared={slot['compared']} agreement={slot['agreement']:.6f}")
    print("failed_events=" + ",".join(report["failed_events"]))
    if report["annotation"]:
        print(f"annotation={report['annotation']}")
    return 0


def _load_prediction_records(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
    else:
        records = payload
    if not isinstance(records, list):
        raise DataError(f"Prediction file has no records list: {path}")
    return records


def _attach_fold(config: dict, args) -> None:
    fold_name = getattr(args, "fold", None)
    if not fold_name:
        return
    requested_output = config.get("output_dir")
    config["fold"] = fold_name
    config["folds_config"] = args.folds
    config["cross_env_check"] = bool(getattr(args, "cross_env_check", False))
    fold = resolve_fold(config)
    explicit_seed = _explicit_override(args.overrides, "seed")
    if explicit_seed is not None and int(explicit_seed) != fold.seed:
        raise ConfigError(
            f"Fold {fold.name} fixes seed {fold.seed}. The command passed seed={explicit_seed}. "
            "Edit the fold entry to change the seed."
        )
    config["seed"] = fold.seed
    config["fold_role"] = fold.role
    config["fold_claim_evidence"] = fold.claim_evidence
    config["holdout_events"] = list(fold.holdout_events)
    apply_fold_data_paths(config, fold)
    config["training_output_dir"] = fold.output_dir or config.get("output_dir")
    if config["cross_env_check"]:
        training_dir = config.get("training_output_dir")
        if not requested_output or not training_dir:
            raise ConfigError("--cross-env-check needs --set output_dir=... and a fold training directory.")
        if Path(requested_output).resolve() == Path(training_dir).resolve():
            raise ConfigError(
                "--cross-env-check must write to a different directory than the fold training output "
                f"({training_dir})."
            )
        config["output_dir"] = requested_output


def _explicit_override(overrides: list[str] | None, key: str):
    found = None
    for item in overrides or []:
        name, separator, raw = item.partition("=")
        if separator and name.strip() == key:
            found = yaml.safe_load(raw)
    return found


if __name__ == "__main__":
    raise SystemExit(main())

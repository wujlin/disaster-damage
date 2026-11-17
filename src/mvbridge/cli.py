"""Command line entry points for the week-1 scaffold.

audit, sagin-smoke, train-xbd, and eval-xbd never start work at import time.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from mvbridge.audit.overlap import format_report, run_audit
from mvbridge.config import apply_overrides, load_yaml
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
    train.add_argument("--dry-run", action="store_true", help="Build ViT-B and run one forward without data or downloads")
    train.add_argument("--max-steps", type=int, default=None)

    evaluate = subparsers.add_parser("eval-xbd", help="Evaluate a damage-head checkpoint on official test or holdout")
    _add_config_args(evaluate, "configs/xbd_dinov2_vitb.yaml")
    evaluate.add_argument("--split", required=True, choices=["test", "holdout"])
    evaluate.add_argument("--checkpoint", required=True)

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
    except (ConfigError, DataError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"error: unknown command {args.command}")
    return 2


def _add_config_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a config key, as in data.root=/data/xbd")


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
    logger = configure_logging(None if args.dry_run else config.get("output_dir"))
    log_config(logger, config)
    run_training(config, logger, dry_run=args.dry_run, max_steps=args.max_steps)
    return 0


def _eval(args) -> int:
    config = apply_overrides(load_yaml(args.config), args.overrides)
    logger = configure_logging(config.get("output_dir"))
    log_config(logger, config)
    run_evaluation(config, logger, split=args.split, checkpoint=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leave-one-reference-event-out folds. Fixtures are synthetic; no xBD download."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from mvbridge.cli import main
from mvbridge.config import load_yaml
from mvbridge.data.folds import (
    FoldHeldOutDataset,
    FoldSpec,
    apply_fold_data_paths,
    TileRecord,
    assert_dataset_excludes_events,
    assert_head_may_score,
    assert_kept_ids_have_zero_holdout,
    assert_zero_holdout,
    bind_event_fold,
    build_manifest,
    count_disaster_images,
    finalize_manifest,
    index_official_tiles,
    kept_tiles,
    load_folds,
    plan_oof,
    resolve_fold,
    route_oof,
    select_oof_head,
    write_manifest,
)
from mvbridge.engine.eval import assert_checkpoint_fold
from mvbridge.engine.runtime import assert_pinned_torch, torch_release
from mvbridge.data.xbd import XBDBuildingDataset
from mvbridge.errors import ConfigError, DataError

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "configs" / "folds" / "event_folds.yaml"


def _logger() -> logging.Logger:
    logger = logging.getLogger("mvbridge-folds-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _write_tile(root: Path, split: str, image_id: str, disaster: str) -> None:
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for when in ("pre", "post"):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(images / f"{image_id}_{when}_disaster.png")
        payload = {
            "features": {
                "xy": [
                    {
                        "properties": {
                            "feature_type": "building",
                            "subtype": "minor-damage" if when == "post" else "no-damage",
                            "uid": "b1",
                        },
                        "wkt": "POLYGON ((4 4, 20 4, 20 20, 4 20, 4 4))",
                    }
                ],
                "lng_lat": [],
            },
            "metadata": {
                "disaster": disaster,
                "img_name": f"{image_id}_{when}_disaster.png",
            },
        }
        (labels / f"{image_id}_{when}_disaster.json").write_text(json.dumps(payload), encoding="utf-8")


def _data_config(root: Path) -> dict:
    return {
        "root": str(root),
        "image_size": 32,
        "crop_padding": 2,
        "split_dirs": {"train": "train", "test": "test", "holdout": "holdout"},
    }


def _populate(root: Path) -> dict[str, set[str]]:
    groups = {name: set() for name in ("michael", "harvey", "palu", "santa", "socal")}
    named = [
        ("hurricane-michael", "michael"),
        ("hurricane-harvey", "harvey"),
        ("palu-tsunami", "palu"),
        ("santa-rosa-wildfire", "santa"),
        ("socal-fire", "socal"),
    ]
    bases = {"train": 1000, "test": 2000, "holdout": 3000}
    for split, base in bases.items():
        for offset, (event, key) in enumerate(named):
            image_id = f"{event}_{base + offset:08d}"
            _write_tile(root, split, image_id, event)
            groups[key].add(image_id)
    # Filename says Harvey; metadata says Michael. Fold membership must follow metadata.
    _write_tile(root, "train", "hurricane-harvey_00000050", "hurricane-michael")
    groups["michael"].add("hurricane-harvey_00000050")
    # Filename says Michael; metadata says Harvey. This tile stays in a Michael fold.
    _write_tile(root, "train", "hurricane-michael_00000051", "hurricane-harvey")
    groups["harvey"].add("hurricane-michael_00000051")
    return groups


def test_shipped_folds_are_an_appendable_list():
    raw = load_yaml(FOLDS)
    assert raw["seed"] == 1
    folds = load_folds(FOLDS)
    by_name = {fold.name: fold for fold in folds}
    michael = by_name["minus-hurricane-michael"]
    assert michael.role == "main"
    assert michael.holdout_events == ("hurricane-michael",)
    assert michael.enabled is True
    assert michael.claim_evidence is True
    assert michael.seed == 1
    palu = by_name["minus-palu-tsunami"]
    assert palu.enabled is True
    assert palu.claim_evidence is True
    assert palu.holdout_events == ("palu-tsunami",)
    assert palu.seed == 1
    santa = by_name["minus-santa-rosa"]
    assert santa.role == "pipeline-check"
    assert santa.claim_evidence is False
    assert santa.holdout_events == ("santa-rosa-wildfire",)
    socal = by_name["minus-socal"]
    assert socal.role == "pipeline-check"
    assert socal.claim_evidence is False
    assert socal.holdout_events == ("socal-fire",)
    seed2 = by_name["minus-hurricane-michael-seed2"]
    assert seed2.seed == 2
    assert seed2.role == "seed-repeat"
    assert seed2.claim_evidence is False
    assert seed2.enabled is True
    assert seed2.holdout_events == michael.holdout_events
    assert seed2.output_dir == "/data/repo/outputs/minus-hurricane-michael-seed2"
    assert seed2.output_dir != michael.output_dir
    assert {fold.name: fold.seed for fold in folds if fold.seed == 1}[michael.name] == 1
    assert all(fold.seed == 1 for fold in folds if fold.name != "minus-hurricane-michael-seed2")
    provenance = load_yaml(FOLDS)["weight_provenance"]
    timm_sha = "6cce660bf1cdf01bc70ad96d53c3ad4d7d02fd4211a669b04abba9aaa848e66c"
    facebook_sha = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
    assert provenance["loaded"]["sha256"] == timm_sha
    assert provenance["loaded"]["bytes"] == 346385151
    assert provenance["original"]["filename"] == "dinov2_vitb14_pretrain.pth"
    assert provenance["original"]["sha256"] == facebook_sha
    assert provenance["original"]["bytes"] == 346378731
    assert provenance["timm"] == "timm/vit_base_patch14_dinov2.lvd142m"
    assert provenance["hf_revision"] == "4685c99dabffe5affac90bd99dbffd25801ae58d"
    assert provenance["upstream"] == "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    wsb_weights = "/data/repo/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth"
    wsa_weights = "/mnt/data_hdd/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth"
    assert michael.host == "wsb"
    assert michael.host_gpu == "RTX 4090"
    assert michael.data_root == "/data/repo/datasets/xbd"
    assert michael.weights_path == wsb_weights
    assert michael.output_dir == "/data/repo/outputs/minus-hurricane-michael"
    assert palu.host == "wsa"
    assert palu.host_gpu == "RTX 5880"
    assert palu.data_root == "/mnt/data_hdd/datasets/xbd"
    assert palu.weights_path == wsa_weights
    assert palu.output_dir == "/mnt/data_hdd/outputs/minus-palu-tsunami"
    assert santa.host == "wsb"
    assert socal.host == "wsb"
    for fold in folds:
        assert fold.holdout_dirname == "hold"
        assert fold.weights_sha256 == timm_sha
        assert fold.weights_bytes == 346385151
        assert fold.batch_size == 32
        assert fold.amp_dtype == "bfloat16"
        assert fold.torch_version == "2.15.0"
    assert palu.batch_size == michael.batch_size == seed2.batch_size
    assert palu.amp_dtype == michael.amp_dtype == seed2.amp_dtype
    assert palu.torch_version == michael.torch_version == seed2.torch_version == "2.15.0"
    assert seed2.weights_sha256 == michael.weights_sha256 == timm_sha
    assert seed2.holdout_dirname == michael.holdout_dirname == "hold"
    assert seed2.host == "wsb"
    assert seed2.data_root == michael.data_root
    assert seed2.weights_path == michael.weights_path
    default = load_yaml(ROOT / "configs" / "xbd_dinov2_vitb.yaml")
    assert "fold" not in default
    assert default["seed"] == 0


def test_michael_seed2_holdout_matches_seed1(tmp_path):
    root = tmp_path / "xbd"
    _populate(root)
    split_dirs = _data_config(root)["split_dirs"]
    tiles = index_official_tiles(root, split_dirs)
    folds = {fold.name: fold for fold in load_folds(FOLDS)}
    seed1 = folds["minus-hurricane-michael"]
    seed2 = folds["minus-hurricane-michael-seed2"]
    assert seed1.seed == 1
    assert seed2.seed == 2
    assert seed2.role == "seed-repeat"
    assert seed2.claim_evidence is False
    assert seed2.holdout_events == seed1.holdout_events == ("hurricane-michael",)
    assert seed2.batch_size == seed1.batch_size == 32
    assert seed2.amp_dtype == seed1.amp_dtype == "bfloat16"
    assert seed2.torch_version == seed1.torch_version == "2.15.0"
    assert seed2.weights_sha256 == seed1.weights_sha256
    assert seed2.holdout_dirname == seed1.holdout_dirname == "hold"
    assert seed2.output_dir == "/data/repo/outputs/minus-hurricane-michael-seed2"
    manifest = finalize_manifest(build_manifest(seed2, tiles, split_dirs))
    kept = {split: kept_tiles(records, seed2.holdout_events) for split, records in tiles.items()}
    assert_zero_holdout(kept, seed2)
    for split in ("train", "test", "holdout"):
        assert manifest["per_event_counts"][split]["hurricane-michael"]["used"] == 0
        assert manifest["splits"][split]["held_out_image_ids"]
        assert set(manifest["splits"][split]["used_image_ids"]).isdisjoint(
            set(manifest["splits"][split]["held_out_image_ids"])
        )


def test_holdout_event_is_absent_and_other_events_stay(tmp_path):
    root = tmp_path / "xbd"
    groups = _populate(root)
    config = _data_config(root)
    tiles = index_official_tiles(root, config["split_dirs"])
    fold = next(item for item in load_folds(FOLDS) if item.name == "minus-hurricane-michael")
    manifest = finalize_manifest(build_manifest(fold, tiles, config["split_dirs"]))
    kept = {split: kept_tiles(records, fold.holdout_events) for split, records in tiles.items()}
    assert_zero_holdout(kept, fold)
    assert_kept_ids_have_zero_holdout(
        root,
        config["split_dirs"],
        {split: manifest["splits"][split]["used_image_ids"] for split in ("train", "test", "holdout")},
        fold,
    )
    for split in ("train", "test", "holdout"):
        used = set(manifest["splits"][split]["used_image_ids"])
        held = set(manifest["splits"][split]["held_out_image_ids"])
        split_ids = {record.image_id for record in tiles[split]}
        michael_ids = {record.image_id for record in tiles[split] if record.disaster == "hurricane-michael"}
        other_ids = split_ids - michael_ids
        assert michael_ids
        assert held == michael_ids
        assert used == other_ids
        assert manifest["per_event_counts"][split]["hurricane-michael"]["used"] == 0
        assert manifest["per_event_counts"][split]["hurricane-michael"]["held_out"] == len(michael_ids)
        plain = XBDBuildingDataset(config, split=split)
        filtered_cfg = dict(config)
        filtered_cfg["exclude_disasters"] = ["hurricane-michael"]
        filtered = XBDBuildingDataset(filtered_cfg, split=split)
        assert filtered.image_ids.isdisjoint(groups["michael"])
        assert filtered.image_ids == plain.image_ids - groups["michael"]
        assert filtered.image_ids == used
        assert_dataset_excludes_events(filtered.examples, ["hurricane-michael"], fold_name=fold.name, split=split)
        assert "hurricane-harvey_00000050" not in filtered.image_ids
        if split == "train":
            assert "hurricane-michael_00000051" in filtered.image_ids
            assert "hurricane-michael_00000051" in plain.image_ids


def test_no_fold_path_matches_the_unfiltered_split(tmp_path):
    root = tmp_path / "xbd"
    _populate(root)
    config = _data_config(root)
    first = XBDBuildingDataset(config, split="train")
    second_cfg = dict(config)
    second_cfg["exclude_disasters"] = []
    second_cfg["include_only_disasters"] = None
    second = XBDBuildingDataset(second_cfg, split="train")
    assert [example.image_id for example in first.examples] == [example.image_id for example in second.examples]
    assert [example.label for example in first.examples] == [example.label for example in second.examples]
    assert first.stats == second.stats
    assert "excluded_fold_images" not in first.stats
    sample = first[0]
    assert set(sample) == {"pre", "post", "label", "image_id", "building_uid", "split"}
    tiles = index_official_tiles(root, config["split_dirs"])
    assert first.image_ids == {record.image_id for record in tiles["train"]}


def test_manifest_is_deterministic(tmp_path):
    root = tmp_path / "xbd"
    _populate(root)
    config = _data_config(root)
    tiles = index_official_tiles(root, config["split_dirs"])
    fold = next(item for item in load_folds(FOLDS) if item.name == "minus-palu-tsunami")
    first = finalize_manifest(build_manifest(fold, tiles, config["split_dirs"]))
    second = finalize_manifest(build_manifest(fold, tiles, config["split_dirs"]))
    assert first["sha256"] == second["sha256"]
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    write_manifest(path_a, first)
    write_manifest(path_b, second)
    assert path_a.read_bytes() == path_b.read_bytes()
    body = json.loads(path_a.read_text(encoding="utf-8"))
    digest = body.pop("sha256")
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == digest
    assert body["seed"] == 1
    assert body["per_event_counts"]["train"]["palu-tsunami"]["used"] == 0
    assert body["per_event_counts"]["test"]["hurricane-michael"]["used"] > 0
    assert "hurricane-harvey_00000050" in body["splits"]["train"]["used_image_ids"]


def test_bind_logs_manifest_and_blocks_a_nonzero_count(tmp_path):
    root = tmp_path / "xbd"
    groups = _populate(root)
    config = {
        "fold": "minus-hurricane-michael",
        "folds_config": str(FOLDS),
        "output_dir": str(tmp_path / "out"),
        "seed": 0,
        "data": _data_config(root),
    }
    fold, manifest = bind_event_fold(config, _logger(), data_filter="exclude-holdout")
    assert fold.name == "minus-hurricane-michael"
    assert config["seed"] == 1
    assert config["data"]["exclude_disasters"] == ["hurricane-michael"]
    manifest_path = tmp_path / "out" / "fold_minus-hurricane-michael_manifest.json"
    first_bytes = manifest_path.read_bytes()
    _, again = bind_event_fold(config, _logger(), data_filter="exclude-holdout")
    assert manifest_path.read_bytes() == first_bytes
    assert again["sha256"] == manifest["sha256"]
    dataset = XBDBuildingDataset(config["data"], split="train")
    assert dataset.image_ids.isdisjoint(groups["michael"])
    bad_kept = {
        "train": ["hurricane-harvey_00000050"],
        "test": [],
        "holdout": [],
    }
    with pytest.raises(DataError, match="Expected 0"):
        assert_kept_ids_have_zero_holdout(root, config["data"]["split_dirs"], bad_kept, fold)
    poisoned = {
        "train": [TileRecord("train", "x", "hurricane-michael")],
        "test": [],
        "holdout": [],
    }
    assert count_disaster_images(poisoned["train"], "hurricane-michael") == 1
    with pytest.raises(DataError, match="Expected 0"):
        assert_zero_holdout(poisoned, fold)


def test_fold_eval_set_is_only_the_held_out_event(tmp_path):
    root = tmp_path / "xbd"
    groups = _populate(root)
    config = {
        "fold": "minus-hurricane-michael",
        "folds_config": str(FOLDS),
        "output_dir": str(tmp_path / "out"),
        "data": _data_config(root),
    }
    fold, manifest = bind_event_fold(config, _logger(), data_filter="held-out-only")
    dataset = FoldHeldOutDataset(config["data"], fold.holdout_events)
    assert dataset.image_ids == groups["michael"]
    assert dataset.image_ids.isdisjoint(groups["harvey"] | groups["palu"] | groups["santa"] | groups["socal"])
    assert set(manifest["fold_eval_image_ids"]) == groups["michael"]
    sample = dataset[0]
    assert sample["image_id"] in groups["michael"]
    for event in {example.disaster for example in dataset.examples}:
        assert_head_may_score(fold, event)
    with pytest.raises(ConfigError, match="must not produce"):
        assert_head_may_score(fold, "palu-tsunami")


def test_oof_routing_never_uses_a_seen_event_head():
    folds = load_folds(FOLDS)
    by_name = {fold.name: fold for fold in folds}
    michael = by_name["minus-hurricane-michael"]
    michael_seed2 = by_name["minus-hurricane-michael-seed2"]
    palu = select_oof_head("palu-tsunami", folds, claim_only=True)
    assert palu.name == "minus-palu-tsunami"
    chosen = select_oof_head("hurricane-michael", folds, claim_only=True)
    assert chosen.name == "minus-hurricane-michael"
    assert chosen.seed == 1
    assert [fold.name for fold in route_oof("hurricane-michael", folds, claim_only=True)] == [
        "minus-hurricane-michael"
    ]
    assert [fold.name for fold in route_oof("hurricane-michael", folds, claim_only=False)] == [
        "minus-hurricane-michael",
        "minus-hurricane-michael-seed2",
    ]
    with pytest.raises(ConfigError, match="Refusing to average"):
        select_oof_head("hurricane-michael", folds, claim_only=False)
    assert resolve_fold({"fold": "minus-hurricane-michael-seed2", "folds_config": str(FOLDS)}).name == (
        "minus-hurricane-michael-seed2"
    )
    assert_head_may_score(michael, "hurricane-michael")
    assert_head_may_score(michael_seed2, "hurricane-michael")
    assert [fold.name for fold in route_oof("palu-tsunami", folds, claim_only=False)] == ["minus-palu-tsunami"]
    with pytest.raises(ConfigError, match="must not produce"):
        assert_head_may_score(palu, "hurricane-michael")
    with pytest.raises(ConfigError, match="must not produce"):
        assert_head_may_score(michael, "palu-tsunami")
    with pytest.raises(ConfigError, match="must not produce"):
        assert_head_may_score(michael_seed2, "palu-tsunami")
    with pytest.raises(ConfigError, match="No enabled fold"):
        select_oof_head("hurricane-harvey", folds, claim_only=True)
    with pytest.raises(ConfigError, match="No enabled fold"):
        select_oof_head("santa-rosa-wildfire", folds, claim_only=True)
    santa = select_oof_head("santa-rosa-wildfire", folds, claim_only=False)
    assert santa.name == "minus-santa-rosa"
    assert santa.claim_evidence is False
    socal = select_oof_head("socal-fire", folds, claim_only=False)
    assert socal.name == "minus-socal"


def test_appended_fold_routes_without_a_router_change(tmp_path):
    extra = FoldSpec(
        name="minus-hurricane-florence",
        holdout_events=("hurricane-florence",),
        role="added-later",
        enabled=True,
        claim_evidence=True,
        seed=1,
        notes="A third reference fold added only in this test.",
    )
    folds = load_folds(FOLDS) + (extra,)
    assert select_oof_head("hurricane-florence", folds, claim_only=True).name == "minus-hurricane-florence"
    assert select_oof_head("hurricane-michael", folds, claim_only=True).name == "minus-hurricane-michael"
    with pytest.raises(ConfigError, match="must not produce"):
        assert_head_may_score(extra, "hurricane-michael")
    text = FOLDS.read_text(encoding="utf-8") + (
        "\n"
        "  - name: minus-hurricane-florence\n"
        "    holdout_events: [hurricane-florence]\n"
        "    role: secondary\n"
        "    enabled: true\n"
        "    claim_evidence: true\n"
        "    seed: 1\n"
        "    notes: Appended in a test. Routing code is unchanged.\n"
    )
    path = tmp_path / "folds.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = load_folds(path)
    assert select_oof_head("hurricane-florence", loaded, claim_only=True).name == "minus-hurricane-florence"
    both = (
        extra,
        FoldSpec(
            name="also-florence",
            holdout_events=("hurricane-florence",),
            role="secondary",
            enabled=True,
            claim_evidence=True,
            seed=1,
        ),
    )
    with pytest.raises(ConfigError, match="Refusing to average"):
        select_oof_head("hurricane-florence", both, claim_only=True)


def test_plan_oof_skips_events_no_head_held_out(tmp_path):
    root = tmp_path / "xbd"
    groups = _populate(root)
    tiles = [record for records in index_official_tiles(root, _data_config(root)["split_dirs"]).values() for record in records]
    groups_by_head = plan_oof(tiles, load_folds(FOLDS), claim_only=True)
    assert "minus-hurricane-michael-seed2" not in groups_by_head
    michael_ids = set(groups_by_head["minus-hurricane-michael"])
    palu_ids = set(groups_by_head["minus-palu-tsunami"])
    assert michael_ids == groups["michael"]
    assert palu_ids == groups["palu"]
    assert michael_ids.isdisjoint(palu_ids)
    assigned = michael_ids | palu_ids
    assert groups["harvey"].isdisjoint(assigned)
    assert groups["santa"].isdisjoint(assigned)
    assert groups["socal"].isdisjoint(assigned)
    assert "minus-santa-rosa" not in groups_by_head
    with pytest.raises(ConfigError, match="Refusing to average"):
        plan_oof(tiles, load_folds(FOLDS), claim_only=False)
    with_checks = plan_oof(
        tiles,
        tuple(fold for fold in load_folds(FOLDS) if fold.claim_evidence or fold.role == "pipeline-check"),
        claim_only=False,
    )
    assert set(with_checks["minus-santa-rosa"]) == groups["santa"]
    assert set(with_checks["minus-socal"]) == groups["socal"]


def test_disabled_fold_is_skipped_and_rejected(tmp_path):
    path = tmp_path / "folds.yaml"
    path.write_text(
        "folds:\n"
        "  - name: minus-example\n"
        "    holdout_events: [example-event]\n"
        "    role: secondary\n"
        "    enabled: false\n"
        "    claim_evidence: true\n"
        "    seed: 1\n",
        encoding="utf-8",
    )
    folds = load_folds(path)
    assert route_oof("example-event", folds, claim_only=True) == []
    with pytest.raises(ConfigError, match="disabled"):
        resolve_fold({"fold": "minus-example", "folds_config": str(path)})


def test_oof_route_cli(capsys):
    code = main(["oof-route", "--disaster", "palu-tsunami", "--folds", str(FOLDS)])
    assert code == 0
    output = capsys.readouterr().out
    assert "fold=minus-palu-tsunami" in output
    assert "seed=1" in output
    assert "averaged=false" in output
    assert "minus-hurricane-michael" not in output
    michael_route = main(["oof-route", "--disaster", "hurricane-michael", "--folds", str(FOLDS)])
    assert michael_route == 0
    michael_text = capsys.readouterr().out
    assert "fold=minus-hurricane-michael\n" in michael_text
    assert "seed=1" in michael_text
    assert "minus-hurricane-michael-seed2" not in michael_text
    assert "averaged=false" in michael_text
    seed2_train = main(
        ["train-xbd", "--fold", "minus-hurricane-michael-seed2", "--folds", str(FOLDS), "--dry-run"]
    )
    assert seed2_train == 0
    refused = main(["oof-route", "--disaster", "hurricane-harvey", "--folds", str(FOLDS)])
    assert refused == 2
    pipeline = main(["oof-route", "--disaster", "santa-rosa-wildfire", "--folds", str(FOLDS)])
    assert pipeline == 2
    allowed = main(
        ["oof-route", "--disaster", "santa-rosa-wildfire", "--folds", str(FOLDS), "--no-claim-only"]
    )
    assert allowed == 0
    assert "fold=minus-santa-rosa" in capsys.readouterr().out


def test_unknown_fold_and_seed_override_do_not_train(tmp_path, capsys):
    unknown = main(["train-xbd", "--fold", "minus-not-a-fold", "--folds", str(FOLDS), "--dry-run"])
    assert unknown == 2
    assert "Unknown fold" in capsys.readouterr().out
    mismatched = main(
        [
            "train-xbd",
            "--fold",
            "minus-hurricane-michael",
            "--folds",
            str(FOLDS),
            "--set",
            "seed=2",
            "--dry-run",
        ]
    )
    assert mismatched == 2
    local_folds = tmp_path / "folds.yaml"
    local_folds.write_text(
        "seed: 1\n"
        "train:\n"
        "  batch_size: 32\n"
        "  amp_dtype: bfloat16\n"
        '  torch_version: "2.15.0"\n'
        "hosts:\n"
        "  local:\n"
        "    gpu: none\n"
        f"    data_root: {tmp_path / 'missing-xbd'}\n"
        "    holdout_dirname: hold\n"
        f"    weights_path: {tmp_path / 'missing.pth'}\n"
        f"    output_root: {tmp_path / 'out'}\n"
        "folds:\n"
        "  - name: minus-palu-tsunami\n"
        "    holdout_events: [palu-tsunami]\n"
        "    role: secondary\n"
        "    enabled: true\n"
        "    claim_evidence: true\n"
        "    seed: 1\n"
        "    host: local\n",
        encoding="utf-8",
    )
    missing_root = main(
        [
            "train-xbd",
            "--config",
            str(ROOT / "configs" / "xbd_dinov2_vitb.yaml"),
            "--fold",
            "minus-palu-tsunami",
            "--folds",
            str(local_folds),
            "--set",
            "data.root=null",
        ]
    )
    assert missing_root == 2
    assert not (tmp_path / "out" / "minus-palu-tsunami" / "checkpoint_last.pt").exists()


def test_checkpoint_fold_must_match_the_request():
    michael = next(fold for fold in load_folds(FOLDS) if fold.name == "minus-hurricane-michael")
    palu = next(fold for fold in load_folds(FOLDS) if fold.name == "minus-palu-tsunami")
    assert_checkpoint_fold({"config": {}}, None)
    assert_checkpoint_fold({"config": {"fold": michael.name}}, michael)
    with pytest.raises(DataError, match="does not match"):
        assert_checkpoint_fold({"config": {"fold": palu.name}}, michael)
    with pytest.raises(DataError, match="Pass --fold"):
        assert_checkpoint_fold({"config": {"fold": michael.name}}, None)


def test_host_paths_and_shared_pins_apply_without_gpu_changes(tmp_path):
    folds = {fold.name: fold for fold in load_folds(FOLDS)}
    michael_cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "xbd_dinov2_vitb.yaml"))
    michael_cfg["fold"] = "minus-hurricane-michael"
    michael_cfg["folds_config"] = str(FOLDS)
    apply_fold_data_paths(michael_cfg, folds["minus-hurricane-michael"])
    palu_cfg = copy.deepcopy(load_yaml(ROOT / "configs" / "xbd_dinov2_vitb.yaml"))
    palu_cfg["fold"] = "minus-palu-tsunami"
    palu_cfg["folds_config"] = str(FOLDS)
    apply_fold_data_paths(palu_cfg, folds["minus-palu-tsunami"])
    assert michael_cfg["data"]["root"] == "/data/repo/datasets/xbd"
    assert michael_cfg["data"]["split_dirs"]["holdout"] == "hold"
    assert michael_cfg["data"]["split_dirs"]["train"] == "train"
    assert michael_cfg["backbone"]["weights_path"].startswith("/data/repo/weights/")
    assert michael_cfg["backbone"]["pretrained"] is False
    assert michael_cfg["output_dir"] == "/data/repo/outputs/minus-hurricane-michael"
    assert palu_cfg["data"]["root"] == "/mnt/data_hdd/datasets/xbd"
    assert palu_cfg["backbone"]["weights_path"] == (
        "/mnt/data_hdd/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth"
    )
    assert palu_cfg["output_dir"] == "/mnt/data_hdd/outputs/minus-palu-tsunami"
    assert palu_cfg["backbone"]["weights_sha256"] == michael_cfg["backbone"]["weights_sha256"]
    assert palu_cfg["batch_size"] == michael_cfg["batch_size"] == 32
    assert palu_cfg["amp_dtype"] == michael_cfg["amp_dtype"] == "bfloat16"
    assert palu_cfg["torch_version"] == michael_cfg["torch_version"] == "2.15.0"
    # A CLI output_dir does not move the Palu checkpoints onto the root disk.
    palu_cfg["output_dir"] = "/tmp/on-the-root-disk"
    apply_fold_data_paths(palu_cfg, folds["minus-palu-tsunami"])
    assert palu_cfg["output_dir"] == "/mnt/data_hdd/outputs/minus-palu-tsunami"
    bad = tmp_path / "folds.yaml"
    bad.write_text(
        "train:\n"
        "  batch_size: 32\n"
        "  amp_dtype: bfloat16\n"
        '  torch_version: "2.15.0"\n'
        "folds:\n"
        "  - name: minus-other\n"
        "    holdout_events: [hurricane-florence]\n"
        "    role: secondary\n"
        "    enabled: true\n"
        "    claim_evidence: true\n"
        "    seed: 1\n"
        "    batch_size: 8\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="does not match the shared pin"):
        load_folds(bad)


def test_torch_pin_rejects_a_different_release():
    logger = _logger()
    pinned = torch_release(__import__("torch").__version__)
    assert_pinned_torch({"torch_version": pinned, "batch_size": 32, "amp_dtype": "bfloat16"}, logger)
    with pytest.raises(ConfigError, match="does not match the pinned"):
        assert_pinned_torch({"torch_version": "0.0.1", "batch_size": 32, "amp_dtype": "bfloat16"}, logger)
    assert torch_release("2.15.0+cu124") == "2.15.0"


def test_dry_run_names_the_fold_and_skips_the_manifest(capsys):
    code = main(
        ["train-xbd", "--fold", "minus-hurricane-michael", "--folds", str(FOLDS), "--dry-run"]
    )
    assert code == 0
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "minus-hurricane-michael" in text
    assert "manifest not written" in text

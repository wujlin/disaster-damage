"""Environment metadata, same-env fold scoring, and cross-env agreement."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mvbridge.cli import main
from mvbridge.engine.runtime import (
    AGREEMENT_NO_EFFECT,
    assert_same_training_env,
    collect_run_env,
    prediction_agreement,
    record_run_env,
    require_same_fold_env,
)
from mvbridge.errors import ConfigError, DataError

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "configs" / "folds" / "event_folds.yaml"


def _logger() -> logging.Logger:
    logger = logging.getLogger("mvbridge-env-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _probe(**overrides) -> dict:
    base = {
        "python": "3.12.3",
        "hostname": "wsb",
        "torch": "2.6.0+cu124",
        "cuda": "12.4",
        "gpus": ["RTX 4090"],
        "nvidia_driver": "560.35.03",
    }
    base.update(overrides)
    return base


def test_run_env_writer_uses_the_probe_and_logs(tmp_path):
    messages = []

    class _Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = _logger()
    logger.addHandler(_Handler())
    env = record_run_env(tmp_path, logger, probe=_probe(gpus=[]))
    # An empty GPU list is the mock for a machine with no CUDA device.
    written = json.loads((tmp_path / "run_env.json").read_text(encoding="utf-8"))
    assert written["hostname"] == "wsb"
    assert written["torch"] == "2.6.0+cu124"
    assert written["cuda"] == "12.4"
    assert written["nvidia_driver"] == "560.35.03"
    assert written["gpus"] == []
    assert written["python"] == "3.12.3"
    assert env == written
    assert any("run_env" in message and "wsb" in message for message in messages)
    mocked = collect_run_env(_probe(hostname="wsa", torch="2.11.0+cu128", cuda="12.8", gpus=["RTX 5880"]))
    assert mocked["hostname"] == "wsa"
    assert mocked["gpus"] == ["RTX 5880"]


def test_fold_score_refuses_a_different_environment(tmp_path):
    logger = _logger()
    training = collect_run_env(_probe())
    record_run_env(tmp_path, logger, probe=_probe())
    config = {
        "fold": "minus-hurricane-michael",
        "output_dir": str(tmp_path),
        "training_output_dir": str(tmp_path),
        "cross_env_check": False,
    }
    require_same_fold_env(config, logger, training)
    other = collect_run_env(_probe(hostname="wsa", torch="2.11.0+cu128", cuda="12.8", gpus=["RTX 5880"]))
    with pytest.raises(ConfigError, match="same host and environment"):
        require_same_fold_env(config, logger, other)
    config["cross_env_check"] = True
    require_same_fold_env(config, logger, other)
    with pytest.raises(DataError, match="does not exist"):
        require_same_fold_env(
            {"fold": "minus-hurricane-michael", "training_output_dir": str(tmp_path / "absent")},
            logger,
            training,
        )


def test_same_training_env_helper_blocks_without_the_flag():
    training = collect_run_env(_probe())
    current = collect_run_env(_probe(nvidia_driver="555.42.02"))
    assert_same_training_env(training, training, cross_env=False)
    with pytest.raises(ConfigError, match="--cross-env-check"):
        assert_same_training_env(training, current, cross_env=False)
    assert_same_training_env(training, current, cross_env=True)


def test_cross_env_check_requires_a_separate_output_dir():
    code = main(
        [
            "eval-xbd",
            "--fold",
            "minus-hurricane-michael",
            "--folds",
            str(FOLDS),
            "--split",
            "fold-eval",
            "--checkpoint",
            "missing.pt",
            "--cross-env-check",
            "--set",
            "output_dir=/data/repo/outputs/minus-hurricane-michael",
        ]
    )
    assert code == 2


def test_no_effect_requires_every_event(tmp_path):
    big = [
        {"image_id": f"big_{index}", "building_uid": "b", "disaster": "hurricane-michael", "prediction": 0}
        for index in range(200)
    ]
    small_left = [
        {"image_id": "small_0", "building_uid": "b", "disaster": "palu-tsunami", "prediction": 1},
        {"image_id": "small_1", "building_uid": "b", "disaster": "palu-tsunami", "prediction": 1},
    ]
    small_right = [
        {"image_id": "small_0", "building_uid": "b", "disaster": "palu-tsunami", "prediction": 1},
        {"image_id": "small_1", "building_uid": "b", "disaster": "palu-tsunami", "prediction": 0},
    ]
    report = prediction_agreement(big + small_left, big + small_right)
    assert report["agreement"] == pytest.approx(201 / 202)
    assert report["agreement"] >= AGREEMENT_NO_EFFECT
    assert report["per_event"]["hurricane-michael"]["agreement"] == pytest.approx(1.0)
    assert report["per_event"]["palu-tsunami"]["agreement"] == pytest.approx(0.5)
    assert report["failed_events"] == ["palu-tsunami"]
    assert report["no_effect"] is False


def test_prediction_keys_must_be_present_and_unique():
    complete = {"image_id": "img", "building_uid": "b1", "disaster": "hurricane-michael", "prediction": 0}
    with pytest.raises(DataError, match="image_id"):
        prediction_agreement([{"building_uid": "b1", "prediction": 0}], [complete])
    with pytest.raises(DataError, match="building_uid"):
        prediction_agreement([{"image_id": "img", "prediction": 0}], [complete])
    with pytest.raises(DataError, match="Duplicate"):
        prediction_agreement([complete, dict(complete)], [complete])


def test_prediction_agreement_is_stratified_and_uses_99_percent(tmp_path, capsys):
    left = {
        "records": [
            {"image_id": "hurricane-michael_1", "building_uid": "a", "disaster": "hurricane-michael", "prediction": 0},
            {"image_id": "hurricane-michael_1", "building_uid": "b", "disaster": "hurricane-michael", "prediction": 1},
            {"image_id": "palu-tsunami_1", "building_uid": "c", "disaster": "palu-tsunami", "prediction": 3},
        ]
    }
    right = {
        "records": [
            {"image_id": "hurricane-michael_1", "building_uid": "a", "disaster": "hurricane-michael", "prediction": 0},
            {"image_id": "hurricane-michael_1", "building_uid": "b", "disaster": "hurricane-michael", "prediction": 2},
            {"image_id": "palu-tsunami_1", "building_uid": "c", "disaster": "palu-tsunami", "prediction": 3},
        ]
    }
    report = prediction_agreement(left["records"], right["records"])
    assert report["compared"] == 3
    assert report["matched"] == 2
    assert report["agreement"] == pytest.approx(2 / 3)
    assert report["threshold"] == AGREEMENT_NO_EFFECT == 0.99
    assert report["no_effect"] is False
    assert "Palu-fold" in report["annotation"]
    assert report["per_event"]["hurricane-michael"]["agreement"] == pytest.approx(0.5)
    assert report["per_event"]["palu-tsunami"]["agreement"] == pytest.approx(1.0)
    identical = prediction_agreement(left["records"], left["records"])
    assert identical["no_effect"] is True
    assert identical["annotation"] is None
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")
    code = main(["cross-env-agree", "--left", str(left_path), "--right", str(right_path)])
    assert code == 0
    text = capsys.readouterr().out
    assert "no_effect=false" in text
    assert "event=hurricane-michael" in text
    assert "event=palu-tsunami" in text

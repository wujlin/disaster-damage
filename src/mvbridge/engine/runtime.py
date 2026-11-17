"""Run environment metadata and the same-env rule for one fold.

Batch size and AMP dtype stay pinned. Torch is recorded, not required to match
across hosts. Scoring a fold must match that fold's training environment
unless ``--cross-env-check`` is set.
"""

from __future__ import annotations

import contextlib
import json
import platform
import socket
import subprocess
from pathlib import Path

import torch

from mvbridge.errors import ConfigError, DataError

_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

RUN_ENV_NAME = "run_env.json"
AGREEMENT_NO_EFFECT = 0.99
FINGERPRINT_FIELDS = ("hostname", "torch", "cuda", "nvidia_driver", "gpus")


def collect_run_env(probe: dict | None = None) -> dict:
    """Environment fingerprint for a train or score run. ``probe`` overrides the machine."""
    source = probe_machine() if probe is None else probe
    cuda = source.get("cuda")
    return {
        "python": source.get("python"),
        "hostname": source.get("hostname"),
        "torch": source.get("torch"),
        "cuda": None if cuda in {None, ""} else str(cuda),
        "gpus": [str(name) for name in (source.get("gpus") or [])],
        "nvidia_driver": source.get("nvidia_driver") or None,
    }


def probe_machine() -> dict:
    return {
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": _gpu_names(),
        "nvidia_driver": _nvidia_driver_version(),
    }


def write_run_env(output_dir: str | Path, env: dict, logger) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_ENV_NAME
    path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "run_env python=%s hostname=%s torch=%s cuda=%s nvidia_driver=%s gpus=%s wrote %s",
        env.get("python"),
        env.get("hostname"),
        env.get("torch"),
        env.get("cuda"),
        env.get("nvidia_driver"),
        env.get("gpus"),
        path,
    )
    return path


def record_run_env(output_dir: str | Path, logger, probe: dict | None = None) -> dict:
    env = collect_run_env(probe)
    write_run_env(output_dir, env, logger)
    return env


def env_fingerprint(env: dict) -> tuple:
    return tuple(env.get(field) if field != "gpus" else tuple(env.get("gpus") or []) for field in FINGERPRINT_FIELDS)


def assert_same_training_env(training: dict, current: dict, *, cross_env: bool) -> None:
    """Refuse a fold score whose host, torch, CUDA, driver, or GPU differs from training."""
    if env_fingerprint(training) == env_fingerprint(current):
        return
    message = (
        "Fold scoring environment does not match the training run_env.json. "
        f"training host={training.get('hostname')} torch={training.get('torch')} "
        f"cuda={training.get('cuda')} driver={training.get('nvidia_driver')} gpus={training.get('gpus')}; "
        f"current host={current.get('hostname')} torch={current.get('torch')} "
        f"cuda={current.get('cuda')} driver={current.get('nvidia_driver')} gpus={current.get('gpus')}. "
        "Training, scoring, and conflict-head evaluation of one fold must use the same host and environment. "
        "Pass --cross-env-check with a separate output directory only to measure agreement."
    )
    if not cross_env:
        raise ConfigError(message)


def load_run_env(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"Run environment file does not exist: {file_path}")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataError(f"Run environment file is not an object: {file_path}")
    return payload


def require_same_fold_env(config: dict, logger, current: dict) -> dict | None:
    """Compare ``current`` with the fold training environment. No-fold runs skip this."""
    if not config.get("fold"):
        return None
    directory = config.get("training_output_dir") or config.get("output_dir")
    if not directory:
        raise ConfigError("A fold score needs the training output directory to find run_env.json.")
    path = Path(directory) / RUN_ENV_NAME
    training = load_run_env(path)
    assert_same_training_env(training, current, cross_env=bool(config.get("cross_env_check")))
    if env_fingerprint(training) != env_fingerprint(current):
        logger.warning(
            "cross-env-check accepted a different environment. training=%s current=%s",
            {field: training.get(field) for field in FINGERPRINT_FIELDS},
            {field: current.get(field) for field in FINGERPRINT_FIELDS},
        )
    return training


def prediction_agreement(left: list[dict], right: list[dict]) -> dict:
    """Agreement of satellite grades, overall and stratified by event."""
    left_map = {_prediction_key(row): row for row in left}
    right_map = {_prediction_key(row): row for row in right}
    shared = sorted(set(left_map) & set(right_map))
    if not shared:
        raise DataError("The two prediction files have no image and building in common.")
    matched = 0
    per_event: dict[str, dict[str, int]] = {}
    for key in shared:
        event = str(left_map[key].get("disaster") or right_map[key].get("disaster") or "")
        slot = per_event.setdefault(event, {"compared": 0, "matched": 0})
        slot["compared"] += 1
        agree = int(left_map[key]["prediction"]) == int(right_map[key]["prediction"])
        slot["matched"] += int(agree)
        matched += int(agree)
    compared = len(shared)
    rate = matched / compared
    per_event_out = {}
    for event, slot in sorted(per_event.items()):
        per_event_out[event] = {
            "compared": slot["compared"],
            "matched": slot["matched"],
            "agreement": slot["matched"] / slot["compared"],
        }
    no_effect = rate >= AGREEMENT_NO_EFFECT
    return {
        "compared": compared,
        "matched": matched,
        "agreement": rate,
        "threshold": AGREEMENT_NO_EFFECT,
        "no_effect": no_effect,
        "per_event": per_event_out,
        "annotation": (
            None
            if no_effect
            else "Annotate Palu-fold results with the torch, CUDA, driver, and GPU difference."
        ),
    }


def _prediction_key(row: dict) -> tuple[str, str]:
    return (str(row.get("image_id") or ""), str(row.get("building_uid") or ""))


def _gpu_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def autocast_for(config: dict, device: torch.device):
    """Return the pinned AMP context, or a no-op when no fold dtype is set."""
    name = config.get("amp_dtype")
    if not name:
        return contextlib.nullcontext()
    dtype = _AMP_DTYPES.get(str(name).strip().lower())
    if dtype is None:
        raise ConfigError(f"amp_dtype must be bfloat16 or float16, got {name!r}")
    return torch.autocast(device_type=device.type, dtype=dtype)

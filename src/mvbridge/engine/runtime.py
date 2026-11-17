"""Pinned fold runtime: one batch size, one AMP dtype, one torch release.

The GPU name is logged. It does not change those pins.
"""

from __future__ import annotations

import contextlib

import torch

from mvbridge.errors import ConfigError

_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def torch_release(version: str) -> str:
    """Drop a local build tag such as +cu124 or +cpu."""
    return str(version).split("+", 1)[0].strip()


def assert_pinned_torch(config: dict, logger) -> None:
    """Log torch, CUDA, and the GPU. Stop if the torch release is not the pin.

    No-op when the config has no torch_version, which is the no-fold path.
    """
    pinned = str(config.get("torch_version") or "").strip()
    if not pinned:
        return
    actual = torch.__version__
    cuda = torch.version.cuda
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "none"
    logger.info(
        "runtime torch=%s cuda=%s gpu=%s pinned_torch=%s batch_size=%s amp_dtype=%s host=%s host_gpu=%s",
        actual,
        cuda,
        gpu,
        pinned,
        config.get("batch_size"),
        config.get("amp_dtype"),
        config.get("host"),
        config.get("host_gpu"),
    )
    if torch_release(actual) != torch_release(pinned):
        raise ConfigError(
            f"torch {actual} does not match the pinned fold version {pinned}. "
            f"CUDA={cuda} GPU={gpu}. Batch size and AMP dtype are not changed to fit this machine."
        )
    host_gpu = str(config.get("host_gpu") or "").strip()
    if gpu == "none":
        logger.warning(
            "CUDA GPU is none. Fold host GPU is %s. Batch size %s and AMP dtype %s stay pinned.",
            host_gpu or "unset",
            config.get("batch_size"),
            config.get("amp_dtype"),
        )
        return
    if host_gpu and host_gpu.lower() not in gpu.lower() and gpu.lower() not in host_gpu.lower():
        logger.warning(
            "GPU %s does not match the fold host GPU %s. "
            "Batch size %s and AMP dtype %s stay pinned and are not changed for this GPU.",
            gpu,
            host_gpu,
            config.get("batch_size"),
            config.get("amp_dtype"),
        )


def autocast_for(config: dict, device: torch.device):
    """Return the pinned AMP context, or a no-op when no fold dtype is set."""
    name = config.get("amp_dtype")
    if not name:
        return contextlib.nullcontext()
    dtype = _AMP_DTYPES.get(str(name).strip().lower())
    if dtype is None:
        raise ConfigError(f"amp_dtype must be bfloat16 or float16, got {name!r}")
    return torch.autocast(device_type=device.type, dtype=dtype)

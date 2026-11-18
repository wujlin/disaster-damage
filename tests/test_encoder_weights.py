"""Local DINOv2 weight checks. Fixtures are tiny tensors, not the real checkpoint."""

from __future__ import annotations

import hashlib
import inspect
import logging

import pytest
import torch
from torch import nn

from mvbridge.errors import ConfigError, DataError
from mvbridge.models.pathways import TimmDinoEncoder, load_local_encoder_weights, prepare_local_backbone


def _logger() -> logging.Logger:
    logger = logging.getLogger("mvbridge-weights-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.embed_dim = 2

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return images.new_zeros((images.shape[0], 2, self.embed_dim))


def test_weights_sha256_is_logged_and_must_match(tmp_path):
    payload = b"not-the-real-dinov2-file"
    path = tmp_path / "dinov2_vitb14_timm_lvd142m.pth"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    messages = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logger = _logger()
    logger.addHandler(_ListHandler())
    config = {
        "backbone": {"weights_path": str(path), "weights_sha256": digest, "pretrained": True},
        "weight_provenance": {
            "loaded": {
                "filename": "dinov2_vitb14_timm_lvd142m.pth",
                "bytes": 346385151,
                "sha256": digest,
            },
            "original": {
                "filename": "dinov2_vitb14_pretrain.pth",
                "bytes": 346378731,
                "sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
            },
            "timm": "timm/vit_base_patch14_dinov2.lvd142m",
            "hf_revision": "4685c99dabffe5affac90bd99dbffd25801ae58d",
            "upstream": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        },
    }
    assert prepare_local_backbone(config, logger) == digest
    assert config["backbone"]["pretrained"] is False
    text = "\n".join(messages)
    assert "6cce660" not in text or digest.startswith("6cce660")
    assert "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73" in text
    assert "timm/vit_base_patch14_dinov2.lvd142m" in text
    assert "4685c99dabffe5affac90bd99dbffd25801ae58d" in text
    assert digest in text
    config["backbone"]["weights_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="does not match"):
        prepare_local_backbone(config, logger)
    missing = tmp_path / "absent.pth"
    with pytest.raises(DataError, match="not replaced by a pretrained download"):
        prepare_local_backbone({"backbone": {"weights_path": str(missing), "weights_sha256": digest}}, logger)


def test_local_state_dict_must_match_exactly(tmp_path):
    module = _Tiny()
    good = tmp_path / "good.pt"
    torch.save(module.state_dict(), good)
    other = _Tiny()
    with torch.no_grad():
        other.fc.weight.zero_()
    load_local_encoder_weights(other, good)
    assert torch.allclose(other.fc.weight, module.fc.weight)
    extra = dict(module.state_dict())
    extra["unexpected.weight"] = torch.zeros(1)
    bad = tmp_path / "extra.pt"
    torch.save(extra, bad)
    with pytest.raises(ConfigError, match="unexpected_keys"):
        load_local_encoder_weights(_Tiny(), bad)
    missing = {key: value for key, value in module.state_dict().items() if key != "fc.weight"}
    gone = tmp_path / "missing.pt"
    torch.save(missing, gone)
    with pytest.raises(ConfigError, match="missing_keys"):
        load_local_encoder_weights(_Tiny(), gone)


def test_size_mismatch_is_rejected_without_interpolation(tmp_path):
    module = _Tiny()
    blob = {key: value.detach().clone() for key, value in module.state_dict().items()}
    blob["fc.weight"] = torch.zeros(4, 3)
    path = tmp_path / "shape.pt"
    torch.save(blob, path)
    target = _Tiny()
    before = target.fc.weight.detach().clone()
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_local_encoder_weights(target, path)
    assert torch.equal(target.fc.weight, before)
    source = inspect.getsource(load_local_encoder_weights)
    assert "interpolat" not in source.lower()


def test_weights_path_never_asks_timm_to_download(tmp_path, monkeypatch):
    created = []

    def fake_create(name, pretrained=False, num_classes=0, img_size=224):
        created.append(pretrained)
        return _Tiny()

    import timm

    monkeypatch.setattr(timm, "create_model", fake_create)
    good = tmp_path / "encoder.pt"
    torch.save(_Tiny().state_dict(), good)
    encoder = TimmDinoEncoder(
        "vit_base_patch14_dinov2",
        224,
        pretrained=True,
        weights_path=str(good),
    )
    assert encoder.out_dim == 2
    assert created == [False]
    bad = tmp_path / "bad.pt"
    torch.save({"nope.weight": torch.zeros(1)}, bad)
    with pytest.raises(ConfigError, match="missing_keys"):
        TimmDinoEncoder("vit_base_patch14_dinov2", 224, pretrained=True, weights_path=str(bad))
    assert created == [False, False]
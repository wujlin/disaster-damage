"""Building-damage examples from the official xBD / xView2 splits.

The dataset reads one of ``train``, ``test``, or ``holdout``. It does not
reshuffle image pairs across those directories.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mvbridge.data.exclusions import read_id_list
from mvbridge.data.images import crop_rgb_tensor, square_crop_box
from mvbridge.data.xbd_labels import (
    CLASS_TO_INDEX,
    DAMAGE_CLASSES,
    OFFICIAL_SPLIT_NAMES,
    SKIP_SUBTYPES,
    image_id_from_filename,
    iter_label_paths,
    load_label_file,
)
from mvbridge.errors import ConfigError, DataError


@dataclass
class BuildingExample:
    image_id: str
    building_uid: str
    label: int
    pre_path: str
    post_path: str
    pre_box: tuple[int, int, int, int]
    post_box: tuple[int, int, int, int]
    split: str


class XBDBuildingDataset(Dataset):
    def __init__(self, config: dict, *, split: str):
        if split not in OFFICIAL_SPLIT_NAMES:
            raise ConfigError(
                f"Split {split!r} is not an official xBD/xView2 split. "
                "Use train, test, or holdout. This scaffold does not build a custom split."
            )
        root = config.get("root")
        if not root:
            raise ConfigError("xbd.root is required")
        root_path = Path(root)
        if not root_path.is_dir():
            raise DataError(f"xBD root does not exist: {root_path}")
        split_dirs = config.get("split_dirs") or {}
        if split not in split_dirs:
            raise ConfigError(f"xbd.split_dirs.{split} is required")
        split_path = root_path / str(split_dirs[split])
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        if not images_dir.is_dir():
            raise DataError(f"Official xBD {split} images directory does not exist: {images_dir}")
        if not labels_dir.is_dir():
            raise DataError(
                f"Official xBD {split} labels directory does not exist: {labels_dir}. "
                "The challenge release sometimes shipped images without JSON labels. "
                "Point this split at the labeled train, test, or holdout directory. "
                "Unlabeled images are not scored, and no custom split is created to replace them."
            )
        self.split = split
        self.image_size = int(config.get("image_size", 224))
        self.crop_padding = int(config.get("crop_padding", 16))
        self.images_dir = images_dir.resolve()
        excluded = read_id_list(config.get("exclude_ids_file"))
        self.examples, self.stats = _index_buildings(
            images_dir=images_dir,
            labels_dir=labels_dir,
            split=split,
            padding=self.crop_padding,
            excluded=excluded,
        )
        if not self.examples:
            raise DataError(
                f"Official xBD {split} split contains no classified buildings in {labels_dir}. "
                f"Index stats: {self.stats}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        pre = crop_rgb_tensor(example.pre_path, example.pre_box, self.image_size)
        post = crop_rgb_tensor(example.post_path, example.post_box, self.image_size)
        return {
            "pre": pre,
            "post": post,
            "label": torch.tensor(example.label, dtype=torch.long),
            "image_id": example.image_id,
            "building_uid": example.building_uid,
            "split": example.split,
        }

    @property
    def class_counts(self) -> dict[str, int]:
        counts = Counter(example.label for example in self.examples)
        return {DAMAGE_CLASSES[index]: int(counts.get(index, 0)) for index in range(len(DAMAGE_CLASSES))}

    @property
    def image_ids(self) -> set[str]:
        return {example.image_id for example in self.examples}


def _index_buildings(
    *,
    images_dir: Path,
    labels_dir: Path,
    split: str,
    padding: int,
    excluded: set[str],
) -> tuple[list[BuildingExample], dict]:
    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        image_id, when = image_id_from_filename(path.name)
        if when not in {"pre", "post"}:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(images_dir.resolve()):
            raise DataError(f"Refusing image outside the official {split} images directory: {resolved}")
        pairs.setdefault(image_id, {})[when] = path
    labels: dict[str, dict[str, Path]] = {}
    for path in iter_label_paths(labels_dir):
        image_id, when = image_id_from_filename(path.name)
        labels.setdefault(image_id, {})[when or "post"] = path
    examples = []
    stats = {
        "image_pairs": 0,
        "incomplete_pairs": 0,
        "missing_labels": 0,
        "excluded_images": 0,
        "skipped_unclassified": 0,
        "skipped_without_box": 0,
    }
    for image_id, images in sorted(pairs.items()):
        if "pre" not in images or "post" not in images:
            stats["incomplete_pairs"] += 1
            continue
        stats["image_pairs"] += 1
        if image_id in excluded:
            stats["excluded_images"] += 1
            continue
        label_files = labels.get(image_id)
        if not label_files or "post" not in label_files:
            stats["missing_labels"] += 1
            continue
        post_label = load_label_file(label_files["post"])
        pre_label = load_label_file(label_files["pre"]) if "pre" in label_files else None
        pre_boxes = {} if pre_label is None else {building.uid: building.xy_box for building in pre_label.buildings}
        from PIL import Image

        with Image.open(images["pre"]) as pre_image:
            pre_size = pre_image.size
        with Image.open(images["post"]) as post_image:
            post_size = post_image.size
        for building in post_label.buildings:
            if building.subtype in SKIP_SUBTYPES:
                stats["skipped_unclassified"] += 1
                continue
            if building.subtype not in CLASS_TO_INDEX:
                stats["skipped_unclassified"] += 1
                continue
            post_box_raw = building.xy_box
            pre_box_raw = pre_boxes.get(building.uid) or post_box_raw
            if post_box_raw is None or pre_box_raw is None:
                stats["skipped_without_box"] += 1
                continue
            examples.append(
                BuildingExample(
                    image_id=image_id,
                    building_uid=building.uid,
                    label=CLASS_TO_INDEX[building.subtype],
                    pre_path=str(images["pre"]),
                    post_path=str(images["post"]),
                    pre_box=square_crop_box(pre_box_raw, padding, pre_size[0], pre_size[1]),
                    post_box=square_crop_box(post_box_raw, padding, post_size[0], post_size[1]),
                    split=split,
                )
            )
    return examples, stats

"""Score adapter over synthetic or precomputed class probabilities.

The inference table is produced from probabilities alone. Evaluation labels
are joined later, one-to-one, on ``(image_id, building_uid)``. Removing or
shuffling those labels must not change the inference scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mvbridge.errors import ConfigError
from mvbridge.score_contract.protocol import CLASS_ALIASES, OFFICIAL_CLASS_ORDER

ALIGNMENT_KEY = ("image_id", "building_uid")
PROBABILITY_SUM_ATOL = 1e-6
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class ClassPrediction:
    """Argmax over the official four-class order, with documented tie handling."""

    class_name: str
    class_index: int
    tied: bool
    tied_indices: tuple[int, ...]


@dataclass(frozen=True)
class ProbabilityScores:
    probabilities: tuple[float, ...]
    entropy: float
    one_minus_max_p: float
    prediction: ClassPrediction


@dataclass(frozen=True)
class InferenceRow:
    image_id: str
    building_uid: str
    probabilities: tuple[float, ...]
    entropy: float
    one_minus_max_p: float
    y_hat_s: str
    y_hat_s_index: int
    tied: bool
    tied_indices: tuple[int, ...]
    disaster: str = ""
    fold: str = ""
    seed: int | None = None
    reporting_model_version: str = ""
    out_of_sample: bool | None = None
    extras: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[str, str]:
        return (self.image_id, self.building_uid)

    def scores_only(self) -> dict[str, Any]:
        """Values that must stay stable if evaluation labels move."""
        return {
            "image_id": self.image_id,
            "building_uid": self.building_uid,
            "probabilities": self.probabilities,
            "entropy": self.entropy,
            "one_minus_max_p": self.one_minus_max_p,
            "y_hat_s": self.y_hat_s,
            "y_hat_s_index": self.y_hat_s_index,
            "tied": self.tied,
            "tied_indices": self.tied_indices,
            "disaster": self.disaster,
            "fold": self.fold,
            "seed": self.seed,
            "reporting_model_version": self.reporting_model_version,
            "out_of_sample": self.out_of_sample,
        }


@dataclass(frozen=True)
class InferenceTable:
    rows: tuple[InferenceRow, ...]
    class_order: tuple[str, ...] = OFFICIAL_CLASS_ORDER

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(row.key for row in self.rows)

    def by_key(self) -> dict[tuple[str, str], InferenceRow]:
        return {row.key: row for row in self.rows}

    def score_payload(self) -> tuple[dict[str, Any], ...]:
        return tuple(row.scores_only() for row in self.rows)


def normalize_class_name(name: str | None) -> str | None:
    if name is None:
        return None
    text = str(name).strip().lower().replace("_", "-")
    if not text:
        return None
    return CLASS_ALIASES.get(text, text)


def class_index(name: str, class_order: Sequence[str] = OFFICIAL_CLASS_ORDER) -> int:
    normalized = normalize_class_name(name)
    if normalized is None:
        raise ConfigError("Class name is empty")
    try:
        return list(class_order).index(normalized)
    except ValueError as exc:
        raise ConfigError(
            f"Class {name!r} is not in the official four-class order {tuple(class_order)}."
        ) from exc


def score_probabilities(
    probabilities: Sequence[float],
    *,
    class_order: Sequence[str] = OFFICIAL_CLASS_ORDER,
) -> ProbabilityScores:
    values = _as_probability_vector(probabilities, class_count=len(class_order))
    prediction = predict_satellite_class(values, class_order=class_order)
    return ProbabilityScores(
        probabilities=tuple(float(value) for value in values),
        entropy=entropy(values),
        one_minus_max_p=one_minus_max_p(values),
        prediction=prediction,
    )


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats of a four-class probability vector.

    The base is natural log. Ranking is invariant to a change of base. A later
    protocol lock of bits or nats would only rescale the number.
    """
    values = _as_probability_vector(probabilities, class_count=len(probabilities))
    positive = values[values > 0]
    return float(-(positive * np.log(positive)).sum())


def one_minus_max_p(probabilities: Sequence[float]) -> float:
    values = _as_probability_vector(probabilities, class_count=len(probabilities))
    return float(1.0 - values.max())


def predict_satellite_class(
    probabilities: Sequence[float],
    *,
    class_order: Sequence[str] = OFFICIAL_CLASS_ORDER,
) -> ClassPrediction:
    """Return the official-order argmax.

    Ties: every class within ``TIE_ATOL`` of the maximum is a tie. The reported
    class is the tied class with the lowest official index
    (no-damage < minor-damage < major-damage < destroyed). The protocol text
    does not yet freeze a tie rule; this is the adapter's deterministic rule.
    """
    values = _as_probability_vector(probabilities, class_count=len(class_order))
    maximum = float(values.max())
    tied = tuple(int(index) for index, value in enumerate(values) if abs(float(value) - maximum) <= TIE_ATOL)
    if not tied:
        raise ConfigError("Internal error: no class reached the maximum probability")
    index = tied[0]
    return ClassPrediction(
        class_name=str(class_order[index]),
        class_index=index,
        tied=len(tied) > 1,
        tied_indices=tied,
    )


def build_inference_table(
    records: Iterable[Mapping[str, Any]],
    *,
    class_order: Sequence[str] = OFFICIAL_CLASS_ORDER,
    label_fields: Sequence[str] = ("y", "Y", "label", "f", "F", "u", "U", "u_e", "u_f"),
) -> InferenceTable:
    """Build the inference table without reading evaluation labels.

    Known label field names are ignored if present, so a joined table can be
    re-scored without leaking Y into the predictions.
    """
    ignored = {field.lower() for field in label_fields}
    rows: list[InferenceRow] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        image_id = str(record.get("image_id") or "").strip()
        building_uid = str(record.get("building_uid") or "").strip()
        if not image_id or not building_uid:
            raise ConfigError(
                f"Inference row {index} needs image_id and building_uid. "
                "Those keys must be unique and one-to-one with evaluation labels."
            )
        key = (image_id, building_uid)
        if key in seen:
            raise ConfigError(f"Duplicate inference key {key}. Prediction rows must be unique.")
        seen.add(key)
        scored = score_probabilities(_row_probabilities(record, class_order), class_order=class_order)
        extras = {
            name: value
            for name, value in record.items()
            if name not in {
                "image_id",
                "building_uid",
                "probabilities",
                "probs",
                "p",
                "disaster",
                "fold",
                "seed",
                "reporting_model_version",
                "out_of_sample",
            }
            and name.lower() not in ignored
        }
        seed_value = record.get("seed")
        rows.append(
            InferenceRow(
                image_id=image_id,
                building_uid=building_uid,
                probabilities=scored.probabilities,
                entropy=scored.entropy,
                one_minus_max_p=scored.one_minus_max_p,
                y_hat_s=scored.prediction.class_name,
                y_hat_s_index=scored.prediction.class_index,
                tied=scored.prediction.tied,
                tied_indices=scored.prediction.tied_indices,
                disaster=str(record.get("disaster") or "").strip(),
                fold=str(record.get("fold") or "").strip(),
                seed=None if seed_value is None or seed_value == "" else int(seed_value),
                reporting_model_version=str(record.get("reporting_model_version") or "").strip(),
                out_of_sample=_optional_bool(record.get("out_of_sample")),
                extras=tuple(sorted(extras.items(), key=lambda item: item[0])),
            )
        )
    rows.sort(key=lambda row: row.key)
    return InferenceTable(rows=tuple(rows), class_order=tuple(class_order))


def align_inference_and_labels(
    table: InferenceTable,
    labels: Iterable[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> tuple[tuple[InferenceRow, dict[str, Any]], ...]:
    """Join evaluation labels onto an already-scored inference table.

    Alignment is one-to-one on ``(image_id, building_uid)``. The inference
    rows are not recomputed.
    """
    label_rows = [_label_record(record, index) for index, record in enumerate(labels)]
    keys = [item[0] for item in label_rows]
    if len(set(keys)) != len(keys):
        duplicate = next(key for key in keys if keys.count(key) > 1)
        raise ConfigError(f"Duplicate evaluation-label key {duplicate}. Alignment must be one-to-one.")
    by_key = dict(label_rows)
    inference_keys = set(table.keys())
    label_keys = set(by_key)
    missing_labels = sorted(inference_keys - label_keys)
    missing_predictions = sorted(label_keys - inference_keys)
    if require_complete and (missing_labels or missing_predictions):
        raise ConfigError(
            "Inference table and evaluation labels are not one-to-one. "
            f"Missing labels for {missing_labels or 'none'}; "
            f"missing predictions for {missing_predictions or 'none'}."
        )
    aligned = []
    for row in table.rows:
        if row.key not in by_key:
            continue
        aligned.append((row, by_key[row.key]))
    return tuple(aligned)


def inference_payload_unchanged(before: InferenceTable, after: InferenceTable) -> bool:
    return before.score_payload() == after.score_payload()


def replace_inference_metadata(row: InferenceRow, **changes: Any) -> InferenceRow:
    return replace(row, **changes)


def _row_probabilities(record: Mapping[str, Any], class_order: Sequence[str]) -> list[float]:
    if record.get("probabilities") is not None:
        return list(record["probabilities"])
    if record.get("probs") is not None:
        return list(record["probs"])
    if record.get("p") is not None:
        return list(record["p"])
    named = []
    for name in class_order:
        key = f"p_{name}"
        if key not in record:
            raise ConfigError(
                "Inference row needs a probabilities vector of length "
                f"{len(class_order)} in official class order {tuple(class_order)}."
            )
        named.append(record[key])
    return named


def _as_probability_vector(probabilities: Sequence[float], *, class_count: int) -> np.ndarray:
    try:
        values = np.asarray(list(probabilities), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Probabilities must be a numeric vector, got {probabilities!r}.") from exc
    if values.ndim != 1 or values.size != class_count:
        raise ConfigError(
            f"Probabilities must have length {class_count} in official class order, got shape {values.shape}."
        )
    if np.any(~np.isfinite(values)):
        raise ConfigError("Probabilities must be finite")
    if np.any(values < -PROBABILITY_SUM_ATOL):
        raise ConfigError(f"Probabilities must be non-negative, got {values.tolist()}.")
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if abs(total - 1.0) > PROBABILITY_SUM_ATOL:
        raise ConfigError(f"Probabilities must sum to 1 plus or minus {PROBABILITY_SUM_ATOL}, got {total}.")
    return values


def _label_record(record: Mapping[str, Any], index: int) -> tuple[tuple[str, str], dict[str, Any]]:
    image_id = str(record.get("image_id") or "").strip()
    building_uid = str(record.get("building_uid") or "").strip()
    if not image_id or not building_uid:
        raise ConfigError(f"Evaluation label {index} needs image_id and building_uid")
    return (image_id, building_uid), dict(record)


def _optional_bool(value) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    raise ConfigError(f"out_of_sample must be a boolean or null, got {value!r}")

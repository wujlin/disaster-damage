"""Classification metrics for the four xBD damage classes."""

from __future__ import annotations

import numpy as np

from mvbridge.data.xbd_labels import DAMAGE_CLASSES


def classification_metrics(y_true, y_pred) -> dict:
    labels = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(y_pred, dtype=np.int64)
    n_classes = len(DAMAGE_CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(labels, predictions):
        if 0 <= int(truth) < n_classes and 0 <= int(pred) < n_classes:
            confusion[int(truth), int(pred)] += 1
    per_class = {}
    f1_values = []
    supports = []
    for index, name in enumerate(DAMAGE_CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        supports.append(support)
    support_sum = int(sum(supports))
    weighted = 0.0
    if support_sum:
        weighted = float(sum(f1 * support for f1, support in zip(f1_values, supports)) / support_sum)
    accuracy = float(np.trace(confusion) / support_sum) if support_sum else 0.0
    return {
        "count": int(labels.size),
        "accuracy": accuracy,
        "weighted_f1": weighted,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "classes": list(DAMAGE_CLASSES),
    }

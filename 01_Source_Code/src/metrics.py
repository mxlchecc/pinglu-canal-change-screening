from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    precision: float
    recall: float
    f1: float
    iou: float
    overall_accuracy: float
    kappa: float

    def to_dict(self) -> dict:
        return {key: value.item() if isinstance(value, np.generic) else value
                for key, value in asdict(self).items()}


class ConfusionAccumulator:
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=bool).ravel()
        target = np.asarray(target, dtype=bool).ravel()
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
        self.tp += int(np.count_nonzero(prediction & target))
        self.tn += int(np.count_nonzero(~prediction & ~target))
        self.fp += int(np.count_nonzero(prediction & ~target))
        self.fn += int(np.count_nonzero(~prediction & target))

    def compute(self) -> BinaryMetrics:
        tn, fp, fn, tp = self.tn, self.fp, self.fn, self.tp
        eps = np.finfo(np.float64).eps
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        total = tn + fp + fn + tp
        oa = (tp + tn) / (total + eps)
        p_yes = ((tp + fp) * (tp + fn)) / ((total * total) + eps)
        p_no = ((fn + tn) * (fp + tn)) / ((total * total) + eps)
        p_e = p_yes + p_no
        kappa = (oa - p_e) / (1 - p_e + eps)
        return BinaryMetrics(tn, fp, fn, tp, precision, recall, f1, iou, oa, kappa)

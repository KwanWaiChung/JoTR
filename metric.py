import torch
from typing import List, Tuple


class DiaactF1:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n_label = 0.0
        self.n_pred = 0.0
        self.n_correct = 0.0

    def compute(self):
        p = self.n_correct / (self.n_pred + 1e-8)
        r = self.n_correct / (self.n_label + 1e-8)
        f1 = (2 * p * r) / (p + r + 1e-8)
        return f1

    def __call__(
        self,
        y_pred: List[Tuple[str, str, str]],
        y_true: List[Tuple[str, str, str]],
    ):
        """
        Args:
            y_pred: List of (domain, intent, slot)
            y_true: List of (domain, intent, slot)

        Return:
            float: The f1 metric of current example.

        """
        y_pred = set(y_pred)
        y_true = set(y_true)
        n_correct = len(y_pred & y_true)
        n_pred = len(y_pred)
        n_label = len(y_true)

        p = n_correct / (n_pred + 1e-12)
        r = n_correct / (n_label + 1e-12)
        f1 = (2 * p * r) / (p + r + 1e-12)

        self.n_pred += n_pred
        self.n_label += n_label
        self.n_correct += n_correct
        return f1
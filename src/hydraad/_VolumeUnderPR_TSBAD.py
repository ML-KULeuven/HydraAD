from typing import Literal

import numpy as np
from dtaianomaly.evaluation import ProbaMetric
from TSB_AD.evaluation.basic_metrics import generate_curve

__all__ = ["VolumeUnderPR_TSBAD"]


class VolumeUnderPR_TSBAD(ProbaMetric):
    sliding_window: int
    version: Literal["opt", "opt_mem"]
    threshold: int

    def __init__(
        self,
        sliding_window: int = 100,
        version: Literal["opt", "opt_mem"] = "opt",
        threshold: int = 250,
    ):
        self.sliding_window = sliding_window
        self.version = version
        self.threshold = threshold

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
            label=y_true,
            score=y_pred,
            slidingWindow=self.sliding_window,
            version=self.version,
            thre=self.threshold,
        )
        return VUS_PR

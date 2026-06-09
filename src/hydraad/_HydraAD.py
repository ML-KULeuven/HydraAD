from typing import Literal

import numba as nb
import numpy as np
import torch
from dtaianomaly.anomaly_detection import BaseDetector, Supervision
from dtaianomaly.utils import get_dimension, is_univariate
from dtaianomaly.windowing import WINDOW_SIZE_TYPE, compute_window_size
from sklearn.preprocessing import StandardScaler

from hydraad._Hydra import Hydra
from hydraad._SparseScaler import SparseScaler

__all__ = ["HydraAD"]


class HydraAD(BaseDetector):
    window_size: WINDOW_SIZE_TYPE
    periodicity: float
    overlap_rate: float
    quantile: float
    scaling: Literal["sparse", "standard", "none"]
    batch_size: int | None = None,
    # Hydra parameters
    n_kernels: int
    n_groups: int
    dilations: list[int] | Literal["auto"]
    first_order_difference: bool
    hard_counts: bool
    soft_counts: bool
    kernel_size: int
    relative_n_channels: float
    max_n_channels: int | None
    device: torch.device | None
    seed: int | None

    hydra_: Hydra

    def __init__(
        self,
        window_size: WINDOW_SIZE_TYPE,
        periodicity: float = 1.0,
        overlap_rate: float = 0.25,
        quantile: float = 0.9,
        scaling: Literal["sparse", "standard", "none"] = "sparse",
        batch_size: int | None = None,
        n_kernels: int = 4,
        n_groups: int = 32,
        dilations: list[int] | Literal["auto"] = "auto",
        first_order_difference: bool = True,
        hard_counts: bool = False,
        soft_counts: bool = True,
        kernel_size: int = 9,
        relative_n_channels: float = 0.3,
        max_n_channels: int | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
    ):
        super().__init__(Supervision.UNSUPERVISED)
        self.window_size = window_size
        self.periodicity = periodicity
        self.overlap_rate = overlap_rate
        self.quantile = quantile
        self.scaling = scaling
        self.batch_size = batch_size

        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.dilations = dilations
        self.first_order_difference = first_order_difference
        assert (hard_counts is True) or (soft_counts is True), (
            "Either the hard counts, the soft counts, or both must be computed"
        )
        self.hard_counts = hard_counts
        self.soft_counts = soft_counts
        self.kernel_size = kernel_size
        self.relative_n_channels = relative_n_channels
        self.max_n_channels = max_n_channels
        self.device = device
        self.seed = seed

    def _fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> None:
        window_size_ = int(
            _compute_window_size_multivariate(X, self.window_size, **kwargs)
            * self.periodicity
        )
        self.window_size_ = max(self.kernel_size, window_size_)

        # Hydra
        self.hydra_ = (
            Hydra(
                n_kernels=self.n_kernels,
                n_groups=self.n_groups,
                dilations=self.dilations,
                first_order_difference=self.first_order_difference,
                hard_counts=self.hard_counts,
                soft_counts=self.soft_counts,
                kernel_size=self.kernel_size,
                relative_n_channels=self.relative_n_channels,
                max_n_channels=self.max_n_channels,
                seed=self.seed,
            )
            .initialize_weights(n_channels=get_dimension(X), n_steps=self.window_size_)
            .to(self.device)
        )

    def _decision_function(self, X: np.ndarray) -> np.array:
        # Initialize the data loader
        stride = max(1, int(self.window_size_ * self.overlap_rate))

        windows = (
            torch.as_tensor(X, dtype=torch.float32)
            .unfold(0, self.window_size_, stride)
            .to(self.device)
        )
        if X.ndim == 1:
            windows = windows.unsqueeze(1)
        with torch.no_grad():
            if self.batch_size is None or len(X) <= self.batch_size:
                embedding = self.hydra_.transform(windows)
            else:
                Z = []
                batches = torch.arange(len(windows)).split(self.batch_size)
                for batch in batches:
                    Z.append(self.hydra_.transform(windows[batch]))
                embedding = torch.cat(Z)

        # Scaling
        if self.scaling == "sparse":
            embedding = SparseScaler().fit_transform(embedding)
        if self.scaling == "standard":
            embedding = torch.Tensor(StandardScaler().fit_transform(embedding))

        # Decision scores
        decision_scores = (
            torch.quantile(embedding.abs(), self.quantile, dim=1).cpu().numpy()
        )
        if decision_scores.min() == decision_scores.max():
            decision_scores -= decision_scores.min()

        return _reverse_sliding_window(
            decision_scores, self.window_size_, stride, X.shape[0]
        )


def _compute_window_size_multivariate(
    X: np.ndarray, window_size: WINDOW_SIZE_TYPE, **kwargs
) -> int:
    # Compute the window size
    if is_univariate(X):
        return int(compute_window_size(X, window_size, **kwargs))
    else:
        default_widow_size = (
            kwargs["default_window_size"] if "default_window_size" in kwargs else -1
        )
        kwargs["default_window_size"] = -1
        window_sizes = list(
            filter(
                lambda i: i > 0,
                [
                    compute_window_size(X[:, i], window_size, **kwargs)
                    for i in range(X.shape[1])
                ],
            )
        )
        if len(window_sizes) == 0:
            assert default_widow_size != -1, (
                "Could not compute window size from the data."
            )
            return kwargs["default_window_size"]

        else:
            return int(np.mean(window_sizes))


@nb.njit(fastmath=True)
def _reverse_sliding_window(
    per_window_anomaly_scores: np.ndarray,
    window_size: int,
    stride: int,
    length_time_series: int,
) -> np.ndarray:
    # Convert to array
    scores_time = np.empty(length_time_series)

    start_window_index = 0
    min_start_window = 0
    end_window_index = 0
    min_end_window = 0
    for t in range(length_time_series - window_size):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        while t >= min_end_window:
            end_window_index += 1
            min_end_window += stride
        scores_time[t] = np.mean(
            per_window_anomaly_scores[start_window_index:end_window_index]
        )

    for t in range(length_time_series - window_size, length_time_series):
        while min_start_window + window_size <= t:
            start_window_index += 1
            min_start_window += stride
        if start_window_index < per_window_anomaly_scores.shape[0]:
            scores_time[t] = np.mean(per_window_anomaly_scores[start_window_index:])
        else:
            scores_time[t] = 0.0

    return scores_time

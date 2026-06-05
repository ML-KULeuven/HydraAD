"""
An adaptation of the Hydra-transform implementation, which allows for
more flexibility on the hyperparameters of the transformation.

Adapted from https://github.com/angus924/hydra/blob/main/code/hydra.py
Dempster, A., Schmidt, D.F. & Webb, G.I. "Hydra: competing convolutional
kernels for fast and accurate time series classification." Data Min Knowl
Disc 37, 1779–1805 (2023). https://doi.org/10.1007/s10618-023-00939-3
"""

from typing import Literal

import numpy as np
import torch

__all__ = ["Hydra"]


class Hydra(torch.nn.Module):
    n_kernels: int
    n_groups: int
    dilations: int | list[int] | Literal["auto"]
    first_order_difference: bool
    hard_counts: bool
    soft_counts: bool
    kernel_size: int
    relative_n_channels: float
    max_n_channels: int | None
    seed: int | None

    h_: int
    dilations_: torch.Tensor
    paddings_: torch.Tensor
    W_: torch.Tensor
    I_: torch.Tensor

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        dilations: int | list[int] | Literal["auto"] = "auto",
        first_order_difference: bool = True,
        hard_counts: bool = True,
        soft_counts: bool = True,
        kernel_size: int = 9,
        relative_n_channels: float = 0.3,
        max_n_channels: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
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
        self.seed = seed

    def fit(self, X: torch.Tensor, y=None) -> "Hydra":
        return self.initialize_weights(X.shape[1], X.shape[2])

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    def fit_transform(self, X, y=None) -> torch.Tensor:
        return self.fit(X, y).transform(X)

    def initialize_weights(self, n_channels: int, n_steps: int) -> "Hydra":
        for attr in ["dilations_", "paddings_", "W_", "I_"]:
            if not hasattr(self, attr):
                self.register_buffer(attr, None)

        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.dilations == "auto":
            max_exponent = np.log2((n_steps - 1) / (self.kernel_size - 1))
            self.dilations_ = 2 ** torch.arange(int(max_exponent) + 1)
        elif isinstance(self.dilations, int):
            self.dilations_ = torch.IntTensor([self.dilations])
        else:
            self.dilations_ = torch.IntTensor(self.dilations)

        self.paddings_ = torch.div(
            (self.kernel_size - 1) * self.dilations_, 2, rounding_mode="floor"
        ).int()

        if self.first_order_difference and self.n_groups > 1:
            n_subsets = 2
        else:
            n_subsets = 1

        self.h_ = self.n_groups // n_subsets

        # Initialize the kernels
        self.W_ = torch.randn(
            self.dilations_.shape[0],  # For every dilation factor
            n_subsets,  # For every subset of groups within the dilation
            self.n_kernels
            * self.h_,  # For every kernel across all groups (of the subset
            1,  # Generate a *univariate* kernel ...
            self.kernel_size,  # ... of the specified kernel size
        )
        self.W_ = self.W_ - self.W_.mean(-1, keepdim=True)
        self.W_ = self.W_ / self.W_.abs().sum(-1, keepdim=True)

        # Initialize the subspaces
        num_channels_per = np.clip(
            int(n_channels * self.relative_n_channels),
            1,
            self.max_n_channels,
        )
        self.I_ = torch.randint(
            low=0,
            high=n_channels,
            size=(
                self.dilations_.shape[0],
                self.W_.shape[1],
                self.h_,
                num_channels_per,
            ),
        )

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n_sequences = X.shape[0]
        time_series_dimension = X.shape[1]

        # Compute the difference if it is necessary
        diff_X = None
        if self.W_.shape[1] == 2:
            diff_X = torch.diff(X)

        Z = []

        for i in range(self.dilations_.shape[0]):
            for diff_index in range(self.W_.shape[1]):
                # Compute the convolutions
                if time_series_dimension > 1:  # Multivariate
                    _Z = torch.nn.functional.conv1d(
                        (
                            X[:, self.I_[i][diff_index]].sum(2)
                            if diff_index == 0
                            else diff_X[:, self.I_[i][diff_index]].sum(2)
                        ),
                        self.W_[i][diff_index],
                        dilation=self.dilations_[i].item(),
                        padding=self.paddings_[i].item(),
                        groups=self.h_,
                    ).view(n_sequences, self.h_, self.n_kernels, -1)

                else:  # Univariate
                    _Z = torch.nn.functional.conv1d(
                        X if diff_index == 0 else diff_X,
                        self.W_[i, diff_index],
                        dilation=self.dilations_[i].item(),
                        padding=self.paddings_[i].item(),
                    ).view(n_sequences, self.h_, self.n_kernels, -1)

                # Compute the soft counts, i.e., the sum of the convolutions
                if self.soft_counts:
                    max_values, max_indices = _Z.max(2)
                    count_max = torch.zeros(
                        n_sequences, self.h_, self.n_kernels, device=self.W_.device
                    )
                    count_max.scatter_add_(-1, max_indices, max_values)
                    Z.append(count_max)

                # Compute the hard counts, i.e., the pure counts
                if self.hard_counts:
                    min_values, min_indices = _Z.min(2)
                    count_min = torch.zeros(
                        n_sequences, self.h_, self.n_kernels, device=self.W_.device
                    )
                    count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))
                    Z.append(count_min)

        return torch.cat(Z, 1).view(n_sequences, -1)

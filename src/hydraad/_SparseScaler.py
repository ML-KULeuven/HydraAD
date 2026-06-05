import torch
from dtaianomaly.type_validation import (
    AttributeValidationMixin,
    BoolAttribute,
    IntegerAttribute,
)
from dtaianomaly.utils import CheckIsFittedMixin

__all__ = ["SparseScaler"]


class SparseScaler(CheckIsFittedMixin, AttributeValidationMixin):
    """
    Adaptation of standard scaling to deal with sparse data :cite:`dempster2023hydra`.

    This scaler applies square root transformation to non-negative data and
    standardizes it using mean and standard deviation. It adds an epsilon term
    to the standard deviation that increases with the sparsity of each feature,
    preventing division by small values for sparse features. Optionally, it can
    mask zero values during transformation to preserve sparsity patterns. Formally,
    the transformation is defined as

    .. math::

        \\hat{Z} = \\frac{\\sqrt{Z} - \\mu_{\\sqrt{Z}}}{\\sigma_{\\sqrt{Z}} + \\alpha^4}

    where :math:`Z` is the input data, :math:`\\mu_{\\sqrt{Z}}` and
    :math:`\\sigma_{\\sqrt{Z}}` are the mean and standard deviation of the
    square root transformed data, and :math:`\\alpha` is the per-feature sparsity.

    Parameters
    ----------
    mask : bool, default=True
        Whether to mask zero values during transformation. If True, zero values
        remain zero after transformation, preserving sparsity patterns.
    exponent : int, default=4
        The exponent applied to the sparsity ratio when computing epsilon.

    Attributes
    ----------
    mu_ : torch.Tensor
        The mean of each feature after square root transformation, computed during
        fitting.
    sigma_ : torch.Tensor
        The standard deviation of each feature after square root transformation,
        with epsilon adjustment for sparsity, computed during fitting.

    See Also
    --------
    fit : Fit the scaling parameters.
    transform : Apply the learned scaling.

    Notes
    -----
    - The code is an adapatation of https://github.com/angus924/hydra/blob/main/code/hydra.py.
    - A small epsilon term is added to the standard deviation to prevent numerical instability,
      especially to avoid division by zero for features that are entirely sparse.

    Examples
    --------
    >>> import torch
    >>> from hydraad import SparseScaler
    >>> X = torch.tensor([[0.0, 1.0, 4.0], [0.0, 0.0, 9.0], [0.0, 4.0, 16.0]])
    >>> scaler = SparseScaler()
    >>> scaler.fit(X)
    >>> scaler.transform(X)
    """

    mask: bool
    exponent: int
    mu_: torch.Tensor
    sigma_: torch.Tensor

    EPSILON: float = 1e-8

    attribute_validation = {
        "mask": BoolAttribute(),
        "exponent": IntegerAttribute(0),
    }

    def __init__(self, mask: bool = True, exponent: int = 4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X: torch.Tensor, y=None) -> "SparseScaler":
        """
        Fit the scaling parameters.

        Computes the mean and standard deviation of each feature after square root
        transformation. An epsilon term is added to the standard deviation based on
        the sparsity ratio of each feature to prevent numerical instability.

        Parameters
        ----------
        X : array-like of type float and shape (n_samples, n_features)
            The input data to fit the scaler on. Must contain non-negative values.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        SparseScaler
            Returns the fitted scaler instance.

        See Also
        --------
        transform : Apply the learned scaling.
        fit_transform : Fit and transform in one step.

        Examples
        --------
        >>> import torch
        >>> from hydraad import SparseScaler
        >>> X = torch.tensor([[0.0, 1.0, 4.0], [0.0, 0.0, 9.0], [0.0, 4.0, 16.0]])
        >>> scaler = SparseScaler().fit(X)
        """
        X = X.clamp(0).sqrt()
        epsilon = (X == 0).float().mean(0) ** self.exponent + self.EPSILON
        self.mu_ = X.mean(0)
        self.sigma_ = X.std(0) + epsilon
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned scaling.

        Applies square root transformation to the input data and standardizes it
        using the mean and standard deviation computed during fitting. If masking
        is enabled, zero values remain zero after transformation.

        Parameters
        ----------
        X : array-like of type float and shape (n_samples, n_features)
            The input data to transform. Must contain non-negative values.

        Returns
        -------
        torch.Tensor
            The transformed data with the same shape as the input.

        See Also
        --------
        fit : Fit the scaling parameters.
        fit_transform : Fit and transform in one step.

        Examples
        --------
        >>> import torch
        >>> from hydraad import SparseScaler
        >>> X = torch.tensor([[0.0, 1.0, 4.0], [0.0, 0.0, 9.0], [0.0, 4.0, 16.0]])
        >>> scaler = SparseScaler().fit(X)
        >>> scaler.transform(X)
        """
        X = X.clamp(0).sqrt()
        if self.mask:
            return ((X - self.mu_) * (X != 0)) / self.sigma_
        else:
            return (X - self.mu_) / self.sigma_

    def fit_transform(self, X: torch.Tensor, y=None) -> torch.Tensor:
        """
        Fit and transform in one step.

        Computes the scaling parameters from the input data and immediately applies
        the transformation. This is equivalent to calling fit(X) followed by
        transform(X), but more convenient and efficient.

        Parameters
        ----------
        X : array-like of type float and shape (n_samples, n_features)
            The input data to fit the scaler on and transform. Must contain
            non-negative values.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        torch.Tensor
            The transformed data with the same shape as the input.

        See Also
        --------
        fit : Fit the scaling parameters.
        transform : Apply the learned scaling.

        Examples
        --------
        >>> import torch
        >>> from hydraad import SparseScaler
        >>> X = torch.tensor([[0.0, 1.0, 4.0], [0.0, 0.0, 9.0], [0.0, 4.0, 16.0]])
        >>> scaler = SparseScaler()
        >>> X_transformed = scaler.fit_transform(X)
        """
        return self.fit(X, y).transform(X)

"""CSR layout container and backend dispatch for sparse kernels.

See docs/sparse_contract.md. This is the minimal setup-commit version:
the layout keeps its arrays and shape and checks only basic structure.
Canonicalization, construction helpers, defensive copies and complete
validation belong to the CSR storage owner.
"""
from __future__ import annotations

import numpy as np

from .ndarray import NDArray

BACKENDS = ("reference", "cpu")


class CSRLayout:
    """Fixed CSR structure of a matrix of shape (m, n) with s stored entries."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, shape: tuple[int, int]):
        if len(shape) != 2:
            raise ValueError(f"shape must have two entries, got {shape!r}")
        m, n = (int(d) for d in shape)
        if m < 0 or n < 0:
            raise ValueError(f"shape must be nonnegative, got {shape!r}")
        for name, arr in (("indptr", indptr), ("indices", indices)):
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional NumPy array")
            if arr.dtype != np.int32:
                raise TypeError(f"{name} must have dtype int32, got {arr.dtype}")
        if indptr.shape != (m + 1,):
            raise ValueError(f"indptr must have shape ({m + 1},), got {indptr.shape}")
        if indptr[0] != 0 or indptr[-1] != indices.shape[0]:
            raise ValueError("indptr must start at 0 and end at len(indices)")
        self.indptr = indptr
        self.indices = indices
        self.shape = (m, n)

    @property
    def nnz(self) -> int:
        """Number of stored entries s."""
        return int(self.indices.shape[0])

    def __repr__(self) -> str:
        return f"CSRLayout(shape={self.shape}, nnz={self.nnz})"


def _impl(backend: str):
    if backend == "reference":
        from . import sparse_reference

        return sparse_reference
    if backend == "cpu":
        try:
            from . import sparse_backend_cpu  # type: ignore[attr-defined]
        except ImportError as e:
            raise RuntimeError(
                "backend='cpu' needs the compiled sparse_backend_cpu extension; "
                "build it with `make lib`. There is no fallback to the reference."
            ) from e
        return sparse_backend_cpu
    raise ValueError(f"unknown sparse backend {backend!r}; expected one of {BACKENDS}")


def spmm(layout: CSRLayout, values: NDArray, X: NDArray, *, backend: str = "reference") -> NDArray:
    """A @ X, shape (m, f)."""
    return _impl(backend).spmm(layout, values, X)


def spmm_transpose(layout: CSRLayout, values: NDArray, G: NDArray, *, backend: str = "reference") -> NDArray:
    """A.T @ G, shape (n, f)."""
    return _impl(backend).spmm_transpose(layout, values, G)


def sampled_dot(layout: CSRLayout, G: NDArray, X: NDArray, *, backend: str = "reference") -> NDArray:
    """dot(G[i], X[j]) for each stored (i, j) in CSR order, shape (s,)."""
    return _impl(backend).sampled_dot(layout, G, X)

"""Sparse Tensor operations (see docs/sparse_contract.md)."""
from __future__ import annotations

__all__ = ["sparse_matmul"]


def sparse_matmul(A, X, *, backend: str = "reference"):
    """Y = A @ X for a SparseMatrix A of shape (m, n) and a Tensor X of shape (n, f).

    Backward: dL/dX = spmm_transpose(layout, values, G) and
    dL/dvalues = sampled_dot(layout, G, X).
    """
    raise NotImplementedError("sparse_matmul is not implemented yet")

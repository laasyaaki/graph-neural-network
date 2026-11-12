"""Tensor-level sparse matrix (see docs/sparse_contract.md)."""
from __future__ import annotations

import numpy as np

from .autograd import Tensor
from .backend_ndarray.sparse import CSRLayout


class SparseMatrix:
    """A CSR matrix with fixed structure and differentiable stored values.

    Not a Needle Tensor itself: `layout` is fixed during differentiation and
    `values` is a Tensor of shape (s,) that optimizers may update.
    """

    def __init__(self, layout: CSRLayout, values: Tensor):
        if not isinstance(layout, CSRLayout):
            raise TypeError(f"layout must be a CSRLayout, got {type(layout).__name__}")
        if not isinstance(values, Tensor):
            raise TypeError(f"values must be a Needle Tensor, got {type(values).__name__}")
        if values.shape != (layout.nnz,):
            raise ValueError(f"values must have shape ({layout.nnz},), got {values.shape}")
        self.layout = layout
        self.values = values

    @property
    def shape(self) -> tuple[int, int]:
        return self.layout.shape

    @property
    def nnz(self) -> int:
        return self.layout.nnz

    @property
    def device(self):
        return self.values.device

    def __matmul__(self, X: Tensor) -> Tensor:
        """A @ X with the reference backend; call sparse_matmul to pick one."""
        from .ops.ops_sparse import sparse_matmul

        return sparse_matmul(self, X)

    def to_dense(self) -> np.ndarray:
        """Dense NumPy copy of A (for tests and debugging; not differentiable)."""
        m, n = self.shape
        dense = np.zeros((m, n), dtype=np.float32)
        rows = np.repeat(np.arange(m), np.diff(self.layout.indptr))
        np.add.at(dense, (rows, self.layout.indices), self.values.numpy())
        return dense

    def __repr__(self) -> str:
        return f"SparseMatrix(shape={self.shape}, nnz={self.nnz})"

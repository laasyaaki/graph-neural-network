"""Sparse Tensor operations (see docs/sparse_contract.md).

A SparseMatrix A has a fixed CSRLayout and differentiable stored values.
Each operation below is a Needle TensorOp whose inputs are Tensors only; the
layout and backend choice live on the op. The three ops are closed under
differentiation, so their gradients are themselves differentiable:

    SparseMatmul(values, X)      -> A @ X           (m, f)
    SparseMatmulT(values, G)     -> A.T @ G         (n, f)
    SampledDot(G, X)             -> dot(G[i], X[j]) for stored (i, j), (s,)
"""
from __future__ import annotations

from ..autograd import NDArray, Tensor, TensorOp
from ..backend_ndarray import sparse as sparse_backend
from ..backend_ndarray.sparse import CSRLayout

__all__ = ["sparse_matmul", "sparse_matmul_transpose", "sampled_dot"]


def _check_backend(backend: str) -> str:
    if backend not in sparse_backend.BACKENDS:
        raise ValueError(
            f"unknown sparse backend {backend!r}; expected one of {sparse_backend.BACKENDS}"
        )
    return backend


class _SparseOp(TensorOp):
    def __init__(self, layout: CSRLayout, backend: str):
        if not isinstance(layout, CSRLayout):
            raise TypeError(f"layout must be a CSRLayout, got {type(layout).__name__}")
        self.layout = layout
        self.backend = _check_backend(backend)


class SparseMatmul(_SparseOp):
    """Y = A @ X with inputs (values, X)."""

    def compute(self, values: NDArray, X: NDArray) -> NDArray:
        return sparse_backend.spmm(self.layout, values, X, backend=self.backend)

    def gradient(self, out_grad: Tensor, node: Tensor):
        values, X = node.inputs
        grad_X = SparseMatmulT(self.layout, self.backend)(values, out_grad)
        grad_values = SampledDot(self.layout, self.backend)(out_grad, X)
        return grad_values, grad_X


class SparseMatmulT(_SparseOp):
    """Z = A.T @ G with inputs (values, G)."""

    def compute(self, values: NDArray, G: NDArray) -> NDArray:
        return sparse_backend.spmm_transpose(self.layout, values, G, backend=self.backend)

    def gradient(self, out_grad: Tensor, node: Tensor):
        values, G = node.inputs
        # Z[j] = sum_p values[p] G[i]  =>  dG = A @ dZ, dvalues[p] = dot(G[i], dZ[j])
        grad_G = SparseMatmul(self.layout, self.backend)(values, out_grad)
        grad_values = SampledDot(self.layout, self.backend)(G, out_grad)
        return grad_values, grad_G


class SampledDot(_SparseOp):
    """S[p] = dot(G[i], X[j]) for stored p = (i, j), with inputs (G, X)."""

    def compute(self, G: NDArray, X: NDArray) -> NDArray:
        return sparse_backend.sampled_dot(self.layout, G, X, backend=self.backend)

    def gradient(self, out_grad: Tensor, node: Tensor):
        G, X = node.inputs
        # With W the matrix whose stored values are out_grad: dG = W @ X, dX = W.T @ G
        grad_G = SparseMatmul(self.layout, self.backend)(out_grad, X)
        grad_X = SparseMatmulT(self.layout, self.backend)(out_grad, G)
        return grad_G, grad_X


def _check_operands(A, X) -> None:
    from ..sparse import SparseMatrix  # avoid a circular import at module load

    if not isinstance(A, SparseMatrix):
        raise TypeError(f"A must be a SparseMatrix, got {type(A).__name__}")
    if not isinstance(X, Tensor):
        raise TypeError(f"X must be a Needle Tensor, got {type(X).__name__}")


def sparse_matmul(A, X: Tensor, *, backend: str = "reference") -> Tensor:
    """Y = A @ X for a SparseMatrix A of shape (m, n) and a Tensor X of shape (n, f).

    Returns a dense Tensor of shape (m, f). Gradients flow to X and to A.values:
    dL/dX = A.T @ G and dL/dvalues[p] = dot(G[i], X[j]) for stored p = (i, j).
    """
    _check_operands(A, X)
    return SparseMatmul(A.layout, backend)(A.values, X)


def sparse_matmul_transpose(A, G: Tensor, *, backend: str = "reference") -> Tensor:
    """Z = A.T @ G for a SparseMatrix A of shape (m, n) and a Tensor G of shape (m, f)."""
    _check_operands(A, G)
    return SparseMatmulT(A.layout, backend)(A.values, G)


def sampled_dot(layout: CSRLayout, G: Tensor, X: Tensor, *, backend: str = "reference") -> Tensor:
    """dot(G[i], X[j]) for each stored (i, j) of `layout`, in CSR order, shape (s,)."""
    return SampledDot(layout, backend)(G, X)

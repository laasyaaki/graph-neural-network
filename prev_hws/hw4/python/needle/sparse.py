"""Fixed CSR structure with differentiable stored values."""
import numpy as np
from .autograd import Tensor


def _validate_layout(layout):
    # Accept the shared CSRLayout protocol until the backend owner supplies it.
    shape = getattr(layout, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or any(
        type(d) is not int or d < 0 for d in shape
    ):
        raise ValueError("layout.shape must contain two nonnegative Python integers")
    for name in ("indptr", "indices"):
        a = getattr(layout, name, None)
        if not isinstance(a, np.ndarray) or a.dtype != np.int32 or a.ndim != 1:
            raise TypeError("layout." + name + " must be a one-dimensional int32 array")
    ptr, idx = layout.indptr, layout.indices
    if ptr.shape != (shape[0] + 1,) or ptr[0] != 0 or ptr[-1] != len(idx):
        raise ValueError("layout.indptr must delimit every stored entry")
    if np.any(ptr[1:] < ptr[:-1]) or np.any(ptr < 0) or np.any(ptr > len(idx)):
        raise ValueError("layout.indptr must be monotone and in bounds")
    if np.any(idx < 0) or np.any(idx >= shape[1]):
        raise ValueError("layout.indices contains an out-of-bounds column")
    for i in range(shape[0]):
        row = idx[ptr[i]:ptr[i + 1]]
        if np.any(row[1:] <= row[:-1]):
            raise ValueError("layout.indices must be strictly increasing within each row")


def _validate_tensor(value, name):
    if not isinstance(value, Tensor):
        raise TypeError(name + " must be a Needle Tensor")
    if value.dtype != np.dtype("float32"):
        raise TypeError(name + " must have dtype float32")
    if getattr(value.device, "name", "cpu") not in ("cpu", "cpu_numpy"):
        raise ValueError(name + " must use a supported CPU device")


class SparseMatrix:
    """Hold a fixed canonical CSR layout and a live Tensor of stored values.

    Pass an nn.Parameter as values to register it with ordinary Needle optimizers.
    The caller must not mutate layout arrays during forward/backward.
    """
    def __init__(self, layout, values):
        _validate_layout(layout)
        _validate_tensor(values, "values")
        if values.shape != (len(layout.indices),):
            raise ValueError("values must have shape (number of stored entries,)")
        self.layout = layout
        self.values = values

    @property
    def shape(self):
        return self.layout.shape


def sparse_matmul(A, X, *, backend="reference"):
    """Return A @ X, differentiating X and each stored value (including zeros)."""
    if backend not in ("reference", "cpu"):
        raise ValueError("backend must be 'reference' or 'cpu'")
    if not isinstance(A, SparseMatrix):
        raise TypeError("A must be a SparseMatrix")
    _validate_layout(A.layout)
    _validate_tensor(A.values, "values")
    _validate_tensor(X, "X")
    if A.values.shape != (len(A.layout.indices),):
        raise ValueError("values must match layout.indices")
    if len(X.shape) != 2 or X.shape[0] != A.shape[1]:
        raise ValueError("X must have shape (A.shape[1], feature width)")
    if A.values.device != X.device:
        raise ValueError("values and X must use the same device")
    from .ops.ops_sparse import SparseMatMul
    return SparseMatMul(A.layout, backend)(A.values, X)

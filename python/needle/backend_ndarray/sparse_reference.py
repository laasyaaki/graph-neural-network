"""NumPy reference kernels for CSR sparse products (see docs/sparse_contract.md).

These define the numerical meaning that the compiled CPU kernels must match.
They favour clarity over speed.
"""
from __future__ import annotations

import numpy as np

from .ndarray import NDArray

_SUPPORTED_DEVICES = ("cpu", "cpu_numpy")


def _check_array(name: str, arr: NDArray, ndim: int) -> None:
    if not isinstance(arr, NDArray):
        raise TypeError(f"{name} must be a Needle NDArray, got {type(arr).__name__}")
    if arr.device.name not in _SUPPORTED_DEVICES:
        raise NotImplementedError(f"{name} is on {arr.device}; sparse kernels support CPU only")
    if arr.dtype != "float32":
        raise TypeError(f"{name} must have dtype float32, got {arr.dtype}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")


def _check_same_device(*named: tuple[str, NDArray]) -> None:
    first_name, first = named[0]
    for name, arr in named[1:]:
        if arr.device != first.device:
            raise ValueError(f"{name} is on {arr.device} but {first_name} is on {first.device}")


def _row_ids(layout) -> np.ndarray:
    """Row index of every stored entry, in CSR order."""
    return np.repeat(np.arange(layout.shape[0], dtype=np.int64), np.diff(layout.indptr))


def _check_values(layout, values: NDArray) -> None:
    _check_array("values", values, 1)
    if values.shape != (layout.nnz,):
        raise ValueError(f"values must have shape ({layout.nnz},), got {values.shape}")


def spmm(layout, values: NDArray, X: NDArray) -> NDArray:
    m, n = layout.shape
    _check_values(layout, values)
    _check_array("X", X, 2)
    _check_same_device(("values", values), ("X", X))
    if X.shape[0] != n:
        raise ValueError(f"X must have {n} rows, got shape {X.shape}")
    x, v = X.numpy(), values.numpy()
    out = np.zeros((m, X.shape[1]), dtype=np.float32)
    np.add.at(out, _row_ids(layout), v[:, None] * x[layout.indices])
    return NDArray(out, device=X.device)


def spmm_transpose(layout, values: NDArray, G: NDArray) -> NDArray:
    m, n = layout.shape
    _check_values(layout, values)
    _check_array("G", G, 2)
    _check_same_device(("values", values), ("G", G))
    if G.shape[0] != m:
        raise ValueError(f"G must have {m} rows, got shape {G.shape}")
    g, v = G.numpy(), values.numpy()
    out = np.zeros((n, G.shape[1]), dtype=np.float32)
    np.add.at(out, layout.indices, v[:, None] * g[_row_ids(layout)])
    return NDArray(out, device=G.device)


def sampled_dot(layout, G: NDArray, X: NDArray) -> NDArray:
    m, n = layout.shape
    _check_array("G", G, 2)
    _check_array("X", X, 2)
    _check_same_device(("G", G), ("X", X))
    if G.shape[0] != m or X.shape[0] != n or G.shape[1] != X.shape[1]:
        raise ValueError(f"G must be ({m}, f) and X ({n}, f); got {G.shape} and {X.shape}")
    g, x = G.numpy(), X.numpy()
    out = np.einsum("pf,pf->p", g[_row_ids(layout)], x[layout.indices]).astype(np.float32)
    return NDArray(out, device=X.device)

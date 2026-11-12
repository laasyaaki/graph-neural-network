"""Reference sparse kernels against dense NumPy (see docs/sparse_contract.md)."""
import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd
from needle.backend_ndarray.sparse import CSRLayout, spmm, spmm_transpose, sampled_dot

DEVICES = [
    pytest.param(nd.cpu_numpy(), id="cpu_numpy"),
    pytest.param(nd.cpu(), id="cpu", marks=pytest.mark.skipif(not nd.cpu().enabled(), reason="CPU extension not built")),
]
ATOL = 1e-5


def random_csr(m, n, density, rng):
    """Canonical CSR (sorted, no duplicates) plus its dense equivalent."""
    dense = np.where(rng.random((m, n)) < density, rng.standard_normal((m, n)), 0).astype(np.float32)
    rows, cols = np.nonzero(dense)  # row-major order, so columns are sorted within rows
    indptr = np.zeros(m + 1, dtype=np.int32)
    np.add.at(indptr, rows + 1, 1)
    layout = CSRLayout(np.cumsum(indptr).astype(np.int32), cols.astype(np.int32), (m, n))
    return layout, dense[rows, cols].astype(np.float32), dense


SHAPES = [(5, 5, 3, 0.4), (4, 7, 2, 0.5), (7, 3, 4, 0.3), (6, 6, 2, 0.0), (0, 4, 3, 0.5), (4, 0, 3, 0.5), (5, 5, 0, 0.5)]


@pytest.mark.parametrize("m,n,f,density", SHAPES)
@pytest.mark.parametrize("device", DEVICES)
def test_kernels_match_dense(m, n, f, density, device):
    rng = np.random.default_rng(0)
    layout, vals, dense = random_csr(m, n, density, rng)
    X = rng.standard_normal((n, f)).astype(np.float32)
    G = rng.standard_normal((m, f)).astype(np.float32)
    v_nd, X_nd, G_nd = (nd.array(a, device=device) for a in (vals, X, G))

    Y = spmm(layout, v_nd, X_nd)
    assert Y.shape == (m, f) and Y.device == device
    np.testing.assert_allclose(Y.numpy(), dense @ X, atol=ATOL)

    Z = spmm_transpose(layout, v_nd, G_nd)
    assert Z.shape == (n, f)
    np.testing.assert_allclose(Z.numpy(), dense.T @ G, atol=ATOL)

    rows = np.repeat(np.arange(m), np.diff(layout.indptr))
    S = sampled_dot(layout, G_nd, X_nd)
    assert S.shape == (layout.nnz,)
    np.testing.assert_allclose(S.numpy(), np.sum(G[rows] * X[layout.indices], axis=1), atol=ATOL)


def test_empty_rows_and_stored_zero():
    # Row 1 is empty; entry (2, 0) is a stored zero and still counts toward s.
    layout = CSRLayout(np.array([0, 2, 2, 3], np.int32), np.array([0, 2, 0], np.int32), (3, 3))
    v = nd.array(np.array([1.0, 2.0, 0.0], np.float32), device=nd.cpu_numpy())
    X = nd.array(np.arange(6, dtype=np.float32).reshape(3, 2), device=nd.cpu_numpy())
    np.testing.assert_allclose(spmm(layout, v, X).numpy(), [[8, 11], [0, 0], [0, 0]])
    assert sampled_dot(layout, X, X).shape == (3,)


def test_unknown_backend_raises():
    layout, vals, _ = random_csr(2, 2, 1.0, np.random.default_rng(0))
    v = nd.array(vals, device=nd.cpu_numpy())
    X = nd.array(np.ones((2, 1), np.float32), device=nd.cpu_numpy())
    with pytest.raises(ValueError, match="backend"):
        spmm(layout, v, X, backend="gpu")


def test_bad_inputs_raise():
    layout, vals, _ = random_csr(3, 4, 0.5, np.random.default_rng(1))
    v = nd.array(vals, device=nd.cpu_numpy())
    with pytest.raises(ValueError, match="X"):
        spmm(layout, v, nd.array(np.ones((3, 2), np.float32), device=nd.cpu_numpy()))
    with pytest.raises(ValueError, match="X"):
        spmm(layout, v, nd.array(np.ones(4, np.float32), device=nd.cpu_numpy()))
    with pytest.raises(TypeError):
        CSRLayout(np.array([0, 0], np.int64), np.array([], np.int32), (1, 1))
    with pytest.raises(ValueError):
        CSRLayout(np.array([0, 1], np.int32), np.array([], np.int32), (1, 1))


def test_needle_imports_and_dense_aggregate():
    A = ndl.Tensor(np.eye(3, dtype=np.float32), device=nd.cpu_numpy())
    H = ndl.Tensor(np.ones((3, 2), dtype=np.float32), device=nd.cpu_numpy())
    assert ndl.nn.dense_aggregate(A, H).shape == (3, 2)
    assert hasattr(ndl, "SparseMatrix") and hasattr(ndl, "sparse_matmul")

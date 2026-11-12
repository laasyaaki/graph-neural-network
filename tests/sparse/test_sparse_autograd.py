"""SparseMatrix / sparse_matmul forward, backward, and optimizer checks."""
import importlib

import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd
from needle.backend_ndarray.sparse import CSRLayout
from needle.ops.ops_sparse import sparse_matmul, sparse_matmul_transpose, sampled_dot


def _sparse_cpu_built() -> bool:
    try:
        importlib.import_module("needle.backend_ndarray.sparse_backend_cpu")
        return True
    except ImportError:
        return False


CPU_READY = nd.cpu().enabled() and _sparse_cpu_built()
CASES = [
    pytest.param(nd.cpu_numpy(), "reference", id="cpu_numpy-reference"),
    pytest.param(nd.cpu(), "reference", id="cpu-reference",
                 marks=pytest.mark.skipif(not nd.cpu().enabled(), reason="CPU extension not built")),
    pytest.param(nd.cpu(), "cpu", id="cpu-cpu",
                 marks=pytest.mark.skipif(not CPU_READY, reason="sparse CPU kernel not built")),
]
ATOL = 1e-4


def random_csr(m, n, density, rng):
    dense = np.where(rng.random((m, n)) < density, rng.standard_normal((m, n)), 0).astype(np.float32)
    rows, cols = np.nonzero(dense)
    counts = np.bincount(rows, minlength=m)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return CSRLayout(indptr, cols.astype(np.int32), (m, n)), dense[rows, cols], dense


def make_sparse(m, n, density, device, seed=0, param=False):
    rng = np.random.default_rng(seed)
    layout, vals, dense = random_csr(m, n, density, rng)
    cls = ndl.nn.Parameter if param else ndl.Tensor
    return ndl.SparseMatrix(layout, cls(vals, device=device)), dense, rng


def tensor(a, device):
    return ndl.Tensor(np.asarray(a, dtype=np.float32), device=device)


def stored_rows(layout):
    return np.repeat(np.arange(layout.shape[0]), np.diff(layout.indptr))


SHAPES = [(6, 6, 3, 0.4), (4, 7, 2, 0.5), (7, 3, 5, 0.3), (5, 5, 2, 0.0), (5, 5, 0, 0.5), (0, 3, 2, 0.5)]


@pytest.mark.parametrize("m,n,f,density", SHAPES)
@pytest.mark.parametrize("device,backend", CASES)
def test_forward_and_backward_match_dense(m, n, f, density, device, backend):
    A, dense, rng = make_sparse(m, n, density, device)
    Xn = rng.standard_normal((n, f)).astype(np.float32)
    Cn = rng.standard_normal((m, f)).astype(np.float32)
    X = tensor(Xn, device)

    Y = sparse_matmul(A, X, backend=backend)
    assert Y.shape == (m, f) and Y.device == device
    np.testing.assert_allclose(Y.numpy(), dense @ Xn, atol=ATOL)

    # Seed dL/dY = C directly: the baseline's sum() fails on empty arrays.
    Y.backward(tensor(Cn, device))
    np.testing.assert_allclose(X.grad.numpy(), dense.T @ Cn, atol=ATOL)
    rows = stored_rows(A.layout)
    expected_dv = np.sum(Cn[rows] * Xn[A.layout.indices], axis=1)
    assert A.values.grad.shape == (A.nnz,)
    np.testing.assert_allclose(A.values.grad.numpy(), expected_dv, atol=ATOL)


def numerical_grad(loss_fn, arr, eps=1e-2):
    """Central differences in float64 around a float32 array."""
    grad = np.zeros(arr.shape)
    for idx in np.ndindex(*arr.shape):
        orig = arr[idx]
        arr[idx] = orig + eps
        hi = loss_fn()
        arr[idx] = orig - eps
        lo = loss_fn()
        arr[idx] = orig
        grad[idx] = (hi - lo) / (2 * eps)
    return grad


@pytest.mark.parametrize("device,backend", CASES)
def test_finite_difference_through_dense_ops(device, backend):
    """Gradient check of a GCN-like loss: sum((A @ (X @ W)) ** 2)."""
    rng = np.random.default_rng(3)
    layout, vals, _ = random_csr(5, 4, 0.5, rng)
    Xn = rng.standard_normal((4, 3)).astype(np.float32)
    Wn = rng.standard_normal((3, 2)).astype(np.float32)

    def loss(v, x, w):
        A = ndl.SparseMatrix(layout, v)
        return (sparse_matmul(A, x @ w, backend=backend) ** 2).sum()

    v, x, w = tensor(vals, device), tensor(Xn, device), tensor(Wn, device)
    loss(v, x, w).backward()

    def value():
        return float(loss(tensor(vals, device), tensor(Xn, device), tensor(Wn, device)).numpy())

    for arr, t in ((vals, v), (Xn, x), (Wn, w)):
        np.testing.assert_allclose(t.grad.numpy(), numerical_grad(value, arr), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device,backend", CASES)
def test_transpose_and_sampled_dot_gradients(device, backend):
    """The two helper ops are differentiable too (used for higher-order grads)."""
    A, dense, rng = make_sparse(4, 6, 0.5, device, seed=7)
    layout, rows = A.layout, stored_rows(A.layout)
    Gn = rng.standard_normal((4, 3)).astype(np.float32)
    Xn = rng.standard_normal((6, 3)).astype(np.float32)

    # Z = A.T @ G; L = sum(Z * C)
    Cn = rng.standard_normal((6, 3)).astype(np.float32)
    G = tensor(Gn, device)
    Z = sparse_matmul_transpose(A, G, backend=backend)
    np.testing.assert_allclose(Z.numpy(), dense.T @ Gn, atol=ATOL)
    (Z * tensor(Cn, device)).sum().backward()
    np.testing.assert_allclose(G.grad.numpy(), dense @ Cn, atol=ATOL)
    np.testing.assert_allclose(A.values.grad.numpy(),
                               np.sum(Gn[rows] * Cn[layout.indices], axis=1), atol=ATOL)

    # S = sampled_dot(G, X); L = sum(S * c)
    cn = rng.standard_normal(layout.nnz).astype(np.float32)
    G, X = tensor(Gn, device), tensor(Xn, device)
    S = sampled_dot(layout, G, X, backend=backend)
    (S * tensor(cn, device)).sum().backward()
    W = np.zeros_like(dense)
    W[rows, layout.indices] = cn
    np.testing.assert_allclose(G.grad.numpy(), W @ Xn, atol=ATOL)
    np.testing.assert_allclose(X.grad.numpy(), W.T @ Gn, atol=ATOL)


def test_dense_and_sparse_aggregation_agree():
    """Same GCN-style layer with dense_aggregate vs sparse_matmul gives the same grads."""
    device = nd.cpu_numpy()
    A, dense, rng = make_sparse(8, 8, 0.3, device, seed=11)
    Xn = rng.standard_normal((8, 4)).astype(np.float32)
    Wn = rng.standard_normal((4, 3)).astype(np.float32)

    def run(aggregate, graph):
        W = ndl.nn.Parameter(Wn, device=device)
        out = aggregate(graph, tensor(Xn, device) @ W)
        (out ** 2).sum().backward()
        return out.numpy(), W.grad.numpy()

    out_d, gw_d = run(ndl.nn.dense_aggregate, tensor(dense, device))
    out_s, gw_s = run(sparse_matmul, A)
    np.testing.assert_allclose(out_s, out_d, atol=ATOL)
    np.testing.assert_allclose(gw_s, gw_d, atol=1e-3)


@pytest.mark.parametrize("opt_cls,kwargs", [(ndl.optim.SGD, {"lr": 0.05}), (ndl.optim.Adam, {"lr": 0.05})])
def test_optimizer_updates_values_in_place(opt_cls, kwargs):
    """Optimizers update A.values; the SparseMatrix sees the change without rebuilding."""
    device = nd.cpu_numpy()
    A, dense, rng = make_sparse(6, 6, 0.5, device, seed=5, param=True)
    X = tensor(rng.standard_normal((6, 2)), device)
    target = tensor(rng.standard_normal((6, 2)), device)
    opt = opt_cls([A.values], **kwargs)

    def step():
        opt.reset_grad()
        loss = ((sparse_matmul(A, X) - target) ** 2).sum()
        loss.backward()
        opt.step()
        return float(loss.numpy())

    before = A.values.numpy().copy()
    losses = [step() for _ in range(20)]
    assert not np.allclose(A.values.numpy(), before)
    assert A.values.shape == (A.nnz,) and A.values.dtype == "float32"
    assert losses[-1] < losses[0]


def test_matmul_operator_and_to_dense():
    device = nd.cpu_numpy()
    A, dense, rng = make_sparse(4, 5, 0.5, device, seed=2)
    X = tensor(rng.standard_normal((5, 2)), device)
    np.testing.assert_allclose((A @ X).numpy(), dense @ X.numpy(), atol=ATOL)
    np.testing.assert_allclose(A.to_dense(), dense)


def test_input_errors():
    device = nd.cpu_numpy()
    A, dense, _ = make_sparse(3, 4, 0.5, device)
    with pytest.raises(ValueError, match="backend"):
        sparse_matmul(A, tensor(np.ones((4, 2)), device), backend="gpu")
    with pytest.raises(TypeError, match="SparseMatrix"):
        sparse_matmul(tensor(dense, device), tensor(np.ones((4, 2)), device))
    with pytest.raises(TypeError, match="Tensor"):
        sparse_matmul(A, np.ones((4, 2), np.float32))
    with pytest.raises(ValueError, match="X"):
        sparse_matmul(A, tensor(np.ones((3, 2)), device))
    with pytest.raises(ValueError, match="values"):
        ndl.SparseMatrix(A.layout, tensor(np.ones(A.nnz + 1), device))
    with pytest.raises(TypeError, match="layout"):
        ndl.SparseMatrix("not a layout", A.values)


@pytest.mark.skipif(_sparse_cpu_built(), reason="sparse CPU kernel is built")
def test_cpu_backend_does_not_fall_back():
    device = nd.cpu_numpy()
    A, _, _ = make_sparse(3, 4, 0.5, device)
    with pytest.raises(RuntimeError, match="sparse_backend_cpu"):
        sparse_matmul(A, tensor(np.ones((4, 2)), device), backend="cpu")

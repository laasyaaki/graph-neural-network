# Sparse interface contract

This is the interface for the setup commit and the three feature branches. The HW4 Needle baseline is tracked at the repository root; the sparse paths described here still need to be implemented and verified.

## Shapes and storage

`CSRLayout(indptr, indices, shape)` represents a two-dimensional matrix `A` of shape `(m, n)` with `s` stored entries. `shape` is a pair of nonnegative Python integers. `indptr` and `indices` are one-dimensional NumPy arrays with dtype `int32`; their shapes are `(m + 1,)` and `(s,)`. `indptr[0] == 0`, `indptr[-1] == s`, and row `i` occupies positions `indptr[i]:indptr[i + 1]`. Each `indices[p]` is in `[0, n)`. Column indices are strictly increasing within each row. Stored zero values are allowed and still count toward `s`. Structure is fixed during differentiation; no gradient is defined for `indptr`, `indices`, or `shape`.

The minimal layout container in the setup commit should retain these arrays and shape. Person 1 owns canonicalization, construction helpers, defensive copies, and complete validation. Reject malformed input rather than silently sorting, coalescing, or truncating it. Full validation must be complete before the CPU backend is used for experiments.

`values` is a one-dimensional Needle NDArray of shape `(s,)`, dtype `float32`. Dense backend inputs and outputs are two-dimensional Needle NDArrays of dtype `float32`. The first backend target is CPU only. Use `cpu()` or `cpu_numpy()` explicitly if the reference implementation supports both; the selected arrays and output must stay on the same device. A CUDA array raises `ValueError` or `NotImplementedError`; it never silently moves to CPU. Invalid ranks, shapes, or dtypes raise `ValueError` or `TypeError` with the offending argument named.

## Backend functions

```python
spmm(layout: CSRLayout, values: NDArray, X: NDArray) -> NDArray
spmm_transpose(layout: CSRLayout, values: NDArray, G: NDArray) -> NDArray
sampled_dot(layout: CSRLayout, G: NDArray, X: NDArray) -> NDArray
```

With `A.shape == (m, n)` and feature width `f`:

| Function | Input shapes | Output shape | Meaning |
|---|---|---|---|
| `spmm` | `values (s,)`, `X (n, f)` | `(m, f)` | `A @ X` |
| `spmm_transpose` | `values (s,)`, `G (m, f)` | `(n, f)` | `A.T @ G` |
| `sampled_dot` | `G (m, f)`, `X (n, f)` | `(s,)` | For stored `(i, j)`, `dot(G[i], X[j])` in CSR order |

The functions return a Needle NDArray, including when `m`, `n`, `f`, or `s` is zero. `sampled_dot` does not take `values`; it computes the gradient for each stored position. The reference implementation may use NumPy inside the function, but it must return an NDArray on the input device. The CPU implementation must have the same signatures and numerical meaning. Compare outputs with a stated `float32` tolerance; do not require bitwise equality from different accumulation orders.

Backend choice is explicit at the wrapper boundary: `backend="reference"` calls the Python implementation and `backend="cpu"` calls the compiled kernel. An unknown backend raises `ValueError`. An unavailable CPU extension raises an error; it does not fall back to the reference. Benchmark output records this choice.

## Tensor interface

```python
class SparseMatrix:
    def __init__(self, layout: CSRLayout, values: Tensor): ...

def sparse_matmul(A: SparseMatrix, X: Tensor, *, backend: str = "reference") -> Tensor: ...
```

`values` is a Needle Tensor of shape `(s,)`, `X` is a Needle Tensor of shape `(n, f)`, and `Y` is a dense Needle Tensor of shape `(m, f)`. All numerical tensors are `float32` on the same supported CPU device. `SparseMatrix` holds fixed `layout` and differentiable `values`; it is not a Needle Tensor itself. The sparse operation's backward returns `spmm_transpose(layout, values, G)` for `X` and `sampled_dot(layout, G, X)` for `values`. Return gradients through Needle's Tensor operation machinery, so ordinary optimizers can update `values` when it is a parameter. Before Person 2 implements the operation, calling it raises `NotImplementedError` and importing `needle` still succeeds.

## GCN convention

```python
def dense_aggregate(graph: Tensor, features: Tensor) -> Tensor:
    return graph @ features

# Within the layer:
projected = features @ self.weight
output = self.aggregate(graph, projected)
```

`features` has shape `(N, f_in)`, `weight` has shape `(f_in, f_out)`, and `graph` is the normalized adjacency `A_hat` of shape `(N, N)`. The aggregator returns shape `(N, f_out)`. Dense mode uses a Needle Tensor and `dense_aggregate`; sparse mode uses `SparseMatrix` and `sparse_matmul`. The layer, loss, optimizer, and training loop use the same calling convention. Build `A_hat = D_tilde^(-1/2) (A_raw + I) D_tilde^(-1/2)` once with self loops; document any feature normalization and the exact Cora split before comparing runs.

## Ownership and checks

Person 1 owns `python/needle/backend_ndarray/sparse.py`, `python/needle/backend_ndarray/sparse_reference.py`, `src/sparse_backend_cpu.cc`, build configuration, and backend tests. Person 2 owns `python/needle/sparse.py`, `python/needle/ops/ops_sparse.py`, autograd tests, package exports, this contract, and CI. Person 3 owns `python/needle/nn/nn_gnn.py`, `graph_data.py`, `apps/train_gcn.py`, GCN tests, and benchmarks.

The setup commit must test all three reference functions on a rectangular matrix and test backward through dense aggregation on the small graph fixture. Subsequent checks compare reference and CPU results, sparse and dense gradients, and sparse and dense GCN outputs on the same normalized adjacency. Do not report reference-kernel timing as sparse CPU performance.

# Graph convolution in Needle

We are adding sparse graph aggregation to Needle and using it to train a graph convolutional network (GCN). A dense adjacency matrix makes the model easy to check, but most of its entries are zero. Compressed sparse row (CSR) storage lets us keep the edges and multiply by node features without allocating the full matrix. The dense path stays as the correctness baseline.

This belongs in Needle because the sparse multiplication needs to participate in Needle's computation graph. The CPU kernel handles the arithmetic, the Tensor operation returns gradients for node features and stored edge weights, and the GCN uses the same layer code with either dense or sparse aggregation. A separate NumPy or PyTorch training script would not test that integration.

The model follows the propagation rule in [Kipf and Welling's GCN paper](https://arxiv.org/abs/1609.02907). We use the normalized adjacency matrix as the graph input. The CSR convention follows [SciPy's canonical CSR format](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html): column indices are sorted within each row and duplicates are removed.

## Dividing the project

| Person                                    | Work                                                                                                   | How to start                                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Laasya: CSR storage and CPU kernels       | CSR construction and validation; forward, transpose, and stored-value gradient kernels; backend tests. | Match each CPU result against the reference kernel, including rectangular and empty-row cases.                             |
| Sameeksha: Sparse Tensor API and autograd | `SparseMatrix`, `sparse_matmul`, backward rules, gradient checks, optimizer compatibility.             | Use the reference kernels first. Switching to `backend="cpu"` should not change gradient logic.                            |
| Dhiren: GCN and experiments               | GCN layer, Cora preparation, dense baseline, training, synthetic graphs, benchmarks.                   | Train with `dense_aggregate` first. Replace the graph object and aggregation function when sparse multiplication is ready. |

After the setup commit, ownership is exclusive unless the owners coordinate a change:

| Owner     | Paths                                                                                                                                                           |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Laasya    | `python/needle/backend_ndarray/sparse.py`, `python/needle/backend_ndarray/sparse_reference.py`, `src/sparse_backend_cpu.cc`, build configuration, backend tests |
| Sameeksha | `python/needle/sparse.py`, `python/needle/ops/ops_sparse.py`, autograd tests, package exports, `docs/sparse_contract.md`, CI                                    |
| Dhiren    | `python/needle/nn/nn_gnn.py`, `graph_data.py`, `apps/train_gcn.py`, GCN tests, benchmark scripts and results                                                    |

## The shared notation

Use `m` for sparse rows, `n` for sparse columns, `f` for feature width, and `s` for the number of stored values. Then `A` has shape `(m, n)`, `X` has shape `(n, f)`, and `Y = A @ X` has shape `(m, f)`. `G = dL/dY` has shape `(m, f)`. The two backward results are `dL/dX = A.T @ G`, shape `(n, f)`, and `dL/dvalues[p] = dot(G[i], X[j])` for stored entry `p = (i, j)`, shape `(s,)`.

For a graph, use `N` nodes and a square adjacency `A` of shape `(N, N)`. Call the raw adjacency `A_raw`, add self loops to get `A_tilde = A_raw + I`, and call `D_tilde^(-1/2) A_tilde D_tilde^(-1/2)` the normalized adjacency `A_hat`. A GCN layer projects first, then aggregates: `projected = features @ weight`; `output = aggregate(graph, projected)`. Use `A_hat` for both dense and sparse runs so the comparison is meaningful. The contract fixes the rest of the conventions.

## Working order

1. Verify the HW4 Needle baseline in `prev_hws/hw4`, add the setup pieces above, and record the setup commit hash.
2. Create three feature branches from that exact hash.
3. Develop against the reference kernels and dense path; keep tests with the component that owns them.
4. Merge components incrementally. Coordinate any change to `docs/sparse_contract.md` before changing an interface.
5. Connect CPU kernels to autograd, then connect sparse aggregation to the GCN. Run gradient checks, dense/sparse equivalence checks, and final CPU benchmarks.

The HW4 Needle source, tests, notebook, and build files are kept unchanged in `prev_hws/hw4` for reference. The working copy lives at the repository root: `python/needle`, `src`, `CMakeLists.txt`, `Makefile`, and the HW4 tests in `tests/hw4`.

From the repository root, `make lib` builds the CPU extension on a supported Unix-like toolchain. `import needle` works without the build; `needle.backend_ndarray.cpu().enabled()` reports whether the extension is available, and `cpu_numpy()` works everywhere. Run `python -m pytest` from the root to run `tests/sparse` (`pytest.ini` sets the Python path). CI in `.github/workflows/ci.yml` builds the extension on Ubuntu and runs the same tests.

The setup commit adds the minimal `CSRLayout`, NumPy reference kernels in `sparse_reference.py`, the backend dispatch in `backend_ndarray/sparse.py`, `SparseMatrix`, a `sparse_matmul` stub, and `nn.dense_aggregate`. The compiled sparse kernel (`src/sparse_backend_cpu.cc`) and its build target are not part of setup; `backend="cpu"` raises until they exist.

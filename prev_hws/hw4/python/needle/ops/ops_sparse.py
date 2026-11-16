"""Sparse Tensor operations; the same derivative graph serves both kernels."""
from importlib import import_module
import numpy as np
from ..autograd import TensorOp
from ..backend_selection import array_api


def _numpy(a):
    return a if isinstance(a, np.ndarray) else a.numpy()


def _reference(name, layout, a, b):
    """Temporary reference adapter, allocating only the output and row temporaries."""
    av, bv = _numpy(a), _numpy(b)
    m, n = layout.shape
    shape = {"spmm": (m, bv.shape[1]),
             "spmm_transpose": (n, bv.shape[1]),
             "sampled_dot": (len(layout.indices),)}[name]
    result = np.zeros(shape, dtype=np.float32)
    for i in range(m):
        for p in range(layout.indptr[i], layout.indptr[i + 1]):
            j = layout.indices[p]
            if name == "spmm":
                result[i] += av[p] * bv[j]
            elif name == "spmm_transpose":
                result[j] += av[p] * bv[i]
            else:
                result[p] = np.dot(av[i], bv[j])
    if isinstance(a, np.ndarray):
        return result
    return array_api.array(result, device=a.device, dtype="float32")


def _kernel(backend, name, layout, a, b):
    module_name = "needle.backend_ndarray." + (
        "sparse_reference" if backend == "reference" else "sparse")
    # NumPy-only Needle cannot load the compiled NDArray backend.
    if backend == "reference" and isinstance(a, np.ndarray):
        return _reference(name, layout, a, b)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        if backend == "reference":
            return _reference(name, layout, a, b)
        raise NotImplementedError("backend='cpu' requires the sparse CPU backend") from exc
    return getattr(module, name)(layout, a, b)


class SparseMatMul(TensorOp):
    def __init__(self, layout, backend="reference"):
        self.layout, self.backend = layout, backend

    def compute(self, values, X):
        return _kernel(self.backend, "spmm", self.layout, values, X)

    def gradient(self, out_grad, node):
        values, X = node.inputs
        return (SampledDot(self.layout, self.backend)(out_grad, X),
                SparseTransposeMatMul(self.layout, self.backend)(values, out_grad))


class SparseTransposeMatMul(SparseMatMul):
    def compute(self, values, G):
        return _kernel(self.backend, "spmm_transpose", self.layout, values, G)

    def gradient(self, out_grad, node):
        values, G = node.inputs
        return (SampledDot(self.layout, self.backend)(G, out_grad),
                SparseMatMul(self.layout, self.backend)(values, out_grad))


class SampledDot(SparseMatMul):
    def compute(self, G, X):
        return _kernel(self.backend, "sampled_dot", self.layout, G, X)

    def gradient(self, out_grad, node):
        G, X = node.inputs
        return (SparseMatMul(self.layout, self.backend)(out_grad, X),
                SparseTransposeMatMul(self.layout, self.backend)(out_grad, G))

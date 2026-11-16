"""Run with PYTHONPATH=python python -m unittest discover -s tests/sparse."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import needle as ndl
from needle.ops import ops_sparse
from needle.backend_selection import BACKEND


def layout(shape=(3, 4), indptr=(0, 2, 2, 4), indices=(0, 3, 1, 2)):
    return SimpleNamespace(shape=shape, indptr=np.array(indptr, dtype=np.int32),
                           indices=np.array(indices, dtype=np.int32))


def tensor(a, **kwargs):
    return ndl.Tensor(np.asarray(a, dtype=np.float32), **kwargs)


class SparseAutogradTests(unittest.TestCase):
    def setUp(self):
        self.layout = layout()
        self.v = np.array([1., 0., -2., 3.], dtype=np.float32)
        self.x = np.arange(8, dtype=np.float32).reshape(4, 2) / 5
        self.g = np.array([[1., -2.], [3., 4.], [-1., 2.]], dtype=np.float32)

    def run_op(self, v=None, x=None):
        v = tensor(self.v if v is None else v)
        x = tensor(self.x if x is None else x)
        y = ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x)
        return v, x, y

    def test_dense_equivalence_and_gradients(self):
        v, x, y = self.run_op()
        dense = np.zeros((3, 4), dtype=np.float32)
        dense[0, [0, 3]] = self.v[:2]
        dense[2, [1, 2]] = self.v[2:]
        a, xd = tensor(dense), tensor(self.x)
        yd = a @ xd
        y.backward(tensor(self.g, requires_grad=False))
        yd.backward(tensor(self.g, requires_grad=False))
        np.testing.assert_allclose(y.numpy(), yd.numpy(), atol=1e-6)
        np.testing.assert_allclose(x.grad.numpy(), xd.grad.numpy(), atol=1e-6)
        np.testing.assert_allclose(v.grad.numpy(), a.grad.numpy()[[0, 0, 2, 2], [0, 3, 1, 2]], atol=1e-6)
        self.assertNotEqual(v.grad.numpy()[1], 0)  # Stored zero remains differentiable.

    def test_finite_difference_both_inputs(self):
        v, x, y = self.run_op()
        y.backward(tensor(self.g, requires_grad=False))
        def loss(values, features):
            return float(np.sum(self.run_op(values, features)[2].numpy() * self.g))
        eps = 1e-3
        for original, analytic, is_values in ((self.v, v.grad.numpy(), True),
                                               (self.x, x.grad.numpy(), False)):
            numeric = np.zeros_like(original)
            for index in np.ndindex(original.shape):
                plus, minus = original.copy(), original.copy()
                plus[index] += eps
                minus[index] -= eps
                numeric[index] = ((loss(plus, self.x) - loss(minus, self.x)) if is_values
                                  else (loss(self.v, plus) - loss(self.v, minus))) / (2 * eps)
            np.testing.assert_allclose(analytic, numeric, atol=8e-4, rtol=2e-3)

    def test_shared_values_accumulate(self):
        v, x, y = self.run_op()
        z = y + ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x)
        z.backward(tensor(self.g))
        v2, x2, y2 = self.run_op()
        y2.backward(tensor(self.g * 2))
        np.testing.assert_allclose(v.grad.numpy(), v2.grad.numpy())
        np.testing.assert_allclose(x.grad.numpy(), x2.grad.numpy())

    def test_frozen_inputs(self):
        for freeze_values in (True, False):
            v = tensor(self.v, requires_grad=not freeze_values)
            x = tensor(self.x, requires_grad=freeze_values)
            y = ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x)
            y.backward(tensor(self.g))
            self.assertIsNotNone((x if freeze_values else v).grad)
        v = tensor(self.v, requires_grad=False)
        x = tensor(self.x, requires_grad=False)
        self.assertFalse(ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x).requires_grad)

    def test_second_derivative(self):
        v, x, y = self.run_op()
        y.backward(tensor(self.g, requires_grad=False))
        # d/dX sum(dL/dvalues) = structure(A).T @ G.
        v.grad.backward(tensor(np.ones(4)))
        expected = np.zeros_like(self.x)
        expected[[0, 3]] = self.g[0]
        expected[[1, 2]] = self.g[2]
        np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_empty_dimensions(self):
        for m, n, f in ((3, 4, 2), (0, 4, 2), (3, 0, 2), (3, 4, 0), (0, 0, 0)):
            a = layout((m, n), [0] * (m + 1), [])
            v, x = tensor(np.zeros(0)), tensor(np.zeros((n, f)))
            y = ndl.sparse_matmul(ndl.SparseMatrix(a, v), x)
            self.assertEqual(y.shape, (m, f))
            y.backward(tensor(np.zeros((m, f))))
            self.assertEqual(v.grad.shape, (0,))
            self.assertEqual(x.grad.shape, (n, f))
            np.testing.assert_array_equal(y.numpy(), np.zeros((m, f)))

    def test_optimizer_updates(self):
        for optimizer in (ndl.optim.SGD, ndl.optim.Adam):
            v = ndl.nn.Parameter(tensor(self.v))
            opt = optimizer([v], lr=0.01)
            for _ in range(2):
                opt.reset_grad()
                ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), tensor(self.x)).backward(tensor(self.g))
                before, grad = v.numpy().copy(), v.grad.numpy().copy()
                opt.step()
                expected = before - .01 * (grad if optimizer is ndl.optim.SGD else grad / (np.abs(grad) + 1e-8))
                np.testing.assert_allclose(v.numpy(), expected, atol=1e-6)

    def test_invalid_arguments(self):
        a = ndl.SparseMatrix(self.layout, tensor(self.v))
        for bad in (np.ones((4, 2)), None):
            with self.assertRaisesRegex(TypeError, "X"):
                ndl.sparse_matmul(a, bad)
        for bad in (np.ones(4), np.ones((3, 2)), np.ones((4, 2, 1))):
            with self.assertRaisesRegex(ValueError, "X"):
                ndl.sparse_matmul(a, tensor(bad))
        with self.assertRaisesRegex(ValueError, "backend"):
            ndl.sparse_matmul(a, tensor(self.x), backend="automatic")
        with self.assertRaisesRegex(TypeError, "A"):
            ndl.sparse_matmul(None, tensor(self.x))
        with self.assertRaisesRegex(ValueError, "values"):
            ndl.SparseMatrix(self.layout, tensor([1]))
        if BACKEND == "np":
            with self.assertRaisesRegex(TypeError, "values"):
                ndl.SparseMatrix(self.layout, ndl.Tensor(self.v, dtype="float64"))
        for bad in (layout(indices=(0, 0, 1, 2)), layout(indices=(0, 4, 1, 2)),
                    layout(indptr=(0, 3, 2, 4)), layout(shape=(-1, 4)),
                    layout(indptr=(0, 1, 2, 3))):
            with self.assertRaises((ValueError, TypeError)):
                ndl.SparseMatrix(bad, tensor(self.v))

    def test_dense_projection_composition(self):
        v, x = tensor(self.v), tensor(self.x)
        w = tensor([[.2, -.4, .6], [.5, .1, -.3]])
        y = ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x @ w)
        loss = (y * y).sum()
        loss.backward()
        dense = np.zeros((3, 4), dtype=np.float32)
        dense[0, [0, 3]], dense[2, [1, 2]] = self.v[:2], self.v[2:]
        xd, wd = tensor(self.x), tensor(w.numpy())
        yd = tensor(dense, requires_grad=False) @ (xd @ wd)
        (yd * yd).sum().backward()
        np.testing.assert_allclose(loss.numpy(), np.sum(yd.numpy() ** 2), atol=1e-5)
        np.testing.assert_allclose(x.grad.numpy(), xd.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(w.grad.numpy(), wd.grad.numpy(), atol=1e-5)

    @unittest.skipUnless(BACKEND == "nd", "native NDArray extension required")
    def test_ndarray_devices(self):
        for device in (ndl.cpu_numpy(), ndl.cpu()):
            v, x = tensor(self.v, device=device), tensor(self.x, device=device)
            y = ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x)
            y.backward(tensor(self.g, device=device))
            self.assertEqual(y.device, device)
            self.assertEqual(v.grad.device, device)
            self.assertEqual(x.grad.device, device)
            np.testing.assert_allclose(y.numpy(), self.run_op()[2].numpy(), atol=1e-6)
        with self.assertRaisesRegex(ValueError, "device"):
            ndl.sparse_matmul(ndl.SparseMatrix(self.layout, tensor(self.v, device=ndl.cpu())),
                              tensor(self.x, device=ndl.cpu_numpy()))
        if ndl.cuda().enabled():
            with self.assertRaisesRegex(ValueError, "CPU"):
                ndl.SparseMatrix(self.layout, tensor(self.v, device=ndl.cuda()))

    def test_backend_dispatch(self):
        # Exercise all three kernel calls without requiring the other owner's extension.
        calls = []
        def kernel(name):
            def run(a, x, y):
                calls.append(name)
                return ops_sparse._reference(name, a, x, y)
            return run
        module = SimpleNamespace(**{name: kernel(name) for name in
                                    ("spmm", "spmm_transpose", "sampled_dot")})
        with patch.object(ops_sparse, "import_module", return_value=module):
            v, x = tensor(self.v), tensor(self.x)
            y = ndl.sparse_matmul(ndl.SparseMatrix(self.layout, v), x, backend="cpu")
            y.backward(tensor(self.g))
        self.assertEqual(calls, ["spmm", "sampled_dot", "spmm_transpose"])
        missing = ModuleNotFoundError("missing sparse backend", name="needle.backend_ndarray.sparse")
        with patch.object(ops_sparse, "import_module", side_effect=missing):
            with self.assertRaisesRegex(NotImplementedError, "cpu"):
                ndl.sparse_matmul(ndl.SparseMatrix(self.layout, tensor(self.v)), tensor(self.x), backend="cpu")


if __name__ == "__main__":
    unittest.main()

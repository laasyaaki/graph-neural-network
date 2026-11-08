"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return (
            out_grad * b * power(a, b - 1),
            out_grad * power(a, b) * log(a),
        )


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (
            out_grad / rhs,
            -out_grad * lhs / power_scalar(rhs, 2),
        )


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        return array_api.swapaxes(a, self.axes[0], self.axes[1])

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape

        extra_dims = len(output_shape) - len(input_shape)
        axes = list(range(extra_dims))

        for i, dim in enumerate(input_shape):
            if dim == 1 and output_shape[i + extra_dims] != 1:
                axes.append(i + extra_dims)

        grad = out_grad
        if axes:
            grad = summation(grad, tuple(axes))

        return reshape(grad, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape

        if self.axes is None:
            reshape_shape = tuple(1 for _ in input_shape)
        else:
            axes = self.axes
            if isinstance(axes, int):
                axes = (axes,)

            axes = tuple(
                axis if axis >= 0 else axis + len(input_shape)
                for axis in axes
            )

            reshape_shape = list(input_shape)
            for axis in axes:
                reshape_shape[axis] = 1
            reshape_shape = tuple(reshape_shape)

        return broadcast_to(reshape(out_grad, reshape_shape), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)

        if lhs_grad.shape != lhs.shape:
            extra = len(lhs_grad.shape) - len(lhs.shape)
            axes = list(range(extra))

            for i, dim in enumerate(lhs.shape):
                if dim == 1 and lhs_grad.shape[i + extra] != 1:
                    axes.append(i + extra)

            if axes:
                lhs_grad = summation(lhs_grad, tuple(axes))

            lhs_grad = reshape(lhs_grad, lhs.shape)

        if rhs_grad.shape != rhs.shape:
            extra = len(rhs_grad.shape) - len(rhs.shape)
            axes = list(range(extra))

            for i, dim in enumerate(rhs.shape):
                if dim == 1 and rhs_grad.shape[i + extra] != 1:
                    axes.append(i + extra)

            if axes:
                rhs_grad = summation(rhs_grad, tuple(axes))

            rhs_grad = reshape(rhs_grad, rhs.shape)

        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * Tensor(node.realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)
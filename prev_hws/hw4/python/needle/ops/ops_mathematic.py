"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


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
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return (a.log() * b).exp() if BACKEND == "nd" else a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return (
            out_grad * b * power(a, b - 1),
            out_grad * power(a, b) * log(a),
        )
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.scalar == 0:
            return out_grad * 0
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (
            out_grad / rhs,
            -out_grad * lhs / power_scalar(rhs, 2),
        )
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = self.axes if self.axes is not None else (-1, -2)
        order = list(range(len(a.shape)))
        i, j = axes[0] % len(order), axes[1] % len(order)
        order[i], order[j] = order[j], order[i]
        return a.permute(tuple(order)) if BACKEND == "nd" else numpy.transpose(a, order)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape) if BACKEND == "nd" else array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND != "nd":
            return array_api.sum(a, axis=self.axes)
        if self.axes is None:
            return a.sum().reshape(())
        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        result = a
        for axis in sorted((axis % len(a.shape) for axis in axes), reverse=True):
            result = result.sum(axis=axis)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        if BACKEND != "nd":
            return array_api.matmul(a, b)
        if len(a.shape) == len(b.shape) == 2:
            return a @ b
        shape = numpy.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        aa = a.broadcast_to(shape + a.shape[-2:]).compact()
        bb = b.broadcast_to(shape + b.shape[-2:]).compact()
        out = array_api.empty(shape + (a.shape[-2], b.shape[-1]), device=a.device)
        for idx in numpy.ndindex(shape):
            out[idx + (slice(None), slice(None))] = aa[idx + (slice(None), slice(None))].compact().reshape(a.shape[-2:]) @ bb[idx + (slice(None), slice(None))].compact().reshape(b.shape[-2:])
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.realize_cached_data() > 0, device=node.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        t = tanh(node.inputs[0])
        return out_grad * (-(t * t) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        arrays = args
        axis = self.axis % (len(arrays[0].shape) + 1)
        shape = list(arrays[0].shape)
        shape.insert(axis, len(arrays))
        out = array_api.empty(tuple(shape), device=arrays[0].device) if BACKEND == "nd" else numpy.empty(shape, dtype=arrays[0].dtype)
        for i, a in enumerate(arrays):
            sl = [slice(None)] * len(shape)
            sl[axis] = slice(i, i + 1)
            out[tuple(sl)] = a.compact().reshape(tuple(shape[:axis] + [1] + shape[axis + 1:])) if BACKEND == "nd" else numpy.expand_dims(a, axis)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = A.shape
        axis = self.axis % len(shape)
        out = []
        for i in range(shape[axis]):
            sl = [slice(None)] * len(shape)
            sl[axis] = slice(i, i + 1)
            part = A[tuple(sl)]
            out.append(part.compact().reshape(shape[:axis] + shape[axis + 1:]) if BACKEND == "nd" else numpy.squeeze(part, axis=axis))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack([out_grad[i] for i in range(len(out_grad))], self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes) if BACKEND == "nd" else numpy.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        shape = list(a.shape)
        for axis in range(len(shape)):
            if axis in self.axes:
                shape[axis] *= self.dilation + 1
        out = array_api.empty(tuple(shape), device=a.device) if BACKEND == "nd" else numpy.zeros(shape, dtype=a.dtype)
        if BACKEND == "nd":
            out.fill(0)
        sl = tuple(slice(None, None, self.dilation + 1) if i in self.axes else slice(None) for i in range(len(shape)))
        out[sl] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        sl = tuple(slice(None, None, self.dilation + 1) if i in self.axes else slice(None) for i in range(len(a.shape)))
        out = a[sl]
        return out.compact() if BACKEND == "nd" else out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        n, h, w, cin = A.shape
        kh, kw, _, cout = B.shape
        p, st = self.padding, self.stride
        if p:
            A = A.pad(((0, 0), (p, p), (p, p), (0, 0))) if BACKEND == "nd" else numpy.pad(A, ((0, 0), (p, p), (p, p), (0, 0)))
        oh = (h + 2 * p - kh) // st + 1
        ow = (w + 2 * p - kw) // st + 1
        strides = A.strides
        if BACKEND == "nd":
            patches = A.as_strided((n, oh, ow, kh, kw, cin), (strides[0], strides[1] * st, strides[2] * st, strides[1], strides[2], strides[3]))
            return (patches.compact().reshape((n * oh * ow, kh * kw * cin)) @ B.compact().reshape((kh * kw * cin, cout))).reshape((n, oh, ow, cout))
        patches = numpy.lib.stride_tricks.as_strided(A, (n, oh, ow, kh, kw, cin), (strides[0], strides[1] * st, strides[2] * st, strides[1], strides[2], strides[3]))
        return (patches.reshape((n * oh * ow, kh * kw * cin)) @ B.reshape((kh * kw * cin, cout))).reshape((n, oh, ow, cout))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        kh, kw, cin, cout = B.shape
        G = out_grad
        if self.stride > 1:
            G = dilate(G, (1, 2), self.stride - 1)
            gh, gw = G.shape[1:3]
            expected_h = A.shape[1] + 2 * self.padding - kh + 1
            expected_w = A.shape[2] + 2 * self.padding - kw + 1
            if (gh, gw) != (expected_h, expected_w):
                raw = G.realize_cached_data()
                raw = raw[:, :min(gh, expected_h), :min(gw, expected_w), :]
                pad_h, pad_w = max(0, expected_h - gh), max(0, expected_w - gw)
                raw = raw.pad(((0, 0), (0, pad_h), (0, pad_w), (0, 0))) if BACKEND == "nd" else numpy.pad(raw, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
                G = Tensor(raw, device=G.device)
        BT = flip(B, (0, 1)).transpose((2, 3))
        dA = conv(G, BT, stride=1, padding=kh - 1 - self.padding)
        AT = A.transpose((0, 3))
        GT = G.transpose((0, 1)).transpose((1, 2))
        dB = conv(AT, GT, stride=1, padding=self.padding).transpose((0, 1)).transpose((1, 2))
        return dA, dB
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



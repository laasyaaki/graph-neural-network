from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=-1, keepdims=True) if BACKEND == "nd" else array_api.max(Z, axis=-1, keepdims=True)

        shifted_Z = Z - max_Z

        log_sum_exp = array_api.log(
            array_api.sum(
                array_api.exp(shifted_Z),
                axis=-1,
                keepdims=True
            )
        )

        return shifted_Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        sum_grad = summation(out_grad, axes=(-1,))

        shape = Z.shape[:-1] + (1,)

        sum_grad = broadcast_to(
            reshape(sum_grad, shape), Z.shape
        )

        return out_grad - exp(node) * sum_grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axes = tuple(range(len(Z.shape))) if self.axes is None else ((self.axes,) if isinstance(self.axes, int) else self.axes)
        axes = tuple(axis % len(Z.shape) for axis in axes)
        maximum = Z
        for axis in axes:
            maximum = maximum.max(axis=axis, keepdims=True) if BACKEND == "nd" else array_api.max(maximum, axis=axis, keepdims=True)
        shifted = Z - array_api.broadcast_to(maximum, Z.shape)
        total = array_api.exp(shifted)
        for axis in sorted(axes, reverse=True):
            total = array_api.sum(total, axis=axis)
        out_shape = tuple(size for i, size in enumerate(Z.shape) if i not in axes)
        total = array_api.reshape(total.compact() if BACKEND == "nd" else total, out_shape)
        return array_api.log(total) + array_api.reshape(maximum.compact() if BACKEND == "nd" else maximum, out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        axes = tuple(a % len(Z.shape) for a in axes)

        shape = list(Z.shape)
        for axis in axes:
            shape[axis] = 1

        shape = tuple(shape)

        logsumexp = broadcast_to(
            reshape(node, shape), Z.shape
        )

        grad = broadcast_to(
            reshape(out_grad, shape), Z.shape
        )

        return grad * exp(Z - logsumexp)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)

from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=-1, keepdims=True)

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
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)

        shifted_Z = Z - max_Z

        sum_exp = array_api.sum(
            array_api.exp(shifted_Z),
            axis=self.axes
        )

        return array_api.log(sum_exp) + array_api.sum(max_Z, axis=self.axes)
        
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
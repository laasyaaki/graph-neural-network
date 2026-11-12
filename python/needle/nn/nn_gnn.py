"""Graph neural network layers (see docs/sparse_contract.md)."""
from __future__ import annotations

from ..autograd import Tensor

__all__ = ["dense_aggregate"]


def dense_aggregate(graph: Tensor, features: Tensor) -> Tensor:
    """Aggregate with a dense normalized adjacency A_hat of shape (N, N)."""
    return graph @ features

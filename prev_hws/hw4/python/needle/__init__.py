from . import ops
from .ops import *
from .autograd import Tensor, cpu, all_devices

from . import init
from .init import ones, zeros, zeros_like, ones_like

from . import nn
from . import optim
from .backend_selection import *

from .sparse import SparseMatrix, sparse_matmul


def __getattr__(name):
    # Dataset imports require the native NDArray extension; load on first use.
    if name == "data":
        from importlib import import_module
        module = import_module(".data", __name__)
        globals()[name] = module
        return module
    raise AttributeError("module %r has no attribute %r" % (__name__, name))

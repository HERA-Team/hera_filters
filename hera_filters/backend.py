# backend.py
import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse.linalg import LinearOperator as sp_LinearOperator

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    from cupyx.scipy.sparse.linalg import LinearOperator as cpx_LinearOperator
    HAS_CUPY = True
except ImportError:
    cp = None
    cpx_sparse = None
    cpx_LinearOperator = None
    HAS_CUPY = False

arraylib = np
sparse_backend = sp_sparse
linear_operator = sp_LinearOperator

def set_backend(backend="numpy"):
    """
    Set the backend for array operations.
    """
    global arraylib, sparse_backend, linear_operator
    if backend == "numpy":
        arraylib = np
        sparse_backend = sp_sparse
        linear_operator = sp_LinearOperator
    elif backend == "cupy":
        if not HAS_CUPY:
            raise ImportError("CuPy not installed.")
        arraylib = cp
        sparse_backend = cpx_sparse
        linear_operator = cpx_LinearOperator
    else:
        raise ValueError(f"Unknown backend: {backend}")
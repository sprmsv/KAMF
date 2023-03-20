from typing import Union

import scipy as sp
import numpy as np


SparseMatrix = Union[
    sp.sparse.csc_matrix,
    sp.sparse.csr_matrix,
]
Matrix = Union[
    np.ndarray,
    SparseMatrix,
]

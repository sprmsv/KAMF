import numpy as np

import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import linalg as spla
from scipy import linalg as la

from src.definitions import Matrix, SparseMatrix

def get_FD_matrix(n: int, d: int, dtype: np.dtype = np.float64, format: str = 'csc') -> SparseMatrix:
    """
    Returns the stiffness matrix for finite difference discretization
    of the Poisson's equation with a uniform grid.

    Args:
        n: Number of interior grid points.
        d: Dimension of the problem.
        dtype: Data type. Defaults to np.float64.
        format: _description_. Defaults to 'csc'.

    Returns:
        SparseArray: The stiffness matrix.
    """

    # Construct blocks
    I = sparse.identity(n=n, dtype=dtype, format=format)
    E = sparse.eye(m=n, k=1, dtype=dtype, format=format)\
        + sparse.eye(m=n, k=-1, dtype=dtype, format=format)
    K = -2 * I + E

    # Construct A
    if d == 1:
        A = K
    elif d == 2:
        A = sparse.kron(I, K) + sparse.kron(K, I)
    elif d == 3:
        A = sparse.kron(I, sparse.kron(I, K)) + sparse.kron(K, sparse.kron(I, I))

    # Change format
    if format == 'csc':
        A = sparse.csc_matrix(A)
    elif format == ' csr':
        A = sparse.csr_matrix(A)
    else:
        raise Exception('Format not supported.')

    return A

def relative_error(approximation: Matrix, exact: Matrix) -> float:
    """
    Returns the relative error of a phi-function approximation against the exact values.
    """

    # Convert to np.ndarray
    if sparse.issparse(exact):
        exact = exact.toarray()
    if sparse.issparse(approximation):
        exact = approximation.toarray()

    # Get the error vector
    err = approximation - exact

    return np.linalg.norm(err) / np.linalg.norm(exact)

# NOTE: Old relative error, like in Niesen
def relative_error_(approximation: Matrix, exact: Matrix) -> float:
    """
    Returns the relative error of a phi-function approximation against the exact values.
    """

    # Convert to np.ndarray
    if sparse.issparse(exact):
        exact = exact.toarray()
    if sparse.issparse(approximation):
        exact = approximation.toarray()

    tol = 10 * np.finfo(exact.dtype).resolution  # CHECK: What tolerance is safe?
    nz = np.where(exact > tol)
    relerr = (approximation[nz] - exact[nz]) / exact[nz]

    return np.linalg.norm(relerr)

def multiply_by_inverse(A: Matrix, B: Matrix, mode: str = 'left') -> np.ndarray:
    """
    Multiplies B by A^{-1} from left or right by solving
    the corresponding linear systems of equations.
    Similar to A \ B in Matlab.

    - left: Returns A^{-1} B
    - left: Returns B A^{-1}
    """

    # Set the solve method based on the type of the input
    if sp.sparse.issparse(A) and sp.sparse.issparse(B):
        solve = spla.spsolve
        sparse = True
    elif (not sp.sparse.issparse(A)) and (not sp.sparse.issparse(B)):
        solve = la.solve
        sparse = False
    else:
        raise Exception('Both matrices should be either sparse or dense.')

    # Solve the system
    if mode == 'left':
        C = solve(A, B)
    elif mode == 'right':
        C = solve(A.T, B.T).T
    else:
        raise Exception('Mode not supported.')

    return C

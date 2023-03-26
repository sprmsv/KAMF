import numpy as np

import scipy as sp
import scipy.sparse as sps
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
    I = sps.identity(n=n, dtype=dtype, format=format)
    E = sps.eye(m=n, k=1, dtype=dtype, format=format)\
        + sps.eye(m=n, k=-1, dtype=dtype, format=format)
    K = -2 * I + E

    # Construct A
    if d == 1:
        A = K
    elif d == 2:
        A = sps.kron(I, K) + sps.kron(K, I)
    elif d == 3:
        A = sps.kron(I, sps.kron(I, K)) + sps.kron(I, sps.kron(K, I)) + sps.kron(K, sps.kron(I, I))

    # Change format
    if format == 'csc':
        A = sps.csc_matrix(A)
    elif format == ' csr':
        A = sps.csr_matrix(A)
    else:
        raise Exception('Format not supported.')

    return ((n + 1) ** 2) * A

def relative_error(approximation: Matrix, exact: Matrix) -> float:
    """
    Returns the relative error of a phi-function approximation against the exact values.
    """

    # Convert to np.ndarray
    if sps.issparse(exact):
        exact = exact.toarray()
    if sps.issparse(approximation):
        exact = approximation.toarray()

    # Get the error vector
    err = approximation - exact

    return np.linalg.norm(err) / np.linalg.norm(exact)

def multiply_by_inverse(A: Matrix, B: Matrix, mode: str = 'left') -> np.ndarray:
    """
    Multiplies B by A^{-1} from left or right by solving
    the corresponding linear systems of equations.
    Similar to A \ B in Matlab.

    - left: Returns A^{-1} B
    - left: Returns B A^{-1}
    """

    # Set the solve method based on the type of the input
    if sps.issparse(A) and sps.issparse(B):
        solve = spla.spsolve
        sparse = True
    elif (not sps.issparse(A)) and (not sps.issparse(B)):
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

def calculate_relgap(A: SparseMatrix) -> float:
    """Calculates the relative gap of a given matrix."""

    sa = sps.linalg.eigsh(A, k=2, which='SA')[0]
    lam_1 = sa[0]
    lam_2 = sa[1]
    lam_n = sps.linalg.eigsh(A, k=1, which='LA')[0]
    return float((lam_2 - lam_1) / (lam_n - lam_1))

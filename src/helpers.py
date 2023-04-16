import numpy as np

import scipy as sp
import scipy.sparse as sps
from scipy.sparse import linalg as spla
from scipy import linalg as la
import matplotlib.pyplot as plt

from src.definitions import Matrix, SparseMatrix

def get_FD_matrix(n: int, d: int, scale: bool = False, dtype: np.dtype = np.float64, format: str = 'csc') -> SparseMatrix:
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

    # Scale if specified
    if scale:
        A *= (n + 1) ** 2

    return A

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

def calculate_relgap(
        A: SparseMatrix,
        lam_min1: float = None, lam_min2: float = None, lam_max1: float = None, lam_max2: float = None,
    ) -> float:
    """Calculates the relative gap of a given Hermitian matrix."""

    if (not lam_min1) or (not lam_min2):
        sa = spla.eigsh(A, k=2, which='SA')[0]
    if not lam_min1:
        lam_min1 = sa[0]
    if not lam_min2:
        lam_min2 = sa[1]
    if (not lam_max1) or (not lam_max2):
        la_ = spla.eigsh(A, k=2, which='LA')[0]
    if not lam_max2:
        lam_max2 = la_[0]
    if not lam_max1:
        lam_max1 = la_[1]

    relgap_l = float((lam_min2 - lam_min1) / (lam_max1 - lam_min1))
    relgap_r = float((lam_max1 - lam_max2) / (lam_max1 - lam_min1))

    return relgap_l, relgap_r

def plot_eigenvalues(As: list[Matrix], legends: list[str] = None, xticks: list = None, range_: tuple = None) -> None:
    """Plots the eigenvalues of several matrices."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for idx, A in enumerate(As):
        if sps.issparse(A):
            A = A.toarray()
        eigs = la.eigvalsh(A)
        ax.scatter(x=eigs, y=[-idx]*len(eigs), s=10, label=(legends[idx] if legends else None))
    ax.set(
        ylim=[-len(As), +1],
        xlabel='$\\lambda$',
        yticks=[],
    )
    if xticks:
        ax.set(xticks = xticks)
    elif range_:
        ax.set(xticks = np.linspace(*range_, 5))
    if legends:
        ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))

def spectral_scale(A: SparseMatrix, a: float, b: float, eigs: tuple = None) -> SparseMatrix:
    """Scales the spectral interval of a Hermitian matrix to a pre-specified domain [a, b]."""

    if not eigs:
        lmin = sps.linalg.eigsh(A, k=1, which='SA', return_eigenvectors=False).item()
        lmax = sps.linalg.eigsh(A, k=1, which='LA', return_eigenvectors=False).item()
    else:
        lmin, lmax = eigs
    I = sps.identity(n=A.shape[0], dtype=A.dtype, format=A.format)

    return (A - lmin * I) * (b - a) / (lmax - lmin) + (a * I)

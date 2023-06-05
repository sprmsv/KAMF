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
    of the Laplace operator with a uniform grid.

    Args:
        n: Number of interior grid points.
        d: Dimension of the problem.
        dtype: Data type. Defaults to np.float64.
        format: The sparse format. Defaults to 'csc'.

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

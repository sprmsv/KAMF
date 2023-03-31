import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.phi import Phi
from src.definitions import Matrix, SparseMatrix
from src.helpers import relative_error


def check_arnoldi(A: Matrix, v: np.ndarray, ms: list = None, title: str = None) -> plt.Figure:
    """
    Plots two measures for evaluating the implemented Arnoldi method.

    - Orthogonality error: $\left\| V_m^*V_m - I \right\|_{2}$
    - Projection error: $\left\| V_m^* A V_m - H_m \right\|_{max}$

    Args:
        A (Matrix): The matrix for building the Krylov subspace.
        v (np.ndarray): The vector for building the Krylov subspace.
        ms: List of Krylov subspace dimensions. Defaults to None.
        title (str, optional): Title of the figure. Defaults to None.

    Returns:
        plt.Figure: Figure containing the plots.
    """

    if not ms:
        ms = [int(m) for m in np.linspace(1, 100, 51)]

    data = {
        'm': [],
        'orthogonality': [],
        'projection': [],
        're-orthogonalization': [],
    }
    for m in ms:
        V_m, H_m = Phi.standardarnoldi(A=A, v=v, m=m, ro=True)
        data['m'].append(m)
        data['orthogonality'].append(np.linalg.norm(V_m.conjugate().T @ V_m - np.eye(m, dtype=A.dtype)))
        data['projection'].append(np.linalg.norm(V_m.conjugate().T @ A @ V_m - H_m))
        data['re-orthogonalization'].append(True)
        V_m, H_m = Phi.standardarnoldi(A=A, v=v, m=m, ro=False)
        data['m'].append(m)
        data['orthogonality'].append(np.linalg.norm(V_m.conjugate().T @ V_m - np.eye(m, dtype=A.dtype)))
        data['projection'].append(np.linalg.norm(V_m.conjugate().T @ A @ V_m - H_m))
        data['re-orthogonalization'].append(False)

    # Plot the errors
    plt.figure()
    fig, axs = plt.subplots(2, figsize=(8,8))
    sns.lineplot(
        data=data,
        x='m',
        y='orthogonality',
        hue='re-orthogonalization',
        ax=axs[0],
    )
    axs[0].set(
        ylabel='Orthogonality error $\left\| V_m^* V_m - I_m \\right\|$',
        yscale='log',
        # xlabel='Dimension of the Krylov subspace $m$',
    )
    sns.lineplot(
        data=data,
        x='m',
        y='projection',
        hue='re-orthogonalization',
        ax=axs[1],
    )
    axs[1].set(
        ylabel='Projection error $\left\| V_m^* A V_m - H_m \\right\|$',
        yscale='log',
        xlabel='Dimension of the Krylov subspace $m$',
    )

    if title:
        fig.suptitle(title)

def approximation_convergence(
        A: SparseMatrix,
        v: np.ndarray,
        m_exact: int,
        ps: list[int] = [0, 1, 2],
        ms: list[int] = [int(m) for m in np.linspace(1, 100, 51)],
) -> dict[str, list]:
    """Gets convergence results of the approximation for multiple phi-functions and plots them."""

    data = {'p': [], 'm': [], 'method': [], 'err': []}
    for p in ps:
        # Create the phi-function
        phi = Phi(p=p)

        # Get the reference evaluations
        exact_PA = phi.standardkrylov(A=A, v=v, m=m_exact)
        poles = np.array([-m_exact / (2 ** .5)] * (m_exact-1) + [np.inf])
        exact_RA = phi.rationalkrylov(A=A, v=v, m=m_exact, poles=poles)

        # get the Krylov subspace method approximation
        for m in ms:
            # Get polynomial krylov approximation error
            data['p'].append(p)
            data['m'].append(m)
            data['method'].append('PA')
            krylov = phi.standardkrylov(A=A, v=v, m=m)
            err = relative_error(approximation=krylov, exact=exact_PA)
            data['err'].append(err)

            # Get rational krylov approximation error
            data['p'].append(p)
            data['m'].append(m)
            data['method'].append('RA')
            poles = np.array([-m / (2 ** .5)] * (m-1) + [np.inf])
            krylov = phi.rationalkrylov(A=A, v=v, m=m, poles=poles)
            err = relative_error(approximation=krylov, exact=exact_RA)
            data['err'].append(err)

    return data

def get_bound_taylor(ps: list, ms: list, alpha: float, vnorm: float = 1):

    l = len(ms)
    ms = np.array([int(m) for m in ms])
    data = {'p': [], 'm': [], 'method': [], 'err': []}
    for p in ps:
        bound = (2 * vnorm * (alpha ** ms)
                / np.array([np.math.factorial(m + p) for m in ms], dtype=np.float64))
        data['p'].extend([p] * l)
        data['m'].extend(ms.tolist())
        data['method'].extend(['PA'] * l)
        data['err'].extend(bound.tolist())

    return data

def get_bound_chebyshev(ps: list, ms: list, alpha: float, vnorm: float = 1):
    """
    Estimates for p's other than 1 are not valid.
    """

    l = len(ms)
    ms = np.array(ms)
    data = {'p': [], 'm': [], 'method': [], 'err': []}
    for p in ps:
        bound_i = (
            vnorm * (5 * (alpha ** 2)) / (ms ** 3)
            * np.exp(- 4 * (ms ** 2) / (5 * alpha))
        )
        bound = (
            vnorm * 64 / (12 * ms - 5 * alpha)
            * ((np.e * alpha / (4 * ms + 2 * alpha)) ** ms)
        )
        bound[np.where(ms < (alpha / 2))] = bound_i[np.where(ms < (alpha / 2))]
        # bound[np.where(ms < np.sqrt(alpha))] = np.nan
        data['p'].extend([p] * l)
        data['m'].extend(ms.tolist())
        data['method'].extend(['PA'] * l)
        data['err'].extend(bound.tolist())

    return data

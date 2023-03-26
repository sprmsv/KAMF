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
        V_m, H_m = Phi.arnoldi(A=A, v=v, m=m, ro=True)
        data['m'].append(m)
        data['orthogonality'].append(np.linalg.norm(V_m.conjugate().T @ V_m - np.eye(m, dtype=A.dtype)))
        data['projection'].append(np.linalg.norm(V_m.conjugate().T @ A @ V_m - H_m))
        data['re-orthogonalization'].append(True)
        V_m, H_m = Phi.arnoldi(A=A, v=v, m=m, ro=False)
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

def study_approximation_convergence(
        A: SparseMatrix,
        v: np.ndarray,
        ps: list[int] = [0, 1, 2],
        ms: list[int] = [int(m) for m in np.linspace(1, 100, 51)],
        title: str = None,
        m_exact: int = None,
) -> None:
    """Gets convergence results of the approximation for multiple phi-functions and plots them."""

    dfs = []
    for p in ps:
        # Create the phi-function
        phi = Phi(p=p)

        # Get the true evaluation
        if not m_exact:
            exact = phi.recursive(A=A, v=v)
        else:
            exact = phi.krylovsubspace(A=A, v=v, m=m_exact)

        # get the Krylov subspace method approximation
        errs = []
        for m in ms:
            krylov = phi.krylovsubspace(A=A, v=v, m=m)
            errs.append(relative_error(approximation=krylov, exact=exact))

        dfs.append(
            pd.DataFrame({
                '$p$': [p] * len(ms),
                '$m$': ms,
                'Relative error': errs,
            })
        )

    # Plot the errors
    data = pd.concat(objs=dfs, axis=0)
    sns.relplot(
        data=data,
        x='$m$',
        y='Relative error',
        hue='$p$',
        kind='line',
        height=5,
        aspect=1.5,
    )
    plt.yscale('log')
    plt.ylim([1e-16, 1e+01])
    if title:
        plt.title(title)
    plt.xlabel('Dimension of the Krylov subspace ($m$)')
    plt.ylabel('Relative error of $\\varphi_p(A)v$')

    return data

def study_approximation_convergence_with_ro(
        A: SparseMatrix, v: np.ndarray,
        ps: list[int] = [1], ms: list = None,
        m_exact: int = None,
) -> None:
    """
    Plots the convergence of the approximation once with re-orthogonalization and once without it.
    # NOTE: It did not have any significant difference.
    """

    dfs = []
    for p in ps:
        # Create the phi-function
        phi = Phi(p=p)

        # Get the true evaluation
        if not m_exact:
            exact = phi.recursive(A=A, v=v)
        else:
            exact = phi.krylovsubspace(A=A, v=v, m=m_exact, ro=True)

        # get the Krylov subspace method approximation
        if not ms:
            ms = [int(m) for m in np.linspace(1, 100, 51)]

        for ro in [True, False]:
            errs = []
            for m in ms:
                krylov = phi.krylovsubspace(A=A, v=v, m=m, ro=ro)
                errs.append(relative_error(approximation=krylov, exact=exact))
            dfs.append(
                pd.DataFrame({
                    '$p$': [p] * len(ms),
                    '$m$': ms,
                    'Relative error': errs,
                    'Re-orthogonalization': ro,
                })
            )

    # Plot the errors
    data = pd.concat(objs=dfs, axis=0)
    sns.relplot(
        data=data,
        x='$m$',
        y='Relative error',
        hue='Re-orthogonalization',
        col='$p$',
        col_wrap=2,
        kind='line',
        height=5,
        aspect=1.5,
    )
    plt.yscale('log')
    plt.ylim([1e-17, 1e+01])

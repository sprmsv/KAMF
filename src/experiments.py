from time import process_time
from typing import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as sps
import seaborn as sns
from baryrat import aaa
from tqdm.notebook import tqdm_notebook as tqdm

from src.definitions import Matrix, SparseMatrix
from src.helpers import get_FD_matrix, relative_error, spectral_scale
from src.solvers import Phi, MatrixFunction, rationalarnoldi, standardarnoldi


def get_test_matrices(n: int, interval: tuple[float], eigs: dict = None):
    # Check the shape of the 2D Laplace matrix
    assert (n ** .5) % 1 == 0

    # Get the interval
    a, b = interval

    # 1D Laplace stiffness matrix
    A1 = spectral_scale(
        A=get_FD_matrix(n=n, d=1, scale=False),
        a=a, b=b,
        eigs=eigs['A1'] if eigs else None,
    )
    # 2D Laplace stiffness matrix
    A2 = spectral_scale(
        A=get_FD_matrix(n=(n ** .5), d=2, scale=False),
        a=a, b=b,
        eigs=eigs['A2'] if eigs else None,
    )
    # Diagonal uniformly distributed eigenvalues
    A3 = sps.diags(np.linspace(a, b, n))
    # Diagonal geometrically distributed eigenvalues
    b_ = b or b + np.sign(a) * 1e-16
    A4 = sps.diags(+np.geomspace(a, b_, n))
    # Diagonal reversed geometrically distributed eigenvalues
    b_ = b or b + np.sign(a) * 1e-16
    A5 = sps.diags((-np.geomspace(a, b_) + b_ + a)[::-1])

    return A1, A2, A3, A4, A5

def check_arnoldi(A: Matrix, v: np.ndarray, ms: list = None, title: str = None) -> plt.Figure:
    """
    Plots two measures for evaluating the implemented Arnoldi method.

    - Orthogonality error: $\left\| V_m^*V_m - I \\right\|_{2}$
    - Projection error: $\left\| V_m^* A V_m - H_m \\right\|_{max}$

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
        V_m, H_m = standardarnoldi(A=A, v=v, m=m, ro=True)
        data['m'].append(m)
        data['orthogonality'].append(np.linalg.norm(V_m.conjugate().T @ V_m - np.eye(m, dtype=A.dtype)))
        data['projection'].append(np.linalg.norm(V_m.conjugate().T @ A @ V_m - H_m))
        data['re-orthogonalization'].append(True)
        V_m, H_m = standardarnoldi(A=A, v=v, m=m, ro=False)
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

def get_convergence_phi(
        funcs: list[MatrixFunction],
        A: Matrix,
        v: np.ndarray,
        ks: list[int],
        mmax_PA: int = None,
        mmax_RA: int = None,
        nms: int = 30,
        poles_dir: str = './data/poles/phi',
) -> dict[str, list]:
    """Gets convergence results of the approximation for multiple phi-functions."""

    poles_dir = Path(poles_dir)
    assert poles_dir.exists()
    for f in funcs:
        assert isinstance(f, Phi)
        for k in ks:
            file = poles_dir / f'p{f.p:02d}_m{k:03d}.npy'
            assert file.exists(), file.as_posix()

    ms_PA = [int(m) for m in np.linspace(5, mmax_PA, nms)] if mmax_PA else []
    ms_RA = [int(m) for m in np.linspace(5, mmax_RA, nms)] if mmax_RA else []

    data = {'f': [], 'm': [], 'method': [], 'err': [], 'time': []}
    pbar = tqdm(total=len(funcs), desc='Matrix functions', leave=False)
    for f in funcs:
        # Refresh the progress bar
        pbar.total = len(funcs)
        pbar.refresh()
        pbar.desc = f'Matrix functions: {repr(f)}'
        pbar.refresh()

        # Create the inside progress bar
        pbar_method = tqdm(total=(len(ks)+3), desc='Methods', leave=False)

        # Get the reference evaluation
        pbar_method.desc = f'Methods: EX'
        pbar_method.refresh()
        start = process_time()
        if sps.issparse(A):
            exact = f.exact(A, v)
        else:
            exact = f.exact_dense(A, v)
        elapsed = process_time() - start
        data['f'].append(str(f))
        data['m'].append(0)
        data['method'].append('EX')
        data['err'].append(0)
        data['time'].append(elapsed)
        pbar_method.update()

        # Get PA errors
        data = get_krylov_convergence(
                f=f,
                Av=(A, v),
                exact=exact,
                name=f'PA',
                ms=ms_PA,
                data=data,
                poles=None,
                pbar=pbar_method,
        )

        # Get RA-ONES errors
        poles = np.array([1])
        data = get_krylov_convergence(
                f=f,
                Av=(A, v),
                exact=exact,
                name=f'RA-ONES',
                ms=ms_RA,
                data=data,
                poles=poles,
                pbar=pbar_method,
        )

        # Get RA-AAAk errors
        for k in ks:
            poles = np.load(poles_dir / f'p{f.p:02d}_m{k:03d}.npy')
            data =  get_krylov_convergence(
                f=f,
                Av=(A, v),
                exact=exact,
                name=f'RA-AAA{k}',
                ms=ms_RA,
                data=data,
                poles=poles,
                pbar=pbar_method,
            )

        pbar_method.close()
        pbar.update()

    pbar.close()

    return data

def get_convergence_trig(
    funcs: list[MatrixFunction],
    A: Matrix,
    v: np.ndarray,
    ks: list[int],
    interval=tuple[float],
    mmax_PA: int = None,
    mmax_RA: int = None,
    nms: int = 30,
):
    """Gets convergence results of the approximation for multiple trigonometric functions."""


    ms_PA = [int(m) for m in np.linspace(5, mmax_PA, nms)] if mmax_PA else []
    ms_RA = [int(m) for m in np.linspace(5, mmax_RA, nms)] if mmax_RA else []
    Z = np.linspace(interval[0], interval[1], 2000)

    data = {'f': [], 'm': [], 'method': [], 'err': [], 'time': []}
    pbar = tqdm(total=len(funcs), desc='Matrix functions', leave=False)
    for f in funcs:
        # Refresh the progress bar
        pbar.total = len(funcs)
        pbar.refresh()
        pbar.desc = f'Matrix functions: {repr(f)}'
        pbar.refresh()

        # Create the inside progress bar
        pbar_method = tqdm(total=(len(ks)+2), desc='Methods', leave=False)

        # Get the reference evaluation
        pbar_method.desc = f'Methods: EX'
        pbar_method.refresh()
        start = process_time()
        if sps.issparse(A):
            exact = f.exact(A, v)
        else:
            exact = f.exact_dense(A, v)
        elapsed = process_time() - start
        data['f'].append(str(f))
        data['m'].append(0)
        data['method'].append('EX')
        data['err'].append(0)
        data['time'].append(elapsed)
        pbar_method.update()

        # Get PA errors
        data = get_krylov_convergence(
            f=f,
            Av=(A, v),
            exact=exact,
            name=f'PA',
            ms=ms_PA,
            data=data,
            poles=None,
            pbar=pbar_method,
        )

        # Get RA-AAAk errors
        for k in ks:
            poles = aaa(Z=Z, F=f.scalar, mmax=(k+1), tol=-1).poles()
            data =  get_krylov_convergence(
                f=f,
                Av=(A, v),
                exact=exact,
                name=f'RA-AAA{k}',
                ms=ms_RA,
                data=data,
                poles=poles,
                pbar=pbar_method,
            )

        pbar_method.close()
        pbar.update()

    pbar.close()

    return data

def get_krylov_convergence(
        f: MatrixFunction,
        Av: tuple[Matrix, np.ndarray],
        exact: np.ndarray,
        name: str,
        ms: list[int],
        data: dict[str, list],
        poles: np.ndarray = None,
        pbar: tqdm = None,
    ) -> dict[str, list]:
    """Gets the convergence results for Krylov subspace method and writes the results to a dictionary."""

    # Update the pbar description
    if pbar:
        pbar.desc = f'Methods: {name}'
        pbar.refresh()

    # Read the inputs
    A, v = Av

    for m in ms:
        # Get the approximation
        start = process_time()
        if poles is None:
            krylov = f.standardkrylov(A=A, v=v, m=m)
        else:
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
        elapsed = process_time() - start
        err = relative_error(approximation=krylov, exact=exact)

        # Append to the data
        data['f'].append(str(f))
        data['m'].append(m)
        data['method'].append(name)
        data['err'].append(err)
        data['time'].append(elapsed)

    # Update the pbar
    if pbar:
        pbar.update()

    return data

def get_bound_taylor(ps: list, mmax: list, nms: int, alpha: float, vnorm: float = 1):

    ms = np.array([int(m) for m in np.linspace(5, mmax, nms)])
    data = {'p': [], 'm': [], 'method': [], 'err': [], 'time': []}
    for p in ps:
        logfrac = ms * np.log(alpha) - sp.special.gammaln(ms + p + 1)
        bound = 2 * vnorm * np.exp(logfrac)
        data['p'].extend([p] * nms)
        data['m'].extend(ms.tolist())
        data['method'].extend(['PA'] * nms)
        data['err'].extend(bound.tolist())
        data['time'].extend([0] * nms)

    return data

def get_bound_chebyshev(ps: list, mmax: list, nms: int, alpha: float, vnorm: float = 1) -> dict[str, list]:
    """
    Estimates for p's other than 1 are not valid.
    """

    ms = np.array([int(m) for m in np.linspace(5, mmax, nms)])
    data = {'p': [], 'm': [], 'method': [], 'err': [], 'time': []}
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
        data['p'].extend([p] * nms)
        data['m'].extend(ms.tolist())
        data['method'].extend(['PA'] * nms)
        data['err'].extend(bound.tolist())
        data['time'].extend([0] * nms)

    return data

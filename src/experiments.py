from time import process_time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def get_convergence(
        A: Matrix,
        v: np.ndarray,
        interval: tuple,
        funcs: list[MatrixFunction],
        mmax_PA: int = None,
        mmax_RA: int = None,
        nms: int = 30,
) -> dict[str, list]:
    """Gets convergence results of the approximation for multiple matrix functions."""

    ms_PA = [int(m) for m in np.linspace(5, mmax_PA, nms)] if mmax_PA else []
    ms_RA = [int(m) for m in np.linspace(5, mmax_RA, nms)] if mmax_RA else []

    data = {'f': [], 'm': [], 'method': [], 'err': [], 'time': []}
    pbar = tqdm(total=len(funcs), desc='Matrix functions', leave=False)
    for f in funcs:
        # Refresh the progress bar
        pbar.desc = f'Matrix functions: {repr(f)}'
        pbar.total = len(funcs)
        pbar.refresh()

        pbar.desc = f'Matrix functions: {repr(f)}'
        pbar.refresh()

        # Create the inside progress bar
        pbar_method = tqdm(total=7, desc='Methods', leave=False)

        # Get the reference evaluation
        pbar_method.desc = f'Methods: EX'
        pbar_method.refresh()
        data['f'].append(str(f))
        data['m'].append(np.nan)
        data['method'].append('EX')
        start = process_time()
        if sps.issparse(A):
            exact = f.exact(A, v)
        else:
            exact = f.exact_dense(A, v)
        elapsed = process_time() - start
        data['err'].append(0)
        data['time'].append(elapsed)
        pbar_method.update()

        # Get PA error
        pbar_method.desc = f'Methods: PA'
        pbar_method.refresh()
        for m in ms_PA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('PA')
            start = process_time()
            krylov = f.standardkrylov(A=A, v=v, m=m)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        # Get RA-ONES error
        pbar_method.desc = f'Methods: RA-ONES'
        pbar_method.refresh()
        for m in ms_RA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('RA-ONES')
            poles = np.array([1] * m)
            start = process_time()
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        # Get the interval and the AAA discretization
        a, b = interval
        if isinstance(f, Phi):
            Z = np.linspace(-1000, b, 1000)
        else:
            Z = np.linspace(a, b, 1000)

        # Get RA-AAA1 error
        pbar_method.desc = f'Methods: RA-AAA1'
        pbar_method.refresh()
        r = aaa(Z=Z, F=f.scalar, mmax=2, tol=-1)
        rpoles = r.poles()
        for m in ms_RA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('RA-AAA1')
            poles = np.concatenate([rpoles] * (m // len(rpoles) + 1))[:m]
            start = process_time()
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        # Get RA-AAA3 error
        pbar_method.desc = f'Methods: RA-AAA3'
        pbar_method.refresh()
        r = aaa(Z=Z, F=f.scalar, mmax=4, tol=-1)
        rpoles = r.poles()
        for m in ms_RA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('RA-AAA3')
            poles = np.concatenate([rpoles] * (m // len(rpoles) + 1))[:m]
            start = process_time()
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        # Get RA-AAA5 error
        pbar_method.desc = f'Methods: RA-AAA5'
        pbar_method.refresh()
        r = aaa(Z=Z, F=f.scalar, mmax=6, tol=-1)
        rpoles = r.poles()
        for m in ms_RA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('RA-AAA5')
            poles = np.concatenate([rpoles] * (m // len(rpoles) + 1))[:m]
            start = process_time()
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        # Get RA-AAA10 error
        pbar_method.desc = f'Methods: RA-AAA10'
        pbar_method.refresh()
        r = aaa(Z=Z, F=f.scalar, mmax=11, tol=-1)
        rpoles = r.poles()
        for m in ms_RA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('RA-AAA10')
            poles = np.concatenate([rpoles] * (m // len(rpoles) + 1))[:m]
            start = process_time()
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        # Get RA-AAAm error
        pbar_method.desc = f'Methods: RA-AAAm'
        pbar_method.refresh()
        for m in ms_RA:
            data['f'].append(str(f))
            data['m'].append(m)
            data['method'].append('RA-AAAm')
            r = aaa(Z=Z, F=f.scalar, mmax=(m + 1), tol=-1)
            poles = r.poles()
            start = process_time()
            krylov = f.rationalkrylov(A=A, v=v, m=m, poles=poles)
            elapsed = process_time() - start
            err = relative_error(approximation=krylov, exact=exact)
            data['err'].append(err)
            data['time'].append(elapsed)
        pbar_method.update()

        pbar_method.close()
        pbar.update()

    pbar.close()

    return data

def get_bound_taylor(ps: list, mmax: list, nms: int, alpha: float, vnorm: float = 1):

    ms = np.array([int(m) for m in np.linspace(5, mmax, nms)])
    data = {'p': [], 'm': [], 'method': [], 'err': [], 'time': []}
    for p in ps:
        mpps = np.ones(shape=(ms.max() + p, len(ms)), dtype=ms.dtype)
        for j, m in enumerate(ms):
            mpps[:(m + p), j] = np.arange(1, m + p + 1, dtype=ms.dtype)
        logfrac = ms * np.log(alpha) - np.sum(np.log(mpps), axis=0)
        bound = 2 * vnorm * np.exp(logfrac)
        data['p'].extend([p] * nms)
        data['m'].extend(ms.tolist())
        data['method'].extend(['PA'] * nms)
        data['err'].extend(bound.tolist())
        data['time'].extend([0] * nms)

    return data

def get_bound_chebyshev(ps: list, mmax: list, nms: int, alpha: float, vnorm: float = 1):
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

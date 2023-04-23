from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy import linalg as la
from scipy.sparse import linalg as spla

from src.helpers import multiply_by_inverse
from src.definitions import Matrix, SparseMatrix


def standardarnoldi(A: Matrix, v: np.ndarray, m: int, ro: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Arnoldi algorithm using matrix multiplications (lecture notes).
    """

    # Check dimensions
    n = len(v)
    dtype = A.dtype
    assert A.shape == (n, n)
    assert m <= n

    # Initialize
    H_m = np.zeros(shape=(m, m), dtype=dtype)
    V_m = np.zeros(shape=(n, m), dtype=dtype)

    V_m[:, 0] = v / np.linalg.norm(v)
    for j in range(0, m):
        w = A @ V_m[:, j]
        H_m[:(j+1), j] = V_m[:, :(j+1)].conjugate().T @ w
        u_hat = w - V_m[:, :(j+1)] @ H_m[:(j+1), j]

        # Reorthogonalization
        if ro:
            u_hat_norm = np.linalg.norm(u_hat)
            w_norm = np.linalg.norm(w)
            if u_hat_norm < .7 * w_norm:
                h_hat = V_m[:, :(j+1)].conjugate().T @ u_hat
                H_m[:(j+1), j] += h_hat
                u_hat -= V_m[:, :(j+1)] @ h_hat

        if j + 1 >= m:
            hmp1 = np.linalg.norm(u_hat)
            vmp1 = u_hat / hmp1
        else:
            H_m[j + 1, j] = np.linalg.norm(u_hat)
            V_m[:, j + 1] = u_hat / H_m[j + 1, j]

    return V_m, H_m

def rationalarnoldi(A: Matrix, v: np.ndarray, m: int, poles: np.ndarray, ro: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Rational Arnoldi algorithm using matrix multiplications (lecture notes).
    """

    # Check dimensions
    n = len(v)
    dtype = A.dtype if not np.iscomplexobj(poles) else poles.dtype
    assert A.shape == (n, n)
    assert m <= n

    # Initialize
    H_m = np.zeros(shape=(m, m), dtype=dtype)
    V_m = np.zeros(shape=(n, m), dtype=dtype)
    I = sps.identity(n, dtype=dtype, format=(A.format if sps.issparse(A) else None))
    solve = spla.spsolve if sps.issparse(A) else la.solve

    V_m[:, 0] = v / np.linalg.norm(v)
    for j in range(0, m):
        w = solve(
            (I - A / poles[j]),
            (A @ V_m[:, j]),
        )
        H_m[:(j+1), j] = V_m[:, :(j+1)].conjugate().T @ w
        u_hat = w - V_m[:, :(j+1)] @ H_m[:(j+1), j]

        # Reorthogonalization
        if ro:
            u_hat_norm = np.linalg.norm(u_hat)
            w_norm = np.linalg.norm(w)
            if u_hat_norm < .7 * w_norm:
                h_hat = V_m[:, :(j+1)].conjugate().T @ u_hat
                H_m[:(j+1), j] += h_hat
                u_hat -= V_m[:, :(j+1)] @ h_hat

        if j + 1 >= m:
            hmp1 = np.linalg.norm(u_hat)
            vmp1 = u_hat / hmp1
        else:
            H_m[j + 1, j] = np.linalg.norm(u_hat)
            V_m[:, j + 1] = u_hat / H_m[j + 1, j]

    return V_m, H_m, vmp1, hmp1

# TODO: Remove
class Phi_old:
    def __init__(self, p: int):
        assert isinstance(p, int)
        self.p = p

    def scalar(self, z: complex) -> complex:
        """Computes the scalar function using the recurrence relation."""

        res = np.sum([(1 / np.math.factorial(l)) * (z ** l) for l in range(self.p)], axis=0)
        phi_z = (np.exp(z) - res) / (z ** self.p)

        return phi_z

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the exact action of the phi-function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Get p
        p = self.p

        # Construct the embedded matrix
        if p == 0:
            return spla.expm_multiply(A, v)
        else:
            A_h = sps.lil_matrix(np.zeros(shape=(n+p, n+p), dtype=dtype))
            A_h[:n, :n] = A
            A_h[:n, n] = v
            A_h[n:n+p-1, n+1:n+p] = sps.identity(p-1, dtype=dtype, format=A_h.format)
            if A.format == 'csc':
                A_h = A_h.tocsc()
            elif A.format == 'csr':
                A_h = A_h.tocsr()

            # Compute the last column of the exponential of the embedded matrix
            enpp = np.zeros(shape=(n+p,), dtype=dtype)
            enpp[-1] = 1
            return spla.expm_multiply(A_h, enpp)[:n]

    # NOTE: Loses accuracy as p grows
    def recursive(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the phi-function on a vector using the recurrence relation."""

        # Check the type of A
        if not sps.issparse(A):
            raise Exception(f'The matrix A of shape {A.shape} must be sparse.')

        return self._recursive(A=A, v=v, p=self.p)

    @staticmethod
    def _recursive(A: SparseMatrix, v: np.ndarray, p: int) -> np.ndarray:
        """Computes the action of the matrix function on a vector using the recurrence relation."""

        if p == 0:
            return spla.expm_multiply(A, v)
        else:
            # TODO: You can pre-compute an LU or Cholesky decomposition of A and improve this function
            # NOTE: x = inv(A) b = inv(U) inv(L) x (call spsolve twice)
            t1 = spla.spsolve(A=A, b=Phi._recursive(A=A, v=v, p=p-1))
            t2 = spla.spsolve(A=A, b=v) / np.math.factorial(p-1)
            return t1 - t2

    def standardkrylov(self, A: SparseMatrix, v: np.ndarray, m: int, ro: bool = True) -> np.ndarray:
        """Computes the action of the phi-function on a vector using the polynomial Krylov approximation."""

        # Check types
        assert isinstance(m, int)

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'
        assert m <= n, f'{m} > {n}'
        assert m > 0, f'{m} <= 0'

        # Arnoldi method
        V_m, H_m = standardarnoldi(A=A, v=v, m=m, ro=ro)

        # Fetch p
        p = self.p

        if p == 0:
            # Define e1
            e1 = np.zeros(shape=(m,), dtype=dtype)
            e1[0] = 1
            # Calculate {\phi}_0(H_m)
            phi_H = la.expm(H_m)
            # Calculate {\phi}_0(H_m) e1
            phi_H_e1 = phi_H @ e1

        else:
            # Construct the H_{hat} matrix
            H_h = np.zeros(shape=(m+p, m+p), dtype=dtype)
            H_h[:m, :m] = H_m
            H_h[0, m] = 1
            H_h[m:m+p-1, m+1:m+p] = np.eye(p-1, dtype=dtype)
            # Calculate {\phi}_p(H_m) e_1
            phi_H_e1 = la.expm(H_h)[:m, -1]

        # Calculate the approximation of {\phi}_p(A) v
        phi_A_v = la.norm(v) * V_m @ phi_H_e1

        return phi_A_v

    def rationalkrylov(self, A: SparseMatrix, v: np.ndarray, m: int, poles: np.ndarray, ro: bool = True) -> np.ndarray:
        """Computes the action of the phi-function on a vector using the rational Krylov approximation."""

        # Check types
        assert isinstance(m, int)

        # Check dimensions
        n = len(v)
        dtype = A.dtype if not np.iscomplexobj(poles) else poles.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'
        assert m <= n, f'{m} > {n}'
        assert m > 0, f'{m} <= 0'
        assert poles.shape == (m,)

        # Rational Arnoldi method
        V_m, H_m, _, _ = rationalarnoldi(A=A, v=v, m=m, poles=poles, ro=ro)
        if poles[-1] == np.inf:
            D_m = np.diag(1 / poles)
            K_m = np.identity(m) + H_m @ D_m
            A_m = multiply_by_inverse(A=K_m, B=H_m, mode='right')
        else:
            A_m = V_m.conjugate().T @ A @ V_m

        # Fetch p
        p = self.p

        if p == 0:
            # Define e1
            e1 = np.zeros(shape=(m,), dtype=dtype)
            e1[0] = 1
            # Calculate {\phi}_0(A_m)
            phi_A = la.expm(A_m)
            # Calculate {\phi}_0(A_m) e1
            phi_A_e1 = phi_A @ e1

        else:
            # Construct the A_{hat} matrix
            A_h = np.zeros(shape=(m+p, m+p), dtype=dtype)
            A_h[:m, :m] = A_m
            A_h[0, m] = 1
            A_h[m:m+p-1, m+1:m+p] = np.eye(p-1, dtype=dtype)
            # Calculate {\phi}_p(H_m) e_1
            phi_A_e1 = la.expm(A_h)[:m, -1]

        # Calculate the approximation of {\phi}_p(A) v
        phi_A_v = la.norm(v) * V_m @ phi_A_e1

        return phi_A_v

class MatrixFunction(ABC):
    """Class for computing the action of a matrix function on a vector."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def scalar(self, z: complex) -> complex:
        """Computes the scalar function."""
        ...

    @abstractmethod
    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by converting it to a dense matrix."""
        ...

    @abstractmethod
    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by building the corresponding embedded matrix."""
        ...

    def standardkrylov(self, A: Matrix, v: np.ndarray, m: int, ro: bool = True) -> np.ndarray:
        """Computes the action of the matrix function on a vector using the polynomial Krylov approximation."""

        # Check types
        assert isinstance(m, int)

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'
        assert m <= n, f'{m} > {n}'
        assert m > 0, f'{m} <= 0'

        # Arnoldi method
        V_m, H_m = standardarnoldi(A=A, v=v, m=m, ro=ro)

        # Calculate the approximation of f(A) v
        e1 = np.zeros(shape=(m,), dtype=dtype)
        e1[0] = 1
        fAv = la.norm(v) * V_m @ self.exact_dense(A=H_m, v=e1)

        return fAv

    def rationalkrylov(self, A: Matrix, v: np.ndarray, m: int, poles: np.ndarray, ro: bool = True) -> np.ndarray:
        """Computes the action of the matrix function on a vector using the rational Krylov approximation."""

        # Check types
        assert isinstance(m, int)

        # Check dimensions
        n = len(v)
        dtype = A.dtype if not np.iscomplexobj(poles) else poles.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'
        assert m <= n, f'{m} > {n}'
        assert m > 0, f'{m} <= 0'
        assert poles.shape == (m,)

        # Rational Arnoldi method
        V_m, H_m, _, _ = rationalarnoldi(A=A, v=v, m=m, poles=poles, ro=ro)
        if poles[-1] == np.inf:
            D_m = np.diag(1 / poles)
            K_m = np.identity(m) + H_m @ D_m
            A_m = multiply_by_inverse(A=K_m, B=H_m, mode='right')
        else:
            A_m = V_m.conjugate().T @ A @ V_m


        # Calculate the approximation of f(A) v
        e1 = np.zeros(shape=(m,), dtype=dtype)
        e1[0] = 1
        fAv = la.norm(v) * V_m @ self.exact_dense(A=A_m, v=e1)

        return fAv

# TODO: Validate
class Phi(MatrixFunction):
    """Class for computing $\\varphi_{p}(tA) v$"""

    def __init__(self, p: int, t: float = 1.):
        super().__init__()
        assert isinstance(p, int)
        self.p = p
        self.t = t

    def scalar(self, z: complex) -> complex:
        """Computes the scalar function using the recurrence relation."""

        z = self.t * z
        res = np.sum([(1 / np.math.factorial(l)) * (z ** l) for l in range(self.p)], axis=0)

        return (np.exp(z) - res) / (z ** self.p)

    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the phi-function on a vector by converting it to a dense matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Convert A to dense matrixs
        if sps.issparse(A):
            A = A.toarray()

        # Fetch p
        p = self.p

        if p == 0:
            return la.expm(self.t * A) @ v

        else:
            # Construct the \hat{A} matrix
            A_h = np.zeros(shape=(n+p, n+p), dtype=A.dtype)
            A_h[:n, :n] = self.t * A
            A_h[:n, n] = v
            A_h[n:n+p-1, n+1:n+p] = np.eye(p-1, dtype=A.dtype)

            return la.expm(A_h)[:n, -1]

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the phi-function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Get p
        p = self.p

        # Construct the embedded matrix
        if p == 0:
            return spla.expm_multiply(self.t * A, v)
        else:
            A_h = sps.lil_matrix(np.zeros(shape=(n+p, n+p), dtype=dtype))
            A_h[:n, :n] = self.t * A
            A_h[:n, n] = v
            A_h[n:n+p-1, n+1:n+p] = sps.identity(p-1, dtype=dtype, format=A_h.format)
            if A.format == 'csc':
                A_h = A_h.tocsc()
            elif A.format == 'csr':
                A_h = A_h.tocsr()

            # Compute the last column of the exponential of the embedded matrix
            enpp = np.zeros(shape=(n+p,), dtype=dtype)
            enpp[-1] = 1
            return spla.expm_multiply(A_h, enpp)[:n]

    # NOTE: Loses accuracy as p grows
    def recursive(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the phi-function on a vector using the recurrence relation."""

        # Check the type of A
        if not sps.issparse(A):
            raise Exception(f'The matrix A of shape {A.shape} must be sparse.')

        return self._recursive(A=A, v=v, p=self.p)

    def _recursive(self, A: SparseMatrix, v: np.ndarray, p: int) -> np.ndarray:
        """Computes the action of the matrix function on a vector using the recurrence relation."""

        if p == 0:
            return spla.expm_multiply(self.t * A, v)
        else:
            # TODO: You can pre-compute an LU or Cholesky decomposition of A and improve this function
            # NOTE: x = inv(A) b = inv(U) inv(L) x (call spsolve twice)
            t1 = spla.spsolve(A=(self.t * A), b=Phi._recursive(A=A, v=v, p=p-1))
            t2 = spla.spsolve(A=(self.t * A), b=v) / np.math.factorial(p-1)
            return t1 - t2

    def __str__(self) -> str:
        return f'$\\varphi_{self.p}(tA)$'

    def __repr__(self) -> str:
        return f'Phi(p={self.p})'

class Cosine(MatrixFunction):
    """Class for computing $cos(tA) v$"""

    def __init__(self, t: float = 1.):
        super().__init__()
        self.t = t

    def scalar(self, z: complex) -> complex:
        return np.cos(self.t * z)

    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by converting it to a dense matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Convert A to dense matrixs
        if sps.issparse(A):
            A = A.toarray()

        return la.cosm(self.t * A) @ v

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Construct the embedded matrix
        A_h = sps.lil_matrix(np.zeros(shape=(n+1, n+1), dtype=dtype))
        A_h[:n, :n] = self.t * A
        A_h[:n, n] = self.t * A @ v
        if A.format == 'csc':
            A_h = A_h.tocsc()
        elif A.format == 'csr':
            A_h = A_h.tocsr()

        # Compute the last column of the exponential of the embedded matrix
        enpp = np.zeros(shape=(n+1,), dtype=dtype)
        enpp[-1] = 1

        if dtype == 'complex':
            return (v + .5 * (spla.expm_multiply(1j * A_h, enpp)[:n] + spla.expm_multiply(-1j * A_h, enpp)[:n]))
        else:
            return (v + spla.expm_multiply(1j * A_h, enpp)[:n].real)

    def __str__(self) -> str:
        return '$\\cos(tA)$'

    def __repr__(self) -> str:
        return f'Cosine(t={self.t:.0e})'

class CosineSqrt(MatrixFunction):
    """Class for computing $\cos(t \sqrt{A}) v$"""

    def __init__(self, t: float = 1.):
        super().__init__()
        self.t = t

    def scalar(self, z: complex) -> complex:
        return np.cos(self.t * np.sqrt(z))

    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by converting it to a dense matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Convert A to dense matrixs
        if sps.issparse(A):
            A = A.toarray()

        # Calculate the square root of A
        H = la.sqrtm(A)

        return la.cosm(self.t * H) @ v

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Calculate the square root
        if sps.issparse(A):
            H = la.sqrtm(A.toarray())
        else:
            H = la.sqrtm(A)

        # Construct the embedded matrix
        A_h = sps.lil_matrix(np.zeros(shape=(n+1, n+1), dtype=dtype))
        A_h[:n, :n] = self.t * H
        A_h[:n, n] = self.t * H @ v
        if A.format == 'csc':
            A_h = A_h.tocsc()
        elif A.format == 'csr':
            A_h = A_h.tocsr()

        # Compute the last column of the exponential of the embedded matrix
        enpp = np.zeros(shape=(n+1,), dtype=dtype)
        enpp[-1] = 1

        if dtype == 'complex':
            return (v + .5 * (spla.expm_multiply(1j * A_h, enpp)[:n] + spla.expm_multiply(-1j * A_h, enpp)[:n]))
        else:
            return (v + spla.expm_multiply(1j * A_h, enpp)[:n].real)

    def __str__(self) -> str:
        return '$\\cos(t \sqrt{A})$'

    def __repr__(self) -> str:
        return f'CosineSqrt(t={self.t:.0e})'

class Sine(MatrixFunction):
    """Class for computing $sin(tA) v$"""

    def __init__(self, t: float = 1.):
        super().__init__()
        self.t = t

    def scalar(self, z: complex) -> complex:
        return np.sin(self.t * z)

    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by converting it to a dense matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Convert A to dense matrixs
        if sps.issparse(A):
            A = A.toarray()

        return la.sinm(self.t * A) @ v

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Construct the embedded matrix
        A_h = sps.lil_matrix(np.zeros(shape=(n+1, n+1), dtype=dtype))
        A_h[:n, :n] = self.t * A
        A_h[:n, n] = self.t * A @ v
        if A.format == 'csc':
            A_h = A_h.tocsc()
        elif A.format == 'csr':
            A_h = A_h.tocsr()

        # Compute the last column of the exponential of the embedded matrix
        enpp = np.zeros(shape=(n+1,), dtype=dtype)
        enpp[-1] = 1


        if dtype == 'complex':
            return (1 / (2j)) * (spla.expm_multiply(1j * A_h, enpp)[:n] - spla.expm_multiply(-1j * A_h, enpp)[:n])
        else:
            return (spla.expm_multiply(1j * A_h, enpp)[:n].imag)

    def __str__(self) -> str:
        return '$\\sin(tA)$'

    def __repr__(self) -> str:
        return f'Sine(t={self.t:.0e})'

class Sinc(MatrixFunction):
    """Class for computing $t sinc(tA) v$"""

    def __init__(self, t: float = 1.):
        super().__init__()
        self.t = t

    def scalar(self, z: complex) -> complex:
        return self.t * (np.sin(self.t * z) / (self.t * z))

    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by converting it to a dense matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Convert A to dense matrixs
        if sps.issparse(A):
            A = A.toarray()

        return la.solve(a=A, b=la.sinm(self.t * A) @ v)

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Construct the embedded matrix
        A_h = sps.lil_matrix(np.zeros(shape=(n+1, n+1), dtype=dtype))
        A_h[:n, :n] = self.t * A
        A_h[:n, n] = v
        if A.format == 'csc':
            A_h = A_h.tocsc()
        elif A.format == 'csr':
            A_h = A_h.tocsr()

        # Compute the last column of the exponential of the embedded matrix
        enpp = np.zeros(shape=(n+1,), dtype=dtype)
        enpp[-1] = 1

        if dtype == 'complex':
            return self.t * (1 / (2j)) * (spla.expm_multiply(1j * A_h, enpp)[:n] - spla.expm_multiply(-1j * A_h, enpp)[:n])
        else:
            return self.t * (spla.expm_multiply(1j * A_h, enpp)[:n].imag)

    def __str__(self) -> str:
        return '$t \\mathrm{sinc}(tA)$'

    def __repr__(self) -> str:
        return f'Sinc(t={self.t:.0e})'

class Sinc2(MatrixFunction):
    """Class for computing $.5 t^2 sinc(.5 t A)^2 v$"""

    def __init__(self, t: float = 1.):
        super().__init__()
        self.t = t
        self.sinc = Sinc(t=t/2)

    def scalar(self, z: complex) -> complex:
        return .5 * (self.t ** 2) * ((np.sin(.5 * self.t * z) / (.5 * self.t * z)) ** 2)

    def exact_dense(self, A: Matrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by converting it to a dense matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        # Convert A to dense matrixs
        if sps.issparse(A):
            A = A.toarray()

        f = self.sinc.exact_dense

        return 2 * f(A=A, v=f(A=A, v=v))

    def exact(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector by building the corresponding embedded matrix."""

        # Check dimensions
        n = len(v)
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'

        f = self.sinc.exact

        return 2 * f(A=A, v=f(A=A, v=v))

    def __str__(self) -> str:
        return '$\\frac{t^2}{2} \\mathrm{sinc}(\\frac{t}{2} A)^2$'

    def __repr__(self) -> str:
        return f'Sinc2(t={self.t:.0e})'
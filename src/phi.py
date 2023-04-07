import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import linalg as spla
from scipy import linalg as la

from src.helpers import multiply_by_inverse
from src.definitions import Matrix, SparseMatrix


class Phi:
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
            t1 = spla.spsolve(A=A, b=Phi._recursive(A=A, v=v, p=p-1))
            t2 = spla.spsolve(A=A, b=v) / np.math.factorial(p-1)
            return t1 - t2

    def standardkrylov(self, A: SparseMatrix, v: np.ndarray, m: int, ro: bool = True) -> np.ndarray:
        """Computes the phi-function evaluation using the Arnoldi iteration and
        the method described int the Niesen's paper.
        """

        # Check types
        assert isinstance(m, int)

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'
        assert m <= n, f'{m} > {n}'
        assert m > 0, f'{m} <= 0'

        # Arnoldi method
        V_m, H_m = self.standardarnoldi(A=A, v=v, m=m, ro=ro)

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
        """Computes the phi-function evaluation using the Arnoldi iteration and
        the method described int the Niesen's paper.
        """

        # Check types
        assert isinstance(m, int)

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n), f'{A.shape} != {(n, n)}'
        assert m <= n, f'{m} > {n}'
        assert m > 0, f'{m} <= 0'
        assert poles.shape == (m,)

        # Rational Arnoldi method
        V_m, H_m, vmp1, hmp1 = self.rationalarnoldi(A=A, v=v, m=m, poles=poles, ro=ro)
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

    @staticmethod
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

    # UNUSED
    @staticmethod
    def standardarnoldi_(A: Matrix, v: np.ndarray, m: int, ro: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard Arnoldi algorithm (Niesen).
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
            w_norm = np.linalg.norm(w)
            for i in range(0, j + 1):
                H_m[i, j] = V_m[:, i].dot(w)
                w = w - (H_m[i, j] * V_m[:, i])

            # Reorthogonalization
            if ro:
                if np.linalg.norm(w) < .7 * w_norm:
                    h_hat = V_m[:, :(j+1)].conjugate().T @ w
                    H_m[:(j+1), j] += h_hat
                    w -= V_m[:, :(j+1)] @ h_hat

            if j + 1 >= m:
                hmp1 = np.linalg.norm(w)
                vmp1 = w / hmp1
            else:
                H_m[j + 1, j] = np.linalg.norm(w)
                V_m[:, j + 1] = w / H_m[j + 1, j]

        return V_m, H_m

    @staticmethod
    def rationalarnoldi(A: SparseMatrix, v: np.ndarray, m: int, poles: np.ndarray, ro: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Rational Arnoldi algorithm using matrix multiplications (lecture notes).
        """

        # Check dimensions
        n = len(v)
        dtype = A.dtype
        assert A.shape == (n, n)
        assert m <= n


        # Initialize
        H_m = np.zeros(shape=(m, m), dtype=dtype)
        V_m = np.zeros(shape=(n, m), dtype=dtype)
        I = sps.identity(n, dtype=dtype, format=A.format)

        V_m[:, 0] = v / np.linalg.norm(v)
        for j in range(0, m):
            w = spla.spsolve(
                A=(I - A / poles[j]),
                b=(A @ V_m[:, j]),
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

    def __call__(self, A, v, m):
        return self.krylov(A, v, m)

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

    def scalar(self, z: float) -> float:
        """Computes the scalar function using the recurrence relation."""

        res = np.sum([(1 / np.math.factorial(l)) * (z ** l) for l in range(self.p)], axis=0)
        phi_z = (np.exp(z) - res) / (z ** self.p)

        return phi_z

    def recursive(self, A: SparseMatrix, v: np.ndarray) -> np.ndarray:
        """Computes the action of the matrix function on a vector using the recurrence relation."""

        # Read info of A
        n = A.shape[0]
        dtype = A.dtype
        if not sps.issparse(A):
            raise Exception
        format = A.format

        # Convert v to sparse matrix
        assert len(v.shape) == 1
        v = v.reshape((-1, 1))
        if format == 'csc':
            v = sps.csc_matrix(v)
        elif format == 'csr':
            v = sps.csr_matrix(v)
        else:
            raise Exception

        # Compute $\exp(A) v$ and return if $p=0$
        eAv = spla.expm_multiply(A, v)
        if not self.p:
            return eAv.toarray().reshape(-1)

        # Compute $\varphi(A) v$ if $p>0$
        if format == 'csc':
            res = sps.csc_matrix((n, n), dtype=dtype)
        elif format == 'csr':
            res = sps.csr_matrix((n, n), dtype=dtype)
        A_powered = sps.eye(n, dtype=dtype, format=format)
        for k in range(self.p):
            res = res + A_powered / np.math.factorial(k)
            A_powered = A @ A_powered

        phiAv = multiply_by_inverse(
            A=(A ** self.p).toarray(),
            B=(eAv - res @ v).toarray(),
            mode='left',
        )

        return phiAv.reshape(-1)

    def krylovsubspace(self, A: SparseMatrix, v: np.ndarray, m: int, ro: bool = True) -> np.ndarray:
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
        V_m, H_m = self.arnoldi(A=A, v=v, m=m, ro=ro)

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

    @staticmethod
    def arnoldi(A: Matrix, v: np.ndarray, m: int, ro: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Arnoldi algorithm (Niesen).
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
                break
            H_m[j + 1, j] = np.linalg.norm(w)
            V_m[:, j + 1] = w / H_m[j + 1, j]

        return V_m, H_m

    @staticmethod
    def arnoldi_(A: np.ndarray, v: np.ndarray, m: int, ro: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Arnoldi algorithm using matrix multiplications (lecture notes).
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
                break
            H_m[j + 1, j] = np.linalg.norm(u_hat)
            V_m[:, j + 1] = u_hat / H_m[j + 1, j]

        return V_m, H_m

    def __call__(self, A, v, m):
        return self.krylov(A, v, m)

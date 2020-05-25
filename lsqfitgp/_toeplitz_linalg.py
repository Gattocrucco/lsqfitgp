# This code originally copied from the TOEPLITZ_CHOLESKY code by John Burkardt,
# released under the LGPL.
# people.sc.fsu.edu/~jburkardt/py_src/toeplitz_cholesky/toeplitz_cholesky.html

import numpy as np

def cholesky_toeplitz(t):
    t = np.asanyarray(t)
    assert len(t.shape) == 1
    if len(t) == 0:
        return np.empty((0, 0), dtype=t.dtype)
    if t[0] <= 0:
        msg = '1-th leading minor is not positive definite'
        raise np.linalg.LinAlgError(msg)

    n = len(t)
    L = np.zeros((n, n), order='F', dtype=t.dtype)
    L[:, 0] = t
    g = np.stack([np.roll(t, 1), t])

    for i in range(1, n):
        assert g[0, i] > 0
        rho = -g[1, i] / g[0, i]
        if np.abs(rho) >= 1:
            msg = '{}-th leading minor is not positive definite'.format(i + 1)
            raise np.linalg.LinAlgError(msg)
        gamma = np.sqrt((1 - rho) * (1 + rho))
        g[:, i:] += g[::-1, i:] * rho
        g[:, i:] /= gamma
        L[i:, i] = g[0, i:]
        g[0, i:] = np.roll(g[0, i:], 1)

    return L

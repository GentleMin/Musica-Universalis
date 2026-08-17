# -*- coding: utf-8 -*-

import numpy as np
import warnings
from scipy.sparse import linalg as spla


def single_eig(A, B, target, nev=5, **kwargs):
    """ Compute the eigenvector, for given eigenvalue """

    C = A - target * B
    clu = spla.splu(C)

    def bmx(x):
        return clu.solve(B.dot(x))

    D = spla.LinearOperator(dtype=np.complex128, shape=np.shape(A), matvec=bmx)

    try:
        evals, u = spla.eigs(D, k=nev, which='LM', **kwargs)
    except spla.ArpackNoConvergence as err_arpack:
        warnings.warn('ARPACKNoConvergence triggered...')
        evals = err_arpack.eigenvalues
        u = err_arpack.eigenvectors

    if evals.size > 0:
        return 1.0 / evals + target, u
    else:
        return np.nan*(1+1j), None


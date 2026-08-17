# -*- coding: utf-8 -*-

import numpy as np
import warnings
from scipy import sparse, linalg
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


def full_eig(K, M):
    """ Compute full eigenspectrum
    """
    K = sparse.csc_array(K)
    M = sparse.csc_array(M)
    KiM = spla.spsolve(K, M).toarray()
    # print(KiM.nbytes)
    # w, _ = linalg.eig(KiM.toarray())
    w = np.linalg.eigvals(KiM)
    w = 1./w

    i = np.isfinite(w) & (np.abs(w) < 1e+10) & (np.abs(w) > 1e-10)
    w = w[i]

    i = np.argsort(np.imag(w))
    w = w[i]
    return w


def full_gevp(K, M):
    """ Full spectrum of general eigenvalue problem
    """
    K = K.toarray()
    M = M.toarray()
    w = linalg.eigvals(K, b=M, overwrite_a=True)

    i = np.isfinite(w) & (np.abs(w) < 1e+10) & (np.abs(w) > 1e-10)
    w = w[i]
    i = np.argsort(np.imag(w))
    w = w[i]
    return w


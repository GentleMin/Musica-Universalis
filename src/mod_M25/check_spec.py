# -*- coding: utf-8 -*-


import os
import time
import argparse
import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as spla


def full_eig(K, M):
    """ Compute full eigenspectrum
    """
    K = sparse.csc_array(K)
    M = sparse.csc_array(M)
    KiM = spla.spsolve(K, M).toarray()
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

def load_matrices(lib_dir):
    fnames = os.listdir(lib_dir)
    M_lib = dict()
    for fname in fnames:
        matname = fname[:-4]
        M_lib[matname] = sparse.load_npz(os.path.join(lib_dir, fname))
    return M_lib


def setup_op_spin_DR(M_lib, Ro):
    B = -M_lib['mass']
    A = M_lib['coriolis'] \
        + Ro*M_lib['differential_rotation']
    if 'tau' in M_lib:
        A += M_lib['tau']
    return A, B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-Ro', type=float, default=1.0)
    args = parser.parse_args()

    M_lib = load_matrices(os.path.join(args.dir, 'ops'))
    A, B = setup_op_spin_DR(M_lib, args.Ro)
    print(
        "\n============================================================\n" +
        "                    Spectral decomposition                   \n" +
        f"    Matrix: {args.dir}; Rossby for DR: {args.Ro:.2f}\n" + 
        "------------------------------------------------------------"
    )

    save_name = os.path.join(args.dir, f'spec-dr{args.Ro:.2f}')
    if os.path.exists(save_name + '.npy'):
        print(f"\tResult already exists... aborting...", flush=True)
        return

    print(f"\t\tDecomposing system of size: {A.shape}")
    starttime = time.perf_counter()
    w = full_eig(A, B)
    elapsed = time.perf_counter() - starttime
    print(f"\t\tSpectral decomposition finished in {elapsed:.2f}s.", flush=True)

    np.save(save_name, -1j*w)
    print(f"\t\tEigenspectrum saved to {save_name}\n", flush=True)


if __name__ == "__main__":
    main()



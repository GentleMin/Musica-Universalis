# -*- coding: utf-8 -*-


import os
import numpy as np
import warnings
import time
import argparse
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


class RescalerCarrington:

    def __call__(self, v):
        return v/456.0

    def reverse(self, v):
        return v*456.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-Ro', type=float, default=1.0)
    parser.add_argument('-w0', type=float, default=0.0)
    parser.add_argument('-d0', type=float, default=0.0)
    parser.add_argument('-ws', type=float)
    parser.add_argument('-ds', type=float)
    parser.add_argument('-v', '--output-vector')
    args = parser.parse_args()

    M_lib = load_matrices(args.dir)
    A, B = setup_op_spin_DR(M_lib, args.Ro)
    print(
        "\n============================================================\n" +
        "                    Quick eigenvalue test                   \n" +
        f"    Matrix: {args.dir}; Rossby for DR: {args.Ro:.2f}\n" + 
        "------------------------------------------------------------"
    )

    rescaler = RescalerCarrington()

    ev0 = args.w0 + 1j*args.d0
    evs = ev0 if args.ws is None else args.ws + 1j*args.ds
    print(f"\t\tTarget:{ev0:+24.4f} | {rescaler(ev0):+24.4f}", flush=True)

    starttime = time.perf_counter()
    ev_solved, vectors = single_eig(A, B, rescaler(ev0), nev=2, maxiter=50, tol=1e-12)
    elapsed = time.perf_counter() - starttime

    print(f"\t\tSolved:")
    for i, ev in enumerate(ev_solved):
        ev_tmp = rescaler.reverse(ev)
        print(f"\t\t{i:6d}:{ev_tmp:+24.4f} | {ev:+24.4f}")
    print(f"\t\tReference:{evs:+21.4f} | {rescaler(evs):+24.4f}")
    print(f"\t\tArnold iterations finished in {elapsed:.2f}s.\n", flush=True)

    if args.output_vector is not None:
        vectors = M_lib['perm'].T @ vectors
        np.save(args.output_vector, vectors)
        print(f"\t\tEigenvectors saved to {args.output_vector}\n", flush=True)


if __name__ == "__main__":
    main()


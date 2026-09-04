# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np
import time
from scipy import sparse
# from . import eig
import eig


def load_matrices(lib_dir):
    fnames = os.listdir(lib_dir)
    M_lib = dict()
    for fname in fnames:
        matname = fname[:-4]
        M_lib[matname] = sparse.load_npz(os.path.join(lib_dir, fname))
    return M_lib


def setup_op_spin_DR(M_lib, E, Em, Le, Ro):
    B = M_lib['mass']
    A = M_lib['coriolis'] \
        + Le*M_lib['lorentz_induction'] \
        + E*M_lib['viscous_diffusion'] \
        + Em*M_lib['magnetic_diffusion'] \
        + Ro*M_lib['differential_rotation']
    if 'tau' in M_lib:
        A += M_lib['tau']
    return A, B


class RescalerCarrington:

    def __call__(self, v):
        return v/456.0

    def reverse(self, v):
        return v*456.0


class RescalerIdentity:
    
    def __call__(self, v): return v
    def reverse(self, v): return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-Ek', type=float, default=1.0)
    parser.add_argument('-Em', type=float, default=1.0)
    parser.add_argument('-Le', type=float, default=1.0)
    parser.add_argument('-Ro', type=float, default=1.0)
    parser.add_argument('-w0', type=float, default=0.0)
    parser.add_argument('-d0', type=float, default=0.0)
    parser.add_argument('-ws', type=float)
    parser.add_argument('-ds', type=float)
    parser.add_argument('-v', '--output-vector')
    args = parser.parse_args()

    M_lib = load_matrices(args.dir)
    A, B = setup_op_spin_DR(M_lib, args.Ek, args.Em, args.Le, args.Ro)
    print(
        "\n============================================================\n" +
        "                   Resolve single eigen-pair                   \n"
    )
    for key, val in vars(args).items():
        print(f"\t{key}={val}")
    print("------------------------------------------------------------", flush=True)

    # rescaler = RescalerCarrington()
    rescaler = RescalerIdentity()

    ev0 = args.d0 + 1j*args.w0
    evs = ev0 if args.ws is None else args.ds + 1j*args.ws
    print(f"\t\tTarget:{ev0:+24.4f} | {rescaler(ev0):+24.4f}", flush=True)

    starttime = time.perf_counter()
    ev_solved, vectors = eig.single_eig(A, -B, rescaler(ev0), nev=1, maxiter=50, tol=1e-12)
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


if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-


import os, h5py
import argparse
import numpy as np
import time
from scipy import sparse
# from . import eig
import eig
from utils import sym_slicer


def load_matrices(lib_dir):
    fnames = os.listdir(lib_dir)
    M_lib = dict()
    for fname in fnames:
        matname = fname[:-4]
        M_lib[matname] = sparse.load_npz(os.path.join(lib_dir, fname))
    return M_lib


def setup_op_spin_DR(M_lib, E, Em, Le, Ro):
    B = M_lib['mass'].copy()
    A = M_lib.get('coriolis', 0) \
        + Le*M_lib.get('lorentz_induction', 0) \
        + E*M_lib.get('viscous_diffusion', 0) \
        + Em*M_lib.get('magnetic_diffusion', 0) \
        + Ro*M_lib.get('differential_rotation', 0) \
        + M_lib.get('tau', 0)
    return A, B


class RescalerCarrington:

    def __call__(self, v):
        return v/456.0

    def reverse(self, v):
        return v*456.0


class RescalerIdentity:
    
    def __call__(self, v): return v
    def reverse(self, v): return v


def main_scanEk_hydro():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-w0-file')
    parser.add_argument('-w0-path')
    parser.add_argument('-Ro', type=float, default=0.0)
    parser.add_argument('-v', '--output-vector')
    args = parser.parse_args()

    with h5py.File(args.w0_file, 'r') as fp:
        Ek_arr = fp["Ek"][()]
        seeds = fp[args.w0_path][()]
    assert Ek_arr.size == seeds.size

    print(
        "\n============================================================\n" +
        "                  Solve eigen-pairs for params                   \n")
    for key, val in vars(args).items():
        print(f"\t{key}={val}")
    print("------------------------------------------------------------\n", flush=True)


    for iEk, Ek in enumerate(Ek_arr):
        
        print(f"------------------------- Ek = {Ek:.02e} -----------------------", flush=True)

        M_lib = load_matrices(args.dir)
        A, B = setup_op_spin_DR(M_lib, E=Ek, Em=0, Le=0, Ro=args.Ro)
        ev0 = seeds[iEk]

        print(f"\t\tTarget:{ev0:+24.4f}", flush=True)

        starttime = time.perf_counter()
        ev_solved, vectors = eig.single_eig(A, -B, ev0, nev=1, maxiter=50, tol=1e-12)
        elapsed = time.perf_counter() - starttime

        print(f"\t\tSolved:")
        for i, ev in enumerate(ev_solved):
            print(f"\t\t{i:6d}:{ev:+24.4f}")
        print(f"\t\tArnold iterations finished in {elapsed:.2f}s.\n", flush=True)

        if args.output_vector is not None:
            vectors = M_lib['perm'].T @ vectors
            save_name = args.output_vector + f"_Ek{Ek:.2e}.npy"
            np.save(save_name, vectors)
            print(f"\t\tEigenvectors saved to {save_name}\n", flush=True)


def main_scanRo_hydro():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-w0-file')
    parser.add_argument('-w0-path')
    parser.add_argument('-hydro', action="store_true")
    parser.add_argument('-res', type=int, nargs=2)
    parser.add_argument('-v', '--output-vector')
    args = parser.parse_args()

    with h5py.File(args.w0_file, 'r') as fp:
        gp = fp[args.w0_path]
        Ek = gp.attrs["Ek"]
        Ro_arr = gp["Ro"][()]
        seeds = gp["w"][()]
    assert Ro_arr.size == seeds.size

    print(
        "\n============================================================\n" +
        "                  Solve eigen-pairs for params                   \n")
    for key, val in vars(args).items():
        print(f"\t{key}={val}")
    print("------------------------------------------------------------\n", flush=True)

    M_lib = load_matrices(args.dir)

    for iRo, Ro in enumerate(Ro_arr):
        
        print(f"------------------------- Ro = {Ro:.02e} -----------------------", flush=True)

        A, B = setup_op_spin_DR(M_lib, E=Ek, Em=0, Le=0, Ro=Ro)
        P = M_lib["perm"]
        ev0 = seeds[iRo]
        
        if not np.isfinite(ev0):
            continue

        if args.hydro:
            N = args.res[0]
            dL = args.res[1]
            A, _ = sym_slicer.slice_Hydro_from_MHD(A, N, dL, perm=P)
            B, P = sym_slicer.slice_Hydro_from_MHD(B, N, dL, perm=P)

        print(f"\t\tTarget:{ev0:+24.4f}", flush=True)

        starttime = time.perf_counter()
        ev_solved, vectors = eig.single_eig(A, -B, ev0, nev=1, maxiter=50, tol=1e-12)
        elapsed = time.perf_counter() - starttime

        print(f"\t\tSolved:")
        for i, ev in enumerate(ev_solved):
            print(f"\t\t{i:6d}:{ev:+24.4f}")
        print(f"\t\tArnold iterations finished in {elapsed:.2f}s.\n", flush=True)

        if args.output_vector is not None:
            vectors = P.T @ vectors
            save_name = args.output_vector + f"_Ro{Ro:.2e}.npy"
            np.save(save_name, vectors)
            print(f"\t\tEigenvectors saved to {save_name}\n", flush=True)
        del A, B, P



def main_hydro():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-Ek', type=float, default=1.0)
    parser.add_argument('-Ro', type=float, default=1.0)
    parser.add_argument('-w0', type=float, default=0.0)
    parser.add_argument('-d0', type=float, default=0.0)
    parser.add_argument('-ws', type=float)
    parser.add_argument('-ds', type=float)
    parser.add_argument('-v', '--output-vector')
    args = parser.parse_args()

    M_lib = load_matrices(args.dir)
    A, B = setup_op_spin_DR(M_lib, E=args.Ek, Em=0, Le=0, Ro=args.Ro)
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


def main_dispersion_mhd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-w0-file')
    parser.add_argument('-w0-path')
    parser.add_argument('-hydro', action="store_true")
    parser.add_argument('-res', type=int, nargs=2)
    parser.add_argument('-symmv', type=int)
    parser.add_argument('-symmb', type=int)
    parser.add_argument('-v', '--output-vector')
    args = parser.parse_args()

    with h5py.File(args.w0_file, 'r') as fp:
        gp = fp[args.w0_path]
        Ek = gp.attrs["Ek"]
        Em = gp.attrs["Em"]
        Le = gp.attrs["Le"]
        Ro = gp.attrs["Ro"]
        m_arr = gp["m"][()]
        seeds = gp["w"][()]
    assert m_arr.size == seeds.size
    N, dL = args.res

    print(
        "\n============================================================\n" +
        "                  Solve eigen-pairs for params                   \n")
    for key, val in vars(args).items():
        print(f"\t{key}={val}")
    print(f"Ek={Ek:+.02e}")
    print(f"Em={Em:+.02e}")
    print(f"Le={Le:+.02e}")
    print(f"Ro={Ro:+.02e}")
    print("------------------------------------------------------------\n", flush=True)

    for im, m in enumerate(m_arr):
        
        print(f"------------------------- m = {m:.02e} -----------------------", flush=True)
        if args.output_vector is not None:
            save_name = args.output_vector + f"_m{m:02d}.npy"
            if os.path.exists(save_name):
                print(f"\t\tEigenvectors already exist... Skipping\n", flush=True)
                continue

        m_dir = os.path.join(args.dir, f"Mat_m{m}_{N}x{m+dL}/ops")
        M_lib = load_matrices(m_dir)

        A, B = setup_op_spin_DR(M_lib, E=Ek, Em=Em, Le=Le, Ro=Ro)
        P = M_lib["perm"]
        ev0 = seeds[im]
        
        if not np.isfinite(ev0):
            continue

        if args.symmv is not None and args.symmb is not None:
            A, _ = sym_slicer.slice_sym_MHD(A, N, dL, sym_v=args.symmv, sym_b=args.symmb, perm=P)
            B, P = sym_slicer.slice_sym_MHD(B, N, dL, sym_v=args.symmv, sym_b=args.symmb, perm=P)

        print(f"\t\tTarget:{ev0:+24.4f}", flush=True)

        starttime = time.perf_counter()
        ev_solved, vectors = eig.single_eig(A, -B, ev0, nev=1, maxiter=50, tol=1e-12)
        elapsed = time.perf_counter() - starttime

        print(f"\t\tSolved:")
        for i, ev in enumerate(ev_solved):
            print(f"\t\t{i:6d}:{ev:+24.4f}")
        print(f"\t\tArnold iterations finished in {elapsed:.2f}s.\n", flush=True)

        if args.output_vector is not None:
            vectors = P.T @ vectors
            np.save(save_name, vectors)
            print(f"\t\tEigenvectors saved to {save_name}\n", flush=True)
        
        del M_lib, A, B, P


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
    # main()
    # main_scanEk_hydro()
    # main_scanRo_hydro()
    main_dispersion_mhd()

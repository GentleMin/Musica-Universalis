# -*- coding: utf-8 -*-


import os, time
import argparse
import numpy as np
from scipy import sparse 
# from models import evp_shell, bgs
# from utils import eigs, sym_slicer
from src.models import evp_shell, bgs
from src.utils import eigs, sym_slicer



def load_matrices(lib_dir):
    fnames = os.listdir(lib_dir)
    M_lib = dict()
    for fname in fnames:
        matname = fname[:-4]
        M_lib[matname] = sparse.load_npz(os.path.join(lib_dir, fname))
    return M_lib


def setup_op_spin_DR(M_lib, E, Em, Le, Ro):
    B = -M_lib['mass']
    A = M_lib['coriolis'] \
        + Le*M_lib['lorentz_induction'] \
        + E*M_lib['viscous_diffusion'] \
        + Em*M_lib['magnetic_diffusion'] \
        + Ro*M_lib['differential_rotation']
    if 'tau' in M_lib:
        A += M_lib['tau']
    return A, B



def scan_Rossby():

    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("-res", type=int, nargs=2)
    parser.add_argument("-symmv", type=int)
    parser.add_argument("-symmb", type=int)
    parser.add_argument("-Ek", type=float)
    parser.add_argument("-Em", type=float)
    parser.add_argument("-Le", type=float)
    args = parser.parse_args()

    Nr, dL = args.res[0], args.res[1]
    m = args.m
    Ek = args.Ek
    Em = args.Em
    Le = args.Le
    L = m + dL
    dir_name = f"./out/bg_SolarTa_LS18-SH16T9/Mat_m{m:d}_{Nr}x{L}"
    M_lib = load_matrices(f"{dir_name}/ops")
    # Ro_arr = np.linspace(0., 1., num=51)
    Ro_arr = 0.01*np.r_[np.arange(0, 30, 2), np.arange(30, 70, 4), np.arange(70, 101, 6)]

    for i_Ro, Ro_r in enumerate(Ro_arr):

        print(f"============================= Rossby: {Ro_r:.2f} ============================")    
        save_name = dir_name + f'/spec_Ek{Ek:.2e}_Em{Em:.2e}_Le{Le:.2e}_Ro{Ro_r:.2f}'
        if args.symmv is not None:
            save_name = save_name + f"_v{args.symmv}-b{args.symmb}"
        if os.path.exists(save_name + '.npy'):
            print(f"\tResult already exists... aborting...")
            continue
        tstart = time.perf_counter()

        A, B = setup_op_spin_DR(M_lib, Ek, Em, Le, Ro_r)
        if args.symmv is not None:
            A, P = sym_slicer.slice_sym_MHD(A, Nr, dL, sym_v=args.symmv, sym_b=args.symmb, perm=M_lib['perm'])
            B, _ = sym_slicer.slice_sym_MHD(B, Nr, dL, sym_v=args.symmv, sym_b=args.symmb, perm=M_lib['perm'])
       
        print(f"\tSolving eigen-system with size {A.shape}...", flush=True)
        w = eigs.full_eig(A, B)
        tlapse = time.perf_counter() - tstart
        print(f"\tCalculation finished in {tlapse:.2f} s.", flush=True)
        np.save(save_name, w)


if __name__ == '__main__':
    scan_Rossby()



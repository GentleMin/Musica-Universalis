# -*- coding: utf-8 -*-

import os, warnings, argparse
import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as spla
from scipy.io import mmread
from utils import timers, sym_slicer, eigs


def quicc_load(lib_dir, name):
    A_cpx = mmread(os.path.join(lib_dir, f'{name}_re.mtx')) \
        + 1j*mmread(os.path.join(lib_dir, f'{name}_im.mtx'))
    return A_cpx


def load_matrices(lib_dir):
    fnames = os.listdir(lib_dir)
    M_lib = dict()
    for fname in fnames:
        matname = fname[:-4]
        M_lib[matname] = sparse.load_npz(os.path.join(lib_dir, fname))
    return M_lib


def setup_op_Alfven(M_lib, Le, Lu, Pm):
    B = M_lib['mass']
    A = (1/Le)*M_lib.get('coriolis', 0) \
        + M_lib.get('lorentz_induction', 0) \
        + (Pm/Lu)*M_lib.get('viscous_diffusion', 0) \
        + (1/Lu)*M_lib.get('magnetic_diffusion', 0) \
        + M_lib.get('tau', 0)
    return A, B

def setup_op_spin(M_lib, E, Em, Le):
    B = M_lib['mass']
    A = M_lib.get('coriolis', 0) \
        + Le*M_lib.get('lorentz_induction', 0) \
        + E*M_lib.get('viscous_diffusion', 0) \
        + Em*M_lib.get('magnetic_diffusion', 0) \
        + M_lib.get('tau', 0)
    return A, B

def setup_op_spin_DR(M_lib, E, Em, Le, Ro):
    B = M_lib['mass']
    A = M_lib.get('coriolis', 0) \
        + Le*M_lib.get('lorentz_induction', 0) \
        + E*M_lib.get('viscous_diffusion', 0) \
        + Em*M_lib.get('magnetic_diffusion', 0) \
        + Ro*M_lib.get('differential_rotation', 0) \
        + M_lib.get('tau', 0)
    return A, B


# ======================================================
#           Main configuration and programs
# ======================================================


def main_scanEk_Hydro():

    parser = argparse.ArgumentParser()
    parser.add_argument('-src')
    parser.add_argument('-dest')
    parser.add_argument('-Ro', type=float, default=1.0)
    parser.add_argument('-gevp', action='store_true')
    parser.add_argument('-symmv', type=int)
    parser.add_argument('-N', type=int)
    parser.add_argument('-dL', type=int)
    parser.add_argument('-d3op', '--dedalus-operators', action='store_true')
    parser.add_argument('-w', '--overwrite', action='store_true')

    args = parser.parse_args()
    par_list = np.logspace(-6, -3, num=25)
    timer = timers.ProcTimer(start=True)
    print("\n============================================================================================")
    print("                     Calculating full spectrum of eigenvalue problem                        ")
    print("                                   Ekman param scan                                         ")
    print("--------------------------------------------------------------------------------------------")
    for key, val in vars(args).items():
        print(f"\t{key}={val}")
    print(f"\nScanning Ek:\n\t{par_list}")
    
    spec_solver = eigs.full_gevp if args.gevp else eigs.full_eig
    M_lib = load_matrices(args.src)

    for i_par, par in enumerate(par_list):

        print(f"\n------------------------------ Ek = {par:+.3e} ---------------------------------------------")
        Ro = args.Ro
        A, B = setup_op_spin_DR(M_lib, E=par, Em=0, Le=0, Ro=Ro)
        fname = os.path.join(args.dest, f'spec_Ek{par:.2e}')
        
        if args.symmv is not None:
            perm = M_lib['perm'] if 'perm' in M_lib else None
            N = args.N if args.dedalus_operators else args.N + 1
            dL = args.dL if args.dedalus_operators else args.dL + 1
            A, P = sym_slicer.slice_sym_HydroTP(A, N, dL, sym=args.symmv, perm=perm)
            B, _ = sym_slicer.slice_sym_HydroTP(B, N, dL, sym=args.symmv, perm=perm)
            fname = fname + f'_v{args.symmv}'
    
        if args.dedalus_operators:
            B *= -1
    
        if os.path.exists(fname + '.npy') and (not args.overwrite):
            timer.flag(loginfo="File exists! Skipping...", print_str=True, mode='0+', flush=True)
            continue

        timer.flag(loginfo=f'Eigensolver starts...', print_str=True, mode='0+', flush=True)
        w = spec_solver(A, B)
        np.save(fname + '.npy', w)
        timer.flag(loginfo=f'Spectrum solved successfully! Results saved to {fname}.npy', print_str=True, mode='0+', flush=True)



def main_scanEk_MHD():

    parser = argparse.ArgumentParser()
    parser.add_argument('-src')
    parser.add_argument('-dest')
    parser.add_argument('-Em', type=float, default=1e-4)
    parser.add_argument('-Ro', type=float, default=1.0)
    parser.add_argument('-Le', type=float, default=1e-2)
    parser.add_argument('-hydro', action='store_true')
    parser.add_argument('-gevp', action='store_true')
    parser.add_argument('-symmv', type=int)
    parser.add_argument('-symmb', type=int)
    parser.add_argument('-N', type=int)
    parser.add_argument('-dL', type=int)
    parser.add_argument('-d3op', '--dedalus-operators', action='store_true')
    parser.add_argument('-w', '--overwrite', action='store_true')

    args = parser.parse_args()
    par_list = np.arange(0.022, 0.04, step=0.002)
    timer = timers.ProcTimer(start=True)
    print("\n============================================================================================")
    print("                     Calculating full spectrum of eigenvalue problem                        ")
    print("                                   Ekman param scan                                         ")
    print("--------------------------------------------------------------------------------------------")
    for key, val in vars(args).items():
        print(f"\t{key}={val}")
    print(f"\nScanning Ek:\n\t{par_list}")
    
    spec_solver = eigs.full_gevp if args.gevp else eigs.full_eig
    M_lib = load_matrices(args.src)

    for i_par, par in enumerate(par_list):

        print(f"\n------------------------------ Ek = {par:+.3e} ---------------------------------------------")
        Em, Ro, Le = args.Em, args.Ro, args.Le
        A, B = setup_op_spin_DR(M_lib, par, Em, Le, Ro)
        # fname = os.path.join(args.dest, f'spec_Ek{par:.2e}_Em{Em:.2e}_Le{Le:.2e}_Ro{Ro:.2f}')
        fname = os.path.join(args.dest, f'spec_Ek{par:.2e}')
        
        if args.symmv is not None and args.symmb is not None:
            perm = M_lib['perm'] if 'perm' in M_lib else None
            N = args.N if args.dedalus_operators else args.N + 1
            dL = args.dL if args.dedalus_operators else args.dL + 1
            A, P = sym_slicer.slice_sym_MHD(A, N, dL, sym_v=args.symmv, sym_b=args.symmb, perm=perm)
            B, _ = sym_slicer.slice_sym_MHD(B, N, dL, sym_v=args.symmv, sym_b=args.symmb, perm=perm)
            fname = fname + f'_v{args.symmv}-b{args.symmb}'
    
        if args.hydro:
            dim = A.shape[0] // 2
            A = A[:dim, :dim]
            B = B[:dim, :dim]
            fname = fname + '_hydro'
    
        if args.dedalus_operators:
            B *= -1
    
        if os.path.exists(fname + '.npy') and (not args.overwrite):
            timer.flag(loginfo="File exists! Skipping...", print_str=True, mode='0+', flush=True)
            continue

        timer.flag(loginfo=f'Eigensolver starts...', print_str=True, mode='0+', flush=True)
        w = spec_solver(A, B)
        np.save(fname + '.npy', w)
        timer.flag(loginfo=f'Spectrum solved successfully! Results saved to {fname}.npy', print_str=True, mode='0+', flush=True)


if __name__ == '__main__':
    main_scanEk_Hydro()

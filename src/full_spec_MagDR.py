# -*- coding: utf-8 -*-

import os, warnings, argparse
import numpy as np
from scipy import sparse, linalg
from scipy.sparse import linalg as spla
from scipy.io import mmread
from utils import timers, sym_slicer
# from .utils import timers, sym_slicer


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
    A = (1/Le)*M_lib['coriolis'] \
        + M_lib['lorentz_induction'] \
        + (Pm/Lu)*M_lib['viscous_diffusion'] \
        + (1/Lu)*M_lib['magnetic_diffusion']
    if 'tau' in M_lib:
        A += M_lib['tau']
    return A, B

def setup_op_spin(M_lib, E, Em, Le):
    B = M_lib['mass']
    A = M_lib['coriolis'] \
        + Le*M_lib['lorentz_induction'] \
        + E*M_lib['viscous_diffusion'] \
        + Em*M_lib['magnetic_diffusion']
    if 'tau' in M_lib:
        A += M_lib['tau']
    return A, B

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


# ======================================================
#           Main configuration and programs
# ======================================================

parser = argparse.ArgumentParser()
parser.add_argument('-src', default='./Mat_S1_32x32x3_BCNS/ops')
parser.add_argument('-dest', default='./Mat_S1_32x32x3_BCNS/')
parser.add_argument('-Ek', type=float, default=1e-4)
parser.add_argument('-Em', type=float, default=1e-4)
parser.add_argument('-Le', type=float, default=1e-4)
parser.add_argument('-Ro', type=float, default=1.0)
parser.add_argument('-hydro', action='store_true')
parser.add_argument('-gevp', action='store_true')
parser.add_argument('-symmv', type=int)
parser.add_argument('-symmb', type=int)
parser.add_argument('-N', type=int)
parser.add_argument('-dL', type=int)
parser.add_argument('-d3op', '--dedalus-operators', action='store_true')

def main():

    args = parser.parse_args()
    timer = timers.ProcTimer(start=True)
    timer.flag(loginfo=f'Calculating full spectrum of eigenvalue problem \n\t{args}', print_str=True, mode='0+', flush=True)
    
    spec_solver = full_gevp if args.gevp else full_eig
    M_lib = load_matrices(args.src)
    Ek, Em, Le, Ro = args.Ek, args.Em, args.Le, args.Ro
    Pm, Lu = Ek/Em, Le/Em

    # A, B = setup_op_Alfven(M_lib, Le, Lu, Pm)
    # fname = os.path.join(args.dest, f'spec_Le{Le:.2e}_Lu{Lu:.2e}_Pm{Pm:.2e}')

    A, B = setup_op_spin_DR(M_lib, Ek, Em, Le, Ro)
    fname = os.path.join(args.dest, f'spec_Ek{Ek:.2e}_Em{Em:.2e}_Le{Le:.2e}_Ro{Ro:.2f}')
    
    if args.symmv is not None and args.symmb is not None:
        # print(4*(args.N+1)*(args.dL+1), A.shape)
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

    timer.flag(loginfo=f'Eigensolver starts...', print_str=True, mode='0+', flush=True)
    w = spec_solver(A, B)
    np.save(fname + '.npy', w)
    timer.flag(loginfo=f'Spectrum solved successfully! Results saved to {fname}.npy', print_str=True, mode='0+', flush=True)


if __name__ == '__main__':
    main()

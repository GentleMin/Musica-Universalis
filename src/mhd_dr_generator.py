# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from scipy import interpolate, sparse
# from models import evp_shell
# from utils import eigs, sym_slicer
from src.models import evp_shell, bgs
from src.utils import eigs, sym_slicer, timers


def main_mhd_dr_setup():

    parser = argparse.ArgumentParser()
    parser.add_argument('-res', type=int, nargs=3)
    parser.add_argument('-dir', default="./Rotating_SphShells/")
    args = parser.parse_args()

    print("============================================================================================")
    print("               Rotating MHD eigenvalue problem with differential rotation                   ")
    print("                                     Operator setup                                         ")
    print("--------------------------------------------------------------------------------------------")
    print(f"Parameters: {args}\n", flush=True)
    os.makedirs(args.dir, exist_ok=True)

    ptimer = timers.ProcTimer(start=True)
    Ri, Ro = 0.71, 1.
    D = Ro - Ri
    Ri /= D
    Ro /= D
    m, L, N = args.res[0], args.res[1], args.res[2]

    # B0 = bgs.MagBG_SolarTa()
    B0 = bgs.MagBG_T2_Linear(Ri, Ro, Bi=1, Bo=0.5)
    U0 = bgs.SolarDiffRot_SHT_LSQ(Ro, L=16, Nr=9)
    model_rmhd = evp_shell.ModelEVP_MHDDiffRShell_TorPol((Ri, Ro), (N, L, m), B0_func=B0, U0_func=U0)
    M_lib = model_rmhd.precomp_submat(set_problem=True,
        v_bc_i='stress-free', v_bc_o='stress-free',
        b_bc_i='perfect-conducting', b_bc_o='insulating')
    M_lib["perm"] = model_rmhd.subprob.pre_left

    for key, mat in M_lib.items():
        sparse.save_npz(os.path.join(args.dir, key), mat)
    ptimer.flag(loginfo=f"Operators saved to {args.dir}", print_str=True, mode='0+', flush=True)


if __name__ == '__main__':
    main_mhd_dr_setup()


# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from scipy import interpolate, sparse
# from models import evp_shell
# from utils import eigs, sym_slicer
from src.models import evp_shell, bgs
from src.utils import eigs, sym_slicer


def get_solar_diff_rot(Omega0: float = 1.):

    resource_dir = "resources/Mukhopadhyay_2025_Data"
    diff_rot_file = os.path.join(resource_dir, "data/diff_rot.npz")
    diff_rot_data = np.load(diff_rot_file)
    diff_rot = diff_rot_data['ome']
    rs = diff_rot_data['r']
    thetas = diff_rot_data['theta']
    f_rot = interpolate.RectBivariateSpline(rs, thetas, diff_rot.T)
    
    def f_diff_rot(r_grid, t_grid):
        dOmega = f_rot(r_grid.flatten(), t_grid.flatten()[::-1]).T[np.newaxis, ::-1, :]
        return (dOmega - Omega0)/Omega0

    return f_diff_rot


def main_shell_dr():

    Ri, Ro = 0.71, 1.
    Nr, L, m = 24, 42, 10
    f_dOmega = get_solar_diff_rot(Omega0=456.)

    # m_dr_shell = evp_shell.ModelEVP_DiffRShell((Ri, Ro), (Nr, L, m), dOmega_func=f_dOmega)
    m_dr_shell = evp_shell.ModelEVP_DiffRShell_TorPol((Ri, Ro), (Nr, L, m), dOmega_func=f_dOmega)
    M_lib = m_dr_shell.precomp_submat(set_problem=True, bc='stress-free')

    Ro = 1
    Ek = 1e-4
    A = M_lib['coriolis'] + Ek*M_lib['viscous_diffusion'] + Ro*M_lib['differential_rotation']
    B = -M_lib['mass']

    A, P = sym_slicer.slice_sym_HydroTP(A, Nr, L-m, sym=1, perm=m_dr_shell.subprob.pre_left)
    B, _ = sym_slicer.slice_sym_HydroTP(B, Nr, L-m, sym=1, perm=m_dr_shell.subprob.pre_left)

    w = eigs.full_eig(A, B)
    np.save("./tmp/model_drTP-a_test_m10", w)


def main_shell_solar_mag():

    parser = argparse.ArgumentParser()
    parser.add_argument('-res', type=int, nargs=3)
    # parser.add_argument('-Le', type=float)
    # parser.add_argument('-Ek', type=float)
    # parser.add_argument('-Em', type=float)
    parser.add_argument('-dir', default="./Rotating_SphShells/")
    args = parser.parse_args()

    print("========================================================================")
    print("               Rotating MHD eigenvalue problem solver                   ")
    print("------------------------------------------------------------------------")
    print(f"Parameters: {args}\n", flush=True)
    Ri, Ro = 0.71, 1.
    D = Ro - Ri
    Ri /= D
    Ro /= D
    m, L, N = args.res[0], args.res[1], args.res[2]
    # Le, Ek, Em = args.Le, args.Ek, args.Em

    B0 = bgs.MagBG_SolarTa()
    model_rmhd = evp_shell.ModelEVP_MHDRShell_TorPol_Alfven((Ri, Ro), (N, L, m), B0)
    M_lib = model_rmhd.precomp_submat(set_problem=True,
        v_bc_i='stress-free', v_bc_o='stress-free',
        b_bc_i='perfect-conducting', b_bc_o='insulating')
    M_lib["perm"] = model_rmhd.subprob.pre_left

    os.makedirs(args.dir, exist_ok=True)
    for key, mat in M_lib.items():
        sparse.save_npz(os.path.join(args.dir, key), mat)


if __name__ == '__main__':
    main_shell_dr()

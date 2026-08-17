# -*- coding: utf-8 -*-


import os, time
import argparse
import numpy as np
from scipy import interpolate
# from models import evp_shell, bgs
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


def scan_Rossby():

    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("-res", type=int, nargs=2)
    parser.add_argument("-symm", type=int)
    args = parser.parse_args()

    Ri, Ro = 0.71, 1.
    Nr, dL = args.res[0], args.res[1]
    m = args.m
    L = m + dL
    dir_name = f"./Rotating_SphShells/hydro_DR_LS18/Incompressible/m_{m:02d}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # f_dOmega = get_solar_diff_rot(Omega0=456.)
    U0 = bgs.SolarDiffRot_SHT_LSQ(Ro, L=16, Nr=9)

    # m_dr_shell = evp_shell.ModelEVP_DiffRShell_TorPol((Ri, Ro), (Nr, L, m), dOmega_func=f_dOmega)
    m_dr_shell = evp_shell.ModelEVP_DiffRShell_TorPol((Ri, Ro), (Nr, L, m), U0_func=U0)
    M_lib = m_dr_shell.precomp_submat(set_problem=True, bc='stress-free')
    
    Ek = 1e-4
    Ro_arr = np.linspace(0., 1., num=51)

    for i_Ro, Ro_r in enumerate(Ro_arr):

        print(f"============================= Rossby: {Ro_r:.2f} ============================")    
        save_name = dir_name + f'/freq-dr{Ro_r:.2f}_{m}x{L}x{Nr}'
        if args.symm is not None:
            save_name = save_name + f"_v{args.symm}"
        if os.path.exists(save_name + '.npy'):
            print(f"\tResult already exists... aborting...")
            continue
        tstart = time.perf_counter()

        A = M_lib['coriolis'] + Ek*M_lib['viscous_diffusion'] + Ro_r*M_lib['differential_rotation']
        B = -M_lib['mass']
        if args.symm is not None:
            A, P = sym_slicer.slice_sym_HydroTP(A, Nr, dL, sym=args.symm, perm=m_dr_shell.subprob.pre_left)
            B, _ = sym_slicer.slice_sym_HydroTP(B, Nr, dL, sym=args.symm, perm=m_dr_shell.subprob.pre_left)
        
        print(f"\tSolving eigen-system with size {A.shape}...", flush=True)
        w = eigs.full_eig(A, B)
        tlapse = time.perf_counter() - tstart
        print(f"\tCalculation finished in {tlapse:.2f} s.", flush=True)
        np.save(save_name, w)


if __name__ == '__main__':
    scan_Rossby()

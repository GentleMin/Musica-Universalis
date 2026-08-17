# -*- coding: utf-8 -*-


import os
import argparse
from models import evp_shell, bgs
from scipy import sparse


parser = argparse.ArgumentParser()
parser.add_argument('-Ri', type=float, default=0.35)
parser.add_argument('-Ro', type=float, default=1.)
parser.add_argument('-L', default='Ro')
parser.add_argument('-B0', default='T2')
parser.add_argument('-vi', default='stress-free')
parser.add_argument('-vo', default='stress-free')
parser.add_argument('-bi', default='insulating')
parser.add_argument('-bo', default='insulating')
parser.add_argument('-res', type=int, nargs=3, default=(36, 36, 2))
parser.add_argument('-o', '--output', default='.')


def main():

    print('\n' + '='*64 + '\n')
    print("Set up eigen operators with parameters:\n")
    args = parser.parse_args()
    for key, val in vars(args).items():
        print(f"\t{key:8}= {val}")
    print()

    Ri, Ro = args.Ri, args.Ro
    L = 1
    if args.L == 'Ro':
        L = Ro
    if args.L == 'D':
        L = Ro - Ri
    Ri /= L
    Ro /= L

    bg = bgs.B0_Lib[args.B0](Ri=Ri, Ro=Ro)
    m_mhd_rshell = evp_shell.ModelEVP_MHDRShell_TorPol_Alfven((Ri, Ro), args.res, bg)
    M_lib = m_mhd_rshell.precomp_submat(set_problem=True, 
        v_bc_i=args.vi, v_bc_o=args.vo, b_bc_i=args.bi, b_bc_o=args.bo)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"\nFollowing operators saved to {args.output}:")
    for key in M_lib:
        sparse.save_npz(os.path.join(args.output, key), M_lib[key])
        print(f"\t{key}")
 
    print('\n' + '='*64 + '\n')


if __name__ == '__main__':
    main()

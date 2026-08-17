# -*- coding: utf-8 -*-


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from scipy import sparse, io

from utils.plottings import spy_cmarker


def main():
    """Command line tool to visualize a matrix file
    """

    parser = argparse.ArgumentParser(
        prog="Matrix Sparsity Visualization with Colormapping",
        description="Command line tool to visualize sparsity pattern of a matrix file."
    )
    parser.add_argument('file', help='Matrix file to view. Currently supported formats: .npz, .mtx')

    args = parser.parse_args()
    if args.file[-3:] == 'npz':
        sp_arr = sparse.load_npz(args.file)
    elif args.file[-3:] == 'mtx':
        sp_arr = io.mmread(args.file)
    else:
        raise NotImplementedError

    fig, ax = plt.subplots(figsize=(5*sp_arr.shape[1]/sp_arr.shape[0] + 1, 5), layout='constrained')
    marker_style = {'marker': 's', 's': 1000/sp_arr.shape[0]}
    pts = spy_cmarker(sp_arr, ax, transform=np.abs, **marker_style, norm=mpl.colors.LogNorm(), cmap='magma_r')
    plt.colorbar(pts, ax=ax)
    ax.set_title('spy_cmarker')

    plt.show()


if __name__ == '__main__':
    main()
    

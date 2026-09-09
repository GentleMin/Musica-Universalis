# -*- coding: utf-8 -*-


import os, warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import sparse, io
from cartopy import crs as ccrs


def figsave(fig, file, formats=('png',), overwrite=False, **kwargs):
    if file is None:
        return False
    saved = False
    for fmt in formats:
        fname = file + '.' + fmt
        if os.path.exists(fname) and (not overwrite):
            warnings.warn(f'Skipping {fname} ... File exists!')
            continue
        fig.savefig(fname, **kwargs)
        saved = True
    return saved


def spy_cmarker(Z, ax: plt.Axes, precision=0, transform=lambda x: x, 
    aspect='equal', origin='upper', **kwargs):
    """Visualize sparsity pattern with values encoded with colormaps
    
    :param Z: Sparse matrix
    :param ax: matplotlib.pyplot.Axes, axes for plotting
    :param precision: float, if element < precision then ignore
    :param transform: Callable, maps elements to values used for colormap
    :param aspect: string, aspect options
    :param origin: string, set origin to top left if origin is 'upper'
    :param kwargs: any further keyword arguments to pass to Axes.scatter (see Axes.scatter documentation)
    """

    Z_sp = sparse.coo_array(Z.copy())
    Z_sp.data[np.abs(Z_sp.data) <= precision] = 0
    Z_sp.eliminate_zeros()
    
    pts = ax.scatter(Z_sp.col, Z_sp.row, c=transform(Z_sp.data), **kwargs)
    
    Nr, Nc = Z_sp.shape
    ax.set_xlim([-0.5, Nc - 0.5])
    ax.set_aspect(aspect)
    if origin == 'upper':
        ax.invert_yaxis()
        ax.set_ylim([Nr - 0.5, -0.5])
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    else:
        ax.set_ylim([-0.5, Nr - 0.5])
    
    return pts


def plot_v_comp(xx, yy, ff, fig, gs, vcap=1e-7, title=None, invert_x=False, handles=False):
    ax = fig.add_subplot(gs)
    ccap = vcap if np.abs(np.real(ff)).max() < vcap else None
    im = ax.pcolormesh(xx, yy, np.real(ff), cmap='coolwarm', norm=mpl.colors.CenteredNorm(halfrange=ccap), shading='gouraud')
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.formatter.set_powerlimits((-2, 2))
    ax.set_title(title)
    ax.set_aspect('equal')
    rmax = np.sqrt(np.max(xx**2 + yy**2))
    ax.set_xlim(0, 1.01*rmax)
    ax.set_ylim(-1.01*rmax, +1.01*rmax)
    if invert_x:
        ax.invert_xaxis()
    ax.axis('off')
    if handles:
        return ax, im, cbar
    return ax


def plot_positive(xx, yy, ff, fig, gs, vcap=1e-7, title=None, invert_x=False, handles=False):
    ax = fig.add_subplot(gs)
    ccap = vcap if np.abs(np.real(ff)).max() < vcap else None
    im = ax.pcolormesh(xx, yy, np.real(ff), cmap='magma', norm=mpl.colors.Normalize(vmin=0, vmax=ccap), shading='gouraud')
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
    cbar.formatter.set_powerlimits((-2, 2))
    ax.set_title(title)
    ax.set_aspect('equal')
    rmax = np.sqrt(np.max(xx**2 + yy**2))
    ax.set_xlim(0, 1.01*rmax)
    ax.set_ylim(-1.01*rmax, +1.01*rmax)
    if invert_x:
        ax.invert_xaxis()
    ax.axis('off')
    if handles:
        return ax, im, cbar
    return ax


def plot_sphere(lon, lat, ff, ax, vcap=1e-7, cbar=True, handles=False):
    ccap = vcap if np.abs(np.real(ff)).max() < vcap else None
    im = ax.pcolormesh(lon, lat, np.real(ff), transform=ccrs.PlateCarree(), 
        cmap='coolwarm', norm=mpl.colors.CenteredNorm(halfrange=ccap), shading='gouraud')
    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=50)
        cbar.formatter.set_powerlimits((-2, 2))
        if handles:
            return ax, im, cbar
        return ax
    if handles:
        return ax, im
    return im, ax


def plot_r_outline(r_lines, tg: np.ndarray, ax: Axes, color='k', linewidth=1, **kw):
    r_lines = [r_lines,] if not isinstance(r_lines, list) else r_lines
    for r in r_lines:
        x = r*np.sin(tg)
        y = r*np.cos(tg)
        ax.plot(x, y, color=color, linewidth=linewidth, **kw)
    return ax


def random_sparse_square_matrix(N, nnzeros, rand_seed=42):

    dat = np.array([], dtype=np.float64)
    row = np.array([], dtype=np.int32)
    col = np.array([], dtype=np.int32)

    # Add unstructured random matrix
    rng = np.random.default_rng(rand_seed)
    idx = rng.choice(N**2, size=nnzeros, replace=False, shuffle=False)
    dat_tmp = rng.standard_normal(size=nnzeros)
    row_tmp, col_tmp = np.unravel_index(idx, (N, N))
    dat = np.r_[dat, dat_tmp]
    row = np.r_[row, row_tmp]
    col = np.r_[col, col_tmp]

    # Add diagonal bands
    for k in range(-5, 6):
        row = np.r_[row, np.arange(max(0, -k), N - max(0, +k))]
        col = np.r_[col, np.arange(max(0, +k), N - max(0, -k))]
        dat = np.r_[dat, np.exp(-np.abs(k)**2)*np.ones(N - np.abs(k))]
        dat = np.r_[dat, dat_tmp]
        row = np.r_[row, row_tmp]
        col = np.r_[col, col_tmp]

    mat = sparse.coo_array((dat, (row, col)), shape=(N, N))
    mat = sparse.csr_array(mat)
    return mat


def demo_spy():
    """Demonstrate visualization function
    """
    
    N = 50
    nnzeros = 100
    sp_arr = random_sparse_square_matrix(N, nnzeros)

    fig, axes = plt.subplots(ncols=2, figsize=(11, 5), layout='constrained')

    ax = axes[0]
    marker_style = {'marker': 's', 'markersize': 3}
    ax.spy(sp_arr, **marker_style)
    ax.set_title('Axes.spy')

    ax = axes[1]
    marker_style = {'marker': 's', 's': 10}
    pts = spy_cmarker(sp_arr, ax, transform=np.abs, **marker_style, norm=mpl.colors.LogNorm(vmin=1e-12), cmap='magma_r')
    plt.colorbar(pts, ax=ax)
    ax.set_title('spy_cmarker')

    plt.show()


if __name__ == '__main__':
    demo_spy()    


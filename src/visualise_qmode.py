# -*- coding: utf-8 -*-


import os, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
from scipy import sparse, special
from cartopy import crs as ccrs
from typing import Optional

import eig
from utils import plottings
from qpost import basis as qbasis
from qpost import field as qfield


def load_operators(lib_dir):
    fnames = os.listdir(lib_dir)
    M_lib = dict()
    for fname in fnames:
        matname = fname[:-4]
        M_lib[matname] = sparse.load_npz(os.path.join(lib_dir, fname))
    return M_lib


def calc_mode(mat_dir, params, w0):

    M_lib = load_operators(mat_dir)
    Ek_val = params['Ek']
    Em_val = params['Em']
    Le_val = params['Le']

    A = (M_lib['coriolis'] 
        + Le_val*M_lib['lorentz_induction'] 
        + Ek_val*M_lib['viscous_diffusion'] 
        + Em_val*M_lib['magnetic_diffusion']
        + M_lib['tau'])
    B = M_lib['mass']

    w, v = eig.single_eig(A, B, w0, nev=1)
    w = w[0]
    v = v[:,0]
    print(f'{w0:+.3e}', f'{w:+.3e}', sep='\n')
    return w, v


def build_mode_phys(geometry, resolution, v):

    ri, ro = geometry
    N, L, m = resolution

    T_basis = qbasis.ChebyshevT(N, interval=(ri, ro))
    Plm_basis = qbasis.LegendrePlm(L, m)
    usp = qfield.ShellVectorTorPol_m(T_basis, Plm_basis, v[:v.size//2])
    bsp = qfield.ShellVectorTorPol_m(T_basis, Plm_basis, v[v.size//2:])
    usp_curl = usp.curl(inplace=False)

    return usp, bsp, usp_curl


def visual_mode(
        usp: qfield.ShellVectorTorPol_m, 
        bsp: qfield.ShellVectorTorPol_m, 
        usp_curl: qfield.ShellVectorTorPol_m, 
        rratio_sect: float = 1.0
    ):

    ri, ro = usp.ri, usp.ro
    m = usp.m

    xg, wx = special.roots_chebyt(100)
    zg, wz = special.roots_legendre(200)
    rg = usp.r_basis.x2g(np.r_[-1., xg, +1.])
    tg = usp.t_basis.z2t(np.r_[-1., zg, 1.])
    pg = np.linspace(0., 2*np.pi, num=500)
    xx = np.outer(np.sin(tg), rg)
    yy = np.outer(np.cos(tg), rg)
    lon, lat = np.meshgrid(np.degrees(pg), np.degrees(np.pi/2 - tg))

    uphys = usp.eval_mesh(rg, tg)
    bphys = bsp.eval_mesh(rg, tg)
    uphys_curl = usp_curl.eval_mesh(rg, tg)

    phi_rotate = -np.angle(uphys['p'].flatten()[np.argmax(np.abs(uphys['p']))])
    norm = 1/np.max(np.abs(uphys['p']))
    phase = np.exp(1j*phi_rotate)
    cfact = norm*phase
    ir = np.argmin(np.abs(rg/ro - rratio_sect))
    r_sect = rg[ir]

    fig = plt.figure(figsize=(12, 10))
    gs = mpl.gridspec.GridSpec(2, 5)

    ax = fig.add_subplot(gs[0,:2], projection=ccrs.Orthographic(central_latitude=20, central_longitude=0))
    fsurf = np.outer(cfact*uphys_curl['r'][:, ir], np.exp(1j*m*pg))
    plottings.plot_sphere(lon, lat, fsurf, ax)
    ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')

    ax = plottings.plot_v_comp(xx, yy, cfact*uphys['r'], fig, gs[0,2], title=r'$u_r$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, cfact*uphys['t'], fig, gs[0,3], title=r'$u_\theta$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, cfact*uphys['p'], fig, gs[0,4], title=r'$u_\phi$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')

    ax = fig.add_subplot(gs[1,:2], projection=ccrs.Orthographic(central_latitude=20, central_longitude=0))
    pg = np.linspace(0., 2*np.pi, num=500)
    fsurf = np.outer(cfact*bphys['r'][:, ir], np.exp(1j*m*pg))
    plottings.plot_sphere(lon, lat, fsurf, ax)
    ax.set_title(r'$b_r$')

    ax = plottings.plot_v_comp(xx, yy, cfact*bphys['r'], fig, gs[1,2], title=r'$b_r$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, cfact*bphys['t'], fig, gs[1,3], title=r'$b_\theta$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, cfact*bphys['p'], fig, gs[1,4], title=r'$b_\phi$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')

    plt.show()


def visual_mode_t(
        usp: qfield.ShellVectorTorPol_m, 
        bsp: qfield.ShellVectorTorPol_m, 
        usp_curl: qfield.ShellVectorTorPol_m, 
        w: complex = 0.,
        t: float = 0.,
        rratio_sect: float = 1.0,
    ):

    ri, ro = usp.ri, usp.ro
    m = usp.m

    xg, wx = special.roots_chebyt(100)
    zg, wz = special.roots_legendre(200)
    rg = usp.r_basis.x2g(np.r_[-1., xg, +1.])
    tg = usp.t_basis.z2t(np.r_[-1., zg, 1.])
    pg = np.linspace(0., 2*np.pi, num=500)
    xx = np.outer(np.sin(tg), rg)
    yy = np.outer(np.cos(tg), rg)
    lon, lat = np.meshgrid(np.degrees(pg), np.degrees(np.pi/2 - tg))

    uphys = usp.eval_mesh(rg, tg)
    bphys = bsp.eval_mesh(rg, tg)
    uphys_curl = usp_curl.eval_mesh(rg, tg)

    phi_rotate = -np.angle(uphys['p'].flatten()[np.argmax(np.abs(uphys['p']))])
    norm = 1/np.max(np.abs(uphys['p']))
    phase = np.exp(1j*phi_rotate)
    cfact = norm*phase
    ir = np.argmin(np.abs(rg/ro - rratio_sect))
    r_sect = rg[ir]

    vmax_surf = np.abs(cfact)*np.max(np.abs(uphys_curl['r'][:, ir]))
    vmax_r = np.abs(cfact)*np.max(np.abs(uphys['r']))
    vmax_t = np.abs(cfact)*np.max(np.abs(uphys['t']))
    vmax_p = np.abs(cfact)*np.max(np.abs(uphys['p']))

    bmax_surf = np.abs(cfact)*np.max(np.abs(bphys['r'][:, ir]))
    bmax_r = np.abs(cfact)*np.max(np.abs(bphys['r']))
    bmax_t = np.abs(cfact)*np.max(np.abs(bphys['t']))
    bmax_p = np.abs(cfact)*np.max(np.abs(bphys['p']))

    tfact = np.exp(w*t)

    fig = plt.figure(figsize=(12, 10))
    gs = mpl.gridspec.GridSpec(2, 5)

    ax = fig.add_subplot(gs[0,:2], projection=ccrs.Orthographic(central_latitude=20, central_longitude=0))
    fsurf = np.outer((cfact*tfact)*uphys_curl['r'][:, ir], np.exp(1j*m*pg))
    plottings.plot_sphere(lon, lat, fsurf, ax, vcap=vmax_surf)
    ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')

    ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*uphys['r'], fig, gs[0,2], vcap=vmax_r, title=r'$u_r$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*uphys['t'], fig, gs[0,3], vcap=vmax_t, title=r'$u_\theta$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*uphys['p'], fig, gs[0,4], vcap=vmax_p, title=r'$u_\phi$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')

    ax = fig.add_subplot(gs[1,:2], projection=ccrs.Orthographic(central_latitude=20, central_longitude=0))
    fsurf = np.outer((cfact*tfact)*bphys['r'][:, ir], np.exp(1j*m*pg))
    plottings.plot_sphere(lon, lat, fsurf, ax, vcap=bmax_surf)
    ax.set_title(r'$b_r$')

    ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*bphys['r'], fig, gs[1,2], vcap=bmax_r, title=r'$b_r$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*bphys['t'], fig, gs[1,3], vcap=bmax_t, title=r'$b_\theta$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*bphys['p'], fig, gs[1,4], vcap=bmax_p, title=r'$b_\phi$')
    plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')

    plt.show()


def visual_mode_batch_t(
        usp: qfield.ShellVectorTorPol_m, 
        bsp: qfield.ShellVectorTorPol_m, 
        usp_curl: qfield.ShellVectorTorPol_m, 
        w: complex = 0.,
        time_pts: np.ndarray = np.array([0]),
        rratio_sect: float = 1.0,
        save_name: Optional[str] = None,
    ):

    ri, ro = usp.ri, usp.ro
    m = usp.m

    xg, wx = special.roots_chebyt(100)
    zg, wz = special.roots_legendre(200)
    rg = usp.r_basis.x2g(np.r_[-1., xg, +1.])
    tg = usp.t_basis.z2t(np.r_[-1., zg, 1.])
    pg = np.linspace(0., 2*np.pi, num=500)
    xx = np.outer(np.sin(tg), rg)
    yy = np.outer(np.cos(tg), rg)
    lon, lat = np.meshgrid(np.degrees(pg), np.degrees(np.pi/2 - tg))

    uphys = usp.eval_mesh(rg, tg)
    bphys = bsp.eval_mesh(rg, tg)
    uphys_curl = usp_curl.eval_mesh(rg, tg)

    phi_rotate = -np.angle(uphys['p'].flatten()[np.argmax(np.abs(uphys['p']))])
    norm = 1/np.max(np.abs(uphys['p']))
    phase = np.exp(1j*phi_rotate)
    cfact = norm*phase
    ir = np.argmin(np.abs(rg/ro - rratio_sect))
    r_sect = rg[ir]

    vmax_surf = np.abs(cfact)*np.max(np.abs(uphys_curl['r'][:, ir]))
    vmax_r = np.abs(cfact)*np.max(np.abs(uphys['r']))
    vmax_t = np.abs(cfact)*np.max(np.abs(uphys['t']))
    vmax_p = np.abs(cfact)*np.max(np.abs(uphys['p']))

    bmax_surf = np.abs(cfact)*np.max(np.abs(bphys['r'][:, ir]))
    bmax_r = np.abs(cfact)*np.max(np.abs(bphys['r']))
    bmax_t = np.abs(cfact)*np.max(np.abs(bphys['t']))
    bmax_p = np.abs(cfact)*np.max(np.abs(bphys['p']))

    fig = plt.figure(figsize=(12, 10))

    for it, t in enumerate(time_pts):

        tfact = np.exp(w*t)
        fig.clear()
        gs = mpl.gridspec.GridSpec(2, 5)

        ax = fig.add_subplot(gs[0,:2], projection=ccrs.Orthographic(central_latitude=20, central_longitude=0))
        fsurf = np.outer((cfact*tfact)*uphys_curl['r'][:, ir], np.exp(1j*m*pg))
        plottings.plot_sphere(lon, lat, fsurf, ax, vcap=vmax_surf)
        ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')

        ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*uphys['r'], fig, gs[0,2], vcap=vmax_r, title=r'$u_r$')
        plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
        ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*uphys['t'], fig, gs[0,3], vcap=vmax_t, title=r'$u_\theta$')
        plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
        ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*uphys['p'], fig, gs[0,4], vcap=vmax_p, title=r'$u_\phi$')
        plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')

        ax = fig.add_subplot(gs[1,:2], projection=ccrs.Orthographic(central_latitude=20, central_longitude=0))
        fsurf = np.outer((cfact*tfact)*bphys['r'][:, ir], np.exp(1j*m*pg))
        plottings.plot_sphere(lon, lat, fsurf, ax, vcap=bmax_surf)
        ax.set_title(r'$b_r$')

        ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*bphys['r'], fig, gs[1,2], vcap=bmax_r, title=r'$b_r$')
        plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
        ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*bphys['t'], fig, gs[1,3], vcap=bmax_t, title=r'$b_\theta$')
        plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')
        ax = plottings.plot_v_comp(xx, yy, (cfact*tfact)*bphys['p'], fig, gs[1,4], vcap=bmax_p, title=r'$b_\phi$')
        plottings.plot_r_outline([ri, ro], tg, ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], tg, ax, linewidth=0.5, linestyle='--')

        fig.suptitle(f't={t:.5f} ($\\times$ rotation period)')
        plt.draw()
        plt.pause(0.01)
        if save_name is not None:
            plottings.figsave(fig, save_name + f'_{it:04d}', formats=('jpg',), dpi=100, overwrite=True)

    plt.show()


def visual_qsphere_t(
        usp: qfield.ShellVectorTorPol_m, 
        bsp: qfield.ShellVectorTorPol_m, 
        usp_curl: qfield.ShellVectorTorPol_m, 
        w: complex = 0.,
        t: float = 0.,
        rratio_sect: np.ndarray = np.array([1.0]),
    ):

    ri, ro = usp.ri, usp.ro
    m = usp.m

    zg, wz = special.roots_legendre(400)
    rg = ro*rratio_sect
    tg = usp.t_basis.z2t(np.r_[-1., zg, 1.])
    pg = np.linspace(-np.pi, np.pi, num=500)
    xx = np.outer(np.sin(tg), rg)
    yy = np.outer(np.cos(tg), rg)
    lon, lat = np.meshgrid(np.degrees(pg), np.degrees(np.pi/2 - tg))

    uphys = usp.eval_mesh(rg, tg)
    bphys = bsp.eval_mesh(rg, tg)
    uphys_curl = usp_curl.eval_mesh(rg, tg)

    phi_rotate = -np.angle(uphys['p'].flatten()[np.argmax(np.abs(uphys['p']))])
    norm = 1/np.max(np.abs(uphys['p']))
    phase = np.exp(1j*phi_rotate)
    cfact = norm*phase

    vmax_surf = np.abs(cfact)*np.max(np.abs(uphys_curl['r']))
    bmax_surf = np.abs(cfact)*np.max(np.abs(bphys['r']))

    tfact = np.exp(w*t)

    proj = ccrs.Orthographic(central_latitude=20, central_longitude=0)
    aspect = (5, 5)
    # proj = ccrs.Mollweide(central_longitude=0)
    # aspect = (8, 5)

    nrows = rratio_sect.size
    fig = plt.figure(figsize=(1*aspect[0], nrows*aspect[1]))
    gs = mpl.gridspec.GridSpec(nrows, 1)

    for ir, r_sect in enumerate(rratio_sect):

        # ax = fig.add_subplot(gs[ir, 0], projection=proj)
        # fsurf = np.outer((cfact*tfact)*uphys_curl['r'][:, ir], np.exp(1j*m*pg))
        # plottings.plot_sphere(lon, lat, fsurf, ax, vcap=vmax_surf)
        # ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')

        # ax = fig.add_subplot(gs[ir, 1], projection=proj)
        # fsurf = np.outer((cfact*tfact)*bphys['r'][:, ir], np.exp(1j*m*pg))
        # plottings.plot_sphere(lon, lat, fsurf, ax, vcap=bmax_surf)
        # ax.set_title(r'$b_r$')

        ax = fig.add_subplot(gs[ir, 0], projection=proj)
        hdim = pg.size // 2
        fsurf = np.outer((cfact*tfact)*uphys_curl['r'][:, ir], np.exp(1j*m*pg[:hdim]))
        plottings.plot_sphere(lon[:,:hdim], lat[:,:hdim], fsurf, ax, vcap=vmax_surf, cbar=False)
        fsurf = np.outer((cfact*tfact)*bphys['r'][:, ir], np.exp(1j*m*pg[hdim:]))
        plottings.plot_sphere(lon[:,hdim:], lat[:,hdim:], fsurf, ax, vcap=bmax_surf, cbar=False)
        # ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')

    plt.show()


def visual_qsphere3D_batch_t(
        usp: qfield.ShellVectorTorPol_m, 
        bsp: qfield.ShellVectorTorPol_m, 
        usp_curl: qfield.ShellVectorTorPol_m, 
        w: complex = 0.,
        t_array: np.ndarray = np.array([0.,]),
        rratio_sect: tuple = (0., 1.),
        save_name: Optional[str] = None
    ):

    ri, ro = usp.ri, usp.ro
    m = usp.m
    rmin, rmax = ro*rratio_sect[0], ro*rratio_sect[1]

    zg, wz = special.roots_legendre(1000)
    rg = np.linspace(rmin, rmax, num=100)
    tg = usp.t_basis.z2t(np.r_[-1., zg, 1.])
    pg = np.linspace(-np.pi, np.pi, num=1000)

    # ip_l = np.argmin(np.abs(pg - (-1/7)*np.pi))
    # ip_r = np.argmin(np.abs(pg - (+1/7)*np.pi))
    # ip_ub = np.argmin(np.abs(pg - (0)*np.pi))

    ip_l = np.argmin(np.abs(pg - (0)*np.pi))
    ip_r = np.argmin(np.abs(pg - (+1/2)*np.pi))
    ip_ub = np.argmin(np.abs(pg - (+1/2)*np.pi))

    xs = np.outer(np.sin(tg), np.cos(pg))
    ys = np.outer(np.sin(tg), np.sin(pg))
    zs = np.outer(np.cos(tg), np.ones_like(pg))
    xl = np.outer(np.sin(tg), rg[1:])*np.cos(pg[ip_l])
    yl = np.outer(np.sin(tg), rg[1:])*np.sin(pg[ip_l])
    xr = np.outer(np.sin(tg), rg[1:])*np.cos(pg[ip_r])
    yr = np.outer(np.sin(tg), rg[1:])*np.sin(pg[ip_r])
    zm = np.outer(np.cos(tg), rg[1:])

    uphys = usp.eval_mesh(rg, tg)
    bphys = bsp.eval_mesh(rg, tg)
    uphys_curl = usp_curl.eval_mesh(rg, tg)

    phi_rotate = -np.angle(uphys['p'].flatten()[np.argmax(np.abs(uphys['p']))])
    norm = 1/np.max(np.abs(uphys['p']))
    phase = np.exp(1j*phi_rotate)
    cfact = norm*phase

    vmax = np.abs(cfact)*np.max(np.abs(uphys_curl['r']))
    bmax = np.abs(cfact)*np.max(np.abs(bphys['r']))

    vnorm = mpl.colors.CenteredNorm(halfrange=vmax)
    bnorm = mpl.colors.CenteredNorm(halfrange=bmax)
    vticks = mpl.ticker.MaxNLocator(nbins=8)
    bticks = mpl.ticker.MaxNLocator(nbins=8)
    vannot = {vnorm(tick): f'{tick:.2e}' for tick in vticks.tick_values(-vmax, +vmax)}
    bannot = {bnorm(tick): f'{tick:.2e}' for tick in bticks.tick_values(-bmax, +bmax)}

    # vlinthresh = 10**(int(np.log10(vmax)) - 3)
    # blinthresh = 10**(int(np.log10(bmax)) - 3)
    # vnorm = mpl.colors.SymLogNorm(vlinthresh, linscale=0.1, vmin=-vmax, vmax=+vmax)
    # bnorm = mpl.colors.SymLogNorm(blinthresh, linscale=0.1, vmin=-bmax, vmax=+bmax)
    # vticks = mpl.scale.SymmetricalLogLocator(linthresh=vlinthresh, base=10, subs=np.logspace(0., 1., num=2)[:-1])
    # bticks = mpl.scale.SymmetricalLogLocator(linthresh=blinthresh, base=10, subs=np.logspace(0., 1., num=2)[:-1])
    # vannot = {vnorm(tick): f'{tick:.2e}' for tick in vticks.tick_values(-vmax, +vmax)}
    # bannot = {bnorm(tick): f'{tick:.2e}' for tick in bticks.tick_values(-bmax, +bmax)}

    vfield = r'$\hat{\mathbf{r}}\cdot\nabla\times \mathbf{v}$'
    bfield = r'$b_r$'

    pl = pv.Plotter(image_scale=2)
    pl.show(interactive=True, interactive_update=True)
    arg_vbar = dict(height=0.5, vertical=True, position_x=0.08, position_y=0.25, n_labels=0, 
        label_font_size=14, font_family='times')
    arg_bbar = dict(height=0.5, vertical=True, position_x=0.9, position_y=0.25, n_labels=0, 
        label_font_size=14, font_family='times')
    lighting = False
    smoothing = True

    for it, t in enumerate(t_array):
        print(f"{it+1}/{t_array.size} snap @ t={t:.2f}. ", end='')
        pl.clear()
        tfact = np.exp(w*t)

        fsurf = np.outer((cfact*tfact)*uphys_curl['r'][:, 0], np.exp(1j*m*pg[:(ip_ub+1)]))
        mesh = pv.StructuredGrid(rmin*xs[:,:(ip_ub+1)], rmin*ys[:,:(ip_ub+1)], rmin*zs[:,:(ip_ub+1)])
        mesh.point_data[vfield] = vnorm(np.ravel(np.real(fsurf), order='F'))
        _ = pl.add_mesh(mesh, lighting=lighting, smooth_shading=smoothing, clim=(0., 1.), 
            cmap='coolwarm', scalar_bar_args=arg_vbar)

        fsurf = np.outer((cfact*tfact)*bphys['r'][:, 0], np.exp(1j*m*pg[(ip_ub):]))
        mesh = pv.StructuredGrid(rmin*xs[:,(ip_ub):], rmin*ys[:,(ip_ub):], rmin*zs[:,(ip_ub):])
        mesh.point_data[bfield] = bnorm(np.ravel(np.real(fsurf), order='F'))
        _ = pl.add_mesh(mesh, lighting=lighting, smooth_shading=smoothing, clim=(0., 1.), 
            cmap='coolwarm', scalar_bar_args=arg_bbar)
        mesh_edge = mesh.extract_feature_edges()
        _ = pl.add_mesh(mesh_edge, line_width=1, color='k')

        fsurf = np.outer((cfact*tfact)*uphys_curl['r'][:, -1], np.exp(1j*m*pg[:(ip_l+1)]))
        mesh = pv.StructuredGrid(rmax*xs[:,:(ip_l+1)], rmax*ys[:,:(ip_l+1)], rmax*zs[:,:(ip_l+1)])
        mesh.point_data[vfield] = vnorm(np.ravel(np.real(fsurf), order='F'))
        _ = pl.add_mesh(mesh, lighting=lighting, smooth_shading=smoothing, clim=(0., 1.), cmap='coolwarm')

        fsurf = np.outer((cfact*tfact)*bphys['r'][:, -1], np.exp(1j*m*pg[ip_r:]))
        mesh = pv.StructuredGrid(rmax*xs[:,ip_r:], rmax*ys[:,ip_r:], rmax*zs[:,ip_r:])
        mesh.point_data[bfield] = bnorm(np.ravel(np.real(fsurf), order='F'))
        _ = pl.add_mesh(mesh, lighting=lighting, smooth_shading=smoothing, clim=(0., 1.), cmap='coolwarm')

        fsurf = (np.exp(1j*m*pg[ip_l])*cfact*tfact)*uphys_curl['r'][:, 1:]
        mesh = pv.StructuredGrid(xl, yl, zm)
        mesh.point_data[vfield] = vnorm(np.ravel(np.real(fsurf), order='F'))
        _ = pl.add_mesh(mesh, lighting=lighting, smooth_shading=smoothing, clim=(0., 1.), cmap='coolwarm', annotations=vannot)
        mesh_edge = mesh.extract_feature_edges()
        _ = pl.add_mesh(mesh_edge, line_width=1, color='k')

        fsurf = (np.exp(1j*m*pg[ip_r])*cfact*tfact)*bphys['r'][:, 1:]
        mesh = pv.StructuredGrid(xr, yr, zm)
        mesh.point_data[bfield] = bnorm(np.ravel(np.real(fsurf), order='F'))
        _ = pl.add_mesh(mesh, lighting=lighting, smooth_shading=smoothing, clim=(0., 1.), cmap='coolwarm', annotations=bannot)
        mesh_edge = mesh.extract_feature_edges()
        _ = pl.add_mesh(mesh_edge, line_width=1, color='k')

        pl.add_text(f"t = {t:<5.2f} (x Rotation Period)", position='upper_edge', font='times', font_size=10)
        pl.camera.position = (16., 0., 0.)
        pl.update(1000)

        if save_name:
            pl.screenshot(save_name + f'_{it:04d}.jpg')
            print(f"Saved to {save_name}_{it:04d}.jpg ", end='')
        print(flush=True)
    
    pl.show()


def visual_qsphere3D_multir_batch_t(
        usp: qfield.ShellVectorTorPol_m, 
        bsp: qfield.ShellVectorTorPol_m, 
        usp_curl: qfield.ShellVectorTorPol_m, 
        w: complex = 0.,
        t_array: np.ndarray = np.array([0.,]),
        psects: tuple = (-np.pi, np.pi),
        fviews: tuple = ('zr',),
        rratios: tuple = (1.,),
        npts_r: int = 500,
        npts_p: int = 500,
        save_name: Optional[str] = None,
        off_screen: bool = False
    ):

    ri, ro = usp.ri, usp.ro
    m = usp.m

    assert len(fviews) == len(psects) - 1
    assert len(rratios) == len(psects) - 1

    r_control = ro*np.sort(np.array(tuple(set(rratios))))
    dr_min = (ro - ri)/npts_r
    rg = list()
    for ir, rmin in enumerate(r_control[:-1]):
        rmax = r_control[ir+1]
        if (rmax - rmin) >= 100*dr_min:
            rg.append(np.linspace(rmin, rmax, num=100)[:-1])
        elif (rmax - rmin) <= 10*dr_min:
            rg.append(np.linspace(rmin, rmax, num=10)[:-1])
        else:
            rg.append(np.arange(rmin, rmax - 1e-2*dr_min, dr_min))
    rg.append(r_control[-1])
    rg = np.r_[*rg]

    zg, _ = special.roots_legendre(1000)
    tg = usp.t_basis.z2t(np.r_[-1., zg, 1.])

    pg = list()
    dp = 2*np.pi/npts_p
    for ip, pmin in enumerate(psects[:-1]):
        pmax = psects[ip+1]
        if (pmax - pmin) >= 10*dp:
            pg.append(np.arange(pmin, pmax - 1e-2*dp, dp))
        else:
            pg.append(np.linspace(pmin, pmax, num=10)[:-1])
    pg.append(psects[-1])
    pg = np.r_[*pg]

    xs = np.outer(np.sin(tg), np.cos(pg))
    ys = np.outer(np.sin(tg), np.sin(pg))
    zs = np.outer(np.cos(tg), np.ones_like(pg))
    sm = np.outer(np.sin(tg), rg)
    zm = np.outer(np.cos(tg), rg)

    uphys = usp.eval_mesh(rg, tg)
    bphys = bsp.eval_mesh(rg, tg)
    uphys_curl = usp_curl.eval_mesh(rg, tg)

    phi_rotate = -np.angle(uphys['p'].flatten()[np.argmax(np.abs(uphys['p']))])
    norm = 1/np.max(np.abs(uphys['p']))
    phase = np.exp(1j*phi_rotate)
    cfact = norm*phase

    vmax = np.abs(cfact)*np.max(np.abs(uphys_curl['r']))
    bmax = np.abs(cfact)*np.max(np.abs(bphys['r']))
    vnorm = mpl.colors.CenteredNorm(halfrange=vmax)
    bnorm = mpl.colors.CenteredNorm(halfrange=bmax)
    vticks = mpl.ticker.MaxNLocator(nbins=8)
    bticks = mpl.ticker.MaxNLocator(nbins=8)
    vannot = {vnorm(tick): f'{tick:.2e}' for tick in vticks.tick_values(-vmax, +vmax)}
    bannot = {bnorm(tick): f'{tick:.2e}' for tick in bticks.tick_values(-bmax, +bmax)}

    fields = {
        'ur': uphys['r'],
        'zr': uphys_curl['r'],
        'br': bphys['r'],
    }
    valnames = {
        'ur': r'$v_r$',
        'zr': r'$\hat{\mathbf{r}}\cdot\nabla\times \mathbf{v}$',
        'br': r'$b_r$'
    }

    pl = pv.Plotter(image_scale=3, off_screen=off_screen)
    pl.show(interactive=False, interactive_update=True)
    arg_cbars = {
        'zr': dict(height=0.5, vertical=True, position_x=0.08, position_y=0.25, n_labels=0, 
                   label_font_size=14, font_family='times'),
        'br': dict(height=0.5, vertical=True, position_x=0.9, position_y=0.25, n_labels=0, 
                   label_font_size=14, font_family='times')
    }
    color_norms = {
        'zr': vnorm,
        'br': bnorm
    }
    arg_mesh = {
        'zr': dict(lighting=False, smooth_shading=True, clim=(0.,1.), cmap='Spectral_r', 
                   scalar_bar_args=arg_cbars['zr'], annotations=vannot),
        'br': dict(lighting=False, smooth_shading=True, clim=(0.,1.), cmap='CET_D1', 
                   scalar_bar_args=arg_cbars['br'], annotations=bannot)
    }

    for it, t in enumerate(t_array):
        print(f"{it+1}/{t_array.size} snap @ t={t:.2f}. ", end='')
        pl.clear()
        pl.enable_lightkit()
        tfact = np.exp(w*t)

        for isect in range(len(psects) - 1):
            
            pmin = psects[isect]
            pmax = psects[isect+1]
            ipmin = np.argmin(np.abs(pg - pmin))
            ipmax = np.argmin(np.abs(pg - pmax))
            rtmp = ro*rratios[isect]
            ir = np.argmin(np.abs(rg - rtmp))
            fcode = fviews[isect]

            # Radial slice
            pslice = slice(ipmin, ipmax+1)
            fsurf = np.outer((cfact*tfact)*fields[fcode][:, ir], np.exp(1j*m*pg[pslice]))
            mesh = pv.StructuredGrid(rtmp*xs[:,pslice], rtmp*ys[:,pslice], rtmp*zs[:,pslice])
            mesh.point_data[valnames[fcode]] = color_norms[fcode](np.ravel(np.real(fsurf), order='F'))
            _ = pl.add_mesh(mesh, **arg_mesh[fcode])
            if not arg_mesh['zr']['lighting']:
                mesh_edge = mesh.extract_feature_edges()
                _ = pl.add_mesh(mesh_edge, line_width=3, color='k')

            # Meridional slice
            if isect >= len(psects) - 2:
                continue

            r_next = ro*rratios[isect+1]
            ir_next = np.argmin(np.abs(rg - r_next))
            if ir == ir_next:
                continue

            rslice = slice(min(ir, ir_next)+1, max(ir, ir_next))
            fmerd = (np.exp(1j*m*pmax)*cfact*tfact)*fields[fcode][:, rslice]
            mesh = pv.StructuredGrid(sm[:, rslice]*np.cos(pmax), sm[:, rslice]*np.sin(pmax), zm[:, rslice])
            mesh.point_data[valnames[fcode]] = color_norms[fcode](np.ravel(np.real(fmerd), order='F'))
            _ = pl.add_mesh(mesh, **arg_mesh[fcode])
            if not arg_mesh['zr']['lighting']:
                mesh_edge = mesh.extract_feature_edges()
                _ = pl.add_mesh(mesh_edge, line_width=3, color='k')

        # pl.add_text(f"$t=$ {t:<5.2f} $T_\\Omega$", position='upper_edge', font='times', font_size=20)
        pl.add_text(f"$t=$ {t:<5.2f} $T_\\Omega$\n$r=$ {rratios[0]:<5.2f} $R_\\odot$", 
            position='upper_edge', font='times', font_size=20)
        pl.camera.position = (16., 0., 4.)
        pl.update(1000)

        if save_name:
            pl.screenshot(save_name + f'_{(it):04d}.jpg')
            print(f"Saved to {save_name}_{(it):04d}.jpg ", end='')
        print(flush=True)
    
    pl.show()


def main():

    ri, ro = 0.71, 1
    D = ro - ri
    ri /= D
    ro /= D

    geometry = (ri, ro)
    resolution = (55, 54 + 25, 25)

    # mat_dir = './Rotating_SphShells/solar_modes/bg_T2CI/Mat_T2_m25_54x79_BCSF/ops'
    # params = {'Ek': 1e-4, 'Em': 1e-4, 'Le': 5e-2}
    # w0 = -9.817e-03+7.977e-02j
    # w0 = -4.468e-02+2.230e-01j
    # w0 = -3.919e-02-1.813e-01j

    mat_dir = './Rotating_SphShells/solar_modes/bg_SolarT/Mat_SolarT_m25_54x79_BCSF/ops'
    params = {'Ek': 1.19e-3, 'Em': 1.19e-3, 'Le': 4.0e-2}
    w0 = -1.102e-01+9.832e-02j
    # w0 = -2.076e-01+2.243e-01j
    # w0 = -2.150e-01-1.887e-01j

    # w, v = calc_mode(mat_dir, params, w0)
    # np.savez(f"{mat_dir[:-3]}eigv_Ek{params['Ek']:.2e}_Em{params['Em']:.2e}_Le{params['Le']:.2e}_SlowAlfven", w=w, v=v)
    
    eigv_obj = np.load(f"{mat_dir[:-3]}eigv_Ek{params['Ek']:.2e}_Em{params['Em']:.2e}_Le{params['Le']:.2e}_SlowAlfven.npz")
    w, v = eigv_obj['w'], eigv_obj['v']
    usp, bsp, usp_curl = build_mode_phys(geometry, resolution, v)
    w = 1j*np.imag(w)

    # visual_mode(usp, bsp, usp_curl)
    # visual_mode_t(usp, bsp, usp_curl, w=w, t=5.)
    # time_pts = np.linspace(0., 100., num=120)
    # visual_mode_batch_t(usp, bsp, usp_curl, w=w, time_pts=time_pts, 
    #     save_name='./Rotating_SphShells/solar_modes/bg_SolarT/mode_snaps/snap')
    # visual_qsphere_t(usp, bsp, usp_curl, w, rratio_sect=np.array([0.9, 1.]))
    # visual_qsphere3D_batch_t(usp, bsp, usp_curl, w, rratio_sect=(0.8, 1.0), t_array=np.linspace(0., 100., num=120)[:1], 
    #     # save_name='./Rotating_SphShells/solar_modes/bg_T2CI/mode_snaps/snap3D'
    # )
    # visual_qsphere3D_vb_batch_t(usp, bsp, usp_curl, w, rratio_sect=1.0, t_array=np.linspace(0., 100., num=120)[:1])
    visual_qsphere3D_multir_batch_t(
        usp, bsp, usp_curl, w, 
        # t_array=np.linspace(0., 2*np.pi/np.abs(np.imag(w)), num=240)[:1], 
        t_array=np.arange(0., 30, 0.1),
        # psects=(-np.pi, -(1/12+1/8)*np.pi, -np.pi/12, +np.pi/4, +(3/8)*np.pi, +np.pi),
        # fviews=('zr', 'zr', 'zr', 'zr', 'zr'), 
        # rratios=(1.0, 0.9, 0.8, 0.9, 1.0),
        # psects=(-np.pi, -np.pi/12, +(5/16)*np.pi, +np.pi),
        # fviews=('br', 'br', 'br'), 
        # rratios=(1.0, 0.85, 1.0),
        psects=(-np.pi, np.pi/12, +np.pi),
        fviews=('zr', 'br'), 
        rratios=(0.85, 0.85),
        save_name='./Rotating_SphShells/solar_modes/bg_SolarT/mode_snaps/snap3D_Slow',
        off_screen=True
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import os, h5py, warnings, argparse
import numpy as np
import dedalus.public as d3
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import interpolate
from scipy.sparse import linalg as spla
from models import evp_shell
from utils import plottings, eigs
cwd = os.getcwd()


def bg_func_axisym(r_grid, t_grid, Ri, Ro, bg='U'):

    if bg == 'U':
        Bp = np.zeros((1, t_grid.size, r_grid.size))
        Bt = -np.ones_like(r_grid)*np.sin(t_grid)
        Br = np.ones_like(r_grid)*np.cos(t_grid)
    
    if bg == 'S1':
        alpha = 1/(4*(Ro - Ri)*(2*Ro - Ri))
        Bp = np.zeros((1, t_grid.size, r_grid.size))
        Bt = alpha*(-9*Ro*r_grid + 2*(2*Ri**2 + 4*Ro**2) - 3*Ri**2*Ro/r_grid)*np.sin(t_grid)
        Br = alpha*(+6*Ro*r_grid - 2*(2*Ri**2 + 4*Ro**2) + 6*Ri**2*Ro/r_grid)*np.cos(t_grid)
    
    if bg == 'T2':
        alpha = 4/(Ro - Ri)**2
        Bp = alpha*(r_grid - Ro)*(r_grid - Ri)*np.sin(2*t_grid)
        Bt = np.zeros((1, t_grid.size, r_grid.size))
        Br = np.zeros((1, t_grid.size, r_grid.size))
    
    return Bp, Bt, Br


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


def quiver_sphere(lon, lat, vx, vy, ax, vcap=1e-7):
    if np.max(np.abs(vx)**2 + np.abs(vy**2)) < vcap**2:
        return ax
    ax.quiver(
        lon, lat, np.real(vx), np.real(vy),
        width=0.001, headlength=2, transform=ccrs.PlateCarree(), zorder=5
    )
    return ax


def calc_hydro_fields(model, v_view, r_sect=1.):

    m_val = model.resolution[-1]
    for var in model.solver.state:
        var['c'] = 0
    model.subprob.subsystems[0].scatter(v_view, model.solver.state)

    u = model.u.evaluate()
    dist = model.fields['dist']
    shell = model.fields['shell']

    # Surface fields
    lon_view = np.linspace(-180, +180, num=500)
    scales_view = (1, 8, 4)
    _, t_view, _ = dist.local_grids(shell, scales=scales_view)
    t_view = t_view[0, :, 0]
    lat_view = np.degrees(np.pi/2 - t_view)

    u_surf = u(r=r_sect).evaluate()
    u_surf.change_scales(scales_view)
    zeta = d3.curl(u)
    zeta = zeta(r=1).evaluate()
    zeta.change_scales(scales_view)

    f_anchor = zeta['g'][2, 0, :, 0]
    f_anchor = f_anchor[np.argmax(np.abs(f_anchor[:f_anchor.size//2]))]
    phi_rotate = -np.angle(f_anchor)/m_val

    phase = np.exp(1j*m_val*(np.radians(lon_view) + phi_rotate))
    u_surf_p = np.outer(u_surf['g'][0, 0, :, 0], phase)
    u_surf_t = np.outer(u_surf['g'][1, 0, :, 0], phase)
    zeta_val = np.outer(zeta['g'][2, 0, :, 0], phase)

    lat_mesh, lon_mesh = np.meshgrid(lat_view, lon_view, indexing='ij')

    # Meridional fields
    phi_view = 0
    arg_extra = np.pi/2
    scales_view = (1, 8, 4)
    phase = np.exp(1j*(m_val*(np.radians(phi_view) + phi_rotate) + arg_extra))

    _, t_view, r_view = dist.local_grids(shell, scales=scales_view)
    u.change_scales(scales_view)
    u_p = u['g'][0, 0, ...]*phase
    u_t = u['g'][1, 0, ...]*phase
    u_r = u['g'][2, 0, ...]*phase
    u_s = u_r*np.sin(t_view[0, ...]) + u_t*np.cos(t_view[0, ...])
    u_z = u_r*np.cos(t_view[0, ...]) - u_t*np.sin(t_view[0, ...])

    t_mesh, r_mesh = np.meshgrid(t_view[0, :, 0], r_view[0, 0, :], indexing='ij')
    s_mesh, z_mesh = r_mesh*np.sin(t_mesh), r_mesh*np.cos(t_mesh)

    coords = {
        'lon': lon_mesh,
        'lat': lat_mesh,
        's': s_mesh,
        'z': z_mesh
    }
    mhd_fields = {
        'u_r': u_r,
        'u_t': u_t,
        'u_p': u_p,
        'u_surf_t': u_surf_t,
        'u_surf_p': u_surf_p,
        'zeta_val': zeta_val,
    }
    return coords, mhd_fields


def calc_mhd_fields(model, v_view):

    m_val = model.resolution[-1]
    for var in model.solver.state:
        var['c'] = 0
    model.subprob.subsystems[0].scatter(v_view, model.solver.state)

    u = model.u
    b = model.b
    dist = model.fields['dist']
    shell = model.fields['shell']

    # Surface fields
    lon_view = np.linspace(-180, +180, num=100)
    scales_view = (1, 4, 1)
    _, t_view, _ = dist.local_grids(shell, scales=scales_view)
    t_view = t_view[0, :, 0]
    lat_view = np.degrees(np.pi/2 - t_view)

    u_surf = u(r=1).evaluate()
    u_surf.change_scales(scales_view)

    up_anchor = u_surf['g'][0, 0, :, 0]
    up_anchor = up_anchor[np.argmax(np.abs(up_anchor[:up_anchor.size//2]))]
    phi_rotate = -np.angle(up_anchor)/m_val

    phase = np.exp(1j*m_val*(np.radians(lon_view) + phi_rotate))
    u_surf_p = np.outer(u_surf['g'][0, 0, :, 0], phase)
    u_surf_t = np.outer(u_surf['g'][1, 0, :, 0], phase)
    zeta = d3.curl(u)
    zeta = zeta(r=1).evaluate()
    zeta.change_scales(scales_view)
    zeta_val = np.outer(zeta['g'][2, 0, :, 0], phase)

    b_surf = b(r=1).evaluate()
    b_surf.change_scales(scales_view)
    b_surf_p = np.outer(b_surf['g'][0, 0, :, 0], phase)
    b_surf_t = np.outer(b_surf['g'][1, 0, :, 0], phase)
    b_surf_r = np.outer(b_surf['g'][2, 0, :, 0], phase)

    lat_mesh, lon_mesh = np.meshgrid(lat_view, lon_view, indexing='ij')

    # Meridional fields
    phi_view = 0
    arg_extra = np.pi/4
    scales_view = (1, 4, 4)
    phase = np.exp(1j*(m_val*(np.radians(phi_view) + phi_rotate) + arg_extra))

    _, t_view, r_view = dist.local_grids(shell, scales=scales_view)
    u.change_scales(scales_view)
    u_p = u['g'][0, 0, ...]*phase
    u_t = u['g'][1, 0, ...]*phase
    u_r = u['g'][2, 0, ...]*phase
    u_s = u_r*np.sin(t_view[0, ...]) + u_t*np.cos(t_view[0, ...])
    u_z = u_r*np.cos(t_view[0, ...]) - u_t*np.sin(t_view[0, ...])

    b_field = b.evaluate()
    b_field.change_scales(scales_view)
    b_p = b_field['g'][0, 0, ...]*phase
    b_t = b_field['g'][1, 0, ...]*phase
    b_r = b_field['g'][2, 0, ...]*phase

    t_mesh, r_mesh = np.meshgrid(t_view[0, :, 0], r_view[0, 0, :], indexing='ij')
    s_mesh, z_mesh = r_mesh*np.sin(t_mesh), r_mesh*np.cos(t_mesh)

    coords = {
        'lon': lon_mesh,
        'lat': lat_mesh,
        's': s_mesh,
        'z': z_mesh
    }
    mhd_fields = {
        'u_r': u_r,
        'u_t': u_t,
        'u_p': u_p,
        'u_surf_t': u_surf_t,
        'u_surf_p': u_surf_p,
        'zeta_val': zeta_val,
        'b_r': b_r,
        'b_t': b_t,
        'b_p': b_p,
        'b_surf_r': b_surf_r,
        'b_surf_t': b_surf_t,
        'b_surf_p': b_surf_p,
    }
    return coords, mhd_fields


def main_DR_branch_Ro():

    file_DR_branch = "./Rotating_SphShells/hydro_diff_rot/Incompressible/branches.h5"
    output_dir = "./Rotating_SphShells/hydro_diff_rot/Incompressible/modes_visual"
    ri, ro = 0.71, 1.
    Nr, L, m = 24, 42, 15
    r_sect = 1.
    f_dOmega = get_solar_diff_rot(Omega0=456.)
    # proj = ccrs.Mollweide(central_longitude=0)
    # ncol_proj = 3
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=20)
    ncol_proj = 2

    with h5py.File(file_DR_branch, 'r') as fp:
        Ro_arr = fp["Sectoral_Rossby"]["Ro"][()]
        w_arr = fp["Sectoral_Rossby"][f"m{m}"][()]

    m_dr_shell = evp_shell.ModelEVP_DiffRShell_TorPol((ri, ro), (Nr, L, m), dOmega_func=f_dOmega)
    M_lib = m_dr_shell.precomp_submat(bc='stress-free')

    fig = plt.figure(figsize=(12, 6))

    for i_Ro, Ro in enumerate(Ro_arr):
        w_target = w_arr[i_Ro]
        Ek = 1e-4
        A = M_lib['coriolis'] + Ek*M_lib['viscous_diffusion'] + Ro*M_lib['differential_rotation']
        B = -M_lib['mass']
        # A, B = m_dr_shell.setup_eigenmat(Ek, Ro, set_problem=True, bc='stress-free')
        w_view, v_view = eigs.single_eig(A, B, w_target, nev=1, tol=1e-13, maxiter=50)
        w_view = w_view[0]
        v_view = m_dr_shell.subprob.pre_right @ v_view[:, 0]
        coords, fields = calc_hydro_fields(m_dr_shell, v_view, r_sect=r_sect)
        print(f"Ro: {Ro:.2f} | Target = {w_target:+.2e} | Solved = {w_view:+.2e}", flush=True)
        norm = 1/np.abs(fields['u_t']).max()

        fig.clear()
        gs = fig.add_gridspec(1, ncol_proj+3)

        ax = fig.add_subplot(gs[0, :ncol_proj], projection=proj)
        plottings.plot_sphere(coords['lon'], coords['lat'], norm*fields['zeta_val'], ax)
        # ix, iy = slice(0, None, 3), slice(0, None, 3)
        # quiver_sphere(coords['lon'][0, ix], coords['lat'][iy, 0], fields['u_surf_p'][iy, ix], -fields['u_surf_t'][iy, ix], ax)
        ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')

        ax = plottings.plot_v_comp(coords['s'], coords['z'], norm*fields['u_r'], fig, gs[0,ncol_proj], title=r'$u_r$')
        plottings.plot_r_outline([ri, ro], np.linspace(0, np.pi, num=100), ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], np.linspace(0, np.pi, num=100), ax, linewidth=0.5, linestyle='--')
        ax = plottings.plot_v_comp(coords['s'], coords['z'], norm*fields['u_t'], fig, gs[0,ncol_proj+1], title=r'$u_\theta$')
        plottings.plot_r_outline([ri, ro], np.linspace(0, np.pi, num=100), ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], np.linspace(0, np.pi, num=100), ax, linewidth=0.5, linestyle='--')
        ax = plottings.plot_v_comp(coords['s'], coords['z'], norm*fields['u_p'], fig, gs[0,ncol_proj+2], title=r'$u_\phi$')
        plottings.plot_r_outline([ri, ro], np.linspace(0, np.pi, num=100), ax, linewidth=0.5)
        plottings.plot_r_outline([r_sect], np.linspace(0, np.pi, num=100), ax, linewidth=0.5, linestyle='--')

        fig.suptitle(
            r'$r_i / r_o$ = %.2f,   $Ek$ = %.1e,   $Ro$ = %.2f,   $m$ = %d,   $\lambda$ = %+.3f%+.3fj' 
            % (ri/ro, Ek, Ro, m, np.real(w_view), np.imag(w_view))
        )
        plottings.figsave(fig, os.path.join(output_dir, f"SectRossby_m{m}_Ro{Ro:.2f}"), formats=('jpg',), dpi=200, overwrite=True)
    plt.show()


if __name__ == '__main__':
    main_DR_branch_Ro()

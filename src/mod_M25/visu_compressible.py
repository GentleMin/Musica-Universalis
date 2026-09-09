import numpy as np
import dedalus.public as d3
import logging
from scipy import interpolate, sparse
import os
from scipy.interpolate import interp1d
import argparse

logger = logging.getLogger(__name__)
logger.info("Start!")

parser = argparse.ArgumentParser(description="""
        This is a test.
        """)
parser.add_argument('m', type=int, help="""
        azimuthal wavenumber
        """)
parser.add_argument('-res', type=int, nargs=2, default=[32, 24])
parser.add_argument('-name', default='Rossby0')
args = parser.parse_args()
print(args)

#loading background from Standard model S
V=np.load('data/background.npz')
rs=V['R']
cs=V['cs']
rho0=V['rho0']
p0=V['p0']
Gamma1=V['Gamma1']
T0=V['T0']
g=V['g']
fcs=interp1d(rs,cs)
frho0=interp1d(rs,rho0)
fp0=interp1d(rs,p0)
fT0=interp1d(rs,T0)
fGamma1=interp1d(rs,Gamma1)
fg=interp1d(rs,g)


# Parameters
Nphi = 2*(args.m + 1)
Ntheta = 96
Nr = 24
Ri = 0.71
Ro = 0.985
dtype = np.complex128

Prandtl = 1
gamma = 5/3 
H = 0.0827 # pressure scale height at the base of convection zone
R_sun = 6.96e10
p_bc = 6.16e13
rho_bc = 0.202
Om_sun = 2.87e-6
Turbulent_viscosity = 1e12
g_bc = 5.3e4
Ma = Om_sun*(R_sun)*np.sqrt(rho_bc/p_bc)
Re = Om_sun*(R_sun)**2/Turbulent_viscosity
Pe = Re*Prandtl
Om0 = 456
del_r=0
dir_name= 'results/compressible_solarDR'

os.makedirs(dir_name, exist_ok = True) 


# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype)
shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dtype=dtype)
sphere = shell.outer_surface
phi, theta, r = dist.local_grids(shell)

# Fields
om = dist.Field(name='om')
u = dist.VectorField(coords, name='u', bases=shell)
p = dist.Field(name='p', bases=shell)
tau_u1 = dist.VectorField(coords, bases=sphere)
tau_u2 = dist.VectorField(coords, bases=sphere)
tau_p = dist.Field()
rho = dist.Field(name='rho', bases=shell)
s = dist.Field(name='s', bases=shell)
tau_s1 = dist.Field(bases=sphere)
tau_s2 = dist.Field(bases=sphere)
Iden = d3.Field(dist=dist, bases=(shell.meridional_basis,shell.meridional_basis), tensorsig=(coords,coords), dtype=dtype, name='Iden') #Identity tensor

#Identity tensor
Iden['g'] = 0
for i in range(3):
    Iden['g'][i,i] = 1


# Substitutions
dt = lambda A: -1j*om*A
rvec = dist.VectorField(coords, bases=shell.meridional_basis)
rvec['g'][2] = r
ez = dist.VectorField(coords, bases=shell.meridional_basis)
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta)
er = dist.VectorField(coords, bases=shell.meridional_basis)
er['g'][2] = 1
lift_basis = shell.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
Sigma= grad_u + d3.transpose(grad_u) -2/3*d3.trace(grad_u)*Iden
g = dist.Field(bases=shell.meridional_basis)
g['g'] = fg(r.reshape(r.size))/fg(Ri)
rho0 = dist.Field(bases=shell.meridional_basis)
p0 = dist.Field(bases=shell.meridional_basis)
T0 = dist.Field(bases=shell.meridional_basis)
strain_rate = d3.grad(u) + d3.transpose(d3.grad(u)) -2/3*d3.div(u)*Iden

# Background quantities
T0['g'] = fT0(r.reshape(r.size))/fT0(Ri)
rho0['g'] = frho0(r.reshape(r.size))/frho0(Ri)
p0['g'] = fp0(r.reshape(r.size))/fp0(Ri)

Ome=dist.Field(bases=shell.meridional_basis)
Ome['g']=0
diff_rot_data = np.load('data/diff_rot.npz')
diff_rot = diff_rot_data['ome']
rs= diff_rot_data['r']
thetas = diff_rot_data['theta']
f_new=interpolate.RectBivariateSpline(rs,thetas, diff_rot.T)
tempd=f_new(r.reshape(r.size),theta.reshape(theta.size)[::-1]).T
norm = (tempd -Om0)/Om0
Ome['g']= norm
v0 = dist.VectorField(coords, bases=shell.meridional_basis)
v0['g'][0]=r*np.sin(theta)*Ome['g']
v0['g'][1]=0
v0['g'][2]=0


grad_s = d3.grad(s) + rvec*lift(tau_s1)
nu = dist.Field(bases=shell.meridional_basis)
nu['g'] =  1                                     
S = (d3.grad(u) + d3.transpose(d3.grad(u))+ rvec*lift(tau_u1))
grad_s0 = dist.VectorField(coords, bases=shell.meridional_basis)
diff_rot_data = np.load('data/diff_rot.npz')
del_th = diff_rot_data['dsdt']

rs= diff_rot_data['r']
thetas = diff_rot_data['theta']
f_new=interpolate.RectBivariateSpline(rs,thetas, del_th.T)
tempd=f_new(r.reshape(r.size),theta.reshape(theta.size)[::-1]).T
grad_s0['g'][1]=-tempd/r
Hp = p0['g']/rho0['g']/g['g']*p_bc/rho_bc/g_bc/R_sun
grad_s0['g'][2]=-del_r/Hp
grav_non=fg(Ri)/(Om_sun)**2/R_sun

# Problem
problem = d3.EVP([ p, u, rho, s, tau_u1, tau_u2, tau_p, tau_s1, tau_s2], eigenvalue=om, namespace=locals())
problem.add_equation("dt(rho) + rho0*trace(grad_u)/Ma**2 + grad(rho0)@u/Ma**2 + div(rho*v0) + tau_p = 0")
problem.add_equation("dt(u) + v0@grad_u + u@grad(v0)  + 2*cross(ez, u) + grad(p/rho0) -  s*g/H*er  - 1/Re/rho0*(div(rho0*nu*Sigma)) + lift(tau_u2) = 0")
problem.add_equation("dt(s) + grad_s0@u/Ma**2 + v0@grad_s - 1/Pe/rho0/T0*div(rho0*T0*grad_s) + lift(tau_s2) = 0")
problem.add_equation("p/p0 - gamma*rho/rho0 - gamma*s = 0")
problem.add_equation("radial(u(r=Ri)) = 0")
problem.add_equation("radial(u(r=Ro)) = 0")
problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
problem.add_equation("integ(p/p0) = 0")
problem.add_equation("radial(grad(s)(r=Ro))=0")
problem.add_equation("radial(grad(s)(r=Ri))=0")

# Solver
solver = problem.build_solver(ncc_cutoff=1e-10)
subproblem = solver.subproblems_by_group[(args.m, None, None)]
# solver.build_matrices([subproblem,], ['M', 'L'])
# L_1 = sparse.csc_array(subproblem.L_min)
# M = sparse.csc_array(subproblem.M_min)


def calc_hydro_fields(v_view, r_sect=1.):

    m_val = args.m
    for var in solver.state:
        var['c'] = 0
    subproblem.subsystems[0].scatter(v_view, solver.state)

    # Surface fields
    lon_view = np.linspace(-180, +180, num=500)
    scales_view = (1, 2, 4)
    _, t_view, _ = dist.local_grids(shell, scales=scales_view)
    t_view = t_view[0, :, 0]
    lat_view = np.degrees(np.pi/2 - t_view)

    u_surf = u(r=r_sect).evaluate()
    u_surf.change_scales(scales_view)
    zeta = d3.curl(u)
    zeta = zeta(r=r_sect).evaluate()
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
    scales_view = (1, 2, 4)
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


mat_dir = f"./results/compressible_solarDR/Mat-r0.985_{args.m}x{args.res[0]}x{args.res[1]}"
vec_view = np.load(os.path.join(mat_dir, f"evec-{args.name}.npy"))
# P = sparse.load_npz(os.path.join(mat_dir, "ops/perm.npz"))
# vec_view = P.T @ (P.T @ vec_view)
r_sect = 0.985
coord_view, field_view = calc_hydro_fields(vec_view[:, 0], r_sect=r_sect)


import matplotlib.pyplot as plt
import plottings
from cartopy import crs as ccrs

def view_spec(v_view):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(np.abs(v_view), '.')
    ax.set_yscale('log')
    ax.grid(which='both')
    # plt.show()

def view_field():

    proj = ccrs.Orthographic(central_longitude=0, central_latitude=20)
    ncol_proj = 2

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, ncol_proj+3)
    
    norm = 1/np.abs(field_view['u_t']).max()
    
    ax = fig.add_subplot(gs[0, :ncol_proj], projection=proj)
    plottings.plot_sphere(coord_view['lon'], coord_view['lat'], norm*field_view['zeta_val'], ax)
    # plottings.plot_sphere(coord_view['lon'], coord_view['lat'], norm*field_view['u_surf_t'], ax)
    # ix, iy = slice(0, None, 3), slice(0, None, 3)
    # quiver_sphere(coords['lon'][0, ix], coords['lat'][iy, 0], fields['u_surf_p'][iy, ix], -fields['u_surf_t'][iy, ix], ax)
    ax.set_title(r'$\hat{\mathbf{r}}\cdot \nabla\times \mathbf{u}$')
    
    ax = plottings.plot_v_comp(coord_view['s'], coord_view['z'], norm*field_view['u_r'], fig, gs[0,ncol_proj], title=r'$u_r$')
    plottings.plot_r_outline([Ri, Ro], np.linspace(0, np.pi, num=100), ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], np.linspace(0, np.pi, num=100), ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(coord_view['s'], coord_view['z'], norm*field_view['u_t'], fig, gs[0,ncol_proj+1], title=r'$u_\theta$')
    plottings.plot_r_outline([Ri, Ro], np.linspace(0, np.pi, num=100), ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], np.linspace(0, np.pi, num=100), ax, linewidth=0.5, linestyle='--')
    ax = plottings.plot_v_comp(coord_view['s'], coord_view['z'], norm*field_view['u_p'], fig, gs[0,ncol_proj+2], title=r'$u_\phi$')
    plottings.plot_r_outline([Ri, Ro], np.linspace(0, np.pi, num=100), ax, linewidth=0.5)
    plottings.plot_r_outline([r_sect], np.linspace(0, np.pi, num=100), ax, linewidth=0.5, linestyle='--')
    
    # plottings.figsave(fig, os.path.join(mat_dir, f"mode-{args.name}_m{args.m:02d}"), formats=('jpg',), dpi=200, overwrite=False)
    # plt.close()
    plt.show()


# view_spec(vec_view[:, 0])
view_field()



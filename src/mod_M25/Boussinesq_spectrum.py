import numpy as np
import dedalus.public as d3
import logging
from scipy import interpolate
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
args = parser.parse_args()

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
Nphi = 34
Ntheta = 124
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
dir_name= 'results/Boussinesq_solarDR'

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
strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))
Sigma= grad_u + d3.transpose(grad_u)
g = dist.Field(bases=shell.meridional_basis)
g['g'] = fg(r.reshape(r.size))/fg(Ri)
p0 = dist.Field(bases=shell.meridional_basis)
T0 = dist.Field(bases=shell.meridional_basis)

# Background quantities

T0['g'] = fT0(r.reshape(r.size))/fT0(Ri)
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
grad_s0['g'][2]=0


# Boussinesq_data=np.load('data/Boussinesq.npz')
# rhom=Boussinesq_data['rhom']/frho0(Ri)
rhom = 1

# Problem
problem = d3.EVP([ p, u, s, tau_u1, tau_u2, tau_p, tau_s1, tau_s2], eigenvalue=om, namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + v0@grad_u + u@grad(v0)  + 2*cross(ez, u) + grad(p/rhom) -  s*g/H*er  - 1/Re*(div(nu*Sigma)) + lift(tau_u2) = 0")
problem.add_equation("dt(s) + grad_s0@u/Ma**2 + v0@grad_s - 1/Pe/T0*div(T0*grad_s) + lift(tau_s2) = 0")
problem.add_equation("radial(u(r=Ri)) = 0")
problem.add_equation("radial(u(r=Ro)) = 0")
problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("radial(grad(s)(r=Ri))=0")
problem.add_equation("radial(grad(s)(r=Ro))=0")

# Solver
solver = problem.build_solver(ncc_cutoff=1e-10)


def solve(m):
    subproblem = solver.subproblems_by_group[(m, None, None)]
    target = -2
    eigr=np.array([])
    eigi=np.array([])
    logger.info('m=%d'%(m))
    logger.info("Target=%f"%(target))
    i=0
    

    while target<2:
        solver.solve_sparse(subproblem, 20, target)
        eigs=np.sort_complex(solver.eigenvalues)
        eigr=np.append(eigr,eigs.real)
        eigi=np.append(eigi,eigs.imag)
        if eigs[-1].real>target:
            target=eigs[-1].real
            i=0
        else:
            if i>4:
                dm=(target-eigs[-1].real)/2
                target+=dm 
            else:       
                target+=0.01  #check if required to change!
            i+=1
        logger.info("Target=%f"%(target))


    np.savez(dir_name+'/freq_MultiIter_%d'%(m),eigr=eigr,eigi=eigi)



m=args.m
solve(m)

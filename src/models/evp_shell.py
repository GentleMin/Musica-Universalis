# -*- coding: utf-8 -*-

import numpy as np
import dedalus.public as d3
from scipy import sparse
from typing import Literal, Callable


def chop_sparse(A: sparse.sparray, thresh: float = 1e-13):
    A.data[np.abs(A.data) < thresh] = 0
    A.eliminate_zeros()
    return A


def V0_axisym(r, t):
    Vp = np.zeros((1, t.size, r.size))
    Vt = np.zeros((1, t.size, r.size))
    Vr = np.zeros((1, t.size, r.size))
    return Vp, Vt, Vr


class ModelEVP_RShell:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int]) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)
        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        s = dist.Field(name='s')
        p = dist.Field(name='p', bases=shell)
        u = dist.VectorField(coords, name='u', bases=shell)

        tau_p = dist.Field(name='tau_p')
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)

        rvec = dist.VectorField(coords, name='rvec', bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        lift_basis = shell.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)

        # Intermediate variables
        grad_u = d3.grad(u) + rvec*lift(tau_u1)
        strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))
        return locals()

    @classmethod
    def setup_problem(cls, Ek, fields_namespace: dict, 
        bc: Literal['no-slip', 'stress-free'] = 'no-slip') -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        u = fields_namespace['u']
        p = fields_namespace['p']
        tau_u1 = fields_namespace['tau_u1']
        tau_u2 = fields_namespace['tau_u2']
        tau_p = fields_namespace['tau_p']
        fine_namespace = fields_namespace | {'Ek': Ek}

        # Equations
        problem = d3.EVP([u, p, tau_u1, tau_u2, tau_p], eigenvalue=s, namespace=fine_namespace)
        problem.add_equation("s*u + 2*cross(ez, u) - Ek*div(grad_u) + grad(p) + lift(tau_u2) = 0")
        problem.add_equation("trace(grad_u) + tau_p = 0")

        # Boundary conditions
        if bc == 'no-slip':
            problem.add_equation("u(r=Ri) = 0")
            problem.add_equation("u(r=Ro) = 0")
        if bc == 'stress-free':
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("radial(u(r=Ro)) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")        

        # Gauge conditions
        problem.add_equation("integ(p) = 0")
        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.fields = self.setup_fields(geometry, resolution)
        self.u = self.fields['u']
        self.p = self.fields['p']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_Ek1, M_Ek1 = self.setup_eigenmat(1, set_problem=set_problem, **params_problem)
        K_Ek2, M_Ek2 = self.setup_eigenmat(2, set_problem=False, **params_problem)

        dK_Ek = chop_sparse(K_Ek2 - K_Ek1)
        dM_Ek = chop_sparse(M_Ek2 - M_Ek1)
        K_Ek0 = chop_sparse(K_Ek1 - dK_Ek)
        M_Ek0 = chop_sparse(M_Ek1 - dM_Ek)

        M_lib = {
            'K_0': K_Ek0, 'M_0': M_Ek0,
            'dK_Ek': dK_Ek, 'dM_Ek': dM_Ek
        }
        return M_lib


class ModelEVP_RShell_TorPol:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int]) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        p_grid, t_grid, r_grid = dist.local_grids(shell)
        sphere = shell.outer_surface

        rvec = dist.VectorField(coords, bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        er = dist.VectorField(coords, bases=shell.meridional_basis)
        er['g'][2] = np.ones_like(r_grid)
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        s = dist.Field(name='s')
        Tor = dist.Field(name='T', bases=shell)
        Pol = dist.Field(name='P', bases=shell)
        # tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
        # tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)
        # tau_pr = dist.Field(name='tau_pr', bases=sphere)
        # tau_curl1 = dist.Field(name='tau_curl1', bases=sphere)
        # tau_curl2 = dist.Field(name='tau_curl2', bases=sphere)
        # tau_t = dist.Field(name='tau_t')
        # tau_p = dist.Field(name='tau_p')

        tau_Pt1 = dist.Field(name='tau_Pt1', bases=sphere)
        tau_Pt2 = dist.Field(name='tau_Pt2', bases=sphere)
        tau_Pp1 = dist.Field(name='tau_Pp1', bases=sphere)
        tau_Pp2 = dist.Field(name='tau_Pp2', bases=sphere)
        tau_Pp3 = dist.Field(name='tau_Pp3', bases=sphere)
        tau_Pp4 = dist.Field(name='tau_Pp4', bases=sphere)
        tau_tu = dist.Field(name='tau_tu')
        tau_pu = dist.Field(name='tau_pu')

        lift_basis = shell.derivative_basis(1)
        lift_t = lambda A: d3.Lift(A, shell, -1)
        lift_u = lambda A: d3.Lift(A, lift_basis, -1)

        u = d3.curl(Tor*rvec) + d3.curl(d3.curl(Pol*rvec))
        # strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))

        L2 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: -ell*(ell + 1))
        Mul_l = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell)
        Mul_lp1 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell + 1)

        r_Curl = lambda x: d3.dot(rvec, d3.curl(x))
        r_Curl2 = lambda x: d3.dot(rvec, d3.curl(d3.curl(x)))

        state_var = [
            Tor, Pol,
            # tau_u1, tau_pr, tau_curl1, tau_curl2,
            tau_Pt1, tau_Pt2, tau_Pp1, tau_Pp2, tau_Pp3, tau_Pp4, 
            # tau_Vt1, tau_Vt2, tau_Vp1, tau_Vp2,
            tau_tu, tau_pu, 
            # tau_tb, tau_pb
        ]

        return locals()

    @classmethod
    def setup_problem(cls, Ek, fields_namespace: dict, 
        bc: Literal['no-slip', 'stress-free'] = 'no-slip') -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        # u = fields_namespace['u']
        # Tor = fields_namespace['Tor']
        # Pol = fields_namespace['Pol']
        # ez = fields_namespace['ez']
        # rvec = fields_namespace['rvec']
        # tau_u1 = fields_namespace['tau_u1']
        # # tau_u2 = fields_namespace['tau_u2']
        # tau_pr = fields_namespace['tau_pr']
        # tau_curl1 = fields_namespace['tau_curl1']
        # tau_curl2 = fields_namespace['tau_curl2']
        # tau_t = fields_namespace['tau_t']
        # tau_p = fields_namespace['tau_p']
        # lift_u = fields_namespace['lift_u']
        # lift_t = fields_namespace['lift_t']

        # # Equations
        # eq_NS = s*u + 2*d3.cross(ez, u) - Ek*d3.div(d3.grad(u) + rvec*lift_u(tau_u1)) + rvec*lift_t(tau_pr)
        # curl1_eq_NS = d3.curl(eq_NS) + rvec*lift_u(tau_curl1)
        # curl2_eq_NS = d3.curl(curl1_eq_NS) + rvec*lift_t(tau_curl2)

        # fine_namespace = fields_namespace | {'Ek': Ek, 'eq_NS': eq_NS, 'curl1_eq_NS': curl1_eq_NS, 'curl2_eq_NS': curl2_eq_NS}
        fine_namespace = fields_namespace | {'Ek': Ek}

        # problem = d3.EVP([Tor, Pol, tau_u1, tau_pr, tau_curl1, tau_curl2, tau_t, tau_p], eigenvalue=s, namespace=fine_namespace)
        # # problem = d3.EVP([Tor, Pol, tau_u1, tau_u2, tau_t, tau_p], eigenvalue=s, namespace=fine_namespace)
        # problem.add_equation("dot(er, curl1_eq_NS) + tau_t = 0")
        # problem.add_equation("dot(er, curl2_eq_NS) + tau_p = 0")

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        problem.add_equation(
            "s*L2(Tor) " 
            "- Ek*div(grad(L2(Tor)) + rvec*lift_u(tau_Pt1)) + lift_u(tau_Pt2)" 
            "- r_Curl(2*cross(ez, u)) + tau_tu = 0" 
        )
        problem.add_equation(
            "s*div(grad(L2(Pol)) + rvec*lift_u(tau_Pp1))" 
            "- Ek*div(grad(div(grad(L2(Pol)) + rvec*lift_u(tau_Pp1)) + lift_u(tau_Pp2)) + rvec*lift_u(tau_Pp3)) + lift_u(tau_Pp4)" 
            "+ r_Curl2(2*cross(ez, u)) + tau_pu = 0"
        )

        # Boundary conditions
        # if bc == 'no-slip':
        #     problem.add_equation("u(r=Ri) = 0")
        #     problem.add_equation("u(r=Ro) = 0")
        # if bc == 'stress-free':
        #     problem.add_equation("radial(u(r=Ri)) = 0")
        #     problem.add_equation("radial(u(r=Ro)) = 0")
        #     problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
        #     problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
        problem.add_equation("Pol(r=Ri) = 0")
        problem.add_equation("Pol(r=Ro) = 0")
        if bc == 'no-slip':
            problem.add_equation("radial(grad(Pol)(r=Ri)) = 0")
            problem.add_equation("radial(grad(Pol)(r=Ro)) = 0")
            problem.add_equation("Tor(r=Ri) = 0")
            problem.add_equation("Tor(r=Ro) = 0")
        elif bc == 'stress-free':
            problem.add_equation("radial(radial(grad(grad(Pol))(r=Ri))) = 0")
            problem.add_equation("radial(radial(grad(grad(Pol))(r=Ro))) = 0")
            problem.add_equation("radial(grad(Tor)(r=Ri)) - Tor(r=Ri)/Ri = 0")
            problem.add_equation("radial(grad(Tor)(r=Ro)) - Tor(r=Ro)/Ro = 0")
        
        # Gauge conditions
        problem.add_equation("integ(Tor) = 0")
        problem.add_equation("integ(Pol) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.fields = self.setup_fields(geometry, resolution)
        self.u = self.fields['u']
        self.Tor = self.fields['Tor']
        self.Pol = self.fields['Pol']
        self.problem = None
        self.subprob = None
        self.solver = None
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_Ek1, M_Ek1 = self.setup_eigenmat(1, set_problem=set_problem, **params_problem)
        K_Ek2, M_Ek2 = self.setup_eigenmat(2, set_problem=False, **params_problem)

        dK_Ek = chop_sparse(K_Ek2 - K_Ek1, thresh=1e-10)
        dM_Ek = chop_sparse(M_Ek2 - M_Ek1, thresh=1e-10)
        K_Ek0 = chop_sparse(K_Ek1 - dK_Ek, thresh=1e-10)
        M_Ek0 = chop_sparse(M_Ek1 - dM_Ek, thresh=1e-10)

        M_lib = {
            'K_0': K_Ek0, 'M_0': M_Ek0,
            'dK_Ek': dK_Ek, 'dM_Ek': dM_Ek
        }
        return M_lib


class ModelEVP_MHDRShell:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray]) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        s = dist.Field(name='s')
        p = dist.Field(name='p', bases=shell)
        V = dist.Field(name='V', bases=shell)
        u = dist.VectorField(coords, name='u', bases=shell)
        A = dist.VectorField(coords, name='A', bases=shell)

        tau_p = dist.Field(name='tau_p')
        tau_V = dist.Field(name='tau_V')
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)
        tau_A1 = dist.VectorField(coords, name='tau_A1', bases=sphere)
        tau_A2 = dist.VectorField(coords, name='tau_A2', bases=sphere)

        rvec = dist.VectorField(coords, name='rvec', bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        # Background field
        B0 = dist.VectorField(coords, name='B0', bases=shell.meridional_basis)
        Bp, Bt, Br = B0_func(r_grid, t_grid)
        B0['g'][0] = Bp
        B0['g'][1] = Bt
        B0['g'][2] = Br

        lift_basis = shell.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)

        # Intermediate variables
        b = d3.curl(A)
        grad_u = d3.grad(u) + rvec*lift(tau_u1)
        grad_A = d3.grad(A) + rvec*lift(tau_A1)
        strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))
        ell_func_o = lambda ell: ell+1
        A_potential_bc_o = d3.radial(d3.grad(A)(r=Ro)) + d3.SphericalEllProduct(A, coords, ell_func_o)(r=Ro)/Ro
        ell_func_i = lambda ell: -ell
        A_potential_bc_i = d3.radial(d3.grad(A)(r=Ri)) + d3.SphericalEllProduct(A, coords, ell_func_i)(r=Ri)/Ri

        return locals()

    @classmethod
    def setup_problem(cls, Ek, Em, Le, fields_namespace: dict, 
        v_bc: Literal['no-slip', 'stress-free'] = 'no-slip', 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        u = fields_namespace['u']
        p = fields_namespace['p']
        A = fields_namespace['A']
        V = fields_namespace['V']
        tau_u1 = fields_namespace['tau_u1']
        tau_u2 = fields_namespace['tau_u2']
        tau_p = fields_namespace['tau_p']
        tau_A1 = fields_namespace['tau_A1']
        tau_A2 = fields_namespace['tau_A2']
        tau_V = fields_namespace['tau_V']
        fine_namespace = fields_namespace | {'Ek': Ek, 'Em': Em, 'Le': Le}

        # Equations
        problem = d3.EVP([u, p, A, V, tau_u1, tau_u2, tau_p, tau_A1, tau_A2, tau_V], eigenvalue=s, namespace=fine_namespace)
        problem.add_equation(
            "s*u + 2*cross(ez, u) - Ek*div(grad_u) + grad(p) + lift(tau_u2) " \
            "+ Le*(cross(lap(A), B0) - cross(curl(B0), b)) = 0")
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("s*A - Em*div(grad_A) + grad(V) + lift(tau_A2) - Le*cross(u, B0) = 0")
        problem.add_equation("trace(grad_A) + tau_V = 0")

        # Boundary conditions
        if v_bc == 'no-slip':
            problem.add_equation("u(r=Ri) = 0")
            problem.add_equation("u(r=Ro) = 0")
        elif v_bc == 'stress-free':
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("radial(u(r=Ro)) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
        else:
            raise NotImplementedError
        
        if m_bc == 'insulating':
            problem.add_equation("A_potential_bc_i = 0")
            problem.add_equation("A_potential_bc_o = 0")
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(p) = 0")
        problem.add_equation("integ(V) = 0")
        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.p = self.fields['p']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, Em, Le, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, Em, Le, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Ek2, M_Ek2 = self.setup_eigenmat(2, 1, 1, set_problem=False, **params_problem)
        K_Em2, M_Em2 = self.setup_eigenmat(1, 2, 1, set_problem=False, **params_problem)
        K_Le2, M_Le2 = self.setup_eigenmat(1, 1, 2, set_problem=False, **params_problem)

        dK_Ek = chop_sparse(K_Ek2 - K_base)
        dM_Ek = chop_sparse(M_Ek2 - M_base)
        dK_Em = chop_sparse(K_Em2 - K_base)
        dM_Em = chop_sparse(M_Em2 - M_base)
        dK_Le = chop_sparse(K_Le2 - K_base)
        dM_Le = chop_sparse(M_Le2 - M_base)
        K_0 = chop_sparse(K_base - dK_Ek - dK_Em - dK_Le)
        M_0 = chop_sparse(M_base - dM_Ek - dM_Em - dM_Le)

        M_lib = {
            'K_0': K_0, 'M_0': M_0,
            'dK_Ek': dK_Ek, 'dM_Ek': dM_Ek,
            'dK_Em': dK_Em, 'dM_Em': dM_Em,
            'dK_Le': dK_Le, 'dM_Le': dM_Le
        }
        return M_lib


class ModelEVP_MHDRShell_Alfven:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray]) -> dict:
        return ModelEVP_MHDRShell.setup_fields(geometry, resolution, B0_func)

    @classmethod
    def setup_problem(cls, Le, Lu, Pm, fields_namespace: dict, 
        v_bc: Literal['no-slip', 'stress-free'] = 'no-slip', 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        u = fields_namespace['u']
        p = fields_namespace['p']
        A = fields_namespace['A']
        V = fields_namespace['V']
        tau_u1 = fields_namespace['tau_u1']
        tau_u2 = fields_namespace['tau_u2']
        tau_p = fields_namespace['tau_p']
        tau_A1 = fields_namespace['tau_A1']
        tau_A2 = fields_namespace['tau_A2']
        tau_V = fields_namespace['tau_V']
        fine_namespace = fields_namespace | {'Le': Le, 'Lu': Lu, 'Pm': Pm}

        # Equations
        problem = d3.EVP([u, p, A, V, tau_u1, tau_u2, tau_p, tau_A1, tau_A2, tau_V], eigenvalue=s, namespace=fine_namespace)
        problem.add_equation(
            "s*u + (2/Le)*cross(ez, u) - (Pm/Lu)*div(grad_u) + grad(p) + lift(tau_u2) " \
            "+ (cross(lap(A), B0) - cross(curl(B0), b)) = 0")
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("s*A - (1/Lu)*div(grad_A) + grad(V) + lift(tau_A2) - cross(u, B0) = 0")
        problem.add_equation("trace(grad_A) + tau_V = 0")

        # Boundary conditions
        if v_bc == 'no-slip':
            problem.add_equation("u(r=Ri) = 0")
            problem.add_equation("u(r=Ro) = 0")
        elif v_bc == 'stress-free':
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("radial(u(r=Ro)) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
        else:
            raise NotImplementedError
        
        if m_bc == 'insulating':
            problem.add_equation("A_potential_bc_i = 0")
            problem.add_equation("A_potential_bc_o = 0")
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(p) = 0")
        problem.add_equation("integ(V) = 0")
        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Le, Lu, Pm, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Le, Lu, Pm, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Le, _ = self.setup_eigenmat(0.5, 1, 1, set_problem=False, **params_problem)
        K_Lu, _ = self.setup_eigenmat(1, 0.5, 1, set_problem=False, **params_problem)
        K_Pm, _ = self.setup_eigenmat(1, 1, 2.0, set_problem=False, **params_problem)

        K_vis = chop_sparse(K_Pm - K_base)
        K_Cor = chop_sparse(K_Le - K_base)
        K_eta = chop_sparse(K_Lu - K_base - K_vis)
        K_0 = chop_sparse(K_base - K_Cor - K_vis - K_eta)

        M_lib = {
            'M_0': M_base,
            'K_0': K_0,
            'K_vis': K_vis, 
            'K_Cor': K_Cor, 
            'K_eta': K_eta, 
        }
        return M_lib


class ModelEVP_ivMHDRShell_Alfven:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray]) -> dict:
        fields = ModelEVP_MHDRShell.setup_fields(geometry, resolution, B0_func)
        dist = fields['dist']
        sphere = fields['sphere']
        tau_u1 = dist.Field(name='tau_u1', bases=sphere)
        tau_u2 = dist.Field(name='tau_u2', bases=sphere)
        lift2 = lambda A: d3.Lift(A, fields['lift_basis'], -2)
        fields['tau_u1'] = tau_u1
        fields['tau_u2'] = tau_u2
        fields['lift2'] = lift2
        return fields

    @classmethod
    def setup_problem(cls, Le, Lu, fields_namespace: dict, 
        v_bc: Literal['no-penetration'] = 'no-penetration', 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        u = fields_namespace['u']
        p = fields_namespace['p']
        A = fields_namespace['A']
        V = fields_namespace['V']
        tau_u1 = fields_namespace['tau_u1']
        tau_u2 = fields_namespace['tau_u2']
        tau_p = fields_namespace['tau_p']
        tau_A1 = fields_namespace['tau_A1']
        tau_A2 = fields_namespace['tau_A2']
        tau_V = fields_namespace['tau_V']
        fine_namespace = fields_namespace | {'Le': Le, 'Lu': Lu}

        # Equations
        problem = d3.EVP([u, p, A, V, tau_u1, tau_u2, tau_p, tau_A1, tau_A2, tau_V], eigenvalue=s, namespace=fine_namespace)
        problem.add_equation(
            "s*u + (2/Le)*cross(ez, u) + grad(p) + rvec*lift(tau_u1)" \
            "+ (cross(lap(A), B0) - cross(curl(B0), b)) = 0")
        problem.add_equation("trace(grad(u) + rvec*rvec*lift(tau_u2)) + tau_p = 0")
        problem.add_equation("s*A - (1/Lu)*div(grad_A) + grad(V) + lift(tau_A2) - cross(u, B0) = 0")
        problem.add_equation("trace(grad_A) + tau_V = 0")

        # Boundary conditions
        if v_bc == 'no-penetration':
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("radial(u(r=Ro)) = 0")
        else:
            raise NotImplementedError
        
        if m_bc == 'insulating':
            problem.add_equation("A_potential_bc_i = 0")
            problem.add_equation("A_potential_bc_o = 0")
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(p) = 0")
        problem.add_equation("integ(V) = 0")
        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Le, Lu, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Le, Lu, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Le, _ = self.setup_eigenmat(0.5, 1, 1, set_problem=False, **params_problem)
        K_Lu, _ = self.setup_eigenmat(1, 0.5, 1, set_problem=False, **params_problem)

        K_Cor = chop_sparse(K_Le - K_base)
        K_eta = chop_sparse(K_Lu - K_base)
        K_0 = chop_sparse(K_base - K_Cor - K_eta)

        M_lib = {
            'M_0': M_base,
            'K_0': K_0,
            'K_Cor': K_Cor, 
            'K_eta': K_eta, 
        }
        return M_lib


class ModelEVP_MHDRShell_TorPol:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros_like(x)) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        rvec = dist.VectorField(coords, bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        er = dist.VectorField(coords, bases=shell.meridional_basis)
        er['g'][2] = np.ones_like(r_grid)
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        # Main fields
        s = dist.Field(name='s')
        Tu = dist.Field(name='Tu', bases=shell)
        Pu = dist.Field(name='Pu', bases=shell)
        Tb = dist.Field(name='Tb', bases=shell)
        Pb = dist.Field(name='Pb', bases=shell)
        u = d3.curl(Tu*rvec) + d3.curl(d3.curl(Pu*rvec))
        b = d3.curl(Tb*rvec) + d3.curl(d3.curl(Pb*rvec))

        # Kinetic tau variables
        # tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
        # tau_pr = dist.Field(name='tau_pr', bases=sphere)
        # tau_curl1 = dist.Field(name='tau_curl1', bases=sphere)
        # tau_curl2 = dist.Field(name='tau_curl2', bases=sphere)
        tau_Pt1 = dist.Field(name='tau_Pt1', bases=sphere)
        tau_Pt2 = dist.Field(name='tau_Pt2', bases=sphere)
        tau_Pp1 = dist.Field(name='tau_Pp1', bases=sphere)
        tau_Pp2 = dist.Field(name='tau_Pp2', bases=sphere)
        tau_Pp3 = dist.Field(name='tau_Pp3', bases=sphere)
        tau_Pp4 = dist.Field(name='tau_Pp4', bases=sphere)
        tau_tu = dist.Field(name='tau_tu')
        tau_pu = dist.Field(name='tau_pu')

        # Magnetic tau variables
        tau_Vt1 = dist.Field(name='tau_Vt1', bases=sphere)
        tau_Vt2 = dist.Field(name='tau_Vt2', bases=sphere)
        tau_Vp1 = dist.Field(name='tau_Vp1', bases=sphere)
        tau_Vp2 = dist.Field(name='tau_Vp2', bases=sphere)
        tau_tb = dist.Field(name='tau_tb')
        tau_pb = dist.Field(name='tau_pb')

        # Background field
        B0 = dist.VectorField(coords, name='B0', bases=shell.meridional_basis)
        Bp, Bt, Br = B0_func(r_grid, t_grid)
        B0['g'][0] = Bp
        B0['g'][1] = Bt
        B0['g'][2] = Br

        # Operators
        lift_basis = shell.derivative_basis(1)
        lift_t = lambda A: d3.Lift(A, shell, -1)
        lift_u = lambda A: d3.Lift(A, lift_basis, -1)

        L2 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: -ell*(ell + 1))
        Mul_l = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell)
        Mul_lp1 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell + 1)

        r_Curl = lambda x: d3.dot(rvec, d3.curl(x))
        r_Curl2 = lambda x: d3.dot(rvec, d3.curl(d3.curl(x)))

        state_var = [
            Tu, Pu, Tb, Pb,
            # tau_u1, tau_pr, tau_curl1, tau_curl2,
            tau_Pt1, tau_Pt2, tau_Pp1, tau_Pp2, tau_Pp3, tau_Pp4, 
            tau_Vt1, tau_Vt2, tau_Vp1, tau_Vp2,
            tau_tu, tau_pu, tau_tb, tau_pb
        ]

        return locals()

    @classmethod
    def setup_problem(cls, Ek, Em, Le, fields_namespace: dict, 
        v_bc: Literal['no-slip', 'stress-free'] = 'no-slip', 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:
        
        # Main variables
        s = fields_namespace['s']
        fine_namespace = {
            'Ek': Ek, 'Em': Em, 'Le': Le,
        }
        fine_namespace = fields_namespace | fine_namespace

        # Hydro only
        # problem = d3.EVP(
        #     [Tu, Pu, tau_u1, tau_pr, tau_curl1, tau_curl2, tau_tu, tau_pu], 
        #     eigenvalue=s, namespace=fine_namespace)
        # Magnetic only
        # problem = d3.EVP(
        #     [Tb, Pb, tau_Vt1, tau_Vt2, tau_Vp1, tau_Vp2, tau_tb, tau_pb], 
        #     eigenvalue=s, namespace=fine_namespace)
        # MHD equation
        # problem = d3.EVP(
        #     [Tu, Pu, Tb, Pb, tau_u1, tau_pr, tau_curl1, tau_curl2, tau_tu, tau_pu, tau_Vt1, tau_Vt2, tau_Vp1, tau_Vp2, tau_tb, tau_pb], 
        #     eigenvalue=s, namespace=fine_namespace)
        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        
        # Momentum equation
        # problem.add_equation("dot(er, curl1_eq_NS) + tau_tu = 0")
        # problem.add_equation("dot(er, curl2_eq_NS) + tau_pu = 0")
        problem.add_equation(
            "s*L2(Tu) " 
            "- Ek*div(grad(L2(Tu)) + rvec*lift_u(tau_Pt1)) + lift_u(tau_Pt2)" 
            "- r_Curl(2*cross(ez, u))" 
            "+ Le*r_Curl(cross(curl(b), B0) + cross(curl(B0), b)) = 0"
        )
        problem.add_equation(
            "s*div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1))" 
            "- Ek*div(grad(div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1)) + lift_u(tau_Pp2)) + rvec*lift_u(tau_Pp3)) + lift_u(tau_Pp4)" 
            "+ r_Curl2(2*cross(ez, u))"
            "- Le*r_Curl2(cross(curl(b), B0) + cross(curl(B0), b)) = 0"
        )
        problem.add_equation(
            "s*L2(Tb) " 
            "- Em*div(grad(L2(Tb)) + rvec*lift_u(tau_Vt1)) + lift_u(tau_Vt2)" 
            "+ Le*r_Curl2(cross(u, B0)) = 0")
        problem.add_equation(
            "s*L2(Pb) " 
            "- Em*div(grad(L2(Pb)) + rvec*lift_u(tau_Vp1)) + lift_u(tau_Vp2)" 
            "+ Le*r_Curl(cross(u, B0)) = 0")

        # Boundary conditions
        if v_bc == 'no-slip':
            problem.add_equation("Tu(r=Ri) = 0")
            problem.add_equation("Tu(r=Ro) = 0")
            problem.add_equation("Pu(r=Ri) = 0")
            problem.add_equation("Pu(r=Ro) = 0")
            problem.add_equation("radial(grad(Pu)(r=Ri)) = 0")
            problem.add_equation("radial(grad(Pu)(r=Ro)) = 0")
        elif v_bc == 'stress-free':
            problem.add_equation("radial(grad(Tu)(r=Ri)) - Tu(r=Ri)/Ri = 0")
            problem.add_equation("radial(grad(Tu)(r=Ro)) - Tu(r=Ro)/Ro = 0")
            problem.add_equation("Pu(r=Ri) = 0")
            problem.add_equation("Pu(r=Ro) = 0")
            problem.add_equation("radial(radial(grad(grad(Pu))(r=Ri))) = 0")
            problem.add_equation("radial(radial(grad(grad(Pu))(r=Ro))) = 0")
        elif v_bc == 'none':
            pass
        else:
            raise NotImplementedError
        
        if m_bc == 'insulating':
            problem.add_equation("Tb(r=Ri) = 0")
            problem.add_equation("Tb(r=Ro) = 0")
            problem.add_equation("radial(grad(Pb)(r=Ri)) - Mul_l(Pb)(r=Ri)/Ri = 0")
            problem.add_equation("radial(grad(Pb)(r=Ro)) + Mul_lp1(Pb)(r=Ro)/Ro = 0")
        elif m_bc == 'none':
            pass
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(Tu) = 0")
        problem.add_equation("integ(Pu) = 0")
        problem.add_equation("integ(Tb) = 0")
        problem.add_equation("integ(Pb) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, Em, Le, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, Em, Le, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Ek2, M_Ek2 = self.setup_eigenmat(2, 1, 1, set_problem=False, **params_problem)
        K_Em2, M_Em2 = self.setup_eigenmat(1, 2, 1, set_problem=False, **params_problem)
        K_Le2, M_Le2 = self.setup_eigenmat(1, 1, 2, set_problem=False, **params_problem)

        dK_Ek = chop_sparse(K_Ek2 - K_base)
        dM_Ek = chop_sparse(M_Ek2 - M_base)
        dK_Em = chop_sparse(K_Em2 - K_base)
        dM_Em = chop_sparse(M_Em2 - M_base)
        dK_Le = chop_sparse(K_Le2 - K_base)
        dM_Le = chop_sparse(M_Le2 - M_base)
        K_0 = chop_sparse(K_base - dK_Ek - dK_Em - dK_Le)
        M_0 = chop_sparse(M_base - dM_Ek - dM_Em - dM_Le)

        M_lib = {
            'K_0': K_0, 'M_0': M_0,
            'dK_Ek': dK_Ek, 'dM_Ek': dM_Ek,
            'dK_Em': dK_Em, 'dM_Em': dM_Em,
            'dK_Le': dK_Le, 'dM_Le': dM_Le
        }
        return M_lib


class ModelEVP_MHDRShell_TorPol_Alfven:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros_like(x)) -> dict:
        return ModelEVP_MHDRShell_TorPol.setup_fields(geometry, resolution, B0_func)

    @classmethod
    def setup_problem(cls, Le, Lu, Pm, fields_namespace: dict, 
        v_bc_i: Literal['no-slip', 'stress-free'] = 'no-slip', 
        v_bc_o: Literal['no-slip', 'stress-free'] = 'no-slip', 
        b_bc_i: Literal['insulating', 'perfect-conducting'] = 'insulating',
        b_bc_o: Literal['insulating', 'perfect-conducting'] = 'insulating'
    ) -> d3.EigenvalueProblem:
        
        s = fields_namespace['s']
        fine_namespace = {'Le': Le, 'Lu': Lu, 'Pm': Pm}
        fine_namespace = fields_namespace | fine_namespace

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        
        problem.add_equation(
            "s*L2(Tu) " 
            "- (Pm/Lu)*div(grad(L2(Tu)) + rvec*lift_u(tau_Pt1)) + lift_u(tau_Pt2)" 
            "- (1/Le)*r_Curl(2*cross(ez, u))" 
            "+ r_Curl(cross(curl(b), B0) + cross(curl(B0), b)) = 0"
        )
        problem.add_equation(
            "s*div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1))" 
            "- (Pm/Lu)*div(grad(div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1)) + lift_u(tau_Pp2)) + rvec*lift_u(tau_Pp3)) + lift_u(tau_Pp4)" 
            "+ (1/Le)*r_Curl2(2*cross(ez, u))"
            "- r_Curl2(cross(curl(b), B0) + cross(curl(B0), b)) = 0"
        )
        problem.add_equation(
            "s*L2(Tb) " 
            "- (1/Lu)*div(grad(L2(Tb)) + rvec*lift_u(tau_Vt1)) + lift_u(tau_Vt2)" 
            "+ r_Curl2(cross(u, B0)) = 0")
        problem.add_equation(
            "s*L2(Pb) " 
            "- (1/Lu)*div(grad(L2(Pb)) + rvec*lift_u(tau_Vp1)) + lift_u(tau_Vp2)" 
            "+ r_Curl(cross(u, B0)) = 0")

        # Boundary conditions
        eq_bcs = {
            'vT': {
                'no-slip': [
                    "Tu(r=Ri) = 0",
                    "Tu(r=Ro) = 0"
                ],
                "stress-free": [
                    "radial(grad(Tu)(r=Ri)) - Tu(r=Ri)/Ri = 0",
                    "radial(grad(Tu)(r=Ro)) - Tu(r=Ro)/Ro = 0"
                ]
            },
            'vP': {
                'no-penetration': [
                    "Pu(r=Ri) = 0",
                    "Pu(r=Ro) = 0"
                ],
                'no-slip': [
                    "radial(grad(Pu)(r=Ri)) = 0",
                    "radial(grad(Pu)(r=Ro)) = 0"
                ],
                "stress-free": [
                    "radial(radial(grad(grad(Pu))(r=Ri))) = 0",
                    "radial(radial(grad(grad(Pu))(r=Ro))) = 0"
                ]
            }, 
            'bT': {
                'insulating': [
                    "Tb(r=Ri) = 0",
                    "Tb(r=Ro) = 0"
                ],
                'perfect-conducting': [
                    "radial(grad(Tb)(r=Ri)) + Tb(r=Ri)/Ri = 0",
                    "radial(grad(Tb)(r=Ro)) + Tb(r=Ro)/Ro = 0"
                ]
            },
            'bP': {
                'insulating': [
                    "radial(grad(Pb)(r=Ri)) - Mul_l(Pb)(r=Ri)/Ri = 0",
                    "radial(grad(Pb)(r=Ro)) + Mul_lp1(Pb)(r=Ro)/Ro = 0"
                ],
                'perfect-conducting': [
                    "Pb(r=Ri) = 0",
                    "Pb(r=Ro) = 0"
                ]
            }
        }
        problem.add_equation(eq_bcs['vT'][v_bc_i][0])
        problem.add_equation(eq_bcs['vT'][v_bc_o][1])
        problem.add_equation(eq_bcs['vP']['no-penetration'][0])
        problem.add_equation(eq_bcs['vP']['no-penetration'][1])
        problem.add_equation(eq_bcs['vP'][v_bc_i][0])
        problem.add_equation(eq_bcs['vP'][v_bc_o][1])
        problem.add_equation(eq_bcs['bT'][b_bc_i][0])
        problem.add_equation(eq_bcs['bT'][b_bc_o][1])
        problem.add_equation(eq_bcs['bP'][b_bc_i][0])
        problem.add_equation(eq_bcs['bP'][b_bc_o][1])

        # Gauge conditions
        problem.add_equation("integ(Tu) = 0")
        problem.add_equation("integ(Pu) = 0")
        problem.add_equation("integ(Tb) = 0")
        problem.add_equation("integ(Pb) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str

    def setup_eigenmat(self, Le, Lu, Pm, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Le, Lu, Pm, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Le, _ = self.setup_eigenmat(0.5, 1, 1, set_problem=False, **params_problem)
        K_Lu, _ = self.setup_eigenmat(1, 0.5, 1, set_problem=False, **params_problem)
        K_Pm, _ = self.setup_eigenmat(1, 1, 2.0, set_problem=False, **params_problem)

        K_vis = chop_sparse(K_Pm - K_base)
        K_Cor = chop_sparse(K_Le - K_base)
        K_eta = chop_sparse(K_Lu - K_base - K_vis)
        K_0 = chop_sparse(K_base - K_Cor - K_vis - K_eta)

        M_lib = {
            'mass': M_base,
            'coriolis': K_Cor, 
            'lorentz_induction': K_0,
            'viscous_diffusion': K_vis, 
            'magnetic_diffusion': K_eta, 
        }
        return M_lib


class ModelEVP_ivMHDRShell_TorPol_Alfven:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros_like(x)) -> dict:

        fields = ModelEVP_MHDRShell_TorPol.setup_fields(geometry, resolution, B0_func)
        fields['state_var'] = [
            fields['Tu'], fields['Pu'], fields['Tb'], fields['Pb'], 
            fields['tau_Pp1'], fields['tau_Pp2'],
            fields['tau_Vt1'], fields['tau_Vt2'], fields['tau_Vp1'], fields['tau_Vp2'], 
            fields['tau_pu'], fields['tau_tu'], 
            fields['tau_pb'], fields['tau_tb'], 
        ]
        return fields

    @classmethod
    def setup_problem(cls, Le, Lu, fields_namespace: dict, 
        v_bc: Literal['no-penetration'] = 'no-penetration', 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:
        
        s = fields_namespace['s']
        fine_namespace = {'Le': Le, 'Lu': Lu}
        fine_namespace = fields_namespace | fine_namespace

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        
        problem.add_equation(
            "s*L2(Tu) "  
            "- (1/Le)*r_Curl(2*cross(ez, u))" 
            "+ r_Curl(cross(curl(b), B0) + cross(curl(B0), b)) + tau_tu = 0"
        )
        problem.add_equation(
            "s*div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1)) + lift_u(tau_Pp2)"  
            "+ (1/Le)*r_Curl2(2*cross(ez, u))"
            "- r_Curl2(cross(curl(b), B0) + cross(curl(B0), b)) + tau_pu = 0"
        )
        problem.add_equation(
            "s*L2(Tb) " 
            "- (1/Lu)*div(grad(L2(Tb)) + rvec*lift_u(tau_Vt1)) + lift_u(tau_Vt2)" 
            "+ r_Curl2(cross(u, B0)) + tau_tb = 0")
        problem.add_equation(
            "s*L2(Pb) " 
            "- (1/Lu)*div(grad(L2(Pb)) + rvec*lift_u(tau_Vp1)) + lift_u(tau_Vp2)" 
            "+ r_Curl(cross(u, B0)) + tau_pb = 0")

        # Boundary conditions
        if v_bc == 'no-penetration':
            problem.add_equation("Pu(r=Ri) = 0")
            problem.add_equation("Pu(r=Ro) = 0")
        elif v_bc == 'none':
            pass
        else:
            raise NotImplementedError
        
        if m_bc == 'insulating':
            problem.add_equation("Tb(r=Ri) = 0")
            problem.add_equation("Tb(r=Ro) = 0")
            problem.add_equation("radial(grad(Pb)(r=Ri)) - Mul_l(Pb)(r=Ri)/Ri = 0")
            problem.add_equation("radial(grad(Pb)(r=Ro)) + Mul_lp1(Pb)(r=Ro)/Ro = 0")
        elif m_bc == 'none':
            pass
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(Tu) = 0")
        problem.add_equation("integ(Pu) = 0")
        problem.add_equation("integ(Tb) = 0")
        problem.add_equation("integ(Pb) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Le, Lu, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Le, Lu, self.fields, **params_problem)
        solver = problem.build_solver()
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Le, _ = self.setup_eigenmat(0.5, 1, 1, set_problem=False, **params_problem)
        K_Lu, _ = self.setup_eigenmat(1, 0.5, 1, set_problem=False, **params_problem)

        K_Cor = chop_sparse(K_Le - K_base)
        K_eta = chop_sparse(K_Lu - K_base)
        K_0 = chop_sparse(K_base - K_Cor - K_eta)

        M_lib = {
            'M_0': M_base,
            'K_0': K_0,
            'K_Cor': K_Cor, 
            'K_eta': K_eta, 
        }
        return M_lib


class ModelEVP_MagShell_TorPol:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int]) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        rvec = dist.VectorField(coords, bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        er = dist.VectorField(coords, bases=shell.meridional_basis)
        er['g'][2] = np.ones_like(r_grid)

        s = dist.Field(name='s')
        Tb = dist.Field(name='Tb', bases=shell)
        Pb = dist.Field(name='Pb', bases=shell)
        b = d3.curl(Tb*rvec) + d3.curl(d3.curl(Pb*rvec))

        tau_b1 = dist.VectorField(coords, name='tau_b1', bases=sphere)
        tau_b = dist.Field(name='tau_b', bases=sphere)
        tau_V = dist.Field(name='tau_V', bases=sphere)
        tau_curl1b = dist.Field(name='tau_curl1b', bases=sphere)
        tau_curl2b = dist.Field(name='tau_curl2b', bases=sphere)
        tau_tb = dist.Field(name='tau_tb')
        tau_pb = dist.Field(name='tau_pb')

        lift_basis = shell.derivative_basis(1)
        lift_t = lambda A: d3.Lift(A, shell, -1)
        lift_u = lambda A: d3.Lift(A, lift_basis, -1)

        # Intermediate variables
        f_ell = lambda ell: ell
        f_ell_p1 = lambda ell: ell+1
        mul_Pb_ell = d3.SphericalEllProduct(Pb, coords, f_ell)
        mul_Pb_ell_p1 = d3.SphericalEllProduct(Pb, coords, f_ell_p1)

        return locals()

    @classmethod
    def setup_problem(cls, Em, fields_namespace: dict, 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        b = fields_namespace['b']
        Tb = fields_namespace['Tb']
        Pb = fields_namespace['Pb']
        rvec = fields_namespace['rvec']
        tau_b1 = fields_namespace['tau_b1']
        tau_V = fields_namespace['tau_V']
        tau_curl1b = fields_namespace['tau_curl1b']
        tau_curl2b = fields_namespace['tau_curl2b']
        tau_tb = fields_namespace['tau_tb']
        tau_pb = fields_namespace['tau_pb']

        lift_u = fields_namespace['lift_u']
        lift_t = fields_namespace['lift_t']

        # Equations
        eq_MI = s*b - Em*d3.div(d3.grad(b) + rvec*lift_u(tau_b1)) + rvec*lift_u(tau_V)
        curl1_eq_MI = d3.curl(eq_MI)

        fine_namespace = {
            'Em': Em, 
            'eq_MI': eq_MI, 'curl1_eq_MI': curl1_eq_MI
        }
        fine_namespace = fields_namespace | fine_namespace

        problem = d3.EVP(
            [Tb, Pb, tau_b1, tau_V, tau_tb, tau_pb],
            # [Tb, Pb, tau_b1, tau_V], 
            eigenvalue=s, namespace=fine_namespace)
        # problem = d3.EVP(
        #     [Tu, Pu, Tb, Pb, tau_u1, tau_pr, tau_curl1, tau_curl2, tau_tu, tau_pu, tau_b, tau_curl1b, tau_curl2b, tau_V, tau_tb, tau_pb], 
        #     eigenvalue=s, namespace=fine_namespace)
        problem.add_equation("dot(er, curl1_eq_MI) + tau_tb = 0")
        problem.add_equation("dot(er, eq_MI) + tau_pb = 0")
        # problem.add_equation("dot(er, curl1_eq_MI) = 0")
        # problem.add_equation("dot(er, eq_MI) = 0")

        # Boundary conditions
        if m_bc == 'insulating':
            problem.add_equation("Tb(r=Ri) = 0")
            problem.add_equation("Tb(r=Ro) = 0")
            problem.add_equation("Pb(r=Ri) = 0")
            problem.add_equation("Pb(r=Ro) = 0")
            # problem.add_equation("radial(grad(Pb)(r=Ri)) - mul_Pb_ell(r=Ri)/Ri = 0")
            # problem.add_equation("radial(grad(Pb)(r=Ro)) + mul_Pb_ell_p1(r=Ro)/Ro = 0")
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(Tb) = 0")
        problem.add_equation("integ(Pb) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.fields = self.setup_fields(geometry, resolution)
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Em, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Em, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
        
        return K, M


class ModelEVP_DecayRShell_TorPol:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.zeros_like(x)) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        rvec = dist.VectorField(coords, bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        er = dist.VectorField(coords, bases=shell.meridional_basis)
        er['g'][2] = np.ones_like(r_grid)

        # Main fields
        s = dist.Field(name='s')
        Tb = dist.Field(name='Tb', bases=shell)
        Pb = dist.Field(name='Pb', bases=shell)
        b = d3.curl(Tb*rvec) + d3.curl(d3.curl(Pb*rvec))

        # Magnetic tau variables
        tau_Vt1 = dist.Field(name='tau_Vt1', bases=sphere)
        tau_Vt2 = dist.Field(name='tau_Vt2', bases=sphere)
        tau_Vp1 = dist.Field(name='tau_Vp1', bases=sphere)
        tau_Vp2 = dist.Field(name='tau_Vp2', bases=sphere)
        tau_tb = dist.Field(name='tau_tb')
        tau_pb = dist.Field(name='tau_pb')

        # Operators
        lift_basis = shell.derivative_basis(1)
        lift_t = lambda A: d3.Lift(A, shell, -1)
        lift_u = lambda A: d3.Lift(A, lift_basis, -1)

        L2 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: -ell*(ell + 1))
        Mul_l = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell)
        Mul_lp1 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell + 1)

        r_Curl = lambda x: d3.dot(rvec, d3.curl(x))
        r_Curl2 = lambda x: d3.dot(rvec, d3.curl(d3.curl(x)))

        state_var = [
            Tb, Pb,
            tau_Vt1, tau_Vt2, tau_Vp1, tau_Vp2,
            tau_tb, tau_pb
        ]

        return locals()

    @classmethod
    def setup_problem(cls, fields_namespace: dict, 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:
        
        # Main variables
        s = fields_namespace['s']
        fine_namespace = fields_namespace

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        
        problem.add_equation(
            "s*L2(Tb) " 
            "- div(grad(L2(Tb)) + rvec*lift_u(tau_Vt1)) + lift_u(tau_Vt2) = 0")
        problem.add_equation(
            "s*L2(Pb) " 
            "- div(grad(L2(Pb)) + rvec*lift_u(tau_Vp1)) + lift_u(tau_Vp2) = 0")

        # Boundary conditions        
        if m_bc == 'insulating':
            problem.add_equation("Tb(r=Ri) = 0")
            problem.add_equation("Tb(r=Ro) = 0")
            problem.add_equation("radial(grad(Pb)(r=Ri)) - Mul_l(Pb)(r=Ri)/Ri = 0")
            problem.add_equation("radial(grad(Pb)(r=Ro)) + Mul_lp1(Pb)(r=Ro)/Ro = 0")
        elif m_bc == 'none':
            pass
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(Tb) = 0")
        problem.add_equation("integ(Pb) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M


class ModelEVP_DecayRShell_lm1D:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int]) -> dict:

        Ri, Ro = geometry
        Nr, lval, mval = resolution

        rcoord = d3.Coordinates('r')
        dist = d3.Distributor(rcoord, dtype=cls.dtype)
        rbasis = d3.ChebyshevT(rcoord, size=Nr, radii=(Ri, Ro), dtype=cls.dtype)
        r_grid = dist.local_grid(rbasis)

        # Main fields
        s = dist.Field(name='s')
        Rb = dist.Field(name='Tb', bases=rbasis)

        # Magnetic tau variables
        tau_Rb1 = dist.Field(name='tau_tb')
        tau_Rb2 = dist.Field(name='tau_pb')

        # coeffs
        cf2 = dist.Field(bases=rbasis)
        cf1 = dist.Field(bases=rbasis)
        cf2['g'] = r_grid**2
        cf1['g'] = r_grid

        # Operators
        lift_basis = rbasis.derivative_basis(1)
        lift_u = lambda A: d3.Lift(A, lift_basis, -1)

        # Intermediate variables
        L = lval*(lval + 1)
        dr = lambda A: d3.Differentiate(A, rcoord)
        dR_dr = dr(Rb) + lift_u(tau_Rb1)
        d2R_dr2 = dr(dR_dr) + lift_u(tau_Rb2)

        state_var = [
            Rb,
            tau_Rb1, tau_Rb2
        ]
        return locals()

    @classmethod
    def setup_problem(cls, fields_namespace: dict, 
        m_bc: Literal['insulating'] = 'insulating',
        m_type: Literal['tor', 'pol'] = 'tor',
    ) -> d3.EigenvalueProblem:
        
        # Main variables
        s = fields_namespace['s']
        fine_namespace = fields_namespace

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        
        problem.add_equation("cf2*d2R_dr2 + cf1*dR_dr - L*Rb - s*cf2*Rb = 0")

        # Boundary conditions        
        if m_bc == 'insulating':
            if m_type == 'tor':
                problem.add_equation("Rb(r=Ri) = 0")
                problem.add_equation("Rb(r=Ro) = 0")
            elif m_type == 'pol':
                problem.add_equation("Rb(r=Ri) - lval*dr(Rb)(r=Ri)/Ri = 0")
                problem.add_equation("Rb(r=Ro) + (lval + 1)*dr(Rb)(r=Ro)/Ro = 0")
            else:
                raise NotImplementedError
        elif m_bc == 'none':
            pass
        else:
            raise NotImplementedError

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.fields = self.setup_fields(geometry, resolution)
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M


class ModelEVP_MHD_Shell:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable[[np.ndarray], np.ndarray]) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        s = dist.Field(name='s')
        p = dist.Field(name='p', bases=shell)
        V = dist.Field(name='V', bases=shell)
        u = dist.VectorField(coords, name='u', bases=shell)
        A = dist.VectorField(coords, name='A', bases=shell)

        tau_p = dist.Field(name='tau_p')
        tau_V = dist.Field(name='tau_V')
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)
        tau_A1 = dist.VectorField(coords, name='tau_A1', bases=sphere)
        tau_A2 = dist.VectorField(coords, name='tau_A2', bases=sphere)

        rvec = dist.VectorField(coords, name='rvec', bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        # Background field
        B0 = dist.VectorField(coords, name='B0', bases=shell.meridional_basis)
        Bp, Bt, Br = B0_func(r_grid, t_grid)
        B0['g'][0] = Bp
        B0['g'][1] = Bt
        B0['g'][2] = Br

        lift_basis = shell.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)

        # Intermediate variables
        b = d3.curl(A)
        grad_u = d3.grad(u) + rvec*lift(tau_u1)
        grad_A = d3.grad(A) + rvec*lift(tau_A1)
        strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))
        ell_func_o = lambda ell: ell+1
        A_potential_bc_o = d3.radial(d3.grad(A)(r=Ro)) + d3.SphericalEllProduct(A, coords, ell_func_o)(r=Ro)/Ro
        ell_func_i = lambda ell: -ell
        A_potential_bc_i = d3.radial(d3.grad(A)(r=Ri)) + d3.SphericalEllProduct(A, coords, ell_func_i)(r=Ri)/Ri

        return locals()

    @classmethod
    def setup_problem(cls, Le, Lu, Pm, fields_namespace: dict, 
        v_bc: Literal['no-slip', 'stress-free'] = 'no-slip', 
        m_bc: Literal['insulating'] = 'insulating'
    ) -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        u = fields_namespace['u']
        p = fields_namespace['p']
        A = fields_namespace['A']
        V = fields_namespace['V']
        tau_u1 = fields_namespace['tau_u1']
        tau_u2 = fields_namespace['tau_u2']
        tau_p = fields_namespace['tau_p']
        tau_A1 = fields_namespace['tau_A1']
        tau_A2 = fields_namespace['tau_A2']
        tau_V = fields_namespace['tau_V']
        fine_namespace = fields_namespace | {'Le': Le, 'Lu': Lu, 'Pm': Pm}

        # Equations
        problem = d3.EVP([u, p, A, V, tau_u1, tau_u2, tau_p, tau_A1, tau_A2, tau_V], eigenvalue=s, namespace=fine_namespace)
        problem.add_equation(
            "s*u + (2/Le)*cross(ez, u) - (Pm/Lu)*div(grad_u) + grad(p) + lift(tau_u2) " \
            "+ (cross(lap(A), B0) - cross(curl(B0), b)) = 0")
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("s*A - (1/Lu)*div(grad_A) + grad(V) + lift(tau_A2) - cross(u, B0) = 0")
        problem.add_equation("trace(grad_A) + tau_V = 0")

        # Boundary conditions
        if v_bc == 'no-slip':
            problem.add_equation("u(r=Ri) = 0")
            problem.add_equation("u(r=Ro) = 0")
        elif v_bc == 'stress-free':
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("radial(u(r=Ro)) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")
        else:
            raise NotImplementedError
        
        if m_bc == 'insulating':
            problem.add_equation("A_potential_bc_i = 0")
            problem.add_equation("A_potential_bc_o = 0")
        else:
            raise NotImplementedError

        # Gauge conditions
        problem.add_equation("integ(p) = 0")
        problem.add_equation("integ(V) = 0")
        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], B0_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Le, Lu, Pm, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Le, Lu, Pm, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, set_problem=set_problem, **params_problem)
        K_Le, _ = self.setup_eigenmat(0.5, 1, 1, set_problem=False, **params_problem)
        K_Lu, _ = self.setup_eigenmat(1, 0.5, 1, set_problem=False, **params_problem)
        K_Pm, _ = self.setup_eigenmat(1, 1, 2.0, set_problem=False, **params_problem)

        K_vis = chop_sparse(K_Pm - K_base)
        K_Cor = chop_sparse(K_Le - K_base)
        K_eta = chop_sparse(K_Lu - K_base - K_vis)
        K_0 = chop_sparse(K_base - K_Cor - K_vis - K_eta)

        M_lib = {
            'M_0': M_base,
            'K_0': K_0,
            'K_vis': K_vis, 
            'K_Cor': K_Cor, 
            'K_eta': K_eta, 
        }
        return M_lib


class ModelEVP_DiffRShell:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        dOmega_func: Callable = lambda x, y: np.zeros_like(x)) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)
        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        sphere = shell.outer_surface
        p_grid, t_grid, r_grid = dist.local_grids(shell)

        s = dist.Field(name='s')
        p = dist.Field(name='p', bases=shell)
        u = dist.VectorField(coords, name='u', bases=shell)

        tau_p = dist.Field(name='tau_p')
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)

        rvec = dist.VectorField(coords, name='rvec', bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        # Background field
        U0 = dist.VectorField(coords, name='U0', bases=shell.meridional_basis)
        dOmega = dOmega_func(r_grid, t_grid)
        U0['g'][0] = r_grid*np.sin(t_grid)*dOmega

        lift_basis = shell.derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)

        # Intermediate variables
        grad_u = d3.grad(u) + rvec*lift(tau_u1)
        strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))
        return locals()

    @classmethod
    def setup_problem(cls, Ek, Ro_r, fields_namespace: dict, 
        bc: Literal['no-slip', 'stress-free'] = 'no-slip') -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        u = fields_namespace['u']
        p = fields_namespace['p']
        tau_u1 = fields_namespace['tau_u1']
        tau_u2 = fields_namespace['tau_u2']
        tau_p = fields_namespace['tau_p']
        fine_namespace = fields_namespace | {'Ek': Ek, 'Ro_r': Ro_r}

        # Equations
        problem = d3.EVP([u, p, tau_u1, tau_u2, tau_p], eigenvalue=s, namespace=fine_namespace)
        problem.add_equation("s*u + Ro_r*(U0 @ grad_u + u @ grad(U0)) + 2*cross(ez, u) - Ek*div(grad_u) + grad(p) + lift(tau_u2) = 0")
        problem.add_equation("trace(grad_u) + tau_p = 0")

        # Boundary conditions
        if bc == 'no-slip':
            problem.add_equation("u(r=Ri) = 0")
            problem.add_equation("u(r=Ro) = 0")
        if bc == 'stress-free':
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("radial(u(r=Ro)) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ri), 0), 0) = 0")
            problem.add_equation("angular(radial(strain_rate(r=Ro), 0), 0) = 0")        

        # Gauge conditions
        problem.add_equation("integ(p) = 0")
        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], dOmega_func: Callable, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.dOmega_func = dOmega_func
        self.fields = self.setup_fields(geometry, resolution, dOmega_func=self.dOmega_func)
        self.u = self.fields['u']
        self.p = self.fields['p']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, Ro_r, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, Ro_r, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K, M = self.setup_eigenmat(1, 1, set_problem=set_problem, **params_problem)
        K_Ek, _ = self.setup_eigenmat(2, 1, set_problem=False, **params_problem)
        K_Ro, _ = self.setup_eigenmat(1, 2, set_problem=False, **params_problem)

        dK_Ek = chop_sparse(K_Ek - K)
        dK_Ro = chop_sparse(K_Ro - K)
        K_0 = chop_sparse(K - dK_Ek - dK_Ro)

        M_lib = {
            'mass': M,
            'coriolis': K_0,
            'viscous_diffusion': dK_Ek,
            'differential_rotation': dK_Ro
        }
        return M_lib


class ModelEVP_DiffRShell_TorPol:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        U0_func: Callable = V0_axisym) -> dict:

        Ri, Ro = geometry
        Nr, Nt, m = resolution
        Np = 2*(m + 1)

        coords = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(coords, dtype=cls.dtype)
        shell = d3.ShellBasis(coords, shape=(Np, Nt, Nr), radii=(Ri, Ro), dtype=cls.dtype)
        p_grid, t_grid, r_grid = dist.local_grids(shell)
        sphere = shell.outer_surface

        rvec = dist.VectorField(coords, bases=shell.meridional_basis)
        rvec['g'][2] = r_grid
        er = dist.VectorField(coords, bases=shell.meridional_basis)
        er['g'][2] = np.ones_like(r_grid)
        ez = dist.VectorField(coords, bases=shell.meridional_basis)
        ez['g'][1] = - np.sin(t_grid)
        ez['g'][2] = + np.cos(t_grid)

        s = dist.Field(name='s')
        Tor = dist.Field(name='T', bases=shell)
        Pol = dist.Field(name='P', bases=shell)

        tau_Pt1 = dist.Field(name='tau_Pt1', bases=sphere)
        tau_Pt2 = dist.Field(name='tau_Pt2', bases=sphere)
        tau_Pp1 = dist.Field(name='tau_Pp1', bases=sphere)
        tau_Pp2 = dist.Field(name='tau_Pp2', bases=sphere)
        tau_Pp3 = dist.Field(name='tau_Pp3', bases=sphere)
        tau_Pp4 = dist.Field(name='tau_Pp4', bases=sphere)
        tau_tu = dist.Field(name='tau_tu')
        tau_pu = dist.Field(name='tau_pu')

        # Background field
        U0 = dist.VectorField(coords, name='U0', bases=shell.meridional_basis)
        Up, Ut, Ur = U0_func(r_grid, t_grid)
        U0['g'][0] = Up
        U0['g'][1] = Ut
        U0['g'][2] = Ur
        # dOmega = dOmega_func(r_grid, t_grid)
        # U0['g'][0] = r_grid*np.sin(t_grid)*dOmega

        lift_basis = shell.derivative_basis(1)
        lift_t = lambda A: d3.Lift(A, shell, -1)
        lift_u = lambda A: d3.Lift(A, lift_basis, -1)

        u = d3.curl(Tor*rvec) + d3.curl(d3.curl(Pol*rvec))
        # strain_rate = d3.grad(u) + d3.transpose(d3.grad(u))

        L2 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: -ell*(ell + 1))
        Mul_l = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell)
        Mul_lp1 = lambda x: d3.SphericalEllProduct(x, coords, lambda ell: ell + 1)

        r_Curl = lambda x: d3.dot(rvec, d3.curl(x))
        r_Curl2 = lambda x: d3.dot(rvec, d3.curl(d3.curl(x)))

        state_var = [
            Tor, Pol,
            tau_Pt1, tau_Pt2, tau_Pp1, tau_Pp2, tau_Pp3, tau_Pp4, 
            tau_tu, tau_pu, 
        ]

        return locals()

    @classmethod
    def setup_problem(cls, Ek, Ro_r, fields_namespace: dict, 
        bc: Literal['no-slip', 'stress-free'] = 'no-slip') -> d3.EigenvalueProblem:

        s = fields_namespace['s']
        fine_namespace = fields_namespace | {'Ek': Ek, 'Ro_r': Ro_r}

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        problem.add_equation(
            "s*L2(Tor) - r_Curl(2*cross(ez, u) + Ro_r*(U0 @ grad(u) + u @ grad(U0)))" 
            "- Ek*div(grad(L2(Tor)) + rvec*lift_u(tau_Pt1))" 
            "+ lift_u(tau_Pt2) + tau_tu = 0" 
        )
        problem.add_equation(
            "s*div(grad(L2(Pol)) + rvec*lift_u(tau_Pp1))"
            "+ r_Curl2(2*cross(ez, u) + Ro_r*(U0 @ grad(u) + u @ grad(U0)))" 
            "- Ek*div(grad(div(grad(L2(Pol)) + rvec*lift_u(tau_Pp1)) + lift_u(tau_Pp2)) + rvec*lift_u(tau_Pp3))" 
            "+ lift_u(tau_Pp4) + tau_pu = 0"
        )

        if bc == 'no-slip':
            problem.add_equation("Tor(r=Ri) = 0")
            problem.add_equation("Tor(r=Ro) = 0")
            problem.add_equation("radial(grad(Pol)(r=Ri)) = 0")
            problem.add_equation("radial(grad(Pol)(r=Ro)) = 0")
        elif bc == 'stress-free':
            problem.add_equation("radial(grad(Tor)(r=Ri)) - Tor(r=Ri)/Ri = 0")
            problem.add_equation("radial(grad(Tor)(r=Ro)) - Tor(r=Ro)/Ro = 0")
            problem.add_equation("radial(radial(grad(grad(Pol))(r=Ri))) = 0")
            problem.add_equation("radial(radial(grad(grad(Pol))(r=Ro))) = 0")
        problem.add_equation("Pol(r=Ri) = 0")
        problem.add_equation("Pol(r=Ro) = 0")
        
        # Gauge conditions
        problem.add_equation("integ(Tor) = 0")
        problem.add_equation("integ(Pol) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        U0_func: Callable = V0_axisym, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.U0_func = U0_func
        self.fields = self.setup_fields(geometry, resolution, U0_func=self.U0_func)
        self.u = self.fields['u']
        self.Tor = self.fields['Tor']
        self.Pol = self.fields['Pol']
        self.problem = None
        self.subprob = None
        self.solver = None
        
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, Ro_r, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, Ro_r, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K, M = self.setup_eigenmat(1, 1, set_problem=set_problem, **params_problem)
        K_Ek, _ = self.setup_eigenmat(2, 1, set_problem=False, **params_problem)
        K_Ro, _ = self.setup_eigenmat(1, 2, set_problem=False, **params_problem)

        dK_Ek = chop_sparse(K_Ek - K, thresh=1e-10)
        dK_Ro = chop_sparse(K_Ro - K, thresh=1e-10)
        K_0 = chop_sparse(K - dK_Ek - dK_Ro, thresh=1e-10)

        M_lib = {
            'mass': M,
            'coriolis': K_0,
            'viscous_diffusion': dK_Ek,
            'differential_rotation': dK_Ro
        }
        return M_lib


class ModelEVP_MHDDiffRShell_TorPol:

    dtype = np.complex128

    @classmethod
    def setup_fields(cls, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable = V0_axisym, U0_func: Callable = V0_axisym) -> dict:

        fields = ModelEVP_MHDRShell_TorPol.setup_fields(geometry, resolution, B0_func)
        # Add background flow
        U0 = fields['dist'].VectorField(fields['coords'], name='U0', bases=fields['shell'].meridional_basis)
        Up, Ut, Ur = U0_func(fields['r_grid'], fields['t_grid'])
        U0['g'][0] = Up
        U0['g'][1] = Ut
        U0['g'][2] = Ur
        fields['U0'] = U0
        return fields

    @classmethod
    def setup_problem(cls, Ek, Em, Le, Ro_r, fields_namespace: dict, 
        v_bc_i: Literal['no-slip', 'stress-free'] = 'no-slip', 
        v_bc_o: Literal['no-slip', 'stress-free'] = 'no-slip', 
        b_bc_i: Literal['insulating', 'perfect-conducting'] = 'insulating',
        b_bc_o: Literal['insulating', 'perfect-conducting'] = 'insulating'
    ) -> d3.EigenvalueProblem:
        
        s = fields_namespace['s']
        fine_namespace = {'Ek': Ek, 'Em': Em, 'Le': Le, 'Ro_r': Ro_r}
        fine_namespace = fields_namespace | fine_namespace

        problem = d3.EVP(
            fields_namespace['state_var'], 
            eigenvalue=s, namespace=fine_namespace)
        
        problem.add_equation(
            "s*L2(Tu) - r_Curl(2*cross(ez, u) + Ro_r*(U0 @ grad(u) + u @ grad(U0)))" 
            "- Ek*div( grad(L2(Tu)) + rvec*lift_u(tau_Pt1))" 
            "+ Le*r_Curl(cross(curl(b), B0) + cross(curl(B0), b))"
            "+ lift_u(tau_Pt2) + tau_tu = 0"
        )
        problem.add_equation(
            "s*div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1))" 
            "+ r_Curl2(2*cross(ez, u) + Ro_r*(U0 @ grad(u) + u @ grad(U0)))"
            "- Ek*div(grad(div(grad(L2(Pu)) + rvec*lift_u(tau_Pp1)) + lift_u(tau_Pp2)) + rvec*lift_u(tau_Pp3))" 
            "- Le*r_Curl2(cross(curl(b), B0) + cross(curl(B0), b))"
            "+ lift_u(tau_Pp4) + tau_pu = 0"
        )
        problem.add_equation(
            "s*L2(Tb) " 
            "- Em*div(grad(L2(Tb)) + rvec*lift_u(tau_Vt1))" 
            "+ r_Curl2(Le*cross(u, B0) + Ro_r*cross(U0, b))"
            "+ lift_u(tau_Vt2) + tau_tb = 0")
        problem.add_equation(
            "s*L2(Pb) " 
            "- Em*div(grad(L2(Pb)) + rvec*lift_u(tau_Vp1))" 
            "+ r_Curl(Le*cross(u, B0) + Ro_r*cross(U0, b))"
            "+ lift_u(tau_Vp2) + tau_pb = 0")

        # Boundary conditions
        eq_bcs = {
            'vT': {
                'no-slip': [
                    "Tu(r=Ri) = 0",
                    "Tu(r=Ro) = 0"
                ],
                "stress-free": [
                    "radial(grad(Tu)(r=Ri)) - Tu(r=Ri)/Ri = 0",
                    "radial(grad(Tu)(r=Ro)) - Tu(r=Ro)/Ro = 0"
                ]
            },
            'vP': {
                'no-penetration': [
                    "Pu(r=Ri) = 0",
                    "Pu(r=Ro) = 0"
                ],
                'no-slip': [
                    "radial(grad(Pu)(r=Ri)) = 0",
                    "radial(grad(Pu)(r=Ro)) = 0"
                ],
                "stress-free": [
                    "radial(radial(grad(grad(Pu))(r=Ri))) = 0",
                    "radial(radial(grad(grad(Pu))(r=Ro))) = 0"
                ]
            }, 
            'bT': {
                'insulating': [
                    "Tb(r=Ri) = 0",
                    "Tb(r=Ro) = 0"
                ],
                'perfect-conducting': [
                    "radial(grad(Tb)(r=Ri)) + Tb(r=Ri)/Ri = 0",
                    "radial(grad(Tb)(r=Ro)) + Tb(r=Ro)/Ro = 0"
                ]
            },
            'bP': {
                'insulating': [
                    "radial(grad(Pb)(r=Ri)) - Mul_l(Pb)(r=Ri)/Ri = 0",
                    "radial(grad(Pb)(r=Ro)) + Mul_lp1(Pb)(r=Ro)/Ro = 0"
                ],
                'perfect-conducting': [
                    "Pb(r=Ri) = 0",
                    "Pb(r=Ro) = 0"
                ]
            }
        }
        problem.add_equation(eq_bcs['vT'][v_bc_i][0])
        problem.add_equation(eq_bcs['vT'][v_bc_o][1])
        problem.add_equation(eq_bcs['vP']['no-penetration'][0])
        problem.add_equation(eq_bcs['vP']['no-penetration'][1])
        problem.add_equation(eq_bcs['vP'][v_bc_i][0])
        problem.add_equation(eq_bcs['vP'][v_bc_o][1])
        problem.add_equation(eq_bcs['bT'][b_bc_i][0])
        problem.add_equation(eq_bcs['bT'][b_bc_o][1])
        problem.add_equation(eq_bcs['bP'][b_bc_i][0])
        problem.add_equation(eq_bcs['bP'][b_bc_o][1])

        # Gauge conditions
        problem.add_equation("integ(Tu) = 0")
        problem.add_equation("integ(Pu) = 0")
        problem.add_equation("integ(Tb) = 0")
        problem.add_equation("integ(Pb) = 0")

        return problem

    def __init__(self, geometry: tuple[float, float], resolution: tuple[int, int, int], 
        B0_func: Callable = V0_axisym, U0_func: Callable = V0_axisym, **params) -> None:

        self.geometry = geometry
        self.resolution = resolution
        self.B0_func = B0_func
        self.U0_func = U0_func
        self.fields = self.setup_fields(geometry, resolution, B0_func=self.B0_func, U0_func=self.U0_func)
        self.u = self.fields['u']
        self.b = self.fields['b']
        self.problem = None
        self.subprob = None
        self.solver = None
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        o_str = f'<{cls_name} geom={self.geometry} res={self.resolution}>'
        return o_str
    
    def setup_eigenmat(self, Ek, Em, Le, Ro_r, set_problem: bool = True, **params_problem):

        m = self.resolution[-1]
        problem = self.setup_problem(Ek, Em, Le, Ro_r, self.fields, **params_problem)
        solver = problem.build_solver(ncc_cutoff=1e-10)
        subprob = solver.subproblems_by_group[(m, None, None)]
        solver.build_matrices([subprob,], ['M', 'L'])
        K = sparse.csc_array(subprob.L_min)
        M = sparse.csc_array(subprob.M_min)
        
        if set_problem:
            self.problem = problem
            self.subprob = subprob
            self.solver = solver
        
        return K, M
    
    def precomp_submat(self, set_problem: bool = True, **params_problem):
        
        K_base, M_base = self.setup_eigenmat(1, 1, 1, 1, set_problem=set_problem, **params_problem)
        K_Ek, _ = self.setup_eigenmat(2, 1, 1, 1, set_problem=False, **params_problem)
        K_Em, _ = self.setup_eigenmat(1, 2, 1, 1, set_problem=False, **params_problem)
        K_Le, _ = self.setup_eigenmat(1, 1, 2, 1, set_problem=False, **params_problem)
        K_Ro, _ = self.setup_eigenmat(1, 1, 1, 2, set_problem=False, **params_problem)

        K_mag = chop_sparse(K_Le - K_base)
        K_vis = chop_sparse(K_Ek - K_base)
        K_eta = chop_sparse(K_Em - K_base)
        K_vel = chop_sparse(K_Ro - K_base)
        K_0 = chop_sparse(K_base - K_mag - K_vel - K_vis - K_eta)

        M_lib = {
            'mass': M_base,
            'coriolis': K_0, 
            'lorentz_induction': K_mag,
            'viscous_diffusion': K_vis, 
            'magnetic_diffusion': K_eta,
            'differential_rotation': K_vel, 
        }
        return M_lib

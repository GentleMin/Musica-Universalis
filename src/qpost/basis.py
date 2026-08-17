# -*- coding: utf-8 -*-


import numpy as np
from scipy import fft, special


def eval_chebyt_recur(Nmesh: np.ndarray, zmesh: np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev polynomial of the 1st kind using recurrence relations
    
    This function is intended to maintain the same signature 
    as `scipy.special.eval_chebyt` and `sympy.chebyshevt`, 
    albeit several restrictions regarding the input params (see note of Jacobi)
    
    :param np.ndarray Nmesh: mesh for degrees *(N,Nz)*
    :param np.ndarray zmesh: mesh for evaluation grid *(N,Nz)*
    """
    assert Nmesh.shape == zmesh.shape
    
    n_array = Nmesh[:, 0]
    z_array = zmesh[0, :]
    Nmax = n_array.max()
    idx_pos = n_array > 0
    
    T_vals = np.zeros_like(zmesh)
    T_vals[n_array == 0, :] = 1.
    if Nmax >= 1:
        T_vals_Nmax = eval_chebyt_recur_Nmax(Nmax, z_array)
        T_vals[idx_pos, :] = T_vals_Nmax[n_array[idx_pos], :]
        
    return T_vals


def eval_chebyt_recur_Nmax(Nmax: int, z: np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev polynomials of 1st kind with recurrence relation up to a degree
    
    This functions generates values for Chebyshev T polynomials from degree 0
    up to a specified degree, using recurrence relations.
    
    :param int Nmax: maximum degree, required to be >= 1
    :param np.ndarray z: 1-D array of grid points where the Jacobi polynomials
        are to be evaluated; assumed to be within interval [-1, +1]
    :returns: Array with shape (Nmax + 1, z.size), values for Jacobi
        polynomials at grid points specified in `z`.
    """
    assert Nmax >= 1
    # Set computing degrees
    n_array = np.arange(Nmax + 1)                                   # O(N)
    # Initializing the matrix: N * M
    T_vals_Nmax = np.zeros((n_array.size, z.size), dtype=z.dtype)   # O(MN)
    # Start from non-negative degrees
    T_vals_Nmax[0, :] = 1.
    T_vals_Nmax[1, :] = z
    if Nmax == 1:
        return T_vals_Nmax
    
    # Use recurrence relations, computation O(kN) + extra Memory O(kN)
    for i_n in range(n_array.size - 2):
        T_vals_Nmax[i_n+2, :] = 2*z*T_vals_Nmax[i_n+1, :] - T_vals_Nmax[i_n, :]
    
    return T_vals_Nmax


def chebyT(N: int, z: np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev polynomials of 1st kind with recurrence relation up to a degree
    """
    n_array = np.arange(N + 1)
    Tn = np.zeros((n_array.size, z.size), dtype=z.dtype)
    Tn[0, :] = 1.
    if N == 0: 
        return Tn

    Tn[1, :] = z
    for i_n in range(2, N+1):
        Tn[i_n, :] = 2*z*Tn[i_n-1, :] - Tn[i_n-2, :]
    return Tn


def chebyU(N: int, z: np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev polynomials of 2nd kind with recurrence relation up to a degree
    """
    n_array = np.arange(N + 1)
    Un = np.zeros((n_array.size, z.size), dtype=z.dtype)
    Un[0, :] = 1.
    if N == 0: 
        return Un

    Un[1, :] = 2*z
    for i_n in range(2, N+1):
        Un[i_n, :] = 2*z*Un[i_n-1, :] - Un[i_n-2, :]
    return Un


def eval_legendrePlm(Lmax, m, x):

    Plm = special.sph_legendre_p_all(Lmax, m, np.arccos(x))[0, m:, m, :]
    return Plm


def alpha(l, m):
    return np.sqrt(((l - m) * (l + m)) / ((2.0 * l - 1.0) * (2.0 * l + 1.0)) * (l >= m))


def lgClm(l, m):
    return 0.5 * (special.loggamma(l - m + 1) - special.loggamma(l + m + 1) + np.log((2. * l + 1.) / 4. / np.pi))
    

def Plm(m, lmax, theta):
    """ compute spherical harmonics (excluding the e^{im\\phi} term), evaluated on theta grid  """
    cos_theta = np.cos(theta).reshape(-1, 1)

    if lmax == m:
        C = np.exp(-0.5 * special.loggamma(2 * m + 1)) * np.sqrt((2. * m + 1.) / (4. * np.pi))
        C_Pmm = (-1.)**m * special.factorial2(2*m - 1) if m > 0 else 1.
        val = C * C_Pmm * (1. - cos_theta**2) ** (m / 2.)
        return val
    elif lmax > m:
        vals = []
        C = np.exp(-0.5 * special.loggamma(2 * m + 1)) * np.sqrt((2. * m + 1.) / (4. * np.pi))
        C_Pmm = (-1.)**m * special.factorial2(2*m - 1) if m > 0 else 1.
        tmp = C * C_Pmm * (1. - cos_theta**2) ** (m / 2.)
        vals.append(tmp)
        for l in range(m, lmax):
            if l > m:
                tmp = (cos_theta * vals[l - m] - alpha(l, m) * vals[l - m - 1]) / alpha(l + 1, m)
            else:
                tmp = cos_theta * vals[l - m] / alpha(l + 1, m)
            vals.append(tmp)
        return np.concatenate(vals, axis=1)


def divs_Plm(m, lmax, theta):
    """ compute value of C_l^m P_l^m(\\cos\\theta) / \\sin\\theta, on theta, for a single m
    Only for m > 0 !"""
    if m > 0:
        plm_m1 = Plm(m - 1, lmax, theta)
        plm_p1 = Plm(m + 1, lmax, theta)

        val = []
        for l in range(m, lmax + 1):
            if l - 1 >= m + 1:
                tmp = -1. / (2. * m) * np.exp(lgClm(l, m) - lgClm(l - 1, m + 1)) * plm_p1[:, l - m - 2] \
                    - (l + m - 1.) * (l + m) / (2. * m) * np.exp(lgClm(l, m) - lgClm(l - 1, m - 1)) * plm_m1[:, l - m]
            else:
                tmp = - (l + m - 1.) * (l + m) / (2. * m) * np.exp(lgClm(l, m) - lgClm(l - 1, m - 1)) * plm_m1[:, l - m]
            val.append(tmp.reshape(-1, 1))
        return np.concatenate(val, axis=1)
    else:
        # return np.zeros((theta.shape[0], lmax-m+1))
        return Plm(m, lmax, theta)/np.sin(theta).reshape(-1, 1)


def Dtheta_Plm(m, lmax, Plm, PlmDivSin, theta):
    """ compute value of D_theta C_l^m P_l^m(\\cos\\theta), for a single m
    m = 0 compute in a different way """

    sin_theta = np.sin(theta).reshape(-1, 1)
    if m == 0:
        val = [np.zeros(sin_theta.shape), -0.5 * np.sqrt(3. / np.pi) * sin_theta]
        for l in range(1, lmax):
            tmp = np.exp(lgClm(l + 1, 0) - lgClm(l - 1, 0)) * val[l - 1] - \
                  (2. * l + 1.) * np.exp(lgClm(l + 1, 0) - lgClm(l, 0)) * sin_theta * Plm[:, l].reshape(-1, 1)
            val.append(tmp.reshape(-1, 1))
        DthetaPlm = np.concatenate(val, axis=1)
    else:
        val = []
        for l in range(m, lmax + 1):
            if l > m:
                tmp = -(l + 1.) * alpha(l, m) * PlmDivSin[:, l - m - 1] + l * alpha(l + 1, m) * PlmDivSin[:, l + 1 - m]
            else:
                tmp = l * alpha(l + 1, m) * PlmDivSin[:, l + 1 - m]
            val.append(tmp.reshape(-1, 1))
        DthetaPlm = np.concatenate(val, axis=1)
    return DthetaPlm


class ChebyshevT:

    def __init__(self, N: int, interval: tuple = (-1., 1.), dealias=1., g=None):

        self.Ns = N
        self.range = interval
        self.a = (self.range[1] - self.range[0])/2
        self.b = (self.range[1] + self.range[0])/2

        if dealias is None:
            self.Ng = fft.next_fast_len(self.Ns)
        else:
            self.Ng = int(dealias*self.Ns)
        self.g0 = self.a*np.cos(np.pi*(np.arange(self.Ng) + 0.5)/self.Ng) + self.b
        self.g = self.g0.copy() if g is None else g
        self.ops = dict()
    
    def x2g(self, x):
        g = self.a*x + self.b
        return g
    
    def g2x(self, g):
        x = (g - self.b)/self.a
        return x
    
    def grids_chebyt(self, N):
        xi = self.a*np.cos(np.pi*(np.arange(N) + 0.5)/N) + self.b
        xi = np.flip(xi)
        wt = self.a*(np.pi/N)*np.ones(N)
        return xi, wt
    
    def grids_legendre(self, N):
        xi, wt = special.roots_legendre(N)
        xi = self.x2g(xi)
        wt *= self.a
        return xi, wt
    
    def grids_jacobi(self, N, a, b):
        xi, wt = special.roots_jacobi(N, a, b)
        xi = self.x2g(xi)
        wt *= self.a
        return xi, wt
    
    def mat_s2p(self, grid=None):
        grid = self.g0 if grid is None else grid
        Cmat = chebyT(self.Ns-1, self.g2x(grid)).T
        Cmat[:, 0] /= 2.
        return Cmat
    
    def mat_p2s(self):
        Cmat = chebyT(self.Ns-1, self.g2x(self.g0))
        Cmat *= 2/self.Ng
        return Cmat
    
    def s2p(self, spec, grid=None):

        assert spec.size == self.Ns

        if grid is None:
            fval = fft.dct(spec, type=3, n=self.Ng)/2
            return fval
        
        Cmat = chebyT(self.Ns-1, self.g2x(grid)).T
        Cmat[:, 0] /= 2.
        fval = Cmat @ spec
        return fval
    
    def p2s(self, fval):

        assert fval.size == self.Ng
        spec = fft.dct(fval, type=2, n=self.Ng)[:self.Ns]/self.Ng
        return spec
    
    def op_id(self, grid = None):

        grid = self.g if grid is None else grid
        op = chebyT(self.Ns-1, self.g2x(grid)).T
        op[:, 0] /= 2.
        return op
    
    def op_d1(self, grid = None):

        grid = self.g if grid is None else grid
        op = np.zeros((grid.size, self.Ns), dtype=np.float64)
        if self.Ns <= 1:
            return op
        
        nrange = np.arange(self.Ns)
        op[:, 1:] = chebyU(self.Ns-2, self.g2x(grid)).T
        op *= nrange/self.a
        return op
    
    def op_d2(self, grid = None):
        
        grid = self.g if grid is None else grid
        op = np.zeros((grid.size, self.Ns), dtype=np.float64)
        if self.Ns <= 2:
            return op
        
        nrem = np.arange(2, self.Ns)
        x = self.g2x(grid)

        ix_upper = np.isclose(x, +1, rtol=1e-7, atol=0)
        op[ix_upper, 2:] = nrem**2*(nrem**2 - 1)/3/self.a**2

        ix_lower = np.isclose(x, -1, rtol=1e-7, atol=0)
        op[ix_lower, 2:] = ((-1)**nrem)*nrem**2*(nrem**2 - 1)/3/self.a**2

        xrem = x[~(ix_upper | ix_lower)]
        op[~(ix_upper | ix_lower), 2:] = (nrem*
            ((nrem + 1)*chebyT(self.Ns-1, xrem)[2:].T - chebyU(self.Ns-1, xrem)[2:].T)
            /(self.a**2*np.reshape(xrem**2 - 1, (-1, 1)))
        )
        return op
    
    def setup_ops(self, grid = None):

        self.ops['id'] = self.op_id(grid=grid)
        self.ops['d1'] = self.op_d1(grid=grid)
        self.ops['d2'] = self.op_d2(grid=grid)


class LegendrePlmx:
    
    def __init__(self, L, m, N_quad=None, dealias=1, g=None):
        
        N_quad = int(np.round(dealias*L)) + 3 if N_quad is None else N_quad
        xi_quad, wt_quad = special.roots_legendre(N_quad)

        self.L = L
        self.m = m
        self.Ns = L - m + 1
        self.Ng = N_quad
        self.range = (-1., 1.)
        self.wt = wt_quad
        self.g0 = xi_quad
        self.g = self.g0.copy() if g is None else g
    
    def __repr__(self):
        o_str = '<AssocLegendre Plmx (L={}, m={}) on Gauss-Legendre grid (N={})>'.format(
            self.L, self.m, self.Ng
        )
        return o_str
    
    def norm_sq(self):
        d = 1/(2*np.pi)*np.ones(self.L - self.m + 1)
        return d
    
    def t2z(self, theta):
        z = np.cos(theta)
        return z
    
    def z2t(self, z):
        theta = np.arccos(z)
        return theta
    
    def mat_s2p(self, grid = None):
        
        grid = self.g if grid is None else grid
        P_Plm = eval_legendrePlm(self.L, self.m, grid)
        P_Plm = P_Plm.T
        # P_Plm = Plmx(self.m, self.L, grid)
        return P_Plm

    def mat_p2s(self):
        
        norm2 = self.norm_sq()
        Plm = eval_legendrePlm(self.L, self.m, self.g0)
        T_Plm = ((Plm*self.wt).T/norm2).T
        # T_Plm = Plmx(self.m, self.L, self.g)
        # T_Plm = (T_Plm/norm2).T*self.wt
        return T_Plm
    
    def s2p(self, spec, grid = None):
        
        assert spec.size == self.Ns
        fval = self.mat_s2p(grid=grid) @ spec
        return fval
    
    def p2s(self, fval):

        assert fval.size == self.Ng
        spec = self.mat_p2s() @ fval
        return spec


class LegendrePlm:
    
    def __init__(self, L, m, N_quad=None, dealias=1, g=None):
        
        N_quad = int(np.round(dealias*L)) + 3 if N_quad is None else N_quad
        xi_quad, wt_quad = special.roots_legendre(N_quad)

        self.L = L
        self.m = m
        self.Ns = L - m + 1
        self.Ng = N_quad
        self.range = (0., np.pi)
        self.wt = wt_quad
        self.g0 = np.arccos(xi_quad)
        self.g = self.g0.copy() if g is None else g
        self.ops = dict()
    
    def __repr__(self):
        o_str = '<AssocLegendre Plm (L={}, m={}) on Gauss-Legendre grid (N={})>'.format(
            self.L, self.m, self.Ng
        )
        return o_str
    
    def norm_sq(self):
        d = 1/(2*np.pi)*np.ones(self.L - self.m + 1)
        return d
    
    def t2z(self, theta):
        z = np.cos(theta)
        return z
    
    def z2t(self, z):
        theta = np.arccos(z)
        return theta
    
    def mat_s2p(self, grid = None):
        
        grid = self.g if grid is None else grid
        P_Plm = Plm(self.m, self.L, grid)
        return P_Plm

    def mat_p2s(self):
        
        T_Plm = Plm(self.m, self.L, self.g0)
        norm2 = self.norm_sq()
        T_Plm = (T_Plm/norm2).T*self.wt
        return T_Plm
    
    def s2p(self, spec, grid = None):
        
        assert spec.size == self.Ns
        fval = self.mat_s2p(grid=grid) @ spec
        return fval
    
    def p2s(self, fval):

        assert fval.size == self.Ng
        T_Plm = Plm(self.m, self.L, self.g0)
        norm2 = self.norm_sq()
        spec = (T_Plm.T @ (fval*self.wt))/norm2
        return spec
    
    def op_id(self, grid = None):
        grid = self.g if grid is None else grid
        op = Plm(self.m, self.L, grid)
        return op
    
    def op_div_sin(self, grid = None):
        grid = self.g if grid is None else grid
        op = divs_Plm(self.m, self.L, grid)
        return op
    
    def op_dtheta(self, grid = None):
        grid = self.g if grid is None else grid
        plm_arr = Plm(self.m, self.L, grid)
        divs_plm_arr = divs_Plm(self.m, self.L+1, grid)
        op = Dtheta_Plm(self.m, self.L, plm_arr, divs_plm_arr, grid)
        return op
    
    def setup_ops(self, grid = None):
        self.ops['id'] = self.op_id(grid=grid)
        self.ops['div_sin'] = self.op_div_sin(grid=grid)
        self.ops['d_theta'] = self.op_dtheta(grid=grid)



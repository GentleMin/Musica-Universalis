# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
from scipy import interpolate
from ..qpost import basis as qbasis
from ..qpost import field as qfield


class AxisymVectorBG:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, r, t):
        return np.zeros_like(r)


class NullVector(AxisymVectorBG):

    def __call__(self, r, t):
        Bp = np.zeros((1, t.size, r.size))
        Bt = np.zeros((1, t.size, r.size))
        Br = np.zeros((1, t.size, r.size))
        return Bp, Bt, Br


"""
Background magnetic fields
"""


class MagBG_U(AxisymVectorBG):

    def __call__(self, r, t):
        Bp = np.zeros((1, t.size, r.size))
        Bt = -np.ones_like(r)*np.sin(t)
        Br = np.ones_like(r)*np.cos(t)
        return Bp, Bt, Br


class MagBG_S1(AxisymVectorBG):

    def __init__(self, *args, Ri=0.35, Ro=1.0, **kwargs):
        self.Ri = Ri
        self.Ro = Ro

    def __call__(self, r, t):
        # S1-Shell, conforming to insulating BC @ arbitrary Ri, Ro
        Ri, Ro = self.Ri, self.Ro
        alpha = 1/(4*(Ro - Ri)*(2*Ro - Ri))
        Bp = np.zeros((1, t.size, r.size))
        Bt = alpha*(-9*Ro*r + 2*(2*Ri**2 + 4*Ro**2) - 3*Ri**2*Ro/r)*np.sin(t)
        Br = alpha*(+6*Ro*r - 2*(2*Ri**2 + 4*Ro**2) + 6*Ri**2*Ro/r)*np.cos(t)
        return Bp, Bt, Br


class MagBG_S1_Waves4(AxisymVectorBG):

    def __call__(self, r, t):
        Ri = 0.35
        alpha = (400/3)*(10/13)*np.sqrt(15/732251)*np.sqrt(3/np.pi)/2
        Bp = np.zeros((1, t.size, r.size))
        Bt = alpha*(-9*r + 2*(2*Ri**2 + 4) - 3*Ri**2/r)*np.sin(t)
        Br = alpha*(+6*r - 2*(2*Ri**2 + 4) + 6*Ri**2/r)*np.cos(t)
        return Bp, Bt, Br
    

class MagBG_S1_Waves4_R1(AxisymVectorBG):

    def __call__(self, r, t):
        Ri = 0.35
        alpha = (400/3)*(10/13)*np.sqrt(15/732251)*np.sqrt(3/np.pi)/2
        Bp = np.zeros((1, t.size, r.size))
        Bt = alpha*(-9*r/0.65 + 2*(2*Ri**2 + 4) - 3*Ri**2*0.65/r)*np.sin(t)
        Br = alpha*(+6*r/0.65 - 2*(2*Ri**2 + 4) + 6*Ri**2*0.65/r)*np.cos(t)
        return Bp, Bt, Br
    

class MagBG_T2(AxisymVectorBG):
    """
    T2-Shell, conforming to insulating BC @ arbitrary Ri, Ro
    """

    def __init__(self, *args, Ri=0.35, Ro=1.0, **kwargs):
        self.Ri = Ri
        self.Ro = Ro

    def __call__(self, r, t):
        Ri, Ro = self.Ri, self.Ro
        alpha = 4/(Ro - Ri)**2
        Bp = alpha*(r - Ro)*(r - Ri)*np.sin(2*t)
        Bt = np.zeros((1, t.size, r.size))
        Br = np.zeros((1, t.size, r.size))
        return Bp, Bt, Br


class MagBG_T2_Linear(AxisymVectorBG):
    """
    T2-Shell, linear profile
    """

    def __init__(self, *args, Ri=0.35, Ro=1.0, Bi=1, Bo=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.Ri = Ri
        self.Ro = Ro
        self.Bi = Bi
        self.Bo = Bo
    
    def __call__(self, r, t):
        Bp = ((r - self.Ri)/(self.Ro - self.Ri)*(self.Bo - self.Bi) + self.Bi)*np.sin(2*t)
        Bt = np.zeros((1, t.size, r.size))
        Br = np.zeros((1, t.size, r.size))
        return Bp, Bt, Br


class MagBG_T2_conductingIC(AxisymVectorBG):
    """
    T2-Shell, conforming to insulating BC @ Ro and perfect conducting BC @ Ri
    """

    def __init__(self, *args, Ri=0.71, Ro=1.0, **kwargs):
        self.Ri = Ri
        self.Ro = Ro
        L_bg, N_bg = 2, 3
        T_basis = qbasis.ChebyshevT(N_bg, interval=(Ri, Ro))
        Plm_basis = qbasis.LegendrePlm(L_bg, 0)
        self.f_B0 = qfield.ShellTorPol_m(T_basis, Plm_basis, 'tor', dtype=np.float64)
        b = (3*Ri - 2*Ro)/(2*Ri - Ro)*Ri
        c = 1/((Ro - Ri)**2*Ri/(Ro - 2*Ri))
        T_func = lambda r: c*(r - b)*(r - Ro)
        self.f_B0.spectrum[2*N_bg:3*N_bg] = T_basis.p2s(T_func(T_basis.g0))

    def __call__(self, r, t):
        B0_val = self.f_B0.eval_mesh(r.flatten(), t.flatten())
        Bp = B0_val['p'][np.newaxis, ...]
        Bt = B0_val['t'][np.newaxis, ...]
        Br = B0_val['r'][np.newaxis, ...]
        return Bp, Bt, Br
    

class MagBG_SolarTa(AxisymVectorBG):

    def __init__(self, *args, Ro=1./0.29, **kwargs):
        self.Ro = Ro
        self.Ro_int = 1./0.29

    def __call__(self, r, t):
        r_int = (self.Ro_int/self.Ro)*r
        frad = (
            +757.564396 + r_int*(
            -1390.500753 + r_int*(
            +1019.975136 + r_int*(
            -373.087646 + r_int*(
            +68.0328443 + r_int*(-4.94813137)))))
        )
        Plm_basis = qbasis.LegendrePlm(10, 0)
        spec_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.00389425606, 0, -0.00419644077, 0, +0.00100597563])
        fcla = (Plm_basis.op_dtheta(grid=t.flatten()) @ spec_Plm).reshape((1, t.size, 1))
        Bp = frad*fcla
        Bt = np.zeros((1, t.size, r.size))
        Br = np.zeros((1, t.size, r.size))
        return Bp, Bt, Br


class MagBG_SolarTa_pro(AxisymVectorBG):

    def __init__(self, Ro=1./0.29, *args, **kwargs):
        self.Ro = Ro
        self.Ro_int = 1./0.29

    def __call__(self, r, t):
        r_int = (self.Ro_int/self.Ro)*r
        frad = (
            +757.564396 + r_int*(
            -1390.500753 + r_int*(
            +1019.975136 + r_int*(
            -373.087646 + r_int*(
            +68.0328443 + r_int*(-4.94813137)))))
        )
        Plm_basis = qbasis.LegendrePlm(10, 0)
        spec_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.0389425606, 0, -0.0419644077, 0, +0.0100597563])
        fcla = (Plm_basis.op_dtheta(grid=t.flatten()) @ spec_Plm).reshape((1, t.size, 1))
        Bp = frad*fcla
        Bt = np.zeros((1, t.size, r.size))
        Br = np.zeros((1, t.size, r.size))
        return Bp, Bt, Br


class MagBG_SolarTa_bimodal(AxisymVectorBG):

    def __init__(self, Ri, Ro, *args, **kwargs):
        self.Ri = Ri
        self.Ro = Ro

        L, Nr = 10, 7
        Plm_basis = qbasis.LegendrePlm(L, 0)
        T_basis = qbasis.ChebyshevT(Nr, interval=(Ri, Ro))
        self.f_B0 = qfield.ShellTorPol_m(T_basis, Plm_basis, 'tor', dtype=np.float64)
        s_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.0389425606, 0, -0.0419644077, 0, +0.0100597563])
        s_T = np.array([ 1.41201805, -0.20244305,  0.0946188 , -0.1392218 , -0.24359233, -0.16432093, -0.05104973])
        for l in range(L+1):
            self.f_B0.spectrum[l*Nr:(l+1)*Nr] = s_Plm[l]*s_T

    def __call__(self, r, t):
        B0_val = self.f_B0.eval_mesh(r.flatten(), t.flatten())
        Bp = B0_val['p'][np.newaxis, ...]
        Bt = B0_val['t'][np.newaxis, ...]
        Br = B0_val['r'][np.newaxis, ...]
        return Bp, Bt, Br


class MagBG_SolarTa_bimodal_varNBC(AxisymVectorBG):

    def __init__(self, Ri, Ro, *args, **kwargs):
        self.Ri = Ri
        self.Ro = Ro

        L, Nr = 10, 5
        Plm_basis = qbasis.LegendrePlm(L, 0)
        T_basis = qbasis.ChebyshevT(Nr, interval=(Ri, Ro))
        self.f_B0 = qfield.ShellTorPol_m(T_basis, Plm_basis, 'tor', dtype=np.float64)
        s_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.0389425606, 0, -0.0419644077, 0, +0.0100597563])
        s_T = np.array([1.59690488, 0, 0.298990390, 0, -0.109705025])
        for l in range(L+1):
            self.f_B0.spectrum[l*Nr:(l+1)*Nr] = s_Plm[l]*s_T

    def __call__(self, r, t):
        B0_val = self.f_B0.eval_mesh(r.flatten(), t.flatten())
        Bp = B0_val['p'][np.newaxis, ...]
        Bt = B0_val['t'][np.newaxis, ...]
        Br = B0_val['r'][np.newaxis, ...]
        return Bp, Bt, Br


class MagBG_SolarTa_Linear(AxisymVectorBG):
    """
    T2-Shell, linear profile
    """

    def __init__(self, *args, Ri=0.35, Ro=1.0, Bi=1, Bo=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.Ri = Ri
        self.Ro = Ro
        self.Bi = Bi
        self.Bo = Bo
        self.Plm_basis = qbasis.LegendrePlm(10, 0)
        self.spec_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.0389425606, 0, -0.0419644077, 0, +0.0100597563])
    
    def __call__(self, r, t):
        frad = (r - self.Ri)/(self.Ro - self.Ri)*(self.Bo - self.Bi) + self.Bi
        fcla = (self.Plm_basis.op_dtheta(grid=t.flatten()) @ self.spec_Plm).reshape((1, t.size, 1))
        Bp = frad*fcla
        Bt = np.zeros((1, t.size, r.size))
        Br = np.zeros((1, t.size, r.size))
        return Bp, Bt, Br


class MagBG_SolarTa_FastDecay(AxisymVectorBG):
    """
    SolarTa toroidal field (angular) x Fast-decaying radial profile
    """

    def __init__(self, Ri, Ro, *args, **kwargs):
        self.Ri = Ri
        self.Ro = Ro

        L, Nr = 10, 6
        Plm_basis = qbasis.LegendrePlm(L, 0)
        T_basis = qbasis.ChebyshevT(Nr, interval=(Ri, Ro))
        self.f_B0 = qfield.ShellTorPol_m(T_basis, Plm_basis, 'tor', dtype=np.float64)
        s_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.0389425606, 0, -0.0419644077, 0, +0.0100597563])
        s_T = np.array([0.97091952, -0.49407104,  0.07285458,  0.00857464, -0.05667983, -0.01613811])
        for l in range(L+1):
            self.f_B0.spectrum[l*Nr:(l+1)*Nr] = s_Plm[l]*s_T

    def __call__(self, r, t):
        B0_val = self.f_B0.eval_mesh(r.flatten(), t.flatten())
        Bp = B0_val['p'][np.newaxis, ...]
        Bt = B0_val['t'][np.newaxis, ...]
        Br = B0_val['r'][np.newaxis, ...]
        return Bp, Bt, Br


class MagBG_SolarTa_iPCoI(AxisymVectorBG):
    """
    SolarTa toroidal field (angular) x lowest-degree Galerkin basis satisfying
        inner Perfectly Conducting (iPC) condition, &
        outer Insulating (oI) condition
    """

    def __init__(self, Ri, Ro, *args, **kwargs):
        self.Ri = Ri
        self.Ro = Ro

        L, Nr = 10, 3
        Plm_basis = qbasis.LegendrePlm(L, 0)
        T_basis = qbasis.ChebyshevT(Nr, interval=(Ri, Ro))
        self.f_B0 = qfield.ShellTorPol_m(T_basis, Plm_basis, 'tor', dtype=np.float64)
        s_Plm = np.array([0, 0, +1.13435906, 0, -0.198508946, 0, +0.0389425606, 0, -0.0419644077, 0, +0.0100597563])
        s_T = np.array([1.14788732, -0.5, -0.07394366])
        for l in range(L+1):
            self.f_B0.spectrum[l*Nr:(l+1)*Nr] = s_Plm[l]*s_T

    def __call__(self, r, t):
        B0_val = self.f_B0.eval_mesh(r.flatten(), t.flatten())
        Bp = B0_val['p'][np.newaxis, ...]
        Bt = B0_val['t'][np.newaxis, ...]
        Br = B0_val['r'][np.newaxis, ...]
        return Bp, Bt, Br


B0_Lib = {
    'U': MagBG_U,
    'S1': MagBG_S1,
    'S1-W4': MagBG_S1_Waves4,
    'S1-W4-R1': MagBG_S1_Waves4_R1,
    'T2': MagBG_T2
}


"""
Differential rotation profiles
"""


class SolarDiffRot_Grid(AxisymVectorBG):
    """
    Interpolation-based background flow due to differential rotation
    """

    default_diff_rot = os.path.join(os.path.dirname(os.path.realpath(__file__)), "diff_rot.npz")

    def __init__(self, Ro, *args, Omega0=456., normalize=456., path_diff_rot=None, **kwargs):
        self.Ro = Ro
        self.Omega0 = Omega0
        self.norm = normalize
        self.path_diff_rot = self.default_diff_rot if path_diff_rot is None else path_diff_rot
        self.f_R = self.build_rot_profile()
    
    def build_rot_profile(self):
        DR_data = np.load(self.path_diff_rot)
        f_rot = interpolate.RectBivariateSpline(DR_data['r'], DR_data['theta'], DR_data['ome'].T)
        return f_rot
    
    def diff_rot(self, r, t):
        dOmega = self.f_R(r.flatten()/self.Ro, t.flatten()[::-1]).T[np.newaxis, ::-1, :]
        DR = (dOmega - self.Omega0)/self.norm
        return DR
    
    def __call__(self, r, t):
        DR = self.diff_rot(r, t)
        Up = r*np.sin(t)*np.reshape(DR, (1, t.size, r.size))
        Ut = np.zeros((1, t.size, r.size))
        Ur = np.zeros((1, t.size, r.size))
        return Up, Ut, Ur


class SolarDiffRot_LS18(AxisymVectorBG):
    """
    Interpolation-based background flow due to differential rotation
    """

    default_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/MDI")

    def __init__(self, Ro, *args, Omega0=456., normalize=456., path_data=None, **kwargs):
        self.Ro = Ro
        self.Omega0 = Omega0
        self.norm = normalize
        self.path_data = self.default_data_dir if path_data is None else path_data
        self.f_R = self.build_rot_profile()
    
    def build_rot_profile(self):

        Omega = pd.read_table(f"{self.path_data}/rot2d.mdifd.ave", sep=r'\s+', header=None).to_numpy(dtype=np.float64).T
        Err = pd.read_table(f"{self.path_data}/err2d.mdifd.ave", sep=r'\s+', header=None).to_numpy(dtype=np.float64).T
        rg = pd.read_table(f"{self.path_data}/rmesh.orig", header=None).to_numpy(dtype=np.float64).flatten()[::4]
        tg = np.radians(np.arange(Omega.shape[0])*15/8)

        Omega = np.r_[Omega, np.flipud(Omega)[1:]]
        Err = np.r_[Err, np.flipud(Err)[1:]]
        tg = np.r_[tg, np.pi - np.flip(tg)[1:]]
        f_rot = interpolate.RectBivariateSpline(rg, tg, Omega.T)
        return f_rot
    
    def diff_rot(self, r, t):
        dOmega = self.f_R(r.flatten()/self.Ro, t.flatten()[::-1]).T[np.newaxis, ::-1, :]
        DR = (dOmega - self.Omega0)/self.norm
        return DR
    
    def __call__(self, r, t):
        DR = self.diff_rot(r, t)
        Up = r*np.sin(t)*np.reshape(DR, (1, t.size, r.size))
        Ut = np.zeros((1, t.size, r.size))
        Ur = np.zeros((1, t.size, r.size))
        return Up, Ut, Ur


class SolarDiffRot_SHT(AxisymVectorBG):
    """
    Differential rotation background flow with band-limited spherical-harmonic-Chebyshev repr
    """

    def __init__(self, Ro, L=8, Nr=5, Omega0=456., normalize=456., **kwargs):
        self.Ro = Ro
        self.L = L
        self.Nr = Nr
        self.Omega0 = Omega0
        self.norm = normalize
        self.path_spec = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"OSpec_SH{L}-T{Nr}.npy")
        self.f_R = self.build_rot_profile()
    
    def build_rot_profile(self):
        spec = np.load(self.path_spec)
        T_basis = qbasis.ChebyshevT(self.Nr, interval=(0.71, 1.))
        Plm_basis = qbasis.LegendrePlm(self.L, 0)
        scalar_Omega = qfield.ShellScalar_m(T_basis, Plm_basis, spec, dtype=np.float64)
        return scalar_Omega
    
    def diff_rot(self, r, t):
        rg = r.flatten()/self.Ro
        if rg.max() > 1.0 + 1e-7 or rg.min() < 0.71 - 1e-7:
            warnings.warn(
                "Spectral representation of differential rotation constructed within radii 0.71 - 1 x R_sun.\n"
                "Evaluating differential rotation outside this radius range might cause unexpected behaviour!\n"
                f"Current radius range: {rg.min():.3f} < r/Ro < {rg.max():.3f}")
        dOmega = self.f_R.eval_mesh(rg, t.flatten())
        DR = (dOmega - self.Omega0)/self.norm
        return DR
    
    def __call__(self, r, t):
        DR = self.diff_rot(r, t)
        Up = r*np.sin(t)*np.reshape(DR, (1, t.size, r.size))
        Ut = np.zeros((1, t.size, r.size))
        Ur = np.zeros((1, t.size, r.size))
        return Up, Ut, Ur


class SolarDiffRot_SHT_LSQ(AxisymVectorBG):
    """
    Differential rotation background flow with least-square-constructed band-limited spherical-harmonic-Chebyshev repr
    """

    def __init__(self, Ro, L=8, Nr=5, Omega0=456., normalize=456., **kwargs):
        self.Ro = Ro
        self.L = L
        self.Nr = Nr
        self.Omega0 = Omega0
        self.norm = normalize
        self.path_spec = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"OSpec_SH{L}-T{Nr}-Lsq.npy")
        self.f_R = self.build_rot_profile()
    
    def build_rot_profile(self):
        spec = np.load(self.path_spec)
        T_basis = qbasis.ChebyshevT(self.Nr, interval=(0.71, 1.))
        Plm_basis = qbasis.LegendrePlm(self.L, 0)
        scalar_Omega = qfield.ShellScalar_m(T_basis, Plm_basis, spec, dtype=np.float64)
        return scalar_Omega
    
    def diff_rot(self, r, t):
        rg = r.flatten()/self.Ro
        if rg.max() > 1.0 + 1e-7 or rg.min() < 0.71 - 1e-7:
            warnings.warn(
                "Spectral representation of differential rotation constructed within radii 0.71 - 1 x R_sun.\n"
                "Evaluating differential rotation outside this radius range might cause unexpected behaviour!\n"
                f"Current radius range: {rg.min():.3f} < r/Ro < {rg.max():.3f}")
        dOmega = self.f_R.eval_mesh(rg, t.flatten())
        DR = (dOmega - self.Omega0)/self.norm
        return DR
    
    def __call__(self, r, t):
        DR = self.diff_rot(r, t)
        Up = r*np.sin(t)*np.reshape(DR, (1, t.size, r.size))
        Ut = np.zeros((1, t.size, r.size))
        Ur = np.zeros((1, t.size, r.size))
        return Up, Ut, Ur


"""
Anelastic background density profiles
"""


class ScalarRProfile:

    def __init__(self, *args, **kwds) -> None:
        super().__init__()

    def __call__(self, r):
        raise NotImplementedError


class rDensity_SCZ_N18_Scale5_5(ScalarRProfile):
    """
    Reciprocal density profile for the solar convection zone
    with 5.5 density scale heights in 0.71R < r < R, 
    polynomial fit within 0.71 to 1 outer sphere radius prioritizing inner region
    """

    def __init__(self, *args, Ro=1.0, **kwds) -> None:
        self.Ro = Ro
        self.Ro_int = 1.0
        super().__init__(*args, **kwds)

    def __call__(self, r):
        r_int = (self.Ro_int/self.Ro)*r
        r_ptb = r_int - 0.71
        rho = (
            1. + r_ptb*(
            7. + r_ptb**2*(
            500. + r_ptb**2*(
            700. + r_ptb**2*(
            1.0e+5 + r_ptb**3*(
            1.0e+7 + r_ptb**8*(7.7e+11))))))
        )
        return rho


class rDensity_SCZ_Scale6(ScalarRProfile):
    """
    Reciprocal density profile for the solar convection zone
    with 6.0 density scale heights in 0.71R < r < R, 
    polynomial fit within 0.71 to 1 outer sphere radius prioritizing inner region
    """

    def __init__(self, *args, Ro=1.0, **kwds) -> None:
        self.Ro = Ro
        self.Ro_int = 1.0
        super().__init__(*args, **kwds)

    def __call__(self, r):
        r_int = (self.Ro_int/self.Ro)*r
        r_ptb = r_int - 0.71
        rho = (
            1. + r_ptb*(
            7. + r_ptb**2*(
            500. + r_ptb**2*(
            700. + r_ptb**2*(
            1.0e+5 + r_ptb**3*(
            1.0e+7 + r_ptb**10*(
            1.0e+13 + r_ptb**30*(1.0e+30)))))))
        )
        return rho
    

class rDensity_SCZ_N24_Scale6_5(ScalarRProfile):
    """
    Reciprocal density profile for the solar convection zone
    with 6.5 density scale heights in 0.71R < r < R (rho(0.71R) = 650 rho(R)), 
    polynomial fit within 0.71 to 1 outer sphere radius prioritizing inner region,
    maximum relative error 20% within 0.71R < r < 0.985R (rho(0.71R) = 205 rho(0.985R))
    """

    def __init__(self, *args, Ro=1.0, **kwds) -> None:
        self.Ro = Ro
        self.Ro_int = 1.0
        super().__init__(*args, **kwds)

    def __call__(self, r):
        r_int = (self.Ro_int/self.Ro)*r
        r_ptb = r_int - 0.71
        rho = (
            1. + r_ptb*(
            7. + r_ptb**2*(
            500. + r_ptb**2*(
            700. + r_ptb**2*(
            1.0e+5 + r_ptb**3*(
            1.0e+7 + r_ptb**14*(4.48e+15))))))
        )
        return rho


class rDensity_SCZ_Scale7_5(ScalarRProfile):
    """
    Reciprocal density profile for the solar convection zone 
    with 7.5 density scale heights in 0.71R < r < R, 
    polynomial fit within 0.71 to 1 outer sphere radius prioritizing inner region
    """

    def __init__(self, *args, Ro=1.0, **kwds) -> None:
        self.Ro = Ro
        self.Ro_int = 1.0
        super().__init__(*args, **kwds)

    def __call__(self, r):
        r_int = (self.Ro_int/self.Ro)*r
        r_ptb = r_int - 0.71
        rho = (
            1. + r_ptb*(
            7. + r_ptb**2*(
            5.0e+2 + r_ptb**2*(
            7.0e+2 + r_ptb**2*(
            1.0e+5 + r_ptb**3*(
            1.0e+7 + r_ptb**10*(
            1.0e+13 + r_ptb**30*(1.0795e+30)))))))
        )
        return rho


class LinearProfile(ScalarRProfile):
    """
    Scalar profile varying linearly between Ri and Ro
    """

    def __init__(self, *args, Ri=0.35, Ro=1.0, fi=1.0, fo=1.0, **kwds) -> None:
        self.Ri = Ri
        self.Ro = Ro
        self.fi = fi
        self.fo = fo
        super().__init__(*args, **kwds)

    def __call__(self, r):
        f_vals = (r - self.Ri)/(self.Ro - self.Ri)*(self.fo - self.fi) + self.fi
        return f_vals


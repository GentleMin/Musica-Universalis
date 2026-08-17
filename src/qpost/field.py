# -*- coding: utf-8 -*-


import numpy as np
from scipy import sparse, special
from dataclasses import dataclass, field
from typing import Optional, Literal, Union
from . import basis as qbasis


def iparity(N, dL, sym=0):
    """Get indices of a scalar spectrum corresponding to symmetry
    """
    idx = np.arange(N*dL).reshape(dL, N)
    idx = idx[sym::2, :].flatten()
    return idx


def iparity_v(N, dL, sym=0, comp='pol'):
    """Get indices of a tor/pol component spectrum corresponding to symmetry
    """
    sym_tp = 1 if comp == 'tor' else 0
    sym = (sym + sym_tp) % 2
    return iparity(N, dL, sym=sym)


@dataclass
class OrderingSpherical_m:
    """
    Spectral ordering for spherical geometry, single m
    """
    N: int
    L: int
    m: int

    def __post_init__(self):
        self.res = (self.N, self.L, self.m)
        self.dim = self.N * (self.L - self.m + 1)

    def idx(self, l, n):
        return (l - self.m) * self.N + n

    def mode_l(self, l):
        return self.idx(l, 0), self.idx(l, self.N-1) + 1
    

class ShellScalar_m:

    def __init__(self, 
        r_basis: qbasis.ChebyshevT, 
        t_basis: qbasis.LegendrePlm, 
        spectrum: Optional[np.ndarray] = None,
        dtype = np.complex128
    ):
        self.r_basis = r_basis
        self.t_basis = t_basis
        
        self.ri = self.r_basis.range[0]
        self.ro = self.r_basis.range[1]
        self.N, self.L, self.m = self.r_basis.Ns, self.t_basis.L, self.t_basis.m
        self.order = OrderingSpherical_m(self.N, self.L, self.m)
        if spectrum is None:
            self.spectrum = np.zeros(self.order.dim, dtype=dtype)
        else:
            if self.order.dim != spectrum.shape[0]:
                raise RuntimeError("Data shape does not match input resolution")
            self.spectrum = spectrum.copy()
        self.energy_spectrum_l = None
        self.ops = {'R': {}, 'T': {}}

    @classmethod
    def from_parity_spectrum(cls, 
        r_basis: qbasis.ChebyshevT, 
        t_basis: qbasis.LegendrePlm, 
        spec: np.ndarray, 
        parity: int
    ):
        """
        Spectral construction given spectral coefficients for a given parity
        """
        N, L, m = r_basis.Ns, t_basis.L, t_basis.m
        order = OrderingSpherical_m(N, L, m)
        sp = np.zeros((order.dim,), dtype=spec.dtype)
        idx_parity = iparity(N, L - m, sym=parity)
        sp[idx_parity] = spec
        return cls(r_basis, t_basis, sp)

    def mode_l(self, l):
        a, b = self.order.mode_l(l)
        return self.spectrum[a:b]

    def calc_espec_l(self):
        """
        Compute energy spectrum in l
        """
        m, L, N = self.m, self.L, self.N
        energy_spec_l = np.zeros(L - m + 1, dtype=np.float64)
        rg, wt = self.r_basis.grids_legendre(N + 3)
        Kernel = self.r_basis.op_id(grid=rg)
        Kernel = Kernel.T @ sparse.diags(wt*rg**2) @ Kernel
        for l in range(m, L+1):
            spec_l = self.mode_l(l)
            energy_spec_l[l-m] = np.abs(spec_l.conj().T @ Kernel @ spec_l)
        self.energy_spectrum_l = energy_spec_l
        return self.energy_spectrum_l

    def eval_mesh(self, rg: np.ndarray, tg: np.ndarray):

        m, L, N = self.m, self.L, self.N
        nrg = rg.size
        ntg = tg.size
        
        self.r_basis.setup_ops(grid=rg)
        self.t_basis.setup_ops(grid=tg)
        R_ops = {
            'id': sparse.block_diag([self.r_basis.ops['id'],]*(L-m+1)),
        }        
        radial = (R_ops['id'] @ self.spectrum).reshape(-1, nrg)
        fval = self.t_basis.ops['id'] @ radial        
        return fval
    
    def laplacian(self, inplace=True):
        """
        Take Laplacian of the component
        """
        field = self if inplace else self.copy()
        m, L, N = field.m, field.L, field.N
        Tid = field.r_basis.op_id(grid=field.r_basis.g0)
        D1T = field.r_basis.op_d1(grid=field.r_basis.g0)
        D2T = field.r_basis.op_d2(grid=field.r_basis.g0)
        for l in range(m, L+1):
            fval_curl = (
                + D2T @ field.mode_l(l)
                + (2/field.r_basis.g0)*(D1T @ field.mode_l(l))
                - (l*(l+1)/field.r_basis.g0**2)*(Tid @ field.mode_l(l))
            )
            i0, i1 = field.order.mode_l(l)
            field.spectrum[i0:i1] = field.r_basis.p2s(fval_curl)
        return field

    def restrict_parity(self, parity: int):
        """
        Set the spectrum of certain parity to be zero
        """
        N, L, m = self.N, self.L, self.m
        idx_parity = iparity(N, L - m, sym=parity)
        self.spectrum[idx_parity] *= 0.

    def padding(self, N, L):
        """
        Pad to higher resolution
        """
        assert L >= self.L
        assert N >= self.N
        N0, L0, m = self.N, self.L, self.m
        r_basis = qbasis.ChebyshevT(N, self.r_basis.range)
        t_basis = qbasis.LegendrePlm(L, m)
        field = ShellScalar_m(r_basis, t_basis, dtype=self.spectrum.dtype)

        for l in range(m, L0):
            i0, i1 = field.order.mode_l(l)
            field.spectrum[i0:i0+N0] = self.mode_l(l)
        return field

    def copy(self):
        return ShellScalar_m(self.r_basis, self.t_basis, spectrum=self.spectrum.copy())


class ShellTorPol_m:
    """
    A single wavenumber m field in spherical shell geometry
    """

    def __init__(self, 
        r_basis: qbasis.ChebyshevT, 
        t_basis: qbasis.LegendrePlm, 
        comp: Literal['pol', 'tor'], 
        spectrum: Optional[np.ndarray] = None,
        dtype = np.complex128
    ):
        self.r_basis = r_basis
        self.t_basis = t_basis
        self.comp = comp
        assert self.comp in ['tor', 'pol'], "field component mush be either 'tor' or 'pol'."
        
        self.ri = self.r_basis.range[0]
        self.ro = self.r_basis.range[1]
        self.N, self.L, self.m = self.r_basis.Ns, self.t_basis.L, self.t_basis.m
        self.order = OrderingSpherical_m(self.N, self.L, self.m)
        if spectrum is None:
            self.spectrum = np.zeros(self.order.dim, dtype=dtype)
        else:
            if self.order.dim != spectrum.shape[0]:
                raise RuntimeError("Data shape does not match input resolution")
            self.spectrum = spectrum.copy()
        self.energy_spectrum_l = None

    @classmethod
    def from_parity_spectrum(cls, 
        r_basis: qbasis.ChebyshevT, 
        t_basis: qbasis.LegendrePlm, 
        comp: Literal['pol', 'tor'], 
        spec: np.ndarray, 
        parity: int
    ):
        """
        Spectral construction given spectral coefficients for a given parity
        """
        N, L, m = r_basis.Ns, t_basis.L, t_basis.m
        order = OrderingSpherical_m(N, L, m)
        sp = np.zeros((order.dim,), dtype=spec.dtype)
        idx_parity = iparity_v(N, L - m, sym=parity, comp=comp)
        sp[idx_parity] = spec
        return cls(r_basis, t_basis, comp, sp)

    def mode_l(self, l):
        a, b = self.order.mode_l(l)
        return self.spectrum[a:b]

    def calc_espec_l(self):
        """
        Compute energy spectrum in l
        """
        m, L, N = self.m, self.L, self.N
        energy_spec_l = np.zeros(L - m + 1, dtype=np.float64)
        rg, wt = self.r_basis.grids_legendre(N + 3)

        if self.comp == 'tor':
            Kernel = self.r_basis.op_id(grid=rg)
            Kernel = Kernel.T @ sparse.diags(wt*rg**2) @ Kernel
            for l in range(m, L+1):
                spec_l = self.mode_l(l)
                energy_spec_l[l-m] = (l*(l+1))*np.abs(spec_l.conj().T @ Kernel @ spec_l)
            self.energy_spectrum_l = energy_spec_l
            return self.energy_spectrum_l
        elif self.comp == 'pol':
            K1 = self.r_basis.op_id(grid=rg)
            K2 = K1 + sparse.diags(rg) @ self.r_basis.op_d1(grid=rg)
            K1 = K1.T @ sparse.diags(wt) @ K1
            K2 = K2.T @ sparse.diags(wt) @ K2
            for l in range(m, L+1):
                L2 = (l*(l+1))
                spec_l = self.mode_l(l)
                energy_spec_l[l-m] = (
                    (L2**2)*np.abs(spec_l.conj().T @ K1 @ spec_l) + 
                    L2*np.abs(spec_l.conj().T @ K2 @ spec_l)
                )
            self.energy_spectrum_l = energy_spec_l
            return self.energy_spectrum_l
        else:
            raise RuntimeError(f"Unknown component {self.comp}, must be either 'pol' or 'tor'.")

    def eval_mesh(self, rg: np.ndarray, tg: np.ndarray):

        m, L, N = self.m, self.L, self.N
        nrg = rg.size
        ntg = tg.size
        
        self.r_basis.setup_ops(grid=rg)
        self.t_basis.setup_ops(grid=tg)
        op_divr = self.r_basis.ops['id']/rg.reshape(-1, 1)
        op_divrDr = op_divr + self.r_basis.ops['d1']
        R_ops = {
            'id': sparse.block_diag([self.r_basis.ops['id'],]*(L-m+1)),
            'divr': sparse.block_diag([op_divr,]*(L-m+1)),
            'divrDr': sparse.block_diag([op_divrDr,]*(L-m+1))
        }

        if self.comp == 'tor':
            radial = (R_ops['id'] @ self.spectrum).reshape(-1, nrg)
            r_comp = np.zeros((ntg, nrg))
            t_comp = 1.j*m*self.t_basis.ops['div_sin'] @ radial if m > 0 else r_comp.copy()
            p_comp = - self.t_basis.ops['d_theta'] @ radial

        elif self.comp == 'pol':
            # radial1 = (worland_transform.operators['divrW'] @ self.spectrum).reshape(-1, nrg)
            radial1 = (R_ops['divr'] @ self.spectrum).reshape(-1, nrg)
            radial2 = (R_ops['divrDr'] @ self.spectrum).reshape(-1, nrg)
            l_factor = sparse.diags([l*(l + 1) for l in range(m, L+1)], dtype=np.int32)
            r_comp = self.t_basis.ops['id'] @ l_factor @ radial1
            t_comp = self.t_basis.ops['d_theta'] @ radial2
            p_comp = 1.0j * m * self.t_basis.ops['div_sin'] @ radial2 if m > 0 else np.zeros_like(t_comp)

        else:
            raise RuntimeError(f"Unknown component {self.comp}, must be either 'pol' or 'tor'.")
        
        return {'r': r_comp, 't': t_comp, 'p': p_comp}
    
    def curl(self, inplace=True):
        """
        Take curl of the component
        """
        field = self if inplace else self.copy()
        if field.comp == "tor":
            field.comp = "pol"
        else:
            field.comp = "tor"
            m, L, N = field.m, field.L, field.N
            Tid = field.r_basis.op_id(grid=field.r_basis.g0)
            D1T = field.r_basis.op_d1(grid=field.r_basis.g0)
            D2T = field.r_basis.op_d2(grid=field.r_basis.g0)
            for l in range(m, L+1):
                fval_curl = (
                    - D2T @ field.mode_l(l)
                    - (2/field.r_basis.g0)*(D1T @ field.mode_l(l))
                    + (l*(l+1)/field.r_basis.g0**2)*(Tid @ field.mode_l(l))
                )
                i0, i1 = field.order.mode_l(l)
                field.spectrum[i0:i1] = field.r_basis.p2s(fval_curl)
        return field
    
    def laplacian(self, inplace=True):
        """
        Take Laplacian of the component
        """
        field = self if inplace else self.copy()
        m, L, N = field.m, field.L, field.N
        Tid = field.r_basis.op_id(grid=field.r_basis.g0)
        D1T = field.r_basis.op_d1(grid=field.r_basis.g0)
        D2T = field.r_basis.op_d2(grid=field.r_basis.g0)
        for l in range(m, L+1):
            fval_curl = (
                + D2T @ field.mode_l(l)
                + (2/field.r_basis.g0)*(D1T @ field.mode_l(l))
                - (l*(l+1)/field.r_basis.g0**2)*(Tid @ field.mode_l(l))
            )
            i0, i1 = field.order.mode_l(l)
            field.spectrum[i0:i1] = field.r_basis.p2s(fval_curl)
        return field

    def restrict_parity(self, parity: int):
        """
        Set the spectrum of certain parity to be zero
        """
        N, L, m = self.N, self.L, self.m
        idx_parity = iparity_v(N, L - m, sym=parity, comp=self.comp)
        self.spectrum[idx_parity] *= 0.

    def padding(self, N, L):
        """
        Pad to higher resolution
        """
        assert L >= self.L
        assert N >= self.N
        N0, L0, m = self.N, self.L, self.m
        r_basis = qbasis.ChebyshevT(N, self.r_basis.range)
        t_basis = qbasis.LegendrePlm(L, m)
        field = ShellTorPol_m(r_basis, t_basis, self.comp, dtype=self.spectrum.dtype)

        for l in range(m, L0):
            i0, i1 = field.order.mode_l(l)
            field.spectrum[i0:i0+N0] = self.mode_l(l)
        return field

    def copy(self):
        return ShellTorPol_m(self.r_basis, self.t_basis, self.comp, spectrum=self.spectrum.copy())


class ShellVectorTorPol_m:
    """
    A single-wavenumber-m vector field in spherical shell geometry
    in toroidal-poloidal representation
    """

    def __init__(self, 
        r_basis: qbasis.ChebyshevT, 
        t_basis: qbasis.LegendrePlm, 
        spectrum: np.ndarray
    ):
        self.r_basis = r_basis
        self.t_basis = t_basis
        
        self.ri = self.r_basis.range[0]
        self.ro = self.r_basis.range[1]
        self.N, self.L, self.m = self.r_basis.Ns, self.t_basis.L, self.t_basis.m
        self.order = OrderingSpherical_m(self.N, self.L, self.m)

        dim = spectrum.size // 2
        if self.order.dim != dim:
            raise RuntimeError("Data shape does not match input resolution")
        self.components = {
            "tor": ShellTorPol_m(r_basis, t_basis, "tor", spectrum[:dim]),
            "pol": ShellTorPol_m(r_basis, t_basis, "pol", spectrum[dim:])
        }

    @classmethod
    def from_components(cls, tor: ShellTorPol_m, pol: ShellTorPol_m):
        """
        Construction from toroidal and poloidal components
        """
        assert tor.order.res == pol.order.res, "Incompatible resolutions for tor and pol components"
        spectrum = np.concatenate([tor.spectrum, pol.spectrum])
        return cls(tor.r_basis, tor.t_basis, spectrum)
    
    @classmethod
    def from_parity_spectrum(cls, 
        r_basis: qbasis.ChebyshevT, 
        t_basis: qbasis.LegendrePlm, 
        spec: np.ndarray, 
        parity: int
    ):
        """
        Spectral construction given spectral coefficients for a given parity
        """
        dim = spec.size // 2
        field_tor = ShellTorPol_m.from_parity_spectrum(r_basis, t_basis, 'tor', spectrum=spec[:dim], parity=parity)
        field_pol = ShellTorPol_m.from_parity_spectrum(r_basis, t_basis, 'pol', spectrum=spec[dim:], parity=parity)
        return cls.from_components(field_tor, field_pol)
    
    def spectrum(self):
        return np.concatenate([self.components["tor"].spectrum, self.components["pol"].spectrum])
    
    def calc_espec_l(self):
        espec_l_tor = self.components['tor'].calc_espec_l()
        espec_l_pol = self.components['pol'].calc_espec_l()
        return espec_l_tor + espec_l_pol
    
    def energy(self):
        return np.sum(self.calc_espec_l())
    
    def __add__(self, v: "ShellVectorTorPol_m") -> "ShellVectorTorPol_m":
        """Add two vector fields of single m together
        """
        assert self.order.res == v.order.res, "Resolution incompatible!"
        return ShellVectorTorPol_m(self.r_basis, self.t_basis, self.spetrum() + v.spetrum())
    
    def __radd__(self, v: "ShellVectorTorPol_m") -> "ShellVectorTorPol_m":
        """Right add
        """
        if v == 0:
            return self
        else:
            return self.__add__(v)

    def eval_mesh(self, rg: np.ndarray, tg: np.ndarray):
        """
        Compute physical fields given Worland transform and associated Legendre transform
        """
        field_tor = self.components["tor"].eval_mesh(rg, tg)
        field_pol = self.components["pol"].eval_mesh(rg, tg)
        field = {k: field_tor[k] + field_pol[k] for k in field_tor}
        return field

    def curl(self, inplace=True):
        """
        Transform to curl of the field
        """
        new_pol = self.components["tor"].curl(inplace=inplace)
        new_tor = self.components["pol"].curl(inplace=inplace)
        if inplace:
            self.components["tor"] = new_tor
            self.components["pol"] = new_pol
            return self
        return ShellVectorTorPol_m.from_components(new_tor, new_pol)
        
    def laplacian(self, inplace=True):
        """
        Transform to Laplacian of the field
        """
        new_tor = self.components["tor"].laplacian(inplace=inplace)
        new_pol = self.components["pol"].laplacian(inplace=inplace)
        if inplace:
            return self
        return ShellVectorTorPol_m.from_components(new_tor, new_pol)

    def restrict_parity(self, parity):
        """
        Set the spectrum of certain parity to be zero
        """
        self.components['tor'].restrict_parity(parity)
        self.components["pol"].restrict_parity(parity)

    def padding(self, N, L):
        """
        Pad zero to a higher resolution
        """
        new_tor = self.components["tor"].padding(N, L)
        new_pol = self.components["pol"].padding(N, L)
        return ShellVectorTorPol_m.from_components(new_tor, new_pol)
    
    def copy(self):
        return ShellVectorTorPol_m(self.r_basis, self.t_basis, self.spetrum())


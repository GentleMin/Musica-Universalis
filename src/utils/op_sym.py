# coding: utf-8 -*-


import numpy as np
import sympy as sym
from sympy import diff, I, S


def dot(a, b):
    c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return c

def cross(a, b):
    c = np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])
    return c

def conj(vA):
    v_conj = np.array([sym.conjugate(vi) for vi in vA])
    return v_conj


class SphericalCoordinates:

    def __init__(self, r, theta, phi):
        self.r = r
        self.t = theta
        self.p = phi

    def surf_comp(self, vA):
        return np.array([0, vA[1], vA[2]])

    def curl(self, a):
        r, t, p = self.r, self.t, self.p
        s = sym.sin(t)
        curl_a = np.array([
            diff(s*a[2], t)/(r*s) - diff(a[1], p)/(r*s),
            diff(a[0], p)/(r*s) - diff(r*a[2], r)/r,
            diff(r*a[1], r)/r - diff(a[0], t)/r
        ])
        return curl_a
    
    def curl_m(self, a, m):
        r, t = self.r, self.t
        s = sym.sin(t)
        curl_a = np.array([
            diff(s*a[2], t)/(r*s) - I*m*a[1]/(r*s),
            I*m*a[0]/(r*s) - diff(r*a[2], r)/r,
            diff(r*a[1], r)/r - diff(a[0], t)/r
        ])
        return curl_a

    def div(self, a):
        r, t, p = self.r, self.t, self.p
        div_a = (
            diff(r**2*a[0], r)/r**2
            + diff(sym.sin(t)*a[1], t)/(r*sym.sin(t))
            + diff(a[2], p)/(r*sym.sin(t))
        )
        return div_a

    def div_m(self, a, m):
        r, t = self.r, self.t
        div_a = (
            diff(r**2*a[0], r)/r**2
            + diff(sym.sin(t)*a[1], t)/(r*sym.sin(t))
            + I*m*a[2]/(r*sym.sin(t))
        )
        return div_a

    def grad(self, a):
        r, t, p = self.r, self.t, self.p
        grad_a = np.array([
            diff(a, r),
            diff(a, t)/r,
            diff(a, p)/(r*sym.sin(t))
        ])
        return grad_a

    def grad_m(self, a, m):
        r, t = self.r, self.t
        grad_a = np.array([
            diff(a, r),
            diff(a, t)/r,
            I*m*a/(r*sym.sin(t))
        ])
        return grad_a

    def lap(self, a):
        return self.div(self.grad(a))

    def lap_m(self, a, m):
        return self.div_m(self.grad_m(a, m), m)
    
    def lap_Ylm(self, a, l):
        r = self.r
        return (diff(r**2*diff(a, r), r) - l*(l+1)*a)/r**2

    def lapv(self, a):
        return self.grad(self.div(a)) - self.curl(self.curl(a))

    def lapv_m(self, a, m):
        return self.grad_m(self.div_m(a, m), m) - self.curl_m(self.curl_m(a, m), m)


class CylindricalCoordinates:

    def __init__(self, s, phi, z):
        self.s = s
        self.p = phi
        self.z = z

    def curl(self, a, m):
        s, p, z = self.s, self.p, self.z
        curl_a = np.array([
            diff(a[2], p)/s - diff(a[1], z),
            diff(a[0], z) - diff(a[2], s),
            diff(s*a[1], s)/s - diff(a[0], p)/s
        ])
        return curl_a
            
    def curl_m(self, a, m):
        s, z = self.s, self.z
        curl_a = np.array([
            I*m/s*a[2] - diff(a[1], z),
            diff(a[0], z) - diff(a[2], s),
            diff(s*a[1], s)/s - I*m/s*a[0]
        ])
        return curl_a


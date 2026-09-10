# -*- coding: utf-8 -*-
"""Constraining matrices to symmetric / anti-symmetric parts
"""


import numpy as np
from scipy import sparse


def idx_sym(N, dL, sym=0):

    idx = np.arange(N*dL).reshape(dL, N)
    idx = idx[sym::2, :].flatten()
    return idx


def idx_sym_torpol(N, dL, sym=0, type_='p'):

    sym_tp = 1 if type_ == 't' else 0
    sym = (sym + sym_tp) % 2
    return idx_sym(N, dL, sym=sym)


def slice_sym_HydroTP(A, N, dL, sym=0, perm=None):
    
    A = sparse.csr_array(A)
    Nd = N*dL

    idx_v_tor = idx_sym_torpol(N, dL, sym=sym, type_='t')
    idx_v_pol = idx_sym_torpol(N, dL, sym=sym, type_='p')
    idx = np.r_[idx_v_tor, Nd + idx_v_pol]

    itau_v_t = np.arange(dL)[((sym+1)%2)::2]
    itau_v_p = np.arange(dL)[((sym+0)%2)::2]
    idx = np.r_[idx,
        2*Nd + 0*dL + itau_v_t, 2*Nd + 1*dL + itau_v_t,
        2*Nd + 2*dL + itau_v_p, 2*Nd + 3*dL + itau_v_p, 2*Nd + 4*dL + itau_v_p, 2*Nd + 5*dL + itau_v_p,
    ]

    if perm is None:
        perm = sparse.eye_array(A.shape[0], format='csr')
    
    ele_arr = np.zeros(A.shape[0])
    ele_arr[idx] = 1
    idx_perm = np.arange(A.shape[0])[np.asarray(perm @ ele_arr, dtype=bool)]

    A = A[np.ix_(idx_perm, idx_perm)]
    perm = perm[np.ix_(idx_perm, idx)]
    return A, perm


def slice_sym_MHD(A, N, dL, sym_v=0, sym_b=0, Ntrail=0, perm=None):

    A = sparse.csr_array(A)
    Nd = N*dL
    # assert A.shape[0] == 4*Nd + Ntrail

    idx_v_tor = idx_sym_torpol(N, dL, sym=sym_v, type_='t')
    idx_v_pol = idx_sym_torpol(N, dL, sym=sym_v, type_='p')
    idx_b_tor = idx_sym_torpol(N, dL, sym=sym_b, type_='t')
    idx_b_pol = idx_sym_torpol(N, dL, sym=sym_b, type_='p')
    idx = np.r_[idx_v_tor, Nd + idx_v_pol, 2*Nd + idx_b_tor, 3*Nd + idx_b_pol]

    # Tau variables
    itau_v_t = np.arange(dL)[((sym_v+1)%2)::2]
    itau_v_p = np.arange(dL)[((sym_v+0)%2)::2]
    itau_b_t = np.arange(dL)[((sym_b+1)%2)::2]
    itau_b_p = np.arange(dL)[((sym_b+0)%2)::2]
    idx = np.r_[idx, 
        4*Nd + 0*dL + itau_v_t, 4*Nd + 1*dL + itau_v_t,
        4*Nd + 2*dL + itau_v_p, 4*Nd + 3*dL + itau_v_p, 4*Nd + 4*dL + itau_v_p, 4*Nd + 5*dL + itau_v_p,
        4*Nd + 6*dL + itau_b_t, 4*Nd + 7*dL + itau_b_t,
        4*Nd + 8*dL + itau_b_p, 4*Nd + 9*dL + itau_b_p,
    ]

    if perm is None:
        perm = sparse.eye_array(A.shape[0], format='csr')
    
    ele_arr = np.zeros(A.shape[0])
    ele_arr[idx] = 1
    idx_perm = np.arange(A.shape[0])[np.asarray(perm @ ele_arr, dtype=bool)]

    A = A[np.ix_(idx_perm, idx_perm)]
    perm = perm[np.ix_(idx_perm, idx)]
    return A, perm


def slice_Hydro_from_MHD(A, N, dL, perm=None):

    A = sparse.csr_array(A)
    Nd = N*dL
    # assert A.shape[0] == 4*Nd + Ntrail

    idx_v_tor = np.arange(N*dL)
    idx_v_pol = np.arange(N*dL)
    idx = np.r_[idx_v_tor, Nd + idx_v_pol]

    # Tau variables
    itau_v_t = np.arange(dL)
    itau_v_p = np.arange(dL)
    idx = np.r_[idx, 
        4*Nd + 0*dL + itau_v_t, 4*Nd + 1*dL + itau_v_t,
        4*Nd + 2*dL + itau_v_p, 4*Nd + 3*dL + itau_v_p, 4*Nd + 4*dL + itau_v_p, 4*Nd + 5*dL + itau_v_p,
    ]

    if perm is None:
        perm = sparse.eye_array(A.shape[0], format='csr')
    
    ele_arr = np.zeros(A.shape[0])
    ele_arr[idx] = 1
    idx_perm = np.arange(A.shape[0])[np.asarray(perm @ ele_arr, dtype=bool)]

    A = A[np.ix_(idx_perm, idx_perm)]
    perm = perm[np.ix_(idx_perm, idx)]
    return A, perm


def slice_Hydro_sym_from_MHD(A, N, dL, sym_v=0, perm=None):

    A = sparse.csr_array(A)
    Nd = N*dL
    # assert A.shape[0] == 4*Nd + Ntrail

    idx_v_tor = idx_sym_torpol(N, dL, sym=sym_v, type_='t')
    idx_v_pol = idx_sym_torpol(N, dL, sym=sym_v, type_='p')
    idx = np.r_[idx_v_tor, Nd + idx_v_pol]

    # Tau variables
    itau_v_t = np.arange(dL)[((sym_v+1)%2)::2]
    itau_v_p = np.arange(dL)[((sym_v+0)%2)::2]
    idx = np.r_[idx, 
        4*Nd + 0*dL + itau_v_t, 4*Nd + 1*dL + itau_v_t,
        4*Nd + 2*dL + itau_v_p, 4*Nd + 3*dL + itau_v_p, 4*Nd + 4*dL + itau_v_p, 4*Nd + 5*dL + itau_v_p,
    ]

    if perm is None:
        perm = sparse.eye_array(A.shape[0], format='csr')
    
    ele_arr = np.zeros(A.shape[0])
    ele_arr[idx] = 1
    idx_perm = np.arange(A.shape[0])[np.asarray(perm @ ele_arr, dtype=bool)]

    A = A[np.ix_(idx_perm, idx_perm)]
    perm = perm[np.ix_(idx_perm, idx)]
    return A, perm

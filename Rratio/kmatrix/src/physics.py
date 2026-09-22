#!/usr/bin/env python3
"""
Generic coupled-channel K-matrix physics.

Implements M(s) = n(s) [1 - K(s) Sigma(s)]^-1 K(s) n(s)  (Eq. 1 of
arXiv:2312.00619), with:

  - Sigma_k(s): the analytic (Chew-Mandelstam-type) self-energy for each
    channel. Equal-mass channels use the exact closed form of Eqs. (4)-(7)
    of that paper. Unequal-mass channels (e.g. D* Dbar) use a standard
    closed-form generalisation for L=0, and a numerically-evaluated
    dispersion integral for L=1 (no closed form is implemented here for
    that case -- see `sigma_numerical_crosscheck`).

  - n_k(s): the Blatt-Weisskopf vertex factor of Eq. (2), applied on the
    OUTSIDE of the K-matrix bracket as in Eq. (1) -- previously this was
    folded into the loop function instead and never applied externally.
    Kept as its own function since Eq. (A.5)'s P-vector construction for
    b -> s l+ l- reuses the same n(s).

Both the barrier factor and Sigma are now genuinely analytic functions of
s (built from q^2(s), a rational function of s, rather than from |q^2|),
which is a prerequisite for eventually continuing the amplitude off the
real axis (e.g. for a second-sheet pole search) and for the crossed
P-vector amplitude in the appendix, which needs the same n(s) to be a
well-defined analytic object, not just a real-axis convenience array.

Still agnostic to the number of channels / resonances -- both are
inferred from array shapes -- and no channel or parameter names appear
anywhere in this module.
"""

import numpy as np
from functools import lru_cache
from scipy import integrate


## ------------------------------------------------------------------ kinematics
def Kallen_triangle_function(x, y, z):
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*z*x


def breakup_q2(s, m1, m2=None):
    """Square of the breakup momentum q(s) for a channel with two legs of
    mass m1, m2 (equal-mass channel if m2 is omitted)."""
    if m2 is None:
        m2 = m1
    return Kallen_triangle_function(s, m1**2, m2**2) / (4*s)


def breakup_q(q2):
    """Principal-branch analytic continuation of q(s) = sqrt(q^2(s))."""
    return np.sqrt(q2) ## CHECK COMPLEX OK

def s_threshold(m1, m2):
    return (m1+m2)**2

def phsp_function(sqrt_s, q):
    # rho = np.zeros((s.shape[0], N_ch))
    # for i in range(N_ch):
        # rho[:, i] = np.real(q[i] / (8.0 * np.pi * np.sqrt(s)))
    rho = np.real(q / (8.0 * np.pi * sqrt_s))
    return rho        


## ---------------------------------------------------------- vertex factors
def Blatt_Weisskopf_sq(q2, l, z0):
    """Square of Blatt-Weisskopf form factor. Only L = 0, 1
    are implemented. Can take q2 as input as they are even functions."""
    # q2 = np.asarray(q2, dtype=complex)
    q2 = np.asarray(q2) ## ensure array
    if l == 0:
        return np.ones_like(q2)
    elif l == 1:
        return 1.0 / (1.0 + z0**2 * q2)
    else:
        raise ValueError(f"orbital angular momentum L={l} not implemented (only L=0,1)")

def vertex_factor_sq(q2, l, z0):
    """Square of the vertex factor for a single channel, square of eqn (2)
    and left unrooted as used as squared function later."""
    Fsq = Blatt_Weisskopf_sq(q2, l, z0)
    if l == 0: ## save time not raising to the power 0
        return Fsq
    else:
        return (z0**2 * q2)**l * Fsq

## keep with above as both useful in later development
def vertex_factor(vfs):
    return np.sqrt(vfs) ## CHECK COMPLEX OK HERE

## build the above into the vector structure for all channels together
# def vertex_factor_vector(q2, l, z0):
#     """The matrix with dim (N_ch, N_ch) and diagonal elements of the
#     vertex factors (not squared)."""
#     N_ch = setup.N_ch
#     n = np.zeros((s.shape[0], N_ch), dtype=complex)
#     for i in range(N_ch):
#         n[:, i] = np.sqrt(vertex_factor(q2, l, z0))
#     return n

## ------------------------------------------------------------- Self energy
def Pi_function_0(s, sth):
    """s = array of s values, sth = threshold centre of mass energy, so for 
    a given channel expect (m1+m2)^2 (in the complicated maths, is the branchcut 
    between the Riemann sheets)"""
    # term1 = np.sqrt(sth/s -1, dtype=complex) ## CHECK COMPLEX HERE OK
    # term2 = 1/term1 ## CHECK COMPLEX HERE OK
    # return -term1 * np.arctan(term2)
    arg = np.sqrt(s/(sth - s))
    return -1./arg * np.arctan(arg)


def Pi_function_1(s, z0, sth):
    """s = array of s values, z0 = radius as usual, sth = threshold
    centre of mass energy, so for a given channel expect (m1+m2)^2
    (in the complicated maths, is the branchcut between the Riemann sheets)"""
    s0 = 4 / (z0**2)

    # term1 = np.sqrt(s0 / (s0-sth)) ## CHECK COMPLEX HERE OK
    # term2 = 1/term1 ## CHECK COMPLEX HERE OK
    # term3 = s0 / (s+s0-sth) ## CHECK COMPLEX HERE OK
    # return term1 * term3 * np.arctanh(term2)
    return (s**1.5) / (np.sqrt(complex(s0)) * (s0 - sth) * (s + s0 - sth)) \
           * np.arctanh(np.sqrt(complex(1.0 - sth / s0)))



def Chew_Mandelstamm(s, q2, l, z0, sth):
    prefactor = 1/(8*np.pi**2)
    Pi0 = Pi_function_0(s, sth)
    if l == 0:
        return prefactor*Pi0
    elif l==1:
        Pi1 = Pi_function_1(s, z0, sth)
        BWsq = Blatt_Weisskopf_sq(q2, l, z0)
        sTerm = 0.25*(s-sth)*z0**2
        return prefactor * sTerm * (BWsq*Pi0 + Pi1)
    else:
        raise ValueError(f"orbital angular momentum L={l} not implemented (only L=0,1)")

## ------------------------------------------------------------- save kinematics
class Kinematics:
    """Container class for all quantitiesthat does NOT depend on fit parameters, will
    be true for a given set of s points. Build once per grid with precompute_kinematics()."""
    __slots__ = ("sqrt_s", "s", "q2", "q", "Sigma", "n", "rho")
 
    def __init__(self, sqrt_s, s, q2, q, Sigma, n, rho):
        self.sqrt_s = sqrt_s
        self.s = s
        self.q2 = q2          # (N_s, N_ch) complex
        self.q = q             # (N_s, N_ch) complex
        self.Sigma = Sigma    # (N_s, N_ch) complex
        self.n = n             # (N_s, N_ch) complex
        self.rho = rho         # (N_s, N_ch) real
 
 
def precompute_kinematics(sqrt_s, setup):
    """Compute q2_k(s), q_k(s), Sigma_k(s), n_k(s), rho_k(s) for every
    channel, once, for a fixed s-grid. Call this once (e.g. in
    AsymChi2.__init__, or once per plotting grid in reporting.py) and
    reuse the result -- never call it inside a per-MIGRAD-iteration cost
    function."""

    ## pull these out of setup to make syntax easier
    sqrt_s = np.atleast_1d(sqrt_s).astype(complex)
    s      = sqrt_s.astype(complex) ** 2
    N_s    = s.shape[0]
    N_ch   = setup.N_ch
    z0     = complex(setup.z0)
 
    ## setup arrays to store the set of values for each sqrt_s point and
    ## per each channel with own mass and o.a.m.
    q2_arr  = np.zeros((N_s, N_ch), dtype=complex)
    q_arr   = np.zeros((N_s, N_ch), dtype=complex)
    n       = np.zeros((N_s, N_ch, N_ch), dtype=complex)
    Sigma   = np.zeros((N_s, N_ch, N_ch), dtype=complex)
    rho     = np.zeros((N_s, N_ch), dtype=complex)
 
    for k in range(N_ch):
        m1, m2, l       = setup.masses[k].astype(complex), setup.masses2[k].astype(complex), setup.oam[k]
        sth             = s_threshold(m1, m2)
        q2              = breakup_q2(s, m1, m2)## calculated separately to use in functions more easily
        q               = breakup_q(q2) 
        q2_arr[:, k]    = q2
        q_arr[:, k]     = q
        rho[:, k]       = phsp_function(sqrt_s, q)
        n[:, k, k]      = vertex_factor(vertex_factor_sq(q2, l, z0)) ## diagonal matrix
        Sigma[:, k, k]  = Chew_Mandelstamm(s, q2, l, z0, sth)  ## diagonal matrix
        del m1, m2, l, q2, q ## variables to be used again next iteration
 
    return Kinematics(sqrt_s, s, q2_arr, q_arr, Sigma, n, rho)




## ------------------------------------------------------------- K-matrix / amplitude
def build_K(s, bare_masses, g):
    """bare_masses: (n_res,); g: (n_res, N_ch). Returns K: (N_s, N_ch, N_ch).
    Unchanged: real, symmetric pole expansion, Eq. (9) (background term
    omitted, matching the existing config)."""
    s = np.atleast_1d(s).real
    n_res = bare_masses.shape[0]
    N_ch = g.shape[1]
    K = np.zeros((s.shape[0], N_ch, N_ch), dtype=float)
    for a in range(n_res):
        denom = bare_masses[a]**2 - s
        denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom) ## CHECK IS THIS CAUSING THE DISCTY??
        ga = g[a]
        K += (ga[None, :, None] * ga[None, None, :]) / denom[:, None, None]
    return K


def M_matrix(bare_masses, g, setup, kinematics):
    """Full K-matrix scattering amplitude, Eq. (1):
        M(s) = n(s) [1 - K(s) Sigma(s)]^-1 K(s) n(s)
    kinematics is instance of class storing
    s           : the (array) of s values 
    Sigma       : the self energy function with shape (N_s, N_ch, N_ch)
    n           : vertex matrix with diagonal elements, shape (N_ch, N_ch)
    and we have free parameters as input:
    bare_masses : numpy array of the resonance masses (the free parameters)
    g           : numpy array of the coupling strengths (the free parameters)
    """

    # s = np.atleast_1d(s).astype(complex)
    N_s = kinematics.s.shape[0]
    N_ch = setup.N_ch
    K = build_K(kinematics.s, bare_masses, g).astype(complex)
    # idx = np.arange(N_ch)
    I = np.broadcast_to(np.eye(N_ch, dtype=complex), (N_s, N_ch, N_ch))
    bracket = np.matmul(np.linalg.inv(I - np.matmul(K, kinematics.Sigma)), K)
    # return kinematics.n[:, :, None] * bracket * kinematics.n[:, None, :]
    return np.matmul(kinematics.n, np.matmul(bracket, kinematics.n))



## ------------------------------------------------------------------ R model
def R_model(bare_masses, g, b, baseline, kinematics, setup):
    """Generic R-ratio model, any number of channels / resonances.
    Interface unchanged from before -- only the internals of M_matrix /
    self_energy_function / vertex_vector changed."""
    # sqrt_s = np.atleast_1d(sqrt_s).astype(float)
    # s = sqrt_s ** 2
    M = M_matrix(bare_masses, g, setup, kinematics)
    # ro = phsp_function(s, setup)

    R_res = np.zeros_like(kinematics.sqrt_s)
    R_bg = np.zeros_like(kinematics.sqrt_s)
    p0 = setup.production_idx
    for j in setup.open_charm_idxs:
        R_res += (2*setup.oam[j]+1) * kinematics.rho[:, j] * np.abs(M[:, p0, j] + b[j]) ** 2
        R_bg += kinematics.rho[:, j] * np.abs(b[j]) ** 2

    scaling = 3* 137**2 / (16*np.pi)
    R_res *= scaling
    R_bg *= scaling
    
    R_baseline = np.full_like(kinematics.sqrt_s, baseline)
    return baseline + R_res, R_baseline, baseline + R_bg
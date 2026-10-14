import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import integrate
from chew_mandelstams import EPSILON, s_threshold, sqrt_func

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
    return sqrt_func(q2)

def phsp_function(sqrt_s, q):
    rho = np.real(q / (8.0 * np.pi * sqrt_s))
    return rho        


## ---------------------------------------------------------- vertex factors
def Blatt_Weisskopf_sq(z, l):
    """Square of Blatt-Weisskopf form factor. Only L = 0, 1
    are implemented. Can take q2 as input as they are even functions."""
    # q2 = np.asarray(q2, dtype=complex)
    z = np.asarray(z) ## ensure array
    if l == 0:
        return np.ones_like(z)
    elif l == 1:
        return 1.0 / (1.0 + z)
    elif l == 2:
        return 1. / (9. + 3.*z + z**2)
    elif l == 3:
        return 1. / (225. + 45.*z + 6*z**2 + z**3)
    else:
        raise ValueError(f"orbital angular momentum L={l} not implemented (only L < 3)")

def vertex_factor_sq(q2, l, z0):
    """Square of the vertex factor for a single channel, square of eqn (2)
    and left unrooted as used as squared function later."""
    z = z0**2 * q2
    Fsq = Blatt_Weisskopf_sq(z, l)
    if l == 0: ## save time not raising to the power 0
        return Fsq
    else:
        return (z0**2 * q2)**l * Fsq

## keep with above as both useful in later development
def vertex_factor(vfs):
    return sqrt_func(vfs) 

## ------------------------------------------------------------- Self energy
def Pi_function_0(s, sth):
    """s = array of s values, sth = threshold centre of mass energy, so for 
    a given channel expect (m1+m2)^2 (in the complicated maths, is the branchcut 
    between the Riemann sheets)
    Manually change the sign of the imag part to compensate for numpy.sqrt 
    approaching the pole from the wrong direction"""
    arg = np.sqrt(s/(sth - s))
    wrongsign =  -1./arg * np.arctan(arg)
    return wrongsign.real - wrongsign.imag * 1j


def Pi_function_1(s, z0, sth):
    """s = array of s values, z0 = radius as usual, sth = threshold
    centre of mass energy, so for a given channel expect (m1+m2)^2
    (in the complicated maths, is the branchcut between the Riemann sheets)
    Manually change the sign *if* a pole exists (even though it cancels) to
    compensate for numpy.sqrt giving wrong root"""
    s0 = 4 / (z0**2)
    if s0 < sth:
        sign = -1
    else:
        sign = 1
    term1 = (s0**1.5) / (np.sqrt(s0-sth) * (s + s0 - sth))
    term2 = np.sqrt(1-sth/s0)
    return term1 * np.arctanh(term2) * sign

def Swave_Chew_Mandelstamm(s, sth, m1, m2, q):
    """Chew-Mandelstam self-energy for an S-wave (L=0) channel with unequal
    particle masses m1 != m2.
    Implements PDG 2022 Review, Section 50.3.5, Eq. (50.40):
        Sigma(s) = 1/(16 pi^2) * [
              2 q(s) / sqrt(s)
                * log( (m1^2 + m2^2 - s + 2 sqrt(s) q(s)) / (2 m1 m2) )
            - (m1^2 - m2^2) * (1/s - 1/s_+) * log(m1 / m2)
    """
    # First term: (2q/sqrt(s)) * log( (m1^2 + m2^2 - s + 2*sqrt(s)*q) / (2*m1*m2) )
    log_arg = (m1**2 + m2**2 - s + 2.0 * np.sqrt(s) * q) / (2.0 * m1 * m2)
    term1   = (2.0 * q / np.sqrt(s)) * np.log(log_arg)
 
    # Second term: -(m1^2 - m2^2) * (1/s - 1/s_+) * log(m1/m2)
    # This term is zero for equal masses and ensures Sigma(s_+) = 0.
    term2   = -(m1**2 - m2**2) * (1.0 / s - 1.0 / sth) * np.log(m1 / m2)
 
    return (term1 + term2) / (16.0 * np.pi**2)



def Chew_Mandelstamm(s, l, q, q2, z0, m1, m2):
    prefactor = 1/(8*np.pi**2)
    sth = s_threshold(m1, m2)
    if np.isclose(m1, m2):
        Pi0 = Pi_function_0(s, sth)
        if l == 0:
            return prefactor*Pi0
        elif l==1:
            Pi1 = Pi_function_1(s, z0, sth)
            z = q2 * z0**2
            BWsq = Blatt_Weisskopf_sq(s, l)
            sTerm = 0.25*(s-sth)*z0**2
            return prefactor * sTerm * (BWsq*Pi0 + Pi1)
    else:
        if l == 0:
            return Swave_Chew_Mandelstamm(s, sth, m1, m2, q)
        else:
            raise ValueError(f"orbital angular momentum L={l} not implemented (only L=0) for inequal masses")

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
    print(f"\033[1;93m[PHYSICS INFO]\033[0m Start calculating the kinematics for {len(sqrt_s)} s-value(s)...")

    ## pull these out of setup to make syntax easier
    sqrt_s = np.atleast_1d(sqrt_s).astype(complex)
    s      = sqrt_s.astype(complex) ** 2
    N_s    = s.shape[0]
    N_ch   = setup.N_ch
    z0     = complex(setup.z0)
    q0     = 1/z0
 
    ## setup arrays to store the set of values for each sqrt_s point and
    ## per each channel with own mass and o.a.m.
    q2_arr  = np.zeros((N_s, N_ch), dtype=complex)
    q_arr   = np.zeros((N_s, N_ch), dtype=complex)
    n       = np.zeros((N_s, N_ch, N_ch), dtype=complex)
    Sigma   = np.zeros((N_s, N_ch, N_ch), dtype=complex)
    rho     = np.zeros((N_s, N_ch), dtype=complex)
 
    for k in range(N_ch):
        m1, m2, l       = setup.masses[k].astype(complex), setup.masses2[k].astype(complex), setup.oam[k]
        q2              = breakup_q2(s, m1, m2)## calculated separately to use in functions more easily
        q               = breakup_q(q2) 
        q2_arr[:, k]    = q2
        q_arr[:, k]     = q
        rho[:, k]       = phsp_function(sqrt_s, q)
        n[:, k, k]      = vertex_factor(vertex_factor_sq(q2, l, z0)) ## diagonal matrix
        Sigma[:, k, k]  = Chew_Mandelstamm(s, l, q, q2, z0, m1, m2)  ## diagonal matrix
        del m1, m2, l, q2, q ## variables to be used again next iteration
 
    print("               ... finished calculation")
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
        denom = np.where(np.abs(denom) < EPSILON, EPSILON, denom)
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
## version with the model only for fitting
def R_model(bare_masses, g, b, baseline, kinematics, setup):
    """Generic R-ratio model, any number of channels / resonances.
    Interface unchanged from before -- only the internals of M_matrix /
    self_energy_function / vertex_vector changed."""
    # sqrt_s = np.atleast_1d(sqrt_s).astype(float)
    # s = sqrt_s ** 2
    M = M_matrix(bare_masses, g, setup, kinematics)
    # ro = phsp_function(s, setup)

    R_res = np.zeros_like(kinematics.sqrt_s)
    # R_bg = np.zeros_like(kinematics.sqrt_s)
    p0 = setup.production_idx
    for j in setup.open_charm_idxs:
        R_res += (2*setup.oam[j]+1) * kinematics.rho[:, j] * np.abs(M[:, p0, j] + b[j]) ** 2
        # R_bg += (2*setup.oam[j]+1) * kinematics.rho[:, j] * np.abs(b[j]) ** 2

    scaling = 3* 137**2 / (16*np.pi)
    R_res = R_res*scaling + baseline
    # R_bg *= scaling
    
    # R_baseline = np.full_like(kinematics.sqrt_s, baseline)
    # return baseline + R_res, R_baseline, baseline + R_bg
    return R_res

## version including the separated parts:
def R_model_separated(bare_masses, g, b, baseline, kinematics, setup):
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
        R_bg += (2*setup.oam[j]+1) * kinematics.rho[:, j] * np.abs(b[j]) ** 2

    scaling = 3* 137**2 / (16*np.pi)
    R_res *= scaling
    R_bg *= scaling
    
    R_baseline = np.full_like(kinematics.sqrt_s, baseline)
    return baseline + R_res, R_baseline, baseline + R_bg
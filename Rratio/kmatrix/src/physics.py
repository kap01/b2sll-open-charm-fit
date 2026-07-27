#!/usr/bin/env python3
"""
Generic coupled-channel K-matrix physics.

Kinematics, K-matrix / T-matrix construction, and the R-ratio model.
Everything here is agnostic to the number of channels (N_ch) and the
number of resonances (n_res): both are inferred from the shapes of the
arrays passed in. No channel names or parameter names appear anywhere
in this module -- all channel-specific information is carried in a
ChannelSetup instance, and all resonance/coupling values are passed in
as plain arrays built elsewhere (see fit.ParameterSpec).
"""

import numpy as np


## ------------------------------------------------------------------ kinematics
def Kallen_triangle_function(x, y, z):
    return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*z*x


def breakup_q2(s, m1, m2=None):
    """Square of the breakup momentum q(s) for a channel with two legs of
    mass m1, m2 (equal-mass channel if m2 is omitted)."""
    if m2 is None:
        m2 = m1
    return Kallen_triangle_function(s, m1**2, m2**2) / (4*s)


def barrier_factor(q2, l, z0):
    """Modified Blatt-Weisskopf form factor. Only L = 0, 1 are implemented --
    this is a genuine physics limitation (higher-L form factors need more
    terms), not something tied to channel/parameter naming."""
    if l == 0:
        return np.ones_like(q2, dtype=float)
    elif l == 1:
        z2 = z0**2
        return (z2 * q2) / (1.0 + z2 * np.abs(q2))
    else:
        raise ValueError(f"orbital angular momentum L={l} not implemented (only L=0,1)")


def phsp_function(s, setup):
    """Complex analytic phase space (+barrier) per channel, shape (N_s, N_ch)."""
    s = np.atleast_1d(s).astype(float)
    N_ch = setup.N_ch
    Sigma = np.zeros((s.shape[0], N_ch), dtype=complex)
    for i in range(N_ch):
        q2 = breakup_q2(s, setup.masses[i])
        q = np.sqrt(q2.astype(complex))
        Sigma[:, i] = (2.0 * q / np.sqrt(s)) * barrier_factor(q2, setup.oam[i], setup.z0)
    return Sigma


def rho_open(s, setup):
    """Physical open-channel phase space (Re -> 0 below threshold)."""
    return np.real(phsp_function(s, setup))


## ------------------------------------------------------------- K- and T-matrix
def build_K(s, bare_masses, g):
    """bare_masses: (n_res,); g: (n_res, N_ch). Returns K: (N_s, N_ch, N_ch)."""
    s = np.atleast_1d(s).astype(float)
    n_res = bare_masses.shape[0]
    N_ch = g.shape[1]
    K = np.zeros((s.shape[0], N_ch, N_ch), dtype=float)
    for a in range(n_res):
        denom = bare_masses[a]**2 - s
        denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)   ## guard exact pole
        ga = g[a]
        K += (ga[None, :, None] * ga[None, None, :]) / denom[:, None, None]
    return K


def T_matrix(s, bare_masses, g, setup):
    s = np.atleast_1d(s).astype(float)
    N_s = s.shape[0]
    N_ch = setup.N_ch
    K = build_K(s, bare_masses, g).astype(complex)
    Sigma_elem = phsp_function(s, setup)
    Sigma = np.zeros((N_s, N_ch, N_ch), dtype=complex)
    idx = np.arange(N_ch)
    Sigma[:, idx, idx] = Sigma_elem
    I = np.broadcast_to(np.eye(N_ch, dtype=complex), (N_s, N_ch, N_ch))
    A = I - 1j * np.matmul(K, Sigma)
    return np.matmul(np.linalg.inv(A), K)


## ------------------------------------------------------------------ R model
def R_model(sqrt_s, bare_masses, g, b, baseline, setup):
    """Generic R-ratio model, any number of channels / resonances.

    bare_masses : (n_res,)
    g           : (n_res, N_ch)  resonance couplings
    b           : (N_ch,)        coherent non-resonant continuum amplitude
                                  per channel (0 for the production channel
                                  and any channel with no continuum term)
    baseline    : scalar          non-resonant light-quark contribution

    Returns (R_total, R_baseline_only, R_baseline_plus_nonresonant_charm).
    """
    sqrt_s = np.atleast_1d(sqrt_s).astype(float)
    s = sqrt_s ** 2
    T = T_matrix(s, bare_masses, g, setup)
    ro = rho_open(s, setup)

    R_res = np.zeros_like(sqrt_s)
    R_bg = np.zeros_like(sqrt_s)
    p0 = setup.production_idx
    for j in setup.open_charm_idxs:
        R_res += ro[:, j] * np.abs(T[:, p0, j] + b[j]) ** 2
        R_bg += ro[:, j] * np.abs(b[j]) ** 2

    R_baseline = np.full_like(sqrt_s, baseline)
    return baseline + R_res, R_baseline, baseline + R_bg
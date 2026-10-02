import sys
PROJECT_ROOT = "/users/sd22284/b2sll_open_charm/repo/Rratio/kmatrix/src/"
sys.path.insert(0, str(PROJECT_ROOT))

import inspect
import warnings
from functools import partial
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import qrules
import sympy as sp
from ampform.dynamics import (
    BlattWeisskopfSquared,
    BreakupMomentumSquared,
    PhaseSpaceFactor,
    PhaseSpaceFactorComplex,
)
from ampform.io import aslatex
from ampform.sympy import unevaluated
from ampform.sympy.math import ComplexSqrt
from scipy.integrate import quad_vec
from sympy.printing.pycode import _unpack_integral_limits
from tqdm.auto import tqdm

from physics import s_threshold

import matplotlib.pyplot as plt
from reporting import mpl_presets
plt.rcParams.update(mpl_presets)

## ---- SETUP SYMPY SYMBOLS  ---- ##
s, m1, m2, s_prime = sp.symbols(r"s m1 m2 s^{\prime}", real=True)
epsilon, q0 = sp.symbols("epsilon q0", positive=True)
z = sp.Symbol("z")
L = sp.Symbol("L", integer=True, positive=True)

s_plus = s_prime + sp.I * epsilon
s_thr = (m1 + m2) ** 2



## ---- CONVERT THE AMPFORM FUNCTIONS INTO CLASSES  ---- ##
@unevaluated
class BreakupMomentum(sp.Expr):
    s: Any
    m1: Any
    m2: Any
    _latex_repr_ = R"q_a\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        q_squared = BreakupMomentumSquared(s, m1, m2)
        return ComplexSqrt(q_squared)


@unevaluated
class ChewMandelstamSWave(sp.Expr):
    s: Any
    m1: Any
    m2: Any
    _latex_repr_ = R"\Sigma\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        # evaluate=False in order to keep same style as PDG
        s, m1, m2 = self.args
        q = BreakupMomentum(s, m1, m2)
        left_term = sp.Mul(
            2 * q / sp.sqrt(s),
            sp.log((m1**2 + m2**2 - s + 2 * sp.sqrt(s) * q) / (2 * m1 * m2)),
            evaluate=False,
        )
        right_term = (m1**2 - m2**2) * (1 / s - 1 / (m1 + m2) ** 2) * sp.log(m1 / m2)
        return sp.Mul(
            1 / (16 * sp.pi**2),
            left_term - right_term,
            evaluate=False,
        )
@unevaluated
class ChewMandelstamPWave(sp.Expr):
    s: Any
    m1: Any
    m2: Any
    q0: Any  # the effective momentum scale
    _latex_repr_ = R"\Sigma_P\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        s, m1, m2, q0 = self.args
        s_thr = (m1 + m2) ** 2
        s0 = 4 * q0**2

        q_sq = BreakupMomentumSquared(s, m1, m2)  # reuse existing class from notebook

        # Blatt-Weisskopf F1^2(q/q0) = 1 / (1 + (q^2/q0^2))
        ff_sq = 1 / (1 + q_sq / q0**2)

        # Pi_0: note ComplexSqrt handles s > s_thr (below threshold) correctly
        sqrt_ratio = ComplexSqrt((s_thr - s) / s)  # = sqrt((s_thr-s)/s)
        Pi0 = -sqrt_ratio * sp.atan(1 / sqrt_ratio)

        # Pi_1: the pole from (s + s0 - s_thr) cancels exactly with F1^2 — see paper
        Pi1 = (
            s0 ** sp.Rational(3, 2)
            / (ComplexSqrt(s0 - s_thr) * (s + s0 - s_thr))
            * sp.atanh(ComplexSqrt(1 - s_thr / s0))
        )

        return sp.Mul(
            1 / (8 * sp.pi**2),
            (s - s_thr) / s0,
            ff_sq * Pi0 + Pi1,
            evaluate=False,
        )

@unevaluated
class FormFactor(sp.Expr):
    s: Any
    m1: Any
    m2: Any
    L: Any
    q0: Any = 1
    _latex_repr_ = R"n_a^2\left({s}\right)"

    def evaluate(self) -> sp.Expr:
        s, m1, m2, L, q0 = self.args
        q_squared = BreakupMomentumSquared(s, m1, m2)
        return BlattWeisskopfSquared(
            z=q_squared / (q0**2),
            angular_momentum=L,
        )



## ---- CONVERT CLASSES INTO EXPRESSIONS ---- ##
cmsw_expr = ChewMandelstamSWave(s, m1, m2)
cmpw_expr = ChewMandelstamPWave(s, m1, m2, q0)
# - different format expression as has the maths inside using the above class:
integrand = (PhaseSpaceFactor(s_prime, m1, m2) * FormFactor(s_prime, m1, m2, L, q0)) / (
    (s_prime - s_thr) * (s_prime - s - epsilon * sp.I)
)
# - diagnostic only:
# q_expr = BreakupMomentum(s, m1, m2)
# rho_expr = PhaseSpaceFactorComplex(s, m1, m2)
# na_expr = FormFactor(s, m1, m2, L, q0)
# bl_expr = BlattWeisskopfSquared(z, L)


## ---- CONVERT EXPRESSIONS INTO LAMBDAS  ---- ##
cmsw_func = sp.lambdify((s_prime, m1, m2, epsilon), cmsw_expr.doit().subs(s, s_plus))
cmpw_func = sp.lambdify((s_prime, m1, m2, q0, epsilon), cmpw_expr.doit().subs(s, s_plus))
integrand_func = sp.lambdify(
    args=(s_prime, s, L, epsilon, m1, m2, q0),
    expr=integrand.doit(),
    modules="numpy",
)
# - diagnostic only:
# rho_func = sp.lambdify(symbols, rho_expr.doit().subs(s, s_plus))

## ---- FUNCTIONS TO ACTUALLY BE CALLED  ---- ##

# - general integral, shouldn't need to use for S-wave, or equal mass P-wave
def CMfunc_general_integral(s_vals, m1_val, m2_val, oam_val, q0_val=1.0, epsilon_val=0.0001):
    s_lo = np.min(s_vals)
    s_hi = np.max(s_vals)
    s_thr_val = (m1_val + m2_val) ** 2
    if oam_val == 0:
        s_thr_val += epsilon_val ## avoid the singularity in the S-wave
    integral_result = quad_vec(
        lambda x: integrand_func(
            x,
            s=s_vals,
            L=oam_val,
            epsilon=epsilon_val,
            m1=m1_val,
            m2=m2_val,
            q0=q0_val,
        ),
        a=s_thr_val, 
        b=np.inf,
    )[0]
    return (s_vals - s_thr_val) / np.pi * integral_result

# - analytical form of the S wave, same function for equal and inequal mass as minimal time saving in evaluatino
def CMfunc_S_wave(s_vals, m1_val, m2_val, epsilon_val=0.0001):
    # - note that epsilon_val should be unused
    return np.array([
        cmsw_func(s,  m1=m1_val, m2=m2_val, epsilon=epsilon_val)
        for s in s_vals])

# - analytical form of the equal mass P wave
def CMfunc_P_wave_equalmass(s_vals, m1_val, m2_val, q0_val, epsilon_val=0.0001):
    # - note that epsilon_val should be unused
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
        return np.array([
            cmpw_func(s, m1=m1_val, m2=m2_val, q0=q0_val, epsilon=epsilon_val)
            for s in s_vals
        ])
 
# - wrapper function
def Chew_Mandelstamm(s_vals, m1_val, m2_val, oam_val, q0_val=1.0, epsilon_val=0.0001):
    print(f"\033[1;93m[CM INFO]\033[0m Computing Chew-Mandelstam for m1={m1_val}, m2={m2_val}, L={oam_val}")
    if oam_val == 0:
        return CMfunc_S_wave(s_vals, m1_val, m2_val)
    elif oam_val == 1 and np.isclose(m1_val, m2_val, 0.000001):
        return CMfunc_P_wave_equalmass(s_vals, m1_val, m2_val, q0_val, epsilon_val)
    else:
        return CMfunc_general_integral(s_vals, m1_val, m2_val, oam_val, q0_val, epsilon_val)


## ---- PLOTTING ---- ##
real_style = {"label": "Real part", "c": "deeppink"}
imag_style = {"label": "Imag part", "c": "forestgreen", "linestyle": "-."}
threshold_style = {"label": R"$s_\mathrm{thr}$", "c": "deepskyblue", "linewidth": 0.9, "linestyle":"dotted"}

def make_CM_plot(cm_vals, s_vals, m1_val, m2_val, oam_val, channelname, out_pdf=None):
    # - don't want to set of a warning when we split up the real and imag parts of cm_vals:
    warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)

    X = ['S', 'P', 'D', 'F'][oam_val]
    s_thr = s_threshold(m1_val, m2_val)

    fig, ax = plt.subplots(figsize=(6, 4))
    # - x axis
    ax.axhline(0, color="black", linewidth=0.75)
    # - mark the s_threshold
    ax.axvline(s_thr, **threshold_style)
    # - plot the components
    ax.plot(s_vals, cm_vals.real, alpha=0.7, **real_style)
    ax.plot(s_vals, cm_vals.imag, alpha=0.7, **imag_style)

    # - graphics:
    ax.legend()
    ax.set_xlim(np.min(s_vals), np.max(s_vals))
    ax.set_xlabel(r"$s$ [GeV$^2$]", loc='right')
    ax.set_ylabel(r"$\Sigma(s)$", rotation=90, fontsize=20) ##text renders smaller
    ax.set_title(f"Dispersion integral for {channelname} {X}-wave:\n$L={oam_val}$, $m_1={m1_val:.2f}$, $m_2={m2_val:.2f}$", fontsize=17)

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"\033[1;93m[CM INFO]\033[0m plot saved to {out_pdf}")
        return None
    else:
        return fig

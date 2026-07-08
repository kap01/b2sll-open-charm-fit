#!/usr/bin/env python3
"""
Coupled-channel K-matrix fit to the BESII R-ratio scan (INSPIRE 552757).

Channels (fixed index order):
    0 : e+e-          production channel (photon couples here), L=0, rho=1
    1 : D+ D-         P-wave (L=1)
    2 : D0 D0bar      P-wave (L=1)
    3 : D*0 D*0bar    S-wave (L=0)

Three K-matrix poles (bare resonances) describe the charmonium structure
in the psi(3770)-psi(4160) region.

Amplitude
---------
K_{ij}(s) = sum_a  g_i^a g_j^a / (M_a^2 - s)            (real, symmetric)
rho_i(s)  = 2 q_i/sqrt(s) * B_{L_i}(q_i^2)              (diag phase space + barrier,
                                                         analytically continued)
T(s)      = [ I - i K(s) rho(s) ]^{-1} K(s)             (T-matrix)

The e+e- -> hadrons cross section is fed by the production element T_{0j}.
A smooth, COHERENT non-resonant charm-production amplitude b_j is added to
each open-charm channel so that R can both sustain the open-charm continuum
(|b_j|^2) and dip below it through resonance-continuum interference
(2 Re[T_{0j} b_j*]) -- an incoherent additive continuum cannot do the latter.

Because R = sigma_had/sigma_mumu and sigma_mumu ~ 1/s, the 1/s cancels:

    R(s) = R_uds + sum_{j in open charm} rho_j^open(s) * | T_{0j}(s) + b_j |^2

Isospin: one coupling gDD^a is shared by D+D- and D0D0bar (their small
threshold splitting is kept via distinct phase spaces); likewise one bDD.

Fit: chi2 with asymmetric errors, minimised with iminuit v2 / MIGRAD.
"""

import numpy as np
import uproot
from iminuit import Minuit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ constants
M_DP, M_D0, M_DST0 = 1.86966, 1.86484, 2.00685            # GeV
CH_MASS = np.array([0.000511, M_DP, M_D0, M_DST0])
CH_L    = np.array([0, 1, 1, 0])
NCH     = 4
BW_R    = 3.0                                             # Blatt-Weisskopf radius (1/GeV)
OPEN_CHARM = (1, 2, 3)

# ------------------------------------------------------------------ kinematics
def breakup_q2(s, m):
    return s / 4.0 - m * m

def barrier(q2, L):
    """Pole-free Blatt-Weisskopf F_L(q^2): L=0 ->1 ; L=1 -> R^2 q^2/(1+R^2|q^2|)."""
    if L == 0:
        return np.ones_like(q2, dtype=float)
    z = BW_R * BW_R
    return (z * q2) / (1.0 + z * np.abs(q2))

def rho_diag(s):
    """Complex analytic phase space (+barrier) per channel, shape (Ns,NCH)."""
    s = np.atleast_1d(s).astype(float)
    rho = np.zeros((s.shape[0], NCH), dtype=complex)
    for i in range(NCH):
        q2 = breakup_q2(s, CH_MASS[i])
        q  = np.sqrt(q2.astype(complex))
        rho[:, i] = (2.0 * q / np.sqrt(s)) * barrier(q2, CH_L[i])
    return rho

def rho_open(s):
    """Physical open-channel phase space (Re -> 0 below threshold)."""
    return np.real(rho_diag(s))

# ------------------------------------------------------------- K- and T-matrix
def build_K(s, masses, g):
    s = np.atleast_1d(s).astype(float)
    K = np.zeros((s.shape[0], NCH, NCH), dtype=float)
    for a, Ma in enumerate(masses):
        den = Ma * Ma - s
        den = np.where(np.abs(den) < 1e-7, 1e-7, den)     # guard exact pole
        ga = g[a]
        K += (ga[None, :, None] * ga[None, None, :]) / den[:, None, None]
    return K

def T_matrix(s, masses, g):
    s = np.atleast_1d(s).astype(float)
    Ns = s.shape[0]
    K = build_K(s, masses, g).astype(complex)
    rho = rho_diag(s)
    Rho = np.zeros((Ns, NCH, NCH), dtype=complex)
    idx = np.arange(NCH)
    Rho[:, idx, idx] = rho
    I = np.broadcast_to(np.eye(NCH, dtype=complex), (Ns, NCH, NCH))
    A = I - 1j * np.matmul(K, Rho)
    return np.matmul(np.linalg.inv(A), K)

# ------------------------------------------------------------------ R model
def _unpack(p):
    masses = np.array([p["M1"], p["M2"], p["M3"]])
    g = np.array([
        [p["ge1"], p["gDD1"], p["gDD1"], p["gDs1"]],
        [p["ge2"], p["gDD2"], p["gDD2"], p["gDs2"]],
        [p["ge3"], p["gDD3"], p["gDD3"], p["gDs3"]],
    ])
    b = np.array([0.0, p["bDD"], p["bDD"], p["bDs"]])     # coherent charm continuum
    return masses, g, b

def R_model(sqrt_s, p):
    """Return (R_total, R_uds_baseline, R_nonres_charm).
       R_nonres_charm is the |b|^2 continuum (resonances switched off)."""
    sqrt_s = np.atleast_1d(sqrt_s).astype(float)
    s = sqrt_s ** 2
    masses, g, b = _unpack(p)
    T = T_matrix(s, masses, g)
    ro = rho_open(s)

    R_res = np.zeros_like(sqrt_s)
    R_bg  = np.zeros_like(sqrt_s)
    for j in OPEN_CHARM:
        R_res += ro[:, j] * np.abs(T[:, 0, j] + b[j]) ** 2
        R_bg  += ro[:, j] * np.abs(b[j]) ** 2
    baseline = np.full_like(sqrt_s, p["R_uds"])
    return p["R_uds"] + R_res, baseline, p["R_uds"] + R_bg

# ------------------------------------------------------------------ data / cost
def load_data(path):
    g = uproot.open(path)["R"]
    x   = np.asarray(g.values("x"), dtype=float)
    y   = np.asarray(g.values("y"), dtype=float)
    eyl = np.asarray(g.errors("low",  "y"), dtype=float)
    eyh = np.asarray(g.errors("high", "y"), dtype=float)
    return x, y, eyl, eyh


class AsymChi2:
    """chi2 with asymmetric errors: sigma = eyh where model>=data else eyl."""
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, eyl, eyh):
        self.x, self.y, self.eyl, self.eyh = x, y, eyl, eyh

    def __call__(self, R_uds, bDD, bDs,
                 M1, ge1, gDD1, gDs1,
                 M2, ge2, gDD2, gDs2,
                 M3, ge3, gDD3, gDs3):
        p = dict(R_uds=R_uds, bDD=bDD, bDs=bDs,
                 M1=M1, ge1=ge1, gDD1=gDD1, gDs1=gDs1,
                 M2=M2, ge2=ge2, gDD2=gDD2, gDs2=gDs2,
                 M3=M3, ge3=ge3, gDD3=gDD3, gDs3=gDs3)
        m, _, _ = R_model(self.x, p)
        if not np.all(np.isfinite(m)):
            return 1e12
        sigma = np.where(m >= self.y, self.eyh, self.eyl)
        return float(np.sum(((m - self.y) / sigma) ** 2))

# ------------------------------------------------------------------ fit driver
def _apply_limits(m):
    m.limits["R_uds"] = (1.5, 2.6)
    m.limits["bDD"]   = (-4.0, 4.0)
    m.limits["bDs"]   = (-4.0, 4.0)
    for i, lo, hi in [(1, 3.70, 3.85), (2, 3.95, 4.12), (3, 4.12, 4.45)]:
        m.limits[f"M{i}"] = (lo, hi)
        for c in ("ge", "gDD", "gDs"):
            m.limits[f"{c}{i}"] = (-8.0, 8.0)

def run_fit(path, n_starts=6, seed=1):
    x, y, eyl, eyh = load_data(path)
    cost = AsymChi2(x, y, eyl, eyh)
    rng = np.random.default_rng(seed)

    base = dict(R_uds=2.20, bDD=1.10, bDs=0.20,
                M1=3.773, ge1=0.30, gDD1=0.50, gDs1=0.10,
                M2=4.040, ge2=0.60, gDD2=0.30, gDs2=0.60,
                M3=4.180, ge3=0.40, gDD3=0.30, gDs3=0.60)

    best = None
    for k in range(n_starts):
        init = dict(base)
        if k > 0:                                        # jittered restarts
            for key in init:
                if key.startswith(("ge", "gDD", "gDs", "bD")):
                    init[key] += rng.normal(0, 0.4)
                elif key.startswith("M"):
                    init[key] += rng.normal(0, 0.02)
        m = Minuit(cost, **init)
        _apply_limits(m)
        m.strategy = 2
        try:
            m.migrad(ncall=400000)
            m.simplex()
            m.migrad()
        except Exception:
            continue
        if m.valid and (best is None or m.fval < best.fval):
            best = m
    best.hesse()
    return best, (x, y, eyl, eyh)

# ------------------------------------------------------------------ reporting
def report(m, data):
    x = data[0]
    ndf = len(x) - m.nfit
    print("=" * 66)
    print(f" MIGRAD valid : {m.valid}     accurate covariance : {m.accurate}")
    print(f" chi2         : {m.fval:.2f}")
    print(f" ndf          : {ndf}   (N = {len(x)}, free params = {m.nfit})")
    print(f" chi2 / ndf   : {m.fval/ndf:.3f}")
    print("-" * 66)
    for p in m.parameters:
        print(f"   {p:8s} = {m.values[p]:+9.4f}  +/- {m.errors[p]:.4f}")
    print("=" * 66)

def make_plot(m, data, out_png):
    x, y, eyl, eyh = data
    p = {k: m.values[k] for k in m.parameters}
    grid = np.linspace(3.55, 4.85, 1600)
    Rtot, Rbase, Rbg = R_model(grid, p)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(x, y, yerr=[eyl, eyh], fmt="o", ms=4, color="black",
                ecolor="0.55", elinewidth=0.8, capsize=1.5, zorder=5,
                label="BESII data (INSPIRE 552757)")
    ax.plot(grid, Rtot, "-", color="crimson", lw=2.1,
            label=r"K-matrix fit ($\chi^2/\mathrm{ndf}=%.2f$)" % (m.fval/(len(x)-m.nfit)))
    ax.plot(grid, Rbg, "--", color="steelblue", lw=1.3,
            label=r"non-resonant charm continuum ($|b|^2$)")
    ax.plot(grid, Rbase, ":", color="grey", lw=1.2, label=r"$uds$ baseline $R_{uds}$")

    for i in (1, 2, 3):
        ax.axvline(p[f"M{i}"], color="0.7", ls=":", lw=0.8)
        ax.text(p[f"M{i}"], 5.05, r"$M_%d$" % i, ha="center", fontsize=8, color="0.4")
    for thr, lab in [(2*M_D0, r"$D\bar D$"), (2*M_DST0, r"$D^{*}\bar{D^{*}}$")]:
        ax.axvline(thr, color="green", ls="-.", lw=0.8, alpha=0.5)
        ax.text(thr, 1.92, lab, ha="center", fontsize=8, color="green", alpha=0.8)

    ax.set_xlabel(r"$\sqrt{s}$  [GeV]")
    ax.set_ylabel(r"$R = \sigma_{\rm had}/\sigma_{\mu\mu}$")
    ax.set_title("Coupled-channel K-matrix fit to BESII $R$  (4 channels, 3 resonances)")
    ax.set_xlim(3.55, 4.85)
    ax.set_ylim(1.85, 5.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[plot saved] {out_png}")

if __name__ == "__main__":
    ROOT_FILE = "../data/ins552757-v1.root"
    m, data = run_fit(ROOT_FILE)
    report(m, data)
    make_plot(m, data, "kmatrix_R_fit.png")

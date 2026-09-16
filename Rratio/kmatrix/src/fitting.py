#!/usr/bin/env python3
"""
Fit machinery: turning the config's named resonances/couplings into a flat
parameter vector Minuit can work with, the asymmetric chi2 cost function,
and a KMatrixFit class that separates *setting up* the fit (starting
values, limits, building the Minuit object) from *running* it (the
multi-start MIGRAD loop).

No parameter names are hardcoded anywhere: they are all generated from the
config (resonance labels and coupling-group names), and the physics module
(physics.py) never sees names at all -- only the plain arrays that
ParameterSpec.unpack() produces.
"""
import numpy as np
from iminuit import Minuit

from physics import R_model


## ------------------------------------------------------------------ parameter bookkeeping
class ParameterSpec:
    """Flat, Minuit-facing view of the fit parameters, plus the index maps
    needed to unpack a flat array back into (bare_masses, g, b, baseline)
    without any function ever referring to a parameter by name.

    Mutable (unlike ChannelSetup): fields can be tweaked after construction,
    e.g. spec.limits[i] = (lo, hi) to loosen a bound between fit attempts
    without rebuilding the whole object from the config."""

    _fields = ("names", "start", "limits", "baseline_idx",
               "res_mass_idx", "res_coupling_idx", "continuum_coupling_idx")

    def __init__(self, names, start, limits, baseline_idx,
                 res_mass_idx, res_coupling_idx, continuum_coupling_idx):
        self.names = names                                   # list of parameter names
        self.start = start                                   # (n_params,) starting values, same order as names
        self.limits = limits                                 # list of (lo, hi), same order as names
        self.baseline_idx = baseline_idx                      # flat-array index of the baseline parameter
        self.res_mass_idx = res_mass_idx                      # [n_res] -> index into the flat array
        self.res_coupling_idx = res_coupling_idx              # [n_res][N_ch] -> index into the flat array
        self.continuum_coupling_idx = continuum_coupling_idx  # [N_ch] -> index into the flat array, or -1

    def __repr__(self):
        fields = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._fields)
        return f"ParameterSpec({fields})"

    def __eq__(self, other):
        if not isinstance(other, ParameterSpec):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f), getattr(other, f))
            if isinstance(getattr(self, f), np.ndarray)
            else getattr(self, f) == getattr(other, f)
            for f in self._fields
        )

    def unpack(self, par):
        n_res = len(self.res_mass_idx)
        N_ch = len(self.continuum_coupling_idx)
        bare_masses = np.array([par[i] for i in self.res_mass_idx])
        g = np.zeros((n_res, N_ch))
        # scaler = 1
        for a in range(n_res):
            g[a] = [par[i] for i in self.res_coupling_idx[a]]
            ## HARDCODING THE RELATIVE COUPLING FOR THE D CHANNELS
            # g[a][2] = scaler * g[a][1] ## first idx for DpDm channel, second idx for DzDzbar channel
            # g[a][5] = scaler * g[a][4] ## first idx for DpDm channel, second idx for DzDzbar channel
        b = np.array([par[i] if i >= 0 else 0.0 for i in self.continuum_coupling_idx])
        ## HARDCODING THE RELATIVE COUPLING FOR THE D CHANNELS
        # b[2] = scaler * b[1]
        # b[5] = scaler * b[4]
        baseline = par[self.baseline_idx]
        return bare_masses, g, b, baseline

    def kind_of(self, i):
        """'mass', 'baseline', or 'coupling' -- used to pick a jitter size
        for restart k>0 in KMatrixFit.run()."""
        if i == self.baseline_idx:
            return "baseline"
        if i in self.res_mass_idx:
            return "mass"
        return "coupling"


def build_parameter_spec(config, setup):
    """Build a ParameterSpec from config['resonances'] / config['continuum'],
    resolved against the channel groups in `setup` (a ChannelSetup)."""
    names, start, limits = [], [], []

    def add(name, start_val, lims):
        idx = len(names)
        names.append(name)
        start.append(start_val)
        limits.append(tuple(lims))
        return idx

    cont_cfg = config["continuum"]
    baseline_cfg = cont_cfg["baseline"]
    baseline_name = baseline_cfg.get("name", "baseline")
    baseline_idx = add(baseline_name, baseline_cfg["start"], baseline_cfg["limits"])

    continuum_group_idx = {}
    for group, spec in cont_cfg.get("couplings", {}).items():
        continuum_group_idx[group] = add(f"b_{group}", spec["start"], spec["limits"])

    res_mass_idx = []
    res_group_idx = []          # [n_res] dict: group name -> flat index
    for r in config["resonances"]:
        label = r["label"]
        res_mass_idx.append(add(f"M_{label}", r["mass"]["start"], r["mass"]["limits"]))
        gmap = {}
        for group, cspec in r["couplings"].items():
            gmap[group] = add(f"g_{label}_{group}", cspec["start"], cspec["limits"])
        res_group_idx.append(gmap)

    res_coupling_idx = []
    for gmap in res_group_idx:
        row = []
        for ch_group in setup.groups:
            if ch_group not in gmap:
                raise ValueError(f"resonance is missing a coupling for group '{ch_group}'")
            row.append(gmap[ch_group])
        res_coupling_idx.append(row)

    continuum_coupling_idx = [
        -1 if i == setup.production_idx else continuum_group_idx.get(ch_group, -1)
        for i, ch_group in enumerate(setup.groups)
    ]

    return ParameterSpec(names=names, start=np.array(start, dtype=float), limits=limits,
                          baseline_idx=baseline_idx, res_mass_idx=res_mass_idx,
                          res_coupling_idx=res_coupling_idx,
                          continuum_coupling_idx=continuum_coupling_idx)


## ------------------------------------------------------------------ cost function
class AsymChi2:
    """chi2 with asymmetric errors: sigma = eyh where model >= data else eyl."""
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, eyl, eyh, param_spec, channel_setup):
        self.x, self.y, self.eyl, self.eyh = x, y, eyl, eyh
        self.param_spec = param_spec
        self.setup = channel_setup

    def __call__(self, par):
        bare_masses, g, b, baseline = self.param_spec.unpack(par)
        m, _, _ = R_model(self.x, bare_masses, g, b, baseline, self.setup)
        if not np.all(np.isfinite(m)):
            return 1e12
        sigma = np.where(m >= self.y, self.eyh, self.eyl)
        return float(np.sum(((m - self.y) / sigma) ** 2))


## ------------------------------------------------------------------ fit driver
class KMatrixFit:
    """Separates configuring the fit (setup) from executing it (run)."""

    def __init__(self, data, param_spec, channel_setup):
        self.x, self.y, self.eyl, self.eyh = data
        self.param_spec = param_spec
        self.channel_setup = channel_setup
        self.cost = AsymChi2(self.x, self.y, self.eyl, self.eyh, param_spec, channel_setup)
        self.minuit = None

    def _new_minuit(self, start):
        m = Minuit(self.cost, start, name=self.param_spec.names)
        for name, lims in zip(self.param_spec.names, self.param_spec.limits):
            m.limits[name] = lims
        m.strategy = 2
        return m

    def setup(self):
        """Build the Minuit object at the configured starting values and
        apply limits. Does not run the minimisation -- lets you inspect
        e.g. the starting chi2 (self.minuit.fval) before committing to a fit."""
        self.minuit = self._new_minuit(self.param_spec.start)
        return self.minuit

    def run(self, n_starts=6, seed=1, jitter=None):
        """Run MIGRAD with n_starts randomised restarts and keep the best
        valid result. setup() must be called first."""
        if self.minuit is None:
            raise RuntimeError("call setup() before run()")

        jitter = jitter or {"mass": 0.02, "coupling": 0.4, "baseline": 0.0}
        rng = np.random.default_rng(seed)
        base_start = np.array(self.param_spec.start, dtype=float)

        best = None
        for k in range(n_starts):
            init = base_start.copy()
            if k > 0:
                for i in range(len(init)):
                    sigma = jitter.get(self.param_spec.kind_of(i), 0.0)
                    if sigma:
                        init[i] += rng.normal(0, sigma)
            m = self._new_minuit(init)
            try:
                m.migrad(ncall=400000)
                m.simplex()
                m.migrad()
            except Exception:
                continue
            if m.valid and (best is None or m.fval < best.fval):
                best = m

        if best is None:
            raise RuntimeError("no valid fit found in any restart")
        best.hesse()
        self.minuit = best
        return best

    def evaluate(self, sqrt_s):
        """R_model at sqrt_s, using whichever parameter values self.minuit
        currently holds -- the best-fit values after run(), or the config's
        starting values if only setup() has been called so far. Effectively
        R_model bound to "the current state of this fit"; scalar or array
        sqrt_s both work (see R_model)."""
        if self.minuit is None:
            raise RuntimeError("call setup() (and usually run()) before evaluate()")
        par = [self.minuit.values[name] for name in self.param_spec.names]
        bare_masses, g, b, baseline = self.param_spec.unpack(par)
        return R_model(sqrt_s, bare_masses, g, b, baseline, self.channel_setup)

import numpy as np
from iminuit import Minuit

from physics import R_model, precompute_kinematics
from reporting import limit_warning, dict_fit_status
from time import time

## ------------------------------------------------------------------ parameter bookkeeping
class ParameterSpec:
    """Flat, Minuit-facing view of the fit parameters, plus the index maps
    needed to unpack a flat array back into (bare_masses, g_k^r, c_ij, baseline)
    without any function ever referring to a parameter by name.

    Mutable (unlike ChannelSetup): fields can be tweaked after construction,
    e.g. spec.limits[i] = (lo, hi) to loosen a bound between fit attempts
    without rebuilding the whole object from the config."""

    _fields = ("names", "start", "limits", "baseline_idx",
               "res_mass_idx", "res_coupling_idx", "non_res_idx")

    def __init__(self, names, start, limits, baseline_idx,
                 res_mass_idx, res_coupling_idx, non_res_idx):
        self.names = names                                   # list of parameter names
        self.start = start                                   # (n_params,) starting values, same order as names
        self.limits = limits                                 # list of (lo, hi), same order as names
        self.baseline_idx = baseline_idx                     # flat-array index of the baseline parameter
        self.res_mass_idx = res_mass_idx                     # [n_res] -> index into the flat array
        self.res_coupling_idx = res_coupling_idx             # [n_res][N_ch] -> index into the flat array
        self.non_res_idx = non_res_idx         # [N_ch, N_ch] -> index into the matrix

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

    @property
    def n_res(self):
        return len(self.res_mass_idx)

    def unpack(self, par):
        """Turns the flat list of idxs/values into the physics categories:
        par         : flat list of all fit parameters in expected order, e.g. extracted from minuit
        bare_masses : (n_res,)
        g           : (n_res, N_ch)
        c           : (N_ch, N_ch)  real, symmetric background K-matrix
        baseline    : scalar
        """
        # n_res = self.n_res ## len(self.res_mass_idx)
        N_ch = len(self.non_res_idx)

        ## pick out the poles from the parameters list
        bare_masses = np.array([par[i] for i in self.res_mass_idx])

        ## pick out the couplings from the parameters list, makes a 2D list of resonance then channel
        ## for g^r_c, returns the same value if channels share coupling group
        g = np.zeros((self.n_res, N_ch))
        for r in range(self.n_res):
            g[r] = [par[k] for k in self.res_coupling_idx[r]]

        ## pick out the non resonant background terms, will make a sym matrix of channels
        c = np.zeros((N_ch, N_ch))
        for i in range(N_ch):
            for j in range(N_ch):
                c[i, j] = par[self.non_res_idx[i][j]]

        ## pick out the continuum term (R_uds)
        baseline = par[self.baseline_idx]

        return bare_masses, g, c, baseline

    def kind_of(self, i):
        """'mass', 'baseline', or 'coupling' -- used to pick a jitter size
        for restart k>0 in KMatrixFit.run()."""
        if i == self.baseline_idx:
            return "baseline"
        elif i in self.res_mass_idx:
            return "mass"
        else:
            for row in self.non_res_idx:
                if i in row:
                    return "non_res"
        return "coupling"


def build_parameter_spec(config, channel_setup):
    """Build a ParameterSpec from config['resonances'] / config['continuum'], 
    (config['channels'] contains no fit parameters and are contained in channel_setup 
    object channel_setup)."""

    names, start, limits = [], [], []

    def add(name, start_val, lims):
        idx = len(names)
        names.append(name)
        start.append(start_val)
        limits.append(lims)
        return idx

    ## - extract baseline continuum contribution, likely fixed but can be a fit parameter
    baseline_cfg = config["baseline"]
    baseline_idx = add(baseline_cfg["name"], baseline_cfg["start"], baseline_cfg["limits"])

    ## - extract poles and couplings for each resonance
    res_mass_idx = []
    res_group_idx = []          # [n_res] dict: group name -> flat index
    for r in config["resonances"]:
        label = r["label"]
        res_mass_idx.append(add(f"M_{label}", r["mass"]["start"], r["mass"]["limits"]))
        ## - build map of coupling_group : idx, one map per res
        gmap = {}
        for group, cspec in r["couplings"].items():
            gmap[group] = add(f"g_{label}_{group}", cspec["start"], cspec["limits"])
        res_group_idx.append(gmap)

    ## - turn the map of coupling groups into channel idx
    ## (can find the idx for the coupling g value of a given channel without needing it's coupling group)
    ## i.e. turning the list of dicts into a list of lists
    res_coupling_idx = []
    for gmap in res_group_idx:
        row = []
        for ch_group in channel_setup.groups:
            if ch_group not in gmap:
                raise ValueError(f"resonance is missing a coupling for group '{ch_group}'")
            row.append(gmap[ch_group])
        res_coupling_idx.append(row)

    ## - extract the non resonant contribution matrix elements, c_ij
    nonres_cfg = config["non_resonant"] ## remember is already validated
    non_res_idx = [[None]*channel_setup.N_ch for _ in range(channel_setup.N_ch)]
    for i in range(channel_setup.N_ch):
        i_group = channel_setup.group_idxs[i] ## convert channel idx -> group idx
        # print(">>> >>>", i, " -> ", i_group)
        for j in range(i+1):   # expect lower triangle; j <= i
            j_group = channel_setup.group_idxs[j] ## convert channel idx -> group idx
            # print(">>>", j, " -> ", j_group)
            st = nonres_cfg["starts"][i_group][j_group]
            lo = nonres_cfg["limits_lo"][i_group][j_group]
            hi = nonres_cfg["limits_hi"][i_group][j_group]

            if lo == "FIXED" or hi == "FIXED":
                lims = "FIXED"
            else:
                lims = [lo, hi]
            
            ## recall we're looping over channels, only add a new fit parameter if it's a new group
            name = f"c_{i_group}_{j_group}" ## NB c is indexing the group not the channels, may want to change naming convention later
            if name not in names:
                flat_idx = add(name, st, lims)
            else:
                flat_idx = names.index(name)
            ## always add the index to the maps bit for referencing
            non_res_idx[i][j] = flat_idx
            non_res_idx[j][i] = flat_idx   # symmetry: same parameter, same index

    return ParameterSpec(names=names, start=np.array(start, dtype=float), limits=limits,
                         baseline_idx=baseline_idx, res_mass_idx=res_mass_idx,
                         res_coupling_idx=res_coupling_idx, non_res_idx=non_res_idx)


## ------------------------------------------------------------------ cost function
class AsymChi2:
    """chi2 with asymmetric errors: sigma = eyh where model >= data else eyl."""
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, eyl, eyh, param_spec, channel_setup):
        self.x, self.y, self.eyl, self.eyh = x, y, eyl, eyh
        self.param_spec = param_spec
        self.setup = channel_setup
        self.kinematics = precompute_kinematics(self.x, channel_setup)
        ## penalty term:
        self.penalty_sqrt_s  = channel_setup.penalty["s"]
        self.penalty_r_max   = channel_setup.penalty["R_max"]
        self.penalty_sigma   = channel_setup.penalty["R_err"]
        self.kinematics_penalty = precompute_kinematics(np.array([self.penalty_sqrt_s]), channel_setup)

    def __call__(self, par):
        bare_masses, g, c, baseline = self.param_spec.unpack(par)
        m = R_model(bare_masses, g, c, baseline, self.kinematics, self.setup)
        if not np.all(np.isfinite(m)):
            return 1e12
        sigma = np.where(m >= self.y, self.eyh, self.eyl)
        chi2 = np.sum(((m.real - self.y) / sigma) ** 2) ## use real part only, m.imag should be 0 anyway

        ## penalty term:
        r_limit = R_model(bare_masses, g, c, baseline,
                     self.kinematics_penalty, self.setup)
        r_limit = r_limit[0].real ## 1 element array and only want real part
        if r_limit > self.penalty_r_max:
            chi2 += ((r_limit - self.penalty_r_max) / self.penalty_sigma) ** 2
        return chi2

        return 


## ------------------------------------------------------------------ fit driver
class KMatrixFit:
    """Separates configuring the fit (setup) from executing it (run)."""

    def __init__(self, data, param_spec, channel_setup):
        self.x, self.y, self.eyl, self.eyh = data
        self.param_spec = param_spec
        self.channel_setup = channel_setup
        self.cost = AsymChi2(self.x, self.y, self.eyl, self.eyh, param_spec, channel_setup)
        self.minuit = None
        self.status = None

    def new_minuit(self, start=None):
        start = self.param_spec.start if start is None else start
        m = Minuit(self.cost, start, name=self.param_spec.names)
        for name, lims in zip(self.param_spec.names, self.param_spec.limits):
            if type(lims) == str:
                if lims == "FIXED":
                    m.fixed[name] = True
                else:
                    print("\033[1;93m[FITTER INFO]\033[0m ***WARNING***")
                    print(f"             unexpected limit value for parameter {name} : {lims}, check config file")
                    print(f"             fixing parameter in fit by default")
                    m.fixed[name] = True
            else:
                m.limits[name] = lims
        m.strategy = 2
        if self.minuit == None:
            self.minuit = m
        return m

    def run(self, n_starts=6, seed=1, jitter=None):
        """Run MIGRAD with n_starts randomised restarts and keep the best
        valid result. setup() must be called first."""
        # if self.minuit is None:
        #     raise RuntimeError("call setup() before run()")

        jitter = jitter or {"mass": 0.02, "coupling": 0.4, "baseline": 0.0}
        rng = np.random.default_rng(seed)
        base_start = np.array(self.param_spec.start, dtype=float)

        best = None ## this variable will store/return the best fit result of the following loop
        for k in range(n_starts):
            start = time()
            init = base_start.copy()
            if k > 0: 
            ## not cross validated but the logic is such that if k>0 self.minuit has
            ## already been assigned to the initial minuit, so we can use self.minuit's
            ## settings to find the fixed fit parameters
                for i in range(len(init)):
                    if not self.minuit.fixed[self.param_spec.names[i]]:
                        sigma = jitter.get(self.param_spec.kind_of(i), 0.0)
                        if sigma:
                            new_init = init[i] + rng.normal(0, sigma)
                            ## ensure not gone out of parameter's fit limits with the jitter
                            if new_init < self.param_spec.limits[i][0]:
                                new_init = self.param_spec.limits[i][0]
                            elif new_init > self.param_spec.limits[i][1]:
                                new_init = self.param_spec.limits[i][1]
                            init[i] = new_init
            m = self.new_minuit(init)
            try:
                m.migrad(ncall=400000)
                if not m.valid: ## escape local minima
                    m.simplex()
                    m.migrad(ncall=400000)
            except Exception as e:
                print("[FIT INFO] Exception during migrad:\n", repr(e))
                continue
                
            if m.valid and (best is None or m.fval < best.fval):
                best = m
            end = time()
            print(f"\033[1;93m[FITTER INFO]\033[0m Completed minimisation round {k+1} of {n_starts} jitters, took {end-start:1f}s")

        if best is None:
            self.status = 4
            raise RuntimeError("no valid fit found in any restart")

        ## more strict validation (have already check it is valid in the above loop)
        best.hesse()
        if not(best.fmin.has_posdef_covar) or  best.fmin.has_made_posdef_covar or best.fmin.hesse_failed:
            print(f"\033[1;93m[FITTER WARNING]\033[0m Best fit reported has an issue, check below:")
            print("                 Hesse() failed:                                   ", best.fmin.hesse_failed)
            print("                 Accurate + positive definite covariance:          ", best.fmin.has_posdef_covar)
            print("                 Tried forcing covariance to be positive definite: ", best.fmin.has_made_posdef_covar)
            self.status = 3 if best.fmin.hesse_failed else 2 if not(best.fmin.has_posdef_covar) else 1 if best.fmin.has_made_posdef_covar else None
        else:
            print("\033[1;93m[FITTER INFO]\033[0m Fit is valid and covariance is positive definite and accurate")
            self.status = 0

        self.minuit = best ## overwrite the initial minuit with the best one in either case
        return best

    def evaluate(self, sqrt_s):
        """R_model at sqrt_s, using whichever parameter values self.minuit
        currently holds -- the best-fit values after run(), or the config's
        starting values if only setup() has been called so far. Effectively
        R_model bound to "the current state of this fit"; scalar or array
        sqrt_s both work (see R_model)."""
        if self.minuit is None:
            raise RuntimeError("call new_minuit() (and usually run()) before evaluate()")
        par = [self.minuit.values[name] for name in self.param_spec.names]
        bare_masses, g, c, baseline = self.param_spec.unpack(par)
        kinematics = precompute_kinematics(sqrt_s, self.channel_setup)
        return R_model(bare_masses, g, c, baseline, kinematics, self.channel_setup)

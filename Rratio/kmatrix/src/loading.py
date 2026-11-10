import numpy as np
import uproot
from yaml import safe_load
from os import makedirs, listdir
from os.path import exists

USER_PATH = "/users/sd22284/b2sll_open_charm/repo/Rratio/"


class ChannelSetup:
    """Everything the physics functions need to know about the channels.
    Built once from the config and
    passed explicitly into every function below -- no module globals.

    Immutable by construction: after __init__ runs, attempting to reassign
    any field raises AttributeError, so a physics function can't silently
    mutate shared setup state that every other caller also depends on."""

    _fields = ("names", "masses", "oam", "groups", "tex", "production_idx",
               "channel_idxs", "group_idxs", "groups_set", "z0", "penalty", "latest_CMs")

    def __init__(self, names, masses, masses2, oam, groups, tex, production_idx,
                 channel_idxs, group_idxs, groups_set, z0, penalty, latest_CMs):
        self.__dict__["names"] = names                         # channel names, index-aligned; for reporting/plotting only
        self.__dict__["masses"] = masses                       # (N_ch,) threshold mass of one leg of each channel - this first particle
        self.__dict__["masses2"] = masses2                     # (N_ch,) threshold mass of one leg of each channel - the second particle
        self.__dict__["oam"] = oam                             # (N_ch,) orbital angular momentum per channel (0 or 1)
        self.__dict__["groups"] = groups                       # (N_ch,) coupling-group name per channel, index-aligned
        self.__dict__["tex"] = tex                             # (N_ch,) tex for each of the channels to label on the plot
        self.__dict__["production_idx"] = production_idx       # index of the channel the photon couples to
        self.__dict__["channel_idxs"] = channel_idxs           # indices of channels summed over in R
        self.__dict__["group_idxs"] = group_idxs               # indices of the group each channel uses, i.e. index version of groups
        self.__dict__["groups_set"] = groups_set                 # set of .groups, order preserved, matches index in group_idx
        self.__dict__["z0"] = z0                               # Blatt-Weisskopf radius parameter
        self.__dict__["penalty"] = penalty                     # Constants for the penalty term
        self.__dict__["latest_CMs"] = latest_CMs               # Store CM path here so can be used with build_CM_filename

    def __setattr__(self, name, value):
        raise AttributeError(
            f"ChannelSetup is immutable; cannot set '{name}' after construction")

    @property
    def N_ch(self):
        return len(self.names)

    @property
    def N_g(self):
        return len(set(self.groups))

    def __repr__(self):
        fields = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._fields)
        return f"ChannelSetup({fields})"

    def __eq__(self, other):
        if not isinstance(other, ChannelSetup):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f), getattr(other, f))
            if isinstance(getattr(self, f), np.ndarray)
            else getattr(self, f) == getattr(other, f)
            for f in self._fields
        )


def load_config(path):
    with open(path, 'r') as infile:
        return safe_load(infile)


def validate_config(config):
    """Structural checks on the config, run before anything else touches it.
    Raises ValueError with a specific message on failure."""

    ## check all top-level keys present
    for key in ("channels", "resonances", "baseline", "non_resonant", "paths", "plotting", "constants"):
        if key not in config:
            raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m config is missing required top-level key '{key}'")

    ## check CM paths present
    if "CM_path" not in config["paths"]:
        raise ValueError("\033[1;93m[CONFIG WARNING]\033[0m No path for Chew Mandelstam lookup table specified")

    channels_cfg = config["channels"]
    resonances_cfg = config["resonances"]
    non_resonant_cfg = config["non_resonant"]
    ## check at least 2 channels
    if len(channels_cfg) < 2:
        raise ValueError("\033[1;93m[CONFIG ERROR]\033[0m need at least two channels (one production, one open-charm)")

    ## check only 1 production channel
    n_production = sum(1 for c in channels_cfg if c.get("role", None) == "production")
    if n_production != 1:
        raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m exactly one channel must have role: production (found {n_production})")

    ## check coupling group specified for all non-prodution channels
    for c in channels_cfg:
        if c.get("role") != "production" and "coupling_group" not in c:
            raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m channel '{c.get('name', '?')}' needs a coupling_group")

    ## check at least one resonances
    if len(resonances_cfg) < 1:
        raise ValueError("\033[1;93m[CONFIG ERROR]\033[0m need at least one resonance")

    ## check each resonances has a mass and list of couplings
    for r in resonances_cfg:
        if "label" not in r or "mass" not in r or "couplings" not in r:
            raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m malformed resonance entry: {r}")
    
    ## check all coupling groups have a channel
    r_cgs = []
    for r in resonances_cfg:
        for cg in r["couplings"].keys():
            if cg not in r_cgs:
                r_cgs.append(cg)
    c_cgs = list(set([c["coupling_group"] for c in channels_cfg]))
    for cg in r_cgs:
        if cg not in c_cgs:
            raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m coupling group '{cg}' used in resonances but does not belong to any channel")

    ## check non resonant matrices present + right shape (lower triangular, dim of N_groups x N_groups)
    N_g = len(c_cgs)
    for key in ["starts", "limits_lo", "limits_hi"]:
        try:
            mat = non_resonant_cfg[key]
        except KeyError:
            raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m Missing the setup {key} in the non_resonant contribution")
        ## check number of rows is correct:
        if len(mat) != N_g:
            raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m Malformed non_resonant {key} matrix, number of rows =/= number of coupling groups")
        for this_len, row in zip(range(1, N_g+1), mat):
            if len(row) != this_len:
                raise ValueError(f"\033[1;93m[CONFIG ERROR]\033[0m Malformed non_resonant {key} matrix, number of columns =/= number of coupling groups")



    ## check fit range specified
    if config['fit'].get('fit_range', None) == None:
        print("[CONFIG WARNING] no fit limits set, using range 0-10 to encompass all expected data points")
        config['fit']['fit_range'] = {}
        config['fit']['fit_range']['low'] = 0.
        config['fit']['fit_range']['high'] = 10.
    

    print("\033[1;93m[CONFIG INFO]\033[0m passed validation")
    print("=" * 64)


def build_channel_setup(config):
    """Turn config['channels'] + config['constants'] into a ChannelSetup."""
    channels_cfg = config["channels"]

    ## get the basic information from each channel
    names = tuple(c["name"] for c in channels_cfg)
    masses  = np.array([c["mass"] for c in channels_cfg], dtype=float)
    masses2 = np.array([c["mass2"] for c in channels_cfg], dtype=float)
    oam = np.array([c["oam"] for c in channels_cfg], dtype=int)
    tex = [c.get("tex", c["name"]) for c in channels_cfg]

    ## find the production channel (expect ee)
    production_idx = next(i for i, c in enumerate(channels_cfg) if c["role"] == "production") # already validated only one production channel

    ## find the coupling group for each channel
    groups = []
    for i, c in enumerate(channels_cfg):
        groups.append(c["coupling_group"])
    groups = tuple(groups)

    ## get the indicies of all non-production channels
    channel_idxs = tuple(i for i in range(len(channels_cfg)) if i != production_idx)

    ## list equivlanet to groups that is the idx of each group in order
    groups_set = list(dict.fromkeys(groups))
    group_idxs = []
    for g in groups:
        group_idxs.append(groups_set.index(g))

    constants = config.get("constants", {})
    if "z0" in constants:
        z0 = constants["z0"]
    elif "q0" in constants:
        z0 = 1.0 / constants["q0"]
    else:
        z0 = 3.0

    penalty = {}
    for c in constants:
        if "penalty_" in c:
            penalty[c.replace("penalty_", "")] = constants[c]

    latest_CMs = config["paths"]["CM_path"]

    return ChannelSetup(names=names, masses=masses, masses2=masses2, oam=oam, groups=groups, tex=tex, production_idx=production_idx,
                        channel_idxs=channel_idxs, group_idxs=group_idxs, groups_set=groups_set, z0=z0, penalty=penalty, latest_CMs=latest_CMs)


def load_data(paths, fit_range, return_id=False):
    x, y, eyl, eyh, ids = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for idx, p in enumerate(paths):
        g = uproot.open(p)["R"]
        x_all = np.asarray(g.values("x"), dtype=float)
        idx_lo = np.searchsorted(x_all, fit_range['low'], side='right')
        idx_hi = np.searchsorted(x_all, fit_range['high'], side='left')

        this_x  = x_all[idx_lo:idx_hi]
        this_y  = np.asarray(g.values("y"), dtype=float)[idx_lo:idx_hi]
        this_eyl = np.asarray(g.errors("low", "y"), dtype=float)[idx_lo:idx_hi]
        this_eyh = np.asarray(g.errors("high", "y"), dtype=float)[idx_lo:idx_hi]

        this_id = np.full_like(this_x, idx)

        x = np.hstack([x, this_x])
        y = np.hstack([y, this_y])
        eyl = np.hstack([eyl, this_eyl])
        eyh = np.hstack([eyh, this_eyh])
        ids = np.hstack([ids, this_id])

    if not return_id:
        return x, y, eyl, eyh
    else:
        return x, y, eyl, eyh, ids

def build_cm_filename(setup, channel_idx, user_path=USER_PATH):
    if "eff_psi" in setup.names[channel_idx]: ## hardcoding that eff_psi_2S and eff_psi_3770 use the same CM function
        use_name = "eff_psi"
    else:
        use_name = setup.names[channel_idx]
    return user_path + setup.latest_CMs +f"chew_mands__z0_{setup.z0}__l_{setup.oam[channel_idx]}__{use_name}.txt"

def validate_cm_file(setup, user_path=USER_PATH):
    outfolder_txt = build_cm_filename(setup, 0, user_path).split("chew_mands")[0]
    if exists(outfolder_txt):
        existing_CMs = listdir(outfolder_txt)
        for i in range(setup.N_ch):
            filename = build_cm_filename(setup, i, USER_PATH).replace(outfolder_txt, "")
            if filename in existing_CMs:
                raise FileExistsError(f"""\033[1;93m[SETUP ERROR]\033[0m CM lookup table for \033[0;93m{setup.names[i]}\033[0m already exists in
                               \033[0;93m{outfolder_txt}\033[0m
                               will not overwrite, remove file, or channel from config""")
        print(f"\033[1;93m[SETUP INFO]\033[0m CM folder at \033[0;93m{outfolder_txt}\033[0m exists,")
        print("             but none of channels in config are present, will calculate all")
    else:
        print("\033[1;93m[SETUP INFO]\033[0m Making newfolder for CMs, all will be calculated")
        makedirs(outfolder_txt)
    return

def load_cm_vals(setup, channel_idx, user_path=USER_PATH):
    filename = build_cm_filename(setup, channel_idx, user_path)
    try:
        lookup_table = np.loadtxt(filename, dtype=complex)
    except:
        raise IOError(f"""\033[1;93m[CM ERROR]\033[0m Couldn't read CM lookup table at
\n\033[0;93m{filename}\033[0m,\n
                    check you have run plot_CM_functions?""") from None

    ref_s = np.sqrt(lookup_table[1].real)
    return ref_s, lookup_table[0]


def load_fit_result(path):
    """Read a fit_result.txt written by reporting.report() and return a plain
    dict that carries everything needed to reproduce the plots."""
    values, errors, fixed = {}, {}, {}
    chi2, ndf, fit_status = None, None, None

    with open(path, 'r') as infile:
        for raw in infile:
            line = raw.strip()

            ## - get the chi2, e.g. " Fit status   : 2    - Fit is valid but covariance not postive definite"
            if line.startswith("Fit status "):
                fit_status = int(line.split(":")[-1].strip().split("-")[0])
            ## - get the chi2, e.g. " chi2         : 71.82"
            if line.startswith("chi2 "):
                chi2 = float(line.split(":")[1].strip())
            ## get the ndf, e.g. "ndf          : 116   (N = 127, free params = 11)"
            elif line.startswith("ndf "):
                ndf = int(line.split(":")[1].split("(")[0].strip())

            ## - get the parameter results, two possibilities for free/fixed results:
            ## free:  "   g_0_ee         =   +0.3412  +/- 0.0231"
            ## fixed: "   M_0            =   +3.6861      ------"
            if "=" in line and not line.startswith("="):
                name, rest = line.split("=", 1)
                name = name.strip()
                rest = rest.strip()

                if "------" in rest: ## must be fixed: value is whatever comes before the dashes
                    val_str = rest.replace("------", "").strip()
                    values[name] = float(val_str)
                    errors[name] = None
                    fixed[name]  = True
                elif "+/-" in rest: ## must be free: there is a value and unc to split up
                    val_str, err_str = rest.split("+/-")
                    values[name] = float(val_str.strip())
                    errors[name] = float(err_str.strip())
                    fixed[name]  = False

    if not values:
        raise ValueError(
            f"\033[1;93m[LOAD ERROR]\033[0m No parameter lines found in {path}.\n"
            "             Check the file was written by reporting.report()."
        )
    if chi2 is None or ndf is None:
        raise ValueError(
            f"\033[1;93m[LOAD ERROR]\033[0m Could not parse chi2 / ndf from {path}."
        )

    print(f"\033[1;93m[LOAD INFO]\033[0m Read fit result from {path}")
    print(f"            chi2={chi2:.2f}  ndf={ndf}  fit_status={fit_status}")
    print(f"            {sum(not v for v in fixed.values())} free + {sum(fixed.values())} fixed parameters")

    return {
        "chi2"       : chi2,
        "ndf"        : ndf,
        "fit_status" : fit_status,
        "values"     : values,
        "errors"     : errors,
        "fixed"      : fixed,
    }


def load_fit_conditions(path):
    """Read a fit_conditions.txt file written by reporting.save_conditions()
    and return a dict that can be used to overwrite options in a config.
    """
    conditions = {}
    in_channels = False
    channels = []
    resonances = []

    with open(path, 'r') as infile:
        for raw in infile:
            line = raw.strip()

            if line.startswith("-- channels --"): ## next loop will go through channels
                in_channels = True
                continue
            elif line.startswith("--") or line.startswith("="): ## next loop will stop going through channels
                in_channels = False
            elif in_channels and line:
                # format: "[0] ee           mass=0.000511  L=0  group=ee       (production)"
                line = line.split("]", 1)[1].strip()
                name, rest = line.split(None, 1)
                tokens = {}
                for token in rest.split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        tokens[k] = v
                    elif token.startswith("(") and token.endswith(")"):
                        tokens["role"] = token.strip("()")
                channels.append({
                    "name":           name,
                    "mass":           float(tokens["mass"]),
                    "oam":            int(tokens["L"]),
                    "coupling_group": tokens["group"].rstrip(","),
                })
            elif line.startswith("Fit range"):
                # format: "   Fit range                 : 3.7 - 3.83"
                value_part = line.split(":", 1)[1].strip()
                lo_str, hi_str = value_part.split("-")
                conditions.setdefault("fit", {})["fit_range"] = {
                    "low":  float(lo_str.strip()),
                    "high": float(hi_str.strip()),
                }
            elif line.startswith("z0 value"):
                # format: "   z0 value (q0 value)       : 3.0 (0.3333)"
                value_part = line.split(":", 1)[1].strip()
                z0_val = value_part.split("(")[0].strip()
                conditions.setdefault("constants", {})["z0"] = int(z0_val)
            elif line.startswith("Number of resonances"):
                value_part = line.split(":", 1)[1].strip()
                n_val = int(value_part.split("(")[0].strip())
                conditions["resonances"] = n_val

    conditions["channels"] = channels

    return conditions
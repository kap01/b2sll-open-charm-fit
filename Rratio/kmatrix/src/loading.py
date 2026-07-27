#!/usr/bin/env python3
"""
Config loading/validation and data loading.

The config schema (see example_config.yml) is deliberately explicit about
the difference between a parameter's *starting value* and its *limits* --
every place a number is given for fitting, it is given as
    {start: <value>, limits: [<lo>, <hi>]}
so there is never ambiguity about which is which.

Channels and resonances are both open-ended lists, so any number of each
is supported without touching this module or physics.py.
"""

import numpy as np
import uproot
from yaml import safe_load

class ChannelSetup:
    """Everything the physics functions need to know about the channels.
    Built once from the config and
    passed explicitly into every function below -- no module globals.

    Immutable by construction: after __init__ runs, attempting to reassign
    any field raises AttributeError, so a physics function can't silently
    mutate shared setup state that every other caller also depends on."""

    _fields = ("names", "masses", "oam", "groups", "tex",
               "production_idx", "open_charm_idxs", "z0")

    def __init__(self, names, masses, oam, groups, tex,
                 production_idx, open_charm_idxs, z0):
        self.__dict__["names"] = names                         # channel names, index-aligned; for reporting/plotting only
        self.__dict__["masses"] = masses                       # (N_ch,) threshold mass of one leg of each channel
        self.__dict__["oam"] = oam                             # (N_ch,) orbital angular momentum per channel (0 or 1)
        self.__dict__["groups"] = groups                       # (N_ch,) coupling-group name per channel, index-aligned
        self.__dict__["tex"] = tex                             # (N_ch,) tex for each of the channels to label on the plot
        self.__dict__["production_idx"] = production_idx       # index of the channel the photon couples to
        self.__dict__["open_charm_idxs"] = open_charm_idxs     # indices of channels summed over in R
        self.__dict__["z0"] = z0                               # Blatt-Weisskopf radius parameter

    def __setattr__(self, name, value):
        raise AttributeError(
            f"ChannelSetup is immutable; cannot set '{name}' after construction")

    @property
    def N_ch(self):
        return len(self.names)

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
    for key in ("channels", "resonances", "continuum", "paths", "plotting"):
        if key not in config:
            raise ValueError(f"[CONFIG ERROR] config is missing required top-level key '{key}'")

    channels = config["channels"]
    resonances = config["resonances"]
    ## check at least 2 channels
    if len(channels) < 2:
        raise ValueError("[CONFIG ERROR] need at least two channels (one production, one open-charm)")

    ## check only 1 production channel
    n_production = sum(1 for c in channels if c.get("role") == "production")
    if n_production != 1:
        raise ValueError(f"[CONFIG ERROR] exactly one channel must have role: production (found {n_production})")

    ## check coupling group specified for all non-prodution channels
    for c in channels:
        if c.get("role") != "production" and "coupling_group" not in c:
            raise ValueError(f"[CONFIG ERROR] channel '{c.get('name', '?')}' needs a coupling_group")

    ## check at least one resonances
    if len(resonances) < 1:
        raise ValueError("[CONFIG ERROR] need at least one resonance")

    ## check each resonances has a mass and list of couplings
    for r in resonances:
        if "label" not in r or "mass" not in r or "couplings" not in r:
            raise ValueError(f"[CONFIG ERROR] malformed resonance entry: {r}")
    
    ## check all coupling groups have a channel
    r_cgs = []
    for r in resonances:
        for cg in r["couplings"].keys():
            if cg not in r_cgs:
                r_cgs.append(cg)
    c_cgs = list(set([c["coupling_group"] for c in channels]))
    for cg in r_cgs:
        if cg not in c_cgs:
            raise ValueError(f"[CONFIG ERROR] coupling group '{cg}' used in resonances but does not belong to any channel")

    

    print("[CONFIG INFO] passed validation")
    print("=" * 64)


def build_channel_setup(config):
    """Turn config['channels'] + config['constants'] into a ChannelSetup."""
    channels_cfg = config["channels"]

    names = tuple(c["name"] for c in channels_cfg)
    masses = np.array([c["mass"] for c in channels_cfg], dtype=float)
    oam = np.array([c["oam"] for c in channels_cfg], dtype=int)
    tex = [c.get("tex", c["name"]) for c in channels_cfg]

    production_idx = next(i for i, c in enumerate(channels_cfg) if c.get("role") == "production")

    groups = []
    for i, c in enumerate(channels_cfg):
        groups.append(c["name"] if i == production_idx else c["coupling_group"])
    groups = tuple(groups)

    open_charm_idxs = tuple(i for i, c in enumerate(channels_cfg) if c.get("open_charm", False))

    const = config.get("constants", {})
    if "z0" in const:
        z0 = const["z0"]
    elif "q0" in const:
        z0 = 1.0 / const["q0"]
    else:
        z0 = 3.0

    return ChannelSetup(names=names, masses=masses, oam=oam, groups=groups, tex=tex,
                         production_idx=production_idx,
                         open_charm_idxs=open_charm_idxs, z0=z0)


def load_data(path):
    g = uproot.open(path)["R"]
    x = np.asarray(g.values("x"), dtype=float)
    y = np.asarray(g.values("y"), dtype=float)
    eyl = np.asarray(g.errors("low", "y"), dtype=float)
    eyh = np.asarray(g.errors("high", "y"), dtype=float)
    return x, y, eyl, eyh
